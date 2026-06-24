"""DB-free microbenchmarks for the spikesorting v2 efficiency evaluation.

Validates the magnitude + scaling of the top hypotheses in
``efficiency-evaluation.md`` without a database or the full pipeline.

Methodology (best practice):
  * fixed RNG seed; 1 warmup iteration discarded; REPEATS timed runs -> median + min
  * memory peak via tracemalloc in a SEPARATE pass from timing
  * each measurement scaled across input sizes to expose the growth curve
  * candidate rewrites checked for byte-identical output before reporting speedup

Run:
  conda run -n spyglass_spikesorting_v2 python \
    .claude/docs/plans/spikesorting-v2/bench_efficiency.py
"""

from __future__ import annotations

import gc
import platform
import time
import tracemalloc

import numpy as np

SEED = 12345
REPEATS = 7
WARMUP = 1
N_REF = 108_000_000  # 1 h @ 30 kHz -- production reference size
FS = 30_000.0


def _timeit(make_call):
    """Median/min wall time of a zero-arg callable, warmup discarded."""
    for _ in range(WARMUP):
        make_call()
    samples = []
    for _ in range(REPEATS):
        gc.collect()
        t0 = time.perf_counter()
        make_call()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return samples[len(samples) // 2], samples[0]


def _peak_mb(make_call):
    """tracemalloc peak (MiB) of a zero-arg callable."""
    gc.collect()
    tracemalloc.start()
    keep = make_call()  # keep ref so the alloc is live at the measurement
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del keep
    return peak / 2**20


def _section(title):
    print(f"\n{'='*72}\n{title}\n{'='*72}")


# --------------------------------------------------------------------------- #
# M1 -- recording.get_times() full-vector cost (root of R2/R3/R7)
# --------------------------------------------------------------------------- #
def m1_get_times():
    _section("M1  recording.get_times() materialization (R2/R3/R7 root)")
    import spikeinterface.full as si

    sizes = [1_000_000, 10_000_000, N_REF]
    print(f"{'n_samples':>12} {'mode':>10} {'dtype':>9} {'MB(array)':>10} "
          f"{'peak MB':>9} {'median ms':>10}")
    for n in sizes:
        secs = n / FS
        rec = si.generate_recording(
            num_channels=1, sampling_frequency=FS, durations=[secs]
        )
        out = rec.get_times()
        arr_mb = out.nbytes / 2**20
        is_concrete = isinstance(out, np.ndarray)
        del out
        peak = _peak_mb(lambda: rec.get_times())
        med, _ = _timeit(lambda: rec.get_times())
        print(f"{n:>12,} {'rate-based':>10} {str(np.dtype('float64')):>9} "
              f"{arr_mb:>10.1f} {peak:>9.1f} {med*1e3:>10.2f}"
              f"   concrete_ndarray={is_concrete}")
        del rec
        gc.collect()

    # explicit timestamps (set_times) at one modest size to confirm same cost
    n = 10_000_000
    rec = si.generate_recording(
        num_channels=1, sampling_frequency=FS, durations=[n / FS]
    )
    rec.set_times(np.arange(n, dtype=np.float64) / FS, segment_index=0)
    out = rec.get_times()
    print(f"{n:>12,} {'explicit':>10} {str(out.dtype):>9} "
          f"{out.nbytes/2**20:>10.1f} {'--':>9} {'--':>10}"
          f"   concrete_ndarray={isinstance(out, np.ndarray)}")
    print(f"\n  -> get_times() materializes float64 = 8 bytes/sample in BOTH "
          f"modes; at N_REF={N_REF:,} that is {N_REF*8/2**20:.0f} MB per call.")


# --------------------------------------------------------------------------- #
# M2 -- artifact_frames magnitude (R2 detail)
# --------------------------------------------------------------------------- #
def m2_artifact_frames():
    _section("M2  artifact_frames = every masked sample (R2 detail)")
    rng = np.random.default_rng(SEED)
    print(f"  N_REF = {N_REF:,} samples (1 h @ 30 kHz)")
    print(f"{'artifact %':>11} {'n_spans':>9} {'frames(M)':>10} "
          f"{'int64 MB':>9} {'concat ms':>10}")
    for frac in (0.001, 0.01, 0.05):
        masked = int(N_REF * frac)
        n_spans = 2000  # artifact bursts scattered across the recording
        span_len = max(1, masked // n_spans)
        starts = np.sort(rng.integers(0, N_REF - span_len, size=n_spans))
        spans = [(int(s), int(s) + span_len) for s in starts]

        def build():
            return np.concatenate(
                [np.arange(s, e, dtype=np.int64) for s, e in spans]
            )

        frames = build()
        med, _ = _timeit(build)
        print(f"{frac*100:>10.1f}% {n_spans:>9,} {frames.size/1e6:>10.2f} "
              f"{frames.nbytes/2**20:>9.1f} {med*1e3:>10.2f}")
        del frames
    print("\n  -> the trigger array passed to SI's RemoveArtifacts scales with "
          "total masked samples; a heavily-artifacted hour reaches tens of "
          "millions of int64 frames.")


# --------------------------------------------------------------------------- #
# M3 -- concat split: members x units mask vs vectorized searchsorted (R10)
# --------------------------------------------------------------------------- #
def _split_vectorized(unit_spike_trains, boundaries):
    """Candidate rewrite: one searchsorted per unit (frames are sorted)."""
    bounds = np.asarray(boundaries, dtype=np.int64)
    starts = np.empty_like(bounds)
    starts[0] = 0
    starts[1:] = bounds[:-1]
    per_member = [dict() for _ in bounds]
    for uid, frames in unit_spike_trains.items():
        frames = np.asarray(frames, dtype=np.int64)
        member = np.searchsorted(bounds, frames, side="right")  # monotonic
        for mi in range(bounds.size):
            lo = int(np.searchsorted(member, mi, side="left"))
            hi = int(np.searchsorted(member, mi, side="right"))
            per_member[mi][uid] = (frames[lo:hi] - starts[mi]).astype(
                np.int64, copy=False
            )
    return per_member


def m3_concat_split():
    _section("M3  concat split: current mask vs vectorized searchsorted (R10)")
    try:
        from spyglass.spikesorting.v2._concat_recording import (
            split_unit_spike_trains as current,
        )
    except Exception as exc:  # pragma: no cover - import guard
        print(f"  SKIP: could not import split_unit_spike_trains ({exc})")
        return
    rng = np.random.default_rng(SEED)
    per_member_samples = 9_000_000  # 5 min @ 30 kHz per member
    spikes_per_unit = 20_000
    print(f"{'members':>8} {'units':>6} {'cur ms':>9} {'vec ms':>9} "
          f"{'speedup':>8} {'equal':>6}")
    for n_members in (2, 5, 10, 20):
        for n_units in (50, 200):
            boundaries = [
                (i + 1) * per_member_samples for i in range(n_members)
            ]
            total = boundaries[-1]
            trains = {
                u: np.sort(rng.integers(0, total, size=spikes_per_unit))
                for u in range(n_units)
            }
            cur = current(trains, boundaries)
            vec = _split_vectorized(trains, boundaries)
            equal = all(
                set(cur[m]) == set(vec[m])
                and all(
                    np.array_equal(cur[m][u], vec[m][u]) for u in cur[m]
                )
                for m in range(n_members)
            )
            cur_med, _ = _timeit(lambda: current(trains, boundaries))
            vec_med, _ = _timeit(lambda: _split_vectorized(trains, boundaries))
            print(f"{n_members:>8} {n_units:>6} {cur_med*1e3:>9.1f} "
                  f"{vec_med*1e3:>9.1f} {cur_med/vec_med:>7.2f}x {str(equal):>6}")


# --------------------------------------------------------------------------- #
# M4 -- UnitMatch _pairs_from_matrix dense n x n allocations (R11)
# --------------------------------------------------------------------------- #
def m4_unitmatch_dense():
    _section("M4  UnitMatch _pairs_from_matrix dense n x n allocations (R11)")
    rng = np.random.default_rng(SEED)
    print(f"{'n_units':>8} {'prob MB':>9} {'mask-build peak MB':>20}")
    for n in (500, 1000, 2000):
        prob = rng.random((n, n)).astype(np.float64)
        thr = 0.5

        def build():
            upper = np.triu(np.ones((n, n), dtype=bool), k=1)
            cross = np.ones((n, n), dtype=bool)
            both = (prob > thr) & (prob.T > thr)
            mask = upper & cross & both
            mean_prob = (prob + prob.T) / 2.0
            return mask, mean_prob

        peak = _peak_mb(build)
        print(f"{n:>8} {prob.nbytes/2**20:>9.1f} {peak:>20.1f}")
        del prob
        gc.collect()
    print("\n  -> several dense n x n arrays on top of UnitMatch's own n x n "
          "prob matrix; upper-tri indices would allocate only candidate pairs.")


# --------------------------------------------------------------------------- #
# M5 -- SI global default n_jobs (R1)
# --------------------------------------------------------------------------- #
def m5_njobs_default():
    _section("M5  SpikeInterface global default n_jobs (R1)")
    import spikeinterface.full as si

    jk = si.get_global_job_kwargs()
    print(f"  si.get_global_job_kwargs() = {jk}")
    print(f"  -> default n_jobs = {jk.get('n_jobs')!r} (1 == single process)")


# --------------------------------------------------------------------------- #
# M6 -- NWB open overhead (R9)
# --------------------------------------------------------------------------- #
def m6_nwb_open():
    _section("M6  NWBHDF5IO open+read overhead per call (R9)")
    import tempfile
    from datetime import datetime, timezone
    from pathlib import Path

    import pynwb

    tmp = Path(tempfile.mkdtemp()) / "units.nwb"
    nwbf = pynwb.NWBFile(
        session_description="b",
        identifier="bench",
        session_start_time=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )
    for uid in range(200):
        nwbf.add_unit(id=uid, spike_times=list(np.sort(
            np.random.default_rng(uid).random(500))))
    with pynwb.NWBHDF5IO(path=str(tmp), mode="w") as io:
        io.write(nwbf)

    def open_read():
        with pynwb.NWBHDF5IO(path=str(tmp), mode="r", load_namespaces=True) as io:
            f = io.read()
            _ = f.units.id[:]

    med, mn = _timeit(open_read)
    print(f"  units NWB (200 units): open+read median={med*1e3:.1f} ms "
          f"min={mn*1e3:.1f} ms  (x2 when a call opens it twice)")


def main():
    print("Spikesorting v2 efficiency microbenchmarks")
    import spikeinterface as si

    print(f"  python={platform.python_version()} numpy={np.__version__} "
          f"spikeinterface={si.__version__}")
    print(f"  platform={platform.platform()}")
    print(f"  SEED={SEED} REPEATS={REPEATS} N_REF={N_REF:,}")
    m5_njobs_default()
    m1_get_times()
    m2_artifact_frames()
    m4_unitmatch_dense()
    m3_concat_split()
    m6_nwb_open()
    print("\nDONE")


if __name__ == "__main__":
    main()

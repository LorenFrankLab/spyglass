"""Pre-refactor baseline-artifact capture for v2 spike-sorting regression
tests.

Why this file exists
--------------------
The v2 tri-part refactor changes the internal shape of
``Recording.make`` / ``ArtifactDetection.make`` / ``Sorting.make`` from
monolithic to tri-part, and switches the ``Recording`` write path to a
streaming HDF5 iterator. Several v2 surfaces are expected to remain
bit-equivalent through the refactor: the ``Recording`` AnalysisNwbfile's
trace + timestamps arrays under deterministic preprocessing, and the
``clusterless_thresholder`` sorter's per-unit spike samples (no
whitening, deterministic peak detection).

To prove the refactor preserves those bits we need a baseline captured
on the **unmodified pre-refactor code**. This module is the capture
machinery:

    1. ``regenerate(...)`` runs ``run_v2_pipeline`` against the 60s MEArec
       polymer fixture under ``preset="franklab_tetrode_clusterless_thresholder"``
       and writes a small bundle of numpy / json files to
       ``tests/spikesorting/v2/_fixtures/phase1_baseline/``.
    2. ``load(...)`` reads that bundle off disk without touching the
       database. Validation tests load through this.

The choice of ``clusterless_thresholder`` is deliberate -- it is the only
shipped sorter whose output is deterministic AND does not depend on
whitening, so it is the natural bit-equivalence gate through the
external-whitening change. Stochastic sorters (MS4 / MS5 / KS4) are
re-baselined separately, not against this bundle.

Regeneration workflow
---------------------
The bundle MUST be regenerated on pre-refactor tip code, before any v2
edits land. The check-in workflow:

    1. ``git stash`` any in-progress refactor edits.
    2. ``git rev-parse HEAD`` and record the SHA -- this becomes the
       baseline-source SHA in ``MANIFEST.json``.
    3. ``rm -rf tests/spikesorting/v2/_fixtures/phase1_baseline/``.
    4. ``pytest tests/spikesorting/v2/test_phase1_baseline_regen.py -q``
       (single-test module that calls ``regenerate``).
    5. ``git diff --stat tests/spikesorting/v2/_fixtures/phase1_baseline/``
       to confirm the bundle was written.
    6. ``git stash pop`` to restore in-progress edits.

The MANIFEST records ``spikeinterface``, ``numpy``, and ``pynwb`` versions
from the regen environment. The validation tests verify the MANIFEST
matches the current environment before comparing baselines; an SI version
drift fails loudly rather than producing a misleading false negative.
"""

from __future__ import annotations

import json
import os
import pickle
import resource
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib.metadata import version as _pkg_version
from pathlib import Path

import numpy as np

# Layout of the on-disk bundle. Keep names short and stable; rename-or-move
# is a breaking change that requires re-regeneration.
PHASE1_BASELINE_DIR = Path(__file__).resolve().parent / "phase1_baseline"
MANIFEST_PATH = PHASE1_BASELINE_DIR / "MANIFEST.json"
RECORDING_NPZ = PHASE1_BASELINE_DIR / "recording_artifacts.npz"
SORTING_PICKLE = PHASE1_BASELINE_DIR / "sorting_spike_times.pkl"
CURATION_PICKLE = PHASE1_BASELINE_DIR / "curation_spike_times.pkl"
STAGE_METRICS_PATH = PHASE1_BASELINE_DIR / "stage_metrics.json"

# 60s MEArec polymer fixture path is shared with the existing
# ``polymer_60s_session`` test fixture. Resolved at call time so the
# fixture-availability check is honest about disk state.
_POLYMER_60S_PATH = (
    Path(__file__).resolve().parent.parent
    / "fixtures"
    / "mearec_polymer_128ch_60s.nwb"
)


@dataclass(frozen=True)
class RecordingBaseline:
    """Pre-refactor byte-equivalence reference for the Recording artifact."""

    cache_hash: str
    object_id: str
    analysis_file_name: str
    traces: np.ndarray  # (n_samples, n_channels) re-read from AnalysisNwbfile
    timestamps: np.ndarray  # (n_samples,) re-read from AnalysisNwbfile


@dataclass(frozen=True)
class SortingBaseline:
    """Pre-refactor reference for the Sorting NWB output."""

    object_id: str
    analysis_file_name: str
    sampling_frequency: float
    # {unit_id: spike-sample ndarray, ints}; clusterless_thresholder
    # returns sample-domain peaks, so we keep them in samples rather than
    # seconds for exact-equality comparison.
    spike_samples_per_unit: dict[int, np.ndarray]


@dataclass(frozen=True)
class CurationBaseline:
    """Pre-refactor reference for the CurationV2 NWB output."""

    object_id: str
    analysis_file_name: str
    sampling_frequency: float
    # {unit_id: spike-time ndarray in seconds, float64}. The curated NWB
    # writes times in seconds via ``add_unit(spike_times=...)``, so we
    # compare in seconds with a 1.5 / fs tolerance per the plan.
    spike_times_per_unit: dict[int, np.ndarray]


@dataclass(frozen=True)
class StageMetrics:
    """Wall-clock + peak-RSS captured around each populate stage.

    These are informational, not gates -- thermal throttling and CI runner
    variance make wall-clock unreliable. Stored for trend-spotting only.
    """

    recording_wall_s: float
    artifact_wall_s: float
    sorting_wall_s: float
    curation_wall_s: float
    peak_rss_kb: int  # max across the whole populate run, via ru_maxrss


@dataclass(frozen=True)
class Phase1Baseline:
    """Container for everything the regen step writes to disk."""

    manifest: dict
    recording: RecordingBaseline
    sorting: SortingBaseline
    curation: CurationBaseline
    metrics: StageMetrics

    # Convenience for tests that only need one slice.
    @property
    def cache_hash(self) -> str:
        return self.recording.cache_hash


def fixture_available() -> bool:
    """Return True if the raw 60s polymer NWB fixture is present on disk."""
    return _POLYMER_60S_PATH.exists()


def baseline_present() -> bool:
    """Return True if a previously-captured baseline bundle is on disk."""
    return all(
        p.exists()
        for p in (
            MANIFEST_PATH,
            RECORDING_NPZ,
            SORTING_PICKLE,
            CURATION_PICKLE,
            STAGE_METRICS_PATH,
        )
    )


def _current_environment_manifest(source_sha: str | None) -> dict:
    """Snapshot package versions + git SHA so a drift fails loudly later."""
    return {
        "source_sha": source_sha or _resolve_git_head(),
        "spikeinterface": _pkg_version("spikeinterface"),
        "numpy": _pkg_version("numpy"),
        "pynwb": _pkg_version("pynwb"),
        "spyglass": _pkg_version("spyglass-neuro"),
        "python": ".".join(map(str, sys.version_info[:3])),
        "preset": "franklab_tetrode_clusterless_thresholder",
        "fixture_filename": _POLYMER_60S_PATH.name,
    }


def _resolve_git_head() -> str | None:
    """Best-effort git SHA at regen time; ``None`` outside a worktree."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parent),
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.SubprocessError, OSError, FileNotFoundError):
        return None
    return sha.decode("ascii").strip() or None


@contextmanager
def _stage_timer(label: str, sink: dict[str, float]):
    """Record wall-clock seconds for the wrapped block under ``label``."""
    start = time.monotonic()
    try:
        yield
    finally:
        sink[label] = time.monotonic() - start


def regenerate(
    *,
    nwb_file_name: str,
    sort_group_id: int,
    team_name: str,
    interval_list_name: str = "raw data valid times",
    description: str = "phase1 baseline capture",
    output_dir: Path = PHASE1_BASELINE_DIR,
) -> Phase1Baseline:
    """Run the clusterless preset and persist the pre-refactor baseline bundle.

    This is the **slow** path; it runs the full ``Recording -> Artifact
    -> Sort -> Curation`` chain inside the isolated test database. The
    caller is expected to have already ingested the 60s MEArec polymer
    fixture via ``copy_and_insert_nwb`` and to have created the
    ``SortGroupV2`` rows for ``nwb_file_name`` (the existing
    ``polymer_60s_session`` and ``polymer_smoke_session`` fixtures handle
    the ingest step; the regen test sets up the sort groups).

    Parameters
    ----------
    nwb_file_name
        The ingested NWB filename (e.g. ``"mearec_polymer_128ch_60s_.nwb"``).
    sort_group_id
        ID of an existing ``SortGroupV2`` row for ``nwb_file_name``.
    team_name
        LabTeam owning the sort.
    interval_list_name
        Defaults to ``"raw data valid times"`` for full-session.
    description
        Free-text description passed to ``CurationV2.insert_curation``.
    output_dir
        Override for the bundle directory; defaults to
        ``tests/spikesorting/v2/_fixtures/phase1_baseline/``.

    Returns
    -------
    Phase1Baseline
        In-memory copy of the artifacts that were just written to disk.
    """
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.pipeline import run_v2_pipeline
    from spyglass.spikesorting.v2.recording import Recording
    from spyglass.spikesorting.v2.sorting import Sorting

    output_dir.mkdir(parents=True, exist_ok=True)

    stage_wall_s: dict[str, float] = {}
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # We run ``run_v2_pipeline`` once and time the populate stages
    # individually by monkey-patching the .populate calls. Simpler: just
    # call run_v2_pipeline as a single block (it's idempotent and short
    # enough that finer breakdown isn't needed for a smoke baseline).
    with _stage_timer("pipeline_wall_s", stage_wall_s):
        manifest = run_v2_pipeline(
            nwb_file_name=nwb_file_name,
            sort_group_id=sort_group_id,
            interval_list_name=interval_list_name,
            team_name=team_name,
            preset="franklab_tetrode_clusterless_thresholder",
            description=description,
        )
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_kb = int(max(rss_before, rss_after))

    recording_baseline = _capture_recording_baseline(
        recording_id=manifest["recording_id"]
    )
    sorting_baseline = _capture_sorting_baseline(
        sorting_id=manifest["sorting_id"]
    )
    curation_baseline = _capture_curation_baseline(
        curation_pk={
            "sorting_id": manifest["sorting_id"],
            "curation_id": manifest["curation_id"],
        }
    )
    stage_metrics = StageMetrics(
        # The pipeline call lumps all stages; record the total under
        # ``sorting_wall_s`` and leave the rest as 0.0 informational
        # placeholders so the dataclass stays stable across runs. A
        # finer breakdown is follow-up work; the gates that care about
        # parallel populate already capture per-stage timing themselves.
        recording_wall_s=0.0,
        artifact_wall_s=0.0,
        sorting_wall_s=stage_wall_s.get("pipeline_wall_s", 0.0),
        curation_wall_s=0.0,
        peak_rss_kb=peak_rss_kb,
    )

    # Sanity probes: stochastic-sorter check is not relevant here because
    # the preset is clusterless_thresholder, which is deterministic.
    # n_units should be > 0 on the 60s polymer fixture; record but do not
    # gate (gating is the validation slice's job).
    if not sorting_baseline.spike_samples_per_unit:
        raise RuntimeError(
            "Pre-refactor baseline regen: sorting produced zero units "
            "on the 60s polymer fixture. Inspect the "
            "clusterless_thresholder default-row threshold; aborting "
            "before writing a useless baseline."
        )

    manifest_blob = _current_environment_manifest(source_sha=None)
    manifest_blob.update(
        {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": int(sort_group_id),
            "interval_list_name": interval_list_name,
            "team_name": team_name,
            "recording_id": str(manifest["recording_id"]),
            "artifact_id": str(manifest["artifact_id"]),
            "sorting_id": str(manifest["sorting_id"]),
            "curation_id": int(manifest["curation_id"]),
            "merge_id": str(manifest["merge_id"]),
            "n_units": len(sorting_baseline.spike_samples_per_unit),
            "n_curated_units": len(curation_baseline.spike_times_per_unit),
        }
    )

    _write_bundle(
        output_dir=output_dir,
        manifest_blob=manifest_blob,
        recording=recording_baseline,
        sorting=sorting_baseline,
        curation=curation_baseline,
        metrics=stage_metrics,
    )

    # Verify the round-trip by reading what we just wrote -- a mismatch
    # here indicates the writer or reader is broken before any
    # validation test even runs.
    reloaded = load(output_dir=output_dir)
    _assert_round_trip(
        reloaded, recording_baseline, sorting_baseline, curation_baseline
    )
    return reloaded


def _capture_recording_baseline(*, recording_id) -> RecordingBaseline:
    """Re-read the Recording's AnalysisNwbfile and snapshot trace+timestamps."""
    import pynwb

    from spyglass.common import AnalysisNwbfile
    from spyglass.spikesorting.v2.recording import Recording

    row = (Recording & {"recording_id": recording_id}).fetch1()
    analysis_path = AnalysisNwbfile.get_abs_path(row["analysis_file_name"])
    with pynwb.NWBHDF5IO(
        path=analysis_path, mode="r", load_namespaces=True
    ) as io:
        nwbfile = io.read()
        # The pre-refactor writer puts the ElectricalSeries under acquisition.
        # We look it up by object_id rather than name so the test does
        # not need the private ``_ELECTRICAL_SERIES_NAME`` constant.
        series = nwbfile.objects[row["object_id"]]
        traces = np.asarray(series.data[:])
        timestamps = np.asarray(series.timestamps[:])
    return RecordingBaseline(
        cache_hash=str(row["cache_hash"]),
        object_id=str(row["object_id"]),
        analysis_file_name=str(row["analysis_file_name"]),
        traces=traces,
        timestamps=timestamps,
    )


def _capture_sorting_baseline(*, sorting_id) -> SortingBaseline:
    """Re-read the Sorting's units NWB and snapshot per-unit spike samples.

    For ``clusterless_thresholder`` the units NWB writes spike times in
    seconds (consistent across sorters); we convert to integer samples
    via the recording's sampling frequency so byte-equivalence comparison
    is exact within float64 round-trip tolerance.
    """
    from spyglass.spikesorting.v2.sorting import Sorting

    row = (Sorting & {"sorting_id": sorting_id}).fetch1()
    sorting_obj = Sorting().get_sorting({"sorting_id": sorting_id})
    fs = float(sorting_obj.get_sampling_frequency())
    spike_samples = {
        int(uid): np.asarray(
            sorting_obj.get_unit_spike_train(unit_id=uid), dtype=np.int64
        )
        for uid in sorting_obj.get_unit_ids()
    }
    return SortingBaseline(
        object_id=str(row["object_id"]),
        analysis_file_name=str(row["analysis_file_name"]),
        sampling_frequency=fs,
        spike_samples_per_unit=spike_samples,
    )


def _capture_curation_baseline(*, curation_pk) -> CurationBaseline:
    """Re-read the CurationV2 NWB and snapshot per-unit spike times (s).

    Curated NWBs may shrink the unit set (merges, label drops); we capture
    whatever the curated NWB carries, so a regression that accidentally
    removes more units would show up as a missing key.
    """
    import pynwb

    from spyglass.common import AnalysisNwbfile
    from spyglass.spikesorting.v2.curation import CurationV2

    row = (CurationV2 & curation_pk).fetch1()
    analysis_path = AnalysisNwbfile.get_abs_path(row["analysis_file_name"])
    with pynwb.NWBHDF5IO(
        path=analysis_path, mode="r", load_namespaces=True
    ) as io:
        nwbfile = io.read()
        units = nwbfile.units
        unit_ids = list(np.asarray(units["id"][:], dtype=np.int64))
        spike_times = [
            np.asarray(units["spike_times"][i]) for i in range(len(unit_ids))
        ]
    by_unit = {int(uid): st for uid, st in zip(unit_ids, spike_times)}
    return CurationBaseline(
        object_id=str(row["object_id"]),
        analysis_file_name=str(row["analysis_file_name"]),
        # Sampling rate lives on the Sorting row, not the CurationV2
        # row directly; pull it through the linked Sorting.
        sampling_frequency=_curation_sampling_frequency(curation_pk),
        spike_times_per_unit=by_unit,
    )


def _curation_sampling_frequency(curation_pk: dict) -> float:
    """Resolve the upstream Sorting's sampling frequency from the curation PK."""
    from spyglass.spikesorting.v2.sorting import Sorting

    sorting_obj = Sorting().get_sorting(
        {"sorting_id": curation_pk["sorting_id"]}
    )
    return float(sorting_obj.get_sampling_frequency())


def _write_bundle(
    *,
    output_dir: Path,
    manifest_blob: dict,
    recording: RecordingBaseline,
    sorting: SortingBaseline,
    curation: CurationBaseline,
    metrics: StageMetrics,
) -> None:
    """Serialize the bundle to ``output_dir``."""
    MANIFEST_PATH_LOCAL = output_dir / MANIFEST_PATH.name
    RECORDING_NPZ_LOCAL = output_dir / RECORDING_NPZ.name
    SORTING_PICKLE_LOCAL = output_dir / SORTING_PICKLE.name
    CURATION_PICKLE_LOCAL = output_dir / CURATION_PICKLE.name
    STAGE_METRICS_LOCAL = output_dir / STAGE_METRICS_PATH.name

    MANIFEST_PATH_LOCAL.write_text(
        json.dumps(manifest_blob, indent=2, sort_keys=True)
    )
    np.savez_compressed(
        RECORDING_NPZ_LOCAL,
        traces=recording.traces,
        timestamps=recording.timestamps,
        cache_hash=np.array([recording.cache_hash]),
        object_id=np.array([recording.object_id]),
        analysis_file_name=np.array([recording.analysis_file_name]),
    )
    with open(SORTING_PICKLE_LOCAL, "wb") as handle:
        pickle.dump(
            {
                "object_id": sorting.object_id,
                "analysis_file_name": sorting.analysis_file_name,
                "sampling_frequency": sorting.sampling_frequency,
                "spike_samples_per_unit": sorting.spike_samples_per_unit,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    with open(CURATION_PICKLE_LOCAL, "wb") as handle:
        pickle.dump(
            {
                "object_id": curation.object_id,
                "analysis_file_name": curation.analysis_file_name,
                "sampling_frequency": curation.sampling_frequency,
                "spike_times_per_unit": curation.spike_times_per_unit,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    STAGE_METRICS_LOCAL.write_text(
        json.dumps(
            {
                "recording_wall_s": metrics.recording_wall_s,
                "artifact_wall_s": metrics.artifact_wall_s,
                "sorting_wall_s": metrics.sorting_wall_s,
                "curation_wall_s": metrics.curation_wall_s,
                "peak_rss_kb": metrics.peak_rss_kb,
            },
            indent=2,
            sort_keys=True,
        )
    )


def load(*, output_dir: Path = PHASE1_BASELINE_DIR) -> Phase1Baseline:
    """Read the previously-captured bundle off disk.

    Raises ``FileNotFoundError`` if the bundle is missing; callers should
    skip the test cleanly with a pointer to the regen workflow.
    """
    MANIFEST_PATH_LOCAL = output_dir / MANIFEST_PATH.name
    RECORDING_NPZ_LOCAL = output_dir / RECORDING_NPZ.name
    SORTING_PICKLE_LOCAL = output_dir / SORTING_PICKLE.name
    CURATION_PICKLE_LOCAL = output_dir / CURATION_PICKLE.name
    STAGE_METRICS_LOCAL = output_dir / STAGE_METRICS_PATH.name

    missing = [
        p
        for p in (
            MANIFEST_PATH_LOCAL,
            RECORDING_NPZ_LOCAL,
            SORTING_PICKLE_LOCAL,
            CURATION_PICKLE_LOCAL,
            STAGE_METRICS_LOCAL,
        )
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Pre-refactor baseline bundle is incomplete; missing files: "
            f"{[str(p) for p in missing]}. Regenerate via "
            "`pytest tests/spikesorting/v2/test_phase1_baseline_regen.py -q` "
            "from a clean checkout of the pre-refactor tip."
        )

    manifest = json.loads(MANIFEST_PATH_LOCAL.read_text())
    with np.load(RECORDING_NPZ_LOCAL, allow_pickle=False) as bundle:
        recording = RecordingBaseline(
            cache_hash=str(bundle["cache_hash"][0]),
            object_id=str(bundle["object_id"][0]),
            analysis_file_name=str(bundle["analysis_file_name"][0]),
            traces=np.asarray(bundle["traces"]),
            timestamps=np.asarray(bundle["timestamps"]),
        )
    with open(SORTING_PICKLE_LOCAL, "rb") as handle:
        s = pickle.load(handle)
    sorting = SortingBaseline(
        object_id=str(s["object_id"]),
        analysis_file_name=str(s["analysis_file_name"]),
        sampling_frequency=float(s["sampling_frequency"]),
        spike_samples_per_unit={
            int(k): np.asarray(v, dtype=np.int64)
            for k, v in s["spike_samples_per_unit"].items()
        },
    )
    with open(CURATION_PICKLE_LOCAL, "rb") as handle:
        c = pickle.load(handle)
    curation = CurationBaseline(
        object_id=str(c["object_id"]),
        analysis_file_name=str(c["analysis_file_name"]),
        sampling_frequency=float(c["sampling_frequency"]),
        spike_times_per_unit={
            int(k): np.asarray(v, dtype=np.float64)
            for k, v in c["spike_times_per_unit"].items()
        },
    )
    m = json.loads(STAGE_METRICS_LOCAL.read_text())
    metrics = StageMetrics(
        recording_wall_s=float(m["recording_wall_s"]),
        artifact_wall_s=float(m["artifact_wall_s"]),
        sorting_wall_s=float(m["sorting_wall_s"]),
        curation_wall_s=float(m["curation_wall_s"]),
        peak_rss_kb=int(m["peak_rss_kb"]),
    )
    return Phase1Baseline(
        manifest=manifest,
        recording=recording,
        sorting=sorting,
        curation=curation,
        metrics=metrics,
    )


def verify_manifest_compatible(manifest: dict) -> list[str]:
    """Return a list of mismatch descriptions between baseline and current env.

    Empty list means the current environment matches the baseline's
    captured manifest. The validation slice surfaces non-empty results
    as a clear skip / xfail.
    """
    current = _current_environment_manifest(source_sha=None)
    mismatches: list[str] = []
    for key in ("spikeinterface", "numpy", "pynwb"):
        if manifest.get(key) != current.get(key):
            mismatches.append(
                f"{key}: baseline={manifest.get(key)!r}, "
                f"current={current.get(key)!r}"
            )
    return mismatches


def _assert_round_trip(
    reloaded: Phase1Baseline,
    recording: RecordingBaseline,
    sorting: SortingBaseline,
    curation: CurationBaseline,
) -> None:
    """Sanity-check that what was written matches what we just held in RAM."""
    if reloaded.recording.cache_hash != recording.cache_hash:
        raise RuntimeError(
            "Pre-refactor baseline round-trip: cache_hash drifted."
        )
    if not np.array_equal(reloaded.recording.traces, recording.traces):
        raise RuntimeError("Pre-refactor baseline round-trip: traces differ.")
    if not np.array_equal(reloaded.recording.timestamps, recording.timestamps):
        raise RuntimeError(
            "Pre-refactor baseline round-trip: timestamps differ."
        )
    if (
        reloaded.sorting.spike_samples_per_unit.keys()
        != sorting.spike_samples_per_unit.keys()
    ):
        raise RuntimeError(
            "Pre-refactor baseline round-trip: unit_id sets differ."
        )
    for uid, arr in sorting.spike_samples_per_unit.items():
        if not np.array_equal(
            reloaded.sorting.spike_samples_per_unit[uid], arr
        ):
            raise RuntimeError(
                f"Pre-refactor baseline round-trip: unit {uid} spike samples differ."
            )
    if (
        reloaded.curation.spike_times_per_unit.keys()
        != curation.spike_times_per_unit.keys()
    ):
        raise RuntimeError(
            "Pre-refactor baseline round-trip: curation unit_id sets differ."
        )
    # Per-unit value check too: a serialization regression that
    # changes spike times without changing the unit count would
    # otherwise pass silently.
    for uid, arr in curation.spike_times_per_unit.items():
        if not np.array_equal(reloaded.curation.spike_times_per_unit[uid], arr):
            raise RuntimeError(
                f"Pre-refactor baseline round-trip: curated unit {uid} spike times differ."
            )

"""DB-free experiment: damage from zero-filled cross-validation halves in the
spyglass UnitMatch bundle (CURRENT: recording split at num_samples//2) vs a
per-unit temporal split of each unit's own sampled spikes (FIXED).

Run:
    cd /Users/edeno/Documents/GitHub/spyglass
    PYTHONPATH=src <env>/bin/python run_experiment.py --seeds 0 1

Outputs land next to this script under results/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

import spikeinterface as si  # noqa: E402

from spyglass.spikesorting.v2._unitmatch_backend import (  # noqa: E402
    UnitMatchBackend,
    _require_unitmatch,
    _zero_center,
    extract_unitmatch_bundle,
)
from spyglass.spikesorting.v2.matcher_protocol import (  # noqa: E402
    SessionMatcherInput,
)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

FS = 30_000.0
DURATION_S = 120.0
N_CHANNELS = 16
N_UNITS = 20
MS_BEFORE = MS_AFTER = 1.5
MAX_SPIKES_PER_HALF = 100
BUNDLE_SEED = 0
JOB_KWARGS = {"n_jobs": 1, "progress_bar": False}


# ----------------------------------------------------------------------------
# Synthetic data
# ----------------------------------------------------------------------------
def make_dataset(seed: int):
    rec, sort = si.generate_ground_truth_recording(
        durations=[DURATION_S],
        sampling_frequency=FS,
        num_channels=N_CHANNELS,
        num_units=N_UNITS,
        generate_probe_kwargs={
            "num_columns": 2,
            "xpitch": 20,
            "ypitch": 20,
            "contact_shapes": "circle",
            "contact_shape_params": {"radius": 6},
        },
        generate_sorting_kwargs={"firing_rates": 10.0, "refractory_period_ms": 4.0},
        noise_kwargs={"noise_levels": 5.0, "strategy": "on_the_fly"},
        seed=seed,
    )
    # SI generates string unit ids; the backend names bundle files with
    # ``np.asarray(unit_ids, dtype=int)``, so use int ids throughout.
    sort = sort.rename_units(np.arange(len(sort.get_unit_ids()), dtype=int))
    return rec, sort


def split_sessions(rec, sort):
    half = int(DURATION_S * FS) // 2
    n = rec.get_num_samples()
    rec_a, sort_a = rec.frame_slice(0, half), sort.frame_slice(0, half)
    rec_b, sort_b = rec.frame_slice(half, n), sort.frame_slice(half, n)
    return (rec_a, sort_a), (rec_b, sort_b)


def drop_spikes(sorting, unit_subset, keep):
    """Return a NumpySorting where units in ``unit_subset`` keep only spikes
    with ``keep(frames)`` True. Other units are untouched. Unit order kept."""
    d = {}
    for uid in sorting.get_unit_ids():
        st = sorting.get_unit_spike_train(uid)
        if uid in unit_subset:
            st = st[keep(st)]
        d[uid] = st.astype(np.int64)
    out = si.NumpySorting.from_unit_dict(d, sorting.sampling_frequency)
    return out


# ----------------------------------------------------------------------------
# FIXED bundle: per-unit temporal split of the unit's own sampled spikes
# ----------------------------------------------------------------------------
def extract_fixed_bundle(session_dir, recording, sorting, *, ms_before, ms_after,
                         max_spikes_per_unit_total, seed):
    um = _require_unitmatch()
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    probe = recording.get_probe()
    if probe.ndim == 3:
        recording = recording.set_probe(probe.to_2d())
    channel_positions = recording.get_channel_locations()

    analyzer = si.create_sorting_analyzer(sorting, recording, sparse=False)
    analyzer.compute("random_spikes", method="uniform",
                     max_spikes_per_unit=max_spikes_per_unit_total, seed=seed)
    analyzer.compute("waveforms", ms_before=ms_before, ms_after=ms_after, **JOB_KWARGS)
    wf = analyzer.get_extension("waveforms").get_data()  # (n_sel, n_samp, n_ch)
    some_spikes = analyzer.get_extension("random_spikes").get_random_spikes()
    unit_ids = np.asarray(sorting.get_unit_ids(), dtype=int)
    n_samp, n_ch = wf.shape[1], wf.shape[2]
    avg = np.zeros((len(unit_ids), n_samp, n_ch, 2))
    for u_idx in range(len(unit_ids)):
        mask = some_spikes["unit_index"] == u_idx
        w = wf[mask]
        s = some_spikes["sample_index"][mask]
        w = w[np.argsort(s, kind="stable")]
        n = w.shape[0]
        h = n // 2
        if h > 0:
            avg[u_idx, :, :, 0] = w[:h].mean(axis=0)
        if n - h > 0:
            avg[u_idx, :, :, 1] = w[h:].mean(axis=0)
    np.save(session_dir / "channel_positions.npy", channel_positions)
    um.extract_raw_data.save_avg_waveforms(
        avg, str(session_dir), unit_ids, unit_ids, extract_good_units_only=False
    )
    rows = [np.array(("cluster_id", "group"))] + [
        np.array((str(i), "good")) for i in unit_ids
    ]
    np.savetxt(session_dir / "cluster_group.tsv", np.vstack(rows),
               fmt=["%s", "%s"], delimiter="\t")


def extract_current_bundle(session_dir, recording, sorting):
    extract_unitmatch_bundle(
        session_dir, recording, sorting,
        ms_before=MS_BEFORE, ms_after=MS_AFTER,
        max_spikes_per_unit=MAX_SPIKES_PER_HALF, seed=BUNDLE_SEED,
        job_kwargs=JOB_KWARGS,
    )


def load_bundle(session_dir, unit_ids):
    return np.stack(
        [np.load(Path(session_dir) / "RawWaveforms" / f"Unit{u}_RawSpikes.npy")
         for u in unit_ids]
    )


# ----------------------------------------------------------------------------
# UnitMatch inference replicated from UnitMatchBackend.match (with recorders)
# ----------------------------------------------------------------------------
def run_unitmatch(session_dirs):
    """Replicates UnitMatchBackend.match line by line and returns diagnostics."""
    um = _require_unitmatch()
    mf = um.overlord.mf
    rec = {"drift": [], "thresholds": []}

    orig_drift = mf.drift_n_sessions
    orig_thr = mf.get_threshold

    def drift_rec(*a, **k):
        out = orig_drift(*a, **k)
        rec["drift"].append(np.asarray(out[0]).tolist())
        return out

    def thr_rec(total_score, within_session, euclid_dist, param, is_first_pass=True):
        out = orig_thr(total_score, within_session, euclid_dist, param, is_first_pass)
        rec["thresholds"].append(float(np.asarray(out).squeeze()))
        rec["euclid_dist"] = euclid_dist
        return out

    mf.drift_n_sessions = drift_rec
    mf.get_threshold = thr_rec
    try:
        param = um.default_params.get_default_param()
        match_threshold = float(param["match_threshold"])
        raw_positions = np.load(Path(session_dirs[0]) / "channel_positions.npy")
        session_dirs = [str(s) for s in session_dirs]
        param["KS_dirs"] = session_dirs
        wave_paths, label_paths, channel_pos = um.utils.paths_from_KS(session_dirs)
        param = um.utils.get_probe_geometry(raw_positions, param)
        (waveform, session_id, session_switch, within_session, good_units, param
         ) = um.utils.load_good_waveforms(wave_paths, label_paths, param,
                                          good_units_only=True)
        assert len(good_units) == len(session_dirs)
        waveform = _zero_center(waveform)
        clus_info = {
            "good_units": good_units,
            "session_switch": session_switch,
            "session_id": session_id,
            "original_ids": np.concatenate(good_units),
        }
        extracted = um.overlord.extract_parameters(waveform, channel_pos, clus_info, param)
        total_score, candidate_pairs, scores_to_include, predictors = (
            um.overlord.extract_metric_scores(
                extracted, session_switch, within_session, param, niter=2
            )
        )
        prior_match = 1 - (param["n_expected_matches"] / param["n_units"] ** 2)
        priors = np.array((prior_match, 1 - prior_match))
        labels = candidate_pairs.astype(int)
        cond = np.unique(labels)
        kernels = um.bayes_functions.get_parameter_kernels(
            scores_to_include, labels, cond, param, add_one=1
        )
        probability = um.bayes_functions.apply_naive_bayes(
            kernels, priors, predictors, param, cond
        )
        prob_matrix = probability[:, 1].reshape(param["n_units"], param["n_units"])
    finally:
        mf.drift_n_sessions = orig_drift
        mf.get_threshold = orig_thr

    return {
        "prob": prob_matrix,
        "total_score": total_score,
        "candidate_pairs": candidate_pairs,
        "scores": scores_to_include,
        "kernels": kernels,
        "cond": cond,
        "param": param,
        "extracted": extracted,
        "within_session": within_session,
        "session_switch": session_switch,
        "original_ids": clus_info["original_ids"],
        "match_threshold": match_threshold,
        "drift": rec["drift"],
        "thresholds": rec["thresholds"],
        "euclid_dist": rec.get("euclid_dist"),
        "prior_match": float(prior_match),
    }


def backend_pairs(session_dirs):
    """Call the real backend for a cross-check of the replicated path."""
    inputs = [
        SessionMatcherInput(
            curation_key={"sorting_id": f"s{i}", "curation_id": i},
            waveform_dir=Path(d),
            channel_positions_path=Path(d) / "channel_positions.npy",
        )
        for i, d in enumerate(session_dirs)
    ]
    pairs = UnitMatchBackend().match(inputs, params={})
    return {(p.unit_a_id, p.unit_b_id): p.match_probability for p in pairs}


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def auc(pos, neg):
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (pos.size * neg.size))


def evaluate(res, S, n_units, thr):
    prob = res["prob"]
    nA = n_units
    S = set(int(s) for s in S)
    mean_prob = (prob + prob.T) / 2.0
    both = (prob > thr) & (prob.T > thr)
    inS = np.array([i in S for i in range(nA)])

    out = {}
    # true pairs (i, i)
    for name, mask in (("S", inS), ("nonS", ~inS)):
        idx = np.where(mask)[0]
        vals = np.array([mean_prob[i, nA + i] for i in idx])
        passed = np.array([both[i, nA + i] for i in idx])
        out[f"true_{name}"] = {
            "n": int(idx.size),
            "mean_prob": float(vals.mean()) if idx.size else float("nan"),
            "min_prob": float(vals.min()) if idx.size else float("nan"),
            "n_pass": int(passed.sum()),
        }
    # false cross-session pairs
    cross = np.zeros((nA, nA), bool)
    for cat in ("SxS", "SxnonS", "nonSxnonS"):
        vals, npass, maxp = [], 0, 0.0
        for i in range(nA):
            for j in range(nA):
                if i == j:
                    continue
                a, b = i in S, j in S
                c = "SxS" if (a and b) else ("nonSxnonS" if (not a and not b) else "SxnonS")
                if c != cat:
                    continue
                p = mean_prob[i, nA + j]
                vals.append(p)
                if both[i, nA + j]:
                    npass += 1
        vals = np.array(vals)
        out[f"false_{cat}"] = {
            "n_pairs": int(vals.size),
            "n_pass": npass,
            "max_prob": float(vals.max()) if vals.size else float("nan"),
            "mean_prob": float(vals.mean()) if vals.size else float("nan"),
        }
    # AUC per group: true partner prob vs all other B units, for A units in group
    for name, mask in (("S", inS), ("nonS", ~inS)):
        pos, neg = [], []
        for i in np.where(mask)[0]:
            pos.append(mean_prob[i, nA + i])
            neg.extend(mean_prob[i, nA + j] for j in range(nA) if j != i)
        out[f"auc_{name}"] = auc(pos, neg)
    # 1-to-1 hit rate: argmax over B for each A unit equals true partner AND passes
    hits = {"S": 0, "nonS": 0}
    for i in range(nA):
        j = int(np.argmax(mean_prob[i, nA:]))
        if j == i and both[i, nA + i]:
            hits["S" if i in S else "nonS"] += 1
    out["argmax_hits"] = hits
    # within-session self-consistency (diag of total_score = cv0 vs cv1 same unit)
    diag = np.diag(res["total_score"])
    out["diag_total_score"] = {
        "A_S": float(np.nanmean(diag[:nA][inS])) if inS.any() else float("nan"),
        "A_nonS": float(np.nanmean(diag[:nA][~inS])),
        "B_S": float(np.nanmean(diag[nA:][inS])) if inS.any() else float("nan"),
        "B_nonS": float(np.nanmean(diag[nA:][~inS])),
    }
    # calibration state
    lab = res["candidate_pairs"].astype(bool)
    ws = res["within_session"] == 0  # 0 == same session in UnitMatchPy
    out["calibration"] = {
        "n_expected_matches": int(res["param"]["n_expected_matches"]),
        "prior_match": res["prior_match"],
        "thresholds": res["thresholds"],
        "drift": res["drift"],
        "n_candidate_pairs_total": int(lab.sum()),
        "n_candidate_within": int((lab & ws).sum()),
        "n_candidate_cross": int((lab & ~ws).sum()),
        "n_candidate_diag": int(np.diag(lab).sum()),
    }
    # score kernels: mean of the "match" kernel per metric
    sv = res["param"]["score_vector"]
    km = {}
    for k, name in enumerate(res["scores"].keys()):
        cols = {}
        for ci, c in enumerate(res["cond"]):
            kern = res["kernels"][:, k, ci]
            kern = kern / kern.sum()
            cols[f"cond{int(c)}"] = float((sv * kern).sum())
        km[name] = cols
    out["kernel_means"] = km
    # per-metric raw score of the true pairs (mean prob-free view)
    per_metric = {}
    for name, sc in res["scores"].items():
        tS = [sc[i, nA + i] for i in range(nA) if i in S]
        tN = [sc[i, nA + i] for i in range(nA) if i not in S]
        per_metric[name] = {
            "true_S": float(np.nanmean(tS)) if tS else float("nan"),
            "true_nonS": float(np.nanmean(tN)),
        }
    out["per_metric_true"] = per_metric
    # per-unit amplitude per cv for S units in A (diagnostic of zero half)
    amp = res["extracted"]["amplitude"]
    out["amplitude_A_S"] = amp[:nA][inS].tolist() if inS.any() else []
    out["amplitude_A_nonS_mean"] = amp[:nA][~inS].mean(axis=0).tolist()
    return out


# ----------------------------------------------------------------------------
def run_scenario(seed, scenario, S, base_rec, base_sort, log):
    t0 = time.time()
    (rec_a, sort_a), (rec_b, sort_b) = split_sessions(base_rec, base_sort)
    half_frames = rec_a.get_num_samples() // 2
    if scenario in ("driftout_A", "driftout_AB"):
        sort_a = drop_spikes(sort_a, set(S), lambda st: st < half_frames)
    if scenario == "driftout_AB":
        sort_b = drop_spikes(sort_b, set(S), lambda st: st >= half_frames)
    unit_ids = np.asarray(base_sort.get_unit_ids(), dtype=int)
    spike_counts = {
        "A": {int(u): int(sort_a.get_unit_spike_train(u).size) for u in unit_ids},
        "B": {int(u): int(sort_b.get_unit_spike_train(u).size) for u in unit_ids},
    }
    root = RESULTS / f"seed{seed}" / scenario
    out = {"seed": seed, "scenario": scenario, "S": [int(s) for s in S],
           "spike_counts": spike_counts, "timings_s": {}}
    conditions = {}
    for cond in ("current", "fixed"):
        dirs = []
        tc = time.time()
        for lbl, (r, s) in (("A", (rec_a, sort_a)), ("B", (rec_b, sort_b))):
            d = root / cond / lbl
            if cond == "current":
                extract_current_bundle(d, r, s)
            else:
                extract_fixed_bundle(d, r, s, ms_before=MS_BEFORE, ms_after=MS_AFTER,
                                     max_spikes_per_unit_total=2 * MAX_SPIKES_PER_HALF,
                                     seed=BUNDLE_SEED)
            dirs.append(d)
        out["timings_s"][f"{cond}_bundle"] = time.time() - tc
        # zero-half verification
        zero = {}
        for lbl, d in zip(("A", "B"), dirs):
            wf = load_bundle(d, unit_ids)
            zero[lbl] = {
                "half0_allzero": [int(u) for u, w in zip(unit_ids, wf) if np.all(w[..., 0] == 0)],
                "half1_allzero": [int(u) for u, w in zip(unit_ids, wf) if np.all(w[..., 1] == 0)],
            }
        tm = time.time()
        res = run_unitmatch(dirs)
        out["timings_s"][f"{cond}_match"] = time.time() - tm
        ev = evaluate(res, S, len(unit_ids), res["match_threshold"])
        ev["zero_halves"] = zero
        # cross-check the replicated path against the real backend
        bp = backend_pairs(dirs)
        mean_prob = (res["prob"] + res["prob"].T) / 2
        both = (res["prob"] > res["match_threshold"]) & (res["prob"].T > res["match_threshold"])
        nA = len(unit_ids)
        mine = {(int(unit_ids[i]), int(unit_ids[j])): float(mean_prob[i, nA + j])
                for i in range(nA) for j in range(nA) if both[i, nA + j]}
        ev["backend_crosscheck"] = {
            "n_backend_pairs": len(bp),
            "n_replicated_pairs": len(mine),
            "same_keys": set(bp) == set(mine),
            "max_abs_prob_diff": float(max((abs(bp[k] - mine[k]) for k in bp if k in mine), default=0.0)),
        }
        np.save(root / cond / "prob_matrix.npy", res["prob"])
        np.save(root / cond / "total_score.npy", res["total_score"])
        conditions[cond] = ev
    out["conditions"] = conditions
    out["timings_s"]["scenario_total"] = time.time() - t0
    log(f"seed={seed} scenario={scenario} done in {out['timings_s']['scenario_total']:.1f}s")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--scenarios", nargs="+",
                    default=["control", "driftout_A", "driftout_AB"])
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    all_out = []

    def log(msg):
        print(msg, flush=True)

    for seed in args.seeds:
        t = time.time()
        rec, sort = make_dataset(seed)
        log(f"seed={seed}: dataset generated in {time.time() - t:.1f}s; "
            f"{rec.get_num_channels()} ch, {len(sort.get_unit_ids())} units, "
            f"{rec.get_num_samples()} samples")
        rng = np.random.default_rng(seed)
        S = sorted(rng.choice(N_UNITS, size=5, replace=False).tolist())
        for scenario in args.scenarios:
            S_used = [] if scenario == "control" else S
            all_out.append(run_scenario(seed, scenario, S_used, rec, sort, log))
            with open(RESULTS / "results.json", "w") as f:
                json.dump(all_out, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    log("all done")


if __name__ == "__main__":
    main()

"""Measure UnitMatch recall for units that fire in only part of a session.

UnitMatch compares two cross-validation templates per unit. How the spyglass
bundle builds those two halves decides whether a unit that drifts out of (or
into) the recording partway through a session can be matched at all: when the
halves are the first and second half of the RECORDING, a unit with no spikes in
one half gets an all-zero template for that half and never matches. This
script measures that on synthetic ground truth, under two bundle
constructions:

``time_half``
    The recording-half construction, kept here as a local copy
    (:func:`extract_time_half_bundle`) so the comparison survives changes to
    the production bundle builder.
``per_unit``
    Whatever the production
    :func:`spyglass.spikesorting.v2._unitmatch_backend.extract_unitmatch_bundle`
    currently builds, called with the arguments the matcher table uses.

Data: one 120 s, 30 kHz, 16-channel ground-truth recording with 20 units per
seed (SpikeInterface ``generate_ground_truth_recording``), cut into two 60 s
sessions A and B with ``frame_slice`` so every unit's true partner is the same
unit id in the other session. Five units per seed (the drift-out set S) are
modified by scenario:

``control``
    No modification (S is empty for scoring).
``driftout_A``
    S units keep only their spikes in the first half of session A.
``driftout_AB``
    As ``driftout_A``, and S units also keep only their spikes in the second
    half of session B.

Every pair goes through :meth:`UnitMatchBackend.match` (a pair passes when both
directed probabilities exceed 0.5). Scored per run: drift-out true-pair recall,
healthy (non-S) true-pair recall, healthy false positives (non-S x non-S,
different ids), S x S false positives and UnitMatch's fitted match prior. Each
drift-out run is also compared with the same condition's ``control`` run on
the same seed, restricted to that seed's non-S units and pairs. Acceptance
gates (:func:`evaluate_gates`) are evaluated for ``per_unit``. When both
conditions run, the bundles are compared file by file (:func:`compare_bundles`).

The dataset, scoring and gate functions are importable; UnitMatchPy is
imported only when a bundle is built or matched.

Usage (from the repo root, in the spikesorting-v2 environment with the
matching extra installed)::

    python tests/spikesorting/v2/scripts/unitmatch_half_split_experiment.py \\
        [--first-seed 10] [--seeds 10] \\
        [--scenarios control driftout_A driftout_AB] \\
        [--conditions time_half per_unit] [--out-dir DIR]

``--seeds N`` runs ``N`` seeds starting at ``--first-seed`` (default 10, so
the default run is seeds 10..19, the same seeds
``test_driftout_units_recovered_pooled`` checks the gates on). Per-run JSON,
``results.json`` and ``summary.md`` are written to ``--out-dir`` (default: a
new temporary directory) and the summary is printed.
"""

from __future__ import annotations

import argparse
import contextlib
import fractions
import io
import json
import tempfile
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

FS = 30_000.0
DURATION_S = 120.0
N_CHANNELS = 16
N_UNITS = 20
N_DRIFT_OUT = 5
MS_BEFORE = MS_AFTER = 1.5
MAX_SPIKES_PER_UNIT = 100
BUNDLE_SEED = 0
MATCH_THRESHOLD = 0.5
JOB_KWARGS = {"n_jobs": 1, "progress_bar": False}

SCENARIOS = ("control", "driftout_A", "driftout_AB")
DRIFT_OUT_SCENARIOS = ("driftout_A", "driftout_AB")
CONDITIONS = ("time_half", "per_unit")
GATED_CONDITION = "per_unit"
BASELINE_CONDITION = "time_half"
SESSIONS = ("A", "B")

MIN_DRIFT_OUT_RECALL = 0.80
MAX_SXS_FALSE_PAIR_RATE = (
    0.015  # diagnostic only, not gated (see evaluate_gates)
)
MAX_HEALTHY_RECALL_DROP = 0.04  # count-based; diagnostic only, not gated
MAX_EXCESS_HEALTHY_RECALL_DROP = 0.04  # count-based; diagnostic only, not gated
MAX_HEALTHY_PROB_DROP = 0.04
MAX_EXCESS_HEALTHY_PROB_DROP = 0.04
MAX_HEALTHY_FALSE_PAIR_RATE_INCREASE = 0.005

#: Default seeds: ``DEFAULT_FIRST_SEED .. DEFAULT_FIRST_SEED +
#: DEFAULT_SEEDS - 1`` (the CLI's and ``test_driftout_units_recovered_pooled``'s
#: shared default).
DEFAULT_FIRST_SEED = 10
DEFAULT_SEEDS = 10


# ----------------------------------------------------------------------------
# Synthetic data
# ----------------------------------------------------------------------------
def make_dataset(seed: int):
    """Generate one ground-truth recording + sorting with integer unit ids.

    Parameters
    ----------
    seed : int
        SpikeInterface generator seed.

    Returns
    -------
    recording : spikeinterface BaseRecording
        ``DURATION_S`` s at ``FS`` Hz, ``N_CHANNELS`` channels.
    sorting : spikeinterface BaseSorting
        ``N_UNITS`` units renamed to ids ``0 .. N_UNITS-1``.
    """
    import spikeinterface as si

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
        generate_sorting_kwargs={
            "firing_rates": 10.0,
            "refractory_period_ms": 4.0,
        },
        noise_kwargs={"noise_levels": 5.0, "strategy": "on_the_fly"},
        seed=seed,
    )
    # SI generates string unit ids; the bundle names files with
    # ``np.asarray(unit_ids, dtype=int)``, so use int ids throughout.
    sort = sort.rename_units(np.arange(len(sort.get_unit_ids()), dtype=int))
    return rec, sort


def choose_drift_out_units(seed: int) -> list[int]:
    """Return the sorted drift-out unit ids S for ``seed``.

    >>> choose_drift_out_units(0) == choose_drift_out_units(0)
    True
    >>> len(choose_drift_out_units(3))
    5
    """
    rng = np.random.default_rng(seed)
    return sorted(
        int(u) for u in rng.choice(N_UNITS, size=N_DRIFT_OUT, replace=False)
    )


def split_sessions(recording, sorting):
    """Cut the recording + sorting into two back-to-back sessions A and B."""
    half = int(DURATION_S * FS) // 2
    n = recording.get_num_samples()
    session_a = (recording.frame_slice(0, half), sorting.frame_slice(0, half))
    session_b = (recording.frame_slice(half, n), sorting.frame_slice(half, n))
    return session_a, session_b


def drop_spikes(sorting, unit_subset, keep):
    """Keep only spikes with ``keep(frames)`` True for units in ``unit_subset``.

    Other units are untouched and the unit order is kept.
    """
    import spikeinterface as si

    trains = {}
    for uid in sorting.get_unit_ids():
        st = sorting.get_unit_spike_train(uid)
        if uid in unit_subset:
            st = st[keep(st)]
        trains[uid] = st.astype(np.int64)
    return si.NumpySorting.from_unit_dict(trains, sorting.sampling_frequency)


def make_scenario_sessions(recording, sorting, scenario, drift_out_units):
    """Return ``{"A": (rec, sort), "B": (rec, sort)}`` for ``scenario``.

    ``driftout_A`` keeps only first-half spikes of session A for the drift-out
    units; ``driftout_AB`` additionally keeps only second-half spikes of
    session B for them. ``control`` leaves both sessions unmodified.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}")
    (rec_a, sort_a), (rec_b, sort_b) = split_sessions(recording, sorting)
    half_frames = rec_a.get_num_samples() // 2
    subset = set(drift_out_units)
    if scenario in DRIFT_OUT_SCENARIOS:
        sort_a = drop_spikes(sort_a, subset, lambda st: st < half_frames)
    if scenario == "driftout_AB":
        sort_b = drop_spikes(sort_b, subset, lambda st: st >= half_frames)
    return {"A": (rec_a, sort_a), "B": (rec_b, sort_b)}


# ----------------------------------------------------------------------------
# Bundle constructions
# ----------------------------------------------------------------------------
def extract_time_half_bundle(
    session_dir,
    recording,
    sorting,
    *,
    ms_before: float = 1.5,
    ms_after: float = 1.5,
    max_spikes_per_unit: int = 100,
    seed: int = 0,
    job_kwargs: dict | None = None,
) -> None:
    """Write a UnitMatch bundle whose halves are the recording's two halves.

    A local copy of the recording-half bundle construction: build a dense
    analyzer on the first and on the second half of the recording, compute
    templates on each, and stack them as the two cross-validation halves. A
    unit with no spikes in one recording half gets an all-zero template for
    that half. Writes ``RawWaveforms/Unit{id}_RawSpikes.npy`` (shape
    ``(spike_width, n_channels, 2)``), ``channel_positions.npy`` and
    ``cluster_group.tsv`` (every unit ``good``).
    """
    from spyglass.spikesorting.v2._params.matcher import UnitMatchParamsSchema
    from spyglass.spikesorting.v2._unitmatch_backend import _require_unitmatch

    validated = UnitMatchParamsSchema(
        ms_before=ms_before,
        ms_after=ms_after,
        max_spikes_per_unit=max_spikes_per_unit,
        seed=seed,
    )
    ms_before = validated.ms_before
    ms_after = validated.ms_after
    max_spikes_per_unit = validated.max_spikes_per_unit
    seed = validated.seed

    import spikeinterface as si

    um = _require_unitmatch()
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    random_seed = seed
    compute_job_kwargs = dict(job_kwargs or {})
    compute_job_kwargs.pop("random_seed", None)

    probe = recording.get_probe()
    if probe.ndim == 3:
        recording = recording.set_probe(probe.to_2d())
    channel_positions = recording.get_channel_locations()
    n_channels = recording.get_num_channels()
    if channel_positions.shape != (n_channels, 2):
        raise ValueError(
            "extract_time_half_bundle: channel_positions have shape "
            f"{channel_positions.shape}, expected (n_channels, 2) = "
            f"({n_channels}, 2)."
        )

    half = recording.get_num_samples() // 2
    t_halves = []
    for a0, a1 in [(0, half), (half, recording.get_num_samples())]:
        analyzer = si.create_sorting_analyzer(
            sorting.frame_slice(a0, a1),
            recording.frame_slice(a0, a1),
            sparse=False,
        )
        analyzer.compute(
            "random_spikes",
            method="uniform",
            max_spikes_per_unit=max_spikes_per_unit,
            seed=random_seed,
        )
        analyzer.compute(
            "waveforms",
            ms_before=ms_before,
            ms_after=ms_after,
            **compute_job_kwargs,
        )
        analyzer.compute("templates", **compute_job_kwargs)
        t_halves.append(analyzer.get_extension("templates").get_data())

    avg_waves = np.stack(t_halves, axis=-1)  # (n_units, spike_width, n_chan, 2)
    unit_ids = np.asarray(sorting.get_unit_ids(), dtype=int)
    np.save(session_dir / "channel_positions.npy", channel_positions)
    um.extract_raw_data.save_avg_waveforms(
        avg_waves,
        str(session_dir),
        unit_ids,
        unit_ids,
        extract_good_units_only=False,
    )
    rows = [np.array(("cluster_id", "group"))] + [
        np.array((str(i), "good")) for i in unit_ids
    ]
    np.savetxt(
        session_dir / "cluster_group.tsv",
        np.vstack(rows),
        fmt=["%s", "%s"],
        delimiter="\t",
    )


def extract_per_unit_bundle(session_dir, recording, sorting) -> list[int]:
    """Build the bundle with production ``extract_unitmatch_bundle``.

    Returns
    -------
    list of int
        Unit ids the production builder reports as excluded from the bundle.
    """
    from spyglass.spikesorting.v2._unitmatch_backend import (
        extract_unitmatch_bundle,
    )

    excluded = extract_unitmatch_bundle(
        session_dir,
        recording,
        sorting,
        ms_before=MS_BEFORE,
        ms_after=MS_AFTER,
        max_spikes_per_unit=MAX_SPIKES_PER_UNIT,
        seed=BUNDLE_SEED,
        job_kwargs=JOB_KWARGS,
    )
    return sorted(int(u) for u in excluded)


def build_bundle(condition, session_dir, recording, sorting) -> list[int]:
    """Build one session bundle under ``condition``; return excluded unit ids."""
    if condition == "time_half":
        extract_time_half_bundle(
            session_dir,
            recording,
            sorting,
            ms_before=MS_BEFORE,
            ms_after=MS_AFTER,
            max_spikes_per_unit=MAX_SPIKES_PER_UNIT,
            seed=BUNDLE_SEED,
            job_kwargs=JOB_KWARGS,
        )
        return []
    if condition == "per_unit":
        return extract_per_unit_bundle(session_dir, recording, sorting)
    raise ValueError(f"unknown condition {condition!r}")


def all_zero_halves(session_dir) -> dict[str, list[int]]:
    """Unit ids whose saved template half 0 / half 1 is exactly all zeros."""
    out = {"half0": [], "half1": []}
    for path in sorted(Path(session_dir, "RawWaveforms").glob("Unit*.npy")):
        unit_id = int(path.name.removeprefix("Unit").split("_")[0])
        wave = np.load(path)
        for k in (0, 1):
            if np.all(wave[..., k] == 0):
                out[f"half{k}"].append(unit_id)
    return {key: sorted(ids) for key, ids in out.items()}


# ----------------------------------------------------------------------------
# Matching
# ----------------------------------------------------------------------------
def _session_index_from_switch(session_switch, n) -> np.ndarray:
    """Stacked-index -> session index (0 = A, 1 = B for a two-session match).

    Exactly the construction ``UnitMatchBackend._pairs_from_matrix`` uses
    (``spyglass/spikesorting/v2/_unitmatch_backend.py:568-570``) to turn
    ``session_switch`` (the stacked boundary between sessions) into a
    per-row session index.
    """
    boundaries = np.asarray(session_switch).ravel()
    return np.searchsorted(boundaries, np.arange(n), side="right") - 1


def true_pair_directed_probs(
    prob_matrix, session_switch, original_ids, unit_id
):
    """The two directed probabilities of one unit's same-id cross-session pair.

    ``prob_matrix`` is UnitMatch's full ``(n, n)`` naive-Bayes probability
    array, built the same way ``UnitMatchBackend.match`` builds it
    (``_unitmatch_backend.py:522-524``: ``probability[:,
    1].reshape(n_units, n_units)``). ``original_ids`` stacks each session's
    kept unit ids in the same order UnitMatch loaded them
    (``_unitmatch_backend.py:501-502``: ``clus_info["original_ids"] =
    np.concatenate(good_units)``); ``session_switch`` marks the stacked
    boundary between sessions, converted to a per-row session index the same
    way ``_pairs_from_matrix`` does it (``_unitmatch_backend.py:568-570``).
    Session 0 is session A, session 1 is session B (the order
    ``UnitMatchBackend.match`` receives ``session_inputs``). Looking a unit up
    by a boolean mask on its id (not by position) is what makes this correct
    for sparse or reordered ids: a unit excluded from one session, or ranked
    differently among kept units between sessions, still resolves to the
    right stacked index.

    Returns
    -------
    (float, float) or None
        ``(p(A_u -> B_u), p(B_u -> A_u))``, i.e. ``(prob_matrix[i, j],
        prob_matrix[j, i])`` for the stacked indices ``i``/``j`` of
        ``unit_id`` in session 0 / session 1 -- the same two directed values
        ``UnitMatchBackend._pairs_from_matrix`` requires both above threshold
        to emit a pair. ``None`` if ``unit_id`` is missing from either
        session (excluded from that session's bundle).

    Examples
    --------
    >>> pm = np.array([[0.0, 0.1, 0.9], [0.2, 0.0, 0.3], [0.8, 0.4, 0.0]])
    >>> true_pair_directed_probs(pm, [0, 2, 3], [5, 9, 5], 5)
    (0.9, 0.8)
    >>> true_pair_directed_probs(pm, [0, 2, 3], [5, 9, 5], 9) is None
    True
    """
    # ``original_ids`` may be a column vector (UnitMatchPy's per-session
    # ``good_units`` entries are ``(n_i, 1)``, so ``np.concatenate`` stacks
    # them into ``(n, 1)``, not ``(n,)``); flatten before comparing against
    # the 1D ``session_ids`` so the two boolean masks combine element-wise
    # instead of broadcasting into an (n, n) array.
    original_ids = np.asarray(original_ids).reshape(-1)
    n = original_ids.shape[0]
    session_ids = _session_index_from_switch(session_switch, n)
    idx_a = np.flatnonzero((session_ids == 0) & (original_ids == unit_id))
    idx_b = np.flatnonzero((session_ids == 1) & (original_ids == unit_id))
    if idx_a.size == 0 or idx_b.size == 0:
        return None
    i, j = int(idx_a[0]), int(idx_b[0])
    return float(prob_matrix[i, j]), float(prob_matrix[j, i])


def match_sessions(
    session_dirs, unit_ids
) -> tuple[list[list], dict | None, dict[int, list[float]]]:
    """Run ``UnitMatchBackend.match`` on two session bundles (A first).

    Also records the fitted prior UnitMatch passes to its naive-Bayes step:
    the backend builds ``priors = (1 - p, p)`` with
    ``p = n_expected_matches / n_units ** 2``, where ``n_expected_matches`` is
    fitted by ``overlord.extract_metric_scores``. The recorder wraps
    ``bayes_functions.apply_naive_bayes`` for the duration of the call, and
    from the SAME wrapped call also captures the full probability matrix
    (reshaped exactly as ``UnitMatchBackend.match`` does it) so every
    requested unit's true cross-session pair directed probabilities
    (:func:`true_pair_directed_probs`) can be recovered without changing
    production code. That lookup also needs the stacked-index ->
    (session, unit id) mapping, which is not returned by ``match()``; a
    second wrap of ``UnitMatchPy.utils.load_good_waveforms`` (the call inside
    ``UnitMatchBackend.match`` that produces ``session_switch`` and
    ``good_units``) records those. Both wraps only observe values the real
    call already computes; neither changes what UnitMatch does.

    Returns
    -------
    pairs : list of [int, int, float]
        Passing pairs as ``[unit_id_in_A, unit_id_in_B, mean_probability]``.
    fitted : dict or None
        ``match_class_prior`` (the match-class prior, ``priors[1]``),
        ``n_expected_matches`` and ``n_units``; ``None`` if UnitMatch never
        reached its naive-Bayes step.
    true_pair_probs : dict[int, list[float]]
        ``{unit_id: [p(A_u -> B_u), p(B_u -> A_u)]}`` for every ``unit_id``
        in ``unit_ids`` present in both sessions' loaded bundle. Empty if
        UnitMatch never reached its naive-Bayes step.
    """
    from spyglass.spikesorting.v2._unitmatch_backend import (
        UnitMatchBackend,
        _require_unitmatch,
    )
    from spyglass.spikesorting.v2.matcher_protocol import SessionMatcherInput

    inputs = [
        SessionMatcherInput(
            curation_key={"sorting_id": label, "curation_id": 0},
            waveform_dir=Path(d),
            channel_positions_path=Path(d) / "channel_positions.npy",
        )
        for label, d in zip(SESSIONS, session_dirs)
    ]
    um = _require_unitmatch()
    original_bayes = um.bayes_functions.apply_naive_bayes
    original_load = um.utils.load_good_waveforms
    fitted = {}
    captured = {}

    def recording_load_good_waveforms(*args, **kwargs):
        result = original_load(*args, **kwargs)
        _, _, session_switch, _, good_units, _ = result
        captured["session_switch"] = session_switch
        captured["original_ids"] = np.concatenate(good_units)
        return result

    def recording_naive_bayes(
        parameter_kernels, priors, predictors, param, cond
    ):
        fitted["match_class_prior"] = float(priors[1])
        fitted["n_expected_matches"] = int(param["n_expected_matches"])
        fitted["n_units"] = int(param["n_units"])
        probability = original_bayes(
            parameter_kernels, priors, predictors, param, cond
        )
        captured["prob_matrix"] = probability[:, 1].reshape(
            param["n_units"], param["n_units"]
        )
        return probability

    um.utils.load_good_waveforms = recording_load_good_waveforms
    um.bayes_functions.apply_naive_bayes = recording_naive_bayes
    try:
        match_pairs = UnitMatchBackend().match(
            inputs, {"match_threshold": MATCH_THRESHOLD}
        )
    finally:
        um.utils.load_good_waveforms = original_load
        um.bayes_functions.apply_naive_bayes = original_bayes
    pairs = []
    for p in match_pairs:
        if (p.session_a_sorting_id, p.session_b_sorting_id) != SESSIONS:
            raise RuntimeError(f"unexpected pair orientation: {p}")
        pairs.append([p.unit_a_id, p.unit_b_id, p.match_probability])

    true_pair_probs: dict[int, list[float]] = {}
    if "prob_matrix" in captured and "session_switch" in captured:
        for uid in unit_ids:
            probs = true_pair_directed_probs(
                captured["prob_matrix"],
                captured["session_switch"],
                captured["original_ids"],
                uid,
            )
            if probs is not None:
                true_pair_probs[int(uid)] = list(probs)
    return pairs, (fitted or None), true_pair_probs


def assert_true_pair_probs_consistent(
    pairs, true_pair_probs, match_threshold, *, seed, scenario, condition
) -> None:
    """Self-check: captured probabilities must agree with the emitted pairs.

    UnitMatch emits a true (same-id) cross-session pair ``(u, u)`` in
    ``pairs`` iff BOTH its directed probabilities clear ``match_threshold``
    (``UnitMatchBackend._pairs_from_matrix``), i.e. iff
    ``min(p_ab, p_ba) > match_threshold``. Comparing the two independently
    derived sets -- one from the captured probability matrix, one from
    UnitMatch's own emitted pairs -- catches a capture bug (wrong indices, a
    stale wrap, ...) at the point of use rather than trusting the capture
    silently. Raises ``RuntimeError`` naming the seed/scenario/condition and
    the exact symmetric difference on any mismatch.
    """
    passed_true_pairs = {int(a) for a, b, _ in pairs if int(a) == int(b)}
    probability_true_pairs = {
        u
        for u, (p_ab, p_ba) in true_pair_probs.items()
        if min(p_ab, p_ba) > match_threshold
    }
    if passed_true_pairs != probability_true_pairs:
        only_by_probability = sorted(probability_true_pairs - passed_true_pairs)
        only_by_pairs = sorted(passed_true_pairs - probability_true_pairs)
        raise RuntimeError(
            "assert_true_pair_probs_consistent: captured true-pair "
            "probabilities disagree with UnitMatchBackend.match's emitted "
            f"pairs for seed={seed} scenario={scenario!r} "
            f"condition={condition!r} (match_threshold={match_threshold}): "
            f"units passing by captured probability but not emitted "
            f"{only_by_probability}; units emitted but not passing by "
            f"captured probability {only_by_pairs}."
        )


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------
def score_pairs(passing_pairs, unit_ids, drift_out_units) -> dict[str, list]:
    """Count passing true and false cross-session pairs by unit group.

    Parameters
    ----------
    passing_pairs : iterable of (int, int) or (int, int, float)
        Passing ``(unit_in_A, unit_in_B[, probability])`` pairs.
    unit_ids : sequence of int
        Every planted unit id (the denominators come from ground truth, so a
        unit missing from a bundle counts as a miss).
    drift_out_units : sequence of int
        The drift-out set S; every other unit is healthy.

    Returns
    -------
    dict
        ``[n_pass, n]`` for ``drift_out_true`` (i, i) with i in S,
        ``healthy_true`` (i, i) with i not in S, ``healthy_false`` (i, j),
        i != j both not in S, ``drift_out_false`` (i, j), i != j both in S,
        and ``mixed_false`` (i, j), i != j with exactly one of i, j in S
        (both orders) -- a false cross-session pair between a drift-out unit
        and a healthy unit. ``mixed_false`` is printed as a diagnostic only;
        no gate is defined on it.

    Examples
    --------
    >>> score_pairs([(0, 0, 0.9), (1, 1, 0.8), (2, 3, 0.7)], range(4), [0, 1])
    {'drift_out_true': [2, 2], 'healthy_true': [0, 2], 'healthy_false': [1, 2], 'drift_out_false': [0, 2], 'mixed_false': [0, 8]}
    """
    passed = {(int(p[0]), int(p[1])) for p in passing_pairs}
    units = [int(u) for u in unit_ids]
    in_s = set(int(u) for u in drift_out_units)
    counts = {
        "drift_out_true": [0, 0],
        "healthy_true": [0, 0],
        "healthy_false": [0, 0],
        "drift_out_false": [0, 0],
        "mixed_false": [0, 0],
    }
    for i in units:
        for j in units:
            if i == j:
                key = "drift_out_true" if i in in_s else "healthy_true"
            elif i in in_s and j in in_s:
                key = "drift_out_false"
            elif i not in in_s and j not in in_s:
                key = "healthy_false"
            else:
                key = "mixed_false"
            counts[key][1] += 1
            counts[key][0] += int((i, j) in passed)
    return counts


def run_one(seed, scenario, condition, sessions, drift_out_units, out_dir):
    """Build both bundles under ``condition``, match, score; return a record.

    ``drift_out_units`` is the seed's S (scored as empty for ``control``).
    Bundles are written to ``out_dir/seed{seed}/{scenario}/{condition}/{A,B}``.
    """
    run_dir = Path(out_dir) / f"seed{seed}" / scenario / condition
    scored_s = [] if scenario == "control" else list(drift_out_units)
    unit_ids = [int(u) for u in sessions["A"][1].get_unit_ids()]
    dirs, excluded, zero = [], {}, {}
    t0 = time.perf_counter()
    # UnitMatchPy prints progress on every save and match; keep it out of the
    # summary.
    with contextlib.redirect_stdout(io.StringIO()):
        for label in SESSIONS:
            rec, sort = sessions[label]
            d = run_dir / label
            excluded[label] = build_bundle(condition, d, rec, sort)
            zero[label] = all_zero_halves(d)
            dirs.append(d)
        t1 = time.perf_counter()
        pairs, fitted, true_pair_probs = match_sessions(dirs, unit_ids)
    assert_true_pair_probs_consistent(
        pairs,
        true_pair_probs,
        MATCH_THRESHOLD,
        seed=seed,
        scenario=scenario,
        condition=condition,
    )
    t2 = time.perf_counter()
    record = {
        "seed": seed,
        "scenario": scenario,
        "condition": condition,
        "unit_ids": unit_ids,
        "drift_out_units": scored_s,
        "seed_drift_out_units": list(drift_out_units),
        "spike_counts": {
            label: {
                str(u): int(sessions[label][1].get_unit_spike_train(u).size)
                for u in unit_ids
            }
            for label in SESSIONS
        },
        "excluded_unit_ids": excluded,
        "all_zero_halves": zero,
        "passing_pairs": pairs,
        "fitted": fitted,
        "true_pair_probs": {str(u): p for u, p in true_pair_probs.items()},
        "counts": score_pairs(pairs, unit_ids, scored_s),
        "timings_s": {"bundles": t1 - t0, "match": t2 - t1},
        "bundle_dirs": {lab: str(d) for lab, d in zip(SESSIONS, dirs)},
    }
    with open(run_dir / "run.json", "w") as f:
        json.dump(record, f, indent=1)
    return record


def _index(records) -> dict[tuple, dict]:
    return {(r["seed"], r["scenario"], r["condition"]): r for r in records}


def paired_counts(record, control_record) -> dict[str, list]:
    """Healthy counts of a drift-out run and its control on the same units.

    Both runs are scored with the drift-out run's S, so the control's healthy
    true pairs, healthy false pairs and mixed (S x non-S) false pairs cover
    exactly the same non-S units and non-S x non-S / S x non-S pairs as the
    drift-out run.
    """
    s = record["seed_drift_out_units"]
    ctrl = score_pairs(
        control_record["passing_pairs"], control_record["unit_ids"], s
    )
    scen = score_pairs(record["passing_pairs"], record["unit_ids"], s)
    return {
        "healthy_true_control": ctrl["healthy_true"],
        "healthy_true_scenario": scen["healthy_true"],
        "mixed_false_control": ctrl["mixed_false"],
        "mixed_false_scenario": scen["mixed_false"],
        "healthy_false_control": ctrl["healthy_false"],
        "healthy_false_scenario": scen["healthy_false"],
    }


def _sum(pairs) -> list[int]:
    pairs = list(pairs)
    return [sum(p[0] for p in pairs), sum(p[1] for p in pairs)]


def _rate_exact(count) -> fractions.Fraction | None:
    """Exact match rate ``count[0] / count[1]``; ``None`` if the pool is empty."""
    return fractions.Fraction(count[0], count[1]) if count[1] else None


def _rate(count) -> float:
    exact = _rate_exact(count)
    return float(exact) if exact is not None else float("nan")


def _exact_diff(
    a: fractions.Fraction | None, b: fractions.Fraction | None
) -> fractions.Fraction | None:
    """Exact ``a - b``; ``None`` if either operand is unavailable."""
    return None if a is None or b is None else a - b


def pooled_counts(records, scenario, condition) -> dict | None:
    """Sum each count over the seeds of one scenario x condition."""
    runs = [
        r
        for r in records
        if r["scenario"] == scenario and r["condition"] == condition
    ]
    if not runs:
        return None
    keys = runs[0]["counts"].keys()
    out = {k: _sum(r["counts"][k] for r in runs) for k in keys}
    priors = [r["fitted"]["match_class_prior"] for r in runs if r["fitted"]]
    out["mean_match_class_prior"] = (
        float(np.mean(priors)) if priors else float("nan")
    )
    out["n_seeds"] = len(runs)
    return out


def pooled_paired_counts(records, scenario, condition) -> dict | None:
    """Sum :func:`paired_counts` over the seeds that have a control run.

    Returns ``None`` if any seed of ``scenario`` x ``condition`` lacks the same
    condition's control run (the paired comparison is then not evaluable).
    """
    index = _index(records)
    runs = [
        r
        for r in records
        if r["scenario"] == scenario and r["condition"] == condition
    ]
    if not runs:
        return None
    per_seed = []
    for r in runs:
        ctrl = index.get((r["seed"], "control", condition))
        if ctrl is None:
            return None
        per_seed.append(paired_counts(r, ctrl))
    out = {k: _sum(p[k] for p in per_seed) for k in per_seed[0]}
    out["healthy_recall_drop"] = _rate(out["healthy_true_control"]) - _rate(
        out["healthy_true_scenario"]
    )
    out["healthy_fp_increase"] = _rate(out["healthy_false_scenario"]) - _rate(
        out["healthy_false_control"]
    )
    out["mixed_fp_increase"] = _rate(out["mixed_false_scenario"]) - _rate(
        out["mixed_false_control"]
    )
    out["seeds"] = sorted(r["seed"] for r in runs)
    return out


def paired_true_pair_probs(
    record, control_record
) -> tuple[list[tuple[int, float, float]], int]:
    """Non-S ``(unit, q_control, q_scenario)`` triples paired against control.

    ``q_u = min(p(A_u -> B_u), p(B_u -> A_u))`` -- the same min
    ``UnitMatchBackend._pairs_from_matrix`` uses to decide whether to emit a
    pair.

    Returns
    -------
    pairs : list of (int, float, float)
        One triple per ``record``'s non-S unit that has a recorded true-pair
        probability in BOTH ``record`` and ``control_record`` (present in
        both sessions' bundle for both runs).
    n_unpaired : int
        The number of ``record``'s non-S units dropped because a probability
        is missing on either side (excluded from a bundle in that run, or
        the whole run's probability capture never ran) -- symmetric: a unit
        missing from the control side or from the scenario side is dropped
        and counted the same way.
    """
    s = set(int(u) for u in record["seed_drift_out_units"])
    non_s = [u for u in record["unit_ids"] if u not in s]
    ctrl_probs = control_record.get("true_pair_probs", {})
    scen_probs = record.get("true_pair_probs", {})
    pairs, n_unpaired = [], 0
    for u in non_s:
        key = str(u)
        if key not in ctrl_probs or key not in scen_probs:
            n_unpaired += 1
            continue
        pairs.append((u, min(ctrl_probs[key]), min(scen_probs[key])))
    return pairs, n_unpaired


def _capture_missing(record) -> bool:
    """A run had non-S units to capture but ``true_pair_probs`` is empty.

    A proxy for "the probability-matrix capture never ran for this run" --
    either ``UnitMatchBackend.match`` returned early (fewer than two
    sessions, or zero good units) or the capture wraps in
    :func:`match_sessions` never observed a naive-Bayes call. Every non-S
    unit of a run like this is already counted in
    :func:`paired_true_pair_probs`'s ``n_unpaired``; this flags WHY, so a
    wholesale capture failure is not silently indistinguishable from
    ordinary per-unit bundle exclusions (a unit with too few sampled spikes).
    """
    return bool(record["unit_ids"]) and not record.get("true_pair_probs")


def pooled_true_pair_prob_drop(records, scenario, condition) -> dict | None:
    """Healthy true-pair probability drop: the two means, the drop, coverage.

    Pools every non-S unit paired (present in both the scenario run and that
    seed's control run) over every seed of ``scenario`` x ``condition`` that
    has a control run -- ONE mean over all pooled units, not a mean of
    per-seed means. Returns ``None`` if there is no run of ``scenario`` x
    ``condition``, any such seed lacks a control run, or no unit is ever
    paired.
    """
    index = _index(records)
    runs = [
        r
        for r in records
        if r["scenario"] == scenario and r["condition"] == condition
    ]
    if not runs:
        return None
    q_control, q_scenario, seeds = [], [], []
    n_unpaired = 0
    n_capture_missing_runs = 0
    for r in runs:
        ctrl = index.get((r["seed"], "control", condition))
        if ctrl is None:
            return None
        if _capture_missing(r):
            n_capture_missing_runs += 1
        if _capture_missing(ctrl):
            n_capture_missing_runs += 1
        pairs, unpaired = paired_true_pair_probs(r, ctrl)
        n_unpaired += unpaired
        for _, c, s in pairs:
            q_control.append(c)
            q_scenario.append(s)
        seeds.append(r["seed"])
    if not q_control:
        return None
    mean_control = float(np.mean(q_control))
    mean_scenario = float(np.mean(q_scenario))
    return {
        "mean_control": mean_control,
        "mean_scenario": mean_scenario,
        "drop": mean_control - mean_scenario,
        "n_paired": len(q_control),
        "n_unpaired": n_unpaired,
        "n_capture_missing_runs": n_capture_missing_runs,
        "seeds": sorted(seeds),
    }


# ----------------------------------------------------------------------------
# Gates
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Gate:
    """One acceptance gate outcome; ``passed`` is ``None`` if not evaluable."""

    name: str
    scenario: str
    value: float
    threshold: float
    comparison: str
    passed: bool | None
    detail: str


def _gate(name, scenario, exact, threshold, comparison, detail) -> Gate:
    """Build one ``Gate``; ``exact`` (a ``Fraction`` or ``None``) decides ``passed``.

    The pass/fail comparison is always done in ``Fraction`` arithmetic against
    ``Fraction(str(threshold))``, so a value that is exactly on the boundary
    (e.g. 7/150 - 1/150 == 1/25 == 0.04) can never flip by float rounding.
    ``value`` on the returned ``Gate`` is ``float(exact)``, kept only for
    display. ``exact is None`` means the gate is not evaluable (empty pool).
    """
    if exact is None:
        return Gate(
            name, scenario, float("nan"), threshold, comparison, None, detail
        )
    exact_threshold = fractions.Fraction(str(threshold))
    ok = (
        exact >= exact_threshold
        if comparison == ">="
        else exact <= exact_threshold
    )
    return Gate(
        name, scenario, float(exact), threshold, comparison, bool(ok), detail
    )


def _gate_float(name, scenario, value, threshold, comparison, detail) -> Gate:
    """Build one ``Gate`` from a plain float ``value`` (mean-probability gates).

    Unlike :func:`_gate`, this compares floats directly rather than exact
    ``Fraction`` counts: the value averages continuous probabilities, so
    there is no integer pool to be exact about. ``value is None`` means the
    gate is not evaluable (no paired unit).
    """
    if value is None:
        return Gate(
            name, scenario, float("nan"), threshold, comparison, None, detail
        )
    ok = value >= threshold if comparison == ">=" else value <= threshold
    return Gate(name, scenario, value, threshold, comparison, bool(ok), detail)


def _as_diagnostic(gate: Gate) -> Gate:
    """Relabel a computed count-based :class:`Gate` as a printed diagnostic.

    Used by the S x S false-pair rate (unstable under UnitMatch's per-run
    calibration), the count-based healthy recall drop and its excess over
    ``time_half`` (sensitive to threshold flips; the template bit-identity and
    mean-probability gates measure the same effect robustly -- see
    :func:`evaluate_gates`) and the S x non-S false-pair rate increase (no
    fixed limit; reported against the healthy false-pair limit for scale).
    Each keeps its computed value/detail for display, but
    :attr:`Gate.passed` is forced to ``None`` -- none of them are part of
    acceptance.
    """
    return Gate(
        f"{gate.name} (diagnostic, not gated)",
        gate.scenario,
        gate.value,
        gate.threshold,
        gate.comparison,
        None,
        gate.detail,
    )


def evaluate_gates(records, condition: str = GATED_CONDITION) -> list[Gate]:
    """Evaluate the acceptance gates for ``condition``, per drift-out scenario.

    Pooled over every seed present.

    - Drift-out recall >= 0.80.
    - Healthy template bit-identity: every non-S unit's saved
      cross-validation-half templates (:func:`non_s_bit_identical_halves`)
      are bit-identical to the same seed's control run, pooled over seeds.
      PASS iff identical == total.
    - Healthy true-pair probability drop: for each non-S unit, ``q_u =
      min(p(A_u -> B_u), p(B_u -> A_u))`` of its true cross-session pair
      (:func:`paired_true_pair_probs`); the drop is the pooled mean ``q_u``
      in the control run minus the pooled mean ``q_u`` in the scenario run,
      over the same paired non-S units, pooled over all seeds
      (:func:`pooled_true_pair_prob_drop`). PASS iff drop <= 0.04.
    - Excess healthy true-pair probability drop vs ``time_half``: the drop
      above for ``condition`` minus the same drop for ``time_half`` on the
      same seeds <= 0.04 (only when ``time_half`` ran on exactly the same
      seeds).
    - Healthy false-pair rate increase (paired against control) <= 0.005.

    The S x S false-pair rate among drift-out units, the count-based
    healthy recall drop (a paired count of healthy true pairs passing
    before/after) and its excess over ``time_half`` are still computed and
    returned, labelled "(diagnostic, not gated)" -- :attr:`Gate.passed` is
    always ``None`` for them; they are printed for context only. The S x S
    rate is a diagnostic because UnitMatch's per-run, data-driven match
    threshold is unstable when a session has few units: a near-tie in the
    threshold search can flip on a single redrawn template and admit a burst
    of false pairs, independent of how the cross-validation halves are
    constructed.

    "S x non-S false-pair rate increase" is also computed and returned as a
    diagnostic, never gated: the healthy false-pair rate increase only
    counts non-S x non-S pairs, so it is blind to a false pair between a
    drift-out unit and a healthy one. This diagnostic covers that gap the
    same way -- paired against the same seed's control, over both (S, non-S)
    orders.

    The drift-out recall, S x S rate, both count-based recall drops, the
    healthy false-pair rate increase and the S x non-S diagnostic are
    evaluated exactly from the underlying integer counts with
    :mod:`fractions`; only the printed/stored ``value`` is a float. The two
    true-pair probability drops average continuous probabilities, so they
    are plain float comparisons.
    """
    gates = []
    bit_identical = non_s_bit_identical_halves(records, condition)
    for scenario in DRIFT_OUT_SCENARIOS:
        pooled = pooled_counts(records, scenario, condition)
        if pooled is None:
            continue
        s_true = pooled["drift_out_true"]
        s_false = pooled["drift_out_false"]
        gates.append(
            _gate(
                "drift-out recall",
                scenario,
                _rate_exact(s_true),
                MIN_DRIFT_OUT_RECALL,
                ">=",
                f"{s_true[0]}/{s_true[1]}",
            )
        )
        gates.append(
            _as_diagnostic(
                _gate(
                    "S x S false-pair rate",
                    scenario,
                    _rate_exact(s_false),
                    MAX_SXS_FALSE_PAIR_RATE,
                    "<=",
                    f"{s_false[0]}/{s_false[1]}",
                )
            )
        )

        bi_counts = bit_identical.get(scenario)
        gates.append(
            _gate(
                "healthy template bit-identity",
                scenario,
                _rate_exact(bi_counts) if bi_counts else None,
                1.0,
                ">=",
                (
                    f"{bi_counts[0]}/{bi_counts[1]}"
                    if bi_counts
                    else "no comparable seed"
                ),
            )
        )

        prob_pooled = pooled_true_pair_prob_drop(records, scenario, condition)
        if prob_pooled is None:
            prob_value = None
            prob_detail = "no paired non-S unit with a recorded probability"
        else:
            prob_value = prob_pooled["drop"]
            prob_detail = (
                f"control mean {prob_pooled['mean_control']:.4f} -> "
                f"scenario mean {prob_pooled['mean_scenario']:.4f} "
                f"(n_paired={prob_pooled['n_paired']}, "
                f"n_unpaired={prob_pooled['n_unpaired']}, "
                f"n_capture_missing_runs="
                f"{prob_pooled['n_capture_missing_runs']})"
            )
        gates.append(
            _gate_float(
                "healthy true-pair probability drop",
                scenario,
                prob_value,
                MAX_HEALTHY_PROB_DROP,
                "<=",
                prob_detail,
            )
        )

        prob_baseline = (
            pooled_true_pair_prob_drop(records, scenario, BASELINE_CONDITION)
            if condition != BASELINE_CONDITION
            else None
        )
        if prob_pooled is None or prob_baseline is None:
            excess_prob_value = None
            excess_prob_detail = (
                "time_half paired probability run not available"
            )
        elif prob_baseline["seeds"] != prob_pooled["seeds"]:
            excess_prob_value = None
            excess_prob_detail = "time_half ran on different seeds"
        else:
            excess_prob_value = prob_pooled["drop"] - prob_baseline["drop"]
            excess_prob_detail = (
                f"{condition} drop {prob_pooled['drop']:.4f} - time_half "
                f"drop {prob_baseline['drop']:.4f}"
            )
        gates.append(
            _gate_float(
                "excess healthy true-pair probability drop vs time_half",
                scenario,
                excess_prob_value,
                MAX_EXCESS_HEALTHY_PROB_DROP,
                "<=",
                excess_prob_detail,
            )
        )

        paired = pooled_paired_counts(records, scenario, condition)
        if paired is None:
            exact_drop = exact_fp_inc = exact_mixed_fp_inc = None
            drop_detail = fp_detail = mixed_fp_detail = (
                "no control run for some seed"
            )
        else:
            c, s = (
                paired["healthy_true_control"],
                paired["healthy_true_scenario"],
            )
            exact_drop = _exact_diff(_rate_exact(c), _rate_exact(s))
            drop_detail = f"control {c[0]}/{c[1]} -> scenario {s[0]}/{s[1]}"
            c, s = (
                paired["healthy_false_control"],
                paired["healthy_false_scenario"],
            )
            exact_fp_inc = _exact_diff(_rate_exact(s), _rate_exact(c))
            fp_detail = f"control {c[0]}/{c[1]} -> scenario {s[0]}/{s[1]}"
            c, s = (
                paired["mixed_false_control"],
                paired["mixed_false_scenario"],
            )
            exact_mixed_fp_inc = _exact_diff(_rate_exact(s), _rate_exact(c))
            mixed_fp_detail = f"control {c[0]}/{c[1]} -> scenario {s[0]}/{s[1]}"
        gates.append(
            _as_diagnostic(
                _gate(
                    "healthy recall drop",
                    scenario,
                    exact_drop,
                    MAX_HEALTHY_RECALL_DROP,
                    "<=",
                    drop_detail,
                )
            )
        )
        baseline = (
            pooled_paired_counts(records, scenario, BASELINE_CONDITION)
            if condition != BASELINE_CONDITION
            else None
        )
        if paired is None or baseline is None:
            exact_excess = None
            excess_detail = "time_half paired run not available"
        elif baseline["seeds"] != paired["seeds"]:
            exact_excess = None
            excess_detail = "time_half ran on different seeds"
        else:
            c, s = (
                baseline["healthy_true_control"],
                baseline["healthy_true_scenario"],
            )
            exact_baseline_drop = _exact_diff(_rate_exact(c), _rate_exact(s))
            exact_excess = _exact_diff(exact_drop, exact_baseline_drop)
            excess_detail = (
                f"{condition} drop {float(exact_drop):.4f} - time_half drop "
                f"{float(exact_baseline_drop):.4f}"
            )
        gates.append(
            _as_diagnostic(
                _gate(
                    "excess healthy recall drop vs time_half",
                    scenario,
                    exact_excess,
                    MAX_EXCESS_HEALTHY_RECALL_DROP,
                    "<=",
                    excess_detail,
                )
            )
        )
        gates.append(
            _gate(
                "healthy false-pair rate increase",
                scenario,
                exact_fp_inc,
                MAX_HEALTHY_FALSE_PAIR_RATE_INCREASE,
                "<=",
                fp_detail,
            )
        )
        gates.append(
            _as_diagnostic(
                _gate(
                    "S x non-S false-pair rate increase",
                    scenario,
                    exact_mixed_fp_inc,
                    MAX_HEALTHY_FALSE_PAIR_RATE_INCREASE,
                    "<=",
                    mixed_fp_detail,
                )
            )
        )
    return gates


# ----------------------------------------------------------------------------
# Bundle fidelity
# ----------------------------------------------------------------------------
def compare_bundles(dir_1, dir_2) -> dict[str, bool]:
    """Whether two bundle directories hold byte-identical files.

    Returns ``raw_waveforms`` (same ``RawWaveforms/*.npy`` file set, every
    file byte-identical), ``channel_positions`` and ``cluster_group``.
    """
    dir_1, dir_2 = Path(dir_1), Path(dir_2)

    def same(rel) -> bool:
        a, b = dir_1 / rel, dir_2 / rel
        return a.is_file() and b.is_file() and a.read_bytes() == b.read_bytes()

    names_1 = sorted(p.name for p in (dir_1 / "RawWaveforms").glob("*.npy"))
    names_2 = sorted(p.name for p in (dir_2 / "RawWaveforms").glob("*.npy"))
    raw = names_1 == names_2 and all(
        same(Path("RawWaveforms") / n) for n in names_1
    )
    return {
        "raw_waveforms": raw,
        "channel_positions": same("channel_positions.npy"),
        "cluster_group": same("cluster_group.tsv"),
    }


def bundle_fidelity(records) -> list[dict]:
    """Compare the two conditions' bundles for every run that has both."""
    index = _index(records)
    rows = []
    for (seed, scenario, condition), rec in sorted(index.items()):
        if condition != CONDITIONS[0]:
            continue
        other = index.get((seed, scenario, CONDITIONS[1]))
        if other is None:
            continue
        for label in SESSIONS:
            rows.append(
                {
                    "seed": seed,
                    "scenario": scenario,
                    "session": label,
                    **compare_bundles(
                        rec["bundle_dirs"][label], other["bundle_dirs"][label]
                    ),
                }
            )
    return rows


def _unit_waveform_paths(session_dir) -> dict[int, Path]:
    """Map unit id -> its saved ``RawWaveforms/Unit{id}_*.npy`` path."""
    return {
        int(path.name.removeprefix("Unit").split("_")[0]): path
        for path in Path(session_dir, "RawWaveforms").glob("Unit*.npy")
    }


def non_s_bit_identical_halves(records, condition) -> dict[str, list[int]]:
    """Per drift-out scenario, non-S template halves bit-identical to control.

    For every seed with both a ``scenario`` and that seed's ``control`` run of
    ``condition``, compares each non-S unit's two saved raw-waveform halves
    (``RawWaveforms/Unit{id}_*.npy[..., 0]`` and ``[..., 1]``), session by
    session. A unit present in one bundle but excluded from the other (fewer
    than two sampled spikes in one run but not the other) counts as NOT
    identical for both of that session's halves -- it is not skipped: a
    unit's bundle presence itself changing between the scenario and control
    run is exactly the kind of drift-out-induced difference this check (and
    the healthy template bit-identity gate built on it) exists to catch. This
    quantifies how much of the paired healthy comparison (recall drop,
    true-pair probability drop, false-pair rate increase) reflects an
    S-unit effect versus resampling noise from SpikeInterface's shared
    per-analyzer RNG, which can perturb a non-S unit's randomly chosen spike
    subset merely because another unit's available spike count changed.

    Returns
    -------
    dict
        ``scenario -> [n_identical, n_total]`` for each of
        :data:`DRIFT_OUT_SCENARIOS` with at least one comparable seed.
        Records without a ``bundle_dirs`` (e.g. hand-built gate-test records
        with no bundles on disk) are skipped rather than raising.
    """
    index = _index(records)
    out = {}
    for scenario in DRIFT_OUT_SCENARIOS:
        n_identical = n_total = 0
        for r in records:
            if r["scenario"] != scenario or r["condition"] != condition:
                continue
            if "bundle_dirs" not in r:
                continue
            ctrl = index.get((r["seed"], "control", condition))
            if ctrl is None or "bundle_dirs" not in ctrl:
                continue
            in_s = set(r["seed_drift_out_units"])
            non_s = [u for u in r["unit_ids"] if u not in in_s]
            for label in SESSIONS:
                scen_waves = _unit_waveform_paths(r["bundle_dirs"][label])
                ctrl_waves = _unit_waveform_paths(ctrl["bundle_dirs"][label])
                for uid in non_s:
                    if uid not in scen_waves or uid not in ctrl_waves:
                        # Present on one side only: both halves count as NOT
                        # identical (the templates cannot be compared, and a
                        # unit whose bundle presence changed is itself a
                        # difference), rather than being skipped.
                        n_total += 2
                        continue
                    scen_wave = np.load(scen_waves[uid])
                    ctrl_wave = np.load(ctrl_waves[uid])
                    for k in (0, 1):
                        n_total += 1
                        if np.array_equal(scen_wave[..., k], ctrl_wave[..., k]):
                            n_identical += 1
        if n_total:
            out[scenario] = [n_identical, n_total]
    return out


# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
def _frac(count) -> str:
    return f"{count[0]}/{count[1]} ({_rate(count):.3f})"


def _fmt_prior(fitted) -> str:
    return f"{fitted['match_class_prior']:.4f}" if fitted else "n/a"


def format_gate_line(g: Gate) -> str:
    """Render one ``Gate`` as a single printed line.

    A diagnostic (``passed is None``) never prints ``value <= threshold`` --
    that reads as an asserted comparison result, which it is not. It prints
    the value next to the limit it is reported against instead, e.g.
    ``0.0200 (limit 0.015, not gated)``. A gated ``Gate`` keeps the original
    ``value <= threshold`` form, since that comparison was actually made.
    """
    verdict = {True: "PASS", False: "FAIL", None: "N/A"}[g.passed]
    if g.passed is None:
        comparison = f"(limit {g.threshold}, not gated)"
    else:
        comparison = f"{g.comparison} {g.threshold}"
    return (
        f"- {verdict} {g.name} [{g.scenario}]: {g.value:.4f} {comparison} "
        f"({g.detail})"
    )


def format_summary(records, gates, fidelity) -> str:
    """Render pooled, per-seed, paired, gate and fidelity tables as markdown."""
    seeds = sorted({r["seed"] for r in records})
    scenarios = [
        s for s in SCENARIOS if any(r["scenario"] == s for r in records)
    ]
    conditions = [
        c for c in CONDITIONS if any(r["condition"] == c for r in records)
    ]
    lines = [
        "# UnitMatch half-split experiment",
        "",
        f"Seeds: {seeds} (n={len(seeds)}); {N_UNITS} units/session, "
        f"|S|={N_DRIFT_OUT} drift-out units per seed; pass = both directed "
        f"probabilities > {MATCH_THRESHOLD}. `control` scores every unit as "
        "healthy.",
        "",
        "## Pooled over seeds",
        "",
        "| scenario | condition | drift-out recall | healthy recall | "
        "healthy FP | SxS FP | S x non-S FP (diagnostic, not gated) | "
        "mean fitted match-class prior |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for scenario in scenarios:
        for condition in conditions:
            p = pooled_counts(records, scenario, condition)
            if p is None:
                continue
            lines.append(
                f"| {scenario} | {condition} | {_frac(p['drift_out_true'])} | "
                f"{_frac(p['healthy_true'])} | {_frac(p['healthy_false'])} | "
                f"{_frac(p['drift_out_false'])} | {_frac(p['mixed_false'])} | "
                f"{p['mean_match_class_prior']:.4f} |"
            )

    lines += [
        "",
        "## Paired against the same condition's control (same non-S units)",
        "",
        "| scenario | condition | healthy recall control -> scenario | "
        "drop | healthy FP control -> scenario | FP increase | S x non-S FP "
        "control -> scenario (diagnostic, not gated) | FP increase "
        "(diagnostic, not gated) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for scenario in (s for s in DRIFT_OUT_SCENARIOS if s in scenarios):
        for condition in conditions:
            p = pooled_paired_counts(records, scenario, condition)
            if p is None:
                continue
            lines.append(
                f"| {scenario} | {condition} | "
                f"{_frac(p['healthy_true_control'])} -> "
                f"{_frac(p['healthy_true_scenario'])} | "
                f"{p['healthy_recall_drop']:+.4f} | "
                f"{_frac(p['healthy_false_control'])} -> "
                f"{_frac(p['healthy_false_scenario'])} | "
                f"{p['healthy_fp_increase']:+.4f} | "
                f"{_frac(p['mixed_false_control'])} -> "
                f"{_frac(p['mixed_false_scenario'])} | "
                f"{p['mixed_fp_increase']:+.4f} |"
            )

    lines += [
        "",
        "## Healthy true-pair mean probability (paired)",
        "",
        "| scenario | condition | control mean q | scenario mean q | drop | "
        "n paired | n unpaired | n capture-missing runs |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for scenario in (s for s in DRIFT_OUT_SCENARIOS if s in scenarios):
        for condition in conditions:
            p = pooled_true_pair_prob_drop(records, scenario, condition)
            if p is None:
                continue
            lines.append(
                f"| {scenario} | {condition} | {p['mean_control']:.4f} | "
                f"{p['mean_scenario']:.4f} | {p['drop']:+.4f} | "
                f"{p['n_paired']} | {p['n_unpaired']} | "
                f"{p['n_capture_missing_runs']} |"
            )

    index = _index(records)
    lines += [
        "",
        "## Per seed",
        "",
        "| seed | S | scenario | condition | drift-out | healthy | healthy FP "
        "| SxS FP | S x non-S FP (diagnostic, not gated) | fitted prior | "
        "excluded A / B | paired healthy ctrl->scen | paired FP ctrl->scen | "
        "paired S x non-S FP ctrl->scen (diagnostic, not gated) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for seed in seeds:
        for scenario in scenarios:
            for condition in conditions:
                r = index.get((seed, scenario, condition))
                if r is None:
                    continue
                c = r["counts"]
                paired_true = paired_fp = paired_mixed_fp = ""
                ctrl = index.get((seed, "control", condition))
                if scenario != "control" and ctrl is not None:
                    pc = paired_counts(r, ctrl)
                    paired_true = (
                        f"{pc['healthy_true_control'][0]} -> "
                        f"{pc['healthy_true_scenario'][0]} "
                        f"/{pc['healthy_true_control'][1]}"
                    )
                    paired_fp = (
                        f"{pc['healthy_false_control'][0]} -> "
                        f"{pc['healthy_false_scenario'][0]} "
                        f"/{pc['healthy_false_control'][1]}"
                    )
                    paired_mixed_fp = (
                        f"{pc['mixed_false_control'][0]} -> "
                        f"{pc['mixed_false_scenario'][0]} "
                        f"/{pc['mixed_false_control'][1]}"
                    )
                ex = r["excluded_unit_ids"]
                lines.append(
                    f"| {seed} | {r['seed_drift_out_units']} | {scenario} | "
                    f"{condition} | {c['drift_out_true'][0]}/"
                    f"{c['drift_out_true'][1]} | {c['healthy_true'][0]}/"
                    f"{c['healthy_true'][1]} | {c['healthy_false'][0]}/"
                    f"{c['healthy_false'][1]} | {c['drift_out_false'][0]}/"
                    f"{c['drift_out_false'][1]} | {c['mixed_false'][0]}/"
                    f"{c['mixed_false'][1]} | {_fmt_prior(r['fitted'])} | "
                    f"{ex['A']} / {ex['B']} | {paired_true} | {paired_fp} | "
                    f"{paired_mixed_fp} |"
                )

    lines += [
        "",
        f"## Acceptance gates ({GATED_CONDITION}, pooled over seeds)",
        "",
    ]
    if not gates:
        lines.append("Not evaluated (no drift-out runs for this condition).")
    for g in gates:
        lines.append(format_gate_line(g))

    lines += ["", "## Bundle fidelity (time_half vs per_unit)", ""]
    if not fidelity:
        lines.append("Not evaluated (both conditions did not run).")
    else:
        n = len(fidelity)
        for key in ("raw_waveforms", "channel_positions", "cluster_group"):
            k = sum(row[key] for row in fidelity)
            lines.append(f"- {key}: bit-identical in {k}/{n} session bundles")
        differing = sorted(
            {
                (row["seed"], row["scenario"], row["session"])
                for row in fidelity
                if not all(
                    row[k]
                    for k in (
                        "raw_waveforms",
                        "channel_positions",
                        "cluster_group",
                    )
                )
            }
        )
        if differing:
            lines.append(f"- differing (seed, scenario, session): {differing}")

    lines += ["", "## All-zero template halves", ""]
    for condition in conditions:
        runs = [r for r in records if r["condition"] == condition]
        n_zero = sum(
            len(set(r["all_zero_halves"][lab]["half0"]))
            + len(set(r["all_zero_halves"][lab]["half1"]))
            for r in runs
            for lab in SESSIONS
        )
        lines.append(
            f"- {condition}: {n_zero} all-zero unit halves across "
            f"{len(runs)} runs"
        )

    lines += [
        "",
        "## Non-S template halves bit-identical to same-seed control",
        "",
    ]
    for condition in conditions:
        per_scenario = non_s_bit_identical_halves(records, condition)
        if not per_scenario:
            lines.append(f"- {condition}: no scenario had a control run")
            continue
        for scenario, count in per_scenario.items():
            lines.append(
                f"- {condition} {scenario}: {count[0]}/{count[1]} non-S "
                "template halves bit-identical to control"
            )
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main(argv=None) -> None:
    """Run the experiment from the command line."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--first-seed",
        type=int,
        default=DEFAULT_FIRST_SEED,
        help="first seed (runs first-seed .. first-seed + seeds - 1)",
    )
    ap.add_argument(
        "--seeds", type=int, default=DEFAULT_SEEDS, help="seed count"
    )
    ap.add_argument(
        "--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS)
    )
    ap.add_argument(
        "--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS)
    )
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    if args.seeds < 1:
        ap.error("--seeds must be >= 1")
    warnings.filterwarnings("ignore")
    # save_avg_waveforms changes the working directory; keep paths absolute.
    out_dir = (
        Path(tempfile.mkdtemp(prefix="unitmatch_half_split_"))
        if args.out_dir is None
        else args.out_dir
    ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"out-dir: {out_dir}", flush=True)

    records = []
    for seed in range(args.first_seed, args.first_seed + args.seeds):
        t0 = time.perf_counter()
        recording, sorting = make_dataset(seed)
        drift_out_units = choose_drift_out_units(seed)
        for scenario in args.scenarios:
            sessions = make_scenario_sessions(
                recording, sorting, scenario, drift_out_units
            )
            for condition in args.conditions:
                records.append(
                    run_one(
                        seed,
                        scenario,
                        condition,
                        sessions,
                        drift_out_units,
                        out_dir,
                    )
                )
        print(
            f"seed {seed} (S={drift_out_units}) done in "
            f"{time.perf_counter() - t0:.1f} s",
            flush=True,
        )

    gates = evaluate_gates(records)
    fidelity = bundle_fidelity(records)
    summary = format_summary(records, gates, fidelity)
    with open(out_dir / "results.json", "w") as f:
        json.dump(
            {
                "records": records,
                "gates": [asdict(g) for g in gates],
                "fidelity": fidelity,
            },
            f,
            indent=1,
        )
    (out_dir / "summary.md").write_text(summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()

"""Shared constants for the MEArec smoke-fixture sort path.

Several test sites need the same low-amplitude ``clusterless_thresholder``
row to find any peaks on the synthetic MEArec smoke fixture (its
templates max out around a few microvolts; the Frank-lab production
default at 100 uV rejects every peak). Hoisting the row's name and
parameter payload here keeps all five call sites
(``baseline_capture.py``, ``test_phase1_baseline_regen.py``,
``test_run_v2_pipeline_clusterless_preset``,
``test_clusterless_thresholder_end_to_end``,
``test_v2_real_data_v1_parity``) in lockstep -- a future param tweak
lands in one place and the v1↔v2 parity gate stays meaningful.

The ``V1_TO_V2_*_NAMES`` dicts encode the v1-vs-v2 row-name asymmetry
that the v1↔v2 parity test crosses: v1 ships ``preproc_param_name=
"default"`` / ``sorter_param_name="default_clusterless"`` while v2
ships ``preprocessing_params_name="default_franklab"`` /
``sorter_params_name="default"``. Captures done with v1's shipping
default names get mapped to v2's equivalents at parity-test time.
"""

from __future__ import annotations

#: Name of the smoke-fixture-friendly ``clusterless_thresholder`` row.
#: Used by both v1 ``SpikeSorterParameters`` (via ``baseline_capture``)
#: and v2 ``SorterParameters`` (via the parity test); the schemas
#: differ but the name is the same.
SMOKE_CLUSTERLESS_PARAM_NAME = "smoke_clusterless_5uv"

#: ~5x-MAD threshold + ``locally_exclusive`` peak detection tuned for the
#: MEArec smoke fixture. ``threshold_unit="mad"`` is set EXPLICITLY (and
#: ``noise_levels`` omitted) so SpikeInterface computes per-channel MAD and
#: ``detect_threshold`` is interpreted as a MAD multiplier (matching the v1
#: baseline-capture row). The schema default unit is 'uv' (the
#: production/real-data threshold), so WITHOUT this explicit 'mad' the 5
#: would be read as raw microvolts and explode the detection count by
#: ~1,400x on this fixture.
SMOKE_CLUSTERLESS_PARAMS: dict = {
    "detect_threshold": 5.0,
    "threshold_unit": "mad",
    "method": "locally_exclusive",
    "peak_sign": "neg",
    "exclude_sweep_ms": 0.1,
    "local_radius_um": 100.0,
}

#: Name of the polymer-60s MountainSort4 ``SpikeSorterParameters`` row.
#: Owned end-to-end by the parity tests on both pipelines (NOT derived
#: from v1's ``franklab_tetrode_hippocampus_30KHz`` /
#: ``franklab_probe_ctx_30KHz`` shipping rows, which are tetrode-tuned;
#: the polymer probe has 32 channels per shank with a different
#: geometry). v2 capture-side test pre-inserts the same row into v2's
#: ``SorterParameters`` so the canonical_sorter_params fingerprint
#: matches on both sides.
MS4_60S_POLYMER_PARAM_NAME = "ms4_60s_polymer"

#: Polymer-probe 60s MountainSort4 params, tuned for v2's preproc
#: (300-6000 Hz bandpass + median CAR -- ``filter=False`` so MS4 does
#: not double-filter). ``clip_size=50`` (≈1.5ms at 32 kHz) is slightly
#: larger than the v1 ``mountain_default`` of 40 (better captures the
#: full polymer-spike waveform). All other fields match v1's
#: ``franklab_probe_ctx_30KHz`` (``mountain_default + freq_min=300``).
#: Must also be insertable into v2's ``SorterParameters`` (validated
#: by ``MountainSort4Schema`` -- ``extra="forbid"``), so no SI-only
#: keys like ``tempdir`` are stored here.
MS4_60S_POLYMER_PARAMS: dict = {
    "detect_sign": -1,
    "adjacency_radius": 100.0,
    "freq_min": 300.0,
    "freq_max": 6000.0,
    "filter": False,
    "whiten": True,
    "num_workers": 1,
    "clip_size": 50,
    "detect_threshold": 3.0,
    "detect_interval": 10,
}

#: Initial broad MS4 parity bands (n_units ± 50%, median FR ± 30%).
#: MS4 is stochastic (no seed control) AND its SI wrapper rewrote
#: between 0.99 → 0.104 (the C++ MS4 1.0.7 binary itself is byte-
#: identical across envs; differences come from SI-side wrapping).
#: Superseded by :data:`MS4_CALIBRATED` after Phase B11 measurement;
#: kept for reference.
MS4_BROAD_TRIAGE: dict = {
    "n_units_rel_band": 0.50,
    "n_units_abs_band": 2,
    "median_fr_rel_band": 0.30,
}

#: Phase B11 calibration of MS4 within-version variance, measured on
#: ``mearec_polymer_128ch_60s`` shanks 0 and 2 via the B10 protocol
#: (2 runs per side per shank). Schema:
#:
#:     {(fixture_stem, "ms4", shank): {kind: {"d_n_units": int,
#:                                            "d_median_fr_hz": float}}}
#:
#: kinds: ``"v1v1"`` (v1↔v1 drift, deterministic), ``"v2v2"`` (v2↔v2
#: drift), ``"v1v2_max"`` (max(v1↔v2) across paired runs).
#:
#: Key observation: v1 MS4 is deterministic on this fixture (run-to-
#: run drift = 0 in both n_units and median_fr); v2 MS4 has a small
#: stochasticity (≤ 1 unit; ≤ 0.74 Hz, i.e. ≤ 3% of median_fr). MS4's
#: C++ binary is byte-identical across envs (1.0.7); v2's drift comes
#: from SI 0.104's wrapper (possibly the rewritten get_noise_levels
#: per SI PR #3359 or threadpool ordering under the new pool engine).
MS4_VARIANCE_TABLE: dict[tuple[str, str, int], dict] = {
    ("mearec_polymer_128ch_60s", "ms4", 0): {
        "v1v1": {"d_n_units": 0, "d_median_fr_hz": 0.0000},
        "v2v2": {"d_n_units": 1, "d_median_fr_hz": 0.7416},
        "v1v2_max": {"d_n_units": 1, "d_median_fr_hz": 0.7416},
    },
    ("mearec_polymer_128ch_60s", "ms4", 2): {
        "v1v1": {"d_n_units": 0, "d_median_fr_hz": 0.0000},
        "v2v2": {"d_n_units": 1, "d_median_fr_hz": 0.2667},
        "v1v2_max": {"d_n_units": 1, "d_median_fr_hz": 0.2667},
    },
}

#: Calibrated MS4 parity bands derived from :data:`MS4_VARIANCE_TABLE`
#: per the Phase B11 rule ``band = max(v1v1, v2v2) + fixed_margin``
#: where ``fixed_margin = (1 unit, 5 percentage points FR)``:
#:
#:   * n_units: max(0, 1) drift + 1-unit margin = 2 absolute (5% of
#:     40-unit shank 0; 4.2% of 48-unit shank 2). The relative band
#:     is set to 10% to absorb across-shank variation a bit.
#:   * median_fr: the shank-0 v2v2 drift is ``d_median_fr_hz=0.7416``
#:     (see :data:`MS4_VARIANCE_TABLE`) against a shank-0 baseline
#:     median_fr of ≈ 24.4 Hz, i.e. 0.7416 / 24.4 ≈ 3.04% rel drift;
#:     max(0%, 3.04%) + 5pp = 8.04% rel, rounded up to 10% for
#:     headroom. (Shank 2's drift is smaller at
#:     ``d_median_fr_hz=0.2667``.)
#:
#: Substantially tighter than :data:`MS4_BROAD_TRIAGE` (was 50% / 30%);
#: replaces it as the active MS4 contract.
MS4_CALIBRATED: dict = {
    "n_units_rel_band": 0.10,
    "n_units_abs_band": 2,
    "median_fr_rel_band": 0.10,
}

#: Active MS4 parity band set. Tests should reference this constant
#: rather than picking between BROAD/CALIBRATED so the active band
#: lives in one place.
MS4_BANDS: dict = MS4_CALIBRATED

#: v1 -> v2 ``preproc_param_name`` translation for shipped default
#: rows. v1's ``"default"`` is the bandpass + common_reference
#: configuration; v2 ships the same semantics under the explicit name
#: ``"default_franklab"``.
V1_TO_V2_PREPROC_PARAM_NAMES: dict = {
    "default": "default_franklab",
}

#: v1 -> v2 ``sorter_params_name`` translation for shipped default
#: rows. v1's ``"default_clusterless"`` is the 100 uV +
#: ``noise_levels=[1.0]`` row; v2 ships the same configuration under
#: the shorter name ``"default"``. The smoke row name is unchanged
#: across pipelines (both sides use ``smoke_clusterless_5uv``).
V1_TO_V2_SORTER_PARAM_NAMES: dict = {
    "default_clusterless": "default",
}


#: ``(fixture_stem, sorter, sort_group_id)`` triples that legitimately
#: skip the v1↔v2 parity gate because the v1 capture cannot produce
#: a non-degenerate baseline on that shank (MEArec planted no
#: detectable units on the shank, MS4 produced ``< 2`` units, etc.).
#:
#: Each entry MUST be evidence-backed: the reason string should cite
#: the MEArec generator log (planted-unit counts per shank) AND the
#: capture-side output (``v1 sort produced 0/1 unit on this shank``).
#: A label without evidence is not acceptable -- see
#: ``parity-extensions.md`` § "Result taxonomy".
#:
#: Populated during capture-side triage (Phase A10 / B-side captures).
#: An unlisted case with missing baseline artifacts under an active
#: ``SPIKESORTING_V2_BASELINE_ROOT`` is a FAIL, not a SKIP, because a
#: broken tmux capture must not pass silently.
EXPECTED_DEGENERATE_CASES: dict[tuple[str, str, int], str] = {
    # smoke shank 3 on mearec_polymer_smoke: v1 baseline writes n_units=1
    # with exactly ONE spike across the whole 4s recording (approx_last_
    # spike_s=1.58s); v2's locally_exclusive (PR #4341, ratio-not-raw
    # amplitude comparison) correctly does NOT promote that single
    # threshold crossing to a unit. Downstream SI's
    # ``estimate_templates_with_accumulator`` then raises
    # ``estimate_templates() need non empty sorting`` because v2's
    # sorting is empty. Both behaviors are scientifically correct on a
    # near-silent shank; the parity test cannot meaningfully compare
    # "1 lone peak" vs "0 units".
    (
        "mearec_polymer_smoke",
        "clusterless_thresholder",
        3,
    ): (
        "smk shank 3: v1 baseline contains 1 lone spike at "
        "approx_last_spike_s=1.58s on a 4s recording (effectively no "
        "planted unit reaches this shank); v2's improved "
        "locally_exclusive (PR #4341) correctly does not promote that "
        "single threshold crossing to a unit, so downstream "
        "estimate_templates raises 'need non empty sorting'. Both "
        "outcomes are correct on a near-silent shank."
    ),
}


def v2_preproc_name_for_v1(v1_name: str) -> str:
    """Map a v1 ``preproc_param_name`` to the v2 equivalent.

    Returns the v1 name unchanged when no entry exists in
    :data:`V1_TO_V2_PREPROC_PARAM_NAMES`; this lets non-default
    baselines (captured under a custom v1 row) be honored verbatim
    on the v2 side as long as the v2 row exists under the same
    name.
    """
    return V1_TO_V2_PREPROC_PARAM_NAMES.get(v1_name, v1_name)


def v2_sorter_name_for_v1(v1_name: str) -> str:
    """Map a v1 ``sorter_param_name`` to the v2 equivalent.

    Same fallback semantics as :func:`v2_preproc_name_for_v1`.
    """
    return V1_TO_V2_SORTER_PARAM_NAMES.get(v1_name, v1_name)

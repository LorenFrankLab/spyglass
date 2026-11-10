"""Shared constants for the MEArec smoke-fixture sorts and v1/v2 parity tests.

- Smoke clusterless row: the fixture's templates are a few microvolts, so the
  100 uV production default finds no peaks.
- ``V1_TO_V2_*_NAMES``: map v1 shipped row names to their v2 equivalents.
"""

from __future__ import annotations

#: Smoke ``clusterless_thresholder`` row name, shared by v1 and v2.
SMOKE_CLUSTERLESS_PARAM_NAME = "smoke_clusterless_5uv"

#: 5x MAD threshold. ``threshold_unit="mad"`` is explicit because the schema
#: default is uV, which would raise detections ~1,400x on this fixture.
SMOKE_CLUSTERLESS_PARAMS: dict = {
    "detect_threshold": 5.0,
    "threshold_unit": "mad",
    "method": "locally_exclusive",
    "peak_sign": "neg",
    "exclude_sweep_ms": 0.1,
    "local_radius_um": 100.0,
}

#: Polymer-probe MS4 row name, inserted on both pipelines by the parity tests.
MS4_60S_POLYMER_PARAM_NAME = "ms4_60s_polymer"

#: Matches v1 ``franklab_probe_ctx_30KHz`` except ``clip_size=50`` for the
#: wider polymer spike. ``filter=False``: preproc already bandpasses.
#: No SI-only keys (``MountainSort4Schema`` forbids extras).
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

#: Broad MS4 parity bands, kept for reference; :data:`MS4_BANDS` is active.
MS4_BROAD_TRIAGE: dict = {
    "n_units_rel_band": 0.50,
    "n_units_abs_band": 2,
    "median_fr_rel_band": 0.30,
}

#: Measured MS4 run-to-run drift (2 runs per side per shank), keyed
#: ``(fixture_stem, "ms4", shank)``. ``v1v1``/``v2v2``: within-version
#: drift; ``v1v2_max``: max cross-version drift.
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

#: ``band = max(v1v1, v2v2) drift + margin (1 unit, 5 pp FR)``, rounded up
#: to 10%. Shank-0 FR drift 0.74 Hz of ~24.4 Hz median is ~3%.
MS4_CALIBRATED: dict = {
    "n_units_rel_band": 0.10,
    "n_units_abs_band": 2,
    "median_fr_rel_band": 0.10,
}

#: Active MS4 parity bands; tests reference this, not BROAD/CALIBRATED.
MS4_BANDS: dict = MS4_CALIBRATED

#: v1 -> v2 ``preproc_param_name`` map. Empty: both ship ``"default"``.
V1_TO_V2_PREPROC_PARAM_NAMES: dict = {}

#: v1 -> v2 ``sorter_params_name`` map for shipped rows.
V1_TO_V2_SORTER_PARAM_NAMES: dict = {
    "default_clusterless": "default",
}


#: ``(fixture_stem, sorter, sort_group_id)`` cases that skip v1/v2 parity
#: because the v1 baseline is degenerate on that shank. Each reason must
#: cite planted-unit counts and the v1 output. Unlisted missing baselines fail.
EXPECTED_DEGENERATE_CASES: dict[tuple[str, str, int], str] = {
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
    """Map a v1 ``preproc_param_name`` to v2; unmapped names pass through."""
    return V1_TO_V2_PREPROC_PARAM_NAMES.get(v1_name, v1_name)


def v2_sorter_name_for_v1(v1_name: str) -> str:
    """Map a v1 ``sorter_param_name`` to v2; unmapped names pass through."""
    return V1_TO_V2_SORTER_PARAM_NAMES.get(v1_name, v1_name)

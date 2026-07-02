"""SpikeInterface sorter-default snapshot pins.

SI is pinned to ==0.104.3 in pyproject.toml. The KS4/MS5/SC2/TDC2/Generic
v2 schemas use ``extra="allow"`` (KS4 stays permissive, NOT made strict), so
every sorter field the schema does not type falls through to SpikeInterface's
per-version default at sort time. These snapshot tests fail loudly when a SI
bump shifts those defaults so the diff can be audited against current sort
outputs before the pin moves.
"""

from __future__ import annotations

import pytest

# KS4's install-independent SI-wrapper defaults (Kilosort4Sorter
# ._si_default_params). The kilosort-ALGORITHM defaults (Th_universal,
# batch_size, nearest_chans, ...) live in kilosort.parameters.DEFAULT_SETTINGS
# and require the kilosort4 package to be installed; the CI SI-0.104 image does
# NOT install KS4 (get_default_sorter_params('kilosort4') then returns only the
# global job-kwargs + a "not installed" warning -- verified). This snapshot
# pins the SI-controlled wrapper subset (always readable). The algorithm-level
# knobs are pinned by the companion skipif test below, which runs only where
# KS4 is installed.
EXPECTED_KS4_SI_DEFAULTS = {
    "do_CAR": True,
    "invert_sign": False,
    "save_extra_vars": False,
    "save_preprocessed_copy": False,
    "torch_device": "auto",
    "bad_channels": None,
    "clear_cache": False,
    "do_correction": True,
    "skip_kilosort_preprocessing": False,
    "keep_good_only": False,
    "use_binary_file": True,
    "delete_recording_dat": True,
}


# MS5's full default surface (minus the global job-kwargs), SI 0.104.3. The
# v2 MS5 schema types only a subset; the remaining ~8 fields are silently
# stripped/defaulted at sort time (documented in the migration guide).
# Snapshotting them here lets that guide name the actual hidden values.
EXPECTED_MS5_DEFAULTS = {
    "scheme": "2",
    "detect_threshold": 5.5,
    "detect_sign": -1,
    "detect_time_radius_msec": 0.5,
    "snippet_T1": 20,
    "snippet_T2": 20,
    "npca_per_channel": 3,
    "npca_per_subdivision": 10,
    "snippet_mask_radius": 250,
    "scheme1_detect_channel_radius": 150,
    "scheme2_phase1_detect_channel_radius": 200,
    "scheme2_detect_channel_radius": 50,
    "scheme2_max_num_snippets_per_training_batch": 200,
    "scheme2_training_duration_sec": 300,
    "scheme2_training_recording_sampling_mode": "uniform",
    "scheme3_block_duration_sec": 1800,
    "freq_min": 300,
    "freq_max": 6000,
    "filter": True,
    "whiten": True,
    "delete_temporary_recording": True,
}

# SpykingCircus2 / TridesClous2 ship with ``extra="allow"`` (untyped) v2
# schemas and EMPTY default-row params, so their effective defaults float
# with the installed SpikeInterface rather than being frozen into the row
# (as v1 did). These snapshots pin SI 0.104.3's defaults so an SI bump
# that would silently change either sorter's behavior surfaces as a test
# failure -- the same guard the KS4/MS5 snapshots give.
EXPECTED_SC2_DEFAULTS = {
    "apply_motion_correction": True,
    "apply_preprocessing": True,
    "apply_whitening": True,
    "cache_preprocessing": {"memory_limit": 0.5, "mode": "memory"},
    "chunk_preprocessing": {"memory_limit": None},
    "cleaning": {
        "max_jitter_ms": 0.2,
        "mean_sd_ratio_threshold": 3,
        "min_snr": 5,
        "sparsify_threshold": 1,
    },
    "clustering": {"method": "iterative-hdbscan", "method_kwargs": {}},
    "debug": False,
    "detection": {
        "method": "matched_filtering",
        "method_kwargs": {"detect_threshold": 5, "peak_sign": "neg"},
        "pipeline_kwargs": {},
    },
    "deterministic_peaks_detection": False,
    "filtering": {
        "filter_order": 2,
        "freq_max": 7000,
        "freq_min": 150,
        "ftype": "bessel",
    },
    "general": {"ms_after": 1.5, "ms_before": 0.5, "radius_um": 100.0},
    "job_kwargs": {},
    "matching": {
        "method": "circus-omp",
        "method_kwargs": {},
        "pipeline_kwargs": {},
    },
    "merging": {"max_distance_um": 50},
    "min_firing_rate": 0.1,
    "motion_correction": {"preset": "dredge_fast"},
    "multi_units_only": False,
    "seed": 42,
    "selection": {
        "method": "uniform",
        "method_kwargs": {
            "min_n_peaks": 100000,
            "n_peaks_per_channel": 5000,
            "select_per_channel": False,
        },
    },
    "whitening": {"mode": "local", "regularize": False},
}

EXPECTED_TDC2_DEFAULTS = {
    "apply_motion_correction": False,
    "apply_preprocessing": True,
    "cache_preprocessing_mode": "auto",
    "clustering_ms_after": 1.5,
    "clustering_ms_before": 0.5,
    "clustering_recursive_depth": 3,
    "debug": False,
    "detect_threshold": 5.0,
    "detection_radius_um": 150.0,
    "features_radius_um": 120.0,
    "freq_max": 6000.0,
    "freq_min": 150.0,
    "gather_mode": "memory",
    "job_kwargs": {},
    "merge_similarity_lag_ms": 0.5,
    "min_firing_rate": 0.1,
    "motion_correction_preset": "dredge_fast",
    "ms_after": 2.5,
    "ms_before": 1.0,
    "n_pca_features": 6,
    "n_peaks_per_channel": 5000,
    "n_svd_components_per_channel": 5,
    "peak_sign": "neg",
    "preprocessing_dict": None,
    "save_array": True,
    "seed": None,
    "split_radius_um": 60.0,
    "template_max_jitter_ms": 0.2,
    "template_min_snr_ptp": 3.5,
    "template_radius_um": 100.0,
    "template_sparsify_threshold": 1.5,
}


def test_kilosort4_si_defaults_unchanged():
    """SI's install-independent KS4 wrapper defaults match the snapshot.

    Pins ``Kilosort4Sorter._si_default_params`` (the SI-controlled overlay,
    readable without the kilosort4 package). A SI bump that shifts any of these
    surfaces as a test failure rather than a silent change to v2 KS4 sort
    behavior. Diff against the pinned snapshot, decide whether v2's typed-5
    KS4 subset still expresses the right knobs, then update the snapshot and
    the CHANGELOG / SI pin together.
    """
    from spikeinterface.sorters.external.kilosort4 import Kilosort4Sorter

    assert Kilosort4Sorter._si_default_params == EXPECTED_KS4_SI_DEFAULTS, (
        "SI's KS4 wrapper defaults shifted. Diff the change against the "
        "pinned EXPECTED_KS4_SI_DEFAULTS, confirm v2's typed KS4 subset still "
        "covers the right knobs, then update the snapshot, the SI pin in "
        "pyproject.toml, and the CHANGELOG together."
    )


# The kilosort-ALGORITHM defaults flagged as the silent-drift risk
# ("a SI upgrade changing batch_size or nearest_chans defaults"). Values read
# authoritatively from kilosort 4.1.7's parameters.py (MAIN_PARAMETERS) and
# surfaced top-level by SI's KS4 wrapper. Asserted as a SUBSET (not full-dict
# equality): KS4 is an optional, UNPINNED package, so the full default dict
# varies harmlessly across kilosort patch releases -- pinning the full dict
# would false-fail on unrelated version differences. The subset pins exactly
# the knobs whose drift changes sort outputs.
EXPECTED_KS4_ALGORITHM_DEFAULTS = {
    "Th_universal": 9,
    "Th_learned": 8,
    "batch_size": 60000,
    "nearest_chans": 10,
}


def _kilosort4_installed():
    import spikeinterface.sorters as sis

    return "kilosort4" in sis.installed_sorters()


@pytest.mark.skipif(
    not _kilosort4_installed(),
    reason="kilosort4 not installed; algorithm defaults are unreadable "
    "without the package (see test_kilosort4_si_defaults_unchanged for the "
    "install-independent wrapper-overlay snapshot that always runs).",
)
def test_kilosort4_algorithm_defaults_unchanged():
    """KS4's algorithm-level defaults (the drift risk) match.

    Runs only where kilosort4 is installed (skipped on the CI SI-0.104 image).
    Pins the specific knobs flagged -- ``Th_universal`` / ``Th_learned``
    / ``batch_size`` / ``nearest_chans`` -- whose silent drift across a SI/KS4
    bump changes sort outputs. Subset assertion (not full-dict equality) so an
    unrelated kilosort patch-version difference does not false-fail; a change
    to ONE of these knobs surfaces loudly. Note: because this is skipped in the
    KS4-less CI lane, continuous protection requires a KS4-enabled CI job
    (infra follow-up).
    """
    import spikeinterface.sorters as sis

    actual = sis.get_default_sorter_params("kilosort4")
    observed = {
        k: actual[k] for k in EXPECTED_KS4_ALGORITHM_DEFAULTS if k in actual
    }
    assert observed == EXPECTED_KS4_ALGORITHM_DEFAULTS, (
        "KS4 algorithm defaults shifted (or a knob was renamed/removed). "
        f"expected (subset) {EXPECTED_KS4_ALGORITHM_DEFAULTS}, got {observed}. "
        "Audit the change against current sort outputs, then update this "
        "snapshot + the SI pin + the CHANGELOG together."
    )


def test_ms5_si_defaults_unchanged():
    """SI's MountainSort5 defaults match the snapshot.

    MS5 is installed in the SI-0.104 env, so its full default surface is
    readable. Excludes the global job-kwargs (``job_keys``) so the snapshot
    is independent of any per-session job-kwargs config and reflects only the
    MS5 algorithm defaults -- including the fields v2's MS5 schema silently
    strips (the migration guide names them from this snapshot).
    """
    import spikeinterface.sorters as sis
    from spikeinterface.core.job_tools import job_keys

    actual = {
        k: v
        for k, v in sis.get_default_sorter_params("mountainsort5").items()
        if k not in job_keys
    }
    assert actual == EXPECTED_MS5_DEFAULTS, (
        "SI's MountainSort5 defaults shifted. Diff against the pinned "
        "EXPECTED_MS5_DEFAULTS, then update the snapshot, the SI pin in "
        "pyproject.toml, and the CHANGELOG together."
    )


def test_spykingcircus2_si_defaults_unchanged():
    """SI's SpykingCircus2 defaults match the snapshot.

    SC2's v2 schema is ``extra="allow"`` and ships an EMPTY default-row
    params blob, so its effective defaults come from the installed SI at
    sort time. This pins SI 0.104.3's defaults so an SI bump that would
    silently change SC2's behavior surfaces as a test failure (loosen only
    alongside the SI pin in pyproject.toml + the CHANGELOG).
    """
    import spikeinterface.sorters as sis
    from spikeinterface.core.job_tools import job_keys

    actual = {
        k: v
        for k, v in sis.get_default_sorter_params("spykingcircus2").items()
        if k not in job_keys
    }
    assert actual == EXPECTED_SC2_DEFAULTS, (
        "SI's SpykingCircus2 defaults shifted. Diff against the pinned "
        "EXPECTED_SC2_DEFAULTS, then update the snapshot, the SI pin in "
        "pyproject.toml, and the CHANGELOG together."
    )


def test_tridesclous2_si_defaults_unchanged():
    """SI's TridesClous2 defaults match the snapshot.

    Same rationale as SpykingCircus2: TDC2's v2 schema is ``extra="allow"``
    with an empty default-row params blob, so the effective defaults float
    with the installed SI. Pin SI 0.104.3 here so a bump is a deliberate,
    audited change.
    """
    import spikeinterface.sorters as sis
    from spikeinterface.core.job_tools import job_keys

    actual = {
        k: v
        for k, v in sis.get_default_sorter_params("tridesclous2").items()
        if k not in job_keys
    }
    assert actual == EXPECTED_TDC2_DEFAULTS, (
        "SI's TridesClous2 defaults shifted. Diff against the pinned "
        "EXPECTED_TDC2_DEFAULTS, then update the snapshot, the SI pin in "
        "pyproject.toml, and the CHANGELOG together."
    )

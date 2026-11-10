"""Snapshots of SpikeInterface 0.104.3 sorter defaults.

The KS4/MS5/SC2/TDC2/Generic schemas use ``extra="allow"``, so untyped fields
take SI's defaults at sort time. An SI bump that shifts them fails here.
"""

from __future__ import annotations

import pytest

# KS4's SI-wrapper defaults (``Kilosort4Sorter._si_default_params``),
# readable without kilosort4 installed. Algorithm defaults are pinned below
# where kilosort4 is installed.
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


# MS5 defaults minus job-kwargs. The v2 schema types a subset; the migration
# guide names the rest from this snapshot.
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

# SC2 / TDC2 default rows have empty params, so their effective defaults come
# from the installed SpikeInterface.
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

    On failure, check v2's typed KS4 subset, then update the snapshot,
    CHANGELOG and SI pin together.
    """
    from spikeinterface.sorters.external.kilosort4 import Kilosort4Sorter

    assert Kilosort4Sorter._si_default_params == EXPECTED_KS4_SI_DEFAULTS, (
        "SI's KS4 wrapper defaults shifted. Diff the change against the "
        "pinned EXPECTED_KS4_SI_DEFAULTS, confirm v2's typed KS4 subset still "
        "covers the right knobs, then update the snapshot, the SI pin in "
        "pyproject.toml, and the CHANGELOG together."
    )


# Kilosort 4.1.7 algorithm defaults whose drift changes sort outputs. A subset,
# since kilosort4 is unpinned and other defaults vary across patch releases.
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
    """KS4's ``Th_universal`` / ``Th_learned`` / ``batch_size`` /
    ``nearest_chans`` defaults match.

    Skipped where kilosort4 is not installed, including CI.
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

    Excludes the global job-kwargs so config does not affect it.
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

    Update only with the SI pin and CHANGELOG.
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

    Update only with the SI pin and CHANGELOG.
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

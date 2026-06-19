"""June-2026 Frank Lab production parameter recipes (parity source of truth).

Independent literal values for the June 2026 Frank Lab production recipes. The
parity tests compare the SHIPPED parameter rows (the ``_DEFAULT_CONTENTS``
blobs) against these constants. A constant is a hand-written literal, NOT
re-derived from the same Pydantic factory that builds the shipped row --
otherwise the parity check would be a tautology that cannot catch a
schema-default drift.

These encode the **v2-expressible** recipe. v1's production preproc blob also
carried ``margin_ms`` / ``seed``; v2's ``PreprocessingParamsSchema`` has no
such fields (it is the nested ``bandpass_filter`` / ``common_reference`` /
``phase_shift`` structure), so those values have no v2 home and are
intentionally omitted.

Naming: ``franklab_<distinguisher>_<YYYY_MM>``. The ``2026_06`` suffix dates
the recipe; a future change is a NEW dated row, never an in-place edit, so a
derived selection id stays reproducible from a name.
"""

from __future__ import annotations

# --- Dated row names ---------------------------------------------------------

# Preprocessing: region high-pass. Filtering happens at the preproc stage
# (the sorter runs ``filter=False``), so the region band lives HERE.
FRANKLAB_HIPPOCAMPUS_2026_06 = "franklab_hippocampus_2026_06"
FRANKLAB_CORTEX_2026_06 = "franklab_cortex_2026_06"

# Sorter (MountainSort4): rate-keyed. The sorter band is inert under
# ``filter=False``, so the only rate-dependent knobs are ``clip_size`` /
# ``detect_interval`` (they hold the ~1.33 ms physical window across rates).
# The tetrode/probe/region distinction is in the preproc row + the preset,
# never the sorter row.
FRANKLAB_30KHZ_MS4_2026_06 = "franklab_30khz_ms4_2026_06"
FRANKLAB_20KHZ_MS4_2026_06 = "franklab_20khz_ms4_2026_06"

# Artifact detection. (v2's 500 uV / 1.0 schema default stays named "default" --
# it is NOT a production recipe and ships unchanged, so no constant is needed.)
FRANKLAB_100UV_P07_2026_06 = "franklab_100uv_p07_2026_06"
FRANKLAB_50UV_P07_2026_06 = "franklab_50uv_p07_2026_06"


# --- Preprocessing params blobs ----------------------------------------------

# Hippocampus: 600 Hz high-pass (hippocampal spikes are denser/narrower);
# 6000 Hz low-pass; 1.5 ms (0.0015 s) min-segment (production keeps the short
# interval slivers the shipped 1.0 s default drops); median common reference.
FRANKLAB_HIPPOCAMPUS_2026_06_PARAMS = {
    "schema_version": 3,
    "phase_shift": None,
    "bandpass_filter": {"freq_min": 600.0, "freq_max": 6000.0},
    "common_reference": {"operator": "median"},
    "whiten": None,
    "min_segment_length": 0.0015,
    "bad_channel_handling": "remove",
}

# Cortex: identical recipe with the 300 Hz high-pass (cortical waveforms are
# wider).
FRANKLAB_CORTEX_2026_06_PARAMS = {
    "schema_version": 3,
    "phase_shift": None,
    "bandpass_filter": {"freq_min": 300.0, "freq_max": 6000.0},
    "common_reference": {"operator": "median"},
    "whiten": None,
    "min_segment_length": 0.0015,
    "bad_channel_handling": "remove",
}


# --- Artifact detection params blobs -----------------------------------------

# Production artifact: 100 uV amplitude threshold, 0.7 proportion-above, 1.0 ms
# removal window. (join_window_ms / min_length_s are not part of the
# production recipe; the schema defaults of 1.0 / 1.0 are used.)
FRANKLAB_100UV_P07_2026_06_PARAMS = {
    "schema_version": 2,
    "detect": True,
    "amplitude_threshold_uv": 100.0,
    "zscore_threshold": None,
    "proportion_above_threshold": 0.7,
    "removal_window_ms": 1.0,
    "join_window_ms": 1.0,
    "min_length_s": 1.0,
}

# The more aggressive 50 uV variant.
FRANKLAB_50UV_P07_2026_06_PARAMS = {
    "schema_version": 2,
    "detect": True,
    "amplitude_threshold_uv": 50.0,
    "zscore_threshold": None,
    "proportion_above_threshold": 0.7,
    "removal_window_ms": 1.0,
    "join_window_ms": 1.0,
    "min_length_s": 1.0,
}


# --- Sorter (MountainSort4) params blobs -------------------------------------

# Shared MS4 core: detect_sign=-1 (conservative downward-going), adjacency
# radius 100 um, filter=False (the preproc stage already bandpassed; the
# sorter must not double-filter), whiten=True (run externally once),
# detect_threshold=3 (sigma of the ZCA-whitened signal, NOT absolute voltage
# and NOT a MAD multiplier), num_workers=1. ``freq_min``/``freq_max`` are
# INERT here (the sorter does not filter) and stay at the schema defaults.
# 30 kHz window: clip_size=40, detect_interval=10.
FRANKLAB_30KHZ_MS4_2026_06_PARAMS = {
    "schema_version": 1,
    "detect_sign": -1,
    "adjacency_radius": 100.0,
    "freq_min": 600.0,
    "freq_max": 6000.0,
    "filter": False,
    "whiten": True,
    "num_workers": 1,
    "clip_size": 40,
    "detect_threshold": 3.0,
    "detect_interval": 10,
}

# 20 kHz window: clip_size=27, detect_interval=7 hold the same ~1.33 ms
# physical clip across the lower rate; everything else matches the 30 kHz row.
FRANKLAB_20KHZ_MS4_2026_06_PARAMS = {
    "schema_version": 1,
    "detect_sign": -1,
    "adjacency_radius": 100.0,
    "freq_min": 600.0,
    "freq_max": 6000.0,
    "filter": False,
    "whiten": True,
    "num_workers": 1,
    "clip_size": 27,
    "detect_threshold": 3.0,
    "detect_interval": 7,
}

"""June-2026 Frank Lab production parameter recipes (parity source of truth).

Hand-written literals compared against the shipped ``_DEFAULT_CONTENTS``
rows; deriving them from the shipped Pydantic factory would make the parity
check a tautology. v1's ``margin_ms`` / ``seed`` have no v2 field and are
omitted.

Naming: ``franklab_<distinguisher>_<YYYY_MM>``. A recipe change is a new
dated row, never an in-place edit, so selection ids stay reproducible.
"""

from __future__ import annotations

# --- Dated row names ---------------------------------------------------------

# Preprocessing: region high-pass. The sorter runs ``filter=False``, so the
# region band lives here.
FRANKLAB_HIPPOCAMPUS_2026_06 = "franklab_hippocampus_2026_06"
FRANKLAB_CORTEX_2026_06 = "franklab_cortex_2026_06"

# Sorter (MountainSort4): keyed by rate only; ``clip_size`` /
# ``detect_interval`` hold a ~1.33 ms window across rates.
FRANKLAB_30KHZ_MS4_2026_06 = "franklab_30khz_ms4_2026_06"
FRANKLAB_20KHZ_MS4_2026_06 = "franklab_20khz_ms4_2026_06"

# Artifact detection. The 500 uV / 1.0 schema default is "default", not a
# production recipe.
FRANKLAB_100UV_P07_2026_06 = "franklab_100uv_p07_2026_06"
FRANKLAB_50UV_P07_2026_06 = "franklab_50uv_p07_2026_06"


# --- Preprocessing params blobs ----------------------------------------------

# Hippocampus: 600-6000 Hz (narrower spikes); 1.5 ms min-segment keeps the
# short slivers the 1.0 s default drops; median common reference.
FRANKLAB_HIPPOCAMPUS_2026_06_PARAMS = {
    "schema_version": 4,
    "phase_shift": None,
    "bandpass_filter": {"freq_min": 600.0, "freq_max": 6000.0},
    "common_reference": {"operator": "median"},
    "min_segment_length": 0.0015,
    "bad_channel_handling": "remove",
}

# Cortex: same recipe with a 300 Hz high-pass (wider waveforms).
FRANKLAB_CORTEX_2026_06_PARAMS = {
    "schema_version": 4,
    "phase_shift": None,
    "bandpass_filter": {"freq_min": 300.0, "freq_max": 6000.0},
    "common_reference": {"operator": "median"},
    "min_segment_length": 0.0015,
    "bad_channel_handling": "remove",
}


# --- Artifact detection params blobs -----------------------------------------

# join_window_ms / min_length_s are not in the recipe; schema defaults apply.
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

# filter=False: preproc already bandpassed, so freq_min/freq_max are inert.
# detect_threshold is in sigma of the whitened signal, not uV or MAD.
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

# Matches the 30 kHz row except the rate-scaled clip_size / detect_interval.
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

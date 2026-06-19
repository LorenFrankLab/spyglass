"""Parity: shipped ``_DEFAULT_CONTENTS`` rows match the production recipes.

Each test compares a shipped parameter-row blob against an INDEPENDENT
hand-written constant in :mod:`_recipe_constants` (not the Pydantic factory
that built the row), so a schema-default drift that silently changes a
shipped recipe is caught. Schema modules connect on import, so the imports
are lazy and the tests are marked ``database`` -- but no fixture or populate
runs, so they are fast.
"""

from __future__ import annotations

import pytest

from tests.spikesorting.v2 import _recipe_constants as rc


def _catalog_by_name(default_contents, name_index=0, params_index=1):
    """Map ``name -> params blob`` for a single-key Lookup's contents."""
    return {row[name_index]: row[params_index] for row in default_contents}


@pytest.mark.database
def test_region_preproc_recipes_match_production(dj_conn):
    """Hippocampus/cortex preproc rows equal the production recipes.

    Region rule: 600 Hz high-pass for hippocampus, 300 Hz for cortex; both
    6000 Hz low-pass and 1.5 ms min-segment. The high-pass is the preproc
    ``bandpass_filter`` (the sorter runs ``filter=False``), so the region band
    is the only difference between the two recipes.
    """
    from spyglass.spikesorting.v2.recording import PreprocessingParameters

    catalog = _catalog_by_name(PreprocessingParameters._DEFAULT_CONTENTS)

    assert (
        catalog[rc.FRANKLAB_HIPPOCAMPUS_2026_06]
        == rc.FRANKLAB_HIPPOCAMPUS_2026_06_PARAMS
    )
    assert (
        catalog[rc.FRANKLAB_CORTEX_2026_06] == rc.FRANKLAB_CORTEX_2026_06_PARAMS
    )

    hippo = catalog[rc.FRANKLAB_HIPPOCAMPUS_2026_06]
    cortex = catalog[rc.FRANKLAB_CORTEX_2026_06]
    assert hippo["bandpass_filter"]["freq_min"] == 600.0
    assert cortex["bandpass_filter"]["freq_min"] == 300.0
    assert hippo["bandpass_filter"]["freq_max"] == 6000.0
    assert hippo["min_segment_length"] == 0.0015
    assert cortex["min_segment_length"] == 0.0015


@pytest.mark.database
def test_production_artifact_recipes_match(dj_conn):
    """The 100/50 uV @0.7 artifact rows match the production recipes.

    The shipped ``"default"`` row is v2's 500 uV bug-fix value, NOT a
    production recipe; it is asserted to be unchanged here (a later sub-task
    renames it honestly once nothing references the ``"default"`` name).
    """
    from spyglass.spikesorting.v2.artifact import ArtifactDetectionParameters

    catalog = _catalog_by_name(ArtifactDetectionParameters._DEFAULT_CONTENTS)

    assert (
        catalog[rc.FRANKLAB_100UV_P07_2026_06]
        == rc.FRANKLAB_100UV_P07_2026_06_PARAMS
    )
    assert (
        catalog[rc.FRANKLAB_50UV_P07_2026_06]
        == rc.FRANKLAB_50UV_P07_2026_06_PARAMS
    )
    assert catalog["default"]["amplitude_threshold_uv"] == 500.0
    assert (
        catalog[rc.FRANKLAB_100UV_P07_2026_06]["proportion_above_threshold"]
        == 0.7
    )


@pytest.mark.database
def test_ms4_rate_keyed_recipes_match(dj_conn):
    """The rate-keyed MS4 rows match the recipes and are region-agnostic.

    ``filter=False`` (the preproc stage filters, not the sorter) is the
    load-bearing fact: the only difference between the 30 kHz and 20 kHz rows
    is the rate window (``clip_size`` / ``detect_interval``), with no region
    or probe-type encoding on the sorter row.
    """
    from spyglass.spikesorting.v2.sorting import SorterParameters

    # SorterParameters rows are (sorter, name, params, schema_version, jk).
    catalog = {
        (row[0], row[1]): row[2] for row in SorterParameters._DEFAULT_CONTENTS
    }
    ms4_30 = catalog[("mountainsort4", rc.FRANKLAB_30KHZ_MS4_2026_06)]
    ms4_20 = catalog[("mountainsort4", rc.FRANKLAB_20KHZ_MS4_2026_06)]

    assert ms4_30 == rc.FRANKLAB_30KHZ_MS4_2026_06_PARAMS
    assert ms4_20 == rc.FRANKLAB_20KHZ_MS4_2026_06_PARAMS

    assert ms4_30["filter"] is False
    assert ms4_30["adjacency_radius"] == 100.0
    assert (ms4_30["clip_size"], ms4_30["detect_interval"]) == (40, 10)
    assert (ms4_20["clip_size"], ms4_20["detect_interval"]) == (27, 7)
    # Region-agnostic: the two rate rows differ ONLY by the rate window.
    diff = {k for k in ms4_30 if ms4_30[k] != ms4_20[k]}
    assert diff == {"clip_size", "detect_interval"}

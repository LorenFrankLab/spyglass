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

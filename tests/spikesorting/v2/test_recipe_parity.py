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

    The shipped ``"default"`` row is v2's schema-default 500 uV value, NOT a
    production recipe; it stays named ``"default"`` (the franklab presets use
    the 100 uV production row instead) and is asserted unchanged here.
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


@pytest.mark.database
def test_presets_reference_shipped_rows(dj_conn):
    """Every pipeline preset references param rows that actually ship.

    This is the invariant a successful preflight depends on -- a preset must
    not point at a Lookup row that ``insert_default`` does not create.
    """
    from spyglass.spikesorting.v2.artifact import ArtifactDetectionParameters
    from spyglass.spikesorting.v2.pipeline import _PIPELINE_PRESETS
    from spyglass.spikesorting.v2.recording import PreprocessingParameters
    from spyglass.spikesorting.v2.sorting import SorterParameters

    preproc = {r[0] for r in PreprocessingParameters._DEFAULT_CONTENTS}
    artifact = {r[0] for r in ArtifactDetectionParameters._DEFAULT_CONTENTS}
    sorter = {(r[0], r[1]) for r in SorterParameters._DEFAULT_CONTENTS}

    for name, p in _PIPELINE_PRESETS.items():
        assert (
            p.preprocessing_params_name in preproc
        ), f"{name}: preproc {p.preprocessing_params_name!r} not shipped"
        assert (
            p.artifact_detection_params_name in artifact
        ), f"{name}: artifact {p.artifact_detection_params_name!r} not shipped"
        assert (
            p.sorter,
            p.sorter_params_name,
        ) in sorter, (
            f"{name}: sorter {(p.sorter, p.sorter_params_name)!r} absent"
        )


@pytest.mark.database
def test_default_preset_is_production_ms4(dj_conn):
    """run_v2_pipeline's default is the production tetrode-hippocampus MS4."""
    import inspect

    from spyglass.spikesorting.v2 import pipeline as pipeline_mod

    default = (
        inspect.signature(pipeline_mod.run_v2_pipeline)
        .parameters["pipeline_preset"]
        .default
    )
    assert default == "franklab_tetrode_hippocampus_30khz_ms4_2026_06"

    preset = pipeline_mod._PIPELINE_PRESETS[default]
    assert preset.sorter == "mountainsort4"
    assert preset.recommendation_status == "production"
    assert preset.target_region == "hippocampus"

    ms5_name = "franklab_tetrode_hippocampus_30khz_ms5_2026_06"
    assert (
        pipeline_mod._PIPELINE_PRESETS[ms5_name].recommendation_status
        == "alternative"
    )

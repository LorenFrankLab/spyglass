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
def test_default_preset_is_runnable_ms5(dj_conn):
    """run_v2_pipeline defaults to the runnable MS5 tetrode-hippocampus recipe.

    The shipped default is MountainSort5, not the production MountainSort4
    recipe: MS4's ``ml_ms4alg`` backend is a numpy<2-era package that does not
    install under the v2 ``numpy>=2`` baseline, so the default must be a sorter
    that runs out of the box. The recommendation_status taxonomy is unchanged
    (MS4 stays the "production" recipe, selectable but needing numpy<2; MS5
    stays "alternative") -- the function default is a separate,
    runnability-driven choice. ``preflight_v2_pipeline`` shares the default.
    """
    import inspect

    from spyglass.spikesorting.v2 import pipeline as pipeline_mod

    default = (
        inspect.signature(pipeline_mod.run_v2_pipeline)
        .parameters["pipeline_preset"]
        .default
    )
    assert default == "franklab_tetrode_hippocampus_30khz_ms5_2026_06"
    assert (
        inspect.signature(pipeline_mod.preflight_v2_pipeline)
        .parameters["pipeline_preset"]
        .default
        == default
    )

    preset = pipeline_mod._PIPELINE_PRESETS[default]
    assert preset.sorter == "mountainsort5"
    assert preset.target_region == "hippocampus"
    # Taxonomy is unchanged: the default (MS5) keeps its "alternative" tier,
    # and the MS4 family stays the "production" recipe.
    assert preset.recommendation_status == "alternative"
    ms4_name = "franklab_tetrode_hippocampus_30khz_ms4_2026_06"
    assert (
        pipeline_mod._PIPELINE_PRESETS[ms4_name].recommendation_status
        == "production"
    )


@pytest.mark.database
def test_neuropixels_ks4_recipe_matches_aind(dj_conn):
    """The KS4 Neuropixels row matches the AIND capsule recipe + whitens once.

    The shipped ``franklab_neuropixels_default`` (kilosort4) row mirrors the
    AIND ``aind-ephys-spikesort-kilosort4`` ``params.json``: its only
    scientifically-meaningful deviation from stock KS4 is non-rigid drift
    correction (``nblocks=5`` vs stock ``1``), and it pins the whitening /
    preprocessing config explicitly. Critically it carries NO ``whiten`` key:
    v2's external float64 whitening only runs when the sorter params hold
    ``whiten=True`` (``_sorting_compute.py``), so omitting it keeps the signal
    whitened exactly once -- by KS4 -- with no double-whitening.
    """
    from spyglass.spikesorting.v2.sorting import SorterParameters

    catalog = {
        (row[0], row[1]): row[2] for row in SorterParameters._DEFAULT_CONTENTS
    }
    params = catalog[("kilosort4", "franklab_neuropixels_default")]

    # AIND params.json deviations from / explicit pins over stock KS4.
    assert params["nblocks"] == 5  # non-rigid drift (AIND); stock is 1
    assert params["whitening_range"] == 32
    assert params["skip_kilosort_preprocessing"] is False
    assert params["highpass_cutoff"] == 300
    assert params["do_correction"] is True
    assert params["keep_good_only"] is False
    # do_CAR / Th come from the schema defaults and match AIND.
    assert params["do_CAR"] is True
    assert params["Th_universal"] == 9.0
    assert params["Th_learned"] == 8.0

    # Whitening-correctness guard: no ``whiten`` key -> v2 does not externally
    # whiten -> KS4 whitens exactly once.
    assert "whiten" not in params


@pytest.mark.database
def test_neuropixels_ks4_preset_wiring(dj_conn):
    """The experimental NP preset bundles the AIND-matched KS4 row + no-CAR-conflict notes."""
    from spyglass.spikesorting.v2 import pipeline as pipeline_mod

    preset = pipeline_mod._PIPELINE_PRESETS[
        "franklab_neuropixels_ks4_2026_06"
    ]
    assert preset.sorter == "kilosort4"
    assert preset.sorter_params_name == "franklab_neuropixels_default"
    assert preset.preprocessing_params_name == "default_neuropixels"
    assert preset.artifact_detection_params_name == "none"
    assert preset.recommendation_status == "experimental"
    assert preset.sampling_rate_hz == 30000
    # The notes must flag the single-whitening guarantee and the reference
    # caveat so a user does not double common-reference before KS4.
    assert "whiten" in preset.notes.lower()
    assert "reference_mode" in preset.notes

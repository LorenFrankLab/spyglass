"""Tests for the pipeline-preset catalog, registration and cloning in
``v2.pipeline``.
"""

from __future__ import annotations

import math
from pathlib import Path

import datajoint as dj
import pytest

from spyglass.spikesorting.v2.pipeline import (
    _PIPELINE_PRESETS,
    clone_pipeline_preset,
    describe_pipeline_preset,
    describe_pipeline_presets,
    describe_recommendation_status,
    list_pipeline_presets,
    register_pipeline_preset,
)

# tests/spikesorting/v2/ -> repo root is parents[3].
_REPO_ROOT = Path(__file__).resolve().parents[3]
_FEATURE_DOC = _REPO_ROOT / "docs" / "src" / "Features" / "SpikeSortingV2.md"
_NOTEBOOK_PY = _REPO_ROOT / "notebooks" / "py_scripts" / "10_Spike_SortingV2.py"
# The Neuropixels/Kilosort4 preset in the catalog.
_CANONICAL_NPX_PRESET = "franklab_neuropixels_ks4_2026_06"
# A placeholder name not in the catalog; user docs must not use it.
_PLACEHOLDER_NPX_PRESET = "franklab_neuropixels_kilosort4"


def _nullable_num_match(df_value, preset_value) -> bool:
    """Compare a DataFrame cell to a preset's nullable numeric field.

    A ``None`` preset field is coerced to ``NaN`` once it shares a pandas
    column with real numbers, so ``NaN`` (DataFrame) and ``None`` (preset)
    must read as equal here.
    """
    if preset_value is None:
        return df_value is None or (
            isinstance(df_value, float) and math.isnan(df_value)
        )
    return df_value == preset_value


_COLUMNS = [
    "pipeline_preset",
    "recommendation_status",
    "probe_type",
    "target_region",
    "sampling_rate_hz",
    "sorter",
    "sorter_family",
    "adjacency_radius_um",
    "preprocessing_params_name",
    "artifact_detection_params_name",
    "sorter_params_name",
    "metric_params_name",
    "auto_curation_rules_name",
    "intended_use",
    "threshold_units",
    "notes",
]


@pytest.mark.unit
def test_describe_pipeline_presets_lists_all():
    """One row per ``list_pipeline_presets()`` entry, sorted by name."""
    df = describe_pipeline_presets()
    assert len(df) == len(list_pipeline_presets())
    assert set(df["pipeline_preset"]) == set(list_pipeline_presets())
    # describe_pipeline_presets documents a name-sorted return.
    assert df["pipeline_preset"].tolist() == sorted(list_pipeline_presets())


@pytest.mark.unit
def test_describe_pipeline_presets_columns():
    """Columns match the specified set, in order."""
    df = describe_pipeline_presets()
    assert list(df.columns) == _COLUMNS


@pytest.mark.unit
def test_describe_recommendation_status_defines_every_used_status():
    """``describe_recommendation_status`` defines exactly the statuses the
    presets use, each non-empty.
    """
    legend = describe_recommendation_status()
    assert list(legend.columns) == ["recommendation_status", "meaning"]
    assert all(meaning.strip() for meaning in legend["meaning"])

    legend_statuses = set(legend["recommendation_status"])
    used_statuses = {
        status
        for status in describe_pipeline_presets()["recommendation_status"]
        if status
    }
    assert used_statuses  # presets actually carry statuses
    assert used_statuses <= legend_statuses, (
        "preset statuses missing a definition: "
        f"{used_statuses - legend_statuses}"
    )
    assert legend_statuses == {"production", "alternative", "experimental"}


@pytest.mark.unit
def test_describe_pipeline_presets_threshold_units_clusterless():
    """The clusterless row's threshold units are µV, never the MAD ``σ``."""
    df = describe_pipeline_presets()
    row = df.loc[df["pipeline_preset"] == "franklab_clusterless_2026_06"].iloc[
        0
    ]
    units = row["threshold_units"]
    assert "µV" in units
    assert "σ" not in units
    assert "sigma" not in units.lower()


@pytest.mark.unit
def test_describe_pipeline_presets_no_db(monkeypatch):
    """``describe_pipeline_presets`` issues no database query."""

    def _boom(*args, **kwargs):
        raise AssertionError(
            "describe_pipeline_presets() issued a database query; it must be DB-free"
        )

    monkeypatch.setattr(dj.Connection, "query", _boom)
    df = describe_pipeline_presets()
    assert set(df["pipeline_preset"]) == set(list_pipeline_presets())


@pytest.mark.unit
def test_describe_pipeline_presets_matches_preset_objects():
    """Each row's static fields equal the backing ``_PipelinePreset`` attributes."""
    df = describe_pipeline_presets().set_index("pipeline_preset")
    for name, pipeline_preset in _PIPELINE_PRESETS.items():
        row = df.loc[name]
        assert row["recommendation_status"] == (
            pipeline_preset.recommendation_status
        )
        assert row["probe_type"] == pipeline_preset.probe_type
        assert row["target_region"] == pipeline_preset.target_region
        # pandas coerces ``None`` to ``NaN`` in these numeric columns.
        assert _nullable_num_match(
            row["sampling_rate_hz"], pipeline_preset.sampling_rate_hz
        )
        assert row["sorter"] == pipeline_preset.sorter
        assert row["sorter_family"] == pipeline_preset.sorter_family
        assert _nullable_num_match(
            row["adjacency_radius_um"], pipeline_preset.adjacency_radius_um
        )
        assert (
            row["preprocessing_params_name"]
            == pipeline_preset.preprocessing_params_name
        )
        assert (
            row["artifact_detection_params_name"]
            == pipeline_preset.artifact_detection_params_name
        )
        assert row["sorter_params_name"] == pipeline_preset.sorter_params_name
        assert row["metric_params_name"] == pipeline_preset.metric_params_name
        assert (
            row["auto_curation_rules_name"]
            == pipeline_preset.auto_curation_rules_name
        )
        assert row["intended_use"] == pipeline_preset.intended_use
        assert row["threshold_units"] == pipeline_preset.threshold_units
        assert row["notes"] == pipeline_preset.notes


@pytest.mark.unit
def test_container_ms4_pipeline_preset_registered():
    """The containerized MS4 preset (Singularity, 30 kHz) is registered as
    ``production`` and names the container sorter row.

    The execution backend lives on the SorterParameters row, not the preset.
    """
    name = "franklab_probe_hippocampus_30khz_ms4_singularity_2026_06"
    df = describe_pipeline_presets().set_index("pipeline_preset")
    assert name in df.index, f"{name} not registered"
    row = df.loc[name]
    assert row["sorter"] == "mountainsort4"
    assert row["sorter_params_name"] == name  # references the container row
    assert row["recommendation_status"] == "production"
    assert "numpy>=2" in row["notes"]
    assert "Singularity" in row["notes"]

    # The container option does not replace the native hippocampus default.
    from spyglass.spikesorting.v2 import pipeline as pipeline_mod
    import inspect

    default = (
        inspect.signature(pipeline_mod.run_v2_pipeline)
        .parameters["pipeline_preset"]
        .default
    )
    assert default == "franklab_probe_hippocampus_30khz_ms4_2026_06"
    assert pipeline_mod._PIPELINE_PRESETS[default].sorter == "mountainsort4"


@pytest.mark.unit
def test_describe_presets_flags_ms4_recommended():
    """``describe_pipeline_presets`` distinguishes container and local MS4
    presets through ``intended_use`` / ``notes``; both are ``production``.
    """
    container = "franklab_probe_hippocampus_30khz_ms4_singularity_2026_06"
    local = "franklab_probe_hippocampus_30khz_ms4_2026_06"
    df = describe_pipeline_presets().set_index("pipeline_preset")
    assert container in df.index and local in df.index

    container_row = df.loc[container]
    local_row = df.loc[local]

    # The container preset is the recommended MS4 path on numpy>=2 hosts.
    intended = container_row["intended_use"].lower()
    assert "recommended-science" in intended
    assert "modern host" in intended
    assert "numpy>=2" in container_row["notes"]

    # The local MS4 runtime is included in the v2 installation.
    assert "numpy>=2" in local_row["notes"]
    assert "spikesorting-v2 extra" in local_row["notes"]
    assert "recommended-science" not in local_row["intended_use"].lower()

    # Local vs container is an execution choice, not a tier.
    assert container_row["recommendation_status"] == "production"
    assert local_row["recommendation_status"] == "production"


@pytest.mark.unit
def test_describe_pipeline_presets_threshold_units_mountainsort():
    """Every MountainSort preset reports σ-of-whitened-signal units, not µV/MAD."""
    df = describe_pipeline_presets().set_index("pipeline_preset")
    ms_presets = df.index[
        df["sorter_family"].isin(["mountainsort4", "mountainsort5"])
    ]
    assert len(ms_presets) >= 2  # the MS4 family + the MS5 alternative
    for name in ms_presets:
        units = df.loc[name, "threshold_units"]
        assert "sigma" in units.lower()
        assert "MAD" not in units
        assert "µV" not in units


@pytest.mark.unit
def test_describe_pipeline_presets_builtins_populate_human_fields():
    """Every shipped preset sets ``intended_use`` / ``threshold_units`` /
    ``notes`` (optional for external presets).
    """
    df = describe_pipeline_presets().set_index("pipeline_preset")
    for name in list_pipeline_presets():
        for col in ("intended_use", "threshold_units", "notes"):
            assert df.loc[name, col].strip(), f"{name}.{col} is blank"


@pytest.mark.unit
def test_docs_reference_canonical_npx_preset():
    """User docs and notebook name the cataloged KS4/Neuropixels preset, not a
    placeholder, and teach ``describe_pipeline_presets()``.
    """
    assert _CANONICAL_NPX_PRESET in list_pipeline_presets()
    assert _PLACEHOLDER_NPX_PRESET not in list_pipeline_presets()

    doc_text = _FEATURE_DOC.read_text()
    notebook_text = _NOTEBOOK_PY.read_text()

    assert "describe_pipeline_presets()" in doc_text
    assert "describe_pipeline_presets" in notebook_text

    # The feature doc names the real dated preset, never the placeholder.
    assert _CANONICAL_NPX_PRESET in doc_text
    assert _PLACEHOLDER_NPX_PRESET not in doc_text
    assert _PLACEHOLDER_NPX_PRESET not in notebook_text


@pytest.mark.unit
def test_describe_pipeline_preset_unknown_name_raises():
    """An unknown preset name raises before any DB read, pointing at discovery."""
    with pytest.raises(ValueError, match="unknown pipeline_preset"):
        describe_pipeline_preset("definitely_not_a_preset")
    # the message routes the user to the discovery helper
    with pytest.raises(ValueError, match="describe_pipeline_presets"):
        describe_pipeline_preset("definitely_not_a_preset")


@pytest.mark.db_unit
@pytest.mark.database
def test_describe_pipeline_preset_missing_row_points_to_initialize_defaults(
    dj_conn, monkeypatch
):
    """A known preset whose Lookup row is absent raises with a
    ``run initialize_v2_defaults()`` pointer.

    Uses a synthetic preset with one bogus row name, so no shared row is
    deleted.
    """
    import spyglass.spikesorting.v2._orchestration.presets as presets_mod

    base = presets_mod._PIPELINE_PRESETS[
        "franklab_tetrode_hippocampus_30khz_ms5_2026_06"
    ]
    bogus = base.model_copy(
        update={"preprocessing_params_name": "missing_preproc_describe"}
    )
    monkeypatch.setattr(
        presets_mod,
        "_PIPELINE_PRESETS",
        {**presets_mod._PIPELINE_PRESETS, "bogus_missing_row_preset": bogus},
    )
    with pytest.raises(ValueError, match="initialize_v2_defaults"):
        describe_pipeline_preset("bogus_missing_row_preset")


def _custom_spec() -> dict:
    """A valid preset bundle copied from a built-in (rows already exist)."""
    base = _PIPELINE_PRESETS["franklab_tetrode_hippocampus_30khz_ms5_2026_06"]
    return base.model_dump()


@pytest.mark.unit
def test_register_pipeline_preset_adds_to_registry(monkeypatch):
    """A registered preset appears in the catalog (no DB row check)."""
    import spyglass.spikesorting.v2._orchestration.presets as presets_mod

    monkeypatch.setattr(
        presets_mod,
        "_PIPELINE_PRESETS",
        dict(presets_mod._PIPELINE_PRESETS),
    )
    register_pipeline_preset(
        "lab_custom_2026_06", _custom_spec(), validate_rows=False
    )
    assert "lab_custom_2026_06" in list_pipeline_presets()


@pytest.mark.unit
def test_register_pipeline_preset_owns_a_validated_copy(monkeypatch):
    """The registry keeps its own validated model: mutating the caller's
    model afterwards does not change the registered recipe, and a model whose
    fields were made invalid before registration is rejected."""
    import spyglass.spikesorting.v2._orchestration.presets as presets_mod
    from spyglass.spikesorting.v2._orchestration.presets import _PipelinePreset

    monkeypatch.setattr(
        presets_mod,
        "_PIPELINE_PRESETS",
        dict(presets_mod._PIPELINE_PRESETS),
    )
    supplied = _PipelinePreset(**_custom_spec())
    original_sorter = supplied.sorter
    register_pipeline_preset("lab_owned_2026_06", supplied, validate_rows=False)
    supplied.sorter = "mutated_after_registration"
    registered = presets_mod._PIPELINE_PRESETS["lab_owned_2026_06"]
    assert registered is not supplied
    assert registered.sorter == original_sorter

    broken = _PipelinePreset(**_custom_spec())
    broken.sorter = None
    with pytest.raises(ValueError):
        register_pipeline_preset(
            "lab_broken_2026_06", broken, validate_rows=False
        )
    assert "lab_broken_2026_06" not in presets_mod._PIPELINE_PRESETS


@pytest.mark.unit
def test_register_pipeline_preset_rejects_duplicate():
    """Re-registering an existing name raises rather than overwriting."""
    existing = next(iter(_PIPELINE_PRESETS))
    with pytest.raises(ValueError, match="already registered"):
        register_pipeline_preset(existing, _custom_spec(), validate_rows=False)


@pytest.mark.unit
def test_register_pipeline_preset_rejects_unknown_field(monkeypatch):
    """Pydantic extra=forbid rejects a typo'd preset field."""
    import spyglass.spikesorting.v2._orchestration.presets as presets_mod

    monkeypatch.setattr(
        presets_mod,
        "_PIPELINE_PRESETS",
        dict(presets_mod._PIPELINE_PRESETS),
    )
    spec = {**_custom_spec(), "bogus_field": 1}
    with pytest.raises(ValueError):
        register_pipeline_preset("lab_bad_2026_06", spec, validate_rows=False)


@pytest.mark.database
@pytest.mark.db_unit
def test_register_pipeline_preset_catches_missing_lookup_row(
    dj_conn, monkeypatch
):
    """Validating against the DB names the missing row and its table."""
    import spyglass.spikesorting.v2._orchestration.presets as presets_mod

    from spyglass.spikesorting.v2 import initialize_v2_defaults

    initialize_v2_defaults()
    monkeypatch.setattr(
        presets_mod,
        "_PIPELINE_PRESETS",
        dict(presets_mod._PIPELINE_PRESETS),
    )
    spec = {**_custom_spec(), "preprocessing_params_name": "does_not_exist_xyz"}
    with pytest.raises(
        ValueError, match="not found in PreprocessingParameters"
    ):
        register_pipeline_preset("lab_missing_2026_06", spec)


@pytest.mark.unit
def test_register_pipeline_preset_rejects_bad_name(monkeypatch):
    """A non-string or blank name is rejected before touching the registry."""
    import spyglass.spikesorting.v2._orchestration.presets as presets_mod

    monkeypatch.setattr(
        presets_mod,
        "_PIPELINE_PRESETS",
        dict(presets_mod._PIPELINE_PRESETS),
    )
    for bad_name in (1, "", "   ", None):
        with pytest.raises(ValueError, match="non-empty string"):
            register_pipeline_preset(
                bad_name, _custom_spec(), validate_rows=False
            )
    assert 1 not in list_pipeline_presets()


# --------------------------------------------------------------------------- #
# clone_pipeline_preset
# --------------------------------------------------------------------------- #

# Clone base: has a nested preprocessing knob (bandpass freq_min=600) and a
# flat sorter knob (detect_threshold=5.5).
_CLONE_BASE = "franklab_tetrode_hippocampus_30khz_ms5_2026_06"
# The same-day concat preset (labelled for concat mode; no motion pin).
_CONCAT_PRESET = "franklab_concat_hippocampus_30khz_ms5_2026_09"


@pytest.fixture
def clone_env(dj_conn, monkeypatch):
    """Install defaults, isolate the registry, drop derived rows after.

    Tests append every parameter-row name they create to the yielded list,
    which is ``delete_quick``-ed on teardown.
    """
    import spyglass.spikesorting.v2._orchestration.presets as presets_mod
    from spyglass.spikesorting.v2 import initialize_v2_defaults

    initialize_v2_defaults()
    monkeypatch.setattr(
        presets_mod,
        "_PIPELINE_PRESETS",
        dict(presets_mod._PIPELINE_PRESETS),
    )
    names: list[str] = []
    yield names

    from spyglass.spikesorting.v2.artifact import ArtifactDetectionParameters
    from spyglass.spikesorting.v2.recording import PreprocessingParameters
    from spyglass.spikesorting.v2.sorting import (
        PreprocessingWaveformParameters,
        SorterParameters,
    )

    for table, col in (
        (PreprocessingWaveformParameters, "preprocessing_params_name"),
        (PreprocessingParameters, "preprocessing_params_name"),
        (ArtifactDetectionParameters, "artifact_detection_params_name"),
        (SorterParameters, "sorter_params_name"),
    ):
        for name in names:
            (table & {col: name}).delete_quick()


def _stage_value(df, stage, key):
    """Read one flattened ``describe_pipeline_preset`` value for a stage/key."""
    row = df[(df["stage"] == stage) & (df["key"] == key)]
    assert len(row) == 1, f"{stage}.{key} not uniquely present: {len(row)} rows"
    return row.iloc[0]["value"]


@pytest.mark.unit
def test_clone_pipeline_preset_unknown_base_raises():
    """An unknown base preset raises before any DB read, pointing at discovery."""
    with pytest.raises(ValueError, match="unknown"):
        clone_pipeline_preset(
            "definitely_not_a_preset", "lab_clone_2026_06", x=1
        )


@pytest.mark.unit
def test_clone_pipeline_preset_rejects_duplicate_new_name():
    """A new_name already in the registry raises rather than overwriting."""
    existing = next(iter(_PIPELINE_PRESETS))
    with pytest.raises(ValueError, match="already registered"):
        clone_pipeline_preset(_CLONE_BASE, existing, detect_threshold=4.0)


@pytest.mark.unit
def test_clone_pipeline_preset_rejects_bad_new_name(monkeypatch):
    """A non-string or blank new_name is rejected before any DB work."""
    import spyglass.spikesorting.v2._orchestration.presets as presets_mod

    monkeypatch.setattr(
        presets_mod,
        "_PIPELINE_PRESETS",
        dict(presets_mod._PIPELINE_PRESETS),
    )
    for bad_name in (1, "", "   ", None):
        with pytest.raises(ValueError, match="non-empty string"):
            clone_pipeline_preset(_CLONE_BASE, bad_name, detect_threshold=4.0)


@pytest.mark.unit
def test_clone_pipeline_preset_requires_at_least_one_override(monkeypatch):
    """Cloning with no overrides raises (use register_pipeline_preset for an alias)."""
    import spyglass.spikesorting.v2._orchestration.presets as presets_mod

    monkeypatch.setattr(
        presets_mod,
        "_PIPELINE_PRESETS",
        dict(presets_mod._PIPELINE_PRESETS),
    )
    with pytest.raises(ValueError, match="at least one override"):
        clone_pipeline_preset(_CLONE_BASE, "lab_no_override_2026_06")


@pytest.mark.database
@pytest.mark.db_unit
def test_clone_pipeline_preset_flat_sorter_override_round_trips(
    dj_conn, clone_env
):
    """A one-knob sorter override forks only the sorter row; other stages
    reuse the base rows, and the base rows are unchanged.
    """
    import spyglass.spikesorting.v2._orchestration.presets as presets_mod

    new_name = "lab_ms5_thresh_2026_06"
    clone_env.append(new_name)

    base = presets_mod._PIPELINE_PRESETS[_CLONE_BASE]
    base_detail = describe_pipeline_preset(_CLONE_BASE)
    base_threshold = _stage_value(base_detail, "sorter", "detect_threshold")
    assert float(base_threshold) == 5.5  # the shipped MS5 default

    returned = clone_pipeline_preset(
        _CLONE_BASE, new_name, detect_threshold=4.0
    )
    assert returned == new_name
    assert new_name in list_pipeline_presets()

    clone = presets_mod._PIPELINE_PRESETS[new_name]
    # Only the sorter row is forked; untouched stages reuse the base rows.
    assert clone.sorter_params_name == new_name
    assert clone.preprocessing_params_name == base.preprocessing_params_name
    assert (
        clone.artifact_detection_params_name
        == base.artifact_detection_params_name
    )
    assert clone.sorter == base.sorter  # mountainsort5 carried over

    detail = describe_pipeline_preset(new_name)
    assert float(_stage_value(detail, "sorter", "detect_threshold")) == 4.0
    # The untouched preprocessing knob is unchanged in the clone.
    assert (
        float(_stage_value(detail, "preprocessing", "bandpass_filter.freq_min"))
        == 600.0
    )

    # The base preset's sorter row is NOT mutated.
    assert (
        float(
            _stage_value(
                describe_pipeline_preset(_CLONE_BASE),
                "sorter",
                "detect_threshold",
            )
        )
        == 5.5
    )


@pytest.mark.database
@pytest.mark.db_unit
def test_clone_pipeline_preset_nested_dotted_override_round_trips(
    dj_conn, clone_env
):
    """A dotted override edits a nested preprocessing key on a forked row."""
    import spyglass.spikesorting.v2._orchestration.presets as presets_mod

    new_name = "lab_hp700_2026_06"
    clone_env.append(new_name)

    clone_pipeline_preset(
        _CLONE_BASE, new_name, **{"bandpass_filter.freq_min": 700.0}
    )

    base = presets_mod._PIPELINE_PRESETS[_CLONE_BASE]
    clone = presets_mod._PIPELINE_PRESETS[new_name]
    assert clone.preprocessing_params_name == new_name
    # The sorter row is untouched, so it is reused, not forked.
    assert clone.sorter_params_name == base.sorter_params_name

    detail = describe_pipeline_preset(new_name)
    assert (
        float(_stage_value(detail, "preprocessing", "bandpass_filter.freq_min"))
        == 700.0
    )
    # freq_max is carried over from the base unchanged.
    assert (
        float(_stage_value(detail, "preprocessing", "bandpass_filter.freq_max"))
        == 6000.0
    )

    # The base preprocessing row keeps its original high-pass.
    assert (
        float(
            _stage_value(
                describe_pipeline_preset(_CLONE_BASE),
                "preprocessing",
                "bandpass_filter.freq_min",
            )
        )
        == 600.0
    )


@pytest.mark.database
@pytest.mark.db_unit
def test_clone_pipeline_preset_preprocessing_override_keeps_waveform_windows(
    dj_conn, clone_env
):
    """A forked preprocessing row keeps its base recipe's analyzer windows.

    The windows are resolved from the source preprocessing row's name, so a
    clone that tunes a preprocessing knob must not fall back to the cortex
    windows, in this process or a later one.
    """
    from spyglass.spikesorting.v2._core.recipe_catalog import (
        HIPPOCAMPUS_PREPROC,
        waveform_params_for_preprocessing,
    )

    new_name = "lab_hp700_windows_2026_06"
    clone_env.append(new_name)

    clone_pipeline_preset(
        _CLONE_BASE, new_name, **{"bandpass_filter.freq_min": 700.0}
    )

    assert waveform_params_for_preprocessing(
        new_name
    ) == waveform_params_for_preprocessing(HIPPOCAMPUS_PREPROC)


@pytest.mark.database
@pytest.mark.db_unit
def test_clone_pipeline_preset_unknown_override_key_raises(dj_conn, clone_env):
    """An override that matches no stage param raises and inserts nothing."""
    new_name = "lab_bad_key_2026_06"
    clone_env.append(new_name)

    with pytest.raises(ValueError, match="does not match any"):
        clone_pipeline_preset(_CLONE_BASE, new_name, no_such_knob=1)

    # Nothing was registered or inserted.
    assert new_name not in list_pipeline_presets()
    from spyglass.spikesorting.v2.recording import PreprocessingParameters
    from spyglass.spikesorting.v2.sorting import SorterParameters

    assert not (
        PreprocessingParameters & {"preprocessing_params_name": new_name}
    )
    assert not (SorterParameters & {"sorter_params_name": new_name})


@pytest.mark.database
@pytest.mark.db_unit
def test_clone_pipeline_preset_invalid_override_value_raises(
    dj_conn, clone_env
):
    """A schema-invalid override raises the same error as a direct insert
    and inserts nothing.
    """
    new_name = "lab_bad_value_2026_06"
    clone_env.append(new_name)

    with pytest.raises((ValueError, TypeError), match="detect_threshold"):
        clone_pipeline_preset(_CLONE_BASE, new_name, detect_threshold=-1.0)

    assert new_name not in list_pipeline_presets()
    from spyglass.spikesorting.v2.sorting import SorterParameters

    assert not (SorterParameters & {"sorter_params_name": new_name})


@pytest.mark.database
@pytest.mark.db_unit
def test_clone_pipeline_preset_duplicate_content_under_new_name_raises(
    dj_conn, clone_env
):
    """A derived row matching another existing row raises
    ``DuplicateParameterContentError``.

    A 300 Hz high-pass reproduces the shipped cortex preprocessing row.
    """
    from spyglass.spikesorting.v2.exceptions import (
        DuplicateParameterContentError,
    )

    new_name = "lab_dup_content_2026_06"
    clone_env.append(new_name)

    with pytest.raises(
        DuplicateParameterContentError, match="franklab_cortex_2026_06"
    ):
        clone_pipeline_preset(
            _CLONE_BASE, new_name, **{"bandpass_filter.freq_min": 300.0}
        )
    assert new_name not in list_pipeline_presets()


@pytest.mark.database
@pytest.mark.db_unit
def test_clone_pipeline_preset_allow_duplicate_params_opts_in(
    dj_conn, clone_env
):
    """``allow_duplicate_params=True`` forks content matching another row."""
    new_name = "lab_dup_optin_2026_06"
    clone_env.append(new_name)

    clone_pipeline_preset(
        _CLONE_BASE,
        new_name,
        allow_duplicate_params=True,
        **{"bandpass_filter.freq_min": 300.0},
    )
    assert new_name in list_pipeline_presets()
    detail = describe_pipeline_preset(new_name)
    assert (
        float(_stage_value(detail, "preprocessing", "bandpass_filter.freq_min"))
        == 300.0
    )


@pytest.mark.database
@pytest.mark.db_unit
def test_clone_pipeline_preset_name_collision_different_content_raises(
    dj_conn, clone_env
):
    """A derived-row name that already exists with different content raises."""
    from spyglass.spikesorting.v2._params.preprocessing import (
        PreprocessingParamsSchema,
    )
    from spyglass.spikesorting.v2.recording import PreprocessingParameters

    new_name = "lab_name_collision_2026_06"
    clone_env.append(new_name)

    # Pre-seed a preprocessing row under new_name with unrelated content
    # (freq_min=777 is unique among the shipped rows).
    PreprocessingParameters.insert1(
        {
            "preprocessing_params_name": new_name,
            "params": PreprocessingParamsSchema.model_validate(
                {"bandpass_filter": {"freq_min": 777.0, "freq_max": 6000.0}}
            ).model_dump(),
        }
    )

    with pytest.raises(ValueError, match="different content"):
        clone_pipeline_preset(
            _CLONE_BASE, new_name, **{"bandpass_filter.freq_min": 700.0}
        )
    assert new_name not in list_pipeline_presets()


@pytest.mark.database
@pytest.mark.db_unit
def test_clone_pipeline_preset_idempotent_rerun(dj_conn, clone_env):
    """Re-running a clone with identical overrides in a fresh process reuses
    the existing rows.
    """
    import spyglass.spikesorting.v2._orchestration.presets as presets_mod

    new_name = "lab_idempotent_2026_06"
    clone_env.append(new_name)

    clone_pipeline_preset(_CLONE_BASE, new_name, detect_threshold=4.0)
    # Simulate a fresh process: the DB rows persist but the registry does not.
    presets_mod._PIPELINE_PRESETS.pop(new_name)

    # Identical re-run must succeed and reflect the same override.
    clone_pipeline_preset(_CLONE_BASE, new_name, detect_threshold=4.0)
    assert new_name in list_pipeline_presets()
    detail = describe_pipeline_preset(new_name)
    assert float(_stage_value(detail, "sorter", "detect_threshold")) == 4.0

    # Exactly one sorter row exists under the derived name.
    from spyglass.spikesorting.v2.sorting import SorterParameters

    assert len(SorterParameters & {"sorter_params_name": new_name}) == 1


# --------------------------------------------------------------------------- #
# preset curation fields (metric / auto-curation)
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_every_preset_declares_curation_params():
    """Every shipped preset names a quality-metric + auto-curation rule set.

    - Polymer/tetrode: ``franklab_default`` with the ISI-aware rules.
    - Neuropixels: ``v1_default_nn_noise_2026_09`` rules.
    - Clusterless: ``minimal`` metrics with ``none`` rules.
    """
    for name, preset in _PIPELINE_PRESETS.items():
        assert preset.metric_params_name.strip(), f"{name}.metric blank"
        assert preset.auto_curation_rules_name.strip(), f"{name}.rules blank"

    # Every shipped preset runs an artifact stage.
    for name, preset in _PIPELINE_PRESETS.items():
        assert preset.artifact_detection_params_name is not None, name

    clusterless = _PIPELINE_PRESETS["franklab_clusterless_2026_06"]
    assert clusterless.metric_params_name == "minimal"
    assert clusterless.auto_curation_rules_name == "none"

    npx = _PIPELINE_PRESETS["franklab_neuropixels_ks4_2026_06"]
    assert npx.metric_params_name == "neuropixels_default"
    assert npx.auto_curation_rules_name == "v1_default_nn_noise_2026_09"

    excluded = {
        "franklab_clusterless_2026_06",
        "franklab_neuropixels_ks4_2026_06",
    }
    for name, preset in _PIPELINE_PRESETS.items():
        if name in excluded:
            continue
        assert preset.metric_params_name == "franklab_default", name
        assert (
            preset.auto_curation_rules_name
            == "franklab_default_auto_curation_2026_09"
        ), name


@pytest.mark.unit
def test_concat_preset_is_registered_and_shaped():
    """The concat preset masks each member with the probe recipe's artifact
    detector, pins no motion recipe, and has no motion-pinned alias."""
    assert _CONCAT_PRESET in list_pipeline_presets()
    assert (
        "franklab_concat_hippocampus_30khz_ms5_2026_06"
        not in list_pipeline_presets()
    )
    preset = _PIPELINE_PRESETS[_CONCAT_PRESET]
    assert (
        preset.artifact_detection_params_name
        == _PIPELINE_PRESETS[
            "franklab_probe_hippocampus_30khz_ms5_2026_06"
        ].artifact_detection_params_name
    )
    assert not any("motion" in field for field in type(preset).model_fields)
    assert preset.recommendation_status == "experimental"
    assert preset.sorter == "mountainsort5"
    assert preset.metric_params_name == "franklab_default"
    assert (
        preset.auto_curation_rules_name
        == "franklab_default_auto_curation_2026_09"
    )


@pytest.mark.database
@pytest.mark.db_unit
def test_run_v2_concat_pipeline_accepts_single_session_preset(dj_conn):
    """A concat run does not require a motion-pinned preset: a single-session
    preset reaches the concat preflight, which rejects only the absent
    session group."""
    from spyglass.spikesorting.v2._core.recipe_catalog import (
        DEFAULT_PIPELINE_PRESET,
    )
    from spyglass.spikesorting.v2.exceptions import PreflightError
    from spyglass.spikesorting.v2.pipeline import run_v2_concat_pipeline

    with pytest.raises(PreflightError, match="does not exist"):
        run_v2_concat_pipeline(
            "no_such_owner_xyz",
            "no_such_group_xyz",
            pipeline_preset=DEFAULT_PIPELINE_PRESET,
        )


_POSITIONAL_SOURCE_FIELDS = {
    "run_v2_pipeline": [
        "nwb_file_name",
        "sort_group_id",
        "interval_list_name",
        "team_name",
    ],
    "run_v2_concat_pipeline": ["session_group_owner", "session_group_name"],
    "estimate_motion": [
        "nwb_file_name",
        "sort_group_id",
        "interval_list_name",
        "team_name",
    ],
    "estimate_concat_motion": ["session_group_owner", "session_group_name"],
    "run_v2_pipeline_session": [
        "nwb_file_name",
        "interval_list_name",
        "team_name",
    ],
    "preflight_v2_pipeline": [
        "nwb_file_name",
        "sort_group_id",
        "interval_list_name",
        "team_name",
    ],
    "preflight_v2_pipeline_session": [
        "nwb_file_name",
        "interval_list_name",
        "team_name",
    ],
}


@pytest.mark.unit
@pytest.mark.parametrize("name", sorted(_POSITIONAL_SOURCE_FIELDS))
def test_entry_points_take_only_source_fields_positionally(name):
    """Each entry point's source fields are its only positional parameters,
    so an option (a preset, a boolean flag) is always named at the call."""
    import inspect

    from spyglass.spikesorting.v2 import pipeline

    params = inspect.signature(getattr(pipeline, name)).parameters.values()
    positional = [p.name for p in params if p.kind is p.POSITIONAL_OR_KEYWORD]
    assert positional == _POSITIONAL_SOURCE_FIELDS[name]
    assert all(
        p.kind is p.KEYWORD_ONLY for p in params if p.name not in positional
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "single, concat",
    [
        ("run_v2_pipeline", "run_v2_concat_pipeline"),
        ("estimate_motion", "estimate_concat_motion"),
    ],
)
def test_single_and_concat_entry_points_share_their_options(single, concat):
    """The single-session and concat entry points take the same options with
    the same defaults; only their source fields differ."""
    import inspect

    from spyglass.spikesorting.v2 import pipeline

    def options(name):
        return {
            p.name: p.default
            for p in inspect.signature(
                getattr(pipeline, name)
            ).parameters.values()
            if p.kind is p.KEYWORD_ONLY
        }

    assert options(single) == options(concat)


@pytest.mark.unit
def test_run_v2_pipeline_rejects_a_none_source_input():
    """A source field passed as ``None`` fails DB-free, naming the called
    entry point's own fields, even when every one of them is ``None``."""
    from spyglass.spikesorting.v2.exceptions import PipelineInputError
    from spyglass.spikesorting.v2.pipeline import (
        estimate_concat_motion,
        run_v2_concat_pipeline,
        run_v2_pipeline,
    )

    with pytest.raises(
        PipelineInputError, match="sort_group_id cannot be None"
    ):
        run_v2_pipeline("x.nwb", None, "i", "t")
    with pytest.raises(
        PipelineInputError, match="session_group_name cannot be None"
    ):
        run_v2_concat_pipeline("owner", None)
    both_missing = (
        r"^{}: required input\(s\) session_group_owner, "
        r"session_group_name cannot be None\.$"
    )
    with pytest.raises(
        PipelineInputError,
        match=both_missing.format("run_v2_concat_pipeline"),
    ):
        run_v2_concat_pipeline(None, None)
    with pytest.raises(
        PipelineInputError,
        match=both_missing.format("estimate_concat_motion"),
    ):
        estimate_concat_motion(
            None, None, motion_correction_params_name="default"
        )


@pytest.mark.unit
def test_run_v2_unit_match_signature_defaults():
    """run_v2_unit_match defaults the matcher to unitmatch_default + no choices.

    ``curation_choices`` defaults to ``None`` so the explicit-choice requirement
    is enforced at call time (never auto-pick a curation). DB-free.
    """
    import inspect

    from spyglass.spikesorting.v2.pipeline import run_v2_unit_match

    params = inspect.signature(run_v2_unit_match).parameters
    assert params["plan"].default is None
    assert params["session_group_name"].default is None
    assert params["matcher_params_name"].default == "unitmatch_default"
    assert params["curation_choices"].default is None
    assert params["session_group_owner"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["session_group_name"].kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.unit
def test_run_v2_unit_match_requires_explicit_curation_choices():
    """run_v2_unit_match rejects ``curation_choices=None`` before any DB
    access; "latest curation" would not be reproducible.
    """
    from spyglass.spikesorting.v2.exceptions import PipelineInputError
    from spyglass.spikesorting.v2.pipeline import run_v2_unit_match

    with pytest.raises(PipelineInputError, match="curation_choices"):
        run_v2_unit_match(
            session_group_owner="owner", session_group_name="group"
        )


@pytest.mark.database
@pytest.mark.db_unit
def test_run_v2_unit_match_unknown_matcher_raises(dj_conn):
    """An unknown matcher_params_name raises PipelineInputError before any
    SessionGroup or curation access.
    """
    from spyglass.spikesorting.v2.exceptions import PipelineInputError
    from spyglass.spikesorting.v2.pipeline import run_v2_unit_match

    with pytest.raises(PipelineInputError, match="unknown matcher_params_name"):
        run_v2_unit_match(
            session_group_owner="no_such_owner",
            session_group_name="no_such_group",
            matcher_params_name="definitely_not_a_matcher_2026",
            curation_choices={0: {"sorting_id": "x", "curation_id": 0}},
            unit_policy="v2_unflagged_single_units",
        )


@pytest.mark.unit
def test_preset_model_artifact_optional_and_no_motion_field():
    """``_PipelinePreset`` makes the artifact recipe optional and forbids extras.

    ``artifact_detection_params_name`` is optional (``None`` for presets that
    run no artifact detection). Presets carry no motion recipe, so a
    ``motion_correction_params_name`` is rejected like any unknown field.
    """
    import spyglass.spikesorting.v2._orchestration.presets as presets_mod

    preset = presets_mod._PipelinePreset(
        preprocessing_params_name="franklab_hippocampus_2026_06",
        artifact_detection_params_name=None,
        sorter="mountainsort5",
        sorter_params_name="franklab_30khz_ms5_2026_06",
        metric_params_name="franklab_default",
        auto_curation_rules_name="v1_default_nn_noise_2026_09",
    )
    assert preset.artifact_detection_params_name is None

    with pytest.raises(ValueError):
        presets_mod._PipelinePreset(
            preprocessing_params_name="franklab_hippocampus_2026_06",
            sorter="mountainsort5",
            sorter_params_name="franklab_30khz_ms5_2026_06",
            metric_params_name="franklab_default",
            auto_curation_rules_name="v1_default_nn_noise_2026_09",
            motion_correction_params_name="auto_default",
        )
    with pytest.raises(ValueError):
        presets_mod._PipelinePreset(
            preprocessing_params_name="franklab_hippocampus_2026_06",
            sorter="mountainsort5",
            sorter_params_name="franklab_30khz_ms5_2026_06",
            metric_params_name="franklab_default",
            auto_curation_rules_name="v1_default_nn_noise_2026_09",
            bogus_field=1,
        )


def _spec_with(**overrides) -> dict:
    """A valid built-in preset spec with field overrides applied."""
    return {**_custom_spec(), **overrides}


@pytest.mark.database
@pytest.mark.db_unit
def test_register_pipeline_preset_catches_missing_metric_row(
    dj_conn, monkeypatch
):
    """A preset naming an absent quality-metric row fails with a clear pointer."""
    import spyglass.spikesorting.v2._orchestration.presets as presets_mod

    from spyglass.spikesorting.v2 import initialize_v2_defaults

    initialize_v2_defaults()
    monkeypatch.setattr(
        presets_mod,
        "_PIPELINE_PRESETS",
        dict(presets_mod._PIPELINE_PRESETS),
    )
    spec = _spec_with(metric_params_name="missing_metric_xyz")
    with pytest.raises(
        ValueError, match="not found in QualityMetricParameters"
    ):
        register_pipeline_preset("lab_missing_metric_2026_06", spec)


@pytest.mark.database
@pytest.mark.db_unit
def test_register_pipeline_preset_catches_missing_auto_curation_row(
    dj_conn, monkeypatch
):
    """A preset naming an absent auto-curation rule set fails clearly."""
    import spyglass.spikesorting.v2._orchestration.presets as presets_mod

    from spyglass.spikesorting.v2 import initialize_v2_defaults

    initialize_v2_defaults()
    monkeypatch.setattr(
        presets_mod,
        "_PIPELINE_PRESETS",
        dict(presets_mod._PIPELINE_PRESETS),
    )
    spec = _spec_with(auto_curation_rules_name="missing_rules_xyz")
    with pytest.raises(ValueError, match="not found in AutoCurationRules"):
        register_pipeline_preset("lab_missing_rules_2026_06", spec)


@pytest.mark.database
@pytest.mark.db_unit
def test_describe_pipeline_preset_surfaces_curation_names(dj_conn, clone_env):
    """The preset detail view surfaces the metric + auto-curation row names."""
    detail = describe_pipeline_preset(_CLONE_BASE)
    assert (
        _stage_value(detail, "preset", "metric_params_name")
        == "franklab_default"
    )
    assert (
        _stage_value(detail, "preset", "auto_curation_rules_name")
        == "franklab_default_auto_curation_2026_09"
    )


@pytest.mark.database
@pytest.mark.db_unit
def test_describe_pipeline_preset_unpacks_curation_values(dj_conn, clone_env):
    """The detail view unpacks metric kwargs and label rules under their own
    stages, not just row names.
    """
    detail = describe_pipeline_preset(_CLONE_BASE)

    # Metric stage unpacks the QualityMetricParameters columns (metric_names /
    # skip_pc_metrics are always emitted; metric_kwargs only when non-empty).
    metric = detail[detail["stage"] == "metric"]
    assert not metric.empty
    metric_keys = set(metric["key"])
    assert "metric_names" in metric_keys
    assert "skip_pc_metrics" in metric_keys
    assert (metric["params_row_name"] == "franklab_default").all()

    # Auto-curation stage carries the master fields plus one row per label rule.
    auto = detail[detail["stage"] == "auto_curation"]
    assert not auto.empty
    assert "auto_merge_preset" in set(auto["key"])
    rule_rows = auto[auto["key"].str.startswith("rule.")]
    assert not rule_rows.empty, "expected at least one auto-curation rule row"
    # Each rule value reads as "<name>: <metric> <op> <threshold> -> <label>".
    assert all("->" in v for v in rule_rows["value"])


@pytest.mark.database
@pytest.mark.db_unit
def test_describe_pipeline_preset_artifact_none_skips_artifact(
    dj_conn, clone_env
):
    """A preset with a ``None`` artifact stage unpacks without an artifact
    section.
    """
    name = "lab_no_artifact_2026_06"
    register_pipeline_preset(
        name,
        _spec_with(artifact_detection_params_name=None),
        validate_rows=False,
    )
    detail = describe_pipeline_preset(name)
    assert (detail["stage"] == "artifact_detection").sum() == 0
    # The other stages still resolve.
    assert (detail["stage"] == "preprocessing").sum() > 0
    assert (detail["stage"] == "sorter").sum() > 0


@pytest.mark.database
@pytest.mark.db_unit
def test_clone_pipeline_preset_no_artifact_base(dj_conn, clone_env):
    """Cloning a no-artifact preset forks only the touched stage and inherits
    the ``None`` artifact.
    """
    import spyglass.spikesorting.v2._orchestration.presets as presets_mod

    new_name = "lab_concat_thresh_2026_06"
    clone_env.append(new_name)

    base_name = "test_concat_no_artifacts"
    clone_env.append(base_name)
    presets_mod._PIPELINE_PRESETS[base_name] = presets_mod._PIPELINE_PRESETS[
        _CONCAT_PRESET
    ].model_copy(update={"artifact_detection_params_name": None})
    clone_pipeline_preset(base_name, new_name, detect_threshold=4.0)

    clone = presets_mod._PIPELINE_PRESETS[new_name]
    assert clone.artifact_detection_params_name is None
    assert clone.sorter_params_name == new_name

    detail = describe_pipeline_preset(new_name)
    assert float(_stage_value(detail, "sorter", "detect_threshold")) == 4.0
    assert (detail["stage"] == "artifact_detection").sum() == 0


def test_auto_curation_reuses_a_twin_presets_child_description():
    """A twin preset's auto-labeled child is reused, not duplicated.

    Twin presets (tetrode and probe labels of one recipe) give the same child
    curation, whose description names the preset that ran first.
    """
    from types import SimpleNamespace

    from spyglass.spikesorting.v2._orchestration.stages import (
        _auto_curation_description,
    )

    def preset(metric, rules):
        return SimpleNamespace(
            metric_params_name=metric, auto_curation_rules_name=rules
        )

    presets = {
        "probe_twin": preset("franklab_default", "franklab_rules"),
        "tetrode_twin": preset("franklab_default", "franklab_rules"),
        "other_rules": preset("franklab_default", "other"),
    }

    def describe(name):
        return f"run_v2_pipeline auto-curation (pipeline_preset={name})"

    # First run: the description names the preset that ran.
    assert _auto_curation_description("probe_twin", set(), presets) == (
        describe("probe_twin")
    )
    # The twin finds the existing child's description.
    assert _auto_curation_description(
        "tetrode_twin", {describe("probe_twin")}, presets
    ) == describe("probe_twin")
    # Its own description wins when both exist.
    assert _auto_curation_description(
        "tetrode_twin",
        {describe("probe_twin"), describe("tetrode_twin")},
        presets,
    ) == describe("tetrode_twin")
    # A preset with other rules gives another child.
    assert _auto_curation_description(
        "other_rules", {describe("probe_twin")}, presets
    ) == describe("other_rules")

"""Unit tests for the pipeline-preset discovery accessors in ``v2.pipeline``.

``describe_pipeline_presets()`` is a pure, database-free read of the in-module
``_PIPELINE_PRESETS`` metadata: one row per shipped pipeline preset with the
discovery axes (recommendation status, probe type, target region, sampling
rate, sorter family, adjacency radius), the per-stage parameter-row names, the
intended use, and the detection-threshold units. These tests pin its contract
(row set, column set/order, parity with the ``_PipelinePreset`` objects, the
clusterless-vs-MountainSort threshold-unit footgun) and assert it issues no
SQL.
"""

from __future__ import annotations

import math
from pathlib import Path

import datajoint as dj
import pytest

from spyglass.spikesorting.v2.pipeline import (
    _PIPELINE_PRESETS,
    clone_preset,
    describe_pipeline_preset,
    describe_pipeline_presets,
    describe_preset,
    list_pipeline_presets,
    register_preset,
)

pytestmark = pytest.mark.unit

# tests/spikesorting/v2/ -> repo root is parents[3].
_REPO_ROOT = Path(__file__).resolve().parents[3]
_FEATURE_DOC = _REPO_ROOT / "docs" / "src" / "Features" / "SpikeSortingV2.md"
_NOTEBOOK_PY = _REPO_ROOT / "notebooks" / "py_scripts" / "10_Spike_SortingV2.py"
# The dated, content-fingerprinted Neuropixels/Kilosort4 preset actually in the
# catalog. Docs/notebooks discover it via describe_pipeline_presets(); this is
# the name they may reference, never an undated placeholder.
_CANONICAL_NPX_PRESET = "franklab_neuropixels_ks4_2026_06"
# A never-shipped placeholder name that must not appear in user docs -- it would
# rot, since the catalog only exposes the dated preset above.
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
    "motion_correction_params_name",
    "intended_use",
    "threshold_units",
    "notes",
]


def test_describe_pipeline_presets_lists_all():
    """One row per ``list_pipeline_presets()`` entry, sorted by name."""
    df = describe_pipeline_presets()
    assert len(df) == len(list_pipeline_presets())
    assert set(df["pipeline_preset"]) == set(list_pipeline_presets())
    # describe_pipeline_presets documents a name-sorted return.
    assert df["pipeline_preset"].tolist() == sorted(list_pipeline_presets())


def test_describe_pipeline_presets_columns():
    """Columns match the specified set, in order."""
    df = describe_pipeline_presets()
    assert list(df.columns) == _COLUMNS


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


def test_describe_pipeline_presets_no_db(monkeypatch):
    """``describe_pipeline_presets`` issues no database query (pure, DB-free).

    Any SQL would route through ``Connection.query``; patch it to raise so an
    accidental table read regresses loudly instead of silently hitting the DB.
    """

    def _boom(*args, **kwargs):
        raise AssertionError(
            "describe_pipeline_presets() issued a database query; it must be DB-free"
        )

    monkeypatch.setattr(dj.Connection, "query", _boom)
    df = describe_pipeline_presets()
    assert set(df["pipeline_preset"]) == set(list_pipeline_presets())


def test_describe_pipeline_presets_matches_preset_objects():
    """Each row's static fields equal the backing ``_PipelinePreset`` attributes.

    Guards against a second hard-coded copy of pipeline-preset data drifting from the
    ``_PIPELINE_PRESETS`` source of truth.
    """
    df = describe_pipeline_presets().set_index("pipeline_preset")
    for name, pipeline_preset in _PIPELINE_PRESETS.items():
        row = df.loc[name]
        assert row["recommendation_status"] == (
            pipeline_preset.recommendation_status
        )
        assert row["probe_type"] == pipeline_preset.probe_type
        assert row["target_region"] == pipeline_preset.target_region
        # ``sampling_rate_hz`` / ``adjacency_radius_um`` are numeric-nullable:
        # pandas coerces a preset's ``None`` to ``NaN`` in the mixed column,
        # so compare NaN-aware rather than with bare ``==``.
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
        # motion is optional (None for single-session presets); pandas coerces a
        # None to NaN only in an all-numeric column, but this column is all
        # strings/None, so compare directly.
        assert (
            row["motion_correction_params_name"]
            == pipeline_preset.motion_correction_params_name
        )
        assert row["intended_use"] == pipeline_preset.intended_use
        assert row["threshold_units"] == pipeline_preset.threshold_units
        assert row["notes"] == pipeline_preset.notes


def test_container_ms4_pipeline_preset_registered():
    """The one containerized MS4 preset (Singularity, 30 kHz) is registered.

    The execution backend is NOT a preset field -- it lives on the referenced
    SorterParameters row -- so ``describe_pipeline_presets`` (DB-free) carries no
    execution columns. The preset names the containerized sorter row, stays
    ``recommendation_status="production"`` (the recommended-science MS4 path),
    and its notes describe the modern-host (numpy>=2) container path; the
    function default remains MountainSort5.
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

    # No shipped preset is the default-switching kind: the default stays MS5.
    from spyglass.spikesorting.v2 import pipeline as pipeline_mod
    import inspect

    default = (
        inspect.signature(pipeline_mod.run_v2_pipeline)
        .parameters["pipeline_preset"]
        .default
    )
    assert pipeline_mod._PIPELINE_PRESETS[default].sorter == "mountainsort5"


def test_describe_presets_flags_ms4_recommended():
    """``describe_pipeline_presets`` distinguishes container vs local MS4 paths.

    The containerized polymer MS4 preset is surfaced as the recommended-science
    MS4 path for modern (``numpy>=2``) hosts; the local polymer MS4 preset is
    surfaced as the compatible-local-runtime (``numpy<2``) path. Both are flagged
    purely through the HUMAN-FACING fields (``recommendation_status`` /
    ``intended_use`` / ``notes``) -- the execution backend is not a preset column
    -- so a scientist reading the catalog can tell which MS4 path to reach for.
    """
    container = "franklab_probe_hippocampus_30khz_ms4_singularity_2026_06"
    local = "franklab_probe_hippocampus_30khz_ms4_2026_06"
    df = describe_pipeline_presets().set_index("pipeline_preset")
    assert container in df.index and local in df.index

    container_row = df.loc[container]
    local_row = df.loc[local]

    # The container preset reads as the recommended-science MS4 path on modern
    # hosts -- its intended_use says so, and its notes confirm the host stays on
    # numpy>=2 because the runtime lives in the image.
    intended = container_row["intended_use"].lower()
    assert "recommended-science" in intended
    assert "modern host" in intended
    assert "numpy>=2" in container_row["notes"]

    # The local MS4 preset is documented for compatible local runtimes: its
    # notes call out the numpy<2 requirement, and it does NOT claim to be the
    # recommended-science modern-host path (that distinction is the point).
    assert "numpy<2" in local_row["notes"]
    assert "recommended-science" not in local_row["intended_use"].lower()

    # Both stay the production MS4 recipe (the tier is unchanged); the
    # local-vs-container split is a runtime-host distinction, not a tier one.
    assert container_row["recommendation_status"] == "production"
    assert local_row["recommendation_status"] == "production"


def test_describe_pipeline_presets_threshold_units_mountainsort():
    """Every MountainSort preset reports σ-of-whitened-signal units, not µV/MAD.

    The complement of the clusterless footgun: MountainSort ``detect_threshold``
    is a multiple of the standard deviation of the ZCA-whitened signal (~3 for
    MS4, ~5.5 for MS5), so mislabeling it with absolute microvolts -- or with a
    MAD multiplier, which belongs to the clusterless ``detect_peaks`` path --
    would be the same class of error in the other direction.
    """
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


def test_describe_pipeline_presets_builtins_populate_human_fields():
    """Every shipped pipeline preset populates the human-facing fields.

    The ``intended_use`` / ``threshold_units`` / ``notes`` fields default to
    ``""`` so external presets need not supply them, but the built-ins must --
    a blank here means a user reading the catalog learns nothing about the
    pipeline preset.
    """
    df = describe_pipeline_presets().set_index("pipeline_preset")
    for name in list_pipeline_presets():
        for col in ("intended_use", "threshold_units", "notes"):
            assert df.loc[name, col].strip(), f"{name}.{col} is blank"


def test_docs_reference_canonical_npx_preset():
    """User docs/notebook discover the KS4/Neuropixels preset, not a placeholder.

    The dated, fingerprinted preset ``franklab_neuropixels_ks4_2026_06`` is the
    one actually in the live catalog, so the docs surface it through
    ``describe_pipeline_presets()`` rather than naming the never-shipped
    ``franklab_neuropixels_kilosort4`` placeholder. This keeps the docs honest:
    they only name a preset the catalog actually exposes, and teach discovery
    rather than a name that could churn.
    """
    # The dated preset is a real catalog entry (so docs can't reference a
    # non-existent name), and the placeholder never made it into the catalog.
    assert _CANONICAL_NPX_PRESET in list_pipeline_presets()
    assert _PLACEHOLDER_NPX_PRESET not in list_pipeline_presets()

    doc_text = _FEATURE_DOC.read_text()
    notebook_text = _NOTEBOOK_PY.read_text()

    # Discovery, not a hardcoded list: both surfaces teach the accessor.
    assert "describe_pipeline_presets()" in doc_text
    assert "describe_pipeline_presets" in notebook_text

    # The feature doc names the real dated preset, never the placeholder.
    assert _CANONICAL_NPX_PRESET in doc_text
    assert _PLACEHOLDER_NPX_PRESET not in doc_text
    assert _PLACEHOLDER_NPX_PRESET not in notebook_text


def test_describe_pipeline_preset_unknown_name_raises():
    """An unknown preset name raises before any DB read, pointing at discovery.

    The name check is the first thing ``describe_pipeline_preset`` does, so this
    path is DB-free; only the value-unpack (a passing name) touches the Lookup
    tables. The full-unpack happy path lives in the integration suite.
    """
    with pytest.raises(ValueError, match="unknown pipeline_preset"):
        describe_pipeline_preset("definitely_not_a_preset")
    # the message routes the user to the discovery helper
    with pytest.raises(ValueError, match="describe_pipeline_presets"):
        describe_pipeline_preset("definitely_not_a_preset")


def test_describe_preset_is_alias():
    """``describe_preset`` is the shorter discovery alias for the same helper."""
    assert describe_preset is describe_pipeline_preset


@pytest.mark.database
def test_describe_pipeline_preset_missing_row_points_to_initialize_defaults(
    dj_conn, monkeypatch
):
    """A known preset whose referenced Lookup row is absent raises with a
    ``run initialize_v2_defaults()`` pointer, not an opaque empty-fetch.

    The name check passes (the preset is registered), so this exercises the
    value-unpack path that reads the Lookup tables: a missing parameter row
    must fail with actionable guidance. A synthetic preset (a copy of the
    runnable default with one bogus row name) drives the missing-row branch
    without deleting any shared default row.
    """
    import spyglass.spikesorting.v2._pipeline_presets as presets_mod

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


def test_register_preset_adds_to_registry(monkeypatch):
    """A registered preset appears in the catalog (no DB row check)."""
    import spyglass.spikesorting.v2._pipeline_presets as presets_mod

    monkeypatch.setattr(
        presets_mod,
        "_PIPELINE_PRESETS",
        dict(presets_mod._PIPELINE_PRESETS),
    )
    register_preset("lab_custom_2026_06", _custom_spec(), validate_rows=False)
    assert "lab_custom_2026_06" in list_pipeline_presets()


def test_register_preset_rejects_duplicate():
    """Re-registering an existing name raises rather than overwriting."""
    existing = next(iter(_PIPELINE_PRESETS))
    with pytest.raises(ValueError, match="already registered"):
        register_preset(existing, _custom_spec(), validate_rows=False)


def test_register_preset_rejects_unknown_field(monkeypatch):
    """Pydantic extra=forbid rejects a typo'd preset field."""
    import spyglass.spikesorting.v2._pipeline_presets as presets_mod

    monkeypatch.setattr(
        presets_mod,
        "_PIPELINE_PRESETS",
        dict(presets_mod._PIPELINE_PRESETS),
    )
    spec = {**_custom_spec(), "bogus_field": 1}
    with pytest.raises(ValueError):
        register_preset("lab_bad_2026_06", spec, validate_rows=False)


def test_register_preset_catches_missing_lookup_row(dj_conn, monkeypatch):
    """Validating against the DB names the missing row and its table."""
    import spyglass.spikesorting.v2._pipeline_presets as presets_mod

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
        register_preset("lab_missing_2026_06", spec)


def test_register_preset_rejects_bad_name(monkeypatch):
    """A non-string or blank name is rejected before touching the registry."""
    import spyglass.spikesorting.v2._pipeline_presets as presets_mod

    monkeypatch.setattr(
        presets_mod,
        "_PIPELINE_PRESETS",
        dict(presets_mod._PIPELINE_PRESETS),
    )
    for bad_name in (1, "", "   ", None):
        with pytest.raises(ValueError, match="non-empty string"):
            register_preset(bad_name, _custom_spec(), validate_rows=False)
    assert 1 not in list_pipeline_presets()


# --------------------------------------------------------------------------- #
# clone_preset
# --------------------------------------------------------------------------- #

# A runnable MS5 built-in used as the clone base across the DB tests: its
# preprocessing row (franklab_hippocampus_2026_06) carries a bandpass_filter
# (freq_min=600) and its mountainsort5 sorter row carries a detect_threshold
# (5.5), so it exercises both a nested preprocessing override and a flat sorter
# override.
_CLONE_BASE = "franklab_tetrode_hippocampus_30khz_ms5_2026_06"
# The same-day concat preset: the one shipped no-artifact / motion-pinned preset.
_CONCAT_PRESET = "franklab_concat_hippocampus_30khz_ms5_2026_06"


@pytest.fixture
def clone_env(dj_conn, monkeypatch):
    """Install defaults, isolate the registry, drop derived rows after.

    The clone base is a shipped preset, so its parameter rows must exist;
    ``initialize_v2_defaults()`` installs them (idempotent). Mutating the
    module-global ``_PIPELINE_PRESETS`` and inserting derived parameter rows
    would leak across tests, so swap in a copy of the registry and record the
    derived-row names to ``delete_quick`` on teardown. Tests append every
    parameter-row name they create (clone-derived or pre-seeded) to the yielded
    list.
    """
    import spyglass.spikesorting.v2._pipeline_presets as presets_mod
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
    from spyglass.spikesorting.v2.sorting import SorterParameters

    for table, col in (
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


def test_clone_preset_unknown_base_raises():
    """An unknown base preset raises before any DB read, pointing at discovery."""
    with pytest.raises(ValueError, match="unknown"):
        clone_preset("definitely_not_a_preset", "lab_clone_2026_06", x=1)


def test_clone_preset_rejects_duplicate_new_name():
    """A new_name already in the registry raises rather than overwriting."""
    existing = next(iter(_PIPELINE_PRESETS))
    with pytest.raises(ValueError, match="already registered"):
        clone_preset(_CLONE_BASE, existing, detect_threshold=4.0)


def test_clone_preset_rejects_bad_new_name(monkeypatch):
    """A non-string or blank new_name is rejected before any DB work."""
    import spyglass.spikesorting.v2._pipeline_presets as presets_mod

    monkeypatch.setattr(
        presets_mod,
        "_PIPELINE_PRESETS",
        dict(presets_mod._PIPELINE_PRESETS),
    )
    for bad_name in (1, "", "   ", None):
        with pytest.raises(ValueError, match="non-empty string"):
            clone_preset(_CLONE_BASE, bad_name, detect_threshold=4.0)


def test_clone_preset_requires_at_least_one_override(monkeypatch):
    """Cloning with no overrides raises (use register_preset for an alias)."""
    import spyglass.spikesorting.v2._pipeline_presets as presets_mod

    monkeypatch.setattr(
        presets_mod,
        "_PIPELINE_PRESETS",
        dict(presets_mod._PIPELINE_PRESETS),
    )
    with pytest.raises(ValueError, match="at least one override"):
        clone_preset(_CLONE_BASE, "lab_no_override_2026_06")


def test_clone_preset_flat_sorter_override_round_trips(dj_conn, clone_env):
    """A one-knob sorter override produces a working, inspectable clone.

    Overriding the MS5 ``detect_threshold`` registers a new preset whose
    sorter row reflects the change while the untouched preprocessing/artifact
    rows are reused (same names as the base), and the base preset's own rows
    are left unchanged.
    """
    import spyglass.spikesorting.v2._pipeline_presets as presets_mod

    new_name = "lab_ms5_thresh_2026_06"
    clone_env.append(new_name)

    base = presets_mod._PIPELINE_PRESETS[_CLONE_BASE]
    base_detail = describe_pipeline_preset(_CLONE_BASE)
    base_threshold = _stage_value(base_detail, "sorter", "detect_threshold")
    assert float(base_threshold) == 5.5  # the shipped MS5 default

    returned = clone_preset(_CLONE_BASE, new_name, detect_threshold=4.0)
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


def test_clone_preset_nested_dotted_override_round_trips(dj_conn, clone_env):
    """A dotted override edits a nested preprocessing key on a forked row."""
    import spyglass.spikesorting.v2._pipeline_presets as presets_mod

    new_name = "lab_hp700_2026_06"
    clone_env.append(new_name)

    clone_preset(_CLONE_BASE, new_name, **{"bandpass_filter.freq_min": 700.0})

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


def test_clone_preset_unknown_override_key_raises(dj_conn, clone_env):
    """An override that matches no stage param raises and inserts nothing."""
    new_name = "lab_bad_key_2026_06"
    clone_env.append(new_name)

    with pytest.raises(ValueError, match="does not match any"):
        clone_preset(_CLONE_BASE, new_name, no_such_knob=1)

    # Nothing was registered or inserted.
    assert new_name not in list_pipeline_presets()
    from spyglass.spikesorting.v2.recording import PreprocessingParameters
    from spyglass.spikesorting.v2.sorting import SorterParameters

    assert not (
        PreprocessingParameters & {"preprocessing_params_name": new_name}
    )
    assert not (SorterParameters & {"sorter_params_name": new_name})


def test_clone_preset_invalid_override_value_raises(dj_conn, clone_env):
    """A schema-invalid override raises the same teaching error as a direct insert.

    ``detect_threshold`` must be > 0; a negative value fails the MS5 Pydantic
    schema, and nothing is registered or inserted.
    """
    new_name = "lab_bad_value_2026_06"
    clone_env.append(new_name)

    with pytest.raises((ValueError, TypeError), match="detect_threshold"):
        clone_preset(_CLONE_BASE, new_name, detect_threshold=-1.0)

    assert new_name not in list_pipeline_presets()
    from spyglass.spikesorting.v2.sorting import SorterParameters

    assert not (SorterParameters & {"sorter_params_name": new_name})


def test_clone_preset_duplicate_content_under_new_name_raises(
    dj_conn, clone_env
):
    """A derived row whose content matches a different existing row raises.

    Overriding the hippocampus preprocessing high-pass to 300 Hz reproduces
    the shipped cortex preprocessing row exactly, so forking it under a new
    name is refused with ``DuplicateParameterContentError`` rather than
    silently duplicating provenance.
    """
    from spyglass.spikesorting.v2.exceptions import (
        DuplicateParameterContentError,
    )

    new_name = "lab_dup_content_2026_06"
    clone_env.append(new_name)

    with pytest.raises(
        DuplicateParameterContentError, match="franklab_cortex_2026_06"
    ):
        clone_preset(
            _CLONE_BASE, new_name, **{"bandpass_filter.freq_min": 300.0}
        )
    assert new_name not in list_pipeline_presets()


def test_clone_preset_allow_duplicate_params_opts_in(dj_conn, clone_env):
    """``allow_duplicate_params=True`` forks content matching another row.

    The same override that is refused by default (it reproduces the cortex
    preprocessing row) is allowed when the duplicate-content opt-in is passed,
    registering the clone under the new name.
    """
    new_name = "lab_dup_optin_2026_06"
    clone_env.append(new_name)

    clone_preset(
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


def test_clone_preset_name_collision_different_content_raises(
    dj_conn, clone_env
):
    """A derived-row name that already exists with different content raises.

    A parameter row already named like the clone, but with different content,
    must not be silently re-pointed; the clone refuses the name collision.
    """
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
        clone_preset(
            _CLONE_BASE, new_name, **{"bandpass_filter.freq_min": 700.0}
        )
    assert new_name not in list_pipeline_presets()


def test_clone_preset_idempotent_rerun(dj_conn, clone_env):
    """Re-running a clone with identical overrides is a no-op, not a fork.

    The derived rows are content-addressed by name, so a second run (after the
    in-memory registration is cleared, as on a fresh process) reuses the
    existing rows without raising a duplicate-content error.
    """
    import spyglass.spikesorting.v2._pipeline_presets as presets_mod

    new_name = "lab_idempotent_2026_06"
    clone_env.append(new_name)

    clone_preset(_CLONE_BASE, new_name, detect_threshold=4.0)
    # Simulate a fresh process: the DB rows persist but the registry does not.
    presets_mod._PIPELINE_PRESETS.pop(new_name)

    # Identical re-run must succeed and reflect the same override.
    clone_preset(_CLONE_BASE, new_name, detect_threshold=4.0)
    assert new_name in list_pipeline_presets()
    detail = describe_pipeline_preset(new_name)
    assert float(_stage_value(detail, "sorter", "detect_threshold")) == 4.0

    # Exactly one sorter row exists under the derived name.
    from spyglass.spikesorting.v2.sorting import SorterParameters

    assert len(SorterParameters & {"sorter_params_name": new_name}) == 1


# --------------------------------------------------------------------------- #
# preset curation fields (metric / auto-curation / motion)
# --------------------------------------------------------------------------- #


def test_every_preset_declares_curation_params():
    """Every shipped preset names a quality-metric + auto-curation rule set.

    These became required fields, so a preset that omitted them would fail to
    build. The shipped mapping: franklab / Neuropixels presets pair their
    ``*_default`` metric set with the noise-labeling ``v1_default_nn_noise``
    rules, while the clusterless preset (no clustered units to merge) pairs the
    ``minimal`` metric set with the inert ``none`` rules.
    """
    for name, preset in _PIPELINE_PRESETS.items():
        assert preset.metric_params_name.strip(), f"{name}.metric blank"
        assert preset.auto_curation_rules_name.strip(), f"{name}.rules blank"

    # The single-session presets run an artifact stage and leave motion
    # correction unset; the same-day concat preset is the inverse (no artifact
    # stage, motion pinned).
    for name, preset in _PIPELINE_PRESETS.items():
        if name == _CONCAT_PRESET:
            continue
        assert preset.artifact_detection_params_name is not None, name
        assert preset.motion_correction_params_name is None, name

    clusterless = _PIPELINE_PRESETS["franklab_clusterless_2026_06"]
    assert clusterless.metric_params_name == "minimal"
    assert clusterless.auto_curation_rules_name == "none"

    ms5 = _PIPELINE_PRESETS[_CLONE_BASE]
    assert ms5.metric_params_name == "franklab_default"
    assert ms5.auto_curation_rules_name == "v1_default_nn_noise"

    npx = _PIPELINE_PRESETS["franklab_neuropixels_ks4_2026_06"]
    assert npx.metric_params_name == "neuropixels_default"
    assert npx.auto_curation_rules_name == "v1_default_nn_noise"


def test_concat_preset_is_registered_and_shaped():
    """The same-day concat preset ships with no artifact stage + pinned motion.

    It is the one shipped preset whose ``artifact_detection_params_name`` is
    None (concat sorts carry no ArtifactDetectionSource row) and whose
    ``motion_correction_params_name`` is set ("auto" -> rigid_fast for a
    same-day group). Otherwise it is the MS5 hippocampus recipe.
    """
    assert _CONCAT_PRESET in list_pipeline_presets()
    preset = _PIPELINE_PRESETS[_CONCAT_PRESET]
    assert preset.artifact_detection_params_name is None
    assert preset.motion_correction_params_name == "auto_default"
    assert preset.sorter == "mountainsort5"
    assert preset.metric_params_name == "franklab_default"
    assert preset.auto_curation_rules_name == "v1_default_nn_noise"


def test_preset_model_artifact_optional_and_motion_field():
    """``_PipelinePreset`` accepts a concat-shaped preset and forbids extras.

    ``artifact_detection_params_name`` is optional (``None`` for concat presets
    that run no artifact detection) and ``motion_correction_params_name`` is an
    optional field a concat preset sets; unknown fields are still rejected.
    """
    import spyglass.spikesorting.v2._pipeline_presets as presets_mod

    preset = presets_mod._PipelinePreset(
        preprocessing_params_name="franklab_hippocampus_2026_06",
        artifact_detection_params_name=None,
        sorter="mountainsort5",
        sorter_params_name="franklab_30khz_ms5_2026_06",
        metric_params_name="franklab_default",
        auto_curation_rules_name="v1_default_nn_noise",
        motion_correction_params_name="auto",
    )
    assert preset.artifact_detection_params_name is None
    assert preset.motion_correction_params_name == "auto"

    with pytest.raises(ValueError):
        presets_mod._PipelinePreset(
            preprocessing_params_name="franklab_hippocampus_2026_06",
            sorter="mountainsort5",
            sorter_params_name="franklab_30khz_ms5_2026_06",
            metric_params_name="franklab_default",
            auto_curation_rules_name="v1_default_nn_noise",
            bogus_field=1,
        )


def _spec_with(**overrides) -> dict:
    """A valid built-in preset spec with field overrides applied."""
    return {**_custom_spec(), **overrides}


def test_register_preset_catches_missing_metric_row(dj_conn, monkeypatch):
    """A preset naming an absent quality-metric row fails with a clear pointer."""
    import spyglass.spikesorting.v2._pipeline_presets as presets_mod

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
        register_preset("lab_missing_metric_2026_06", spec)


def test_register_preset_catches_missing_auto_curation_row(
    dj_conn, monkeypatch
):
    """A preset naming an absent auto-curation rule set fails clearly."""
    import spyglass.spikesorting.v2._pipeline_presets as presets_mod

    from spyglass.spikesorting.v2 import initialize_v2_defaults

    initialize_v2_defaults()
    monkeypatch.setattr(
        presets_mod,
        "_PIPELINE_PRESETS",
        dict(presets_mod._PIPELINE_PRESETS),
    )
    spec = _spec_with(auto_curation_rules_name="missing_rules_xyz")
    with pytest.raises(ValueError, match="not found in AutoCurationRules"):
        register_preset("lab_missing_rules_2026_06", spec)


def test_register_preset_catches_missing_motion_row(dj_conn, monkeypatch):
    """A preset naming an absent motion-correction row fails clearly.

    The motion row is only checked when the preset sets it (single-session
    presets leave it None and skip the check).
    """
    import spyglass.spikesorting.v2._pipeline_presets as presets_mod

    from spyglass.spikesorting.v2 import initialize_v2_defaults

    initialize_v2_defaults()
    monkeypatch.setattr(
        presets_mod,
        "_PIPELINE_PRESETS",
        dict(presets_mod._PIPELINE_PRESETS),
    )
    spec = _spec_with(motion_correction_params_name="missing_motion_xyz")
    with pytest.raises(
        ValueError, match="not found in MotionCorrectionParameters"
    ):
        register_preset("lab_missing_motion_2026_06", spec)


def test_describe_pipeline_preset_surfaces_curation_names(dj_conn, clone_env):
    """The preset detail view surfaces the metric + auto-curation row names."""
    detail = describe_pipeline_preset(_CLONE_BASE)
    assert (
        _stage_value(detail, "preset", "metric_params_name")
        == "franklab_default"
    )
    assert (
        _stage_value(detail, "preset", "auto_curation_rules_name")
        == "v1_default_nn_noise"
    )


def test_describe_pipeline_preset_artifact_none_skips_artifact(
    dj_conn, clone_env
):
    """A preset with no artifact stage unpacks without an artifact section.

    Registered with ``validate_rows=False`` (it references real preproc / sorter
    / curation rows but a ``None`` artifact), the detail view must skip the
    artifact stage rather than raise on a ``None`` row name.
    """
    name = "lab_no_artifact_2026_06"
    register_preset(
        name,
        _spec_with(artifact_detection_params_name=None),
        validate_rows=False,
    )
    detail = describe_pipeline_preset(name)
    assert (detail["stage"] == "artifact_detection").sum() == 0
    # The other stages still resolve.
    assert (detail["stage"] == "preprocessing").sum() > 0
    assert (detail["stage"] == "sorter").sum() > 0


def test_clone_preset_no_artifact_base(dj_conn, clone_env):
    """Cloning a no-artifact preset works and forks only the touched stage.

    The concat preset runs no artifact stage, so a clone that tunes the sorter
    must not try to fetch a base artifact row; only the sorter stage is forked
    and the clone inherits the None artifact.
    """
    import spyglass.spikesorting.v2._pipeline_presets as presets_mod

    new_name = "lab_concat_thresh_2026_06"
    clone_env.append(new_name)

    clone_preset(_CONCAT_PRESET, new_name, detect_threshold=4.0)

    clone = presets_mod._PIPELINE_PRESETS[new_name]
    assert clone.artifact_detection_params_name is None
    assert clone.sorter_params_name == new_name
    assert clone.motion_correction_params_name == "auto_default"

    detail = describe_pipeline_preset(new_name)
    assert float(_stage_value(detail, "sorter", "detect_threshold")) == 4.0
    assert (detail["stage"] == "artifact_detection").sum() == 0

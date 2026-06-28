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

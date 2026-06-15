"""Unit tests for the pipeline-preset discovery accessors in ``v2.pipeline``.

``describe_pipeline_presets()`` is a pure, database-free read of the in-module
``_PIPELINE_PRESETS`` metadata: one row per shipped pipeline preset with the
sorter, the per-stage parameter-row names, the intended use, and the
detection-threshold units. These tests pin its contract (row set, column
set/order, parity with the ``_PipelinePreset`` objects, the clusterless
threshold-unit footgun) and assert it issues no SQL.
"""

from __future__ import annotations

import datajoint as dj
import pytest

from spyglass.spikesorting.v2.pipeline import (
    _PIPELINE_PRESETS,
    describe_pipeline_presets,
    list_pipeline_presets,
)

pytestmark = pytest.mark.unit

_COLUMNS = [
    "pipeline_preset",
    "sorter",
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
    row = df.loc[
        df["pipeline_preset"] == "franklab_tetrode_clusterless_thresholder"
    ].iloc[0]
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
        assert row["sorter"] == pipeline_preset.sorter
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


def test_describe_pipeline_presets_threshold_units_mountainsort():
    """Both MountainSort presets report a MAD-multiplier threshold, not µV.

    The complement of the clusterless footgun: MountainSort ``detect_threshold``
    is a noise-relative MAD multiplier, so mislabeling it with absolute
    microvolts would be the same class of error in the other direction.
    """
    df = describe_pipeline_presets().set_index("pipeline_preset")
    for name in (
        "franklab_tetrode_mountainsort4",
        "franklab_tetrode_mountainsort5",
    ):
        units = df.loc[name, "threshold_units"]
        assert "MAD" in units
        assert "µV" not in units
        assert "sigma" not in units.lower()


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

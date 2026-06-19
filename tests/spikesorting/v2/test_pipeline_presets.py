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

import datajoint as dj
import pytest

from spyglass.spikesorting.v2.pipeline import (
    _PIPELINE_PRESETS,
    describe_pipeline_presets,
    list_pipeline_presets,
)

pytestmark = pytest.mark.unit


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
    row = df.loc[
        df["pipeline_preset"] == "franklab_clusterless_2026_06"
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

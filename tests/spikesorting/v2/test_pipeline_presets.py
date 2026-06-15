"""Unit tests for the preset-discovery accessors in ``v2.pipeline``.

``describe_presets()`` is a pure, database-free read of the in-module
``_PRESETS`` metadata: one row per shipped preset with the sorter, the
per-stage parameter-row names, the intended use, and the detection-threshold
units. These tests pin its contract (row set, column set/order, parity with
the ``_Preset`` objects, the clusterless threshold-unit footgun) and assert it
issues no SQL.
"""

from __future__ import annotations

import datajoint as dj
import pytest

from spyglass.spikesorting.v2.pipeline import (
    _PRESETS,
    describe_presets,
    list_presets,
)

pytestmark = pytest.mark.unit

_COLUMNS = [
    "preset",
    "sorter",
    "preprocessing_params_name",
    "artifact_detection_params_name",
    "sorter_params_name",
    "intended_use",
    "threshold_units",
    "notes",
]


def test_describe_presets_lists_all():
    """One row per ``list_presets()`` entry, sorted by preset name."""
    df = describe_presets()
    assert len(df) == len(list_presets())
    assert set(df["preset"]) == set(list_presets())
    # describe_presets documents a name-sorted return.
    assert df["preset"].tolist() == sorted(list_presets())


def test_describe_presets_columns():
    """Columns match the specified set, in order."""
    df = describe_presets()
    assert list(df.columns) == _COLUMNS


def test_describe_presets_threshold_units_clusterless():
    """The clusterless row's threshold units are µV, never the MAD ``σ``."""
    df = describe_presets()
    row = df.loc[
        df["preset"] == "franklab_tetrode_clusterless_thresholder"
    ].iloc[0]
    units = row["threshold_units"]
    assert "µV" in units
    assert "σ" not in units
    assert "sigma" not in units.lower()


def test_describe_presets_no_db(monkeypatch):
    """``describe_presets`` issues no database query (pure, DB-free).

    Any SQL would route through ``Connection.query``; patch it to raise so an
    accidental table read regresses loudly instead of silently hitting the DB.
    """

    def _boom(*args, **kwargs):
        raise AssertionError(
            "describe_presets() issued a database query; it must be DB-free"
        )

    monkeypatch.setattr(dj.Connection, "query", _boom)
    df = describe_presets()
    assert set(df["preset"]) == set(list_presets())


def test_describe_presets_matches_preset_objects():
    """Each row's static fields equal the backing ``_Preset`` attributes.

    Guards against a second hard-coded copy of preset data drifting from the
    ``_PRESETS`` source of truth.
    """
    df = describe_presets().set_index("preset")
    for name, preset in _PRESETS.items():
        row = df.loc[name]
        assert row["sorter"] == preset.sorter
        assert row["preprocessing_params_name"] == preset.preprocessing_params_name
        assert row["artifact_detection_params_name"] == preset.artifact_detection_params_name
        assert row["sorter_params_name"] == preset.sorter_params_name
        assert row["intended_use"] == preset.intended_use
        assert row["threshold_units"] == preset.threshold_units
        assert row["notes"] == preset.notes


def test_describe_presets_threshold_units_mountainsort():
    """Both MountainSort presets report a MAD-multiplier threshold, not µV.

    The complement of the clusterless footgun: MountainSort ``detect_threshold``
    is a noise-relative MAD multiplier, so mislabeling it with absolute
    microvolts would be the same class of error in the other direction.
    """
    df = describe_presets().set_index("preset")
    for name in (
        "franklab_tetrode_mountainsort4",
        "franklab_tetrode_mountainsort5",
    ):
        units = df.loc[name, "threshold_units"]
        assert "MAD" in units
        assert "µV" not in units
        assert "sigma" not in units.lower()


def test_describe_presets_builtins_populate_human_fields():
    """Every shipped preset populates the human-facing fields (no blanks).

    The ``intended_use`` / ``threshold_units`` / ``notes`` fields default to
    ``""`` so external presets need not supply them, but the built-ins must --
    a blank here means a user reading the catalog learns nothing about the
    preset.
    """
    df = describe_presets().set_index("preset")
    for name in list_presets():
        for col in ("intended_use", "threshold_units", "notes"):
            assert df.loc[name, col].strip(), f"{name}.{col} is blank"

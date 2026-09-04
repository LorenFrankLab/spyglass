"""DB-free review-import validation tests."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest


class _UnitRelation:
    def __and__(self, _restriction):
        return self

    def fetch(self, attribute):
        assert attribute == "unit_id"
        return [1, 2]


class _FakeCurationV2:
    Unit = _UnitRelation()

    @staticmethod
    def _labels_by_unit(_key):
        return {1: ["artifact"]}


def _review():
    return SimpleNamespace(
        parent=SimpleNamespace(
            as_key=lambda: {"sorting_id": "sort", "curation_id": 0}
        ),
        profile=SimpleNamespace(
            label_options=("accept", "mua", "noise"),
            label_import_mode="replace",
        ),
    )


def test_review_import_preserves_inherited_label_outside_palette(monkeypatch):
    """A valid parent label need not be offered as a new review choice."""
    from spyglass.spikesorting.v2.review_api import _normalize_review_edits

    fake_module = ModuleType("spyglass.spikesorting.v2.curation")
    fake_module.CurationV2 = _FakeCurationV2
    monkeypatch.setitem(
        sys.modules, "spyglass.spikesorting.v2.curation", fake_module
    )

    labels, groups, conflicts, count = _normalize_review_edits(
        _review(), {1: ["artifact"], 2: ["noise"]}, []
    )

    assert dict(labels) == {1: ("artifact",), 2: ("noise",)}
    assert groups == ()
    assert conflicts == ()
    assert count == 2


def test_review_import_rejects_new_label_outside_palette(monkeypatch):
    """An inherited label cannot be copied to a different unit."""
    from spyglass.spikesorting.v2.review_api import _normalize_review_edits

    fake_module = ModuleType("spyglass.spikesorting.v2.curation")
    fake_module.CurationV2 = _FakeCurationV2
    monkeypatch.setitem(
        sys.modules, "spyglass.spikesorting.v2.curation", fake_module
    )

    with pytest.raises(ValueError, match="new labels outside"):
        _normalize_review_edits(
            _review(), {1: ["artifact"], 2: ["artifact"]}, []
        )


def test_merge_resolution_may_keep_inherited_label_outside_palette():
    """A conflict can resolve to a label carried by either contributor."""
    from spyglass.spikesorting.v2.review_api import (
        MergeLabelConflict,
        _unknown_conflict_resolution_labels,
    )

    conflict = MergeLabelConflict(
        merged_unit_id=3,
        contributor_unit_ids=(1, 2),
        contributor_labels={1: ("artifact",), 2: ("noise",)},
    )
    palette = {"accept", "mua", "noise"}

    assert not _unknown_conflict_resolution_labels(
        ("artifact",), palette, conflict
    )
    assert _unknown_conflict_resolution_labels(
        ("new-site-label",), palette, conflict
    ) == ["new-site-label"]

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


def _change_set(**overrides):
    import uuid

    from spyglass.spikesorting.v2.review_api import (
        CurationChangeSet,
        MergeLabelConflict,
    )

    review = SimpleNamespace(
        review_id=uuid.UUID(int=7),
        parent=SimpleNamespace(curation_id=0, sorting_id=uuid.UUID(int=1)),
        profile=SimpleNamespace(review_profile_name="minimal_profile"),
    )
    fields = dict(
        review=review,
        annotations_hash="abc",
        labels_before={1: ("accept",), 2: ("noise",), 3: ()},
        labels_after={1: ("accept",), 2: (), 3: ("accept",)},
        merge_groups=((1, 3),),
        unit_count_before=3,
        unit_count_after=2,
        label_conflicts=(
            MergeLabelConflict(
                merged_unit_id=4,
                contributor_unit_ids=(1, 3),
                contributor_labels={1: ("accept",), 3: ("mua",)},
            ),
        ),
        newer_sibling_curations=(SimpleNamespace(curation_id=9),),
        reviewed_parent_created_at=None,
        reviewed_parent_created_by="tester",
    )
    fields.update(overrides)
    return CurationChangeSet(**fields)


def test_change_set_summary_and_changed_units_are_derived_from_fields():
    """The compact summary / table replace the nested dataclass dump: label
    additions/removals, proposed merges, counts, conflicts and siblings, all
    from the preview's own fields (no database)."""
    changes = _change_set()
    assert changes.has_changes
    table = changes.changed_units().set_index("unit_id")
    assert list(table.index) == [1, 2, 3]
    assert table.loc[2, "removed"] == "noise" and table.loc[2, "added"] == ""
    assert table.loc[3, "added"] == "accept"
    assert table.loc[1, "merge_group"] == "1,3"  # unchanged labels, merged
    assert table.loc[2, "merge_group"] == ""
    text = changes.summary()
    assert "units: 3 -> 2" in text
    assert "label changes: 2 unit(s)" in text
    assert "proposed merges: 1,3" in text
    assert "merged unit 4 <- 1:accept | 3:mua" in text
    assert "newer sibling curation(s)" in text and "9" in text
    assert "no changes" not in text

    unchanged = _change_set(
        labels_after={1: ("accept",), 2: ("noise",), 3: ()},
        merge_groups=(),
        unit_count_after=3,
        label_conflicts=(),
        newer_sibling_curations=(),
    )
    assert not unchanged.has_changes
    assert unchanged.changed_units().empty
    assert "confirm_no_changes=True" in unchanged.summary()

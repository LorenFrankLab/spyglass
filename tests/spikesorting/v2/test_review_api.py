"""DB-free review-import validation tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def _profile():
    return SimpleNamespace(
        label_options=("accept", "mua", "noise"),
        label_import_mode="replace",
    )


def test_review_import_preserves_inherited_label_outside_palette():
    """A valid parent label need not be offered as a new review choice."""
    from spyglass.spikesorting.v2.review_api import _normalize_review_edits

    labels, groups, conflicts, count = _normalize_review_edits(
        _profile(),
        {1: ["artifact"], 2: ["noise"]},
        [],
        unit_ids={1, 2},
        labels_before={1: ("artifact",)},
    )

    assert dict(labels) == {1: ("artifact",), 2: ("noise",)}
    assert groups == ()
    assert conflicts == ()
    assert count == 2


def test_review_import_rejects_new_label_outside_palette():
    """An inherited label cannot be copied to a different unit."""
    from spyglass.spikesorting.v2.review_api import _normalize_review_edits

    with pytest.raises(ValueError, match="new labels outside"):
        _normalize_review_edits(
            _profile(),
            {1: ["artifact"], 2: ["artifact"]},
            [],
            unit_ids={1, 2},
            labels_before={1: ("artifact",)},
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


def test_review_and_persistence_agree_on_multiple_merge_ids():
    """Reordered groups with gaps in source IDs keep conflicts and labels aligned."""
    from spyglass.spikesorting.v2._curation_transforms import (
        build_curated_unit_rows,
    )
    from spyglass.spikesorting.v2.review_api import (
        _child_labels,
        _normalize_review_edits,
    )

    unit_ids = (1, 4, 6, 9, 12)
    groups = [[9, 6], [4, 1]]
    labels, normalized_groups, conflicts, count = _normalize_review_edits(
        _profile(),
        {
            1: ["accept"],
            4: ["accept"],
            6: ["accept"],
            9: ["noise"],
            12: ["mua"],
        },
        groups,
        unit_ids=set(unit_ids),
        labels_before={1: ("artifact",)},
    )
    assert count == 3
    assert [(c.merged_unit_id, c.contributor_unit_ids) for c in conflicts] == [
        (14, (6, 9))
    ]
    changes = _change_set(
        labels_after=labels,
        merge_groups=normalized_groups,
        label_conflicts=conflicts,
    )
    assert _child_labels(
        changes, {14: ("noise",)}, parent_units=set(unit_ids)
    ) == {
        12: ["mua"],
        13: ["accept"],
        14: ["noise"],
    }

    source_rows = [
        {
            "unit_id": uid,
            "nwb_file_name": "test.nwb",
            "electrode_group_name": "0",
            "electrode_id": uid,
            "peak_amplitude_uv": 10.0,
            "n_spikes": 100,
        }
        for uid in unit_ids
    ]
    rows, contributors = build_curated_unit_rows(
        "sort", source_rows, groups, 1, True
    )
    assert contributors == {12: [12], 13: [4, 1], 14: [9, 6]}
    assert [row["unit_id"] for row in rows] == [12, 13, 14]
    # Within-group order breaks equal-amplitude ties, independently of the
    # canonical order used to allocate the groups' new IDs.
    assert [row["electrode_id"] for row in rows] == [12, 4, 9]


def _change_set(**overrides):
    import uuid

    from spyglass.spikesorting.v2.review_api import (
        CurationChangeSet,
        MergeLabelConflict,
    )

    review = SimpleNamespace(
        review_id=uuid.UUID(int=7),
        is_hosted=False,
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


def test_next_step_lines_name_the_state_consistently():
    """Saved-not-committed, nothing-to-commit, awaiting verification, and
    available-for-analysis each get one consistent line."""
    from spyglass.spikesorting.v2.review_api import ReviewImportReceipt

    edited = _change_set()
    line = edited.next_step()
    assert line.startswith(
        "Saved browser edits differ from the reviewed parent"
    )
    assert "Preview and commit" in line and "final labels" in line
    edited.review.is_hosted = True
    assert "commit()" in edited.next_step()
    assert "conflict_resolutions" in edited.next_step()
    clean = _change_set(
        labels_after={1: ("accept",), 2: ("noise",), 3: ()},
        merge_groups=(),
        unit_count_after=3,
        label_conflicts=(),
    )
    assert "Save draft" in clean.next_step()
    assert "Record reviewed — no changes" in clean.next_step()
    clean.review.is_hosted = True
    assert "Save Annotations" in clean.next_step()
    assert "confirm_no_changes=True" in clean.next_step()

    def receipt(needs):
        return ReviewImportReceipt(
            curation=SimpleNamespace(curation_id=5),
            evaluation=None,
            changes=edited,
            warnings=(),
            stages=(),
            needs_merge_verification=needs,
        )

    assert (
        receipt(True)
        .next_step()
        .startswith("Merged result (curation 5) awaiting verification")
    )
    assert (
        receipt(False)
        .next_step()
        .startswith("Result available for analysis: curation 5")
    )


def test_configured_lab_label_can_be_added_during_review():
    from spyglass.spikesorting.v2.review_api import _normalize_review_edits

    profile = _profile()
    profile.label_options = (*profile.label_options, "lab_cell")
    labels, _, conflicts, _ = _normalize_review_edits(
        profile,
        {1: ["lab_cell"]},
        [],
        unit_ids={1},
        labels_before={},
    )
    assert labels[1] == ("lab_cell",)
    assert conflicts == ()

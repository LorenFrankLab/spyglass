"""Parent-state child-curation composition for CurationV2.

A child's units, spike trains, labels and merge namespace come from the parent
curation, not ``Sorting.Unit``. Tested on a parent with an applied merge, whose
fresh ``max+1`` id is absent from ``Sorting.Unit``.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.spikesorting.v2._ingest_helpers import clear_curations_for


@pytest.fixture
def merged_parent(planted_three_unit_sort):
    """A committed applied-merge parent curation over three planted units.

    Merges raw units ``[0, 1]`` into fresh id ``3``, leaving parent units
    ``{2, 3}``. Yields a dict with the sort key, the parent curation key, the
    raw unit ids, the parent unit set, and the merged id. Clears curations
    before and after.
    """
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting

    sort_pk = planted_three_unit_sort
    clear_curations_for(sort_pk)

    raw_units = sorted(
        int(u) for u in (Sorting.Unit & sort_pk).fetch("unit_id")
    )
    assert raw_units == [0, 1, 2], raw_units
    merged_id = max(raw_units) + 1  # 3

    parent = CurationV2.insert_curation(
        sort_pk,
        merge_groups=[[0, 1]],
        description="merged parent (raw [0,1] -> fresh 3)",
        apply_merge=True,
    )
    parent_units = sorted(
        int(u) for u in (CurationV2.Unit & parent).fetch("unit_id")
    )
    assert parent_units == [2, merged_id], parent_units

    yield {
        "sort_pk": sort_pk,
        "parent": parent,
        "raw_units": raw_units,
        "parent_units": parent_units,
        "merged_id": merged_id,
    }
    clear_curations_for(sort_pk)


@pytest.mark.slow
@pytest.mark.integration
def test_child_curation_composes_from_merged_parent(merged_parent):
    """A label-only child of a merged parent keeps the parent's fresh merged
    unit id and does NOT resurrect the absorbed raw contributors.
    """
    from spyglass.spikesorting.v2.curation import CurationV2

    sort_pk = merged_parent["sort_pk"]
    parent = merged_parent["parent"]
    merged_id = merged_parent["merged_id"]  # 3

    child = CurationV2.insert_curation(
        sort_pk,
        labels={merged_id: ["mua"]},
        parent_curation_id=parent["curation_id"],
        apply_merge=False,
        description="label edit on merged parent",
    )

    child_units = set(
        int(u) for u in (CurationV2.Unit & child).fetch("unit_id")
    )
    # Parent namespace {2, 3}, NOT the raw {0, 1, 2}: the absorbed raw
    # contributors 0 and 1 are not resurrected as separate child units.
    assert child_units == {2, merged_id}, child_units
    assert 0 not in child_units and 1 not in child_units

    # The label landed on the fresh merged id (a valid parent-namespace input).
    child_labels = {
        (int(r["unit_id"]), r["curation_label"])
        for r in (CurationV2.UnitLabel & child).fetch(as_dict=True)
    }
    assert (merged_id, "mua") in child_labels, child_labels

    # A label-only child of a merged parent is committed, not a preview, even
    # though its raw MergeGroup carries the parent's merge provenance.
    assert CurationV2.is_committed_curation(child)
    assert not CurationV2.has_unapplied_proposed_merges(child)


@pytest.mark.slow
@pytest.mark.integration
def test_child_merge_groups_are_parent_namespace(merged_parent):
    """A child merge can reference a merged-parent unit id; the child writes
    the correct merged spike train and expands raw provenance through the
    parent's contributors.
    """
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting

    sort_pk = merged_parent["sort_pk"]
    parent = merged_parent["parent"]
    merged_id = merged_parent["merged_id"]  # 3

    # Merge parent units {2, 3}; 3 is absent from Sorting.Unit.
    child_merged_id = max(merged_parent["parent_units"]) + 1  # 4
    child = CurationV2.insert_curation(
        sort_pk,
        merge_groups=[[2, merged_id]],
        parent_curation_id=parent["curation_id"],
        description="child merge in parent namespace",
        apply_merge=True,
    )

    child_units = set(
        int(u) for u in (CurationV2.Unit & child).fetch("unit_id")
    )
    assert child_units == {child_merged_id}, child_units

    # Child unit 4 expands through the parent's MergeGroup to raw units:
    # raw(2)={2} U raw(3)={0,1}.
    raw_contribs = {
        int(r["contributor_unit_id"])
        for r in (
            CurationV2.MergeGroup & child & {"unit_id": child_merged_id}
        ).fetch(as_dict=True)
    }
    assert raw_contribs == {0, 1, 2}, raw_contribs

    # Spike conservation: the child's merged train is the union of the parent
    # units 2 and 3's trains (the planted units are non-coincident, so the
    # 0.4 ms dedup removes nothing). Parent unit 3 is itself raw(0) U raw(1),
    # so the child merged unit holds raw 0+1+2 spikes.
    raw_sorting = Sorting().get_sorting(sort_pk)
    expected_n = sum(
        len(raw_sorting.get_unit_spike_train(unit_id=u)) for u in (0, 1, 2)
    )
    child_sorting = CurationV2().get_sorting(child)
    merged_train = np.asarray(
        child_sorting.get_unit_spike_train(unit_id=child_merged_id)
    )
    assert len(merged_train) == expected_n, (len(merged_train), expected_n)

    # n_spikes invariant survives the parent-composition write.
    n_spikes = (CurationV2.Unit & child & {"unit_id": child_merged_id}).fetch1(
        "n_spikes"
    )
    assert n_spikes == expected_n


@pytest.mark.slow
@pytest.mark.integration
def test_parent_operation_provenance_records_parent_units(merged_parent):
    """ParentMergeGroup records parent unit ids (including a merged id absent
    from Sorting.Unit), separately from raw-contributor provenance.
    """
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting

    sort_pk = merged_parent["sort_pk"]
    parent = merged_parent["parent"]
    merged_id = merged_parent["merged_id"]  # 3
    child_merged_id = max(merged_parent["parent_units"]) + 1  # 4

    child = CurationV2.insert_curation(
        sort_pk,
        merge_groups=[[2, merged_id]],
        parent_curation_id=parent["curation_id"],
        apply_merge=True,
    )

    parent_op = {
        int(r["parent_unit_id"])
        for r in (
            CurationV2.ParentMergeGroup & child & {"unit_id": child_merged_id}
        ).fetch(as_dict=True)
    }
    assert parent_op == {2, merged_id}, parent_op

    # Parent id 3 is absent from Sorting.Unit, so it is a validated int, not
    # an FK.
    raw_ids = set(int(u) for u in (Sorting.Unit & sort_pk).fetch("unit_id"))
    assert merged_id not in raw_ids


@pytest.mark.slow
@pytest.mark.integration
def test_parent_merge_group_insert_rejects_invalid_provenance(merged_parent):
    """A direct ParentMergeGroup insert cannot forge invalid provenance.

    ``parent_unit_id`` is not an FK, so ``insert`` enforces:

    - a root curation gets no rows;
    - a child row's ``parent_unit_id`` is a unit of the immediate parent.
    """
    from spyglass.spikesorting.v2.curation import CurationV2

    sort_pk = merged_parent["sort_pk"]
    parent = merged_parent["parent"]  # a merged ROOT (no ParentMergeGroup rows)
    merged_id = merged_parent["merged_id"]  # 3
    parent_units = merged_parent["parent_units"]  # [2, 3]

    # A ROOT curation must carry no ParentMergeGroup rows (their presence is the
    # child discriminator), so a direct insert onto one is rejected.
    assert not (CurationV2.ParentMergeGroup & parent)
    with pytest.raises(ValueError, match="ROOT"):
        CurationV2.ParentMergeGroup.insert1(
            {**parent, "unit_id": merged_id, "parent_unit_id": merged_id}
        )

    # A child of the merged parent may reference only parent units {2, 3};
    # raw unit 0 was absorbed into the merge.
    child = CurationV2.insert_curation(
        sort_pk,
        labels={merged_id: ["mua"]},
        parent_curation_id=parent["curation_id"],
        apply_merge=False,
        description="child for provenance guard",
    )
    assert 0 not in parent_units
    with pytest.raises(ValueError, match="not units in the parent curation"):
        CurationV2.ParentMergeGroup.insert1(
            {**child, "unit_id": merged_id, "parent_unit_id": 0}
        )


@pytest.mark.slow
@pytest.mark.integration
def test_preview_child_of_merged_parent_lazy_merges_in_parent_namespace(
    merged_parent,
):
    """A preview child of a merged parent reads and applies its proposed merge
    in the parent namespace (ParentMergeGroup), not the raw MergeGroup.

    Merging the parent's merged unit is a valid draft, reported as a preview,
    and the lazy merge collapses the parent units over the child's trains.
    """
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting

    sort_pk = merged_parent["sort_pk"]
    parent = merged_parent["parent"]
    merged_id = merged_parent["merged_id"]  # 3
    child_merged_id = max(merged_parent["parent_units"]) + 1  # 4

    preview = CurationV2.insert_curation(
        sort_pk,
        merge_groups=[[2, merged_id]],
        parent_curation_id=parent["curation_id"],
        description="preview further-merge of a merged parent",
        apply_merge=False,
    )

    # Preview keeps every parent unit; [2, 3] reads as a proposed merge in the
    # parent namespace, not the raw MergeGroup (where 3 -> raw [0, 1]).
    assert set(
        int(u) for u in (CurationV2.Unit & preview).fetch("unit_id")
    ) == {2, merged_id}
    own_groups = CurationV2.get_unit_contributor_groups(preview)
    assert sorted(own_groups[2]) == [2, merged_id]
    assert CurationV2.has_unapplied_proposed_merges(preview) is True
    assert CurationV2.is_committed_curation(preview) is False

    # Lazy merge applies [2, 3] over the child's stored parent-namespace trains,
    # assigning max(parent units)+1; spike count is conserved (raw 0+1+2).
    raw_sorting = Sorting().get_sorting(sort_pk)
    expected_n = sum(
        len(raw_sorting.get_unit_spike_train(unit_id=u)) for u in (0, 1, 2)
    )
    merged = CurationV2().get_merged_sorting(preview)
    assert set(int(u) for u in merged.get_unit_ids()) == {child_merged_id}
    assert len(merged.get_unit_spike_train(unit_id=child_merged_id)) == (
        expected_n
    )
    # Raw provenance for the preview's proposed-merge leader still expands to
    # the original raw units, queryable on MergeGroup.
    raw_contribs = {
        int(r["contributor_unit_id"])
        for r in (CurationV2.MergeGroup & preview & {"unit_id": 2}).fetch(
            as_dict=True
        )
    }
    assert raw_contribs == {0, 1, 2}, raw_contribs


@pytest.mark.slow
@pytest.mark.integration
def test_grandchild_composition_chains_raw_and_parent_provenance(
    planted_three_unit_sort,
):
    """A child-of-a-child keeps raw provenance chained to Sorting.Unit and
    parent-op provenance in its immediate parent's namespace.
    """
    from spyglass.spikesorting.v2.curation import CurationV2

    sort_pk = planted_three_unit_sort
    clear_curations_for(sort_pk)
    try:
        root = CurationV2.insert_curation(sorting_key=sort_pk)
        # Child A merges raw [0,1] -> 3; units {2, 3}.
        child_a = CurationV2.insert_curation(
            sort_pk,
            merge_groups=[[0, 1]],
            parent_curation_id=root["curation_id"],
            apply_merge=True,
        )
        # Grandchild B merges A's units [2, 3] -> 4; units {4}. Unit 3 is a
        # fresh A-namespace id absent from Sorting.Unit.
        grandchild_b = CurationV2.insert_curation(
            sort_pk,
            merge_groups=[[2, 3]],
            parent_curation_id=child_a["curation_id"],
            apply_merge=True,
        )
        assert set(
            int(u) for u in (CurationV2.Unit & grandchild_b).fetch("unit_id")
        ) == {4}

        # Raw provenance chains all the way back to the original raw units.
        raw_contribs = {
            int(r["contributor_unit_id"])
            for r in (
                CurationV2.MergeGroup & grandchild_b & {"unit_id": 4}
            ).fetch(as_dict=True)
        }
        assert raw_contribs == {0, 1, 2}, raw_contribs

        # Parent-op provenance is in child A's namespace (the immediate parent).
        parent_op = {
            int(r["parent_unit_id"])
            for r in (
                CurationV2.ParentMergeGroup & grandchild_b & {"unit_id": 4}
            ).fetch(as_dict=True)
        }
        assert parent_op == {2, 3}, parent_op
    finally:
        clear_curations_for(sort_pk)


@pytest.mark.slow
@pytest.mark.integration
def test_child_labels_inherit_and_merge(planted_three_unit_sort):
    """Child curations inherit parent labels by default; a merged child unit
    receives the UNION of contributor labels unless explicitly overridden;
    ``label_policy="replace"`` clears inherited labels.
    """
    from spyglass.spikesorting.v2.curation import CurationV2

    sort_pk = planted_three_unit_sort
    clear_curations_for(sort_pk)
    try:
        # A committed, LABELED parent: merge raw [0,1]->3, label unit 2 "mua"
        # and the merged unit 3 "noise".
        labeled_parent = CurationV2.insert_curation(
            sort_pk,
            merge_groups=[[0, 1]],
            labels={2: ["mua"], 3: ["noise"]},
            description="labeled merged parent",
            apply_merge=True,
        )

        def labels_of(key):
            out: dict[int, set] = {}
            for r in (CurationV2.UnitLabel & key).fetch(as_dict=True):
                out.setdefault(int(r["unit_id"]), set()).add(
                    r["curation_label"]
                )
            return out

        assert labels_of(labeled_parent) == {2: {"mua"}, 3: {"noise"}}

        # (1) Default policy = inherit: a plain child carries the parent labels.
        inherit_child = CurationV2.insert_curation(
            sort_pk,
            parent_curation_id=labeled_parent["curation_id"],
            apply_merge=False,
            description="inherit",
        )
        assert labels_of(inherit_child) == {2: {"mua"}, 3: {"noise"}}

        # (2) A committed merge of the labeled parent units inherits the UNION
        # of contributor labels on the merged unit.
        merge_child = CurationV2.insert_curation(
            sort_pk,
            merge_groups=[[2, 3]],
            parent_curation_id=labeled_parent["curation_id"],
            description="merge inherits union",
            apply_merge=True,
        )
        child_merged_id = 4
        assert labels_of(merge_child) == {child_merged_id: {"mua", "noise"}}

        # (3) An explicit override on the merged unit replaces the inherited
        # union for that unit only.
        override_child = CurationV2.insert_curation(
            sort_pk,
            merge_groups=[[2, 3]],
            labels={4: ["accept"]},
            parent_curation_id=labeled_parent["curation_id"],
            description="merge override",
            apply_merge=True,
        )
        assert labels_of(override_child) == {child_merged_id: {"accept"}}

        # (4) label_policy="replace": the supplied labels are the FULL child
        # state; inherited parent labels are dropped.
        replace_child = CurationV2.insert_curation(
            sort_pk,
            labels={2: ["accept"]},
            parent_curation_id=labeled_parent["curation_id"],
            apply_merge=False,
            label_policy="replace",
            description="replace",
        )
        assert labels_of(replace_child) == {2: {"accept"}}
    finally:
        clear_curations_for(sort_pk)


@pytest.mark.slow
@pytest.mark.integration
def test_insert_curation_rejects_preview_parent(planted_two_unit_sort):
    """A child cannot branch from a preview parent.

    A preview is a draft; a child would make it look committed and inherit the
    unapplied merge as provenance. Committed parents still accept children.
    """
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting

    sorting_key = dict(planted_two_unit_sort)
    unit_ids = sorted(
        int(u) for u in (Sorting.Unit & sorting_key).fetch("unit_id")
    )
    clear_curations_for(planted_two_unit_sort)
    try:
        root = CurationV2.insert_curation(sorting_key=sorting_key)
        preview = CurationV2.insert_curation(
            sorting_key,
            merge_groups=[[unit_ids[0], unit_ids[1]]],
            parent_curation_id=root["curation_id"],
            apply_merge=False,
        )
        assert not CurationV2.is_committed_curation(preview)
        with pytest.raises(ValueError, match="preview/draft"):
            CurationV2.insert_curation(
                sorting_key=sorting_key,
                parent_curation_id=preview["curation_id"],
                labels={unit_ids[0]: ["mua"]},
            )
        # Sanity: a committed parent still accepts a child (guard is
        # preview-only, not "any child").
        child = CurationV2.insert_curation(
            sorting_key=sorting_key,
            parent_curation_id=root["curation_id"],
            labels={unit_ids[0]: ["mua"]},
        )
        assert CurationV2.is_committed_curation(child)
    finally:
        clear_curations_for(planted_two_unit_sort)


@pytest.mark.slow
@pytest.mark.integration
def test_insert_curation_retries_on_curation_id_collision(
    planted_three_unit_sort, monkeypatch
):
    """A concurrent insert claiming the allocated curation_id is recovered.

    The ``max+1`` id is allocated outside the insert transaction. Forcing
    ``_next_curation_id`` to return a taken id first, ``insert_curation`` must
    retry with a fresh id.
    """
    from spyglass.spikesorting.v2.curation import CurationV2

    sort_pk = planted_three_unit_sort
    clear_curations_for(sort_pk)
    try:
        root = CurationV2.insert_curation(sort_pk)  # curation_id 0
        child_a = CurationV2.insert_curation(
            sort_pk,
            parent_curation_id=root["curation_id"],
            labels={0: ["noise"]},
        )  # curation_id 1
        assert child_a["curation_id"] == 1

        # First allocation returns the already-taken id 1 (collision), every
        # later call returns the real next id (2).
        calls = {"n": 0}

        def _fake_next(cls, sorting_id):
            calls["n"] += 1
            return 1 if calls["n"] == 1 else 2

        monkeypatch.setattr(
            CurationV2, "_next_curation_id", classmethod(_fake_next)
        )

        child_b = CurationV2.insert_curation(
            sort_pk,
            parent_curation_id=root["curation_id"],
            labels={2: ["mua"]},  # distinct content -> not a reuse of child_a
        )
        # Retried past the collision to the fresh id, no crash.
        assert child_b["curation_id"] == 2
        assert calls["n"] >= 2  # initial allocation + at least one retry
        assert CurationV2 & {**sort_pk, "curation_id": 2}
    finally:
        clear_curations_for(sort_pk)


@pytest.mark.slow
@pytest.mark.integration
def test_insert_curation_retry_reuses_concurrent_winner(
    planted_three_unit_sort, monkeypatch
):
    """Under a curation_id collision with the same logical child, the retry
    returns that child (reuse_existing) instead of inserting a duplicate.
    """
    from spyglass.spikesorting.v2._curation import insert as _curation_insert
    from spyglass.spikesorting.v2.curation import CurationV2

    sort_pk = planted_three_unit_sort
    clear_curations_for(sort_pk)
    try:
        root = CurationV2.insert_curation(sort_pk)  # curation_id 0
        # The "concurrent winner": a child with specific content at id 1.
        winner = CurationV2.insert_curation(
            sort_pk,
            parent_curation_id=root["curation_id"],
            labels={0: ["noise"]},
        )
        assert winner["curation_id"] == 1

        # Pre-loop reuse check MISSES (as if the winner is not yet visible);
        # the retry's re-check finds it via the real implementation.
        real_find = _curation_insert.find_matching_child_curation
        find_calls = {"n": 0}

        def fake_find(table_cls, **kwargs):
            find_calls["n"] += 1
            if find_calls["n"] == 1:
                return None
            return real_find(table_cls, **kwargs)

        monkeypatch.setattr(
            _curation_insert, "find_matching_child_curation", fake_find
        )
        # Force the id allocation to collide with the winner (1).
        id_calls = {"n": 0}

        def fake_next_id(cls, sorting_id):
            id_calls["n"] += 1
            return 1 if id_calls["n"] == 1 else 2

        monkeypatch.setattr(
            CurationV2, "_next_curation_id", classmethod(fake_next_id)
        )

        # Insert the SAME logical child again with reuse on.
        result = CurationV2.insert_curation(
            sort_pk,
            parent_curation_id=root["curation_id"],
            labels={0: ["noise"]},
            reuse_existing=True,
        )
        # Returned the winner; did NOT stage a duplicate child at id 2.
        assert result["curation_id"] == winner["curation_id"] == 1
        assert not (CurationV2 & {**sort_pk, "curation_id": 2})
    finally:
        clear_curations_for(sort_pk)


@pytest.mark.slow
@pytest.mark.integration
def test_curation_delete_refuses_to_orphan_descendant(planted_three_unit_sort):
    """``CurationV2.delete`` refuses to remove a parent while a descendant
    references it via ``parent_curation_id`` (not a self-FK), naming the
    (parent, child) pair. Leaves and parents without descendants are allowed.
    """
    from spyglass.spikesorting.v2.curation import CurationV2

    sort_pk = planted_three_unit_sort
    clear_curations_for(sort_pk)
    try:
        root = CurationV2.insert_curation(sorting_key=sort_pk)
        child = CurationV2.insert_curation(
            sorting_key=sort_pk,
            parent_curation_id=root["curation_id"],
            description="lineage-delete guard test",
        )
        # Deleting the parent ALONE while the child references it is REFUSED,
        # with a message naming the offending (parent, child) pair; both survive.
        with pytest.raises(ValueError, match="descendant curations") as excinfo:
            (CurationV2 & root).delete(safemode=False)
        assert f"({root['curation_id']}, {child['curation_id']})" in str(
            excinfo.value
        )
        assert CurationV2 & root
        assert CurationV2 & child

        # A leaf child is not refused. Removal itself goes through the
        # merge-aware clear_curations_for.
        (CurationV2 & child).delete(safemode=False)
        clear_curations_for(child)
        assert not (CurationV2 & child)

        # With the descendant gone, the parent is not refused.
        (CurationV2 & root).delete(safemode=False)
    finally:
        clear_curations_for(sort_pk)


@pytest.mark.slow
@pytest.mark.integration
def test_audit_orphaned_lineage_flags_bypass_orphan(planted_three_unit_sort):
    """audit_orphaned_lineage is empty for intact lineage and flags a child
    whose parent was removed by super_delete.
    """
    from spyglass.spikesorting.v2.curation import CurationV2

    def _flags(audit, curation_key):
        return any(
            o["sorting_id"] == curation_key["sorting_id"]
            and o["curation_id"] == curation_key["curation_id"]
            for o in audit
        )

    sort_pk = planted_three_unit_sort
    clear_curations_for(sort_pk)
    try:
        root = CurationV2.insert_curation(sorting_key=sort_pk)
        child = CurationV2.insert_curation(
            sorting_key=sort_pk,
            parent_curation_id=root["curation_id"],
            description="orphan-audit test",
        )
        # Intact lineage: the child's parent exists, so it is not flagged.
        assert not _flags(CurationV2.audit_orphaned_lineage(), child)

        # Administrative bypass: super_delete the parent (merge-aware), leaving
        # the child pointing at a now-missing parent.
        clear_curations_for(root)
        assert not (CurationV2 & root)
        audit = CurationV2.audit_orphaned_lineage()
        assert _flags(audit, child)
        entry = next(
            o for o in audit if o["curation_id"] == child["curation_id"]
        )
        assert entry["parent_curation_id"] == root["curation_id"]
    finally:
        clear_curations_for(sort_pk)

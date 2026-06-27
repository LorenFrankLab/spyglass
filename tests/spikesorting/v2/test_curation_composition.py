"""Parent-state child-curation composition for CurationV2.

A child curation edits the PARENT curation's committed state: its unit rows,
spike trains, labels, and merge namespace all come from the parent
``CurationV2`` row rather than the raw ``Sorting.Unit`` rows. These tests pin
that contract against a parent that already applied a merge (so its unit set
contains a fresh ``max+1`` merged id that is NOT in ``Sorting.Unit``) -- the
case where the old raw-sourced path silently resurrected the absorbed
contributors.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.spikesorting.v2._ingest_helpers import clear_curations_for


@pytest.fixture
def merged_parent(planted_three_unit_sort):
    """A committed applied-merge parent curation over three planted units.

    Merges raw units ``[0, 1]`` into a fresh id ``3`` (``max(0,1,2)+1``),
    leaving the committed parent unit set ``{2, 3}`` -- unit ``3`` is the
    merged id and is absent from ``Sorting.Unit``. Yields a dict with the
    sort key, the merged-parent curation key, the raw unit ids, the parent
    unit set, and the fresh merged id. Clears curations around itself so the
    persistent test DB does not leak rows across runs.
    """
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting

    sort_pk = planted_three_unit_sort
    clear_curations_for(sort_pk)

    raw_units = sorted(int(u) for u in (Sorting.Unit & sort_pk).fetch("unit_id"))
    assert raw_units == [0, 1, 2], raw_units
    merged_id = max(raw_units) + 1  # 3

    parent = CurationV2.create_merged_curation(
        sort_pk,
        merge_groups=[[0, 1]],
        description="merged parent (raw [0,1] -> fresh 3)",
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

    child_units = set(int(u) for u in (CurationV2.Unit & child).fetch("unit_id"))
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

    # A label-only child of a merged parent is COMMITTED, not a preview -- even
    # though its raw MergeGroup carries multi-contributor provenance inherited
    # from the parent's merge (the own-namespace view must not misread that).
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

    # Merge the two PARENT units {2, 3} -- where 3 is the fresh merged id from
    # the parent, absent from Sorting.Unit. The new merged id is
    # max(parent units) + 1 = 4.
    child_merged_id = max(merged_parent["parent_units"]) + 1  # 4
    child = CurationV2.create_merged_curation(
        sort_pk,
        merge_groups=[[2, merged_id]],
        parent_curation_id=parent["curation_id"],
        description="child merge in parent namespace",
    )

    child_units = set(int(u) for u in (CurationV2.Unit & child).fetch("unit_id"))
    assert child_units == {child_merged_id}, child_units

    # Raw-contributor provenance: child unit 4 expands through the parent's
    # MergeGroup to the ORIGINAL raw units -- raw(2)={2} U raw(3)={0,1}.
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
    n_spikes = (
        CurationV2.Unit & child & {"unit_id": child_merged_id}
    ).fetch1("n_spikes")
    assert n_spikes == expected_n


@pytest.mark.slow
@pytest.mark.integration
def test_parent_operation_provenance_records_parent_units(merged_parent):
    """The child merge operation provenance (ParentMergeGroup) records PARENT
    unit ids -- including the fresh merged parent id that is not in
    Sorting.Unit -- stored separately from the raw-contributor provenance.
    """
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting

    sort_pk = merged_parent["sort_pk"]
    parent = merged_parent["parent"]
    merged_id = merged_parent["merged_id"]  # 3
    child_merged_id = max(merged_parent["parent_units"]) + 1  # 4

    child = CurationV2.create_merged_curation(
        sort_pk,
        merge_groups=[[2, merged_id]],
        parent_curation_id=parent["curation_id"],
    )

    parent_op = {
        int(r["parent_unit_id"])
        for r in (
            CurationV2.ParentMergeGroup & child & {"unit_id": child_merged_id}
        ).fetch(as_dict=True)
    }
    assert parent_op == {2, merged_id}, parent_op

    # The fresh merged parent id 3 is genuinely absent from Sorting.Unit, so it
    # could not be FK'd there -- the part stores it as a validated int, not an
    # FK to the raw sort.
    raw_ids = set(int(u) for u in (Sorting.Unit & sort_pk).fetch("unit_id"))
    assert merged_id not in raw_ids


@pytest.mark.slow
@pytest.mark.integration
def test_parent_merge_group_insert_rejects_invalid_provenance(merged_parent):
    """A direct ParentMergeGroup insert cannot forge invalid provenance.

    ``parent_unit_id`` is intentionally not a DataJoint FK (a merged parent id
    is absent from ``Sorting.Unit``), so ``insert`` enforces the two invariants
    ``get_merge_groups`` relies on: a ROOT curation gets no rows, and a child
    row's ``parent_unit_id`` must be a unit in the immediate parent. Without the
    guard a forged row could misclassify a committed curation as a preview or
    break merged-sorting reconstruction.
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

    # A child of the merged parent: its parent-namespace rows must reference
    # parent units {2, 3} only. Raw unit 0 was absorbed into the merge and is
    # NOT in the parent set -- exactly the resurrection the design forbids.
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
    """A PREVIEW child of a merged parent reads/applies its proposed merge in
    the PARENT namespace (via ParentMergeGroup), not the raw MergeGroup.

    This pins the own-namespace path that `get_merge_groups` /
    `has_unapplied_proposed_merges` / `get_merged_sorting` take for a child:
    proposing to further-merge the parent's merged unit id is a valid draft, is
    reported as a preview (not silently committed), and the lazy merge collapses
    the parent units over the child's stored trains.
    """
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting

    sort_pk = merged_parent["sort_pk"]
    parent = merged_parent["parent"]
    merged_id = merged_parent["merged_id"]  # 3
    child_merged_id = max(merged_parent["parent_units"]) + 1  # 4

    preview = CurationV2.propose_merge_curation(
        sort_pk,
        merge_groups=[[2, merged_id]],
        parent_curation_id=parent["curation_id"],
        description="preview further-merge of a merged parent",
    )

    # Preview keeps every parent unit; the own-namespace merge view is the
    # PARENT namespace (ParentMergeGroup), so [2, 3] reads as a real proposed
    # merge -- NOT misread from the raw MergeGroup (where unit 3 -> raw [0,1]).
    assert set(
        int(u) for u in (CurationV2.Unit & preview).fetch("unit_id")
    ) == {2, merged_id}
    own_groups = CurationV2.get_merge_groups(preview)
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
        child_a = CurationV2.create_merged_curation(
            sort_pk, merge_groups=[[0, 1]], parent_curation_id=root["curation_id"]
        )
        # Grandchild B merges A's units [2, 3] -> 4; units {4}. Unit 3 is a
        # fresh A-namespace id absent from Sorting.Unit.
        grandchild_b = CurationV2.create_merged_curation(
            sort_pk,
            merge_groups=[[2, 3]],
            parent_curation_id=child_a["curation_id"],
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
        labeled_parent = CurationV2.create_merged_curation(
            sort_pk,
            merge_groups=[[0, 1]],
            labels={2: ["mua"], 3: ["noise"]},
            description="labeled merged parent",
        )

        def labels_of(key):
            out: dict[int, set] = {}
            for r in (CurationV2.UnitLabel & key).fetch(as_dict=True):
                out.setdefault(int(r["unit_id"]), set()).add(r["curation_label"])
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
        merge_child = CurationV2.create_merged_curation(
            sort_pk,
            merge_groups=[[2, 3]],
            parent_curation_id=labeled_parent["curation_id"],
            description="merge inherits union",
        )
        child_merged_id = 4
        assert labels_of(merge_child) == {child_merged_id: {"mua", "noise"}}

        # (3) An explicit override on the merged unit replaces the inherited
        # union for that unit only.
        override_child = CurationV2.create_merged_curation(
            sort_pk,
            merge_groups=[[2, 3]],
            labels={4: ["accept"]},
            parent_curation_id=labeled_parent["curation_id"],
            description="merge override",
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

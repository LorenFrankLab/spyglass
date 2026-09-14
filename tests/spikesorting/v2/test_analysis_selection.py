"""``select_units_for_analysis``: the explicit curation -> analysis handoff.

Pure policy semantics (DB-free) plus a database round trip on the planted
three-unit sort: the receipt's verdicts must equal what the downstream
``SortedSpikesGroup.fetch_spike_data`` actually returns, rejected / artifact /
noise units can never enter the canonical filtered handoff, and MUA /
unlabeled handling is explicit per policy.
"""

from __future__ import annotations

import pytest


def test_v2_policies_deny_unusable_labels_and_make_mua_explicit():
    """Policy content is exactly as documented; production rows untouched."""
    from spyglass.spikesorting.v2.analysis_selection import (
        DEFAULT_UNIT_SELECTION_POLICY,
        V2_UNIT_SELECTION_POLICIES,
        apply_unit_selection_policy,
    )

    single = V2_UNIT_SELECTION_POLICIES["v2_accepted_single_units"]
    neural = V2_UNIT_SELECTION_POLICIES["v2_accepted_neural_units"]
    assert DEFAULT_UNIT_SELECTION_POLICY == "v2_accepted_single_units"
    for policy in (single, neural):
        assert {"noise", "reject", "artifact"} <= set(policy["exclude_labels"])
    assert "mua" in single["exclude_labels"]
    assert "mua" in neural["include_labels"]

    labels = {
        1: ["accept"],
        2: ["mua"],
        3: ["noise"],
        4: ["reject"],
        5: ["artifact"],
        6: [],  # unlabeled
        7: ["accept", "reject"],  # a denied label always wins
        8: ["accept", "mua"],
    }
    included, excluded = apply_unit_selection_policy(
        labels, range(1, 9), single
    )
    assert included == (1,)
    assert set(excluded) == {2, 3, 4, 5, 6, 7, 8}
    assert "denied label(s): mua" == excluded[2]
    assert "denied label(s): mua" == excluded[8]  # accept+mua: denial wins
    assert "unlabeled" in excluded[6]
    assert "denied label(s): reject" == excluded[7]

    included, excluded = apply_unit_selection_policy(
        labels, range(1, 9), neural
    )
    assert included == (1, 2, 8)
    assert set(excluded) == {3, 4, 5, 6, 7}

    # The explicit expert choice keeps everything, unlabeled included.
    all_units = {"include_labels": [], "exclude_labels": []}
    included, excluded = apply_unit_selection_policy(
        labels, range(1, 9), all_units
    )
    assert included == tuple(range(1, 9)) and not excluded


@pytest.mark.slow
@pytest.mark.integration
def test_select_units_for_analysis_round_trip(
    planted_three_unit_sort, monkeypatch
):
    """Receipt verdicts == downstream fetch; rejected units cannot enter."""
    import numpy as np

    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    from spyglass.spikesorting.analysis.v1 import group as group_module
    from spyglass.spikesorting.analysis.v1.group import (
        SortedSpikesGroup,
        UnitSelectionParams,
    )
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.analysis_selection import (
        select_units_for_analysis,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.curation_api import CurationRef
    from spyglass.spikesorting.v2.sorting import Sorting

    sorting_key = dict(planted_three_unit_sort)
    unit_ids = sorted(
        int(u) for u in (Sorting.Unit & sorting_key).fetch("unit_id")
    )
    assert len(unit_ids) == 3
    clear_curations_for(sorting_key)
    root = CurationV2.insert_curation(sorting_key)
    labeled = CurationV2.insert_curation(
        sorting_key,
        labels={
            unit_ids[0]: ["accept"],
            unit_ids[1]: ["reject"],
            # unit_ids[2] stays unlabeled
        },
        parent_curation_id=root["curation_id"],
    )
    ref = CurationRef.from_key(labeled)
    # The downstream label filter is disabled under pytest for the shared
    # legacy fixtures; the handoff contract is exactly that filter, so enable
    # it for this test.
    monkeypatch.setattr(group_module, "test_mode", False)
    created = []
    try:
        receipt = select_units_for_analysis(ref)
        created.extend(g.group_key for g in receipt.groups)
        assert receipt.curation == ref
        assert receipt.policy_name == "v2_accepted_single_units"
        assert receipt.included_unit_ids == (unit_ids[0],)
        assert set(receipt.excluded_units) == {unit_ids[1], unit_ids[2]}
        assert "reject" in receipt.excluded_units[unit_ids[1]]
        assert receipt.unlabeled_unit_ids == (unit_ids[2],)
        assert len(receipt.groups) == 1
        assert receipt.groups[0].status == "created"
        assert receipt.groups[0].merge_id == ref.merge_id
        # The policy row exists with the documented content.
        row = (
            UnitSelectionParams
            & {"unit_filter_params_name": "v2_accepted_single_units"}
        ).fetch1()
        assert list(row["include_labels"]) == ["accept"]
        assert set(row["exclude_labels"]) == {
            "mua",
            "noise",
            "reject",
            "artifact",
        }

        # What analysis reads is exactly the receipt's included set -- the
        # rejected and unlabeled units never enter.
        spikes, ids = receipt.fetch_spike_data(return_unit_ids=True)
        assert [d["unit_id"] for d in ids] == list(receipt.included_unit_ids)
        assert all(d["spikesorting_merge_id"] == ref.merge_id for d in ids)
        expected = SpikeSortingOutput().get_spike_times(
            {"merge_id": ref.merge_id}
        )
        # The raw merge accessor returns EVERY unit; the handoff does not.
        assert len(expected) == 3 and len(spikes) == 1
        np.testing.assert_array_equal(spikes[0], expected[0])
        frame = receipt.describe()
        assert bool(frame.loc[unit_ids[0], "included"]) is True
        assert bool(frame.loc[unit_ids[1], "included"]) is False

        # Idempotent: the same call reuses the group.
        again = select_units_for_analysis(ref)
        assert again.groups[0].status == "reused"
        assert again.groups[0].group_key == receipt.groups[0].group_key

        # A wider policy admits MUA but still never a rejected unit.
        neural = select_units_for_analysis(
            ref, policy="v2_accepted_neural_units"
        )
        created.extend(g.group_key for g in neural.groups)
        assert neural.included_unit_ids == (unit_ids[0],)
        assert unit_ids[1] in neural.excluded_units

        # The expert all-units choice is explicit and reports the unlabeled unit.
        everything = select_units_for_analysis(ref, policy="all_units")
        created.extend(g.group_key for g in everything.groups)
        assert everything.included_unit_ids == tuple(unit_ids)
        assert everything.unlabeled_unit_ids == (unit_ids[2],)

        # A merge preview is not a result: refused.
        preview = CurationV2.insert_curation(
            sorting_key,
            merge_groups=[unit_ids[:2]],
            apply_merge=False,
            parent_curation_id=root["curation_id"],
        )
        with pytest.raises(ValueError, match="preview"):
            select_units_for_analysis(preview)
    finally:
        for group_key in created:
            (SortedSpikesGroup & dict(group_key)).super_delete(warn=False)
        clear_curations_for(sorting_key)

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

    # Auto-label-only handoff: nothing flagged is kept, MUA and unlabeled
    # included (explicitly).
    unflagged = V2_UNIT_SELECTION_POLICIES["v2_unflagged_units"]
    assert unflagged["include_labels"] == []
    included, excluded = apply_unit_selection_policy(
        labels, range(1, 9), unflagged
    )
    assert included == (1, 2, 6, 8)
    assert set(excluded) == {3, 4, 5, 7}

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

        # The auto-label-only policy keeps the unlabeled unit and says so.
        unflagged = select_units_for_analysis(ref, policy="v2_unflagged_units")
        created.extend(g.group_key for g in unflagged.groups)
        assert unflagged.included_unit_ids == (unit_ids[0], unit_ids[2])
        assert unflagged.included_unlabeled_unit_ids == (unit_ids[2],)
        assert unit_ids[1] in unflagged.excluded_units

        # The expert all-units choice is explicit and reports the unlabeled unit.
        everything = select_units_for_analysis(ref, policy="all_units")
        created.extend(g.group_key for g in everything.groups)
        assert everything.included_unit_ids == tuple(unit_ids)
        assert everything.unlabeled_unit_ids == (unit_ids[2],)
        assert everything.included_unlabeled_unit_ids == (unit_ids[2],)

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


def test_receipt_summary_explains_counts_and_empty_selections(monkeypatch):
    """``summary()`` reports counts from the receipt and names why a
    selection is empty, without raising or changing anything."""
    import uuid
    from types import MappingProxyType

    from spyglass.spikesorting.v2 import analysis_selection as mod
    from spyglass.spikesorting.v2.curation_api import CurationRef

    ref = CurationRef(
        sorting_id=uuid.uuid4(), curation_id=3, curation_uuid=uuid.uuid4()
    )
    labels = {1: ["accept"], 2: ["mua"], 3: ["reject"], 4: []}
    monkeypatch.setattr(mod, "_labels_by_unit", lambda curation: labels)
    group = mod.SelectedGroup(
        nwb_file_name="s.nwb",
        group_key=MappingProxyType(
            {
                "nwb_file_name": "s.nwb",
                "sorted_spikes_group_name": "g",
                "unit_filter_params_name": "v2_unflagged_units",
            }
        ),
        merge_id=uuid.uuid4(),
        member_index=None,
        status="created",
    )

    def receipt(policy_name, policy, unit_ids):
        included, excluded = mod.apply_unit_selection_policy(
            labels, unit_ids, policy
        )
        return mod.UnitSelectionReceipt(
            curation=ref,
            policy_name=policy_name,
            policy=MappingProxyType(policy),
            included_unit_ids=included,
            excluded_units=MappingProxyType(excluded),
            unlabeled_unit_ids=tuple(u for u in unit_ids if not labels.get(u)),
            groups=(group,),
        )

    unflagged = mod.V2_UNIT_SELECTION_POLICIES["v2_unflagged_units"]
    text = receipt("v2_unflagged_units", unflagged, [1, 2, 3, 4]).summary()
    assert "4 total, 3 selected, 1 excluded (1 unlabeled overall)" in text
    assert "selected: 1 labeled mua, 1 unlabeled" in text
    assert "deny ['noise', 'reject', 'artifact']" in text
    assert "empty selection" not in text

    single = mod.V2_UNIT_SELECTION_POLICIES["v2_accepted_single_units"]
    # No unit carries a required label -> the actionable explanation.
    text = receipt("v2_accepted_single_units", single, [2, 3, 4]).summary()
    assert "0 selected" in text
    assert "no unit carries a required label (one of ['accept'])" in text
    assert "Apply one of those labels" in text
    assert "v2_unflagged_units" in text
    # A valid MUA-only policy gets the same policy-derived guidance, not an
    # instruction to accept units.
    mua_only = {"include_labels": ["mua"], "exclude_labels": ["noise"]}
    text = receipt("mua_only", mua_only, [1, 3, 4]).summary()
    assert "(one of ['mua'])" in text and "accept" not in text.lower().replace(
        "accepted", ""
    )
    # Everything excluded although a required label exists (accept + denied).
    labels[5] = ["accept", "noise"]
    text = receipt("v2_accepted_single_units", single, [5]).summary()
    assert "every unit was excluded by the policy" in text
    # Zero-unit curation.
    text = receipt("v2_accepted_single_units", single, []).summary()
    assert "holds no units" in text

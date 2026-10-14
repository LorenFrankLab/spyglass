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
    assert unflagged["include_labels"] == ()
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


def test_shipped_policies_and_receipt_policy_are_read_only():
    """Deriving a custom policy from a shallow copy of a shipped one cannot
    edit the catalog, and a receipt's policy snapshot cannot be edited
    independently of its recorded verdicts."""
    from types import MappingProxyType

    from spyglass.spikesorting.v2 import analysis_selection as mod

    shipped = mod.V2_UNIT_SELECTION_POLICIES["v2_accepted_single_units"]
    custom = dict(shipped)
    with pytest.raises((AttributeError, TypeError)):
        custom["exclude_labels"].remove("mua")
    custom["exclude_labels"] = tuple(
        v for v in shipped["exclude_labels"] if v != "mua"
    )
    assert "mua" in shipped["exclude_labels"]
    assert (
        "mua"
        in mod.V2_UNIT_SELECTION_POLICIES["v2_accepted_single_units"][
            "exclude_labels"
        ]
    )
    with pytest.raises(TypeError):
        shipped["exclude_labels"] = ()

    receipt = mod.UnitSelectionReceipt(
        curation=None,
        policy_name="v2_accepted_single_units",
        policy=mod._label_policy(["accept"], ["mua"]),
        included_unit_ids=(1,),
        excluded_units=MappingProxyType({2: "denied label(s): mua"}),
        unlabeled_unit_ids=(),
        groups=(),
    )
    with pytest.raises((AttributeError, TypeError)):
        receipt.policy["exclude_labels"].clear()
    with pytest.raises(TypeError):
        receipt.policy["exclude_labels"] = ()
    assert receipt.policy["exclude_labels"] == ("mua",)


@pytest.mark.slow
@pytest.mark.integration
def test_select_units_for_analysis_round_trip(
    planted_three_unit_sort, monkeypatch
):
    """Receipt verdicts == downstream fetch; rejected units cannot enter."""
    import numpy as np

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
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

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

        # Metrics are selected from this exact evaluation and frozen in the
        # group, rather than re-read from the NWB's legacy metric columns.
        evaluation = ref.evaluate(
            metric_params_name="minimal", auto_curation_rules_name="none"
        )
        observed = receipt.observation
        assert observed.duration_s > 0 and not observed.unknown_sources
        assert observed.contains(spikes[0]).all()
        metrics = evaluation.metrics
        assert metrics.loc[unit_ids[0], "observed_duration_s"] == pytest.approx(
            observed.duration_s
        )
        assert metrics.loc[
            unit_ids[0], "observed_firing_rate_hz"
        ] == pytest.approx(len(spikes[0]) / observed.duration_s)
        assert "firing_rate" in metrics  # raw SI definition remains available
        criterion = {"observed_firing_rate_hz": {">=": 0}}
        metric_selection = select_units_for_analysis(
            ref,
            policy="all_units",
            evaluation=evaluation,
            unit_criteria=criterion,
        )
        created.extend(g.group_key for g in metric_selection.groups)
        assert metric_selection.included_unit_ids == tuple(unit_ids)
        assert metric_selection.selection_provenance["evaluation_id"] == str(
            evaluation.evaluation_id
        )
        # A direct consumer reads the frozen membership even if the named
        # policy now excludes everything. Restore the shared default promptly.
        policy_key = {"unit_filter_params_name": "all_units"}
        try:
            UnitSelectionParams.update1(
                {**policy_key, "unit_criteria": {"missing_column": {">": 0}}}
            )
            _, selected_ids = SortedSpikesGroup.fetch_spike_data(
                dict(metric_selection.group_key), return_unit_ids=True
            )
            assert [row["unit_id"] for row in selected_ids] == unit_ids
        finally:
            UnitSelectionParams.update1({**policy_key, "unit_criteria": None})
        empty = select_units_for_analysis(
            ref,
            policy="all_units",
            evaluation=evaluation,
            unit_criteria={"firing_rate": {"<": 0}},
        )
        created.extend(g.group_key for g in empty.groups)
        assert empty.group_key != metric_selection.group_key
        assert empty.included_unit_ids == ()
        assert empty.fetch_spike_data() == []
        assert len(empty.excluded_units) == len(unit_ids)
        with pytest.raises(ValueError, match="curation"):
            select_units_for_analysis(
                CurationRef.from_key(root), evaluation=evaluation
            )

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


@pytest.mark.slow
@pytest.mark.integration
def test_nullable_annotation_criteria_freeze_the_selected_population(
    planted_three_unit_sort,
):
    """A missing boolean stays unavailable through storage, filtering and reads."""
    import pandas as pd

    from spyglass.spikesorting.analysis.v1.group import SortedSpikesGroup
    from spyglass.spikesorting.v2.analysis_selection import (
        select_units_for_analysis,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.curation_api import CurationRef
    from spyglass.spikesorting.v2.unit_annotation import (
        CurationUnitAnnotationSet,
        UnitAnnotationDefinition,
    )
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    sorting_key = dict(planted_three_unit_sort)
    clear_curations_for(sorting_key)
    groups = []
    try:
        root = CurationRef.from_key(
            CurationV2.create_initial_curation(sorting_key)
        )
        units = sorted(
            int(u) for u in (CurationV2.Unit & root.as_key()).fetch("unit_id")
        )
        definition = UnitAnnotationDefinition.insert_definition(
            "nullable_population_flag", 1, "bool"
        )
        annotation = CurationUnitAnnotationSet.from_dataframe(
            root,
            definition,
            pd.DataFrame(
                {"flag": pd.array([True, None, False], dtype="boolean")},
                index=units,
            ),
            producer="nullable-population-regression",
        )
        for criterion in (
            {"==": True},
            {"!=": False},
            {"isin": [True]},
            {"notin": [False]},
        ):
            receipt = select_units_for_analysis(
                root,
                policy="all_units",
                annotation_sets=[annotation],
                unit_criteria={annotation.column_name: criterion},
            )
            groups.extend(group.group_key for group in receipt.groups)
            assert receipt.included_unit_ids == (units[0],)
            assert receipt.excluded_units[units[1]].startswith(
                "unavailable metric:"
            )
            _, identities = receipt.fetch_spike_data(return_unit_ids=True)
            assert [identity["unit_id"] for identity in identities] == [
                units[0]
            ]

        # The shared filter also serves legacy numeric and list-valued columns.
        for values in (
            pd.Series([1, None, 0], dtype="Int64"),
            pd.Series([1.0, None, 0.0]),
            pd.Series([1, None, 0], dtype=object),
        ):
            for criterion in (
                {">": 0},
                {"!=": 0},
                {"notin": [0]},
                {"between": [1, 2]},
            ):
                assert SortedSpikesGroup.filter_units_by_criteria(
                    values.to_frame("value"), {"value": criterion}
                ).tolist() == [True, False, False]
        assert SortedSpikesGroup.filter_units_by_criteria(
            pd.DataFrame({"labels": [[], ["noise"], None, pd.NA]}),
            {"labels": {"notin": ["noise"]}},
        ).tolist() == [True, False, False, False]
        assert not SortedSpikesGroup.filter_units_by_criteria(
            pd.DataFrame({"flag": pd.array([None, None], dtype="boolean")}),
            {"flag": {"!=": False}},
        ).any()
    finally:
        for group_key in groups:
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

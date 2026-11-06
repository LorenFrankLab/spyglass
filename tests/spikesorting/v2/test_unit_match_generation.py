"""Reviewed match plans refuse deleted and recreated curation generations."""

import uuid

import pytest


@pytest.fixture
def reviewed_curation(planted_two_unit_sort):
    from spyglass.spikesorting.v2.curation import CurationV2
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    clear_curations_for(planted_two_unit_sort)
    try:
        root = CurationV2.insert_curation(planted_two_unit_sort)
        child = CurationV2.insert_curation(
            planted_two_unit_sort,
            parent_curation_id=root["curation_id"],
            labels={0: ["noise"]},
        )
        yield root, child
    finally:
        clear_curations_for(planted_two_unit_sort)


def _selection_snapshot():
    """Track both registrations and their pinned generations on failure."""
    from spyglass.spikesorting.v2.unit_matching import UnitMatchSelection

    return (
        UnitMatchSelection.fetch("KEY", as_dict=True, order_by="unitmatch_id"),
        UnitMatchSelection.Input.fetch(
            "unitmatch_id",
            "input_index",
            "curation_uuid",
            as_dict=True,
            order_by="unitmatch_id, input_index",
        ),
        UnitMatchSelection.InputRecording.fetch(
            "KEY",
            as_dict=True,
            order_by="unitmatch_id, input_index, recording_index",
        ),
    )


@pytest.mark.database
@pytest.mark.integration
@pytest.mark.stage
@pytest.mark.parametrize("source", ["sorts", "group"])
def test_match_plan_rejects_recreated_curation(reviewed_curation, source):
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.pipeline import (
        plan_v2_unit_match,
        plan_v2_unit_match_from_sorts,
        run_v2_unit_match,
    )
    from spyglass.spikesorting.v2.recording import RecordingSelection
    from spyglass.spikesorting.v2.session_group import SessionGroup
    from spyglass.spikesorting.v2.sorting import SortingSelection
    from spyglass.spikesorting.v2.unit_matching import (
        MatcherParameters,
        UnitMatchSelection,
    )

    root, child = reviewed_curation
    sorting_key = {"sorting_id": root["sorting_id"]}
    MatcherParameters.insert_default()
    original_uuid = (CurationV2 & child).fetch1("curation_uuid")
    assert CurationV2().get_matchable_unit_ids(child).tolist() == [1]
    group_key = None
    try:
        if source == "sorts":
            plan = plan_v2_unit_match_from_sorts(
                [root["sorting_id"]], curation_strategy="final_curated"
            )
            pins = plan.curations
        else:
            recording = SortingSelection.resolve_source(sorting_key)
            row = (RecordingSelection & recording.key).fetch1()
            group_key = {
                "session_group_owner": row["team_name"],
                "session_group_name": f"generation_{uuid.uuid4().hex[:12]}",
            }
            SessionGroup.create_group(
                group_key["session_group_owner"],
                group_key["session_group_name"],
                [
                    {
                        name: row[name]
                        for name in (
                            "nwb_file_name",
                            "sort_group_id",
                            "interval_list_name",
                            "team_name",
                        )
                    }
                ],
            )
            plan = plan_v2_unit_match(
                **group_key, curation_strategy="final_curated"
            )
            pins = list(plan.curation_choices.values())
        assert plan.ok, plan.errors
        assert pins[0]["curation_uuid"] == str(original_uuid)
        assert plan.as_dataframe().iloc[0]["curation_uuid"] == str(
            original_uuid
        )

        (CurationV2 & child).delete(safemode=False)
        replacement = CurationV2.insert_curation(
            sorting_key,
            parent_curation_id=root["curation_id"],
            labels={1: ["noise"]},
            description="replacement after review",
        )
        assert replacement == child
        replacement_uuid = (CurationV2 & replacement).fetch1("curation_uuid")
        assert replacement_uuid != original_uuid
        assert CurationV2().get_matchable_unit_ids(replacement).tolist() == [0]

        # The replacement's actual generation can be selected and frozen.
        pk = UnitMatchSelection.insert_inputs(
            [
                {
                    **replacement,
                    "curation_uuid": str(replacement_uuid),
                }
            ],
            "unitmatch_default",
        )
        try:
            assert (UnitMatchSelection.Input & pk).fetch1(
                "curation_uuid"
            ) == replacement_uuid
            # Even an existing selection for the replacement must not make
            # the stale plan succeed through a find-existing fast path.
            before = _selection_snapshot()
            with pytest.raises(
                ValueError, match="changed curation_uuid since planning"
            ):
                run_v2_unit_match(plan)
            assert _selection_snapshot() == before
            assert (
                UnitMatchSelection.insert_inputs(
                    [replacement], "unitmatch_default"
                )
                == pk
            )
            if source == "group":
                fresh_plan = plan_v2_unit_match(
                    **group_key, curation_strategy="final_curated"
                )
                assert (
                    UnitMatchSelection.insert_selection(
                        **group_key,
                        matcher_params_name="unitmatch_default",
                        curation_choices=fresh_plan.curation_choices,
                    )
                    == pk
                )
            else:
                fresh_plan = plan_v2_unit_match_from_sorts(
                    [root["sorting_id"]], curation_strategy="final_curated"
                )
                assert (
                    UnitMatchSelection.insert_inputs(
                        fresh_plan.curations, "unitmatch_default"
                    )
                    == pk
                )
        finally:
            (UnitMatchSelection & pk).super_delete(warn=False, safemode=False)
    finally:
        if group_key is not None:
            (SessionGroup & group_key).super_delete(warn=False, safemode=False)

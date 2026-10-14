"""Real-table checks for reviewed curation pins and frozen match eligibility."""

import copy
import uuid

import pytest

pytestmark = [pytest.mark.stage, pytest.mark.integration]


@pytest.fixture
def accepted_three_unit_curation(planted_three_unit_sort):
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.curation_api import CurationRef
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    clear_curations_for(planted_three_unit_sort)
    try:
        root = CurationV2.insert_curation(planted_three_unit_sort)
        child = CurationV2.insert_curation(
            planted_three_unit_sort,
            parent_curation_id=root["curation_id"],
            labels={0: ["accept"], 1: ["noise"], 2: ["mua"]},
        )
        yield CurationRef.from_key(child)
    finally:
        clear_curations_for(planted_three_unit_sort)


def _selection_count():
    from spyglass.spikesorting.v2.unit_matching import UnitMatchSelection

    return len(UnitMatchSelection())


def test_direct_match_preflight_reads_existing_assets_without_writes(
    accepted_three_unit_curation, monkeypatch
):
    import datajoint as dj
    from datajoint.autopopulate import AutoPopulate

    from spyglass.common import AnalysisNwbfile, IntervalList
    from spyglass.spikesorting.analysis.v1.group import SortedSpikesGroup
    from spyglass.spikesorting.v2 import analysis_selection
    from spyglass.spikesorting.v2.motion import MotionCorrectedRecording
    from spyglass.spikesorting.v2.pipeline import (
        describe_epochs,
        plan_v2_unit_match_from_curations,
        preflight_v2_unit_match,
    )
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
    )
    from spyglass.spikesorting.v2.session_group import ConcatenatedRecording
    from spyglass.spikesorting.v2.sorting import SortingSelection
    from spyglass.spikesorting.v2.unit_matching import UnitMatchSelection

    recording = (
        RecordingSelection * SortingSelection.RecordingSource
        & accepted_three_unit_curation.as_key()
    ).fetch1()
    interval_key = {
        "nwb_file_name": recording["nwb_file_name"],
        "interval_list_name": recording["interval_list_name"],
    }
    assert interval_key["interval_list_name"] == "raw data valid times"
    expected_bounds = tuple(
        tuple(map(float, bound))
        for bound in (IntervalList & interval_key).fetch1("valid_times")
    )
    recording_key = {"recording_id": recording["recording_id"]}

    def forbidden(*args, **kwargs):
        pytest.fail("Matching preflight attempted a mutation or asset rebuild.")

    before = (_selection_count(), len(SortedSpikesGroup()))
    with monkeypatch.context() as guard:
        guard.setattr(dj.Table, "insert", forbidden)
        guard.setattr(dj.Table, "update1", forbidden)
        guard.setattr(dj.Table, "delete", forbidden)
        guard.setattr(AutoPopulate, "populate", forbidden)
        guard.setattr(AnalysisNwbfile, "create", forbidden)
        guard.setattr(SortedSpikesGroup, "create_group", forbidden)
        guard.setattr(
            analysis_selection, "ensure_v2_unit_selection_policies", forbidden
        )
        for table in (
            Recording,
            ConcatenatedRecording,
            MotionCorrectedRecording,
        ):
            guard.setattr(table, "get_recording", forbidden)
        guard.setattr(Recording, "resolve_stored_traces", forbidden)
        epochs = describe_epochs(
            recording["nwb_file_name"], sort_group_id=recording["sort_group_id"]
        )
        chosen = epochs[
            epochs["interval_list_name"] == interval_key["interval_list_name"]
        ]
        assert len(chosen) == 1
        row = chosen.iloc[0]
        assert row["interval_key"] == interval_key
        assert row["interval_status"] == "valid"
        assert row["interval_bounds"] == expected_bounds
        assert row["start_time"] == expected_bounds[0][0]
        assert row["stop_time"] == expected_bounds[-1][1]
        assert row["duration_s"] == pytest.approx(
            sum(stop - start for start, stop in expected_bounds)
        )
        assert recording_key in row["recording_keys"]
        assert recording_key in row["ready_recording_keys"]
        assert row["recording_readiness"] == "cache_present"
        candidate = next(
            value
            for value in row["recordings"]
            if value["recording_id"] == recording_key["recording_id"]
        )
        assert candidate["recording_computed"]
        assert candidate["cache_present"]
        assert candidate["sort_group_id"] == recording["sort_group_id"]
        plan = plan_v2_unit_match_from_curations([accepted_three_unit_curation])
        assert plan.ok, plan.errors
        assert plan.unit_policy == "v2_accepted_single_units"
        assert plan.unit_selections[0].included_unit_ids == (0,)
        assert plan.unit_selections[0].excluded_units == {
            1: "denied label(s): noise",
            2: "denied label(s): mua",
        }
        report = preflight_v2_unit_match(plan)
        assert report.ok, report.errors
    assert (_selection_count(), len(SortedSpikesGroup())) == before
    assert not (
        UnitMatchSelection.Input
        & {"sorting_id": accepted_three_unit_curation.sorting_id}
    )


def test_policy_drift_blocks_plan_but_existing_selection_keeps_frozen_units(
    accepted_three_unit_curation, monkeypatch
):
    from spyglass.spikesorting.analysis.v1.group import UnitSelectionParams
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.pipeline import (
        plan_v2_unit_match_from_curations,
        preflight_v2_unit_match,
    )
    from spyglass.spikesorting.v2.unit_matching import (
        TrackedUnit,
        UnitMatch,
        UnitMatchSelection,
    )

    policy = {
        "unit_filter_params_name": f"match_accept_{uuid.uuid4().hex[:12]}",
        "include_labels": ["accept"],
        "exclude_labels": ["mua", "noise", "reject", "artifact"],
        "unit_criteria": None,
    }
    UnitSelectionParams.insert1(policy)
    selection_key = None
    try:
        plan = plan_v2_unit_match_from_curations(
            [accepted_three_unit_curation],
            unit_policy=policy["unit_filter_params_name"],
        )
        assert plan.ok, plan.errors
        selection_key = UnitMatchSelection.insert_inputs(
            plan.curations,
            plan.matcher_params_name,
            unit_selections=[
                value.snapshot() for value in plan.unit_selections
            ],
            expected_input_set_hash=plan.input_set_hash,
        )
        stored = (UnitMatchSelection.Input & selection_key).fetch1(
            "selection_snapshot"
        )
        assert stored["included_unit_ids"] == [0]
        assert stored["curation_uuid"] == str(
            accepted_three_unit_curation.curation_uuid
        )
        assert set(stored["labels_by_unit"]) == {"0", "1", "2"}

        UnitSelectionParams.update1(
            {**policy, "exclude_labels": ["accept", "mua", "noise"]}
        )
        before = _selection_count()
        report = preflight_v2_unit_match(plan)
        assert not report.ok
        assert any("changed since planning" in error for error in report.errors)
        assert _selection_count() == before

        def no_live_exclusion_filter(*args, **kwargs):
            pytest.fail(
                "Matching must use the frozen selected IDs, not live labels."
            )

        monkeypatch.setattr(
            CurationV2, "get_matchable_unit_ids", no_live_exclusion_filter
        )
        fetched = UnitMatch().make_fetch(selection_key)
        assert fetched.input_plan[0]["matchable_unit_ids"] == [0]
        assert fetched.input_plan[0]["selection_snapshot"] == stored
        UnitMatch.populate(selection_key, reserve_jobs=False)
        TrackedUnit.populate(selection_key, reserve_jobs=False)
        assert (UnitMatch.MatchableUnit & selection_key).fetch(
            "unit_id"
        ).tolist() == [0]
        assert (TrackedUnit.Member & selection_key).fetch(
            "unit_id"
        ).tolist() == [0]
    finally:
        if selection_key is not None:
            (UnitMatchSelection & selection_key).super_delete(warn=False)
        (
            UnitSelectionParams
            & {"unit_filter_params_name": policy["unit_filter_params_name"]}
        ).super_delete(warn=False)


@pytest.mark.pipeline
@pytest.mark.parametrize(
    "missing", ["curated_units", "sorting_spans", "effective_concat_traces"]
)
def test_preflight_blocks_missing_assets_without_a_geometry_hook(
    daily_concat_match_inputs, monkeypatch, missing
):
    from pathlib import Path

    from spyglass.common import AnalysisNwbfile
    from spyglass.spikesorting.v2._matching.coverage import stored_analysis_path
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.curation_api import CurationRef
    from spyglass.spikesorting.v2.pipeline import (
        plan_v2_unit_match_from_curations,
        preflight_v2_unit_match,
    )
    from spyglass.spikesorting.v2.recording import Recording
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection
    from spyglass.spikesorting.v2.unit_matching import MatcherParameters
    from tests.spikesorting.v2._unitmatch_helpers import (
        install_fixture_pairer,
        restore_matcher_registry,
    )

    curations = daily_concat_match_inputs["curations"]
    refs = [
        CurationRef.from_key(curations[name])
        for name in ("single_b", "concat_day2")
    ]
    params_name = "missing_asset_preflight_params"
    saved = install_fixture_pairer(
        monkeypatch,
        matcher_name="missing_asset_pairer",
        matcher_params_name=params_name,
        pairs=[],
    )
    ref = refs[1]
    if missing == "curated_units":
        name = (CurationV2 & ref.as_key()).fetch1("analysis_file_name")
    elif missing == "sorting_spans":
        name = (Sorting & ref.as_key()).fetch1("analysis_file_name")
    else:
        name = SortingSelection.resolve_effective_source(
            ref.as_key()
        ).traces.row["analysis_file_name"]
    path = Path(stored_analysis_path(name))
    held = path.with_suffix(".preflight-held")
    try:
        plan = plan_v2_unit_match_from_curations(
            refs,
            unit_policy="v2_unflagged_units",
            matcher_params_name=params_name,
        )
        assert plan.ok, plan.errors
        before = _selection_count()
        path.rename(held)

        def forbidden(*args, **kwargs):
            pytest.fail(
                "Missing asset preflight attempted a fallback or rebuild."
            )

        with monkeypatch.context() as guard:
            guard.setattr(AnalysisNwbfile, "get_abs_path", forbidden)
            guard.setattr(Recording, "resolve_stored_traces", forbidden)
            guard.setattr(
                SortingSelection, "ensure_effective_traces", forbidden
            )
            report = preflight_v2_unit_match(plan)
            assert not report.ok
            assert any(
                "missing its local NWB artifact" in error
                for error in report.errors
            )
            blocked_plan = plan_v2_unit_match_from_curations(
                refs,
                unit_policy="v2_unflagged_units",
                matcher_params_name=params_name,
            )
            assert not blocked_plan.ok
            assert not path.exists()

            # The user restores the original artifact, then reviews a new
            # plan. Historical missing-file findings must not keep fresh
            # preflight blocked or change the frozen source/population.
            held.rename(path)
            repaired_report = preflight_v2_unit_match(blocked_plan)
            assert repaired_report.ok, repaired_report.errors
            repaired_plan = plan_v2_unit_match_from_curations(
                refs,
                unit_policy="v2_unflagged_units",
                matcher_params_name=params_name,
            )
            assert repaired_plan.ok, repaired_plan.errors
            assert repaired_plan.input_set_hash == plan.input_set_hash
            assert [
                selection.selection_hash
                for selection in repaired_plan.unit_selections
            ] == [
                selection.selection_hash for selection in plan.unit_selections
            ]
        assert _selection_count() == before
        assert path.exists()
    finally:
        if held.exists():
            held.rename(path)
        (MatcherParameters & {"matcher_params_name": params_name}).super_delete(
            warn=False
        )
        restore_matcher_registry(saved)


@pytest.mark.pipeline
def test_preflight_blocks_unimportable_matching_runtime_before_inserts(
    daily_concat_match_inputs, monkeypatch
):
    from spyglass.spikesorting.v2._matching import unitmatch_backend
    from spyglass.spikesorting.v2.curation_api import CurationRef
    from spyglass.spikesorting.v2.pipeline import (
        plan_v2_unit_match_from_curations,
    )

    refs = [
        CurationRef.from_key(daily_concat_match_inputs["curations"][name])
        for name in ("single_b", "concat_day2")
    ]
    before = _selection_count()

    def unavailable():
        raise ImportError(
            "UnitMatchPy native dependency is unavailable; repair the matching environment."
        )

    monkeypatch.setattr(unitmatch_backend, "_require_unitmatch", unavailable)
    plan = plan_v2_unit_match_from_curations(
        refs, unit_policy="v2_unflagged_units"
    )
    assert not plan.ok
    assert any(
        "native dependency is unavailable" in error for error in plan.errors
    )
    assert _selection_count() == before


def test_direct_match_plan_refuses_recreated_numeric_curation_id(
    accepted_three_unit_curation,
):
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.exceptions import PreflightError
    from spyglass.spikesorting.v2.pipeline import (
        plan_v2_unit_match_from_curations,
        preflight_v2_unit_match,
        run_v2_unit_match,
    )

    ref = accepted_three_unit_curation
    plan = plan_v2_unit_match_from_curations([ref])
    assert plan.ok, plan.errors
    key = ref.as_key()
    parent = (CurationV2 & key).fetch1("parent_curation_id")
    (CurationV2 & key).delete(safemode=False)
    replacement = CurationV2.insert_curation(
        {"sorting_id": ref.sorting_id},
        parent_curation_id=parent,
        labels={0: ["accept"], 1: ["noise"], 2: ["mua"]},
        description="replacement with the same numeric id",
    )
    assert replacement == key
    assert (CurationV2 & replacement).fetch1(
        "curation_uuid"
    ) != ref.curation_uuid
    before = _selection_count()
    report = preflight_v2_unit_match(plan)
    assert not report.ok
    assert any("generation" in error for error in report.errors)
    with pytest.raises(PreflightError, match="generation"):
        run_v2_unit_match(plan)
    assert _selection_count() == before


def test_single_input_preflight_requires_registered_matcher_metadata(
    accepted_three_unit_curation,
):
    """Even a zero-pair run must identify its registered metadata producer."""
    from spyglass.spikesorting.v2.pipeline import (
        plan_v2_unit_match_from_curations,
        preflight_v2_unit_match,
    )
    from spyglass.spikesorting.v2.unit_matching import MatcherParameters
    from tests.spikesorting.v2.test_unitmatch_concat import _raw_update

    plan = plan_v2_unit_match_from_curations([accepted_three_unit_curation])
    assert plan.ok, plan.errors
    before = _selection_count()
    # A persisted custom recipe can outlive its process-local registration.
    with _raw_update(
        MatcherParameters,
        {"matcher_params_name": plan.matcher_params_name},
        "matcher",
        "missing_single_matcher",
    ):
        report = preflight_v2_unit_match(plan)
        assert not report.ok
        assert any("Unknown matcher" in error for error in report.errors)
    assert _selection_count() == before


@pytest.mark.parametrize(
    "corruption", ["phantom_excluded_unit", "malformed_coverage"]
)
def test_worker_refuses_forged_snapshot_before_artifact_access(
    accepted_three_unit_curation, monkeypatch, corruption
):
    from spyglass.spikesorting.v2._curation.unit_selection import (
        unit_selection_hash,
    )
    from spyglass.spikesorting.v2._matching import coverage, fetch
    from spyglass.spikesorting.v2.exceptions import (
        UnitMatchSelectionIntegrityError,
    )
    from spyglass.spikesorting.v2.pipeline import (
        plan_v2_unit_match_from_curations,
    )
    from spyglass.spikesorting.v2.unit_matching import (
        UnitMatch,
        UnitMatchSelection,
    )
    from tests.spikesorting.v2._unitmatch_helpers import reforge_selection

    plan = plan_v2_unit_match_from_curations([accepted_three_unit_curation])
    assert plan.ok, plan.errors
    selection_key = UnitMatchSelection.insert_inputs(
        plan.curations,
        plan.matcher_params_name,
        unit_selections=[value.snapshot() for value in plan.unit_selections],
        expected_input_set_hash=plan.input_set_hash,
    )
    try:
        if corruption == "phantom_excluded_unit":
            data = copy.deepcopy(
                (UnitMatchSelection.Input & selection_key).fetch1(
                    "selection_snapshot"
                )
            )
            data["labels_by_unit"]["999"] = []
            data["excluded_units"]["999"] = "unlabeled; policy requires accept"
            data["selection_hash"] = unit_selection_hash(data)
            reforge_selection(
                selection_key,
                input_edits={0: {"selection_snapshot": data}},
                rehash=True,
            )
        else:
            data = copy.deepcopy(
                (UnitMatchSelection.InputRecording & selection_key).fetch1(
                    "source_coverage"
                )
            )
            data["sample_ranges"] = [[5, 4]]
            reforge_selection(
                selection_key,
                recording_edits={(0, 0): {"source_coverage": data}},
            )

        def forbidden(*args, **kwargs):
            pytest.fail("Forged matching metadata reached artifact I/O.")

        with monkeypatch.context() as guard:
            guard.setattr(fetch, "_member_match_files", forbidden)
            guard.setattr(fetch, "_input_stored_units", forbidden)
            guard.setattr(coverage, "open_base_recording", forbidden)
            guard.setattr(
                coverage, "resolve_recording_source_coverage", forbidden
            )
            with pytest.raises(UnitMatchSelectionIntegrityError):
                UnitMatch().make_fetch(selection_key)
    finally:
        (UnitMatchSelection & selection_key).super_delete(warn=False)


@pytest.mark.pipeline
@pytest.mark.parametrize(
    "input_names, expected_unit_ids",
    [
        pytest.param(
            ("concat_day1", "concat_day2"), (0, 3), id="daily-concatenations"
        ),
        pytest.param(
            ("single_a_first", "single_b"), (0, 0), id="standalone-recordings"
        ),
    ],
)
def test_direct_reviewed_curation_populations_reach_match_artifact(
    daily_concat_match_inputs, monkeypatch, input_names, expected_unit_ids
):
    """Reviewed single-unit populations keep their evidence and known matches.

    The fixture declares one unit per input, firing from frame 1000 every
    5000 frames. Its original raw clocks and declared member intervals define
    the expected downstream events. Evaluations are evidence pins here;
    selective metric filtering and native matcher inference are separate tests.
    """
    import numpy as np

    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.curation_api import CurationRef
    from spyglass.spikesorting.v2.pipeline import (
        plan_v2_unit_match_from_curations,
        resolve_unit_selection,
        run_v2_unit_match,
    )
    from spyglass.spikesorting.v2.unit_matching import (
        MatcherParameters,
        TrackedUnit,
        UnitMatch,
        UnitMatchSelection,
    )
    from tests.spikesorting.v2._unitmatch_helpers import (
        install_fixture_pairer,
        restore_matcher_registry,
    )
    from tests.spikesorting.v2.conftest import (
        DAILY_CONCAT_INTERVALS,
        DAILY_CONCAT_UNIT_IDS,
    )
    from tests.spikesorting.v2._scenario_oracles import _raw_acquisition_clock

    fx = daily_concat_match_inputs
    roots = fx["curations"]
    assert tuple(
        DAILY_CONCAT_UNIT_IDS.get(name, 0) for name in input_names
    ) == (expected_unit_ids)
    member_specs = {
        "concat_day1": (("a", "first"), ("a", "second")),
        "concat_day2": (("c", "first"), ("c", "second")),
        "single_a_first": (("a", "first"),),
        "single_b": (("b", None),),
    }
    raw_clocks = {
        tag: _raw_acquisition_clock(fx["nwb_file_names"][tag])[0]
        for tag in {
            tag for name in input_names for tag, _ in member_specs[name]
        }
    }
    expected_member_events = {}
    for name in input_names:
        member_clocks = []
        for tag, part in member_specs[name]:
            raw_clock = raw_clocks[tag]
            if part is None:
                selected_clock = raw_clock
                interval_name = "raw data valid times"
            else:
                start, stop = np.asarray(DAILY_CONCAT_INTERVALS[part]) + (
                    raw_clock[0]
                )
                selected_clock = raw_clock[
                    (raw_clock >= start) & (raw_clock <= stop)
                ]
                interval_name = f"unitmatch_daily_{part}"
            assert selected_clock.size > 2000
            member_clocks.append(
                (fx["nwb_file_names"][tag], interval_name, selected_clock)
            )
        # This is the fixture's declared planted train, not a sort/output read.
        frames = np.arange(
            1000, sum(len(clock) for _, _, clock in member_clocks) - 1000, 5000
        )
        offset = 0
        for recording_index, (nwb_name, interval_name, clock) in enumerate(
            member_clocks
        ):
            local_frames = (
                frames[(frames >= offset) & (frames < offset + len(clock))]
                - offset
            )
            assert local_frames.size > 0
            expected_member_events[name, recording_index] = (
                nwb_name,
                interval_name,
                clock[local_frames],
            )
            offset += len(clock)

    chosen = []
    for name, unit_id in zip(input_names, expected_unit_ids, strict=True):
        root = roots[name]
        child = CurationV2.insert_curation(
            {"sorting_id": root["sorting_id"]},
            parent_curation_id=root["curation_id"],
            labels={unit_id: ["accept"]},
        )
        chosen.append(CurationRef.from_key(child))
    evaluations = [
        ref.evaluate(
            metric_params_name="minimal", auto_curation_rules_name="none"
        )
        for ref in chosen
    ]
    snapshots = [
        resolve_unit_selection(
            ref,
            evaluation=evaluation,
        )
        for ref, evaluation in zip(chosen, evaluations, strict=True)
    ]
    seen = []
    params_name = "direct_reviewed_population_params"
    saved = install_fixture_pairer(
        monkeypatch,
        matcher_name="reviewed_population_pairer",
        matcher_params_name=params_name,
        pairs=[list(expected_unit_ids)],
        seen_unit_ids=seen,
    )
    selection_key = None
    try:
        for ref, unit_id, snapshot in zip(
            chosen, expected_unit_ids, snapshots, strict=True
        ):
            assert snapshot.curation == ref
            assert snapshot.included_unit_ids == (unit_id,)
            assert snapshot.excluded_units == {}
            assert snapshot.labels_by_unit == {unit_id: ("accept",)}
            assert snapshot.unlabeled_unit_ids == ()
            assert snapshot.unit_criteria == {}
        plan = plan_v2_unit_match_from_curations(
            list(reversed(chosen)),
            matcher_params_name=params_name,
            unit_selections=list(reversed(snapshots)),
        )
        assert plan.ok, plan.errors
        # A different terminal branch must not replace an explicitly reviewed pin.
        CurationV2.insert_curation(
            {"sorting_id": roots[input_names[0]]["sorting_id"]},
            parent_curation_id=roots[input_names[0]]["curation_id"],
            labels={expected_unit_ids[0]: ["reject"]},
            description="unselected terminal branch",
        )
        result = run_v2_unit_match(plan)
        selection_key = {"unitmatch_id": result["unit_match_id"]}
        rows = (UnitMatchSelection.Input & selection_key).fetch(
            as_dict=True, order_by="input_index"
        )
        assert [row["curation_uuid"] for row in rows] == [
            ref.curation_uuid for ref in chosen
        ]
        assert [
            row["selection_snapshot"]["included_unit_ids"] for row in rows
        ] == [[unit] for unit in expected_unit_ids]
        assert [
            row["selection_snapshot"]["excluded_units"] for row in rows
        ] == [
            {},
            {},
        ]
        assert [
            row["selection_snapshot"]["labels_by_unit"] for row in rows
        ] == [{str(unit): ["accept"]} for unit in expected_unit_ids]
        assert seen == [[unit] for unit in expected_unit_ids]
        expected_nodes = [
            (str(ref.sorting_id), ref.curation_id, unit)
            for ref, unit in zip(chosen, expected_unit_ids, strict=True)
        ]
        pairs = (UnitMatch.Pair & selection_key).fetch(as_dict=True)
        assert len(pairs) == 1
        pair = pairs[0]
        assert tuple(
            (
                str(pair[f"session_{side}_sorting_id"]),
                int(pair[f"session_{side}_curation_id"]),
                int(pair[f"unit_{side}_id"]),
            )
            for side in ("a", "b")
        ) == tuple(expected_nodes)
        assert pair["match_probability"] == pytest.approx(0.99)
        matchable = (UnitMatch.MatchableUnit & selection_key).fetch(
            as_dict=True
        )
        assert {
            (
                str(row["sorting_id"]),
                int(row["curation_id"]),
                int(row["unit_id"]),
            )
            for row in matchable
        } == set(expected_nodes)
        tracked = (TrackedUnit & selection_key).fetch(as_dict=True)
        assert len(tracked) == 1
        assert (
            tracked[0]["n_matching_inputs"],
            tracked[0]["n_sessions_detected"],
        ) == (2, 2)
        assert {
            (
                str(row["sorting_id"]),
                int(row["curation_id"]),
                int(row["unit_id"]),
            )
            for row in (TrackedUnit.Member & selection_key).fetch(as_dict=True)
        } == set(expected_nodes)
        spikes = TrackedUnit().get_member_spike_times(selection_key)
        input_by_sorting = {
            str(ref.sorting_id): (name, ref.curation_id, unit)
            for name, ref, unit in zip(
                input_names, chosen, expected_unit_ids, strict=True
            )
        }
        observed_members = set()
        for member in spikes.itertuples():
            name, curation_id, unit_id = input_by_sorting[
                str(member.sorting_id)
            ]
            assert (member.curation_id, member.unit_id) == (
                curation_id,
                unit_id,
            )
            identity = (name, member.recording_index)
            assert identity not in observed_members
            observed_members.add(identity)
            nwb_name, interval_name, expected_times = expected_member_events[
                identity
            ]
            assert (member.nwb_file_name, member.interval_list_name) == (
                nwb_name,
                interval_name,
            )
            assert len(member.spike_times) == len(expected_times) > 0
            np.testing.assert_array_equal(member.spike_times, expected_times)
        assert observed_members == set(expected_member_events)
        db_inputs, _ = UnitMatch().get_input_provenance(selection_key)
        nwb_inputs, _ = UnitMatch().get_input_provenance(
            selection_key, from_nwb=True
        )
        assert (
            db_inputs["selection_snapshot"].tolist()
            == nwb_inputs["selection_snapshot"].tolist()
        )
        assert all(
            snapshot["policy_name"] == "v2_accepted_single_units"
            and snapshot["selection_hash"]
            for snapshot in nwb_inputs["selection_snapshot"]
        )
        assert [
            snapshot["selection_provenance"]["evaluation_id"]
            for snapshot in nwb_inputs["selection_snapshot"]
        ] == [str(evaluation.evaluation_id) for evaluation in evaluations]
        assert all(
            snapshot["unit_criteria"] == {}
            and snapshot["selection_provenance"]["evaluation_recipes"]
            == {
                "metric_params_name": "minimal",
                "auto_curation_rules_name": "none",
            }
            for snapshot in nwb_inputs["selection_snapshot"]
        )
    finally:
        if selection_key is not None:
            (UnitMatchSelection & selection_key).super_delete(warn=False)
        (MatcherParameters & {"matcher_params_name": params_name}).super_delete(
            warn=False
        )
        restore_matcher_registry(saved)

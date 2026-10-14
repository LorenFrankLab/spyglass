"""Match separately sorted, disjoint epochs on one original NWB clock."""

import importlib.util
import json
import os
import uuid
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.pipeline

_FIGPACK_MISSING = (
    importlib.util.find_spec("figpack") is None
    or importlib.util.find_spec("figpack_spike_sorting") is None
)


def _plant_three_epoch_units(recording, **kwargs):
    """Replace sorter inference while keeping the public pipeline's NWB writes."""
    import spikeinterface as si

    samples = np.arange(1500, recording.get_num_samples() - 1500, 2500)
    frames = np.concatenate([samples + 200 * unit for unit in range(3)])
    labels = np.repeat(np.arange(3), len(samples))
    order = np.argsort(frames)
    return si.NumpySorting.from_samples_and_labels(
        samples_list=[frames[order].astype(np.int64)],
        labels_list=[labels[order].astype(np.int32)],
        sampling_frequency=recording.get_sampling_frequency(),
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    "reviewed,unit_policy,expected_units",
    [
        pytest.param(
            True,
            None,
            [0],
            id="reviewed_accepted",
            marks=pytest.mark.skipif(
                _FIGPACK_MISSING
                and not os.environ.get("SPYGLASS_REQUIRE_BROWSER_TESTS"),
                reason="requires the spikesorting-v2-curation extra (figpack)",
            ),
        ),
        pytest.param(
            False,
            "v2_unflagged_units",
            [0, 1, 2],
            id="auto_only_unflagged",
        ),
    ],
)
def test_public_epoch_matching_journey(
    daily_concat_match_inputs,
    monkeypatch,
    reviewed,
    unit_policy,
    expected_units,
):
    """Discover, sort, curate, plan, match and reuse through supported APIs.

    Only sorter inference and pair inference are substituted. Preprocessing,
    evaluations, optional review commits, dense waveform extraction, original
    clocks, match provenance and tracked-unit persistence are real.
    """
    from spyglass.spikesorting.v2._orchestration import presets
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.curation_api import CurationRef
    from spyglass.spikesorting.v2.exceptions import PipelineInputError
    from spyglass.spikesorting.v2.pipeline import (
        describe_epochs,
        plan_v2_unit_match_from_curations,
        preflight_v2_unit_match,
        register_pipeline_preset,
        run_v2_pipeline,
        run_v2_unit_match,
    )
    from spyglass.spikesorting.v2.recording import RecordingSelection
    from spyglass.spikesorting.v2.sorting import SortingSelection
    from spyglass.spikesorting.v2.unit_matching import (
        MatcherParameters,
        TrackedUnit,
        UnitMatch,
        UnitMatchSelection,
    )
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for
    from tests.spikesorting.v2._sorter_stub import plant_sorter
    from tests.spikesorting.v2._unitmatch_helpers import (
        install_fixture_pairer,
        restore_matcher_registry,
    )

    fx = daily_concat_match_inputs
    selected = [
        (RecordingSelection & fx["recording_keys"][name]).fetch1()
        for name in ("a_first", "a_second")
    ]
    nwb_name = fx["nwb_file_names"]["a"]
    epochs = describe_epochs(
        nwb_name, sort_group_id=selected[0]["sort_group_id"]
    ).set_index("interval_list_name")
    interval_names = [row["interval_list_name"] for row in selected]
    assert epochs.loc[interval_names, "interval_status"].tolist() == [
        "valid",
        "valid",
    ]
    assert all(
        epochs.loc[name, "ready_recording_keys"] for name in interval_names
    )
    assert (
        epochs.loc[interval_names[0], "stop_time"]
        < epochs.loc[interval_names[1], "start_time"]
    )

    params_name = f"epoch_public_{uuid.uuid4().hex[:10]}"
    register_pipeline_preset(
        params_name,
        {
            "preprocessing_params_name": "default",
            "sorter": "mountainsort4",
            "sorter_params_name": "franklab_30khz_ms4_2026_06",
            "metric_params_name": "minimal",
            "auto_curation_rules_name": "none",
            "artifact_detection_params_name": None,
        },
    )
    expected_spike_times = {}

    def known_epoch_sorter(recording, sorting_id, **kwargs):
        sorting = _plant_three_epoch_units(recording, **kwargs)
        source_times = np.asarray(recording.get_times())
        # Capture physical event times at the deterministic sorter boundary,
        # independently of every downstream curation and matching reader.
        expected_spike_times[str(sorting_id)] = {
            int(unit): source_times[sorting.get_unit_spike_train(unit)].copy()
            for unit in sorting.unit_ids
        }
        return sorting

    plant_sorter(monkeypatch, known_epoch_sorter)
    runs, final_refs = [], []
    saved_matcher, match_key = None, None
    try:
        for row in selected:
            run = run_v2_pipeline(
                nwb_file_name=nwb_name,
                sort_group_id=row["sort_group_id"],
                interval_list_name=row["interval_list_name"],
                team_name=row["team_name"],
                pipeline_preset=params_name,
                auto_curate=True,
                require_units=True,
            )
            runs.append(run)
            automatic = run.auto_labeled_curation
            assert automatic is not None
            assert not CurationV2._labels_by_unit(automatic.as_key())
            assert sorted(
                map(
                    int, (CurationV2.Unit & automatic.as_key()).fetch("unit_id")
                )
            ) == [0, 1, 2]
            if reviewed:
                from spyglass.spikesorting.v2._review.operations import (
                    OPERATION_FILE,
                    run_operation,
                )
                from spyglass.spikesorting.v2.pipeline import FigPackReview
                from tests.spikesorting.v2.test_review_api_integration import (
                    _ensure_test_profile,
                    _write_edits,
                )

                review = run.start_review(_ensure_test_profile(), upload=False)
                assert review.parent == automatic
                with pytest.raises(ValueError, match="not completed"):
                    review.result()
                _write_edits(
                    review.uri, {0: ["accept"], 1: ["noise"], 2: ["mua"]}, []
                )
                changes = review.preview_import()
                assert changes.has_changes and not changes.merge_groups
                # Execute the same connected worker the browser's commit uses.
                run_operation(
                    review_id=review.review_id,
                    bundle=review.uri,
                    request={
                        "action": "commit",
                        "annotations_hash": changes.annotations_hash,
                    },
                    operation_id=f"epoch-{uuid.uuid4().hex}",
                )
                operation = json.loads(
                    (Path(review.uri) / OPERATION_FILE).read_text()
                )
                assert operation["status"] == "complete", operation
                final = FigPackReview.resume(review.review_id).result()
                assert final.parent == automatic
                final_refs.append(final)
            else:
                final_refs.append(automatic)

        seen_units, fed = [], []
        saved_matcher = install_fixture_pairer(
            monkeypatch,
            matcher_name=params_name,
            matcher_params_name=params_name,
            pairs=[[unit, unit] for unit in expected_units],
            seen_unit_ids=seen_units,
            fed=fed,
            read_bundles=True,
        )
        before = len(UnitMatchSelection())
        if not reviewed:
            accepted = plan_v2_unit_match_from_curations(
                final_refs, matcher_params_name=params_name
            )
            assert not accepted.ok
            assert any(
                "No units satisfy policy" in error for error in accepted.errors
            )
        policy_kwargs = (
            {} if unit_policy is None else {"unit_policy": unit_policy}
        )
        # Named order is deliberately reversed; source clocks determine feed order.
        plan = plan_v2_unit_match_from_curations(
            list(reversed(final_refs)),
            matcher_params_name=params_name,
            **policy_kwargs,
        )
        assert plan.ok, plan.errors
        assert plan.unit_policy == (unit_policy or "v2_accepted_single_units")
        assert plan.as_dataframe()["sorting_id"].tolist() == [
            str(ref.sorting_id) for ref in final_refs
        ]
        for selection in plan.unit_selections:
            assert list(selection.included_unit_ids) == expected_units
            if reviewed:
                assert selection.excluded_units == {
                    1: "denied label(s): noise",
                    2: "denied label(s): mua",
                }
        report = preflight_v2_unit_match(plan)
        assert report.ok, report.errors
        assert report.input_set_hash == plan.input_set_hash
        assert len(UnitMatchSelection()) == before

        # A whole-session sort cannot be matched with one of its own epochs.
        overlapping = plan_v2_unit_match_from_curations(
            [CurationRef.from_key(fx["curations"]["single_a"]), final_refs[0]],
            matcher_params_name=params_name,
            unit_policy="v2_unflagged_units",
        )
        assert not overlapping.ok
        assert any("raw samples" in error for error in overlapping.errors)
        with pytest.raises(PipelineInputError, match="raw samples"):
            run_v2_unit_match(overlapping)
        assert len(UnitMatchSelection()) == before

        summary = run_v2_unit_match(plan)
        match_key = {"unitmatch_id": summary["unit_match_id"]}
        assert summary["unit_match_status"] == "computed"
        assert summary["tracked_unit_status"] == "computed"
        assert summary["n_pairs"] == len(expected_units)
        assert summary["n_tracked_units"] == len(expected_units)
        assert seen_units == [expected_units, expected_units]
        assert [sorting_id for _, sorting_id in fed] == [
            str(ref.sorting_id) for ref in final_refs
        ]
        tracked = (TrackedUnit & match_key).fetch(as_dict=True)
        assert all(
            (row["n_matching_inputs"], row["n_sessions_detected"]) == (2, 1)
            for row in tracked
        )
        regions = TrackedUnit().get_unit_brain_regions(match_key)
        assert regions["detected"].all()
        assert set(regions["nwb_file_name"]) == {nwb_name}
        spikes = TrackedUnit().get_member_spike_times(match_key)
        assert len(spikes) == 2 * len(expected_units)
        assert set(spikes["unit_id"]) == set(expected_units)
        for _, member in spikes.iterrows():
            expected = expected_spike_times[str(member["sorting_id"])][
                int(member["unit_id"])
            ]
            assert len(member["spike_times"]) == len(expected) > 0
            np.testing.assert_array_equal(member["spike_times"], expected)
            epoch = epochs.loc[member["interval_list_name"]]
            assert np.all(member["spike_times"] >= epoch["start_time"])
            assert np.all(member["spike_times"] <= epoch["stop_time"])
        inputs, recordings = UnitMatch().get_input_provenance(match_key)
        nwb_inputs, nwb_recordings = UnitMatch().get_input_provenance(
            match_key, from_nwb=True
        )
        assert inputs.equals(nwb_inputs)
        assert recordings[list(nwb_recordings.columns)].equals(nwb_recordings)
        assert [
            row["included_unit_ids"] for row in inputs["selection_snapshot"]
        ] == [
            expected_units,
            expected_units,
        ]

        reordered = plan_v2_unit_match_from_curations(
            final_refs, matcher_params_name=params_name, **policy_kwargs
        )
        assert reordered.ok, reordered.errors
        assert reordered.input_set_hash == plan.input_set_hash
        reused = run_v2_unit_match(reordered)
        assert reused["unit_match_id"] == summary["unit_match_id"]
        assert (
            reused["unit_match_status"]
            == reused["tracked_unit_status"]
            == "reused"
        )
        assert len(UnitMatchSelection()) == before + 1
        assert seen_units == [expected_units, expected_units]
    finally:
        if match_key is not None:
            (UnitMatchSelection & match_key).super_delete(warn=False)
        (MatcherParameters & {"matcher_params_name": params_name}).super_delete(
            warn=False
        )
        if saved_matcher is not None:
            restore_matcher_registry(saved_matcher)
        for run in reversed(runs):
            sorting_key = {"sorting_id": run["sorting_id"]}
            clear_curations_for(sorting_key)
            (SortingSelection & sorting_key).super_delete(warn=False)
        presets._PIPELINE_PRESETS.pop(params_name, None)


def test_disjoint_epochs_match_without_inflating_detected_sessions(
    daily_concat_match_inputs, monkeypatch
):
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.exceptions import OverlappingMatchInputError
    from spyglass.spikesorting.v2.matcher_protocol import get_input_preparer
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection
    from spyglass.spikesorting.v2.unit_matching import (
        MatcherParameters,
        TrackedUnit,
        UnitMatch,
        UnitMatchSelection,
    )
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for
    from tests.spikesorting.v2._sorter_stub import plant_sorter
    from tests.spikesorting.v2._unitmatch_helpers import (
        install_fixture_pairer,
        restore_matcher_registry,
    )
    from tests.spikesorting.v2.conftest import _plant_spread_unit

    fx = daily_concat_match_inputs
    early = fx["curations"]["single_a_first"]
    plant_sorter(monkeypatch, _plant_spread_unit)
    late_sort = SortingSelection.insert_selection(
        {
            **fx["recording_keys"]["a_second"],
            "sorter": "mountainsort5",
            "sorter_params_name": "franklab_30khz_ms5_2026_06",
        }
    )
    Sorting.populate(late_sort, reserve_jobs=False)
    late = CurationV2.insert_curation(sorting_key=late_sort)
    params_name = "disjoint_epoch_pairer_params"
    saved = install_fixture_pairer(
        monkeypatch,
        matcher_name="disjoint_epoch_pairer",
        matcher_params_name=params_name,
        pairs=[[0, 0]],
    )

    def extract(directory, recording, sorting, **kwargs):
        Path(directory).mkdir(parents=True, exist_ok=True)
        return []

    monkeypatch.setattr(
        get_input_preparer("disjoint_epoch_pairer"), "extract", extract
    )
    pk = None
    try:
        # Reverse caller order: original acquisition position must win.
        pk = UnitMatchSelection.insert_inputs([late, early], params_name)
        assert (
            UnitMatchSelection.insert_inputs([early, late], params_name) == pk
        )
        recordings = (UnitMatchSelection.InputRecording & pk).fetch(
            as_dict=True, order_by="input_index"
        )
        assert len(recordings) == 2
        assert {r["nwb_file_name"] for r in recordings} == {
            fx["nwb_file_names"]["a"]
        }
        assert (
            recordings[0]["source_coverage"]["sample_ranges"][-1][1]
            <= recordings[1]["source_coverage"]["sample_ranges"][0][0]
        )
        with pytest.raises(OverlappingMatchInputError, match="raw samples"):
            UnitMatchSelection.insert_inputs(
                [fx["curations"]["single_a"], early], params_name
            )
        UnitMatch.populate(pk, reserve_jobs=False)
        TrackedUnit.populate(pk, reserve_jobs=False)
        tracked = (TrackedUnit & pk).fetch1()
        assert (
            tracked["n_matching_inputs"],
            tracked["n_sessions_detected"],
        ) == (2, 1)
        assert len(UnitMatch.Pair & pk) == 1
        db_inputs, db_recordings = UnitMatch().get_input_provenance(pk)
        nwb_inputs, nwb_recordings = UnitMatch().get_input_provenance(
            pk, from_nwb=True
        )
        assert db_inputs.equals(nwb_inputs)
        assert db_recordings[list(nwb_recordings.columns)].equals(
            nwb_recordings
        )
        assert all(
            snapshot["included_unit_ids"] == [0]
            and snapshot["policy_name"] == "v2_unflagged_units"
            and snapshot["selection_hash"]
            for snapshot in nwb_inputs["selection_snapshot"]
        )
        assert nwb_recordings["source_coverage"].tolist() == [
            row["source_coverage"] for row in recordings
        ]
        spikes = TrackedUnit().get_member_spike_times(pk)
        assert len(spikes) == 2
        assert len(set(spikes["interval_list_name"])) == 2
        for _, row in spikes.iterrows():
            expected = CurationV2.get_sorting(
                {
                    "sorting_id": row["sorting_id"],
                    "curation_id": row["curation_id"],
                },
                as_dataframe=True,
            ).loc[row["unit_id"], "spike_times"]
            np.testing.assert_array_equal(row["spike_times"], expected)
    finally:
        if pk is not None:
            (UnitMatchSelection & pk).super_delete(warn=False)
        (MatcherParameters & {"matcher_params_name": params_name}).super_delete(
            warn=False
        )
        restore_matcher_registry(saved)
        clear_curations_for(late_sort)
        (SortingSelection & late_sort).super_delete(warn=False)

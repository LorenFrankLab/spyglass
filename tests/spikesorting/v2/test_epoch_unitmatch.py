"""Match separately sorted, disjoint epochs on one original NWB clock."""

import importlib.util
import json
import os
import uuid
from pathlib import Path

import numpy as np
import pytest

from tests.spikesorting.v2._scenario_oracles import _raw_acquisition_clock

pytestmark = pytest.mark.pipeline

_FIGPACK_MISSING = (
    importlib.util.find_spec("figpack") is None
    or importlib.util.find_spec("figpack_spike_sorting") is None
)

# Stable neuron 0 becomes sorter unit 10 and fires more rapidly in the active
# epoch, with physiological 5 ms bursts. Neuron 1 is absent later; neuron 11 is
# first detected later. Their interspike intervals exceed 2 ms. Noise and pooled
# MUA are explicit nuisance streams; MUA's sub-2 ms pairs belong to two neurons.
_EPOCH_BOUNDS = {
    "epoch_z_rest": (0.4, 2.0),
    "epoch_a_active": (2.5, 4.5),
}
_EPOCH_SPIKE_FRAMES = {
    "epoch_z_rest": {
        0: (3300, 5400, 14100, 18300, 21600, 36600, 38100, 42900),
        1: (3900, 9600, 17400, 25800, 33900, 45600),
        2: (2700, 10200, 15900, 30900, 34200, 43800),
        3: (6600, 6618, 7200, 19500, 19515, 27000, 40800, 40812),
    },
    "epoch_a_active": {
        10: (
            2700,
            5400,
            5550,
            5910,
            11700,
            18600,
            24300,
            26400,
            30900,
            31050,
            31470,
            35700,
            42000,
            49200,
            51900,
            56400,
        ),
        11: (4200, 8700, 15300, 27300, 36300, 36450, 36810, 47400, 52800),
        12: (5100, 12900, 17100, 32400, 47100, 57000),
        13: (3300, 3312, 8700, 21900, 21918, 38700, 38712, 44100, 53700),
    },
}
_EPOCH_LABELS = {
    "epoch_z_rest": {0: ["accept"], 1: ["accept"], 2: ["noise"], 3: ["mua"]},
    "epoch_a_active": {
        10: ["accept"],
        11: ["accept"],
        12: ["noise"],
        13: ["mua"],
    },
}


@pytest.mark.integration
@pytest.mark.parametrize(
    "reviewed,unit_policy,expected_units",
    [
        pytest.param(
            True,
            None,
            ([0, 1], [10, 11]),
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
            ([0, 1, 2, 3], [10, 11, 12, 13]),
            id="auto_only_unflagged",
        ),
    ],
)
def test_public_epoch_matching_journey(
    dj_conn, tmp_path, monkeypatch, reviewed, unit_policy, expected_units
):
    """Carry known cells, changing activity and missing cells through the API.

    A small waveform-bearing raw NWB contains one stable neuron, an early-only
    neuron, a newly detected neuron, and explicit noise/MUA. Stable unit IDs
    differ between sorts. Only sorter and pair inference are substituted;
    preprocessing, evaluations, reviews, waveform extraction and persistence
    remain real. This tests workflow correctness; native recovery benchmarks
    establish sorter and matcher accuracy.
    """
    import spikeinterface as si

    from spyglass.common import IntervalList
    from spyglass.spikesorting.v2._orchestration import presets
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.exceptions import PipelineInputError
    from spyglass.spikesorting.v2.pipeline import (
        describe_epochs,
        plan_v2_unit_match_from_curations,
        preflight_v2_unit_match,
        register_pipeline_preset,
        run_v2_pipeline,
        run_v2_unit_match,
    )
    from spyglass.spikesorting.v2.unit_matching import (
        MatcherParameters,
        TrackedUnit,
        UnitMatch,
        UnitMatchSelection,
    )
    from tests.spikesorting.v2._biological_recordings import (
        write_biological_nwb,
    )
    from tests.spikesorting.v2._ingest_helpers import (
        _clean_session_v2,
        configure_v2_run_inputs,
        copy_and_insert_nwb,
    )
    from tests.spikesorting.v2._sorter_stub import plant_sorter
    from tests.spikesorting.v2._unitmatch_helpers import (
        install_fixture_pairer,
        restore_matcher_registry,
    )

    fs = 30_000
    interval_names = list(_EPOCH_BOUNDS)
    frames_by_epoch = {
        name: {
            unit: np.asarray(frames, dtype=np.int64)
            for unit, frames in units.items()
        }
        for name, units in _EPOCH_SPIKE_FRAMES.items()
    }
    footprints = {
        0: (90, 55, 18, 5),
        10: (90, 55, 18, 5),
        1: (8, 80, 35, 4),
        11: (4, 15, 85, 35),
    }
    spike_sources, artifact_sources = [], []
    for name in interval_names:
        origin = round(_EPOCH_BOUNDS[name][0] * fs)
        units = frames_by_epoch[name]
        for unit, frames in units.items():
            absolute = origin + frames
            if unit in footprints:
                assert np.all(np.diff(frames) >= 0.002 * fs)
                spike_sources.append((absolute, footprints[unit]))
            elif unit in (2, 12):
                artifact_sources.append((absolute, (45, 45, 45, 45)))
            else:
                assert np.any(np.diff(frames) < 0.002 * fs)
                # Two physical neurons each obey refractoriness; their pooled
                # MUA train has close pairs and alternating channel footprints.
                for events, footprint in (
                    (absolute[::2], (42, 23, 6, 2)),
                    (absolute[1::2], (3, 8, 25, 43)),
                ):
                    assert np.all(np.diff(events) >= 0.002 * fs)
                    spike_sources.append((events, footprint))
    assert (
        len(frames_by_epoch[interval_names[1]][10]) / 2
        > len(frames_by_epoch[interval_names[0]][0]) / 1.6
    )

    token = uuid.uuid4().hex[:10]
    params_name = f"epoch_biological_{token}"
    raw_path = write_biological_nwb(
        tmp_path / f"{params_name}.nwb",
        fixture_name=params_name,
        duration_s=5,
        spike_sources=spike_sources,
        artifact_sources=artifact_sources,
    )
    # Freeze all scientific expectations from this raw input and the declared
    # events before ingestion or any public pipeline call.
    raw_clock, source_object_id = _raw_acquisition_clock(raw_path)
    expected_coverage, expected_events = {}, {}
    for name, (lo, hi) in _EPOCH_BOUNDS.items():
        raw_frames = np.flatnonzero((raw_clock >= lo) & (raw_clock <= hi))
        assert len(raw_frames) and np.all(np.diff(raw_frames) == 1)
        expected_coverage[name] = {
            "version": 1,
            "source_object_id": source_object_id,
            "sample_ranges": [[int(raw_frames[0]), int(raw_frames[-1]) + 1]],
            "time_ranges": [
                [
                    float(raw_clock[raw_frames[0]]),
                    float(raw_clock[raw_frames[-1]]),
                ]
            ],
        }
        expected_events[name] = {
            unit: raw_clock[raw_frames[frames]].copy()
            for unit, frames in frames_by_epoch[name].items()
        }
    expected_components = [((0, 0), (1, 10)), ((0, 1),), ((1, 11),)]
    if not reviewed:
        expected_components += [((0, 2),), ((0, 3),), ((1, 12),), ((1, 13),)]

    nwb_name = copy_and_insert_nwb(raw_path)
    run_inputs = configure_v2_run_inputs(nwb_name, "v2_epoch_journey_team")
    session_key = {"nwb_file_name": nwb_name}
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
    active_frames = {}

    def known_sorter(**kwargs):
        frames = np.concatenate(list(active_frames.values()))
        labels = np.concatenate(
            [
                np.full(len(events), unit)
                for unit, events in active_frames.items()
            ]
        )
        order = np.argsort(frames)
        return si.NumpySorting.from_samples_and_labels(
            samples_list=[frames[order]],
            labels_list=[labels[order].astype(np.int32)],
            sampling_frequency=fs,
        )

    plant_sorter(monkeypatch, known_sorter)
    runs, final_refs = [], []
    saved_matcher, match_key = None, None
    try:
        for name, bounds in _EPOCH_BOUNDS.items():
            IntervalList.insert1(
                {
                    "nwb_file_name": nwb_name,
                    "interval_list_name": name,
                    "valid_times": np.asarray([bounds]),
                    "pipeline": "biological_epoch_journey",
                }
            )
        epochs = describe_epochs(
            nwb_name, sort_group_id=run_inputs["sort_group_id"]
        ).set_index("interval_list_name")
        for name, (lo, hi) in _EPOCH_BOUNDS.items():
            assert epochs.loc[name, "interval_status"] == "valid"
            assert epochs.loc[name, "interval_bounds"] == ((lo, hi),)
            assert epochs.loc[name, "start_time"] == lo
            assert epochs.loc[name, "stop_time"] == hi
            assert epochs.loc[name, "duration_s"] == pytest.approx(hi - lo)
            assert not epochs.loc[name, "ready_recording_keys"]

        for name in interval_names:
            active_frames = frames_by_epoch[name]
            run = run_v2_pipeline(
                **{**run_inputs, "interval_list_name": name},
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
            ) == sorted(frames_by_epoch[name])
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
                _write_edits(review.uri, _EPOCH_LABELS[name], [])
                changes = review.preview_import()
                assert changes.has_changes and not changes.merge_groups
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

        # The overlapping full-session input has a real, independently sorted
        # root curation. It need not be scored or reviewed to prove the guard.
        active_frames = {
            99: np.concatenate(
                [
                    round(_EPOCH_BOUNDS[interval_names[0]][0] * fs)
                    + frames_by_epoch[interval_names[0]][0],
                    round(_EPOCH_BOUNDS[interval_names[1]][0] * fs)
                    + frames_by_epoch[interval_names[1]][10],
                ]
            )
        }
        whole = run_v2_pipeline(
            **run_inputs, pipeline_preset=params_name, require_units=True
        )
        runs.append(whole)
        after_sorting = describe_epochs(
            nwb_name, sort_group_id=run_inputs["sort_group_id"]
        ).set_index("interval_list_name")
        assert all(
            after_sorting.loc[name, "ready_recording_keys"]
            for name in interval_names
        )
        names_by_sorting = {
            str(ref.sorting_id): name
            for ref, name in zip(final_refs, interval_names, strict=True)
        }
        populations = {
            name: units
            for name, units in zip(interval_names, expected_units, strict=True)
        }
        seen_units, fed = [], []
        saved_matcher = install_fixture_pairer(
            monkeypatch,
            matcher_name=params_name,
            matcher_params_name=params_name,
            pairs=[[0, 10]],
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
        # Names sort in the opposite order and references are passed backwards.
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
            name = names_by_sorting[str(selection.curation.sorting_id)]
            assert list(selection.included_unit_ids) == populations[name]
            assert selection.snapshot()["labels_by_unit"] == {
                str(unit): _EPOCH_LABELS[name][unit] if reviewed else []
                for unit in frames_by_epoch[name]
            }
            if reviewed:
                assert selection.excluded_units == (
                    {2: "denied label(s): noise", 3: "denied label(s): mua"}
                    if name == interval_names[0]
                    else {
                        12: "denied label(s): noise",
                        13: "denied label(s): mua",
                    }
                )
        report = preflight_v2_unit_match(plan)
        assert report.ok, report.errors
        assert report.input_set_hash == plan.input_set_hash
        assert len(UnitMatchSelection()) == before

        overlapping = plan_v2_unit_match_from_curations(
            [whole.root_curation, final_refs[0]],
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
        assert summary["n_pairs"] == 1
        assert summary["n_tracked_units"] == len(expected_components)
        assert seen_units == list(expected_units)
        assert [sorting_id for _, sorting_id in fed] == [
            str(ref.sorting_id) for ref in final_refs
        ]
        member_groups = {}
        for row in (TrackedUnit.Member & match_key).fetch(as_dict=True):
            member_groups.setdefault(row["tracked_unit_id"], set()).add(
                (
                    str(row["sorting_id"]),
                    int(row["curation_id"]),
                    int(row["unit_id"]),
                )
            )
        concrete_components = {
            frozenset(
                (
                    str(final_refs[index].sorting_id),
                    final_refs[index].curation_id,
                    unit,
                )
                for index, unit in members
            )
            for members in expected_components
        }
        assert {
            frozenset(members) for members in member_groups.values()
        } == concrete_components
        for tracked in (TrackedUnit & match_key).fetch(as_dict=True):
            assert tracked["n_matching_inputs"] == len(
                member_groups[tracked["tracked_unit_id"]]
            )
            assert tracked["n_sessions_detected"] == 1
        assert {
            (
                str(row["session_a_sorting_id"]),
                int(row["session_a_curation_id"]),
                int(row["unit_a_id"]),
                str(row["session_b_sorting_id"]),
                int(row["session_b_curation_id"]),
                int(row["unit_b_id"]),
            )
            for row in (UnitMatch.Pair & match_key).fetch(as_dict=True)
        } == {
            (
                str(final_refs[0].sorting_id),
                final_refs[0].curation_id,
                0,
                str(final_refs[1].sorting_id),
                final_refs[1].curation_id,
                10,
            )
        }

        expected_member_keys = {
            (str(ref.sorting_id), ref.curation_id, unit, index, name)
            for index, (ref, name) in enumerate(
                zip(final_refs, interval_names, strict=True)
            )
            for unit in populations[name]
        }
        regions = TrackedUnit().get_unit_brain_regions(match_key)
        assert regions["detected"].all()
        assert set(regions["nwb_file_name"]) == {nwb_name}
        for row in regions.to_dict(orient="records"):
            assert row["n_spikes"] == len(
                expected_events[row["interval_list_name"]][row["unit_id"]]
            )
        spikes = TrackedUnit().get_member_spike_times(match_key)
        assert len(spikes) == sum(map(len, expected_units))
        assert {
            (
                str(row["sorting_id"]),
                int(row["curation_id"]),
                int(row["unit_id"]),
                int(row["input_index"]),
                row["interval_list_name"],
            )
            for row in spikes.to_dict(orient="records")
        } == expected_member_keys
        for _, member in spikes.iterrows():
            expected = expected_events[member["interval_list_name"]][
                int(member["unit_id"])
            ]
            assert len(member["spike_times"]) == len(expected) > 0
            np.testing.assert_array_equal(member["spike_times"], expected)
        # Missing/new cells remain singletons: no fabricated zero-spike member
        # or borrowed identity is inserted in the other epoch.
        assert not any(
            row["input_index"] == 1
            and row["unit_id"] == 1
            or row["input_index"] == 0
            and row["unit_id"] == 11
            for row in spikes.to_dict(orient="records")
        )
        inputs, recordings = UnitMatch().get_input_provenance(match_key)
        nwb_inputs, nwb_recordings = UnitMatch().get_input_provenance(
            match_key, from_nwb=True
        )
        assert inputs.equals(nwb_inputs)
        assert recordings[list(nwb_recordings.columns)].equals(nwb_recordings)
        assert nwb_inputs["curation_uuid"].tolist() == [
            ref.curation_uuid for ref in final_refs
        ]
        assert recordings["interval_list_name"].tolist() == interval_names
        for recording in recordings.to_dict(orient="records"):
            coverage = expected_coverage[recording["interval_list_name"]]
            assert recording["source_coverage"] == coverage
            assert recording["start_sample"] == 0
            assert (
                recording["end_sample"]
                == coverage["sample_ranges"][0][1]
                - coverage["sample_ranges"][0][0]
            )
        assert [
            row["included_unit_ids"] for row in inputs["selection_snapshot"]
        ] == list(expected_units)

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
        assert seen_units == list(expected_units)
    finally:
        if match_key is not None:
            (UnitMatchSelection & match_key).super_delete(warn=False)
        (MatcherParameters & {"matcher_params_name": params_name}).super_delete(
            warn=False
        )
        if saved_matcher is not None:
            restore_matcher_registry(saved_matcher)
        presets._PIPELINE_PRESETS.pop(params_name, None)
        _clean_session_v2(session_key)


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

"""Notebook sorting, reviewed curation, and downstream population journeys."""

from __future__ import annotations

import importlib.util
import os
import uuid

import numpy as np
import pytest

_FIGPACK_MISSING = (
    importlib.util.find_spec("figpack") is None
    or importlib.util.find_spec("figpack_spike_sorting") is None
)

pytestmark = [pytest.mark.pipeline, pytest.mark.integration]


@pytest.mark.parametrize(
    "curation_path",
    [
        "manual",
        pytest.param(
            "browser",
            marks=pytest.mark.skipif(
                _FIGPACK_MISSING
                and not os.environ.get("SPYGLASS_REQUIRE_BROWSER_TESTS"),
                reason="requires the spikesorting-v2-curation extra (figpack)",
            ),
        ),
    ],
)
def test_pipeline_reviewed_generation_to_selected_analysis(
    dj_conn, tmp_path, monkeypatch, curation_path
):
    """Merge an oversplit bursting cell; exclude pooled MUA and artifact events.

    The raw signal contains each declared physical neuron's waveform. Only
    sorter inference is substituted; QC, merge/review, NWB writes and downstream
    population selection run through the real public workflow.
    """
    if curation_path == "browser" and _FIGPACK_MISSING:
        pytest.fail(
            "Browser review workflows are required in this lane; install "
            "the spikesorting-v2-curation extra (figpack)."
        )
    import spikeinterface as si

    from spyglass.spikesorting.analysis.v1.group import SortedSpikesGroup
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2._orchestration.presets import (
        _PIPELINE_PRESETS,
    )
    from spyglass.spikesorting.v2.curation_api import (
        CurationRef,
        save_manual_curation,
    )
    from spyglass.spikesorting.v2.pipeline import (
        preflight_v2_pipeline,
        register_pipeline_preset,
        run_v2_pipeline,
        select_units_for_analysis,
    )
    from tests.spikesorting.v2._ingest_helpers import (
        _clean_session_v2,
        configure_v2_run_inputs,
        copy_and_insert_nwb,
    )
    from tests.spikesorting.v2._sorter_stub import plant_sorter
    from tests.spikesorting.v2._scenario_oracles import _raw_acquisition_clock
    from tests.spikesorting.v2._biological_recordings import (
        write_biological_nwb,
    )

    # One irregular cell was oversplit into complementary clusters 0/1. Its
    # 3--10 ms bursts respect the declared 2 ms refractory/QC threshold.
    frames_by_unit = {
        0: (9000, 18000, 18210, 69000, 84000, 121500, 135000, 166500),
        1: (9150, 18120, 29400, 69300, 120000, 121620, 150900, 166710),
        # Two distinct neurons were pooled by the sorter. Each has a safe
        # refractory period; their combined cluster has eight sub-2 ms pairs.
        2: (
            15000,
            15018,
            39000,
            39030,
            57000,
            57042,
            81000,
            81027,
            105000,
            105036,
            129000,
            129021,
            147000,
            147033,
            171000,
            171045,
        ),
        # Broad correlated voltage transients, rather than neuronal pulses.
        3: (12000, 36000, 63000, 96000, 132000, 159000),
    }
    expected_cell_frames = np.array(
        [
            9000,
            9150,
            18000,
            18120,
            18210,
            29400,
            69000,
            69300,
            84000,
            120000,
            121500,
            121620,
            135000,
            150900,
            166500,
            166710,
        ],
        dtype=np.int64,
    )
    expected_labels = {2: ["mua"], 3: ["noise"], 4: ["accept"]}
    expected_selected_ids = (4,)  # documented fresh merged ID: max(0..3) + 1
    assert np.all(np.diff(expected_cell_frames) >= 90)
    assert np.count_nonzero(np.diff(frames_by_unit[2]) < 60) == 8
    assert np.all(np.diff(frames_by_unit[2][::2]) >= 60)
    assert np.all(np.diff(frames_by_unit[2][1::2]) >= 60)

    token = uuid.uuid4().hex[:12]
    raw_source = write_biological_nwb(
        tmp_path / f"user_curation_{token}.nwb",
        fixture_name=f"user_curation_{token}",
        duration_s=6.0,
        spike_sources=[
            (expected_cell_frames, [130.0, 65.0, 25.0, 10.0]),
            (np.asarray(frames_by_unit[2][::2]), [25.0, 110.0, 40.0, 15.0]),
            (np.asarray(frames_by_unit[2][1::2]), [15.0, 30.0, 120.0, 50.0]),
        ],
        artifact_sources=[
            (np.asarray(frames_by_unit[3]), [900.0, 820.0, 860.0, 760.0])
        ],
    )
    raw_clock, _ = _raw_acquisition_clock(raw_source)
    expected_spike_times = raw_clock[expected_cell_frames].copy()
    assert raw_clock[0] == 0.0
    assert raw_clock[1] == pytest.approx(1 / 30_000)

    # A distinct session and preset avoid changing shared scientific recipes.
    nwb_file_name = copy_and_insert_nwb(
        raw_source, dest_name=f"user_curation_{token}.nwb"
    )
    session = {"nwb_file_name": nwb_file_name}
    preset_name = f"user_curation_{token}"
    created_groups = []

    def known_sorter(**kwargs):
        return si.NumpySorting.from_samples_and_labels(
            samples_list=[np.concatenate(list(frames_by_unit.values()))],
            labels_list=[
                np.concatenate(
                    [
                        np.full(len(frames), unit, dtype=np.int32)
                        for unit, frames in frames_by_unit.items()
                    ]
                )
            ],
            sampling_frequency=30_000,
        )

    plant_sorter(monkeypatch, known_sorter)
    try:
        run_kwargs = configure_v2_run_inputs(
            nwb_file_name, "v2_user_curation_team"
        )
        register_pipeline_preset(
            preset_name,
            {
                "preprocessing_params_name": "default",
                "sorter": "mountainsort5",
                "sorter_params_name": "franklab_30khz_ms5_2026_06",
                "metric_params_name": "minimal",
                "auto_curation_rules_name": "none",
                "artifact_detection_params_name": None,
            },
        )
        run_kwargs.update(pipeline_preset=preset_name, auto_curate=True)
        report = preflight_v2_pipeline(**run_kwargs)
        assert report.ok, report.summary()
        run = run_v2_pipeline(**run_kwargs)
        assert run["n_units"] == 4
        assert run["sorting_status"] == "computed"
        parent = run.auto_labeled_curation
        assert parent is not None and parent.parent == run.root_curation

        empty = select_units_for_analysis(parent)
        created_groups.extend(dict(group.group_key) for group in empty.groups)
        assert empty.policy_name == "v2_accepted_single_units"
        assert empty.included_unit_ids == ()
        assert set(empty.excluded_units) == {0, 1, 2, 3}
        assert empty.fetch_spike_data(return_unit_ids=True) == ([], [])

        evaluation = parent.evaluate(
            metric_params_name="minimal", auto_curation_rules_name="none"
        )
        assert evaluation.curation == parent
        assert set(evaluation.metrics.index) == {0, 1, 2, 3}
        assert evaluation.metrics.loc[0, "isi_violation"] == 0
        assert evaluation.metrics.loc[1, "isi_violation"] == 0
        assert evaluation.metrics.loc[2, "isi_violation"] == pytest.approx(
            8 / 15
        )

        if curation_path == "manual":
            merged = evaluation.merge_and_evaluate([[0, 1]])
            merged_curation, merged_evaluation = (
                merged.child,
                merged.evaluation,
            )
            final = save_manual_curation(
                parent_curation=merged_curation, labels=expected_labels
            )
            assert final.operation_type.producer == "manual"
        else:
            from spyglass.spikesorting.v2._review.operations import (
                run_operation,
            )
            from spyglass.spikesorting.v2.review_api import FigPackReview
            from tests.spikesorting.v2.test_review_api_integration import (
                _ensure_test_profile,
                _write_edits,
            )

            review = run.start_review(_ensure_test_profile(), upload=False)
            assert review.parent == parent
            assert review.evaluation.curation == parent
            with pytest.raises(ValueError):
                review.result()
            _write_edits(
                review.uri,
                {0: ["accept"], 1: ["accept"], 2: ["mua"], 3: ["noise"]},
                [[0, 1]],
            )
            preview = review.preview_import()
            assert preview.has_changes and preview.merge_groups == ((0, 1),)
            # Execute the actual connected-browser commit worker, with saved
            # edits in its local bundle, rather than synthesizing completion.
            run_operation(
                review_id=review.review_id,
                bundle=review.uri,
                request={
                    "action": "commit",
                    "annotations_hash": preview.annotations_hash,
                },
                operation_id=f"user_curation_{token}",
            )
            with pytest.raises(ValueError):
                review.result()  # a merge still needs its explicit verification
            # The supported idempotent commit receipt reopens the exact child
            # whose inspection the connected browser is awaiting.
            committed = preview.commit()
            assert committed.needs_merge_verification
            merged_curation = committed.curation
            merged_evaluation = committed.evaluation
            verification = committed.continue_review()
            assert verification.parent == merged_curation
            verification_preview = verification.preview_import()
            assert not verification_preview.has_changes
            run_operation(
                review_id=verification.review_id,
                bundle=verification.uri,
                request={
                    "action": "commit",
                    "annotations_hash": verification_preview.annotations_hash,
                    "confirm_no_changes": True,
                },
                operation_id=f"user_curation_{token}_verification",
            )
            final = FigPackReview.resume(review.review_id).result()
            assert final == review.result()
            assert final.operation_type.producer == "figpack"

        assert merged_evaluation is not None
        assert set(merged_evaluation.metrics.index) == {2, 3, 4}
        assert merged_evaluation.metrics.loc[4, "isi_violation"] == 0
        assert merged_evaluation.metrics.loc[
            2, "isi_violation"
        ] == pytest.approx(8 / 15)
        assert merged_evaluation.metrics.loc[4, "firing_rate"] == pytest.approx(
            16 / 6
        )
        assert merged_evaluation.metrics.loc[4, "snr"] > 5
        assert final.parent == merged_curation
        assert final.commit_status == "committed"
        assert final.curation_uuid != parent.curation_uuid
        selected = select_units_for_analysis(final)
        created_groups.extend(
            dict(group.group_key) for group in selected.groups
        )
        assert selected.curation == final
        assert selected.included_unit_ids == expected_selected_ids
        assert set(selected.excluded_units) == {2, 3}
        verdicts = selected.describe()
        assert set(verdicts.index) == {2, 3, 4}
        for unit, labels in expected_labels.items():
            assert verdicts.loc[unit, "labels"] == labels
            assert bool(verdicts.loc[unit, "included"]) == (
                unit in expected_selected_ids
            )
        spikes, units = selected.fetch_spike_data(return_unit_ids=True)
        assert tuple(unit["unit_id"] for unit in units) == expected_selected_ids
        assert all(
            unit["spikesorting_merge_id"] == final.merge_id for unit in units
        )
        all_spikes = SpikeSortingOutput().get_spike_times(
            {"merge_id": final.merge_id}
        )
        assert len(all_spikes) == 3 and len(spikes) == 1
        assert len(spikes[0]) == len(expected_spike_times) == 16
        np.testing.assert_allclose(
            spikes[0], expected_spike_times, rtol=0, atol=1e-12
        )
        assert selected.observation.contains(spikes[0]).all()

        # A later sibling cannot silently replace the deliberately chosen child.
        save_manual_curation(parent_curation=parent, labels={0: ["reject"]})
        if curation_path == "browser":
            assert FigPackReview.resume(review.review_id).result() == final
        restored = CurationRef.from_key(
            {**final.as_key(), "curation_uuid": final.curation_uuid}
        )
        again = select_units_for_analysis(restored)
        assert again.groups[0].status == "reused"
        assert again.group_key == selected.group_key
        assert again.included_unit_ids == expected_selected_ids
        persisted_spikes, persisted_units = again.fetch_spike_data(
            return_unit_ids=True
        )
        assert (
            tuple(unit["unit_id"] for unit in persisted_units)
            == expected_selected_ids
        )
        np.testing.assert_allclose(
            persisted_spikes[0], expected_spike_times, rtol=0, atol=1e-12
        )
        assert empty.fetch_spike_data() == []
    finally:
        for key in created_groups:
            (SortedSpikesGroup & key).super_delete(warn=False)
        _PIPELINE_PRESETS.pop(preset_name, None)
        _clean_session_v2(session)

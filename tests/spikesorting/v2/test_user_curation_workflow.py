"""Notebook sorting, reviewed curation, and downstream population journeys."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
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
    dj_conn, smoke_nwb, monkeypatch, curation_path
):
    """Unreviewed output stays empty; the chosen reviewed child drives analysis."""
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
    from spyglass.spikesorting.v2.curation_api import save_manual_curation
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

    # A distinct session and preset avoid changing shared scientific recipes.
    token = uuid.uuid4().hex[:12]
    nwb_file_name = copy_and_insert_nwb(
        smoke_nwb, dest_name=f"user_curation_{token}.nwb"
    )
    session = {"nwb_file_name": nwb_file_name}
    preset_name = f"user_curation_{token}"
    created_groups = []
    expected_spike_times = {}

    def known_sorter(*, recording, **kwargs):
        frames = np.array(
            [
                500,
                1500,
                2500,
                3500,
                700,
                1700,
                2700,
                3700,
                900,
                1900,
                2900,
                3900,
            ],
            dtype=np.int64,
        )
        # Freeze the independent oracle before any sorting/curation NWB writes.
        # These four events per unit index the retained source recording clock.
        source_times = np.asarray(recording.get_times())
        for unit, unit_frames in enumerate(frames.reshape(3, 4)):
            expected_spike_times[unit] = source_times[unit_frames].copy()
        return si.NumpySorting.from_samples_and_labels(
            samples_list=[frames],
            labels_list=[np.repeat(np.arange(3, dtype=np.int32), 4)],
            sampling_frequency=recording.get_sampling_frequency(),
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
        assert run["n_units"] == 3
        assert run["sorting_status"] == "computed"
        parent = run.auto_labeled_curation
        assert parent is not None and parent.parent == run.root_curation

        empty = select_units_for_analysis(parent)
        created_groups.extend(dict(group.group_key) for group in empty.groups)
        assert empty.policy_name == "v2_accepted_single_units"
        assert empty.included_unit_ids == ()
        assert set(empty.excluded_units) == {0, 1, 2}
        assert empty.fetch_spike_data(return_unit_ids=True) == ([], [])

        labels = {0: ["accept"], 1: ["noise"], 2: ["mua"]}
        if curation_path == "manual":
            evaluation = parent.evaluate(
                metric_params_name="minimal", auto_curation_rules_name="none"
            )
            assert evaluation.curation == parent
            assert set(evaluation.metrics.index) == {0, 1, 2}
            final = save_manual_curation(parent_curation=parent, labels=labels)
            assert final.operation_type.producer == "manual"
        else:
            from spyglass.spikesorting.v2._review.operations import (
                OPERATION_FILE,
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
            with pytest.raises(ValueError, match="not completed"):
                review.result()
            _write_edits(review.uri, labels, [])
            preview = review.preview_import()
            assert preview.has_changes and not preview.merge_groups
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
            operation = json.loads(
                (Path(review.uri) / OPERATION_FILE).read_text()
            )
            assert operation["status"] == "complete", operation
            final = FigPackReview.resume(review.review_id).result()
            assert final == review.result()
            assert final.operation_type.producer == "figpack"

        assert final.parent == parent
        assert final.commit_status == "committed"
        assert final.curation_uuid != parent.curation_uuid
        selected = select_units_for_analysis(final)
        created_groups.extend(
            dict(group.group_key) for group in selected.groups
        )
        assert selected.curation == final
        assert selected.included_unit_ids == (0,)
        assert set(selected.excluded_units) == {1, 2}
        spikes, units = selected.fetch_spike_data(return_unit_ids=True)
        assert units == [
            {"spikesorting_merge_id": final.merge_id, "unit_id": 0}
        ]
        all_spikes = SpikeSortingOutput().get_spike_times(
            {"merge_id": final.merge_id}
        )
        assert len(all_spikes) == 3 and len(spikes) == 1
        assert len(spikes[0]) == len(expected_spike_times[0]) == 4
        np.testing.assert_array_equal(spikes[0], expected_spike_times[0])
        np.testing.assert_array_equal(spikes[0], all_spikes[0])
        assert selected.observation.contains(spikes[0]).all()
        stored = (
            SortedSpikesGroup.UnitSelection & dict(selected.group_key)
        ).fetch1()
        assert list(stored["selected_unit_ids"]) == [0]
        assert stored["selection_provenance"]["curation_uuid"] == str(
            final.curation_uuid
        )

        # A later sibling cannot silently replace the deliberately chosen child.
        save_manual_curation(parent_curation=parent, labels={0: ["reject"]})
        if curation_path == "browser":
            assert FigPackReview.resume(review.review_id).result() == final
        again = select_units_for_analysis(final)
        assert again.groups[0].status == "reused"
        assert again.group_key == selected.group_key
        assert again.included_unit_ids == (0,)
        assert empty.fetch_spike_data() == []
    finally:
        for key in created_groups:
            (SortedSpikesGroup & key).super_delete(warn=False)
        _PIPELINE_PRESETS.pop(preset_name, None)
        _clean_session_v2(session)

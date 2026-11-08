"""Browser-first review facade integration contracts."""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

_FIGPACK_MISSING = (
    importlib.util.find_spec("figpack") is None
    or importlib.util.find_spec("figpack_spike_sorting") is None
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.integration,
    pytest.mark.skipif(
        _FIGPACK_MISSING,
        reason="requires the spikesorting-v2-curation extra (figpack)",
    ),
]


def _write_edits(uri: str, labels: dict, groups: list[list[int]]) -> None:
    from spyglass.spikesorting.v2._review.annotations import (
        labels_and_merges_to_annotations,
    )

    (Path(uri) / "annotations.json").write_text(
        json.dumps(labels_and_merges_to_annotations(labels, groups))
    )


def _ensure_test_profile() -> str:
    from spyglass.spikesorting.v2.review_profile import CurationReviewProfile

    name = "test_browser_minimal_2026_09"
    CurationReviewProfile.insert1(
        {
            "review_profile_name": name,
            "metric_params_name": "minimal",
            "auto_curation_rules_name": "none",
            "displayed_unit_properties": [
                "snr",
                "isi_violation",
                "firing_rate",
            ],
            "label_options": ["accept", "mua", "noise"],
            "label_import_mode": "replace",
        },
        skip_duplicates=True,
    )
    return name


@pytest.mark.parametrize("inherited_label", ["artifact", "custom_cell"])
def test_browser_review_preview_commit_resume_and_continue(
    planted_two_unit_sort, curation_evaluation_defaults, inherited_label
):
    """The full review journey is pinned, pure, resumable, and merge-safe."""
    from unittest.mock import patch

    from spyglass.spikesorting.v2._review.annotations import (
        curation_annotations_to_labels_and_merges,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.curation_api import RunResult
    from spyglass.spikesorting.v2.exceptions import (
        ReviewChangedSincePreviewError,
        UnresolvedMergeLabelConflictError,
    )
    from spyglass.spikesorting.v2.figpack_curation import (
        _load_annotations_json,
    )
    from spyglass.spikesorting.v2.review_api import FigPackReview
    from spyglass.spikesorting.v2.sorting import Sorting
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    sorting_key = dict(planted_two_unit_sort)
    clear_curations_for(sorting_key)
    unit_ids = sorted(map(int, (Sorting.Unit & sorting_key).fetch("unit_id")))
    profile_name = _ensure_test_profile()
    try:
        root_key = CurationV2.create_initial_curation(
            sorting_key,
            labels={
                # Canonical and custom labels outside the profile palette
                # must survive both a no-change import and a resolved merge.
                unit_ids[0]: [inherited_label],
                unit_ids[1]: ["noise"],
            },
            allow_custom_labels=True,
        )
        run = RunResult(
            {
                "sorting_id": sorting_key["sorting_id"],
                "root_curation_id": root_key["curation_id"],
                "root_curation_uuid": (CurationV2 & root_key).fetch1(
                    "curation_uuid"
                ),
                "auto_labeled_curation_id": None,
                "auto_labeled_curation_uuid": None,
            }
        )
        with pytest.raises(ValueError, match="source='root'"):
            run.start_review(profile_name, source="auto_labeled")

        review = run.start_review(profile_name, source="root", upload=False)
        assert {stage.name: stage.status for stage in review.stages}[
            "verification_view_ready"
        ] == "computed"
        reused_review = run.start_review(
            profile_name, source="root", upload=False
        )
        assert {stage.name: stage.status for stage in reused_review.stages}[
            "verification_view_ready"
        ] == "reused"
        shutil.rmtree(review.uri)
        review = run.start_review(profile_name, source="root", upload=False)
        assert {stage.name: stage.status for stage in review.stages}[
            "verification_view_ready"
        ] == "computed"
        assert review.parent == run.root_curation
        assert review.profile.review_profile_name == profile_name
        assert review.evaluation.spec == review.profile.evaluation_spec
        assert (
            review.evaluation.missing_qc_inputs().eq("").all()
        )  # no enabled rules
        assert "0/2 units" in review.summary()
        resumed = FigPackReview.resume(review.review_id)
        assert resumed.review_id == review.review_id
        assert resumed.parent == review.parent
        assert resumed.profile == review.profile
        assert resumed.uri == review.uri

        figure_config = json.loads(
            (Path(review.uri) / "spyglass_curation.json").read_text()
        )
        assert figure_config["curation_uuid"] == str(
            review.parent.curation_uuid
        )
        assert figure_config["review"]["profile_hash"] == (
            review.profile.profile_hash
        )
        # The display budget is part of the review's identity and resumes
        # with it.
        assert figure_config["review"]["display"] == (
            review.display_options.as_dict()
        )
        assert resumed.display_options == review.display_options
        bounded = run.start_review(
            profile_name,
            source="root",
            upload=False,
            display_options={
                "max_amplitudes_per_unit": 5,
                "amplitude_sampling_seed": 3,
            },
        )
        assert bounded.review_id != review.review_id
        assert bounded.display_options.max_amplitudes_per_unit == 5
        assert (
            FigPackReview.resume(bounded.review_id).display_options
            == bounded.display_options
        )
        # Identical display options on a rebuilt bundle sample identically.
        bounded_again = run.start_review(
            profile_name,
            source="root",
            upload=False,
            display_options={
                "max_amplitudes_per_unit": 5,
                "amplitude_sampling_seed": 3,
            },
        )
        assert bounded_again.review_id == bounded.review_id
        # Continuing carries the display budget onto the child's review.
        continued = (
            bounded.preview_import()
            .commit(confirm_no_changes=True)
            .continue_review()
        )
        assert continued.display_options == bounded.display_options
        assert continued.display_options.max_amplitudes_per_unit == 5
        assert figure_config["review"]["evaluation_spec"] == {
            "metric_params_name": "minimal",
            "auto_curation_rules_name": "none",
        }

        pristine_bytes = (Path(review.uri) / "annotations.json").read_bytes()
        children_before = len(review.parent.children)
        connection = CurationV2.connection
        with patch.object(connection, "query", wraps=connection.query) as query:
            no_change = review.preview_import()
        for table in (CurationV2.Unit, CurationV2.UnitLabel):
            assert (
                sum(
                    call.args[0].lstrip().upper().startswith("SELECT")
                    and table.full_table_name in call.args[0]
                    for call in query.call_args_list
                )
                == 1
            )
        assert no_change.reviewed_parent_created_at == review.parent.created_at
        assert no_change.reviewed_parent_created_by == review.parent.created_by
        assert dict(no_change.labels_before) == dict(no_change.labels_after)
        assert no_change.labels_after[unit_ids[0]] == (inherited_label,)
        assert no_change.merge_groups == ()
        assert len(review.parent.children) == children_before
        assert (Path(review.uri) / "annotations.json").read_bytes() == (
            pristine_bytes
        )
        _write_edits(
            review.uri,
            {unit_ids[0]: [inherited_label], unit_ids[1]: [inherited_label]},
            [],
        )
        with pytest.raises(ValueError, match="new labels outside"):
            review.preview_import()
        (Path(review.uri) / "annotations.json").write_bytes(pristine_bytes)
        with pytest.raises(ValueError, match="confirm_no_changes"):
            no_change.commit()
        no_change_receipt = no_change.commit(confirm_no_changes=True)
        assert CurationV2._labels_by_unit(
            no_change_receipt.curation.as_key()
        ) == {
            unit_ids[0]: [inherited_label],
            unit_ids[1]: ["noise"],
        }
        assert no_change_receipt.curation.parent == review.parent
        assert not no_change_receipt.needs_merge_verification

        first_edits = {
            unit_ids[0]: [inherited_label],
            unit_ids[1]: ["noise"],
        }
        _write_edits(review.uri, first_edits, [unit_ids])
        stale_preview = review.preview_import()
        assert stale_preview.label_conflicts
        _write_edits(
            review.uri,
            {unit_ids[0]: ["accept"], unit_ids[1]: ["accept"]},
            [unit_ids],
        )
        with pytest.raises(ReviewChangedSincePreviewError):
            stale_preview.commit(
                conflict_resolutions={max(unit_ids) + 1: ("accept",)}
            )

        _write_edits(review.uri, first_edits, [unit_ids])
        changes = review.preview_import()
        assert changes.unit_count_before == 2
        assert changes.unit_count_after == 1
        assert changes.merge_groups == (tuple(unit_ids),)
        assert changes.label_conflicts[0].merged_unit_id == max(unit_ids) + 1
        assert no_change_receipt.curation in changes.newer_sibling_curations
        with pytest.raises(UnresolvedMergeLabelConflictError):
            changes.commit()

        children_before = len(review.parent.children)
        for bad_id in (max(unit_ids) + 1.9, True, str(max(unit_ids) + 1)):
            with pytest.raises(ValueError, match="unit_id must be an integer"):
                changes.commit(conflict_resolutions={bad_id: ("accept",)})
        with pytest.raises(ValueError, match="new labels outside"):
            changes.commit(
                conflict_resolutions={max(unit_ids) + 1: ("new_custom_cell",)}
            )
        assert len(review.parent.children) == children_before

        receipt = changes.commit(
            conflict_resolutions={max(unit_ids) + 1: (inherited_label,)}
        )
        assert receipt.curation.parent == review.parent
        assert receipt.curation != no_change_receipt.curation
        assert receipt.needs_merge_verification
        assert receipt.evaluation is not None
        assert receipt.evaluation.spec == review.profile.evaluation_spec
        assert receipt.curation.merge_id is not None
        assert receipt.curation.member_merge_ids == {}
        assert {child.curation_id for child in review.parent.children} >= {
            receipt.curation.curation_id,
            no_change_receipt.curation.curation_id,
        }

        continuation = receipt.continue_review()
        assert continuation.parent == receipt.curation
        assert continuation.profile == review.profile
        assert continuation.evaluation == receipt.evaluation
        assert set(continuation.evaluation.missing_qc_inputs().index) == set(
            continuation.evaluation.metrics.index
        )
        resumed_continuation = FigPackReview.resume(continuation.review_id)
        assert resumed_continuation.parent == continuation.parent
        assert resumed_continuation.profile == continuation.profile
        labels, pending_merges = curation_annotations_to_labels_and_merges(
            _load_annotations_json(continuation.uri)
        )
        assert pending_merges == []
        assert labels == {max(unit_ids) + 1: [inherited_label]}
    finally:
        clear_curations_for(sorting_key)


def test_worker_retry_reuses_committed_child_after_evaluation_failure(
    planted_two_unit_sort, curation_evaluation_defaults, monkeypatch
):
    """A failed merge evaluation can resume without committing another child."""
    from spyglass.spikesorting.v2._review.operations import (
        OPERATION_FILE,
        RESULT_FILE,
        run_operation,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.curation_api import CurationRef
    from spyglass.spikesorting.v2.review_api import FigPackReview
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    sorting_key = dict(planted_two_unit_sort)
    clear_curations_for(sorting_key)
    try:
        root = CurationRef.from_key(CurationV2.insert_curation(sorting_key))
        units = sorted(
            map(int, (CurationV2.Unit & root.as_key()).fetch("unit_id"))
        )
        review = root.start_review(_ensure_test_profile())
        _write_edits(review.uri, {unit: ["accept"] for unit in units}, [units])
        request = {
            "action": "commit",
            "annotations_hash": review.preview_import().annotations_hash,
        }
        original_evaluate = CurationRef.evaluate
        evaluated = []

        def evaluate(self, **kwargs):
            evaluated.append(self.curation_uuid)
            if len(evaluated) == 1:
                raise RuntimeError("Injected evaluation failure after commit")
            return original_evaluate(self, **kwargs)

        monkeypatch.setattr(CurationRef, "evaluate", evaluate)
        run_operation(
            review_id=review.review_id,
            bundle=review.uri,
            request=request,
            operation_id="first",
        )
        bundle = Path(review.uri)
        pending = json.loads((bundle / RESULT_FILE).read_text())
        state = json.loads((bundle / OPERATION_FILE).read_text())
        assert state["status"] == "failed"
        assert "Injected evaluation failure" in state["message"]
        assert pending["pending"]
        (child,) = root.children
        assert pending["curation"]["curation_uuid"] == str(child.curation_uuid)
        assert len(CurationV2 & sorting_key) == 2  # root + durable child
        with pytest.raises(ValueError, match="not completed"):
            review.result()

        # Resume from persisted review metadata; do not reuse a mocked receipt.
        resumed = FigPackReview.resume(review.review_id)
        run_operation(
            review_id=resumed.review_id,
            bundle=resumed.uri,
            request=request,
            operation_id="retry",
        )
        completed = json.loads((bundle / RESULT_FILE).read_text())
        assert (
            json.loads((bundle / OPERATION_FILE).read_text())["status"]
            == "complete"
        )
        assert not completed["pending"]
        assert completed["curation"] == pending["curation"]
        assert len(CurationV2 & sorting_key) == 2
        assert root.children == (child,)
        assert len(evaluated) >= 2 and set(evaluated) == {child.curation_uuid}
        verification = FigPackReview.resume(completed["verification_review_id"])
        assert verification.parent == child
        assert verification.evaluation.curation == child
        with pytest.raises(ValueError, match="not completed"):
            resumed.result()

        run_operation(
            review_id=verification.review_id,
            bundle=verification.uri,
            request={
                "action": "commit",
                "annotations_hash": verification.preview_import().annotations_hash,
                "confirm_no_changes": True,
            },
            operation_id="verify",
        )
        assert resumed.result().parent == child
        assert root.children == (child,)
    finally:
        clear_curations_for(sorting_key)


def _non_default_metric_waveform_recipe(curation) -> str:
    """A metric waveform recipe other than the curation's default one."""
    from spyglass.spikesorting.v2.metric_curation import (
        CurationEvaluationSelection,
    )

    default_key = CurationEvaluationSelection.insert_by_curation_id(
        curation.sorting_id, curation.curation_id, "minimal", "none"
    )
    default_recipe = (CurationEvaluationSelection & default_key).fetch1(
        "metric_waveform_params_name"
    )
    return next(
        name
        for name in (
            "franklab_hippocampus_metric_waveforms",
            "franklab_cortex_metric_waveforms",
        )
        if name != default_recipe
    )


def _evaluate_with_recipe(curation, recipe: str) -> dict:
    from spyglass.spikesorting.v2.metric_curation import (
        CurationEvaluation,
        CurationEvaluationSelection,
    )

    key = CurationEvaluationSelection.insert_selection(
        {
            **curation.as_key(),
            "metric_params_name": "minimal",
            "auto_curation_rules_name": "none",
            "metric_waveform_params_name": recipe,
        }
    )
    CurationEvaluation.populate(key, reserve_jobs=False)
    return key


def test_start_review_reports_the_evaluation_it_displays(
    planted_two_unit_sort, curation_evaluation_defaults
):
    """The evaluation stage describes the evaluation the review shows.

    An evaluation of the parent under another metric waveform recipe is a
    different evaluation: it must not make the review's own evaluation look
    reused. A supplied, populated evaluation is reused, not computed.
    """
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.curation_api import (
        CurationRef,
        EvaluationResult,
    )
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    sorting_key = dict(planted_two_unit_sort)
    clear_curations_for(sorting_key)
    try:
        root = CurationRef.from_key(
            CurationV2.create_initial_curation(sorting_key)
        )
        profile = _ensure_test_profile()
        other_key = _evaluate_with_recipe(
            root, _non_default_metric_waveform_recipe(root)
        )

        review = root.start_review(profile, upload=False)
        assert str(review.evaluation.evaluation_id) != str(
            other_key["curation_evaluation_id"]
        )
        assert {stage.name: stage.status for stage in review.stages}[
            "evaluation_populated"
        ] == "computed"

        supplied = EvaluationResult.from_key(other_key).start_review(
            profile, upload=False
        )
        assert {stage.name: stage.status for stage in supplied.stages}[
            "evaluation_populated"
        ] == "reused"
    finally:
        clear_curations_for(sorting_key)


def test_review_commit_evaluates_the_merged_child_with_the_review_recipe(
    planted_two_unit_sort, curation_evaluation_defaults
):
    """A committed merge is evaluated with the reviewed evaluation's recipe.

    A review can display an evaluation with an explicit metric waveform
    recipe. The merged child's evaluation must use the same recipe, or its
    metrics come from different waveform windows than the ones reviewed.
    """
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.curation_api import (
        CurationRef,
        EvaluationResult,
    )
    from spyglass.spikesorting.v2.metric_curation import (
        CurationEvaluationSelection,
    )
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    sorting_key = dict(planted_two_unit_sort)
    clear_curations_for(sorting_key)
    try:
        root = CurationRef.from_key(
            CurationV2.create_initial_curation(sorting_key)
        )
        units = sorted(
            map(int, (CurationV2.Unit & root.as_key()).fetch("unit_id"))
        )
        recipe = _non_default_metric_waveform_recipe(root)
        review = EvaluationResult.from_key(
            _evaluate_with_recipe(root, recipe)
        ).start_review(_ensure_test_profile(), upload=False)
        _write_edits(review.uri, {unit: ["accept"] for unit in units}, [units])

        receipt = review.preview_import().commit()

        assert receipt.evaluation.curation == receipt.curation
        child_selection = CurationEvaluationSelection & {
            "curation_evaluation_id": receipt.evaluation.evaluation_id
        }
        assert child_selection.fetch1("metric_waveform_params_name") == recipe
    finally:
        clear_curations_for(sorting_key)


def test_continue_review_after_label_commit_keeps_the_review_recipe(
    planted_two_unit_sort, curation_evaluation_defaults
):
    """The follow-up review of a label-only commit uses the reviewed recipe.

    A label-only commit carries no child evaluation, so ``continue_review``
    evaluates the child itself. It must use the reviewed evaluation's metric
    waveform recipe, not the sort's default one.
    """
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.curation_api import (
        CurationRef,
        EvaluationResult,
    )
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    sorting_key = dict(planted_two_unit_sort)
    clear_curations_for(sorting_key)
    try:
        root = CurationRef.from_key(
            CurationV2.create_initial_curation(sorting_key)
        )
        units = sorted(
            map(int, (CurationV2.Unit & root.as_key()).fetch("unit_id"))
        )
        recipe = _non_default_metric_waveform_recipe(root)
        review = EvaluationResult.from_key(
            _evaluate_with_recipe(root, recipe)
        ).start_review(_ensure_test_profile(), upload=False)
        _write_edits(review.uri, {unit: ["accept"] for unit in units}, [])
        receipt = review.preview_import().commit()
        assert receipt.evaluation is None

        child_review = receipt.continue_review()

        assert child_review.evaluation.curation == receipt.curation
        assert child_review.evaluation._metric_waveform_params_name() == recipe
    finally:
        clear_curations_for(sorting_key)


def test_start_review_evaluates_with_a_requested_recipe(
    planted_two_unit_sort, curation_evaluation_defaults
):
    """``start_review`` without an evaluation can name the waveform recipe.

    Reopening a parent from one of its reviews evaluates it this way, so the
    parent review shows metrics from the same waveform windows.
    """
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.curation_api import CurationRef
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    sorting_key = dict(planted_two_unit_sort)
    clear_curations_for(sorting_key)
    try:
        root = CurationRef.from_key(
            CurationV2.create_initial_curation(sorting_key)
        )
        recipe = _non_default_metric_waveform_recipe(root)
        other_key = _evaluate_with_recipe(root, recipe)

        review = root.start_review(
            _ensure_test_profile(),
            upload=False,
            metric_waveform_params_name=recipe,
        )

        assert str(review.evaluation.evaluation_id) == str(
            other_key["curation_evaluation_id"]
        )
        assert {stage.name: stage.status for stage in review.stages}[
            "evaluation_populated"
        ] == "reused"
    finally:
        clear_curations_for(sorting_key)

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
    from spyglass.spikesorting.v2._figpack_curation import (
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

    from spyglass.spikesorting.v2._figpack_curation import (
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
        # The display budget is persisted with the review, resumes with it,
        # and is part of its identity: different display options are a
        # DIFFERENT review of the same parent/profile, not a silent reuse.
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
        # Committing (no edits, confirmed) and CONTINUING the bounded review
        # carries its display budget onto the child's review.
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
        assert (
            no_change_receipt.created_at
            == no_change_receipt.curation.created_at
        )
        assert (
            no_change_receipt.created_by
            == no_change_receipt.curation.created_by
        )

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


def test_review_explicit_annotation_set_identity_and_display(
    planted_two_unit_sort, curation_evaluation_defaults
):
    """Selected custom properties are hashed, resumable, and displayed."""
    import pandas as pd

    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.curation_api import CurationRef
    from spyglass.spikesorting.v2.review_api import FigPackReview
    from spyglass.spikesorting.v2.sorting import Sorting
    from spyglass.spikesorting.v2.unit_annotation import (
        CurationUnitAnnotationSet,
        UnitAnnotationDefinition,
    )
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    sorting_key = dict(planted_two_unit_sort)
    clear_curations_for(sorting_key)
    try:
        root = CurationRef.from_key(
            CurationV2.create_initial_curation(sorting_key)
        )
        unit_ids = sorted(
            map(int, (Sorting.Unit & sorting_key).fetch("unit_id"))
        )
        definition = UnitAnnotationDefinition.insert_definition(
            "phase4_figpack_score", 1, "float"
        )
        annotation_set = CurationUnitAnnotationSet.from_dataframe(
            root,
            definition,
            pd.DataFrame(
                {"phase4_figpack_score": [0.25, 0.75]},
                index=pd.Index(unit_ids, name="unit_id"),
            ),
            producer="phase4-review-test",
        )
        profile = _ensure_test_profile()

        baseline = root.start_review(profile, upload=False)
        explicit_empty = root.start_review(
            profile, upload=False, annotation_sets=[]
        )
        assert explicit_empty.review_id == baseline.review_id

        selected = root.start_review(
            profile,
            upload=False,
            annotation_sets=[annotation_set],
        )
        assert selected.review_id != baseline.review_id
        assert selected.annotation_sets == (annotation_set,)
        resumed = FigPackReview.resume(selected.review_id)
        assert resumed.annotation_sets == (annotation_set,)

        config = json.loads(
            (Path(selected.uri) / "spyglass_curation.json").read_text()
        )
        snapshot = config["review"]["annotation_sets"][0]
        assert snapshot["set_hash"] == annotation_set.set_hash
        assert snapshot["curation_uuid"] == str(root.curation_uuid)

        serialized = "\n".join(
            path.read_text() for path in Path(selected.uri).rglob("*.json")
        )
        assert annotation_set.column_name in serialized
    finally:
        clear_curations_for(sorting_key)

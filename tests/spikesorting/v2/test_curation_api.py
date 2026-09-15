"""Identity-safe curation facade, receipts, and lifecycle contracts."""

from __future__ import annotations

import inspect

import pytest
import uuid


def test_public_curation_api_is_reexported():
    """Facade value objects are discoverable beside the pipeline runner."""
    from spyglass.spikesorting.v2 import pipeline
    from spyglass.spikesorting.v2.curation_api import (
        CurationRef,
        EvaluationResult,
        EvaluationSpec,
        MergeEvaluateReceipt,
        RunResult,
    )
    from spyglass.spikesorting.v2.review_api import (
        CurationChangeSet,
        FigPackReview,
        ReviewImportReceipt,
        ReviewProfileRef,
    )

    assert pipeline.CurationRef is CurationRef
    assert pipeline.EvaluationResult is EvaluationResult
    assert pipeline.EvaluationSpec is EvaluationSpec
    assert pipeline.MergeEvaluateReceipt is MergeEvaluateReceipt
    assert pipeline.RunResult is RunResult
    assert pipeline.ReviewProfileRef is ReviewProfileRef
    assert pipeline.FigPackReview is FigPackReview
    assert pipeline.CurationChangeSet is CurationChangeSet
    assert pipeline.ReviewImportReceipt is ReviewImportReceipt


def test_non_root_facade_requires_typed_parent():
    """Child facade operations expose no numeric root-sentinel escape hatch."""
    from spyglass.spikesorting.v2.curation_api import (
        commit_merges,
        merge_and_evaluate,
        preview_merges,
        save_manual_curation,
    )

    for function in (
        commit_merges,
        merge_and_evaluate,
        preview_merges,
        save_manual_curation,
    ):
        parameter = inspect.signature(function).parameters["parent_curation"]
        assert parameter.default is inspect.Parameter.empty
        with pytest.raises(
            TypeError, match="requires parent_curation=CurationRef"
        ):
            if function is save_manual_curation:
                function(parent_curation={"curation_id": -1})
            elif function is merge_and_evaluate:
                function(
                    parent_curation={"curation_id": -1},
                    spec=None,
                    groups=[[0, 1]],
                )
            else:
                function(parent_curation={"curation_id": -1}, groups=[[0, 1]])


def test_merge_group_normalization_is_lossless():
    """Malformed merge requests fail loudly instead of selecting other units.

    ``int()`` truncated ``1.9`` to ``1``, accepted ``True`` as ``1`` and split
    the string group ``"12"`` into units 1 and 2 -- all of which could pass
    membership checks as a merge the caller never asked for. Only integers
    (Python or NumPy) in non-string containers are accepted; the shape checks
    run before any database read.
    """
    import numpy as np

    from spyglass.spikesorting.v2.curation_api import (
        _lossless_int,
        _normalize_merge_groups,
    )

    assert _normalize_merge_groups([[1, 2], (3, 4)]) == [[1, 2], [3, 4]]
    assert _normalize_merge_groups([np.array([5, 6], dtype=np.int64)]) == [
        [5, 6]
    ]
    assert _normalize_merge_groups([[np.int32(7), np.uint8(8)]]) == [[7, 8]]
    for malformed, match in (
        ([[1.9, 2.9]], "unit id must be an integer"),
        ([[True, 2]], "unit id must be an integer"),
        ([["1", "2"]], "unit id must be an integer"),
        (["12"], "sequence of unit ids"),
        ("12", "sequence of unit-id sequences"),
        ([{"a": 1, "b": 2}], "sequence of unit ids"),
        ([], "at least one group"),
        ([[1]], "at least two"),
        ([[1, 1]], "duplicate"),
        ([[1, 2], [2, 3]], "disjoint"),
    ):
        with pytest.raises(ValueError, match=match):
            _normalize_merge_groups(malformed)
    assert _lossless_int(np.int64(3), "curation_id") == 3
    with pytest.raises(ValueError, match="curation_id must be an integer"):
        _lossless_int(3.5, "curation_id")


@pytest.mark.slow
@pytest.mark.integration
def test_curation_ref_state_operation_lineage_and_merge_id(
    planted_three_unit_sort,
):
    """Lifecycle fields are orthogonal and merge ids resolve the exact child."""
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.exceptions import CurationNotFoundError
    from spyglass.spikesorting.v2.curation_api import (
        CurationRef,
        RunResult,
        create_initial_curation,
        save_manual_curation,
    )
    from spyglass.spikesorting.v2.sorting import Sorting

    sorting_key = dict(planted_three_unit_sort)
    unit_ids = sorted(map(int, (Sorting.Unit & sorting_key).fetch("unit_id")))
    clear_curations_for(sorting_key)
    try:
        root = create_initial_curation(sorting_key)
        labeled = save_manual_curation(
            parent_curation=root,
            labels={unit_ids[0]: ["mua"]},
            description="typed label child",
        )
        preview = labeled.preview_merges([[unit_ids[1], unit_ids[2]]])
        merged = labeled.commit_merges([[unit_ids[1], unit_ids[2]]])

        assert root.is_root and root.commit_status == "committed"
        assert not root.is_leaf and root.has_committed_children
        assert labeled.operation_type.producer == "manual"
        assert labeled.operation_type.change_kind == "label"
        assert preview.commit_status == "preview"
        assert preview.operation_type.change_kind == "merge"
        assert merged.commit_status == "committed"
        assert merged.operation_type.change_kind == "merge"
        assert not hasattr(root, "superseded")

        assert [ref.curation_id for ref in merged.lineage()] == [
            root.curation_id,
            labeled.curation_id,
            merged.curation_id,
        ]
        tree = merged.visualize_lineage()
        assert f"* curation {merged.curation_id}" in tree
        assert "manual/merge" in tree

        expected_merge = (
            SpikeSortingOutput.CurationV2 & merged.as_key()
        ).fetch1("merge_id")
        assert merged.merge_id == expected_merge
        assert merged.merge_id != root.merge_id

        run = RunResult(
            {
                "sorting_id": root.sorting_id,
                "root_curation_id": root.curation_id,
                "root_curation_uuid": root.curation_uuid,
                "auto_labeled_curation_id": None,
                "auto_labeled_curation_uuid": None,
            }
        )
        assert run["sorting_id"] == root.sorting_id
        assert run.root_curation == root
        assert run.auto_labeled_curation is None
        run["auto_labeled_curation_id"] = merged.curation_id
        run["auto_labeled_curation_uuid"] = merged.curation_uuid
        assert run.auto_labeled_curation == merged
        # The receipt pins the GENERATION: a receipt carrying a stale uuid for
        # the same numeric id must not resolve to a replacement row.
        stale_receipt = RunResult(
            {
                "sorting_id": root.sorting_id,
                "root_curation_id": merged.curation_id,
                "root_curation_uuid": uuid.uuid4(),
                "auto_labeled_curation_id": None,
                "auto_labeled_curation_uuid": None,
            }
        )
        with pytest.raises(CurationNotFoundError):
            stale_receipt.root_curation

        # Delete and recreate the highest numeric id: the old ref must reject
        # the replacement generation even though the DataJoint PK is reused.
        stale = save_manual_curation(
            parent_curation=root, description="generation one"
        )
        stale_id = stale.curation_id
        (CurationV2 & stale.as_key()).delete(safemode=False)
        replacement = save_manual_curation(
            parent_curation=root, description="generation two"
        )
        assert replacement.curation_id == stale_id
        assert replacement.curation_uuid != stale.curation_uuid
        from spyglass.spikesorting.v2.exceptions import CurationNotFoundError

        with pytest.raises(CurationNotFoundError, match="was replaced"):
            stale.as_key()
        # Serializing a ref preserves its identity guarantee: a mapping that
        # carries the stale curation_uuid is refused too, a mapping carrying
        # the replacement's uuid resolves it, and a bare numeric key resolves
        # the CURRENT generation.
        from dataclasses import asdict

        with pytest.raises(CurationNotFoundError, match="stale curation_uuid"):
            CurationRef.from_key(asdict(stale))
        assert CurationRef.from_key(asdict(replacement)) == replacement
        assert (
            CurationRef.from_key(
                {"sorting_id": root.sorting_id, "curation_id": stale_id}
            )
            == replacement
        )
    finally:
        clear_curations_for(sorting_key)


@pytest.mark.slow
@pytest.mark.integration
def test_subtree_delete_is_leaf_up(planted_three_unit_sort):
    """The supported subtree delete leaves no dangling lineage edges."""
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.curation_api import (
        create_initial_curation,
        save_manual_curation,
    )
    from spyglass.spikesorting.v2.exceptions import CurationNotFoundError
    from spyglass.spikesorting.v2.sorting import Sorting

    sorting_key = dict(planted_three_unit_sort)
    unit_ids = sorted(map(int, (Sorting.Unit & sorting_key).fetch("unit_id")))
    clear_curations_for(sorting_key)
    try:
        root = create_initial_curation(sorting_key)
        child = save_manual_curation(
            parent_curation=root, labels={unit_ids[0]: ["mua"]}
        )
        grandchild = child.commit_merges([[unit_ids[1], unit_ids[2]]])

        preview = root.preview_curation_delete()
        assert [ref.curation_id for ref in preview.leaf_first] == [
            grandchild.curation_id,
            child.curation_id,
            root.curation_id,
        ]
        receipt = root.delete_subtree(safemode=False)
        assert receipt.count == 3
        assert not (CurationV2 & sorting_key)
        assert not [
            row
            for row in CurationV2.audit_orphaned_lineage()
            if row["sorting_id"] == sorting_key["sorting_id"]
        ]
        with pytest.raises(CurationNotFoundError):
            root.as_key()
    finally:
        clear_curations_for(sorting_key)


@pytest.mark.slow
@pytest.mark.integration
def test_evaluation_facade_snapshot_labels_and_merge_receipt(
    planted_two_unit_sort, curation_evaluation_defaults
):
    """The scripted journey is identity-safe, idempotent, and spec-preserving."""
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.curation_api import (
        EvaluationSpec,
        create_initial_curation,
        merge_and_evaluate,
    )
    from spyglass.spikesorting.v2.metric_curation import (
        CurationEvaluation,
        CurationEvaluationSelection,
    )
    from spyglass.spikesorting.v2.sorting import Sorting

    sorting_key = dict(planted_two_unit_sort)
    unit_ids = sorted(map(int, (Sorting.Unit & sorting_key).fetch("unit_id")))
    clear_curations_for(sorting_key)
    try:
        root = create_initial_curation(
            sorting_key, labels={unit_ids[0]: ["mua"]}
        )
        evaluation = root.evaluate(
            metric_params_name="minimal",
            auto_curation_rules_name="none",
        )
        assert evaluation.spec.metric_params_name == "minimal"
        assert evaluation.spec.auto_curation_rules_name == "none"

        metrics = evaluation.metrics
        original_columns = list(metrics.columns)
        metrics["local_only"] = 1.0
        assert list(evaluation.metrics.columns) == original_columns
        labels = evaluation.proposed_labels
        labels[999] = ["noise"]
        assert 999 not in evaluation.proposed_labels
        suggestions = evaluation.suggested_merges
        suggestions.append([998, 999])
        assert [998, 999] not in evaluation.suggested_merges

        children_before = len(
            CurationV2
            & {
                "sorting_id": root.sorting_id,
                "parent_curation_id": root.curation_id,
            }
        )
        with pytest.raises(ValueError, match="absent from parent"):
            evaluation.merge_and_evaluate([[unit_ids[0], 999]])
        assert (
            len(
                CurationV2
                & {
                    "sorting_id": root.sorting_id,
                    "parent_curation_id": root.curation_id,
                }
            )
            == children_before
        )

        with CurationV2.connection.transaction:
            with pytest.raises(RuntimeError, match="outside.*transaction"):
                evaluation.merge_and_evaluate([[unit_ids[0], unit_ids[1]]])
            with pytest.raises(RuntimeError, match="outside.*transaction"):
                merge_and_evaluate(
                    parent_curation=root,
                    spec=EvaluationSpec("minimal", "none"),
                    groups=[[unit_ids[0], unit_ids[1]]],
                )

        # The expert commit is intentionally separate and performs no
        # evaluation. merge_and_evaluate then resumes from that child.
        committed = evaluation.commit_merges([[unit_ids[0], unit_ids[1]]])
        assert not (
            CurationEvaluationSelection & committed.as_key()
        ), "commit_merges must not create an evaluation selection"

        first = evaluation.merge_and_evaluate([[unit_ids[0], unit_ids[1]]])
        assert first.child == committed
        assert first.curation_status == "reused"
        assert first.evaluation.spec == evaluation.spec
        assert first.evaluation.curation == first.child
        assert CurationEvaluation & {
            "curation_evaluation_id": first.evaluation.evaluation_id
        }

        second = evaluation.merge_and_evaluate([[unit_ids[0], unit_ids[1]]])
        assert second.child == first.child
        assert set(second.stage_statuses.values()) == {"reused"}
        assert second.evaluation.evaluation_id == first.evaluation.evaluation_id

        replaced = evaluation.accept_labels(mode="replace")
        overlaid = evaluation.accept_labels(mode="overlay")
        assert not (CurationV2.UnitLabel & replaced.as_key())
        assert {
            (int(row["unit_id"]), row["curation_label"])
            for row in (CurationV2.UnitLabel & overlaid.as_key()).fetch(
                as_dict=True
            )
        } == {(unit_ids[0], "mua")}
    finally:
        clear_curations_for(sorting_key)

"""Identity-safe curation facade, receipts, and lifecycle contracts."""

from __future__ import annotations

import inspect

import pytest
import uuid


def test_non_root_facade_requires_typed_parent():
    """Child facade operations expose no numeric root-sentinel escape hatch."""
    from spyglass.spikesorting.v2.curation_api import save_manual_curation

    parameter = inspect.signature(save_manual_curation).parameters[
        "parent_curation"
    ]
    assert parameter.default is inspect.Parameter.empty
    with pytest.raises(TypeError, match="requires parent_curation=CurationRef"):
        save_manual_curation(parent_curation={"curation_id": -1})


def test_merge_group_normalization_is_lossless():
    """Malformed merge requests fail loudly instead of selecting other units.

    Only integers in non-string containers are accepted (no ``1.9``,
    ``True`` or ``"12"``), checked before any database read.
    """
    import numpy as np

    from spyglass.spikesorting.v2._core.lookup_validation import lossless_int
    from spyglass.spikesorting.v2.curation_api import _normalize_merge_groups

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
    assert lossless_int(np.int64(3), "curation_id") == 3
    with pytest.raises(ValueError, match="curation_id must be an integer"):
        lossless_int(3.5, "curation_id")


@pytest.mark.parametrize(
    "recipe_field", ["metric_params_name", "auto_curation_rules_name"]
)
@pytest.mark.parametrize("replacement", ["changed_recipe", None])
def test_evaluation_facade_refuses_changed_or_missing_recipes(
    monkeypatch, recipe_field, replacement
):
    """Persisted reads must still name the recipes of the cached result."""
    import sys
    from types import SimpleNamespace

    import pandas as pd

    from spyglass.spikesorting.v2.curation_api import (
        CurationRef,
        EvaluationResult,
        EvaluationSpec,
    )

    class Rows:
        def __init__(self, rows):
            self.rows = rows

        def __and__(self, key):
            return Rows(
                [
                    row
                    for row in self.rows
                    if all(
                        row.get(name) == value for name, value in key.items()
                    )
                ]
            )

        def fetch(self, *, as_dict):
            assert as_dict
            return [dict(row) for row in self.rows]

        def __len__(self):
            return len(self.rows)

    ref = CurationRef(uuid.uuid4(), 1, uuid.uuid4())
    result = EvaluationResult(
        curation=ref,
        spec=EvaluationSpec("minimal", "none"),
        evaluation_id=uuid.uuid4(),
        metrics=pd.DataFrame({"snr": [7.0]}, index=[0]),
        suggested_merges=[],
        proposed_labels={},
    )
    key = {"curation_evaluation_id": result.evaluation_id}
    row = {
        **key,
        "sorting_id": ref.sorting_id,
        "curation_id": ref.curation_id,
        "metric_params_name": result.spec.metric_params_name,
        "auto_curation_rules_name": result.spec.auto_curation_rules_name,
    }
    monkeypatch.setattr(
        CurationRef, "as_key", lambda self: self._unchecked_key()
    )
    monkeypatch.setitem(
        sys.modules,
        "spyglass.spikesorting.v2.metric_curation",
        SimpleNamespace(
            CurationEvaluationSelection=Rows([row]),
            CurationEvaluation=Rows([key]),
        ),
    )
    assert result._current_evaluation_key() == key
    if replacement is None:
        row.pop(recipe_field)
    else:
        row[recipe_field] = replacement
    with pytest.raises(LookupError, match="recipes no longer match"):
        result.plots.metrics()


def test_evaluation_plots_delegate_to_curation_evaluation(monkeypatch):
    """``result.plots`` calls the matching ``CurationEvaluation`` method."""
    import sys
    from types import SimpleNamespace

    import pandas as pd

    from spyglass.spikesorting.v2.curation_api import (
        CurationRef,
        EvaluationResult,
        EvaluationSpec,
    )

    calls = []

    class FakeEvaluation:
        def __getattr__(self, method):
            def record(key, *args, **kwargs):
                calls.append((method, key, args, kwargs))

            return record

    result = EvaluationResult(
        curation=CurationRef(uuid.uuid4(), 1, uuid.uuid4()),
        spec=EvaluationSpec("minimal", "none"),
        evaluation_id=uuid.uuid4(),
        metrics=pd.DataFrame({"snr": [7.0]}, index=[0]),
        suggested_merges=[],
        proposed_labels={},
    )
    key = {"curation_evaluation_id": result.evaluation_id}
    monkeypatch.setattr(
        EvaluationResult, "_current_evaluation_key", lambda self: key
    )
    monkeypatch.setitem(
        sys.modules,
        "spyglass.spikesorting.v2.metric_curation",
        SimpleNamespace(CurationEvaluation=FakeEvaluation),
    )

    result.plots.si_quality_metrics(compute_missing=True)
    result.plots.si_template_metrics(backend="ipywidgets")
    result.plots.pair_peaks([(0, 1)])
    assert calls == [
        ("plot_si_quality_metrics", key, (), {"compute_missing": True}),
        ("plot_si_template_metrics", key, (), {"backend": "ipywidgets"}),
        ("investigate_pair_peaks", key, ([(0, 1)],), {}),
    ]


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
        save_manual_curation,
    )
    from spyglass.spikesorting.v2.sorting import Sorting

    sorting_key = dict(planted_three_unit_sort)
    unit_ids = sorted(map(int, (Sorting.Unit & sorting_key).fetch("unit_id")))
    clear_curations_for(sorting_key)
    try:
        root = CurationRef.from_key(
            CurationV2.create_initial_curation(sorting_key)
        )
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
        for ref in (root, labeled, preview, merged):
            operation = ref.operation_type
            assert (
                f"curation {ref.curation_id} [{ref.commit_status}; "
                f"{operation.producer}/{operation.change_kind}]"
            ) in tree

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
        # A stale uuid for the same numeric id must not resolve.
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

        # The earlier ref rejects a recreated row that reuses the PK.
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
        # Mappings: stale uuid refused, new uuid resolves, bare key resolves
        # the current generation.
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
        CurationRef,
        save_manual_curation,
    )
    from spyglass.spikesorting.v2.exceptions import CurationNotFoundError
    from spyglass.spikesorting.v2.sorting import Sorting

    sorting_key = dict(planted_three_unit_sort)
    unit_ids = sorted(map(int, (Sorting.Unit & sorting_key).fetch("unit_id")))
    clear_curations_for(sorting_key)
    try:
        root = CurationRef.from_key(
            CurationV2.create_initial_curation(sorting_key)
        )
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
    from spyglass.spikesorting.v2.curation_api import CurationRef
    from spyglass.spikesorting.v2.metric_curation import (
        CurationEvaluation,
        CurationEvaluationSelection,
    )
    from spyglass.spikesorting.v2.sorting import Sorting

    sorting_key = dict(planted_two_unit_sort)
    unit_ids = sorted(map(int, (Sorting.Unit & sorting_key).fetch("unit_id")))
    clear_curations_for(sorting_key)
    try:
        root = CurationRef.from_key(
            CurationV2.create_initial_curation(
                sorting_key, labels={unit_ids[0]: ["mua"]}
            )
        )
        evaluation = root.evaluate(
            metric_params_name="minimal",
            auto_curation_rules_name="none",
        )
        assert evaluation.spec.metric_params_name == "minimal"
        assert evaluation.spec.auto_curation_rules_name == "none"
        assert "unit_properties" not in root.summarize()
        properties = root.summarize(evaluation=evaluation)["unit_properties"]
        assert properties.index.tolist() == unit_ids
        assert "snr" in properties.columns

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

        # The expert commit does not evaluate; merge_and_evaluate resumes.
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
        assert {
            second.curation_status,
            second.evaluation_selection_status,
            second.evaluation_status,
        } == {"reused"}
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


@pytest.mark.slow
@pytest.mark.integration
def test_merge_and_evaluate_keeps_the_parent_metric_waveform_recipe(
    planted_two_unit_sort, curation_evaluation_defaults
):
    """The merged child is evaluated with the parent's metric waveform
    recipe, not the default."""
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.curation_api import (
        CurationRef,
        EvaluationResult,
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
        root = CurationRef.from_key(
            CurationV2.create_initial_curation(sorting_key)
        )
        default_key = CurationEvaluationSelection.insert_by_curation_id(
            root.sorting_id, root.curation_id, "minimal", "none"
        )
        default_recipe = (CurationEvaluationSelection & default_key).fetch1(
            "metric_waveform_params_name"
        )
        recipe = next(
            name
            for name in (
                "franklab_hippocampus_metric_waveforms",
                "franklab_cortex_metric_waveforms",
            )
            if name != default_recipe
        )
        parent_key = CurationEvaluationSelection.insert_selection(
            {
                "sorting_id": root.sorting_id,
                "curation_id": root.curation_id,
                "metric_params_name": "minimal",
                "auto_curation_rules_name": "none",
                "metric_waveform_params_name": recipe,
            }
        )
        CurationEvaluation.populate(parent_key, reserve_jobs=False)

        receipt = EvaluationResult.from_key(parent_key).merge_and_evaluate(
            [[unit_ids[0], unit_ids[1]]]
        )

        child_selection = CurationEvaluationSelection & {
            "curation_evaluation_id": receipt.evaluation.evaluation_id
        }
        assert child_selection.fetch1("metric_waveform_params_name") == recipe
        assert receipt.evaluation_selection_status == "computed"
    finally:
        clear_curations_for(sorting_key)


@pytest.mark.slow
@pytest.mark.integration
def test_evaluate_then_merge_and_evaluate_uses_a_requested_recipe(
    planted_two_unit_sort, curation_evaluation_defaults
):
    """``evaluate(...).merge_and_evaluate`` keeps a named recipe throughout."""
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.curation_api import CurationRef
    from spyglass.spikesorting.v2.metric_curation import (
        CurationEvaluation,
        CurationEvaluationSelection,
    )
    from spyglass.spikesorting.v2.sorting import Sorting

    sorting_key = dict(planted_two_unit_sort)
    unit_ids = sorted(map(int, (Sorting.Unit & sorting_key).fetch("unit_id")))
    clear_curations_for(sorting_key)
    try:
        root = CurationRef.from_key(
            CurationV2.create_initial_curation(sorting_key)
        )
        default_key = CurationEvaluationSelection.insert_by_curation_id(
            root.sorting_id, root.curation_id, "minimal", "none"
        )
        default_recipe = (CurationEvaluationSelection & default_key).fetch1(
            "metric_waveform_params_name"
        )
        recipe = next(
            name
            for name in (
                "franklab_hippocampus_metric_waveforms",
                "franklab_cortex_metric_waveforms",
            )
            if name != default_recipe
        )

        receipt = root.evaluate(
            metric_params_name="minimal",
            auto_curation_rules_name="none",
            metric_waveform_params_name=recipe,
        ).merge_and_evaluate([[unit_ids[0], unit_ids[1]]])

        parent_recipes = set(
            (
                CurationEvaluationSelection * CurationEvaluation.proj()
                & root.as_key()
            ).fetch("metric_waveform_params_name")
        )
        assert parent_recipes == {recipe}
        child_selection = CurationEvaluationSelection & {
            "curation_evaluation_id": receipt.evaluation.evaluation_id
        }
        assert child_selection.fetch1("metric_waveform_params_name") == recipe
    finally:
        clear_curations_for(sorting_key)

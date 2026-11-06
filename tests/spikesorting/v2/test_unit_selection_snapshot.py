"""Read-only population resolution and persisted matching snapshot integrity."""

import sys
import uuid
from types import SimpleNamespace

import pandas as pd
import pytest

from spyglass.spikesorting.v2 import analysis_selection as selection
from spyglass.spikesorting.v2._curation.unit_selection import (
    unit_selection_hash,
    validate_unit_selection_snapshot,
)
from spyglass.spikesorting.v2.curation_api import (
    CurationRef,
    EvaluationResult,
    EvaluationSpec,
)
from spyglass.spikesorting.v2.exceptions import CurationNotFoundError

pytestmark = pytest.mark.unit


class ReadOnlyRows:
    """Minimal relation with real restrictions and forbidden mutations."""

    def __init__(self, rows):
        self.rows = rows

    def __and__(self, key):
        return ReadOnlyRows(
            [
                row
                for row in self.rows
                if all(row.get(name) == value for name, value in key.items())
            ]
        )

    def fetch(self, *attributes, as_dict=False, **kwargs):
        if as_dict:
            return [dict(row) for row in self.rows]
        assert len(attributes) == 1
        return [row[attributes[0]] for row in self.rows]

    def fetch1(self):
        assert len(self.rows) == 1
        return dict(self.rows[0])

    def __len__(self):
        return len(self.rows)

    def __getattr__(self, name):
        if name.startswith(("insert", "populate", "update", "delete")):
            raise AssertionError(f"Read-only resolver attempted {name}.")
        raise AttributeError(name)


@pytest.fixture
def population(monkeypatch):
    ref = CurationRef(uuid.uuid4(), 2, uuid.uuid4())
    key = {"sorting_id": ref.sorting_id, "curation_id": ref.curation_id}
    row = {**key, "curation_uuid": ref.curation_uuid, "merges_applied": True}
    labels = {1: ["accept"], 2: ["accept"], 3: ["noise"], 4: ["mua"], 5: []}
    master = ReadOnlyRows([row])
    master.Unit = ReadOnlyRows([{**key, "unit_id": unit} for unit in labels])
    master._labels_by_unit = lambda key: labels

    def assert_committed(key, *, context, merges_applied):
        if not merges_applied:
            raise ValueError("Cannot select a merge preview.")

    master.assert_committed_curation = assert_committed
    monkeypatch.setitem(
        sys.modules,
        "spyglass.spikesorting.v2.curation",
        SimpleNamespace(CurationV2=master),
    )
    policies = [
        {
            "unit_filter_params_name": name,
            "include_labels": list(policy["include_labels"]),
            "exclude_labels": list(policy["exclude_labels"]),
            "unit_criteria": None,
        }
        for name, policy in selection.V2_UNIT_SELECTION_POLICIES.items()
    ]
    policies.append(
        {
            "unit_filter_params_name": "all_units",
            "include_labels": [],
            "exclude_labels": [],
            "unit_criteria": None,
        }
    )
    monkeypatch.setitem(
        sys.modules,
        "spyglass.spikesorting.analysis.v1.group",
        SimpleNamespace(UnitSelectionParams=ReadOnlyRows(policies)),
    )
    monkeypatch.setitem(
        sys.modules,
        "spyglass.spikesorting.v2.unit_annotation",
        SimpleNamespace(CurationUnitAnnotationSet=object),
    )
    monkeypatch.setattr(
        selection,
        "ensure_v2_unit_selection_policies",
        lambda: pytest.fail("Resolver must not seed policies."),
    )
    check_evaluation_key = EvaluationResult._current_evaluation_key
    monkeypatch.setattr(
        EvaluationResult,
        "_current_evaluation_key",
        lambda self: {"curation_evaluation_id": self.evaluation_id},
    )
    evaluation = EvaluationResult(
        curation=ref,
        spec=EvaluationSpec("metrics", "rules"),
        evaluation_id=uuid.uuid4(),
        metrics=pd.DataFrame({"snr": [7.0, None, 9.0, 6.0, 8.0]}, index=labels),
        suggested_merges=(),
        proposed_labels={},
    )
    return SimpleNamespace(
        ref=ref,
        row=row,
        labels=labels,
        policies=policies,
        evaluation=evaluation,
        check_evaluation_key=check_evaluation_key,
    )


def test_resolver_freezes_exact_population_without_group_or_cache_work(
    population,
):
    snapshot = selection.resolve_unit_selection(
        population.ref,
        evaluation=population.evaluation,
        unit_criteria={"snr": {">": 5}},
    )
    assert snapshot.curation == population.ref
    assert snapshot.included_unit_ids == (1,)
    assert snapshot.excluded_units[2] == "unavailable metric: snr"
    assert snapshot.excluded_units[3] == "denied label(s): noise"
    assert snapshot.unlabeled_unit_ids == (5,)
    payload = snapshot.snapshot()
    assert payload["curation_uuid"] == str(population.ref.curation_uuid)
    assert payload["selection_provenance"]["evaluation_id"] == str(
        population.evaluation.evaluation_id
    )
    assert validate_unit_selection_snapshot(payload) == payload


def test_label_only_resolution_does_not_read_unit_properties(
    population, monkeypatch
):
    from spyglass.spikesorting.v2 import annotation_api

    monkeypatch.setattr(
        annotation_api,
        "read_unit_properties",
        lambda *args, **kwargs: pytest.fail(
            "Label selection needs only metadata."
        ),
    )
    assert selection.resolve_unit_selection(
        population.ref
    ).included_unit_ids == (1, 2)


def test_explicit_annotation_pins_and_nullable_values(population, monkeypatch):
    from spyglass.spikesorting.v2.annotation_api import AnnotationSetRef

    annotation = AnnotationSetRef(population.ref, "good", 1, "bool", "a" * 64)
    monkeypatch.setattr(AnnotationSetRef, "_current_row", lambda self: {})
    table = SimpleNamespace(
        _dataframe_from_ref=lambda ref: pd.DataFrame(
            {
                "good": pd.array(
                    [True, None, False, False, False], dtype="boolean"
                )
            },
            index=population.labels,
        )
    )
    monkeypatch.setitem(
        sys.modules,
        "spyglass.spikesorting.v2.unit_annotation",
        SimpleNamespace(
            CurationUnitAnnotationSet=table, UnitAnnotationDefinition=object
        ),
    )
    snapshot = selection.resolve_unit_selection(
        population.ref,
        policy="all_units",
        annotation_sets=[annotation],
        unit_criteria={annotation.column_name: {"!=": False}},
    )
    assert snapshot.included_unit_ids == (1,)
    assert snapshot.excluded_units[2].startswith("unavailable metric:")
    assert snapshot.selection_provenance["annotation_sets"] == [
        annotation.snapshot()
    ]
    assert (
        validate_unit_selection_snapshot(snapshot.snapshot())
        == snapshot.snapshot()
    )
    wrong = AnnotationSetRef(
        CurationRef(uuid.uuid4(), 1, uuid.uuid4()), "good", 1, "bool", "b" * 64
    )
    with pytest.raises(ValueError, match="different curation"):
        selection.resolve_unit_selection(
            population.ref, annotation_sets=[wrong]
        )


def test_snapshot_is_defensive_and_detects_policy_and_label_drift(population):
    original = selection.resolve_unit_selection(population.ref)
    data = original.snapshot()
    data["policy"]["include_labels"].clear()
    data["labels_by_unit"]["1"].clear()
    assert original.policy["include_labels"] == ("accept",)
    assert original.labels_by_unit[1] == ("accept",)
    with pytest.raises(TypeError):
        original.excluded_units[3] = "selected"
    assert original.revalidate().selection_hash == original.selection_hash
    population.labels[1] = ["reject"]
    assert original.included_unit_ids == (1, 2)
    assert original.labels_by_unit[1] == ("accept",)
    with pytest.raises(ValueError, match="changed since planning"):
        original.revalidate()
    population.labels[1] = ["accept"]
    population.policies[0]["exclude_labels"].append("accept")
    with pytest.raises(ValueError, match="changed since planning"):
        original.revalidate()


def test_hash_includes_criteria_even_when_membership_does_not_change(
    population,
):
    first = selection.resolve_unit_selection(
        population.ref,
        evaluation=population.evaluation,
        unit_criteria={"snr": {">": 5}},
    )
    second = selection.resolve_unit_selection(
        population.ref,
        evaluation=population.evaluation,
        unit_criteria={"snr": {">": 6}},
    )
    assert first.included_unit_ids == second.included_unit_ids == (1,)
    assert first.selection_hash != second.selection_hash
    other = selection.resolve_unit_selection(population.ref, policy="all_units")
    assert other.included_unit_ids == (1, 2, 3, 4, 5)
    assert other.selection_hash != first.selection_hash


def test_stored_criteria_drift_is_rechecked(population):
    population.policies[0]["unit_criteria"] = {"snr": {">": 5}}
    snapshot = selection.resolve_unit_selection(
        population.ref, evaluation=population.evaluation
    )
    population.policies[0]["unit_criteria"]["snr"][">"] = 10
    assert snapshot.unit_criteria == {"snr": {">": 5}}
    with pytest.raises(ValueError, match="changed since planning"):
        snapshot.revalidate()


def test_missing_policy_is_actionable_without_seeding(population):
    with pytest.raises(ValueError, match="initialize_v2_defaults"):
        selection.resolve_unit_selection(population.ref, policy="missing")


def test_stale_generation_and_merge_preview_are_refused(population):
    snapshot = selection.resolve_unit_selection(population.ref)
    population.row["curation_uuid"] = uuid.uuid4()
    with pytest.raises(CurationNotFoundError, match="generation"):
        snapshot.revalidate()
    population.row["curation_uuid"] = population.ref.curation_uuid
    population.row["merges_applied"] = False
    with pytest.raises(ValueError, match="preview"):
        selection.resolve_unit_selection(population.ref)


def test_metric_evidence_must_match_the_exact_curation(population):
    object.__setattr__(
        population.evaluation,
        "curation",
        CurationRef(uuid.uuid4(), 1, uuid.uuid4()),
    )
    with pytest.raises(ValueError, match="different curation"):
        selection.resolve_unit_selection(
            population.ref, evaluation=population.evaluation
        )


@pytest.mark.parametrize(
    "edit",
    [
        lambda data: data.update(curation_id=True),
        lambda data: data.update(curation_uuid="invalid"),
        lambda data: data.update(included_unit_ids=[1, 1, 2]),
        lambda data: data.update(included_unit_ids=[1, 2.5]),
        lambda data: data["excluded_units"].update({"1": "denied"}),
        lambda data: data["labels_by_unit"].pop("5"),
        lambda data: data["labels_by_unit"].update({"1": ["reject"]}),
        lambda data: data["selection_provenance"].update(
            curation_uuid=str(uuid.uuid4())
        ),
    ],
)
def test_malformed_snapshots_fail_even_with_recomputed_hash(population, edit):
    data = selection.resolve_unit_selection(population.ref).snapshot()
    edit(data)
    data["selection_hash"] = unit_selection_hash(data)
    with pytest.raises(ValueError):
        validate_unit_selection_snapshot(data)


def test_tampered_hash_and_missing_metric_columns_fail(population):
    data = selection.resolve_unit_selection(population.ref).snapshot()
    data["selection_hash"] = "0" * 64
    with pytest.raises(ValueError, match="selection_hash"):
        validate_unit_selection_snapshot(data)
    with pytest.raises(ValueError, match="not in units table"):
        selection.resolve_unit_selection(
            population.ref, unit_criteria={"missing_metric": {">": 0}}
        )


@pytest.mark.parametrize("operator", [{"!=": False}, {"notin": [False]}])
def test_nullable_boolean_evidence_stays_unavailable(population, operator):
    population.evaluation._metrics["flag"] = pd.array(
        [True, None, False, True, True], dtype="boolean"
    )
    snapshot = selection.resolve_unit_selection(
        population.ref,
        evaluation=population.evaluation,
        unit_criteria={"flag": operator},
    )
    assert snapshot.included_unit_ids == (1,)
    assert snapshot.excluded_units[2] == "unavailable metric: flag"


def test_criteria_key_order_does_not_change_population_identity(population):
    first = selection.resolve_unit_selection(
        population.ref,
        evaluation=population.evaluation,
        unit_criteria={"snr": {">": 8, "<": 10}},
    )
    second = selection.resolve_unit_selection(
        population.ref,
        evaluation=population.evaluation,
        unit_criteria={"snr": {"<": 10, ">": 8}},
    )
    assert first.snapshot() == second.snapshot()


@pytest.mark.parametrize(
    "recipe_field", ["metric_params_name", "auto_curation_rules_name"]
)
def test_snapshot_revalidation_refuses_changed_evaluation_recipes(
    population, monkeypatch, recipe_field
):
    row = {
        "curation_evaluation_id": population.evaluation.evaluation_id,
        "sorting_id": population.ref.sorting_id,
        "curation_id": population.ref.curation_id,
        "metric_params_name": population.evaluation.spec.metric_params_name,
        "auto_curation_rules_name": population.evaluation.spec.auto_curation_rules_name,
    }
    monkeypatch.setitem(
        sys.modules,
        "spyglass.spikesorting.v2.metric_curation",
        SimpleNamespace(
            CurationEvaluationSelection=ReadOnlyRows([row]),
            CurationEvaluation=ReadOnlyRows(
                [
                    {
                        "curation_evaluation_id": population.evaluation.evaluation_id
                    }
                ]
            ),
        ),
    )
    monkeypatch.setattr(
        EvaluationResult,
        "_current_evaluation_key",
        population.check_evaluation_key,
    )
    snapshot = selection.resolve_unit_selection(
        population.ref,
        evaluation=population.evaluation,
        unit_criteria={"snr": {">": 5}},
    )
    assert snapshot.revalidate().selection_hash == snapshot.selection_hash
    row[recipe_field] = "replacement_recipe"
    with pytest.raises(LookupError, match="recipes no longer match"):
        snapshot.revalidate()
    assert snapshot.included_unit_ids == (1,)


@pytest.mark.parametrize(
    "edit",
    [
        lambda p: p.pop("evaluation_id"),
        lambda p: p.pop("evaluation_recipes"),
        lambda p: p.update(evaluation_id="not-a-uuid"),
        lambda p: p.update(evaluation_id=None),
        lambda p: p.update(evaluation_recipes=None),
        lambda p: p.update(
            evaluation_recipes={"metric_params_name": "metrics"}
        ),
        lambda p: p["evaluation_recipes"].update(metric_params_name=""),
        lambda p: p["evaluation_recipes"].update(auto_curation_rules_name=1),
    ],
)
def test_malformed_evaluation_pins_fail_even_with_recomputed_hash(
    population, edit
):
    data = selection.resolve_unit_selection(
        population.ref, evaluation=population.evaluation
    ).snapshot()
    edit(data["selection_provenance"])
    data["selection_hash"] = unit_selection_hash(data)
    with pytest.raises(ValueError):
        validate_unit_selection_snapshot(data)


@pytest.mark.parametrize(
    "edit",
    [
        lambda a: a.pop("annotation_name"),
        lambda a: a.update(annotation_name=""),
        lambda a: a.update(annotation_version=True),
        lambda a: a.update(annotation_version=1.0),
        lambda a: a.update(annotation_version=0),
        lambda a: a.update(curation_id=True),
        lambda a: a.update(value_type="unknown"),
        lambda a: a.update(value_type=[]),
        lambda a: a.update(set_hash="a" * 63),
        lambda a: a.update(set_hash="A" * 64),
        lambda a: a.update(column_name="annotation:good:v1:wrong_hash"),
    ],
)
def test_malformed_annotation_pins_fail_even_with_recomputed_hash(
    population, edit
):
    data = selection.resolve_unit_selection(population.ref).snapshot()
    annotation = {
        **{
            key: data[key]
            for key in ("sorting_id", "curation_id", "curation_uuid")
        },
        "annotation_name": "good",
        "annotation_version": 1,
        "value_type": "bool",
        "set_hash": "a" * 64,
        "column_name": f"annotation:good:v1:{'a' * 64}",
    }
    data["selection_provenance"]["annotation_sets"] = [annotation]
    data["selection_hash"] = unit_selection_hash(data)
    assert validate_unit_selection_snapshot(data) == data
    edit(annotation)
    data["selection_hash"] = unit_selection_hash(data)
    with pytest.raises(ValueError):
        validate_unit_selection_snapshot(data)


@pytest.mark.parametrize("annotations", [None, {}, "annotation", [{}]])
def test_annotation_collection_requires_complete_pins(population, annotations):
    data = selection.resolve_unit_selection(population.ref).snapshot()
    data["selection_provenance"]["annotation_sets"] = annotations
    data["selection_hash"] = unit_selection_hash(data)
    with pytest.raises(ValueError):
        validate_unit_selection_snapshot(data)

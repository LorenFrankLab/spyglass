"""Database-backed tests for migrating positional UnitAnnotation ids."""

from uuid import uuid4

import datajoint as dj
import pandas as pd
import pytest

pytestmark = pytest.mark.integration


class _FakeSpikeSortingOutput:
    """Minimal merge relation that returns synthetic NWB fetch payloads."""

    def __init__(self, payloads, merge_id=None):
        self.payloads = payloads
        self.merge_id = merge_id

    def __and__(self, restriction):
        return type(self)(self.payloads, restriction["merge_id"])

    def fetch_nwb(self):
        return [self.payloads[self.merge_id]]


@pytest.fixture(scope="module")
def annotation_table(dj_conn):
    """Create an isolated table using the production migration methods."""
    from spyglass.spikesorting.analysis.v1.unit_annotation import (
        UnitAnnotation,
    )
    from spyglass.utils import SpyglassMixin

    class MigrationUnitAnnotation(SpyglassMixin, dj.Manual):
        definition = """
        spikesorting_merge_id: uuid
        unit_id: int
        """

        class Annotation(SpyglassMixin, dj.Part):
            definition = """
            -> master
            annotation: varchar(128)
            ---
            label = NULL: varchar(128)
            quantification = NULL: float
            """

        audit_positional_unit_ids = classmethod(
            UnitAnnotation.audit_positional_unit_ids.__func__
        )
        migrate_positional_unit_ids = classmethod(
            UnitAnnotation.migrate_positional_unit_ids.__func__
        )

    context = {"MigrationUnitAnnotation": MigrationUnitAnnotation}
    schema = dj.Schema(
        "test_unit_annotation_migration",
        context=context,
        connection=dj_conn,
    )
    schema(MigrationUnitAnnotation)

    yield MigrationUnitAnnotation

    previous_level = dj.logger.level
    dj.logger.setLevel("ERROR")
    schema.drop(force=True)
    dj.logger.setLevel(previous_level)


@pytest.fixture
def annotation_case(annotation_table, monkeypatch):
    """Insert sparse and dense namespaces with positional annotation ids."""
    import spyglass.spikesorting.analysis.v1.unit_annotation as module

    sparse_id = uuid4()
    dense_id = uuid4()
    payloads = {
        sparse_id: {"object_id": pd.DataFrame(index=pd.Index([2, 3, 4, 10]))},
        dense_id: {"object_id": pd.DataFrame(index=pd.Index([0, 1, 2, 3]))},
    }
    monkeypatch.setattr(
        module, "SpikeSortingOutput", _FakeSpikeSortingOutput(payloads)
    )

    annotation_table.insert(
        [
            {"spikesorting_merge_id": merge_id, "unit_id": unit_id}
            for merge_id in (sparse_id, dense_id)
            for unit_id in range(4)
        ]
    )
    annotation_table.Annotation.insert(
        [
            {
                "spikesorting_merge_id": sparse_id,
                "unit_id": unit_id,
                "annotation": f"annotation-{unit_id}",
                "label": f"label-{unit_id}",
                "quantification": unit_id + 0.5,
            }
            for unit_id in range(4)
        ]
    )

    yield {
        "table": annotation_table,
        "sparse_id": sparse_id,
        "dense_id": dense_id,
        "payloads": payloads,
    }

    annotation_table.Annotation.delete_quick()
    annotation_table.delete_quick()


def _fetch_rows(relation):
    """Return fetched rows in deterministic primary-key order."""
    return sorted(
        relation.fetch(as_dict=True),
        key=lambda row: tuple(str(row[key]) for key in relation.primary_key),
    )


def test_audit_lists_sparse_namespace_only(annotation_case):
    case = annotation_case

    audit = case["table"].audit_positional_unit_ids()

    assert audit.columns.tolist() == [
        "spikesorting_merge_id",
        "n_units",
        "true_unit_ids",
        "stored_unit_ids",
    ]
    assert audit.to_dict("records") == [
        {
            "spikesorting_merge_id": case["sparse_id"],
            "n_units": 4,
            "true_unit_ids": [2, 3, 4, 10],
            "stored_unit_ids": [0, 1, 2, 3],
        }
    ]


def test_dry_run_returns_mapping_without_writes(annotation_case):
    case = annotation_case
    table = case["table"]
    masters_before = _fetch_rows(table)
    annotations_before = _fetch_rows(table.Annotation)

    plan = table.migrate_positional_unit_ids()

    assert plan == {case["sparse_id"]: {0: 2, 1: 3, 2: 4, 3: 10}}
    assert _fetch_rows(table) == masters_before
    assert _fetch_rows(table.Annotation) == annotations_before


def test_apply_rewrites_ids_and_preserves_payload(annotation_case):
    case = annotation_case
    table = case["table"]
    sparse_restriction = {"spikesorting_merge_id": case["sparse_id"]}
    payload_before = sorted(
        (
            row["annotation"],
            row["label"],
            row["quantification"],
        )
        for row in (table.Annotation & sparse_restriction).fetch(as_dict=True)
    )

    plan = table.migrate_positional_unit_ids(dry_run=False)

    assert plan == {case["sparse_id"]: {0: 2, 1: 3, 2: 4, 3: 10}}
    assert sorted(
        int(unit_id)
        for unit_id in (table & sparse_restriction).fetch("unit_id")
    ) == [2, 3, 4, 10]
    assert sorted(
        int(unit_id)
        for unit_id in (
            table & {"spikesorting_merge_id": case["dense_id"]}
        ).fetch("unit_id")
    ) == [0, 1, 2, 3]
    payload_after = sorted(
        (
            row["annotation"],
            row["label"],
            row["quantification"],
        )
        for row in (table.Annotation & sparse_restriction).fetch(as_dict=True)
    )
    assert payload_after == payload_before

    # The audit identifies sparse namespaces, so it remains a candidate audit
    # after a successful one-time migration. Validate the actual postcondition.
    audit = table.audit_positional_unit_ids()
    assert audit.iloc[0].stored_unit_ids == audit.iloc[0].true_unit_ids


def test_invalid_position_aborts_without_writes(annotation_case):
    case = annotation_case
    table = case["table"]
    table.Annotation.delete_quick()
    table.delete_quick()
    table.insert(
        [
            {"spikesorting_merge_id": case["sparse_id"], "unit_id": 0},
            {"spikesorting_merge_id": case["sparse_id"], "unit_id": 7},
        ]
    )
    table.Annotation.insert1(
        {
            "spikesorting_merge_id": case["sparse_id"],
            "unit_id": 7,
            "annotation": "invalid-position",
            "label": "preserve-me",
            "quantification": 7.5,
        }
    )
    masters_before = _fetch_rows(table)
    annotations_before = _fetch_rows(table.Annotation)

    with pytest.raises(ValueError, match="outside the positional range"):
        table.migrate_positional_unit_ids(dry_run=False)

    assert _fetch_rows(table) == masters_before
    assert _fetch_rows(table.Annotation) == annotations_before

"""Database-backed tests for migrating positional UnitAnnotation ids."""

from uuid import uuid4

import pandas as pd
import pytest

from tests.spikesorting._annotation_fixtures import (
    FakeSpikeSortingOutput,
    drop_annotation_tables,
    make_annotation_tables,
)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def annotation_table(dj_conn):
    """Create an isolated table using the production migration methods."""
    table, schema = make_annotation_tables(
        dj_conn, "test_unit_annotation_migration"
    )

    yield table

    drop_annotation_tables(schema)


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
        module, "SpikeSortingOutput", FakeSpikeSortingOutput(payloads)
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
        "marker": annotation_table._positional_id_migration_table,
        "sparse_id": sparse_id,
        "dense_id": dense_id,
        "payloads": payloads,
    }

    annotation_table._positional_id_migration_table.delete_quick()
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

    marker = case["marker"] & sparse_restriction
    assert marker.fetch1("migration_version") == 1
    assert table.audit_positional_unit_ids().empty

    masters_after = _fetch_rows(table)
    annotations_after = _fetch_rows(table.Annotation)
    assert table.migrate_positional_unit_ids(dry_run=False) == {}
    assert _fetch_rows(table) == masters_after
    assert _fetch_rows(table.Annotation) == annotations_after


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

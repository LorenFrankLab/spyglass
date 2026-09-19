"""Database-backed tests for the UnitAnnotation unit-id write boundary.

``UnitAnnotation`` stores true NWB unit ids. Rows written under the older
positional contract mean something different, so a merge still carrying them
must not accept a new annotation until it is migrated, and a merge whose
first-ever annotation is written today must be recorded as needing nothing.
"""

from contextlib import contextmanager
from uuid import uuid4

import pandas as pd
import pytest

from tests.spikesorting._annotation_fixtures import (
    FakeSpikeSortingOutput,
    drop_annotation_tables,
    make_annotation_tables,
)

pytestmark = pytest.mark.integration

# A namespace whose true ids differ from their positions: stored id 2 means
# the third unit under the old contract and the unit named 2 under the new
# one, so the two contracts are distinguishable.
SPARSE_TRUE_IDS = [0, 2, 5, 7]
DENSE_TRUE_IDS = [0, 1, 2, 3]
POSITIONAL_IDS = [0, 1, 2, 3]


@contextmanager
def _failing_insert1(part_table):
    """Make a part table's ``insert1`` raise, then restore the inherited one."""

    def _raise(self, *args, **kwargs):
        raise RuntimeError("injected")

    part_table.insert1 = _raise
    try:
        yield
    finally:
        del part_table.insert1


def _count_nwb_reads(monkeypatch):
    """Record the merge id of every fake ``fetch_nwb`` call from now on."""
    reads = []
    original = FakeSpikeSortingOutput.fetch_nwb

    def _counting_fetch_nwb(self):
        reads.append(self.merge_id)
        return original(self)

    monkeypatch.setattr(
        FakeSpikeSortingOutput, "fetch_nwb", _counting_fetch_nwb
    )
    return reads


def _unit_ids(table, merge_id):
    """Return the stored unit ids of one merge, sorted."""
    return sorted(
        int(unit_id)
        for unit_id in (table & {"spikesorting_merge_id": merge_id}).fetch(
            "unit_id"
        )
    )


@pytest.fixture(scope="module")
def annotation_table(dj_conn):
    """Create an isolated table using the production annotation methods."""
    table, schema = make_annotation_tables(
        dj_conn, "test_unit_annotation_boundary"
    )

    yield table

    drop_annotation_tables(schema)


@pytest.fixture
def annotation_case(annotation_table, monkeypatch):
    """Plant positional rows on a sparse and a dense namespace.

    Also supplies ``fresh_id``: a sparse-namespace merge with no annotation
    rows at all, standing in for a merge first annotated under the current
    contract.
    """
    import spyglass.spikesorting.analysis.v1.unit_annotation as module

    sparse_id = uuid4()
    dense_id = uuid4()
    fresh_id = uuid4()
    payloads = {
        sparse_id: {"object_id": pd.DataFrame(index=pd.Index(SPARSE_TRUE_IDS))},
        dense_id: {"object_id": pd.DataFrame(index=pd.Index(DENSE_TRUE_IDS))},
        fresh_id: {"object_id": pd.DataFrame(index=pd.Index(SPARSE_TRUE_IDS))},
    }
    monkeypatch.setattr(
        module, "SpikeSortingOutput", FakeSpikeSortingOutput(payloads)
    )

    annotation_table.insert(
        [
            {"spikesorting_merge_id": merge_id, "unit_id": unit_id}
            for merge_id in (sparse_id, dense_id)
            for unit_id in POSITIONAL_IDS
        ]
    )
    annotation_table.Annotation.insert(
        [
            {
                "spikesorting_merge_id": sparse_id,
                "unit_id": unit_id,
                "annotation": f"annotation-{unit_id}",
                "label": f"label-{unit_id}",
            }
            for unit_id in POSITIONAL_IDS
        ]
    )

    yield {
        "table": annotation_table,
        "marker": annotation_table._positional_id_migration_table,
        "sparse_id": sparse_id,
        "dense_id": dense_id,
        "fresh_id": fresh_id,
    }

    annotation_table._positional_id_migration_table.delete_quick()
    annotation_table.Annotation.delete_quick()
    annotation_table.delete_quick()


def test_add_annotation_refuses_unmigrated_sparse_merge(annotation_case):
    case = annotation_case
    table = case["table"]
    key = {
        "spikesorting_merge_id": case["sparse_id"],
        "unit_id": 2,
        "annotation": "x",
    }

    with pytest.raises(ValueError, match="migrate_positional_unit_ids"):
        table().add_annotation(key)

    assert not (table.Annotation & key)
    assert _unit_ids(table, case["sparse_id"]) == POSITIONAL_IDS


def test_restricted_instance_refuses_unmigrated_sparse_merge(annotation_case):
    case = annotation_case
    table = case["table"]
    sparse = {"spikesorting_merge_id": case["sparse_id"]}
    # A restriction matching none of the merge's rows: row presence must be
    # judged on the whole table, as the audit is, or the merge looks fresh.
    restricted = table() & {"unit_id": 99}
    assert not restricted

    with pytest.raises(ValueError, match="predate the true-id contract"):
        restricted.add_annotation(
            {**sparse, "unit_id": 2, "annotation": "restricted"}
        )

    assert not (case["marker"] & sparse)
    assert _unit_ids(table, case["sparse_id"]) == POSITIONAL_IDS


def test_add_annotation_marks_first_write(annotation_case):
    case = annotation_case
    table = case["table"]
    fresh = {"spikesorting_merge_id": case["fresh_id"]}

    table().add_annotation({**fresh, "unit_id": 5, "annotation": "first"})

    assert (case["marker"] & fresh).fetch1("migration_version") == 1
    # Unmarked, the stored id 5 would read as a position outside the
    # four-unit namespace and the migration would refuse to run at all.
    plan = table.migrate_positional_unit_ids(dry_run=False)
    assert plan.get(case["fresh_id"], {}) == {}
    assert _unit_ids(table, case["fresh_id"]) == [5]


def _unmigrated_sparse_blocks_writes(case):
    """Positional rows refuse a write and stay listed as needing migration."""
    table = case["table"]
    sparse = {"spikesorting_merge_id": case["sparse_id"]}

    with pytest.raises(ValueError, match="predate the true-id contract"):
        table().add_annotation(
            {**sparse, "unit_id": 5, "annotation": "blocked"}
        )

    assert not (case["marker"] & sparse)
    audit = table.audit_positional_unit_ids(merge_ids=[case["sparse_id"]])
    assert audit["stored_unit_ids"].tolist() == [POSITIONAL_IDS]


def _migrated_sparse_accepts_writes(case):
    """After migration the rows carry true ids and new writes land."""
    table = case["table"]
    sparse = {"spikesorting_merge_id": case["sparse_id"]}
    new_key = {**sparse, "unit_id": 2, "annotation": "post-migration"}

    plan = table.migrate_positional_unit_ids(dry_run=False)
    assert plan == {case["sparse_id"]: {1: 2, 2: 5, 3: 7}}
    assert _unit_ids(table, case["sparse_id"]) == SPARSE_TRUE_IDS

    table().add_annotation(new_key)
    assert (table.Annotation & new_key).fetch1("annotation") == (
        "post-migration"
    )

    assert table.migrate_positional_unit_ids(dry_run=False) == {}
    assert (table.Annotation & new_key).fetch1("unit_id") == 2
    assert _unit_ids(table, case["sparse_id"]) == SPARSE_TRUE_IDS


def _fresh_merge_needs_no_migration(case):
    """A merge first annotated today is left alone by a later migration."""
    table = case["table"]
    fresh = {"spikesorting_merge_id": case["fresh_id"]}

    table().add_annotation({**fresh, "unit_id": 7, "annotation": "fresh"})

    assert table.audit_positional_unit_ids(merge_ids=[case["fresh_id"]]).empty
    plan = table.migrate_positional_unit_ids(dry_run=False)
    assert case["fresh_id"] not in plan
    assert _unit_ids(table, case["fresh_id"]) == [7]


def _failed_first_write_rolls_back(case):
    """A failed first write leaves neither a marker nor a master row."""
    table = case["table"]
    fresh = {"spikesorting_merge_id": case["fresh_id"]}

    with _failing_insert1(table.Annotation):
        with pytest.raises(RuntimeError, match="injected"):
            table().add_annotation(
                {**fresh, "unit_id": 5, "annotation": "rolled-back"}
            )

    assert not (case["marker"] & fresh)
    assert not (table & fresh)


SCENARIOS = {
    "unmigrated_sparse_blocks_writes": _unmigrated_sparse_blocks_writes,
    "migrated_sparse_accepts_writes": _migrated_sparse_accepts_writes,
    "fresh_merge_needs_no_migration": _fresh_merge_needs_no_migration,
    "failed_first_write_rolls_back": _failed_first_write_rolls_back,
}


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_annotation_state_transitions(scenario, annotation_case):
    SCENARIOS[scenario](annotation_case)


def test_dense_namespace_accepts_write_and_is_marked(
    annotation_case, monkeypatch
):
    case = annotation_case
    table = case["table"]
    dense = {"spikesorting_merge_id": case["dense_id"]}
    new_key = {**dense, "unit_id": 1, "annotation": "dense"}

    assert table.audit_positional_unit_ids(merge_ids=[case["dense_id"]]).empty
    nwb_reads = _count_nwb_reads(monkeypatch)

    table().add_annotation(new_key)

    assert (table.Annotation & new_key).fetch1("unit_id") == 1
    # Nothing to migrate is exactly what the marker records, so the audit's
    # NWB read happens once and not on every later write.
    assert (case["marker"] & dense).fetch1("migration_version") == 1
    assert nwb_reads == [case["dense_id"]]

    table().add_annotation({**dense, "unit_id": 1, "annotation": "dense-2"})

    assert sorted((table.Annotation & dense).fetch("annotation")) == [
        "dense",
        "dense-2",
    ]
    assert nwb_reads == [case["dense_id"]]


def test_batch_annotation_writes_inside_caller_transaction(annotation_case):
    """Several annotations commit together under one caller transaction."""
    case = annotation_case
    table = case["table"]
    fresh = {"spikesorting_merge_id": case["fresh_id"]}

    with table.connection.transaction:
        table().add_annotation({**fresh, "unit_id": 5, "annotation": "one"})
        table().add_annotation({**fresh, "unit_id": 7, "annotation": "two"})

    assert (case["marker"] & fresh).fetch1("migration_version") == 1
    assert _unit_ids(table, case["fresh_id"]) == [5, 7]
    assert sorted((table.Annotation & fresh).fetch("annotation")) == [
        "one",
        "two",
    ]


def test_caller_transaction_failure_rolls_back_first_write(annotation_case):
    """The caller's rollback covers writes this method participated in."""
    case = annotation_case
    table = case["table"]
    fresh = {"spikesorting_merge_id": case["fresh_id"]}

    with pytest.raises(RuntimeError, match="caller failed"):
        with table.connection.transaction:
            table().add_annotation(
                {**fresh, "unit_id": 5, "annotation": "doomed"}
            )
            raise RuntimeError("caller failed")

    assert not (case["marker"] & fresh)
    assert not (table & fresh)
    assert not (table.Annotation & fresh)

"""Staging an ingestion plan: identity across attempts, and the two hashes."""

import pytest


@pytest.fixture
def staged(common, mini_copy_name):
    """A plan for the mini file, staged, cleared afterwards."""
    from spyglass.common.common_usage import IngestionPlanLog
    from spyglass.data_import.planner import plan_nwbfile

    log = IngestionPlanLog()
    key = log.stage(plan_nwbfile(mini_copy_name))
    yield log, key, plan_nwbfile(mini_copy_name)
    # Parts first: the master cannot go while its rows reference it.
    (log.Entry & key).delete_quick()
    (log.Problem & key).delete_quick()
    (log & key).delete_quick()


def test_a_plan_stages_every_entry_it_holds(staged):
    """The staging area mirrors the plan, entry for entry."""
    log, key, plan = staged
    entries = log.Entry & key

    assert len(entries) == plan.entry_count, (
        "Every planned row should be staged: "
        f"{len(entries)} staged vs {plan.entry_count} planned"
    )
    assert (log & key).fetch1("verdict") == plan.verdict
    assert (log & key).fetch1("status") == "open", "A fresh plan is open"


def test_the_two_hashes_answer_different_questions(staged):
    """`key_hash` is identity; `blob_hash` is content.

    If both covered the whole entry they could never disagree, and
    divergence -- same key, different values -- would be indistinguishable
    from novelty. That is the whole reason there are two.
    """
    log, key, _ = staged
    rows = (log.Entry & key).fetch(as_dict=True)

    assert any(
        row["key_hash"] != row["blob_hash"] for row in rows
    ), "key_hash must not simply repeat blob_hash"

    # Identity is per table: two tables keyed only by nwb_file_name will
    # legitimately share a key_hash, which is why table_name is in the key.
    per_table = {}
    for row in rows:
        per_table.setdefault(row["table_name"], []).append(row["key_hash"])
    for table_name, hashes in per_table.items():
        assert len(hashes) == len(
            set(hashes)
        ), f"{table_name} staged two entries under one key_hash"


def test_restaging_updates_rather_than_appends(staged):
    """An entry keeps its identity across attempts.

    This is what lets a second attempt stage N+M where the first staged N,
    rather than accumulating duplicates of the N it already had.
    """
    log, key, plan = staged
    first_attempt = (log & key).fetch1("attempt")
    before = {
        (row["table_name"], row["key_hash"])
        for row in (log.Entry & key).fetch(as_dict=True)
    }

    log.stage(plan)

    after = {
        (row["table_name"], row["key_hash"])
        for row in (log.Entry & key).fetch(as_dict=True)
    }
    assert after == before, "Re-planning the same file restages the same rows"
    assert (log & key).fetch1(
        "attempt"
    ) == first_attempt + 1, "The attempt counter advances"
    assert len(log & key) == 1, "One live plan per file"


def test_entries_carry_their_payload_and_state(staged):
    """A staged entry is re-insertable without re-parsing."""
    log, key, _ = staged
    rows = (log.Entry & key).fetch(as_dict=True)

    assert all(
        row["state"] in {"planned", "blocked", "failed"} for row in rows
    ), "A freshly staged plan holds no inserted or migrated entries"

    payloads = [row["entry_blob"] for row in rows if row["entry_blob"]]
    assert payloads, "Entries should keep their blob until they are migrated"
    assert all(
        isinstance(blob, dict) for blob in payloads
    ), "A staged entry round-trips as the mapping it was"


def test_inserting_a_plan_closes_its_staging_area(common, mini_copy_name):
    """A stored entry keeps its hashes and loses its payload.

    The invariant from the design: no `entry_blob` is retained for an entry
    that exists in its own table. Keeping it would make the log a second copy
    of the data, which is the failure this whole shape exists to avoid.
    """
    from spyglass.common.common_usage import IngestionPlanLog
    from spyglass.data_import.planner import insert_plan, plan_nwbfile

    log = IngestionPlanLog()
    key = log.stage(plan_nwbfile(mini_copy_name))

    result = insert_plan(plan_nwbfile(mini_copy_name), on_divergence="accept")

    assert not result, f"Nothing should have blocked:\n{result}"
    assert (
        len(log.Entry & key & "entry_blob IS NOT NULL") == 0
    ), "A stored entry must not keep its payload"
    assert all(
        state in {"exists", "inserted"}
        for state in (log.Entry & key).fetch("state")
    ), "Every entry should be accounted for once the plan is applied"
    assert (log & key).fetch1("status") == "complete"

    (log.Entry & key).delete_quick()
    (log.Problem & key).delete_quick()
    (log & key).delete_quick()


def test_entries_are_keyed_by_where_the_row_is_going(common, mini_copy_name):
    """A table plans rows for other tables, and those are staged by target.

    `TaskEpoch` plans `Task` rows; `PositionSource` plans `RawPosition` and
    `IntervalList` rows. Recording those under the planning table's name
    instead of the target's leaves them unmatched, and so unmigrated.
    """
    from spyglass.common.common_usage import IngestionPlanLog
    from spyglass.data_import.planner import plan_nwbfile

    log = IngestionPlanLog()
    plan = plan_nwbfile(mini_copy_name)
    key = log.stage(plan)

    staged = set((log.Entry & key).fetch("table_name"))
    planning = {table_plan.table_name for table_plan in plan.table_plans}

    assert staged - planning, (
        "Expected entries staged under tables that plan nothing themselves, "
        "e.g. Task or RawPosition"
    )

    (log.Entry & key).delete_quick()
    (log.Problem & key).delete_quick()
    (log & key).delete_quick()

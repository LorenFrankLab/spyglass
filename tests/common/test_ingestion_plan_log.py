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
    log._clear(key)  # parts then master, in one place


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
    that exists in its own table *and matches it*. Keeping it would make the
    log a second copy of the data, which is the failure this whole shape
    exists to avoid.

    This file holds no divergence, so every entry here is in that case. The
    one exception is pinned separately, below.
    """
    from spyglass.common.common_usage import IngestionPlanLog
    from spyglass.data_import.planner import insert_plan, plan_nwbfile

    log = IngestionPlanLog()
    key = log.stage(plan_nwbfile(mini_copy_name))

    result = insert_plan(plan_nwbfile(mini_copy_name))

    assert not result, f"Nothing should have blocked:\n{result}"
    assert (
        len(log.Entry & key & "entry_blob IS NOT NULL") == 0
    ), "A stored entry must not keep its payload"
    assert all(
        state in {"exists", "inserted"}
        for state in (log.Entry & key).fetch("state")
    ), "Every entry should be accounted for once the plan is applied"
    assert (log & key).fetch1("status") == "complete"

    log._clear(key)  # parts then master, in one place


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

    log._clear(key)  # parts then master, in one place


def test_staging_records_each_table_and_its_read_set(
    common, mini_copy_name, mini_insert
):
    """A staged plan keeps per-table provenance, not just per-entry.

    The read-set digest is what lets a later attempt skip a table whose inputs
    did not change. Storing it per table is the half of that which the log has
    to carry; the comparison itself belongs to the planner.
    """
    from spyglass.common.common_usage import IngestionPlanLog
    from spyglass.data_import.planner import plan_nwbfile

    plan = plan_nwbfile(mini_copy_name)
    IngestionPlanLog().stage(plan)

    key = {"nwb_file_name": mini_copy_name}
    staged = IngestionPlanLog.Table & key

    assert len(staged) == len(
        plan.table_plans
    ), "One row per table the plan covers"

    rows = staged.fetch(as_dict=True)
    by_name = {row["table_name"]: row for row in rows}

    for table_plan in plan.table_plans:
        row = by_name[table_plan.table_name]
        assert row["status"] == table_plan.status
        assert row["entry_count"] == table_plan.entry_count
        assert row["read_set_digest"] == table_plan.read_set_digest

    digests = [r["read_set_digest"] for r in rows if r["read_set_digest"]]
    assert digests, "The mini file is hashable, so digests should be present"
    assert len(set(digests)) > 1, (
        "Different tables read different objects, so their digests must "
        "differ -- one digest for everything would make reuse meaningless"
    )


def test_restaging_replaces_the_table_rows(common, mini_copy_name, mini_insert):
    """Re-planning updates the per-table rows rather than appending them.

    One live plan per file: a second attempt describes the same file, so its
    table rows replace the first attempt's.
    """
    from spyglass.common.common_usage import IngestionPlanLog
    from spyglass.data_import.planner import plan_nwbfile

    key = {"nwb_file_name": mini_copy_name}
    log = IngestionPlanLog()

    log.stage(plan_nwbfile(mini_copy_name))
    first = len(IngestionPlanLog.Table & key)

    log.stage(plan_nwbfile(mini_copy_name))
    second = len(IngestionPlanLog.Table & key)

    assert first == second, f"Table rows accumulated: {first} -> {second}"


def test_a_conflicting_entry_keeps_its_payload(
    common, mini_copy_name, mini_insert, monkeypatch
):
    """The one entry that keeps its blob, and why it has to.

    D7 made a divergence a warning rather than a prompt, which only works if
    the reader can act on it afterwards: the planned value is the thing they
    need, and re-deriving it means re-parsing the file. So a `conflict` row
    keeps its payload where `exists` and `inserted` lose theirs.

    Written because the invariant test above cannot see this case — the mini
    file produces no divergence, so it asserted "no blob survives, states are
    exists or inserted" and passed for want of a counter-example rather than
    because the rule held.
    """
    from spyglass.common.common_usage import IngestionPlanLog
    from spyglass.data_import.planner import insert_plan, plan_nwbfile
    from spyglass.utils.ingestion_plan import PlannedEntries

    target = common.SampleCount().full_table_name

    def _changed(self, source, ctx):
        entries = PlannedEntries()
        entries.add(
            self, [dict(ctx.base_key, sample_count_object_id="deadbeef" * 5)]
        )
        return entries

    monkeypatch.setattr(
        type(common.SampleCount()), "entries_for_row", _changed, raising=False
    )

    log = IngestionPlanLog()
    log.clear(mini_copy_name)
    plan = plan_nwbfile(mini_copy_name, force_replan=True)

    assert any(
        p.code == "divergence" and p.table == target for p in plan.problems
    ), "Premise: SampleCount disagrees with the stored row"

    key = log.stage(plan)
    result = insert_plan(plan)

    assert not result, f"A divergence must not block:\n{result}"

    rows = (log.Entry & key).fetch(as_dict=True)
    conflicts = [r for r in rows if r["state"] == "conflict"]
    assert conflicts, "The disagreeing row must be staged as a conflict"
    assert all(
        r["entry_blob"] is not None for r in conflicts
    ), "A conflict keeps the planned value, or the warning cannot be acted on"

    others = [r for r in rows if r["state"] != "conflict"]
    assert all(
        r["entry_blob"] is None for r in others
    ), "Everything else still loses its payload"
    assert {r["state"] for r in others} <= {"exists", "inserted"}

    log._clear(key)


def test_clearing_a_plan_leaves_no_part_behind(common, mini_copy_name):
    """Every part is cleared, including one added after this was written.

    Derived from `parts()` on both sides -- the code that clears and the
    assertion that checks it -- so a fourth part is covered without anyone
    remembering to extend either. The hardcoded list this replaced announced
    itself by breaking every call site with a foreign-key error when `Table`
    was added, because `delete_quick` does not cascade.
    """
    from spyglass.common.common_usage import IngestionPlanLog
    from spyglass.data_import.planner import plan_nwbfile

    log = IngestionPlanLog()
    key = log.stage(plan_nwbfile(mini_copy_name, force_replan=True))

    parts = log.parts(as_objects=True)
    assert parts, "Premise: the master has parts"
    assert any(
        len(part & key) for part in parts
    ), "Premise: staging filled at least one of them"

    log._clear(key)

    for part in parts:
        assert not len(part & key), f"{part.full_table_name} kept rows"
    assert not len(log & key), "and the master row is gone"

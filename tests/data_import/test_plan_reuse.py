"""Per-table plan reuse: re-parse only what changed (D8).

The workflow this exists for is ingest → read the report → edit the file → try
again. Between attempts the file changes, so a whole-file cache key is useless;
what survives is per table, keyed on the NWB objects that table actually read.

These tests assert the invariant by counting **parses**, not by timing.
"""

import pytest


@pytest.fixture
def staged(common, mini_copy_name, mini_insert):
    """A staged plan for the mini file, so there is something to reuse."""
    from spyglass.common.common_usage import IngestionPlanLog
    from spyglass.data_import.planner import plan_nwbfile

    log = IngestionPlanLog()
    log.stage(plan_nwbfile(mini_copy_name))

    yield log

    log.clear(mini_copy_name)


def test_an_unchanged_file_reuses_every_table_that_found_something(
    common, mini_copy_name, staged, parse_counter
):
    """Nothing changed, so no table that planned anything parses again.

    Not *zero* parses, and the exception is not a shortfall. A table that found
    no source object read nothing, so its read-set is empty and its digest is a
    digest of nothing -- add the object it was looking for and that digest is
    still unchanged. An empty read-set can detect a modification but not an
    appearance, so those tables must always parse. They are also the cheap
    ones: finding no source is an early return.
    """
    from spyglass.data_import.planner import plan_nwbfile

    before = plan_nwbfile(mini_copy_name, force_replan=True)
    productive = {tp.table_name for tp in before.table_plans if tp.entry_count}
    assert productive, "The mini file should fill some tables"

    parse_counter.clear()
    plan = plan_nwbfile(mini_copy_name)

    assert plan.entry_count > 0, "The plan should still describe the file"
    reparsed = productive & set(parse_counter)
    assert not reparsed, (
        "A table that planned entries should not parse again when nothing it "
        + f"read changed; re-parsed: {sorted(reparsed)}"
    )


def test_force_replan_parses_anyway(
    common, mini_copy_name, staged, parse_counter
):
    """The escape hatch, for when the suspicion is the reuse check itself."""
    from spyglass.data_import.planner import plan_nwbfile

    plan_nwbfile(mini_copy_name, force_replan=True)

    assert parse_counter, "force_replan must parse regardless of what is staged"


def test_a_changed_read_set_re_parses_only_its_table(
    common, mini_copy_name, staged, parse_counter, monkeypatch
):
    """Editing one object re-parses the tables that read it, and no others.

    Simulated by making the hasher report a different digest for one table's
    read-set, which is what an edit to an object it read would do. The point is
    the *scope* of invalidation: an unrelated fix must cost nothing.
    """
    from spyglass.common.common_usage import IngestionPlanLog
    from spyglass.data_import.planner import plan_nwbfile
    from spyglass.utils.nwb_hash import NwbfileHasher

    target = common.SampleCount().full_table_name
    stored = (
        IngestionPlanLog.Table
        & {"nwb_file_name": mini_copy_name, "table_name": target}
    ).fetch1()
    changed = set(stored["reads"] or [])
    assert changed, "Premise: SampleCount recorded what it read"

    original = NwbfileHasher.read_set_digest

    def shifted(self, object_ids):
        digest = original(self, object_ids)
        if set(object_ids) == changed:  # as if one of these objects was edited
            return "0" * len(digest)
        return digest

    monkeypatch.setattr(NwbfileHasher, "read_set_digest", shifted)

    plan_nwbfile(mini_copy_name)

    parsed = set(parse_counter)
    assert target in parsed, "The table whose read-set changed must re-parse"

    # Tables that found nothing always parse again -- see the test above --
    # so the claim is about the ones that did find something.
    productive = {
        tp.table_name
        for tp in plan_nwbfile(mini_copy_name, force_replan=True).table_plans
        if tp.entry_count
    }
    collateral = (parsed & productive) - {target}
    assert not collateral, (
        "Only the table whose read-set changed should re-parse; also "
        + f"re-parsed: {sorted(collateral)}"
    )


def test_a_reused_plan_describes_the_same_entries(
    common, mini_copy_name, staged
):
    """Reuse must be invisible in the result, or it is not reuse.

    Same entries per table, from storage as from a parse. If these diverged,
    the saving would be bought with a plan that no longer describes the file.
    """
    from spyglass.data_import.planner import plan_nwbfile

    reused = plan_nwbfile(mini_copy_name)
    parsed = plan_nwbfile(mini_copy_name, force_replan=True)

    def per_target(plan):
        return {
            (tp.table_name, getattr(t, "full_table_name", str(t))): len(rows)
            for tp in plan.table_plans
            for t, rows in tp.entries
            if rows
        }

    assert per_target(reused) == per_target(
        parsed
    ), "A reused plan and a freshly parsed one must hold the same entries"
    assert reused.verdict == parsed.verdict


def test_nothing_staged_means_parse(
    common, mini_copy_name, mini_insert, parse_counter
):
    """With no staged plan there is nothing to reuse, so everything parses.

    Stated because the alternative -- treating "no record" as "unchanged" --
    is the failure mode that would quietly plan an empty file.
    """
    from spyglass.common.common_usage import IngestionPlanLog
    from spyglass.data_import.planner import plan_nwbfile

    IngestionPlanLog().clear(mini_copy_name)

    plan_nwbfile(mini_copy_name)

    assert parse_counter, "Nothing staged must mean everything parses"

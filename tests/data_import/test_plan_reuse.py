"""Per-table plan reuse: re-parse only what changed.

The workflow this exists for is ingest → read the report → edit the file → try
again. Between attempts the file changes, so a whole-file cache key is useless;
what survives is per table, keyed on the NWB objects that table actually read.

These tests assert the invariant by counting **parses**, not by timing.
"""

import pytest


@pytest.fixture
def staged(common, mini_copy_name, mini_insert, fresh_plan):
    """A staged plan for the mini file, so there is something to reuse.

    Cleared and re-parsed rather than planned from whatever is already
    staged: a plan built *through* reuse inherits whatever an earlier test
    left behind, so these tests would be asserting about that instead of
    about reuse. Seen as a plan with zero entries reaching
    `test_an_unchanged_file_reuses_every_table_that_found_something`.
    """
    from spyglass.common.common_usage import IngestionPlanLog
    from spyglass.data_import.planner import plan_nwbfile

    log = IngestionPlanLog()
    log.stage(fresh_plan(mini_copy_name))

    yield log

    log.clear(mini_copy_name)


def test_an_unchanged_file_reuses_every_table_that_found_something(
    common, mini_copy_name, staged, parse_counter
):
    """Nothing changed, so no table that planned anything parses again.

    Not *zero* parses, and the exception is not a shortfall. A table that found
    no source object read nothing, so its read-set is empty and its digest is a
    digest of nothing. Add the object it was looking for and that digest is
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

    # Tables that found nothing always parse again, as the test above shows,
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

    Stated because the alternative, treating "no record" as "unchanged", is
    the failure mode that would quietly plan an empty file.
    """
    from spyglass.common.common_usage import IngestionPlanLog
    from spyglass.data_import.planner import plan_nwbfile

    IngestionPlanLog().clear(mini_copy_name)

    plan_nwbfile(mini_copy_name)

    assert parse_counter, "Nothing staged must mean everything parses"


def test_a_plan_from_another_spyglass_version_is_not_reused(
    common, mini_copy_name, staged, parse_counter
):
    """A code change invalidates a staged plan; the file alone cannot say so.

    The read-set digests cover NWB objects, so they notice the *file*
    changing and nothing else. Reuse fires precisely where a previous attempt
    left an incomplete plan, which is precisely where a Spyglass-side fix has
    just changed what a parse would produce. Without this, a user whose
    ingest was blocked by a bug, who upgrades and retries, is served the plan
    the broken version built.
    """
    from spyglass.common.common_usage import IngestionPlanLog
    from spyglass.data_import.planner import plan_nwbfile

    log = IngestionPlanLog()
    key = {"nwb_file_name": mini_copy_name}
    assert log & key, "Premise: the fixture staged a plan"

    # Asserted on the tables that planned something, never on "any parse at
    # all": a table that found no source reads nothing and so always
    # re-parses, and a bare `assert parse_counter` is satisfied by those nine
    # whether the gate works or not. Sabotaging the gate proved exactly that.
    productive = {
        tp.table_name
        for tp in plan_nwbfile(mini_copy_name, force_replan=True).table_plans
        if tp.entry_count
    }
    assert productive, "Premise: the mini file fills some tables"

    # Stand in for the upgrade, after the force_replan above restaged under
    # the real version.
    log.update1({**key, "spyglass_version": "0.0.0-not-this-one"})

    parse_counter.clear()
    plan = plan_nwbfile(mini_copy_name)

    missed = productive - set(parse_counter)
    assert not missed, (
        "A plan staged by a different version must not be reused; reused "
        + f"instead: {sorted(missed)}"
    )
    assert plan.entry_count > 0, "and the re-parse must still describe the file"


def test_a_changed_stored_row_re_parses_the_table_that_read_it(
    common, mini_copy_name, mini_insert, parse_counter
):
    """A cross-reference resolved out of another table joins the reuse key.

    `DIOEvents` reads `Raw.interval_list_name` and writes it into every one
    of its own entries, so reuse here does not merely cost nothing: it
    serves a plan naming an interval the insert would not write.

    Planned for `DIOEvents` alone so the lookup resolves against the stored
    `Raw` row: with `Raw` in the same pass its *planned* row wins, and no
    edit to storage could be seen.
    """
    from spyglass.common.common_usage import IngestionPlanLog
    from spyglass.data_import.planner import plan_nwbfile

    log = IngestionPlanLog()
    file_restr = {"nwb_file_name": mini_copy_name}
    raw = common.Raw() & file_restr
    assert raw, "Premise: the mini file's Raw row is stored"

    was = raw.fetch1("interval_list_name")
    names = (common.IntervalList & file_restr).fetch("interval_list_name")
    other = next(name for name in names if name != was)

    def plan_dio():
        log.clear(mini_copy_name)
        plan = plan_nwbfile(mini_copy_name, tables=[common.DIOEvents])
        log.stage(plan)
        return plan

    first = plan_dio()
    dio = common.DIOEvents().full_table_name
    staged = {tp.table_name: tp for tp in first.table_plans}[dio]
    assert staged.status == "ok" and staged.entry_count, (
        "Premise: DIOEvents plans entries on its own, so there is something "
        + f"to reuse; status {staged.status}, {staged.entry_count} entries"
    )

    def intervals_named(plan):
        return {
            row.get("interval_list_name")
            for tp in plan.table_plans
            for target, rows in tp.entries
            for row in rows
            if getattr(target, "full_table_name", None) == dio
        }

    assert intervals_named(first) == {was}, (
        "Premise: DIOEvents writes the interval it read from Raw into its "
        + "own entries"
    )
    assert staged.xrefs, (
        "Premise: the parse recorded what it resolved out of Raw; without "
        + "it this test cannot distinguish the fix from its absence"
    )

    try:
        common.Raw().update1({**raw.fetch1("KEY"), "interval_list_name": other})

        parse_counter.clear()
        second = plan_nwbfile(mini_copy_name, tables=[common.DIOEvents])

        assert dio in set(parse_counter), (
            "A table whose cross-reference now resolves differently must "
            + "re-parse; the file alone cannot say so"
        )
        assert intervals_named(second) == {other}, (
            "and the re-planned entries must name the interval Raw holds "
            + f"now, not {was}"
        )
    finally:
        common.Raw().update1({**raw.fetch1("KEY"), "interval_list_name": was})
        log.clear(mini_copy_name)


def test_a_staged_plan_records_its_provenance(
    common, mini_copy_name, mini_insert
):
    """The file hash and version are recorded, not left NULL.

    Both columns existed from the start and were filled only by the plan
    cache, which nothing enabled, so every staged plan carried NULL until the
    version became a reuse gate. The hash costs nothing extra, since the
    read-set index already computes it.
    """
    from spyglass import __version__
    from spyglass.common.common_usage import IngestionPlanLog
    from spyglass.data_import.planner import plan_nwbfile

    log = IngestionPlanLog()
    log.clear(mini_copy_name)
    log.stage(plan_nwbfile(mini_copy_name, force_replan=True))

    row = (log & {"nwb_file_name": mini_copy_name}).fetch1()

    assert row["spyglass_version"] == __version__
    assert row["nwb_hash"], "The file hash is computed anyway; record it"

    log.clear(mini_copy_name)


def test_a_changed_config_re_parses(
    common, mini_copy_name, parse_counter, clean_staging
):
    """A config edit invalidates the staged plan.

    Reuse gates on the NWB reads and the cross-references, neither of which a
    config edit touches, so without the config digest an edited override was
    reported as reused and the old rows served.
    """
    from spyglass.common.common_usage import IngestionPlanLog
    from spyglass.data_import.planner import plan_nwbfile

    log = IngestionPlanLog()
    log.clear(mini_copy_name)

    first = {"Electrode": [{"electrode_id": 0, "name": "from_first_config"}]}
    log.stage(plan_nwbfile(mini_copy_name, config=first, force_replan=True))

    # Same config: the gate passes, so reuse is available.
    parse_counter.clear()
    plan_nwbfile(mini_copy_name, config=first)
    reused = set(parse_counter)

    # Changed config: every table re-parses, the whole-file digest being
    # deliberately coarser than the keys each table reads.
    parse_counter.clear()
    changed = {"Electrode": [{"electrode_id": 0, "name": "from_second"}]}
    plan_nwbfile(mini_copy_name, config=changed)
    reparsed = set(parse_counter)

    assert len(reparsed) > len(reused), (
        "A changed config must invalidate the staged plan; re-parsed "
        f"{len(reparsed)} tables against {len(reused)} with the config "
        "unchanged"
    )

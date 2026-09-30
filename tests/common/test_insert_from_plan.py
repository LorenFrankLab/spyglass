"""Inserting from a plan instead of table by table.

The payoff of planning: check the whole file, then write what was checked. The
tests below are mostly about equivalence -- a planned run must land the same
rows as the per-table run it replaces -- and about the one thing the per-table
run could not do, which is decline to half-ingest a file.
"""

import pytest

# Tables spanning the shapes an ingestion produces: a master, a part, a
# blob-carrying table, one fed by another table, and one shared across files.
WATCHED = (
    "Session",
    "Electrode",
    "ElectrodeGroup",
    "IntervalList",
    "TaskEpoch",
    "Task",
    "Raw",
    "SampleCount",
    "DIOEvents",
    "SensorData",
    "VideoFile",
)


@pytest.fixture
def counts(common, mini_restr):
    """Row counts per watched table, scoped to the mini file where possible."""

    def _counts():
        found = {}
        for name in WATCHED:
            table = getattr(common, name)()
            # `Task` is shared across files and is not file-keyed, so count all
            # of it; everything else is restricted to the file under test.
            restricted = (
                table
                if "nwb_file_name" not in table.primary_key
                else table & mini_restr
            )
            found[name] = len(restricted)
        return found

    return _counts


@pytest.fixture
def emptied_leaves(common, mini_copy_name, mini_insert):
    """Remove two leaf tables' rows so an insert has something to do.

    `super_delete` with `safemode=False`: the cautious path follows a row
    delete by removing the NWB file from disk, and the prompt would hang a
    suite. Restored afterwards regardless of how the test ends, so the next
    test sees a complete file whether or not this one passed.
    """
    tables = [common.SampleCount(), common.DIOEvents()]
    key = {"nwb_file_name": mini_copy_name}

    for table in tables:
        if not (table & key):
            table.insert_from_nwbfile(mini_copy_name)
        (table & key).super_delete(warn=False, safemode=False)

    yield tables, key

    for table in tables:
        if not (table & key):
            table.insert_from_nwbfile(mini_copy_name)


def test_planned_run_inserts_what_the_plan_reported(
    common, mini_copy_name, emptied_leaves, counts
):
    """The two passes agree: what was planned is what lands.

    The whole point of re-deriving nothing. The first pass checked these rows;
    the second writes those same rows rather than parsing the file again.
    """
    from spyglass.common.populate_all_common import populate_all_common

    tables, key = emptied_leaves
    before = counts()

    result = populate_all_common(
        mini_copy_name, use_plan=True, on_divergence="accept"
    )

    assert not result, f"A good file should insert cleanly: {list(result)}"
    for table in tables:
        assert table & key, f"{table.camel_name} was planned but not inserted"

    after = counts()
    for name in ("SampleCount", "DIOEvents"):
        assert after[name] > before[name], f"{name} gained no rows"


def test_planned_run_matches_the_per_table_run(
    common, mini_copy_name, emptied_leaves, counts
):
    """Equivalence with the path it replaces, table for table.

    A different set of rows would make this a rewrite rather than a
    refactor, however much tidier the new path reads.
    """
    from spyglass.common.populate_all_common import populate_all_common

    tables, key = emptied_leaves

    populate_all_common(mini_copy_name, use_plan=True, on_divergence="accept")
    planned = counts()

    # Empty the same tables again and let the legacy path refill them.
    for table in tables:
        (table & key).super_delete(warn=False, safemode=False)
    populate_all_common(mini_copy_name)
    legacy = counts()

    assert planned == legacy, (
        "A planned run and a per-table run must leave the same rows; "
        + f"planned={planned} legacy={legacy}"
    )


def test_planned_run_is_idempotent(common, mini_copy_name, mini_insert, counts):
    """A second planned run on a finished file adds nothing and says so.

    The per-table path logged a duplicate error per table here. A plan answers
    in one word, and writes nothing.
    """
    from spyglass.common.populate_all_common import populate_all_common

    before = counts()

    result = populate_all_common(
        mini_copy_name, use_plan=True, on_divergence="accept"
    )

    assert counts() == before, "A finished file should gain no rows"
    assert result.verdict == "no_op", f"Expected no_op, got {result.verdict}"
    assert not result, "Nothing to do is not a failure"


def test_planned_run_writes_nothing_when_the_plan_blocks(
    common, mini_copy_name, emptied_leaves, counts, monkeypatch
):
    """The behaviour the per-table path could not offer.

    Inserting table by table, a file with one bad table still wrote every table
    before it. Planning first means a blocking problem stops the whole run, so
    the file is left as it was rather than partly ingested.
    """
    from spyglass.common.populate_all_common import populate_all_common
    from spyglass.utils.mixins.ingestion import IngestionMixin

    tables, key = emptied_leaves
    before = counts()

    original = IngestionMixin._parse

    def _parse(self, ctx):
        if self.camel_name == "SampleCount":
            raise ValueError("induced failure")
        return original(self, ctx)

    monkeypatch.setattr(IngestionMixin, "_parse", _parse)

    result = populate_all_common(
        mini_copy_name, use_plan=True, on_divergence="accept"
    )

    assert result, "A file with a broken table must not report success"
    assert counts() == before, (
        "A blocking problem must write nothing at all, not everything up to "
        + "the failure"
    )


def test_raise_err_raises_after_the_whole_file_is_checked(
    common, mini_copy_name, emptied_leaves, monkeypatch
):
    """`raise_err` still raises, but only once the report is complete.

    The difference that matters: the old path raised at the first bad table, so
    the report named one problem. Here the pass finishes, every problem is
    collected, and the exception carries the lot.
    """
    from spyglass.common.common_usage import IngestionPlanLog
    from spyglass.common.populate_all_common import populate_all_common
    from spyglass.utils.mixins.ingestion import IngestionMixin

    # Nothing staged, so nothing is reusable -- otherwise the tables this test
    # breaks are served from the last attempt and never parse at all.
    IngestionPlanLog().clear(mini_copy_name)

    original = IngestionMixin._parse
    broken = {"SampleCount", "DIOEvents"}

    def _parse(self, ctx):
        if self.camel_name in broken:
            raise ValueError(f"induced failure in {self.camel_name}")
        return original(self, ctx)

    monkeypatch.setattr(IngestionMixin, "_parse", _parse)

    with pytest.raises(ValueError) as err:
        populate_all_common(
            mini_copy_name,
            use_plan=True,
            raise_err=True,
            on_divergence="accept",
        )

    message = str(err.value)
    for name in broken:
        assert (
            name.lower() in message.lower() or name in message
        ), f"The report should name {name}; got:\n{message}"


def test_the_default_path_is_unchanged(common, mini_copy_name, mini_insert):
    """D4: planning is opt-in until a later release flips it.

    Without `use_plan`, the return value is still the old one -- None, or a
    list of InsertError keys -- not a plan.
    """
    from spyglass.common.populate_all_common import populate_all_common
    from spyglass.utils.ingestion_plan import IngestionPlan

    result = populate_all_common(mini_copy_name)

    assert not isinstance(
        result, IngestionPlan
    ), "The default path must not start returning a plan"

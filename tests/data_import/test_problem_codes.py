"""Every problem code the planner emits, caused rather than constructed.

The report is the only channel a user has: nothing prompts, and a plan that
blocks says why here or nowhere. So each code needs a test that makes the
system *emit* it, which a hand-built `Problem` carrying the same string does
not do -- that checks the renderer, not the detector.
"""

import pytest

from spyglass.utils.ingestion_plan import BLOCKING, REMEDIES


def _codes(plan):
    return {problem.code for problem in plan.problems}


def _emit_rows(table_type, rows):
    """Replace a table's parse output with `rows`."""

    def _entries(self, source, ctx):
        from spyglass.utils.ingestion_plan import PlannedEntries

        entries = PlannedEntries()
        entries.add(self, [dict(ctx.base_key, **row) for row in rows])
        return entries

    return _entries


def test_missing_attribute_when_a_required_column_is_absent(
    common, mini_copy_name, mini_insert
):
    """A required column the row does not supply blocks its table.

    Asserted against `check_planned_rows` rather than a whole plan, because
    a parse can no longer produce such a row for a table that adjusts its own
    keys: the pass that collapses entries drops anything whose required
    attributes are absent. The detector still matters for targets that do not
    adjust -- a parent emitted alongside another table's rows -- and for any
    future caller that hands it a row directly.
    """
    from spyglass.data_import.planner import VirtualKeySpace

    institution = common.Institution()
    problems = institution.check_planned_rows(
        [{}], VirtualKeySpace(), table=institution
    )

    codes = {problem.code for problem in problems}
    assert "missing_attribute" in codes
    assert all(
        problem.severity in BLOCKING
        for problem in problems
        if problem.code == "missing_attribute"
    ), "An absent required column must block, not warn"


def test_value_too_long_when_a_string_exceeds_its_column(
    common, mini_copy_name, mini_insert, monkeypatch
):
    """A value the column cannot hold is caught before the insert raises."""
    from spyglass.data_import.planner import plan_nwbfile

    limit = int(
        common.Institution()
        .heading.attributes["institution_name"]
        .type[len("varchar(") : -1]
    )
    monkeypatch.setattr(
        type(common.Institution()),
        "entries_for_row",
        _emit_rows(
            common.Institution, [{"institution_name": "x" * (limit + 5)}]
        ),
        raising=False,
    )

    plan = plan_nwbfile(mini_copy_name, force_replan=True)

    assert "value_too_long" in _codes(plan)
    assert any(
        str(limit) in problem.message
        for problem in plan.problems
        if problem.code == "value_too_long"
    ), "The message should say what the column holds"


def test_file_not_registered_when_nothing_names_the_file(common):
    """Planning a file Spyglass has never seen, with no handle to read."""
    from spyglass.data_import.planner import plan_nwbfile

    plan = plan_nwbfile("_no_such_file_at_all_.nwb")

    assert "file_not_registered" in _codes(plan)
    assert plan.verdict == "fatal", "A file that cannot be read is not a no-op"
    assert plan, "and the plan must be truthy, carrying why"


def test_file_unreadable_when_the_file_is_not_nwb(common, raw_dir, mini_insert):
    """A registered path that pynwb cannot open."""
    from spyglass.common.common_nwbfile import Nwbfile
    from spyglass.data_import.planner import plan_nwbfile

    name = "notactually_.nwb"
    path = raw_dir / name
    path.write_bytes(b"this is not an HDF5 file")
    Nwbfile.insert_from_relative_file_name(name)

    try:
        plan = plan_nwbfile(name)

        assert "file_unreadable" in _codes(plan)
        assert plan.verdict == "fatal"
    finally:
        (Nwbfile & {"nwb_file_name": name}).delete_quick()
        path.unlink(missing_ok=True)


def test_entry_too_large_stages_hashes_without_the_payload(
    common, mini_copy_name, mini_insert, monkeypatch
):
    """Over the staging cap, an entry keeps its hashes and loses its blob.

    The cap is driven down rather than the entry inflated: a genuinely
    oversized row would make the test slow and the fixture unreadable, and
    the behaviour under test is the comparison, not the size.
    """
    from spyglass.common.common_usage import IngestionPlanLog
    from spyglass.data_import.planner import plan_nwbfile

    log = IngestionPlanLog()
    monkeypatch.setattr(type(log), "_entry_blob_cap", 8, raising=False)

    log.clear(mini_copy_name)
    key = log.stage(plan_nwbfile(mini_copy_name, force_replan=True))

    rows = (log.Entry & key).fetch(as_dict=True)
    assert rows, "Premise: the file stages entries"
    assert all(
        row["entry_blob"] is None for row in rows
    ), "Over the cap, no payload is kept"
    assert all(
        row["blob_hash"] for row in rows
    ), "but the hashes stay, so change detection still works"
    assert any(
        row["problem_code"] == "entry_too_large" for row in rows
    ), "and the entry says why it has no payload"

    log._clear(key)


@pytest.mark.parametrize(
    "code",
    sorted(
        {
            "divergence",
            "duplicate_key",
            "entry_too_large",
            "extension_unmet",
            "file_not_registered",
            "file_unreadable",
            "missing_attribute",
            "missing_parent",
            "parse_error",
            "planner_miss",
            "value_too_long",
        }
    ),
)
def test_every_blocking_code_offers_a_remedy(code):
    """A problem that stops work must say what to do about it.

    Checked per code so a new one added without guidance fails here rather
    than reaching a user as a bare message. `file_will_be_registered` is
    excluded deliberately: it is `info`, a note that something was resolved,
    and there is nothing for the reader to do.
    """
    assert code in REMEDIES, f"{code} reaches users with no guidance"
    assert REMEDIES[code].strip(), f"{code} has an empty remedy"


def test_blocking_is_the_only_definition_of_blocking():
    """Severity decides; no code is special-cased into or out of blocking."""
    assert BLOCKING == ("fatal", "hard")


def test_one_plan_pass_opens_the_file_once(
    common, mini_copy_name, mini_insert, monkeypatch
):
    """Every table reads through the shared handle, not its own.

    Read-set capture depends on it: the context records what it hands a
    table, so a table that opened the file itself would read objects nothing
    recorded, leaving a read-set that is missing entries and a digest that
    cannot notice them changing. That is the bug class where a table's parse
    looks reusable and is not.

    Asserted by counting opens rather than by inspection, because the
    property is about what runs, and a new table added with its own
    `get_nwb_file` call would pass review and fail here.

    The cache is cleared first so the count does not depend on test order:
    open handles are cached, so a warm pass opens nothing and the same
    assertion would read as a pass for the wrong reason.
    """
    import pynwb

    from spyglass.data_import.planner import plan_nwbfile
    from spyglass.utils import nwb_helper_fn
    from spyglass.utils.nwb_helper_fn import close_nwb_files

    opens = []
    real_io = pynwb.NWBHDF5IO

    class CountingIO(real_io):
        def __init__(self, *args, **kwargs):
            opens.append(args[:1] or kwargs.get("path"))
            super().__init__(*args, **kwargs)

    real_get = nwb_helper_fn.get_nwb_file

    def counting_get(*args, **kwargs):
        opens.append(args[:1])
        return real_get(*args, **kwargs)

    monkeypatch.setattr(pynwb, "NWBHDF5IO", CountingIO)
    monkeypatch.setattr(nwb_helper_fn, "get_nwb_file", counting_get)

    close_nwb_files()
    plan_nwbfile(mini_copy_name, force_replan=True)

    assert len(opens) == 1, (
        "A plan pass must open the file once and share it; opened "
        + f"{len(opens)} times: {opens}"
    )

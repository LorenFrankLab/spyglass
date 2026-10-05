"""What the ingestion report says, and to whom."""

import pytest


@pytest.fixture
def plan_types():
    """The plan dataclasses, which need no database."""
    from spyglass.utils import ingestion_plan

    return ingestion_plan


def _broken(plan_types):
    """A plan with one problem of each interesting shape."""
    return plan_types.IngestionPlan(
        nwb_file_name="broken_.nwb",
        novel={"`common`.`session`": 1},
        table_plans=(
            plan_types.TablePlan(
                table_name="`common`.`session`",
                entries=plan_types.PlannedEntries(),
                status="failed",
                problems=(
                    plan_types.Problem(
                        "hard",
                        "missing_attribute",
                        "institution_name is required and absent",
                        table="`common`.`session`",
                        nwb_object_id="abc-123",
                    ),
                    plan_types.Problem(
                        "soft",
                        "divergence",
                        "already stored with different values",
                        table="`common`.`subject`",
                        suggested_revision={"sex": "M"},
                        primary_key={"subject_id": "54321"},
                    ),
                    plan_types.Problem(
                        "soft",
                        "entry_too_large",
                        "over the cap",
                        table="`common`.`interval_list`",
                    ),
                ),
            ),
            plan_types.TablePlan(
                table_name="`common`.`electrode`",
                entries=plan_types.PlannedEntries(),
                status="blocked",
            ),
        ),
    )


def test_a_clean_plan_is_one_line(plan_types):
    """Padding a good result with empty sections trains people to skim."""
    report = plan_types.IngestionPlan(nwb_file_name="clean_.nwb").report(
        log=False
    )

    assert report.count("\n") == 0, f"Expected one line, got:\n{report}"
    assert "no_op" in report and "clean_.nwb" in report


def test_the_verdict_leads(plan_types):
    """The first line answers 'what happened', before any detail.

    A divergence no longer decides it: the verdict says what the run will do,
    and a disagreement about one stored row does not stop the rest (D7).
    """
    report = _broken(plan_types).report(log=False)
    first = report.splitlines()[0]

    assert first.startswith("broken_.nwb: partial_new"), first


def test_problems_group_by_severity_worst_first(plan_types):
    """Blocking problems come before advisory ones.

    One hard problem, not two: the divergence is soft now and renders in its
    own section rather than among the severities.
    """
    report = _broken(plan_types).report(verbose=True, log=False)

    assert report.index("Hard (1):") < report.index(
        "Soft (1):"
    ), "Hard problems must precede soft ones"
    assert "Hard (1)" in report, "The count tells you how much to read"


def test_a_problem_names_the_object_and_the_remedy(plan_types):
    """The message says what happened; the remedy says what to change."""
    report = _broken(plan_types).report(verbose=True, log=False)

    assert "(object abc-123)" in report, "The object id locates it in the file"
    assert "_spyglass_config.yaml" in report, "missing_attribute has a remedy"
    assert (
        report.count("-> The NWB file does not supply") == 1
    ), "A remedy is per code, not repeated under every occurrence"


def test_a_divergence_revision_is_pasteable(plan_types):
    """A suggested revision should be usable without retyping it.

    It renders inside the divergence section now, and in the *default*
    report: a warning the reader has to ask for is not a warning (D7).
    """
    report = _broken(plan_types).report(log=False)

    assert "{'sex': 'M'}" in report, "The revision renders as a dict literal"
    assert "Disagrees with stored rows (1):" in report
    assert "`common`.`subject`" in report, "grouped under its table"


def test_the_default_report_shows_warnings_but_not_resolutions(plan_types):
    """Soft problems show by default; only `info` is opt-in.

    Reversed with D7. Hiding soft problems meant a divergence, once it became
    a warning rather than a blocker, would have vanished from the report that
    replaced the prompt -- quieter than the behaviour it replaced.
    """
    quiet = _broken(plan_types).report(log=False)

    assert "Soft (1):" in quiet, "An advisory problem is still news"
    assert "Hard (1):" in quiet, "Blocking problems always show"

    plan = plan_types.IngestionPlan(
        nwb_file_name="chatty_.nwb",
        table_plans=(
            plan_types.TablePlan(
                table_name="`common`.`session`",
                entries=plan_types.PlannedEntries(),
                problems=(
                    plan_types.Problem(
                        "info",
                        "file_will_be_registered",
                        "planned as though registered",
                    ),
                ),
            ),
        ),
    )

    assert "Info (1):" not in plan.report(log=False), "info is opt-in"
    assert "Info (1):" in plan.report(verbose=True, log=False)


def test_blocked_tables_are_named_once_at_the_end(plan_types):
    """A blocked table is a consequence, not a separate failure."""
    report = _broken(plan_types).report(verbose=True, log=False)

    assert "Blocked by the above (1): `common`.`electrode`" in report


def test_entry_digest_sees_inside_a_large_array(plan_types):
    """A change-detection hash must not abbreviate what it hashes.

    `datajoint.hash.key_hash` hashes `str(value)`, and numpy renders a large
    array as `[0. 1. 2. ... 9997. 9998. 9999.]`. An edit in the middle of an
    `IntervalList.valid_times` hashes identically under it. That is the one
    answer a change-detection hash may not give.
    """
    import numpy as np
    from datajoint.hash import key_hash

    original = np.arange(10_000.0)
    edited = original.copy()
    edited[5_000] = -1.0

    one = {"nwb_file_name": "f_.nwb", "valid_times": original}
    other = {"nwb_file_name": "f_.nwb", "valid_times": edited}

    assert key_hash(one) == key_hash(
        other
    ), "Premise: key_hash cannot see this change, which is why we do not use it"
    assert plan_types.entry_digest(one) != plan_types.entry_digest(
        other
    ), "entry_digest must see a change anywhere in an array"


def test_entry_digest_notices_a_gained_or_lost_column(plan_types):
    """Attribute names are part of the content, not just their values."""
    base = {"a": 1}

    assert plan_types.entry_digest(base) != plan_types.entry_digest(
        {"b": 1}
    ), "The same value under a different name is a different entry"
    assert plan_types.entry_digest(base) != plan_types.entry_digest(
        {"a": 1, "c": None}
    ), "Gaining a column is a change"


def test_entry_digest_is_stable_across_numpy_and_python_scalars(plan_types):
    """A value fetched back from the database must hash as it went in."""
    import numpy as np

    assert plan_types.entry_digest({"epoch": 1}) == plan_types.entry_digest(
        {"epoch": np.int64(1)}
    ), "np.int64(1) and 1 are the same stored value"


def test_a_clean_plan_is_falsy_when_returned(plan_types):
    """The old contract: an empty error list means nothing went wrong."""
    clean = plan_types.IngestionPlan(nwb_file_name="clean_.nwb")

    assert not clean, "A clean plan must be falsy, as the old list was"
    assert len(clean) == 0
    assert list(clean) == []
    assert "no_op" in str(clean), "str() gives the report"


def test_a_returned_plan_carries_only_blocking_problems(plan_types):
    """The list it replaces held failures, not advisories."""
    plan = _broken(plan_types)

    assert plan, "A blocked plan must be truthy"
    assert len(plan) == 1, "One hard problem; both soft ones excluded"
    assert all(
        problem.severity in plan_types.BLOCKING for problem in plan
    ), "Iterating yields the problems that actually blocked"
    assert plan.blocking == tuple(
        plan
    ), "One definition of blocking, whichever way it is asked for"


def test_a_fatal_plan_is_not_a_no_op(plan_types):
    """A file that could not be read has not 'already been ingested'.

    `fatal` used to fall through to the entry count: a file nothing could be
    planned from has no novel entries, so the verdict read `no_op`, and
    `insert_plan` logged "already ingested" and closed the staging area for
    a file it had never opened.
    """
    unreadable = plan_types.IngestionPlan(
        nwb_file_name="broken_.nwb",
        fatal=(
            plan_types.Problem(
                "fatal", "file_unreadable", "truncated at byte 0"
            ),
        ),
    )

    assert unreadable.verdict == "fatal", "Not no_op, and not clean"
    assert not unreadable.is_clean
    assert len(unreadable.blocking) == 1, "fatal blocks, as hard does"
    assert "already ingested" not in unreadable.report(log=False)

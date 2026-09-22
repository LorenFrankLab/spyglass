"""What the ingestion report says, and to whom."""

import pytest


@pytest.fixture
def plan_types():
    """The plan dataclasses, which need no database."""
    from spyglass.data_import import ingestion_plan

    return ingestion_plan


def _broken(plan_types):
    """A plan with one problem of each interesting shape."""
    return plan_types.IngestionPlan(
        nwb_file_name="broken_.nwb",
        novel={"`common`.`session`": 1},
        table_plans=(
            plan_types.TablePlan(
                table_name="`common`.`session`",
                entries=plan_types.PlannedEntries().freeze(),
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
                        "hard",
                        "divergence",
                        "already stored with different values",
                        table="`common`.`subject`",
                        suggested_revision={"sex": "M"},
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
                entries=plan_types.PlannedEntries().freeze(),
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
    """The first line answers 'what happened', before any detail."""
    report = _broken(plan_types).report(log=False)
    first = report.splitlines()[0]

    assert first.startswith("broken_.nwb: conflict"), first


def test_problems_group_by_severity_worst_first(plan_types):
    """Blocking problems come before advisory ones."""
    report = _broken(plan_types).report(verbose=True, log=False)

    assert report.index("Hard (2):") < report.index(
        "Soft (1):"
    ), "Hard problems must precede soft ones"
    assert "Hard (2)" in report, "The count tells you how much to read"


def test_a_problem_names_the_object_and_the_remedy(plan_types):
    """The message says what happened; the remedy says what to change."""
    report = _broken(plan_types).report(verbose=True, log=False)

    assert "(object abc-123)" in report, "The object id locates it in the file"
    assert "_spyglass_config.yaml" in report, "missing_attribute has a remedy"
    assert (
        report.count("-> The NWB file does not supply") == 1
    ), "A remedy is per code, not repeated under every occurrence"


def test_a_divergence_revision_is_pasteable(plan_types):
    """A suggested revision should be usable without retyping it."""
    report = _broken(plan_types).report(verbose=True, log=False)

    assert "{'sex': 'M'}" in report, "The revision renders as a dict literal"
    assert "Suggested revisions, to apply as-is:" in report


def test_soft_problems_are_hidden_unless_asked_for(plan_types):
    """The default report is what blocks you, not everything noticed."""
    quiet = _broken(plan_types).report(log=False)

    assert "Soft (1):" not in quiet, "Advisory problems are opt-in"
    assert "Hard (2):" in quiet, "Blocking problems always show"


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


def test_report_result_is_falsy_when_clean(plan_types):
    """The old contract: an empty error list means nothing went wrong."""
    clean = plan_types.ReportResult(
        plan_types.IngestionPlan(nwb_file_name="clean_.nwb")
    )

    assert not clean, "A clean result must be falsy, as the old list was"
    assert len(clean) == 0
    assert list(clean) == []
    assert "no_op" in str(clean), "str() gives the report"


def test_report_result_carries_only_blocking_problems(plan_types):
    """The list it replaces held failures, not advisories."""
    result = plan_types.ReportResult(_broken(plan_types))

    assert result, "A blocked result must be truthy"
    assert len(result) == 2, "Two hard problems, the soft one excluded"
    assert all(
        problem.severity in plan_types.BLOCKING for problem in result
    ), "Iterating yields the problems that actually blocked"


def test_report_result_still_answers_fetch_key(plan_types):
    """Legacy callers reached for `.fetch("KEY")`; that still works."""
    result = plan_types.ReportResult(_broken(plan_types))
    keys = result.fetch("KEY")

    assert len(keys) == len(result), "One key per blocking problem"
    assert all(
        "nwb_file_name" in key and "table" in key for key in keys
    ), "Keys keep the shape callers indexed into"

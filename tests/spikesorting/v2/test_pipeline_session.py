"""Tests for the whole-session preflight + runner in ``v2.pipeline``.

``preflight_v2_pipeline_session`` and ``run_v2_pipeline_session`` loop the
single-group helpers over every (or selected) ``SortGroupV2`` row in a session.
Three tiers:

    * ``unit`` (DB-free): the shared target resolver's preset validation, the
      per-group aggregation, and the runner's preflight-gate / continue-on-error
      control flow -- exercised with monkeypatched resolver / preflight / run so
      they touch no database.
    * ``database`` (dj_conn, no ingest): the resolver's "no sort groups" path
      against a never-ingested session name.
    * ``integration`` (Docker MySQL + SI 0.104): the resolver subset/missing
      behavior, the real session preflight, and a real 4-shank session run +
      idempotent re-run on the polymer smoke fixture.

The module-level imports are DB-free (``pipeline`` and ``exceptions`` import no
schema); schema tables are imported lazily inside the tests that need them.
"""

from __future__ import annotations

from pathlib import Path

import datajoint as dj
import pytest

import spyglass.spikesorting.v2._pipeline_run as plr
import spyglass.spikesorting.v2.pipeline as pl
from spyglass.spikesorting.v2._pipeline_preflight import (
    _resolve_session_sort_group_ids,
)
from spyglass.spikesorting.v2.exceptions import (
    PipelineInputError,
    PipelineStageError,
    PreflightError,
    ZeroUnitSortError,
)
from spyglass.spikesorting.v2.pipeline import (
    PreflightCheck,
    PreflightReport,
    PreflightSessionReport,
    preflight_v2_pipeline_session,
    run_v2_pipeline_session,
)
from tests.spikesorting.v2._ingest_helpers import (
    configure_v2_run_inputs,
    copy_and_insert_nwb,
)

# A real, fast, rate-agnostic catalog preset (peak detection, no clustering):
# keeps the integration run cheap and avoids the sampling-rate preflight check.
_PRESET = "franklab_clusterless_2026_06"

_FIXTURE_NAME = "mearec_polymer_smoke"
_FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / f"{_FIXTURE_NAME}.nwb"
)
_TEAM = "session_runner_test_team"
_INTERVAL = "raw data valid times"


def _no_db(monkeypatch) -> None:
    """Make any DataJoint query raise, to prove a path is DB-free."""

    def _boom(*args, **kwargs):
        raise AssertionError("unexpected database query in a DB-free path")

    monkeypatch.setattr(dj.Connection, "query", _boom)


def _ok_report(sort_group_id: int, pipeline_preset: str) -> PreflightReport:
    """A passing single-group ``PreflightReport`` stub."""
    return PreflightReport(
        ok=True,
        errors=[],
        warnings=[],
        resolved_pipeline_preset=pipeline_preset,
        expected_ids={
            "recording_id": {"id": f"rec-{sort_group_id}", "exists": False}
        },
        checks=[PreflightCheck("pipeline_preset_known", True, "")],
    )


# ---------------------------------------------------------------------------
# unit tier -- shared target resolver (DB-free preset validation)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_resolve_requires_pipeline_preset(monkeypatch):
    """A missing ``pipeline_preset`` is rejected before any DB access."""
    _no_db(monkeypatch)
    with pytest.raises(PipelineInputError) as exc:
        _resolve_session_sort_group_ids(
            nwb_file_name="s.nwb",
            pipeline_preset=None,
            sort_group_ids=None,
            caller="run_v2_pipeline_session",
        )
    assert "pipeline_preset is required" in str(exc.value)
    assert "describe_pipeline_presets()" in str(exc.value)


@pytest.mark.unit
def test_resolve_unknown_preset(monkeypatch):
    """An unknown ``pipeline_preset`` is rejected before any DB access."""
    _no_db(monkeypatch)
    with pytest.raises(PipelineInputError) as exc:
        _resolve_session_sort_group_ids(
            nwb_file_name="s.nwb",
            pipeline_preset="not_a_real_preset",
            sort_group_ids=None,
            caller="run_v2_pipeline_session",
        )
    assert "unknown pipeline_preset" in str(exc.value)
    assert "not_a_real_preset" in str(exc.value)


# ---------------------------------------------------------------------------
# unit tier -- session preflight aggregation (monkeypatched per-group preflight)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_preflight_session_all_groups(monkeypatch):
    """One ``group_reports`` entry per target, aggregated ``ok``; no DB."""
    monkeypatch.setattr(
        "spyglass.spikesorting.v2._pipeline_preflight."
        "_resolve_session_sort_group_ids",
        lambda **kw: [0, 1, 2],
    )
    monkeypatch.setattr(
        "spyglass.spikesorting.v2._pipeline_preflight.preflight_v2_pipeline",
        lambda *, sort_group_id, pipeline_preset, **kw: _ok_report(
            sort_group_id, pipeline_preset
        ),
    )
    _no_db(monkeypatch)

    report = preflight_v2_pipeline_session(
        nwb_file_name="s.nwb",
        interval_list_name=_INTERVAL,
        team_name=_TEAM,
        pipeline_preset=_PRESET,
    )

    assert isinstance(report, PreflightSessionReport)
    assert report.ok is True and bool(report) is True
    assert report.errors == [] and report.warnings == []
    assert report.resolved_pipeline_preset == _PRESET
    assert [r["sort_group_id"] for r in report.group_reports] == [0, 1, 2]
    for row in report.group_reports:
        assert set(row) == {
            "sort_group_id",
            "ok",
            "errors",
            "warnings",
            "expected_ids",
            "checks",
        }
        assert row["ok"] is True
        assert row["expected_ids"]  # carried through from the per-group report


@pytest.mark.unit
def test_preflight_session_collects_group_errors(monkeypatch):
    """A failed group flips ``ok``, aggregates prefixed errors/warnings.

    Other group rows stay inspectable; ``ok`` reflects errors only, so a
    warnings-only group does not flip it.
    """
    monkeypatch.setattr(
        "spyglass.spikesorting.v2._pipeline_preflight."
        "_resolve_session_sort_group_ids",
        lambda **kw: [0, 1, 2],
    )

    def fake_preflight(*, sort_group_id, pipeline_preset, **kw):
        if sort_group_id == 1:
            return PreflightReport(
                ok=False,
                errors=["LabTeam 't' does not exist."],
                warnings=["heads up"],
                resolved_pipeline_preset=pipeline_preset,
                expected_ids={},
                checks=[PreflightCheck("team_exists", False, "create it")],
            )
        return _ok_report(sort_group_id, pipeline_preset)

    monkeypatch.setattr(
        "spyglass.spikesorting.v2._pipeline_preflight.preflight_v2_pipeline",
        fake_preflight,
    )

    report = preflight_v2_pipeline_session(
        nwb_file_name="s.nwb",
        interval_list_name=_INTERVAL,
        team_name=_TEAM,
        pipeline_preset=_PRESET,
    )

    assert report.ok is False and bool(report) is False
    assert report.errors == ["sort_group_id=1: LabTeam 't' does not exist."]
    assert report.warnings == ["sort_group_id=1: heads up"]
    by_id = {r["sort_group_id"]: r for r in report.group_reports}
    assert by_id[1]["ok"] is False
    assert by_id[1]["errors"] == ["LabTeam 't' does not exist."]
    assert by_id[0]["ok"] is True and by_id[2]["ok"] is True


# ---------------------------------------------------------------------------
# unit tier -- runner control flow (monkeypatched resolver/preflight/run)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_session_runner_requires_pipeline_preset(monkeypatch):
    """The runner requires an explicit preset, before any DB or run."""
    _no_db(monkeypatch)
    monkeypatch.setattr(
        plr,
        "run_v2_pipeline",
        lambda **kw: pytest.fail("run_v2_pipeline must not be called"),
    )
    with pytest.raises(PipelineInputError, match="pipeline_preset is required"):
        run_v2_pipeline_session(
            nwb_file_name="s.nwb",
            interval_list_name=_INTERVAL,
            team_name=_TEAM,
        )


@pytest.mark.unit
def test_session_runner_unknown_preset(monkeypatch):
    """An unknown preset is rejected before any group runs."""
    _no_db(monkeypatch)
    monkeypatch.setattr(
        plr,
        "run_v2_pipeline",
        lambda **kw: pytest.fail("run_v2_pipeline must not be called"),
    )
    with pytest.raises(PipelineInputError, match="unknown pipeline_preset"):
        run_v2_pipeline_session(
            nwb_file_name="s.nwb",
            interval_list_name=_INTERVAL,
            team_name=_TEAM,
            pipeline_preset="not_a_real_preset",
        )


def _session_report(group_oks: dict[int, bool]) -> PreflightSessionReport:
    """Build a session preflight report stub from {sort_group_id: ok}."""
    rows = []
    errors = []
    for sort_group_id, ok in sorted(group_oks.items()):
        group_errors = [] if ok else [f"group {sort_group_id} misconfigured"]
        rows.append(
            {
                "sort_group_id": sort_group_id,
                "ok": ok,
                "errors": group_errors,
                "warnings": [],
                "expected_ids": {},
                "checks": [],
            }
        )
        errors.extend(
            f"sort_group_id={sort_group_id}: {e}" for e in group_errors
        )
    return PreflightSessionReport(
        ok=not errors,
        errors=errors,
        warnings=[],
        resolved_pipeline_preset=_PRESET,
        group_reports=rows,
    )


@pytest.mark.unit
def test_session_runner_preflight_fail_fast(monkeypatch):
    """A failed session preflight raises before any group is sorted."""
    monkeypatch.setattr(
        plr, "_resolve_session_sort_group_ids", lambda **kw: [0, 1]
    )
    monkeypatch.setattr(
        plr,
        "preflight_v2_pipeline_session",
        lambda **kw: _session_report({0: True, 1: False}),
    )
    calls: list[int] = []
    monkeypatch.setattr(
        plr,
        "run_v2_pipeline",
        lambda *, sort_group_id, **kw: calls.append(sort_group_id),
    )

    with pytest.raises(PreflightError) as exc:
        run_v2_pipeline_session(
            nwb_file_name="s.nwb",
            interval_list_name=_INTERVAL,
            team_name=_TEAM,
            pipeline_preset=_PRESET,
            continue_on_error=False,
        )
    assert "sort_group_id=1" in str(exc.value)
    assert calls == [], "no group should run when preflight fails fast"


@pytest.mark.unit
def test_session_runner_preflight_continue(monkeypatch):
    """``continue_on_error`` skips a failed-preflight group, runs the rest.

    The groups that run are called with ``preflight=False`` (the session
    preflight already covered the DB checks).
    """
    monkeypatch.setattr(
        plr, "_resolve_session_sort_group_ids", lambda **kw: [0, 1]
    )
    monkeypatch.setattr(
        plr,
        "preflight_v2_pipeline_session",
        lambda **kw: _session_report({0: True, 1: False}),
    )
    calls: list[dict] = []

    def fake_run(**kw):
        calls.append(kw)
        return {"root_merge_id": f"m{kw['sort_group_id']}", "n_units": 2}

    monkeypatch.setattr(plr, "run_v2_pipeline", fake_run)

    results = run_v2_pipeline_session(
        nwb_file_name="s.nwb",
        interval_list_name=_INTERVAL,
        team_name=_TEAM,
        pipeline_preset=_PRESET,
        continue_on_error=True,
    )

    assert [c["sort_group_id"] for c in calls] == [0]
    assert all(c["preflight"] is False for c in calls)
    assert [r["sort_group_id"] for r in results] == [0, 1]
    by_id = {r["sort_group_id"]: r for r in results}
    assert by_id[0]["outcome"] == "ok" and by_id[0]["root_merge_id"] == "m0"
    assert by_id[1]["outcome"] == "failed"
    assert by_id[1]["partial_run_summary"] is None
    assert by_id[1]["error_type"] == "PreflightError"
    assert "group 1 misconfigured" in by_id[1]["error"]
    assert by_id[1]["pipeline_preset"] == _PRESET


@pytest.mark.unit
def test_session_runner_continue_on_error(monkeypatch):
    """A per-group sort failure becomes one failed entry; others succeed.

    The failed entry carries both the error string and the
    ``partial_run_summary`` from the ``PipelineStageError``.
    """
    monkeypatch.setattr(
        plr, "_resolve_session_sort_group_ids", lambda **kw: [0, 1, 2]
    )
    monkeypatch.setattr(
        plr,
        "preflight_v2_pipeline_session",
        lambda **kw: _session_report({0: True, 1: True, 2: True}),
    )

    def fake_run(*, sort_group_id, **kw):
        if sort_group_id == 1:
            raise PipelineStageError(
                "sorting",
                {"recording_id": "r1"},
                "sorter crashed",
                original_type="IndexError",
            )
        return {"root_merge_id": f"m{sort_group_id}", "n_units": 3}

    monkeypatch.setattr(plr, "run_v2_pipeline", fake_run)

    results = run_v2_pipeline_session(
        nwb_file_name="s.nwb",
        interval_list_name=_INTERVAL,
        team_name=_TEAM,
        pipeline_preset=_PRESET,
        continue_on_error=True,
    )

    assert [r["sort_group_id"] for r in results] == [0, 1, 2]
    by_id = {r["sort_group_id"]: r for r in results}
    assert by_id[0]["outcome"] == "ok" and by_id[0]["root_merge_id"] == "m0"
    assert by_id[2]["outcome"] == "ok" and by_id[2]["root_merge_id"] == "m2"
    assert by_id[1]["outcome"] == "failed"
    assert by_id[1]["partial_run_summary"] == {"recording_id": "r1"}
    assert by_id[1]["error_type"] == "PipelineStageError"
    assert "sorting" in by_id[1]["error"]
    # Structured triage fields: the failing stage and the wrapped error type.
    assert by_id[1]["stage"] == "sorting"
    assert by_id[1]["original_error_type"] == "IndexError"


@pytest.mark.unit
def test_session_runner_propagates_preflight_warnings(monkeypatch):
    """Per-group preflight warnings reach EVERY entry -- ok, failed-preflight,
    and failed-run -- so describe_run / the batch warning count do not
    under-report (review R2-D + R3-C). Each OK group runs with preflight=False,
    so the warnings must come from the session report, not a per-group rerun.
    """
    monkeypatch.setattr(
        plr, "_resolve_session_sort_group_ids", lambda **kw: [0, 1, 2]
    )

    def fake_session_report(**kw):
        return PreflightSessionReport(
            ok=False,  # group 1 carries a blocking error
            errors=["sort_group_id=1: group 1 misconfigured"],
            warnings=[
                "sort_group_id=0: w-ok",
                "sort_group_id=1: w-failpre",
                "sort_group_id=2: w-failrun",
            ],
            resolved_pipeline_preset=_PRESET,
            group_reports=[
                {
                    "sort_group_id": 0,
                    "ok": True,
                    "errors": [],
                    "warnings": ["w-ok"],
                    "expected_ids": {},
                    "checks": [],
                },
                {
                    "sort_group_id": 1,
                    "ok": False,
                    "errors": ["group 1 misconfigured"],
                    "warnings": ["w-failpre"],
                    "expected_ids": {},
                    "checks": [],
                },
                {
                    "sort_group_id": 2,
                    "ok": True,
                    "errors": [],
                    "warnings": ["w-failrun"],
                    "expected_ids": {},
                    "checks": [],
                },
            ],
        )

    monkeypatch.setattr(
        plr, "preflight_v2_pipeline_session", fake_session_report
    )

    def fake_run(*, sort_group_id, **kw):
        if sort_group_id == 2:  # passes preflight, fails mid-run
            raise PipelineStageError("sorting", {"recording_id": "r2"}, "boom")
        return {
            "root_merge_id": f"m{sort_group_id}",
            "n_units": 1,
            "warnings": [],
        }

    monkeypatch.setattr(plr, "run_v2_pipeline", fake_run)

    results = run_v2_pipeline_session(
        nwb_file_name="s.nwb",
        interval_list_name=_INTERVAL,
        team_name=_TEAM,
        pipeline_preset=_PRESET,
        continue_on_error=True,
    )
    by_id = {r["sort_group_id"]: r for r in results}
    # ok group: its preflight warning is folded into the success summary.
    assert by_id[0]["outcome"] == "ok"
    assert "w-ok" in by_id[0]["warnings"]
    # failed-preflight group: its warning rides on the failed entry.
    assert by_id[1]["outcome"] == "failed"
    assert "w-failpre" in by_id[1]["warnings"]
    # failed-run group: its preflight warning rides on the failed entry.
    assert by_id[2]["outcome"] == "failed"
    assert "w-failrun" in by_id[2]["warnings"]


@pytest.mark.unit
def test_session_runner_fail_fast(monkeypatch):
    """Without ``continue_on_error`` a per-group sort failure propagates."""
    monkeypatch.setattr(
        plr, "_resolve_session_sort_group_ids", lambda **kw: [0, 1, 2]
    )
    monkeypatch.setattr(
        plr,
        "preflight_v2_pipeline_session",
        lambda **kw: _session_report({0: True, 1: True, 2: True}),
    )

    def fake_run(*, sort_group_id, **kw):
        if sort_group_id == 1:
            raise PipelineStageError(
                "sorting", {"recording_id": "r1"}, "sorter crashed"
            )
        return {"root_merge_id": f"m{sort_group_id}", "n_units": 3}

    monkeypatch.setattr(plr, "run_v2_pipeline", fake_run)

    with pytest.raises(PipelineStageError, match="sorting") as exc:
        run_v2_pipeline_session(
            nwb_file_name="s.nwb",
            interval_list_name=_INTERVAL,
            team_name=_TEAM,
            pipeline_preset=_PRESET,
            continue_on_error=False,
        )
    # Fail-fast re-raises the original exception unwrapped: the partial run
    # summary it carries survives intact (not reconstructed or dropped).
    assert exc.value.partial_run_summary == {"recording_id": "r1"}


@pytest.mark.unit
def test_session_runner_continue_on_error_zero_units(monkeypatch):
    """A per-group ``ZeroUnitSortError`` becomes a failed entry (no partial).

    Exercises the third arm of the per-group catch tuple and the
    ``getattr(exc, "partial_run_summary", None)`` fallback for an exception that
    carries no partial summary.
    """
    monkeypatch.setattr(
        plr, "_resolve_session_sort_group_ids", lambda **kw: [0, 1]
    )
    monkeypatch.setattr(
        plr,
        "preflight_v2_pipeline_session",
        lambda **kw: _session_report({0: True, 1: True}),
    )

    def fake_run(*, sort_group_id, **kw):
        if sort_group_id == 1:
            raise ZeroUnitSortError("zero units on a quiet shank")
        return {"root_merge_id": f"m{sort_group_id}", "n_units": 4}

    monkeypatch.setattr(plr, "run_v2_pipeline", fake_run)

    results = run_v2_pipeline_session(
        nwb_file_name="s.nwb",
        interval_list_name=_INTERVAL,
        team_name=_TEAM,
        pipeline_preset=_PRESET,
        require_units=True,
        continue_on_error=True,
    )
    by_id = {r["sort_group_id"]: r for r in results}
    assert by_id[0]["outcome"] == "ok" and by_id[0]["root_merge_id"] == "m0"
    assert by_id[1]["outcome"] == "failed"
    assert by_id[1]["partial_run_summary"] is None  # ZeroUnitSortError has none
    assert by_id[1]["error_type"] == "ZeroUnitSortError"
    assert "zero units" in by_id[1]["error"]


@pytest.mark.unit
def test_session_runner_preflight_false_skips_session_preflight(monkeypatch):
    """``preflight=False`` runs each group directly with ``preflight=False``."""
    monkeypatch.setattr(
        plr, "_resolve_session_sort_group_ids", lambda **kw: [0, 1]
    )
    monkeypatch.setattr(
        plr,
        "preflight_v2_pipeline_session",
        lambda **kw: pytest.fail("session preflight must be skipped"),
    )
    calls: list[dict] = []

    def fake_run(**kw):
        calls.append(kw)
        return {"root_merge_id": f"m{kw['sort_group_id']}"}

    monkeypatch.setattr(plr, "run_v2_pipeline", fake_run)

    results = run_v2_pipeline_session(
        nwb_file_name="s.nwb",
        interval_list_name=_INTERVAL,
        team_name=_TEAM,
        pipeline_preset=_PRESET,
        preflight=False,
        curation_description="batch run",
        require_units=True,
    )
    assert [c["sort_group_id"] for c in calls] == [0, 1]
    assert all(c["preflight"] is False for c in calls)
    # Per-call kwargs are forwarded to each single-group run unchanged.
    assert all(c["curation_description"] == "batch run" for c in calls)
    assert all(c["require_units"] is True for c in calls)
    assert all(c["interval_list_name"] == _INTERVAL for c in calls)
    assert all(c["pipeline_preset"] == _PRESET for c in calls)
    assert all(r["outcome"] == "ok" for r in results)


# ---------------------------------------------------------------------------
# database tier (dj_conn, no ingest)
# ---------------------------------------------------------------------------


@pytest.mark.database
def test_resolve_no_sort_groups(dj_conn):
    """A session with no SortGroupV2 rows raises, naming set_group_by_shank."""
    with pytest.raises(PipelineInputError) as exc:
        _resolve_session_sort_group_ids(
            nwb_file_name="definitely_not_a_session_.nwb",
            pipeline_preset=_PRESET,
            sort_group_ids=None,
            caller="run_v2_pipeline_session",
        )
    assert "no SortGroupV2 rows" in str(exc.value)
    assert "set_group_by_shank" in str(exc.value)


# ---------------------------------------------------------------------------
# integration tier (Docker MySQL + SI 0.104; 4-shank polymer smoke fixture)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def session_inputs(dj_conn):
    """Ingest + configure the smoke fixture (defaults + team + sort groups).

    Distinct ``nwb_file_name`` so this module's rows never collide with the
    other v2 pipeline modules. ``configure_v2_run_inputs`` builds one sort group
    per shank (the polymer fixture yields several), which is exactly what the
    batch runner targets. No ``populate`` here.
    """
    if not _FIXTURE_PATH.exists():
        pytest.skip(
            f"Generated MEArec fixture {_FIXTURE_PATH.name} not found. Run "
            "`python tests/spikesorting/v2/fixtures/generate_mearec.py "
            "--smoke` first."
        )
    nwb_file_name = copy_and_insert_nwb(
        _FIXTURE_PATH, dest_name="mearec_session_runner_smoke.nwb"
    )
    configure_v2_run_inputs(
        nwb_file_name,
        _TEAM,
        interval_list_name=_INTERVAL,
        team_description="session runner tests",
    )
    return {
        "nwb_file_name": nwb_file_name,
        "interval_list_name": _INTERVAL,
        "team_name": _TEAM,
    }


def _available_sort_group_ids(nwb_file_name: str) -> list[int]:
    from spyglass.spikesorting.v2.recording import SortGroupV2

    return sorted(
        int(g)
        for g in (SortGroupV2 & {"nwb_file_name": nwb_file_name}).fetch(
            "sort_group_id"
        )
    )


@pytest.mark.integration
def test_resolve_subset_and_missing(session_inputs):
    """A subset is honored (sorted/deduped); an absent id raises with the list."""
    nwb_file_name = session_inputs["nwb_file_name"]
    available = _available_sort_group_ids(nwb_file_name)
    assert len(available) >= 2

    subset = available[:2]
    got = _resolve_session_sort_group_ids(
        nwb_file_name=nwb_file_name,
        pipeline_preset=_PRESET,
        # reversed + duplicate -> resolver must sort and dedup.
        sort_group_ids=[*reversed(subset), subset[0]],
        caller="run_v2_pipeline_session",
    )
    assert got == sorted(subset)

    with pytest.raises(PipelineInputError) as exc:
        _resolve_session_sort_group_ids(
            nwb_file_name=nwb_file_name,
            pipeline_preset=_PRESET,
            sort_group_ids=[999],
            caller="run_v2_pipeline_session",
        )
    assert "[999]" in str(exc.value)
    assert str(available) in str(exc.value)


@pytest.mark.integration
def test_preflight_session_real_all_ok(session_inputs):
    """The real per-group preflight passes for every group of a good session."""
    report = preflight_v2_pipeline_session(
        **session_inputs, pipeline_preset=_PRESET
    )
    assert report.ok is True, report.errors
    available = set(_available_sort_group_ids(session_inputs["nwb_file_name"]))
    assert {r["sort_group_id"] for r in report.group_reports} == available
    for row in report.group_reports:
        assert set(row) == {
            "sort_group_id",
            "ok",
            "errors",
            "warnings",
            "expected_ids",
            "checks",
        }
        assert row["ok"] is True


@pytest.fixture(scope="module")
def session_run(session_inputs):
    """Run the whole-session sort once (shared by the run assertions)."""
    return run_v2_pipeline_session(
        **session_inputs, pipeline_preset=_PRESET, continue_on_error=False
    )


@pytest.mark.slow
@pytest.mark.integration
def test_session_runner_all_groups(session_run, session_inputs):
    """One ``outcome="ok"`` entry per SortGroupV2 row; ids match the session."""
    available = set(_available_sort_group_ids(session_inputs["nwb_file_name"]))
    assert {r["sort_group_id"] for r in session_run} == available
    assert len(session_run) == len(available)
    for entry in session_run:
        assert entry["outcome"] == "ok"
        # A successful entry is a full single-group run summary.
        assert "root_merge_id" in entry and "n_units" in entry
        assert entry["pipeline_preset"] == _PRESET
        # Fresh session: the first run computes every stage. Paired with the
        # idempotent test's all-"reused" assertion, this makes the
        # computed/reused distinction load-bearing (a bug that always reported
        # "reused" would fail here).
        for status_key in (
            "recording_status",
            "artifact_detection_status",
            "sorting_status",
            "curation_status",
        ):
            assert entry[status_key] == "computed", (
                entry["sort_group_id"],
                status_key,
            )
    # Distinct root_merge_ids per group (each shank is its own sort).
    merge_ids = [r["root_merge_id"] for r in session_run]
    assert len(set(merge_ids)) == len(merge_ids)


@pytest.mark.slow
@pytest.mark.integration
def test_session_runner_idempotent(session_run, session_inputs):
    """A second run returns the same root_merge_ids and every stage ``"reused"``."""
    second = run_v2_pipeline_session(
        **session_inputs, pipeline_preset=_PRESET, continue_on_error=False
    )
    first_by_id = {r["sort_group_id"]: r for r in session_run}
    second_by_id = {r["sort_group_id"]: r for r in second}
    assert set(second_by_id) == set(first_by_id)
    for sort_group_id, entry in second_by_id.items():
        assert entry["outcome"] == "ok"
        assert (
            entry["root_merge_id"]
            == first_by_id[sort_group_id]["root_merge_id"]
        )
        for status_key in (
            "recording_status",
            "artifact_detection_status",
            "sorting_status",
            "curation_status",
        ):
            assert entry[status_key] == "reused", (sort_group_id, status_key)

"""Tests for PathTrackingMixin and the check_missing tracking tables.

1. PathScanner - state per on-disk condition
2. TrackedPath - declaration validation
3. check_missing() - findings, exclusions, coverage
4. run_all, refresh_all, summary - orchestration
5. MissingPath - lost, unreadable, refresh_missing, last_covered_run
"""

import os
import shutil

import datajoint as dj
import pytest

pytestmark = pytest.mark.order("last")


@pytest.fixture(scope="module")
def mixin_mod(dj_conn):
    """Return the check_missing mixin module."""
    import spyglass.utils.mixins.check_missing as mod

    yield mod


@pytest.fixture(scope="module")
def scan(mixin_mod):
    """Return a scan helper taking the path and its declaration."""

    def _scan(path, kind="dir", markers=()):
        tracked = mixin_mod.TrackedPath("p", kind=kind, markers=markers)
        return mixin_mod.PathScanner(tracked).scan(str(path))

    yield _scan


@pytest.fixture(scope="module")
def tracking_tables(dj_conn):
    """Return the tracking tables."""
    from spyglass.common.common_file_tracking import (
        CheckMissingRun,
        MissingPath,
    )

    yield CheckMissingRun, MissingPath


@pytest.fixture(scope="module")
def tracked_table(dj_conn, mixin_mod, teardown):
    """Declare a table that tracks one directory path."""
    from spyglass.utils import SpyglassMixin

    schema = dj.schema("test_check_missing")

    @schema
    class PathHolder(mixin_mod.PathTrackingMixin, SpyglassMixin, dj.Manual):
        definition = """
        holder_id: int
        ---
        some_path: varchar(1000)
        """
        _tracked_paths = (
            mixin_mod.TrackedPath("some_path", kind="dir", markers=("flag",)),
        )

        def _skip_second(self):
            return {(2,)}

    yield PathHolder()

    if teardown:
        schema.drop(force=True)


# ============================== PathScanner ==================================


def test_scan_ok_dir(scan, tmp_path):
    """A directory holding its marker is ok."""
    (tmp_path / "flag").write_text("x")
    found = scan(tmp_path, markers=("flag",))
    assert found["state"] == "ok"
    assert found["n_files"] == 1


def test_scan_partial_dir(scan, tmp_path):
    """A directory holding files but no marker is partial."""
    (tmp_path / "other").write_text("x")
    assert scan(tmp_path, markers=("flag",))["state"] == "partial"


def test_scan_empty_dir(scan, tmp_path):
    """An emptied directory is distinguished from an absent one."""
    found = scan(tmp_path, markers=("flag",))
    assert found["state"] == "empty"
    assert found["mtime"] is not None


def test_scan_missing(scan, tmp_path):
    """An absent path whose parent survives is missing."""
    found = scan(tmp_path / "gone")
    assert found["state"] == "missing"
    assert found["mtime"] is None


def test_scan_parent_missing(scan, tmp_path):
    """An absent path whose parent is also absent is parent_missing."""
    assert scan(tmp_path / "gone" / "deeper")["state"] == "parent_missing"


def test_scan_broken_link(scan, tmp_path):
    """A symlink outliving its target is broken_link, not ok."""
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "nope")
    assert scan(link)["state"] == "broken_link"


def test_scan_not_a_dir(scan, tmp_path):
    """A file where a directory is declared is not_a_dir."""
    target = tmp_path / "f"
    target.write_text("x")
    assert scan(target)["state"] == "not_a_dir"


def test_scan_file_kind(scan, tmp_path):
    """A declared file is ok when present, missing when not."""
    target = tmp_path / "f.json"
    target.write_text("xy")
    found = scan(target, kind="file")
    assert (found["state"], found["size_bytes"]) == ("ok", 2)
    assert scan(tmp_path / "no.json", kind="file")["state"] == "missing"


def test_scan_denied(scan, mixin_mod, tmp_path):
    """An unreadable directory is denied, which is not a loss."""
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    (blocked / "inner").mkdir()
    os.chmod(blocked, 0o000)
    try:
        assert scan(blocked / "inner")["state"] == "denied"
        assert "denied" not in mixin_mod.LOST_STATES
    finally:
        os.chmod(blocked, 0o755)


# ============================== TrackedPath ==================================


def test_tracked_path_rejects_bad_kind(mixin_mod):
    """kind must be 'dir' or 'file'."""
    with pytest.raises(ValueError, match="kind must be"):
        mixin_mod.TrackedPath("some_path", kind="folder")


def test_tracked_path_rejects_markers_on_file(mixin_mod):
    """Markers describe a directory, so a file may not carry them."""
    with pytest.raises(ValueError, match="markers need"):
        mixin_mod.TrackedPath("some_path", kind="file", markers=("flag",))


def test_declaring_table_is_registered(mixin_mod, tracked_table):
    """Declaring the mixin registers the table, with no hand-kept list."""
    registered = [
        t.full_table_name for t in mixin_mod.PathTrackingMixin.tracking_tables()
    ]
    assert tracked_table.full_table_name in registered


# ============================= check_missing =================================


def test_check_missing_records_only_findings(
    tracked_table, tracking_tables, tmp_path
):
    """A whole path yields no row; an absent one yields a finding."""
    _, MissingPath = tracking_tables
    whole = tmp_path / "whole"
    whole.mkdir()
    (whole / "flag").write_text("x")

    tracked_table.insert(
        [
            dict(holder_id=1, some_path=str(whole)),
            dict(holder_id=2, some_path=str(tmp_path / "absent")),
        ]
    )
    try:
        assert tracked_table.check_missing() == {"some_path": 1}

        found = MissingPath() & {
            "full_table_name": tracked_table.full_table_name
        }
        assert len(found) == 1
        entry = found.fetch1()
        assert entry["state"] == "missing"
        assert entry["source_key"] == {"holder_id": 2}
    finally:
        (
            MissingPath() & {"full_table_name": tracked_table.full_table_name}
        ).delete_quick()
        tracked_table.delete_quick()


def test_check_missing_honours_exclusion(
    tracked_table, mixin_mod, tmp_path, monkeypatch
):
    """An excluded key is not reported, even when its path is absent."""
    from spyglass.common.common_file_tracking import CheckMissingRun

    monkeypatch.setattr(
        type(tracked_table),
        "_tracked_paths",
        (
            mixin_mod.TrackedPath(
                "some_path", kind="dir", exclude="_skip_second"
            ),
        ),
    )

    tracked_table.insert1(dict(holder_id=2, some_path=str(tmp_path / "gone")))
    try:
        assert tracked_table.check_missing() == {"some_path": 0}

        # Coverage separates 'nothing absent' from 'nothing looked at'
        row = (
            CheckMissingRun.Coverage
            & {"full_table_name": tracked_table.full_table_name}
        ).fetch(order_by="run_id DESC", as_dict=True)[0]
        assert (row["n_checked"], row["n_excluded"]) == (0, 1)
    finally:
        tracked_table.delete_quick()


def test_check_missing_records_coverage(
    tracked_table, tracking_tables, tmp_path
):
    """Coverage names what the run reached, so a skipped attr is visible."""
    CheckMissingRun, MissingPath = tracking_tables

    tracked_table.insert1(dict(holder_id=3, some_path=str(tmp_path / "gone")))
    try:
        tracked_table.check_missing()
        coverage = CheckMissingRun.Coverage & {
            "full_table_name": tracked_table.full_table_name,
            "attr": "some_path",
        }
        assert len(coverage) >= 1
        row = coverage.fetch(order_by="run_id DESC", as_dict=True)[0]
        assert (row["n_checked"], row["n_missing"]) == (1, 1)
        assert row["completed"] == 1
    finally:
        (
            MissingPath() & {"full_table_name": tracked_table.full_table_name}
        ).delete_quick()
        tracked_table.delete_quick()


def test_last_covered_run_bounds_the_loss(
    tracked_table, tracking_tables, tmp_path
):
    """last_covered_run names the run before the one that flagged it."""
    _, MissingPath = tracking_tables
    present = tmp_path / "present"
    present.mkdir()
    (present / "flag").write_text("x")

    tracked_table.insert1(dict(holder_id=4, some_path=str(present)))
    try:
        tracked_table.check_missing()  # run A: path whole, no finding
        shutil.rmtree(present)
        tracked_table.check_missing()  # run B: path gone, finding

        entry = (
            MissingPath() & {"full_table_name": tracked_table.full_table_name}
        ).fetch1("KEY")
        seen = MissingPath().last_covered_run(entry)
        flagged = (MissingPath() & entry).fetch1("run_id")
        assert seen is not None and seen < flagged
    finally:
        (
            MissingPath() & {"full_table_name": tracked_table.full_table_name}
        ).delete_quick()
        tracked_table.delete_quick()


def test_last_covered_run_bounds_the_attr_not_the_row(
    tracked_table, tracking_tables, tmp_path
):
    """The bound is the attribute's coverage, not the row's lifetime.

    Coverage is recorded per attribute, so a row inserted after a completed
    run still resolves to that run even though it did not exist then. The
    bound is therefore an upper bound, vacuous for a row newer than it.
    """
    _, MissingPath = tracking_tables

    tracked_table.insert1(dict(holder_id=5, some_path=str(tmp_path / "never")))
    try:
        tracked_table.check_missing()
        entry = (
            MissingPath() & {"full_table_name": tracked_table.full_table_name}
        ).fetch1("KEY")
        # Earlier tests banked completed coverage for this attribute
        assert MissingPath().last_covered_run(entry) is not None
    finally:
        (
            MissingPath() & {"full_table_name": tracked_table.full_table_name}
        ).delete_quick()
        tracked_table.delete_quick()


def test_refresh_drops_recovered_paths(
    tracked_table, tracking_tables, tmp_path
):
    """refresh_missing drops a restored path, so `lost` self-heals.

    Without it the table keeps reporting a loss after the file came back.
    A second loss then gets a fresh row naming the run that found it, so
    the bound describes the current absence and not the first one.
    """
    _, MissingPath = tracking_tables
    target = tmp_path / "comes_back"
    tracked_table.insert1(dict(holder_id=6, some_path=str(target)))
    mine = {"full_table_name": tracked_table.full_table_name}

    try:
        tracked_table.check_missing()
        first_run = (MissingPath() & mine).fetch1("run_id")

        target.mkdir()
        (target / "flag").write_text("x")
        assert tracked_table.refresh_missing() == 1
        assert len(MissingPath() & mine) == 0

        shutil.rmtree(target)
        tracked_table.check_missing()
        assert (MissingPath() & mine).fetch1("run_id") > first_run
    finally:
        (MissingPath() & mine).delete_quick()
        tracked_table.delete_quick()


# =============================== run_all =====================================


def test_run_all_covers_every_declared_table(
    tracked_table, tracking_tables, tmp_path
):
    """One run_all covers all declared tables under a single run_id.

    One run across tables is what lets `last_covered_run` compare like
    with like.
    """
    CheckMissingRun, MissingPath = tracking_tables
    tracked_table.insert1(dict(holder_id=7, some_path=str(tmp_path / "gone")))
    mine = {"full_table_name": tracked_table.full_table_name}

    try:
        run_id = CheckMissingRun().run_all(verbose=False)
        assert run_id is not None

        covered = CheckMissingRun.Coverage & {"run_id": run_id}
        # The fixture table registers, so a real run reaches it
        assert len(covered & mine) == 1
        # Closed in a finally, so the run is stamped either way
        assert (CheckMissingRun & {"run_id": run_id}).fetch1("finished")
    finally:
        (MissingPath() & mine).delete_quick()
        tracked_table.delete_quick()


def test_run_all_only_filters_by_class_name(tracking_tables):
    """`only` naming no registered table records nothing and says so."""
    CheckMissingRun, _ = tracking_tables
    before = len(CheckMissingRun())

    assert CheckMissingRun().run_all(only=["NotATable"], verbose=False) is None
    assert len(CheckMissingRun()) == before  # no run opened


def test_run_all_survives_a_failing_table(
    tracked_table, tracking_tables, monkeypatch
):
    """A table that raises leaves its coverage absent, not the run dead.

    Absent coverage is how a partial run stays visible to
    `last_covered_run`, rather than passing as clean.
    """
    CheckMissingRun, _ = tracking_tables

    def boom(self, *args, **kwargs):
        raise RuntimeError("scan exploded")

    monkeypatch.setattr(type(tracked_table), "check_missing", boom)

    run_id = CheckMissingRun().run_all(verbose=False)

    assert run_id is not None  # the run still completed and closed
    gap = CheckMissingRun.Coverage & {
        "run_id": run_id,
        "full_table_name": tracked_table.full_table_name,
    }
    assert len(gap) == 0


def test_summary_returns_coverage_for_a_run(
    tracked_table, tracking_tables, tmp_path
):
    """summary() hands the report a frame, defaulting to the latest run."""
    CheckMissingRun, MissingPath = tracking_tables
    tracked_table.insert1(dict(holder_id=8, some_path=str(tmp_path / "gone")))
    mine = {"full_table_name": tracked_table.full_table_name}

    try:
        run_id = CheckMissingRun().run_all(verbose=False)
        frame = CheckMissingRun().summary(run_id)

        assert not frame.empty
        assert set(frame.columns) >= {"attr", "n_checked", "n_missing"}
        assert CheckMissingRun().summary().equals(frame)  # latest by default
    finally:
        (MissingPath() & mine).delete_quick()
        tracked_table.delete_quick()


def test_lost_excludes_unreadable(tracking_tables, mixin_mod):
    """`lost` is the restore list, so it must not include denied paths."""
    _, MissingPath = tracking_tables
    assert "denied" not in mixin_mod.LOST_STATES
    assert "ok" not in mixin_mod.LOST_STATES
    assert set(MissingPath().lost.fetch("state")).isdisjoint({"denied"})

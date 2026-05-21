import os
from pathlib import Path

import pytest


class _FakeExternalTable:
    def __init__(self, external_paths):
        self._external_paths = external_paths

    def fetch_external_paths(self):
        return self._external_paths


class _EmptyQuery:
    def proj(self):
        return self

    def __bool__(self):
        return False

    def __len__(self):
        return 0

    def delete_quick(self):
        raise AssertionError("empty query should not be deleted")


class _FakeRegistry:
    def __init__(self):
        self.all_classes = []
        self.blocked = False
        self.unblocked = False

    def block_new_inserts(self, dry_run):
        assert dry_run is False
        self.blocked = True

    def unblock_new_inserts(self):
        self.unblocked = True


@pytest.fixture
def common_nwbfile(common):
    """Return a common NWBFile object."""
    return common.common_nwbfile


@pytest.fixture
def lockfile(base_dir, teardown):
    lockfile = base_dir / "temp.lock"
    lockfile.touch()
    os.environ["NWB_LOCK_FILE"] = str(lockfile)
    yield lockfile
    if teardown:
        os.remove(lockfile)


def test_add_to_lock(common_nwbfile, lockfile, mini_copy_name):
    common_nwbfile.Nwbfile.add_to_lock(mini_copy_name)
    with lockfile.open("r") as f:
        assert mini_copy_name in f.read()

    with pytest.raises(FileNotFoundError):
        common_nwbfile.Nwbfile.add_to_lock("non-existent-file.nwb")


def test_nwbfile_cleanup(common_nwbfile):
    before = len(common_nwbfile.Nwbfile.fetch())
    common_nwbfile.Nwbfile.cleanup(delete_files=False)
    after = len(common_nwbfile.Nwbfile.fetch())
    assert before == after, "Nwbfile cleanup changed table entry count."


def _cleanup_plan(common_nwbfile, *, scanned, tracked, delete):
    scanned_files = {Path(f"scan_{i}.nwb") for i in range(scanned)}
    files_to_delete = set(list(scanned_files)[:delete])
    return common_nwbfile.CleanupPlan(
        scanned_files=scanned_files,
        tracked_files={Path(f"tracked_{i}.nwb") for i in range(tracked)},
        files_to_delete=files_to_delete,
        empty_files=set(),
        untracked_files=files_to_delete,
    )


def test_build_untracked_file_plan_uses_resolved_external_paths(
    common_nwbfile, tmp_path
):
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()

    tracked = analysis_dir / "tracked.nwb"
    tracked.write_text("tracked")

    untracked = analysis_dir / "untracked.nwb"
    untracked.write_text("untracked")

    empty = analysis_dir / "empty.nwb"
    empty.touch()

    table = object.__new__(common_nwbfile.AnalysisNwbfile)
    table.__dict__["_analysis_dir"] = str(analysis_dir)
    table._ext_tbl = _FakeExternalTable(
        [("tracked", analysis_dir / "subdir" / ".." / "tracked.nwb")]
    )

    plan = table._build_untracked_file_plan(custom_tables=[])

    assert plan.scanned_files == {
        tracked.resolve(),
        untracked.resolve(),
        empty.resolve(),
    }
    assert plan.tracked_files == {tracked.resolve()}
    assert plan.empty_files == {empty.resolve()}
    assert plan.untracked_files == {untracked.resolve()}
    assert plan.files_to_delete == {empty.resolve(), untracked.resolve()}


def test_analysis_cleanup_plan_rejects_delete_without_tracked_files(
    common_nwbfile,
):
    plan = _cleanup_plan(common_nwbfile, scanned=1, tracked=0, delete=1)

    with pytest.raises(RuntimeError, match="no tracked analysis files"):
        common_nwbfile.AnalysisNwbfile._validate_cleanup_plan(plan)


def test_analysis_cleanup_plan_rejects_high_delete_fraction(common_nwbfile):
    plan = _cleanup_plan(common_nwbfile, scanned=4, tracked=3, delete=2)

    with pytest.raises(RuntimeError, match="above the safety limit"):
        common_nwbfile.AnalysisNwbfile._validate_cleanup_plan(
            plan,
            max_delete_fraction=0.25,
            max_delete_to_tracked_ratio=10.0,
        )


def test_analysis_cleanup_plan_rejects_high_delete_to_tracked_ratio(
    common_nwbfile,
):
    plan = _cleanup_plan(common_nwbfile, scanned=100, tracked=1, delete=11)

    with pytest.raises(RuntimeError, match="11.0x"):
        common_nwbfile.AnalysisNwbfile._validate_cleanup_plan(
            plan,
            max_delete_fraction=1.0,
            max_delete_to_tracked_ratio=10.0,
        )


def test_analysis_cleanup_plan_accepts_plausible_delete_plan(common_nwbfile):
    plan = _cleanup_plan(common_nwbfile, scanned=8, tracked=6, delete=1)

    common_nwbfile.AnalysisNwbfile._validate_cleanup_plan(plan)


def test_analysis_cleanup_plan_accepts_empty_delete(common_nwbfile):
    """delete_count == 0 short-circuits before any other validation kicks in."""
    # tracked=0 would normally trigger "no tracked analysis files"; the empty
    # delete plan must return before that check.
    plan = _cleanup_plan(common_nwbfile, scanned=3, tracked=0, delete=0)

    common_nwbfile.AnalysisNwbfile._validate_cleanup_plan(plan)


def test_analysis_cleanup_plan_accepts_fraction_at_threshold(common_nwbfile):
    """Fraction exactly at max_delete_fraction must pass (strict > guard)."""
    plan = _cleanup_plan(common_nwbfile, scanned=10, tracked=10, delete=9)

    common_nwbfile.AnalysisNwbfile._validate_cleanup_plan(
        plan, max_delete_fraction=0.9, max_delete_to_tracked_ratio=10.0
    )


def test_analysis_cleanup_plan_rejects_fraction_just_above_threshold(
    common_nwbfile,
):
    """Fraction just above max_delete_fraction must raise."""
    plan = _cleanup_plan(common_nwbfile, scanned=11, tracked=10, delete=10)

    with pytest.raises(RuntimeError, match="above the safety limit"):
        common_nwbfile.AnalysisNwbfile._validate_cleanup_plan(
            plan, max_delete_fraction=0.9, max_delete_to_tracked_ratio=100.0
        )


def test_analysis_cleanup_plan_accepts_ratio_at_threshold(common_nwbfile):
    """Ratio exactly at max_delete_to_tracked_ratio must pass."""
    plan = _cleanup_plan(common_nwbfile, scanned=100, tracked=1, delete=10)

    common_nwbfile.AnalysisNwbfile._validate_cleanup_plan(
        plan, max_delete_fraction=1.0, max_delete_to_tracked_ratio=10.0
    )


def test_build_untracked_file_plan_skips_symlinks(
    common_nwbfile, tmp_path, caplog
):
    """Symlinks under analysis_dir must not appear in any plan set,
    and must emit a warning naming the symlink and its target."""
    outside_target = tmp_path / "outside" / "shared.nwb"
    outside_target.parent.mkdir()
    outside_target.write_text("shared data")

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()

    real = analysis_dir / "real.nwb"
    real.write_text("real")

    link = analysis_dir / "escape.nwb"
    link.symlink_to(outside_target)

    table = object.__new__(common_nwbfile.AnalysisNwbfile)
    table.__dict__["_analysis_dir"] = str(analysis_dir)
    table._ext_tbl = _FakeExternalTable([])

    with caplog.at_level("WARNING"):
        plan = table._build_untracked_file_plan(custom_tables=[])

    assert outside_target.resolve() not in plan.scanned_files
    assert outside_target.resolve() not in plan.files_to_delete
    assert plan.scanned_files == {real.resolve()}
    assert any(
        "Skipping symlink" in record.message and str(link) in record.message
        for record in caplog.records
    ), "symlink skip should emit a warning naming the symlink path"


def test_remove_untracked_files_refuses_path_outside_analysis_dir(
    common_nwbfile, tmp_path
):
    """Plans containing paths outside analysis_dir must not unlink anything."""
    outside_target = tmp_path / "outside" / "shared.nwb"
    outside_target.parent.mkdir()
    outside_target.write_text("shared data")

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()

    plan = common_nwbfile.CleanupPlan(
        scanned_files={outside_target.resolve()},
        tracked_files=set(),
        files_to_delete={outside_target.resolve()},
        empty_files=set(),
        untracked_files={outside_target.resolve()},
    )

    table = object.__new__(common_nwbfile.AnalysisNwbfile)
    table.__dict__["_analysis_dir"] = str(analysis_dir)

    table._remove_untracked_files(custom_tables=[], dry_run=False, plan=plan)

    assert outside_target.exists()


def test_analysis_cleanup_dry_run_warns_on_refused_plan(
    common_nwbfile, monkeypatch, caplog
):
    """dry_run=True must surface a validation refusal as a warning, not raise."""
    registry = _FakeRegistry()
    # block_new_inserts is hard-coded to expect dry_run=False; relax for this test.
    monkeypatch.setattr(
        _FakeRegistry,
        "block_new_inserts",
        lambda self, dry_run: setattr(self, "blocked", True),
    )
    bad_plan = _cleanup_plan(common_nwbfile, scanned=4, tracked=3, delete=2)

    monkeypatch.setattr(common_nwbfile, "AnalysisRegistry", lambda: registry)
    monkeypatch.setattr(
        common_nwbfile.AnalysisNwbfile,
        "_build_untracked_file_plan",
        lambda self, custom_tables: bad_plan,
    )
    monkeypatch.setattr(
        common_nwbfile.AnalysisNwbfile,
        "_remove_untracked_files",
        lambda self, custom_tables, dry_run, plan: (set(), set()),
    )
    monkeypatch.setattr(
        common_nwbfile.AnalysisNwbfile,
        "get_orphans",
        lambda self: _EmptyQuery(),
    )
    monkeypatch.setattr(
        common_nwbfile.AnalysisNwbfile,
        "cleanup_external",
        lambda self, dry_run, delete_external_files: [],
    )

    table = object.__new__(common_nwbfile.AnalysisNwbfile)
    with caplog.at_level("WARNING"):
        common_nwbfile.AnalysisNwbfile.cleanup(
            table, dry_run=True, max_delete_fraction=0.25
        )

    assert any(
        "Cleanup plan would be refused" in record.message
        for record in caplog.records
    )


def test_analysis_cleanup_validates_plan_before_unlink(
    common_nwbfile, monkeypatch
):
    """cleanup() must abort destructive unlink when the plan exceeds limits."""
    registry = _FakeRegistry()
    bad_plan = _cleanup_plan(common_nwbfile, scanned=4, tracked=3, delete=2)

    def _remove_untracked_files(self, custom_tables, dry_run=True, plan=None):
        raise AssertionError("dangerous final cleanup plan should not unlink")

    monkeypatch.setattr(common_nwbfile, "AnalysisRegistry", lambda: registry)
    monkeypatch.setattr(
        common_nwbfile.AnalysisNwbfile,
        "_build_untracked_file_plan",
        lambda self, custom_tables: bad_plan,
    )
    monkeypatch.setattr(
        common_nwbfile.AnalysisNwbfile,
        "_remove_untracked_files",
        _remove_untracked_files,
    )
    monkeypatch.setattr(
        common_nwbfile.AnalysisNwbfile,
        "get_orphans",
        lambda self: _EmptyQuery(),
    )
    monkeypatch.setattr(
        common_nwbfile.AnalysisNwbfile,
        "cleanup_external",
        lambda self, dry_run, delete_external_files: [],
    )

    table = object.__new__(common_nwbfile.AnalysisNwbfile)
    with pytest.raises(RuntimeError, match="above the safety limit"):
        common_nwbfile.AnalysisNwbfile.cleanup(
            table, dry_run=False, max_delete_fraction=0.25
        )

    assert registry.blocked
    assert registry.unblocked


def test_cleanup_preserves_original_exception_when_unblock_raises(
    common_nwbfile, monkeypatch
):
    """If both the cleanup body AND unblock_new_inserts raise, the original
    cleanup-body exception must propagate (unblock failure becomes a log line,
    not the user-visible error)."""

    class _RaisingRegistry:
        all_classes = []
        blocked = False

        def block_new_inserts(self, dry_run):
            self.blocked = True

        def unblock_new_inserts(self):
            raise RuntimeError("unblock_kaboom")

    registry = _RaisingRegistry()
    bad_plan = _cleanup_plan(common_nwbfile, scanned=4, tracked=3, delete=2)

    monkeypatch.setattr(common_nwbfile, "AnalysisRegistry", lambda: registry)
    monkeypatch.setattr(
        common_nwbfile.AnalysisNwbfile,
        "_build_untracked_file_plan",
        lambda self, custom_tables: bad_plan,
    )
    monkeypatch.setattr(
        common_nwbfile.AnalysisNwbfile,
        "get_orphans",
        lambda self: _EmptyQuery(),
    )
    monkeypatch.setattr(
        common_nwbfile.AnalysisNwbfile,
        "cleanup_external",
        lambda self, dry_run, delete_external_files: [],
    )

    table = object.__new__(common_nwbfile.AnalysisNwbfile)
    # The validator should raise "above the safety limit", not "unblock_kaboom".
    with pytest.raises(RuntimeError, match="above the safety limit"):
        common_nwbfile.AnalysisNwbfile.cleanup(
            table, dry_run=False, max_delete_fraction=0.25
        )


def test_cleanup_propagates_unblock_failure_when_body_succeeds(
    common_nwbfile, monkeypatch
):
    """If the cleanup body succeeds but unblock_new_inserts raises, the unblock
    failure must propagate so the operator notices the stuck-blocked state."""

    class _RaisingRegistry:
        all_classes = []
        blocked = False

        def block_new_inserts(self, dry_run):
            self.blocked = True

        def unblock_new_inserts(self):
            raise RuntimeError("unblock_kaboom")

    registry = _RaisingRegistry()
    good_plan = _cleanup_plan(common_nwbfile, scanned=8, tracked=6, delete=1)

    monkeypatch.setattr(common_nwbfile, "AnalysisRegistry", lambda: registry)
    monkeypatch.setattr(
        common_nwbfile.AnalysisNwbfile,
        "_build_untracked_file_plan",
        lambda self, custom_tables: good_plan,
    )
    monkeypatch.setattr(
        common_nwbfile.AnalysisNwbfile,
        "_remove_untracked_files",
        lambda self, custom_tables, dry_run, plan: (set(), set()),
    )
    monkeypatch.setattr(
        common_nwbfile.AnalysisNwbfile,
        "get_orphans",
        lambda self: _EmptyQuery(),
    )
    monkeypatch.setattr(
        common_nwbfile.AnalysisNwbfile,
        "cleanup_external",
        lambda self, dry_run, delete_external_files: [],
    )

    table = object.__new__(common_nwbfile.AnalysisNwbfile)
    with pytest.raises(RuntimeError, match="unblock_kaboom"):
        common_nwbfile.AnalysisNwbfile.cleanup(
            table, dry_run=False, max_delete_fraction=0.9
        )

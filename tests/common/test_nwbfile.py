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
    untracked = analysis_dir / "untracked.nwb"
    empty = analysis_dir / "empty.nwb"
    tracked.write_text("tracked")
    untracked.write_text("untracked")
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
        common_nwbfile.AnalysisNwbfile._validate_cleanup_plan(plan)


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


def test_analysis_cleanup_revalidates_post_cleanup_plan(
    common_nwbfile, monkeypatch
):
    registry = _FakeRegistry()
    plans = [
        _cleanup_plan(common_nwbfile, scanned=100, tracked=100, delete=0),
        _cleanup_plan(common_nwbfile, scanned=4, tracked=3, delete=2),
    ]

    def _build_plan(self, custom_tables):
        return plans.pop(0)

    def _remove_untracked_files(self, custom_tables, dry_run=True, plan=None):
        raise AssertionError("dangerous final cleanup plan should not unlink")

    monkeypatch.setattr(common_nwbfile, "AnalysisRegistry", lambda: registry)
    monkeypatch.setattr(
        common_nwbfile.AnalysisNwbfile,
        "_build_untracked_file_plan",
        _build_plan,
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
        common_nwbfile.AnalysisNwbfile.cleanup(table, dry_run=False)

    assert registry.blocked
    assert registry.unblocked
    assert plans == []

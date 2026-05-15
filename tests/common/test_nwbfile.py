import os
from pathlib import Path

import pytest


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
            max_untracked_to_tracked_ratio=10.0,
        )


def test_analysis_cleanup_plan_accepts_plausible_delete_plan(common_nwbfile):
    plan = _cleanup_plan(common_nwbfile, scanned=8, tracked=6, delete=1)

    common_nwbfile.AnalysisNwbfile._validate_cleanup_plan(plan)

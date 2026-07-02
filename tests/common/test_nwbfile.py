import os
import subprocess

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


@pytest.mark.parametrize(
    "exc",
    [
        FileNotFoundError("conda not found"),
        subprocess.CalledProcessError(1, ["conda"]),
    ],
    ids=["conda_absent", "conda_failed"],
)
def test_conda_export_failure_does_not_abort_write(
    common_nwbfile, monkeypatch, exc
):
    """A conda-less / uv-only environment can still write analysis NWB: a
    failed ``conda env export`` records an 'environment capture unavailable'
    marker rather than raising and aborting the write."""
    analysis_tbl = common_nwbfile.AnalysisNwbfile()

    def _boom(*args, **kwargs):
        raise exc

    monkeypatch.setattr(subprocess, "check_output", _boom)

    info = analysis_tbl._logged_env_info()

    assert "environment capture unavailable" in info
    # The rest of the env info is still produced (write is not aborted).
    assert "spyglass=" in info

import os

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
        lockfile.unlink()


def test_get_file_name_error(common_nwbfile):
    """Test that an error is raised when trying non-existent file."""
    with pytest.raises(ValueError):
        common_nwbfile.NWBFile.get_file_name("non-existent-file.nwb")


def test_add_to_lock(common_nwbfile, lockfile, mini_copy_name):
    common_nwbfile.NWBFile.add_to_lock(mini_copy_name)
    with lockfile.open("r") as f:
        assert mini_copy_name in f.read()

    with pytest.raises(AssertionError):
        common_nwbfile.NWBFile.add_to_lock(mini_copy_name)


def test_nwbfile_cleanup(common_nwbfile):
    before = len(common_nwbfile.NWBFile.fetch())
    common_nwbfile.NWBFile.add_to_lock(delete_files=False)
    after = len(common_nwbfile.NWBFile.fetch())
    assert before == after, "Nwbfile cleanup changed table entry count."

import shutil
import warnings
from pathlib import Path

import pynwb
import pytest
from hdmf.backends.warnings import BrokenLinkWarning


@pytest.fixture(scope="session")
def copy_nwb_link_raw_ephys(data_import):
    from spyglass.data_import.insert_sessions import (
        copy_nwb_link_raw_ephys,
    )  # noqa: E402

    return copy_nwb_link_raw_ephys


def test_open_path(mini_path, mini_open):
    this_acq = mini_open.acquisition
    assert "e-series" in this_acq, "Ephys link no longer exists"
    assert (
        str(mini_path) == this_acq["e-series"].data.file.filename
    ), "Path of ephys link is incorrect"


@pytest.mark.slow
def test_copy_link(mini_path, settings, mini_closed, copy_nwb_link_raw_ephys):
    """Test readability after moving the linking raw file, breaking link"""
    new_path = Path(settings.raw_dir) / "no_ephys.nwb"
    new_moved = Path(settings.temp_dir) / "no_ephys_moved.nwb"

    copy_nwb_link_raw_ephys(mini_path.name, new_path.name)
    shutil.move(new_path, new_moved)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        with pynwb.NWBHDF5IO(path=str(new_moved), mode="r") as io:
            with pytest.warns(BrokenLinkWarning):
                nwb_acq = io.read().acquisition
    assert "e-series" not in nwb_acq, "Ephys link still exists after move"


@pytest.fixture
def two_raw_files(common, data_import):
    """Two minimal ingestible raw files.

    Built with pynwb's mocks rather than copied from the shared test file:
    this test only needs each file to reach Session, and a mock is kilobytes
    against the test file's tens of megabytes.
    """
    from pynwb.testing.mock.file import mock_NWBFile, mock_Subject

    from spyglass.settings import raw_dir
    from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename

    names = ["multi_insert_a.nwb", "multi_insert_b.nwb"]
    paths = [Path(raw_dir) / name for name in names]

    for index, path in enumerate(paths):
        nwbfile = mock_NWBFile(
            identifier=f"multi_insert_{index}",
            session_description="Mock file for multi-file insert_sessions",
        )
        mock_Subject(nwbfile=nwbfile)
        path.unlink(missing_ok=True)
        with pynwb.NWBHDF5IO(path, mode="w") as io:
            io.write(nwbfile)

    copy_names = [get_nwb_copy_filename(name) for name in names]

    yield names, copy_names

    # Always clean up: a leftover Nwbfile entry gives populate() phantom work
    # on the next run. See tests/README.md on fixture teardown.
    for copy_name in copy_names:
        (common.Nwbfile & {"nwb_file_name": copy_name}).delete(safemode=False)
    for path in paths:
        path.unlink(missing_ok=True)


def test_insert_sessions_processes_every_file(two_raw_files, common):
    """A list argument ingests every file, not only the first.

    `insert_sessions` used to return from inside its loop, so files after the
    first were silently skipped.
    """
    from spyglass.data_import import insert_sessions

    names, copy_names = two_raw_files

    results = insert_sessions(names, raise_err=True)

    assert isinstance(results, list), "Expected one result per file"
    assert len(results) == len(names), (
        f"Processed {len(results)} of {len(names)} files -- a file after the "
        + "first was skipped"
    )

    for copy_name in copy_names:
        assert common.Nwbfile & {
            "nwb_file_name": copy_name
        }, f"{copy_name} was never registered"
        assert common.Session & {
            "nwb_file_name": copy_name
        }, f"{copy_name} registered but not ingested"

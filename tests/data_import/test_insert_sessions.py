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

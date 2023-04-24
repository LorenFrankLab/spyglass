import datetime
import os
import pathlib
import shutil

import datajoint as dj
import pynwb
import pytest
from hdmf.backends.warnings import BrokenLinkWarning
from spyglass.data_import.insert_sessions import copy_nwb_link_raw_ephys


@pytest.fixture()
def new_nwbfile_raw_file_name(tmp_path):
    nwbfile = pynwb.NWBFile(
        session_description="session_description",
        identifier="identifier",
        session_start_time=datetime.datetime.now(datetime.timezone.utc),
    )

    device = nwbfile.create_device("dev1")
    group = nwbfile.create_electrode_group(
        "tetrode1", "tetrode description", "tetrode location", device
    )
    nwbfile.add_electrode(
        id=1,
        x=1.0,
        y=2.0,
        z=3.0,
        imp=-1.0,
        location="CA1",
        filtering="none",
        group=group,
        group_name="tetrode1",
    )
    region = nwbfile.create_electrode_table_region(
        region=[0], description="electrode 1"
    )

    es = pynwb.ecephys.ElectricalSeries(
        name="test_ts",
        data=[1, 2, 3],
        timestamps=[1.0, 2.0, 3.0],
        electrodes=region,
    )
    nwbfile.add_acquisition(es)

    spyglass_base_dir = tmp_path / "nwb-data"
    os.environ["SPYGLASS_BASE_DIR"] = str(spyglass_base_dir)
    os.mkdir(os.environ["SPYGLASS_BASE_DIR"])

    raw_dir = spyglass_base_dir / "raw"
    os.mkdir(raw_dir)

    dj.config["stores"] = {
        "raw": {"protocol": "file", "location": str(raw_dir), "stage": str(raw_dir)},
    }

    file_name = "raw.nwb"
    file_path = raw_dir / file_name
    with pynwb.NWBHDF5IO(str(file_path), mode="w") as io:
        io.write(nwbfile)

    return file_name


@pytest.fixture()
def new_nwbfile_no_ephys_file_name():
    return "raw_no_ephys.nwb"


@pytest.fixture()
def moved_nwbfile_no_ephys_file_path(tmp_path, new_nwbfile_no_ephys_file_name):
    return tmp_path / new_nwbfile_no_ephys_file_name


def test_copy_nwb(
    new_nwbfile_raw_file_name,
    new_nwbfile_no_ephys_file_name,
    moved_nwbfile_no_ephys_file_path,
):
    copy_nwb_link_raw_ephys(new_nwbfile_raw_file_name, new_nwbfile_no_ephys_file_name)

    # new file should not have ephys data
    base_dir = pathlib.Path(os.getenv("SPYGLASS_BASE_DIR", None))
    new_nwbfile_raw_file_name_abspath = base_dir / "raw" / new_nwbfile_raw_file_name
    out_nwb_file_abspath = base_dir / "raw" / new_nwbfile_no_ephys_file_name
    with pynwb.NWBHDF5IO(path=str(out_nwb_file_abspath), mode="r") as io:
        nwbfile = io.read()
        assert (
            "test_ts" in nwbfile.acquisition
        )  # this still exists but should be a link now
        assert nwbfile.acquisition["test_ts"].data.file.filename == str(
            new_nwbfile_raw_file_name_abspath
        )

    # test readability after moving the linking raw file (paths are stored as relative paths in NWB)
    # so this should break the link (moving the linked-to file should also break the link)
    shutil.move(out_nwb_file_abspath, moved_nwbfile_no_ephys_file_path)
    with pynwb.NWBHDF5IO(path=str(moved_nwbfile_no_ephys_file_path), mode="r") as io:
        with pytest.warns(BrokenLinkWarning):
            nwbfile = io.read()  # should raise BrokenLinkWarning
        assert "test_ts" not in nwbfile.acquisition

"""Ingestion of a file's `invalid_times` into IntervalList. See #1336.

`epochs` and `invalid_times` are both `TimeIntervals` tables read through the
same mapping, so the interesting case is that the two do not collide: an
untagged row of either names itself from its row id, and only the prefix keeps
`invalid_times` row 0 distinct from `epochs` row 0.
"""

from pathlib import Path

import pytest
from pynwb import NWBHDF5IO
from pynwb.testing.mock.file import mock_NWBFile, mock_Subject


@pytest.fixture(scope="module")
def import_invalid_times_nwb(verbose_context):
    from spyglass.common import Nwbfile
    from spyglass.data_import import insert_sessions
    from spyglass.settings import raw_dir

    nwbfile = mock_NWBFile(
        identifier="invalid_times_import_demo",
        session_description="Mock NWB file demonstrating invalid_times import",
    )
    mock_Subject(nwbfile=nwbfile)

    # One tagged epoch and one untagged, so both naming paths are exercised.
    # An epochs table with a tags column requires tags on every row.
    nwbfile.add_epoch(start_time=0.0, stop_time=10.0, tags=["01_r1"])
    nwbfile.add_epoch(start_time=10.0, stop_time=20.0, tags=[])

    # Untagged, so these name themselves `interval_0` and `interval_1` --
    # the same names epochs row 0 and row 1 would take without their tags.
    nwbfile.add_invalid_time_interval(start_time=3.0, stop_time=4.0)
    nwbfile.add_invalid_time_interval(start_time=15.0, stop_time=16.0)

    raw_file_name = "test_invalid_times.nwb"
    copy_file_name = "test_invalid_times_.nwb"
    file_path = Path(raw_dir) / raw_file_name
    nwb_dict = dict(nwb_file_name=copy_file_name)
    file_path.unlink(missing_ok=True)

    with NWBHDF5IO(file_path, mode="w") as io:
        io.write(nwbfile)

    insert_sessions([str(file_path)], raise_err=True)

    yield nwb_dict

    with verbose_context:
        file_path.unlink(missing_ok=True)
        (Nwbfile & nwb_dict).delete(safemode=False)


def test_invalid_times_imported(common, import_invalid_times_nwb):
    """Both tables land, and the invalid ones are namespaced apart."""
    key = import_invalid_times_nwb
    names = set((common.IntervalList & key).fetch("interval_list_name"))

    expected = {
        "01_r1",
        "interval_1",
        "invalid_interval_0",
        "invalid_interval_1",
    }
    assert expected <= names, (
        f"Missing interval list names {expected - names}. "
        + f"IntervalList holds {sorted(names)}"
    )


def test_invalid_times_values(common, import_invalid_times_nwb):
    """An invalid interval keeps the start/stop time it was written with."""
    key = import_invalid_times_nwb
    valid_times = (
        common.IntervalList & key & {"interval_list_name": "invalid_interval_0"}
    ).fetch1("valid_times")

    assert valid_times.tolist() == [
        [3.0, 4.0]
    ], f"Unexpected valid_times for invalid_interval_0: {valid_times}"

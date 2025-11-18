from pynwb import NWBHDF5IO
from pynwb.testing.mock.file import mock_NWBFile, mock_Subject
from pynwb.behavior import SpatialSeries, CompassDirection
from pathlib import Path
import numpy as np
import pytest


@pytest.fixture(scope="module")
def import_compass_nwb(
    verbose_context,
):
    from spyglass.settings import raw_dir
    from spyglass.data_import import insert_sessions
    from spyglass.common import Nwbfile

    nwbfile = mock_NWBFile(
        identifier="compass_direction_bug_demo",
        session_description="Mock NWB file demonstrating Spyglass CompassDirection import bug",
    )
    mock_Subject(nwbfile=nwbfile)
    behavior_module = nwbfile.create_processing_module(
        name="behavior",
        description="Behavioral data including position and compass direction",
    )

    compass_data = []
    for i in range(2):
        timestamps = np.linspace(i, i + 1, 1000)

        direction_spatial_series = SpatialSeries(
            name=f"head_direction {i}",
            description="Horizontal angle of the head (yaw) in radians",
            data=np.zeros_like(timestamps),
            timestamps=timestamps,
            reference_frame="arena coordinates",
            unit="radians",
        )
        compass_data.append(direction_spatial_series)
    compass_direction_obj = CompassDirection(spatial_series=compass_data)
    behavior_module.add(compass_direction_obj)

    # --- Write to file
    raw_file_name = "test_imported_compass.nwb"
    copy_file_name = "test_imported_compass_.nwb"
    file_path = Path(raw_dir) / raw_file_name
    nwb_dict = dict(nwb_file_name=copy_file_name)
    if file_path.exists():
        file_path.unlink(missing_ok=True)

    with NWBHDF5IO(file_path, mode="w") as io:
        io.write(nwbfile)

    # --- Insert compass direction data
    insert_sessions([str(file_path)], raise_err=True)

    yield nwb_dict

    with verbose_context:
        file_path.unlink(missing_ok=True)
        (Nwbfile & nwb_dict).delete(safemode=False)


def test_imported_compass(common, import_compass_nwb):
    key = import_compass_nwb

    query = common.RawCompassDirection & key
    assert (
        len(query) == 2
    ), f"Expected 2 imported compass direction entries, found {len(query)}"

    assert all(
        [
            x in query.fetch("interval_list_name")
            for x in ["compass 1 valid times", "compass 2 valid times"]
        ]
    ), "Imported compass direction interval list names do not match expected names"

    assert (
        query.fetch_nwb()[0]["compass"].data.size == 1000
    ), "Imported compass direction data size does not match expected size"

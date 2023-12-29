import datetime
import shutil
from pathlib import Path

import pynwb
import pytest
from hdmf.backends.warnings import BrokenLinkWarning


@pytest.fixture(scope="session")
def new_raw_name():
    return "raw.nwb"


@pytest.fixture(scope="session")
def write_new_raw(new_raw_name, settings):
    nwbfile = pynwb.NWBFile(
        session_description="session_description",
        identifier="identifier",
        session_start_time=datetime.datetime.now(datetime.timezone.utc),
    )

    nwbfile.add_electrode(
        id=1,
        x=1.0,
        y=2.0,
        z=3.0,
        imp=-1.0,
        location="CA1",
        filtering="none",
        group=nwbfile.create_electrode_group(
            "tetrode1",
            "tetrode description",
            "tetrode location",
            nwbfile.create_device("dev1"),
        ),
        group_name="tetrode1",
    )

    nwbfile.add_acquisition(
        pynwb.ecephys.ElectricalSeries(
            name="test_ts",
            data=[1, 2, 3],
            timestamps=[1.0, 2.0, 3.0],
            electrodes=nwbfile.create_electrode_table_region(
                region=[0], description="electrode 1"
            ),
        ),
    )

    file_path = Path(settings.raw_dir) / new_raw_name

    with pynwb.NWBHDF5IO(str(file_path), mode="w") as io:
        io.write(nwbfile)


@pytest.fixture(scope="session")
def no_ephys_name():
    return "raw_no_ephys.nwb"


@pytest.fixture(scope="session")
def no_ephys_path_moved(settings, no_ephys_name):
    from pathlib import Path

    return Path(settings.temp_dir) / no_ephys_name


def test_copy_nwb(
    new_raw_name,
    no_ephys_name,
    no_ephys_path_moved,
    copy_nwb_link_raw_ephys,
    settings,
    write_new_raw,
    minirec_content,
):
    copy_nwb_link_raw_ephys(new_raw_name, no_ephys_name)
    raw_path = Path(settings.raw_dir)

    # new file should not have ephys data
    new_raw_abspath = raw_path / new_raw_name
    no_ephys_abspath = raw_path / no_ephys_name
    with pynwb.NWBHDF5IO(path=str(no_ephys_abspath), mode="r") as io:
        nwb_acq = io.read().acquisition
        assert nwb_acq["test_ts"].data.file.filename == str(new_raw_abspath)

    assert "test_ts" in nwb_acq, "Ephys link no longer exists"

    # test readability after moving the linking raw file (paths are stored as
    # relative paths in NWB) so this should break the link (moving the
    # linked-to file should also break the link)

    shutil.move(no_ephys_abspath, no_ephys_path_moved)

    with pynwb.NWBHDF5IO(path=str(no_ephys_path_moved), mode="r") as io:
        with pytest.warns(BrokenLinkWarning):
            nwb_acq = io.read().acquisition
    assert "test_ts" not in nwb_acq, "Ephys link still exists"


def trim_file(
    file_in="beans20190718.nwb",
    file_out="beans20190718_trimmed.nwb",
    old_spatial_series=True,
):
    file_in = "beans20190718.nwb"
    file_out = "beans20190718_trimmed.nwb"

    n_timestamps_to_keep = 20  # / 20000 Hz sampling rate = 1 ms

    with pynwb.NWBHDF5IO(file_in, "r", load_namespaces=True) as io:
        nwbfile = io.read()
        orig_eseries = nwbfile.acquisition.pop("e-series")

        # create a new ElectricalSeries with a subset of the data and timestamps
        data = orig_eseries.data[0:n_timestamps_to_keep, :]
        ts = orig_eseries.timestamps[0:n_timestamps_to_keep]

        electrodes = nwbfile.create_electrode_table_region(
            region=orig_eseries.electrodes.data[:].tolist(),
            name=orig_eseries.electrodes.name,
            description=orig_eseries.electrodes.description,
        )
        new_eseries = pynwb.ecephys.ElectricalSeries(
            name=orig_eseries.name,
            description=orig_eseries.description,
            data=data,
            timestamps=ts,
            electrodes=electrodes,
        )
        nwbfile.add_acquisition(new_eseries)

        # create a new analog TimeSeries with a subset of the data and timestamps
        orig_analog = nwbfile.processing["analog"]["analog"].time_series.pop(
            "analog"
        )
        data = orig_analog.data[0:n_timestamps_to_keep, :]
        ts = orig_analog.timestamps[0:n_timestamps_to_keep]
        new_analog = pynwb.TimeSeries(
            name=orig_analog.name,
            description=orig_analog.description,
            data=data,
            timestamps=ts,
            unit=orig_analog.unit,
        )
        nwbfile.processing["analog"]["analog"].add_timeseries(new_analog)

        if old_spatial_series:
            # remove last two columns of all SpatialSeries data (xloc2, yloc2)
            # because it does not conform with NWB 2.5 and they are all zeroes
            # anyway

            new_spatial_series = list()
            for spatial_series_name in list(
                nwbfile.processing["behavior"]["position"].spatial_series
            ):
                spatial_series = nwbfile.processing["behavior"][
                    "position"
                ].spatial_series.pop(spatial_series_name)
                assert isinstance(spatial_series, pynwb.behavior.SpatialSeries)
                data = spatial_series.data[:, 0:2]
                ts = spatial_series.timestamps[0:n_timestamps_to_keep]
                new_spatial_series.append(
                    pynwb.behavior.SpatialSeries(
                        name=spatial_series.name,
                        description=spatial_series.description,
                        data=data,
                        timestamps=spatial_series.timestamps,
                        reference_frame=spatial_series.reference_frame,
                    )
                )

        for spatial_series in new_spatial_series:
            nwbfile.processing["behavior"]["position"].add_spatial_series(
                spatial_series
            )

        with pynwb.NWBHDF5IO(file_out, "w") as export_io:
            export_io.export(io, nwbfile)

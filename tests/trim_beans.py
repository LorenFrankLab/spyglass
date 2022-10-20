import pynwb

# import ndx_franklab_novela

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
    orig_analog = nwbfile.processing["analog"]["analog"].time_series.pop("analog")
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

    # remove last two columns of all SpatialSeries data (xloc2, yloc2) because
    # it does not conform with NWB 2.5 and they are all zeroes anyway
    new_spatial_series = list()
    for spatial_series_name in list(
        nwbfile.processing["behavior"]["position"].spatial_series
    ):
        spatial_series = nwbfile.processing["behavior"]["position"].spatial_series.pop(
            spatial_series_name
        )
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
        nwbfile.processing["behavior"]["position"].add_spatial_series(spatial_series)

    with pynwb.NWBHDF5IO(file_out, "w") as export_io:
        export_io.export(io, nwbfile)

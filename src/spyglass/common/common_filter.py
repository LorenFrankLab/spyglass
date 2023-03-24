# code to define filters that can be applied to continuous time data
import warnings

import datajoint as dj
import matplotlib.pyplot as plt
import numpy as np
import psutil
import pynwb
import scipy.signal as signal

from ..utils.nwb_helper_fn import get_electrode_indices

schema = dj.schema("common_filter")


def _import_ghostipy():
    try:
        import ghostipy as gsp

        return gsp
    except ImportError as e:
        raise ImportError(
            "You must install ghostipy to use filtering methods. Please note that to install ghostipy on "
            "an Mac M1, you must first install pyfftw from conda-forge."
        ) from e


@schema
class FirFilter(dj.Manual):
    definition = """
    filter_name: varchar(80)           # descriptive name of this filter
    filter_sampling_rate: int          # sampling rate for this filter
    ---
    filter_type: enum("lowpass", "highpass", "bandpass")
    filter_low_stop = 0: float         # lowest frequency for stop band for low frequency side of filter
    filter_low_pass = 0: float         # lowest frequency for pass band of low frequency side of filter
    filter_high_pass = 0: float        # highest frequency for pass band for high frequency side of filter
    filter_high_stop = 0: float        # highest frequency for stop band of high frequency side of filter
    filter_comments: varchar(2000)     # comments about the filter
    filter_band_edges: blob            # numpy array containing the filter bands (redundant with individual parameters)
    filter_coeff: longblob             # numpy array containing the filter coefficients
    """

    def add_filter(self, filter_name, fs, filter_type, band_edges, comments=""):
        gsp = _import_ghostipy()

        # add an FIR bandpass filter of the specified type ('lowpass', 'highpass', or 'bandpass').
        # band_edges should be as follows:
        #   low pass filter: [high_pass high_stop]
        #   high pass filter: [low stop low pass]
        #   band pass filter: [low_stop low_pass high_pass high_stop].
        if filter_type not in ["lowpass", "highpass", "bandpass"]:
            print(
                "Error in Filter.add_filter: filter type {} is not "
                "lowpass"
                ", "
                "highpass"
                " or "
                """bandpass""".format(filter_type)
            )
            return None

        p = 2  # transition spline will be quadratic
        if filter_type == "lowpass" or filter_type == "highpass":
            # check that two frequencies were passed in and that they are in the right order
            if len(band_edges) != 2:
                print(
                    "Error in Filter.add_filter: lowpass and highpass filter requires two band_frequencies"
                )
                return None
            tw = band_edges[1] - band_edges[0]

        elif filter_type == "bandpass":
            if len(band_edges) != 4:
                print(
                    "Error in Filter.add_filter: bandpass filter requires four band_frequencies."
                )
                return None
            # the transition width is the mean of the widths of left and right transition regions
            tw = (
                (band_edges[1] - band_edges[0]) + (band_edges[3] - band_edges[2])
            ) / 2.0

        else:
            raise Exception(f"Unexpected filter type: {filter_type}")

        numtaps = gsp.estimate_taps(fs, tw)
        filterdict = dict()
        filterdict["filter_name"] = filter_name
        filterdict["filter_sampling_rate"] = fs
        filterdict["filter_comments"] = comments

        # set the desired frequency response
        if filter_type == "lowpass":
            desired = [1, 0]
            filterdict["filter_low_stop"] = 0
            filterdict["filter_low_pass"] = 0
            filterdict["filter_high_pass"] = band_edges[0]
            filterdict["filter_high_stop"] = band_edges[1]
        elif filter_type == "highpass":
            desired = [0, 1]
            filterdict["filter_low_stop"] = band_edges[0]
            filterdict["filter_low_pass"] = band_edges[1]
            filterdict["filter_high_pass"] = 0
            filterdict["filter_high_stop"] = 0
        else:
            desired = [0, 1, 1, 0]
            filterdict["filter_low_stop"] = band_edges[0]
            filterdict["filter_low_pass"] = band_edges[1]
            filterdict["filter_high_pass"] = band_edges[2]
            filterdict["filter_high_stop"] = band_edges[3]
        filterdict["filter_type"] = filter_type
        filterdict["filter_band_edges"] = np.asarray(band_edges)
        # create 1d array for coefficients
        filterdict["filter_coeff"] = np.array(
            gsp.firdesign(numtaps, band_edges, desired, fs=fs, p=p), ndmin=1
        )
        # add this filter to the table
        self.insert1(filterdict, skip_duplicates=True)

    def plot_magnitude(self, filter_name, fs):
        filter = (
            self & {"filter_name": filter_name} & {"filter_sampling_rate": fs}
        ).fetch(as_dict=True)
        f = filter[0]
        plt.figure()
        w, h = signal.freqz(filter[0]["filter_coeff"], worN=65536)
        magnitude = 20 * np.log10(np.abs(h))
        plt.plot(w / np.pi * fs / 2, magnitude)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.title("Frequency Response")
        plt.xlim(0, np.max(f["filter_coeffand_edges"] * 2))
        plt.ylim(np.min(magnitude), -1 * np.min(magnitude) * 0.1)
        plt.grid(True)

    def plot_fir_filter(self, filter_name, fs):
        filter = (
            self & {"filter_name": filter_name} & {"filter_sampling_rate": fs}
        ).fetch(as_dict=True)
        f = filter[0]
        plt.figure()
        plt.clf()
        plt.plot(f["filter_coeff"], "k")
        plt.xlabel("Coefficient")
        plt.ylabel("Magnitude")
        plt.title("Filter Taps")
        plt.grid(True)

    def filter_delay(self, filter_name, fs):
        # return the filter delay
        filter = (
            self & {"filter_name": filter_name} & {"filter_sampling_rate": fs}
        ).fetch(as_dict=True)
        return self.calc_filter_delay(filter["filter_coeff"])

    def filter_data_nwb(
        self,
        analysis_file_abs_path,
        eseries,
        filter_coeff,
        valid_times,
        electrode_ids,
        decimation,
    ):
        """
        :param analysis_nwb_file_name: str   full path to previously created analysis nwb file where filtered data
        should be stored. This also has the name of the original NWB file where the data will be taken from
        :param eseries: electrical series with data to be filtered
        :param filter_coeff: numpy array with filter coefficients for FIR filter
        :param valid_times: 2D numpy array with start and stop times of intervals to be filtered
        :param electrode_ids: list of electrode_ids to filter
        :param decimation: int decimation factor
        :return: The NWB object id of the filtered data (str), list containing first and last timestamp

        This function takes data and timestamps from an NWB electrical series and filters them using the ghostipy
        package, saving the result as a new electricalseries in the nwb_file_name, which should have previously been
        created and linked to the original NWB file using common_session.AnalysisNwbfile.create()
        """
        gsp = _import_ghostipy()

        data_on_disk = eseries.data
        timestamps_on_disk = eseries.timestamps
        n_dim = len(data_on_disk.shape)
        n_samples = len(timestamps_on_disk)
        # find the
        time_axis = 0 if data_on_disk.shape[0] == n_samples else 1
        electrode_axis = 1 - time_axis
        n_electrodes = data_on_disk.shape[electrode_axis]
        input_dim_restrictions = [None] * n_dim

        # to get the input dimension restrictions we need to look at the electrode table for the eseries and get
        # the indices from that
        input_dim_restrictions[electrode_axis] = np.s_[
            get_electrode_indices(eseries, electrode_ids)
        ]

        indices = []
        output_shape_list = [0] * n_dim
        output_shape_list[electrode_axis] = len(electrode_ids)
        output_offsets = [0]

        timestamp_size = timestamps_on_disk[0].itemsize
        timestamp_dtype = timestamps_on_disk[0].dtype
        data_size = data_on_disk[0][0].itemsize
        data_dtype = data_on_disk[0][0].dtype

        filter_delay = self.calc_filter_delay(filter_coeff)
        for a_start, a_stop in valid_times:
            if a_start < timestamps_on_disk[0]:
                warnings.warn(
                    f"Interval start time {a_start} is smaller than first timestamp {timestamps_on_disk[0]}, "
                    "using first timestamp instead"
                )
                a_start = timestamps_on_disk[0]
            if a_stop > timestamps_on_disk[-1]:
                warnings.warn(
                    f"Interval stop time {a_stop} is larger than last timestamp {timestamps_on_disk[-1]}, "
                    "using last timestamp instead"
                )
                a_stop = timestamps_on_disk[-1]
            frm, to = np.searchsorted(timestamps_on_disk, (a_start, a_stop))
            if to > n_samples:
                to = n_samples
            indices.append((frm, to))
            shape, dtype = gsp.filter_data_fir(
                data_on_disk,
                filter_coeff,
                axis=time_axis,
                input_index_bounds=[frm, to - 1],
                output_index_bounds=[filter_delay, filter_delay + to - frm],
                describe_dims=True,
                ds=decimation,
                input_dim_restrictions=input_dim_restrictions,
            )
            output_offsets.append(output_offsets[-1] + shape[time_axis])
            output_shape_list[time_axis] += shape[time_axis]

        # open the nwb file to create the dynamic table region and electrode series, then write and close the file
        with pynwb.NWBHDF5IO(
            path=analysis_file_abs_path, mode="a", load_namespaces=True
        ) as io:
            nwbf = io.read()
            # get the indices of the electrodes in the electrode table
            elect_ind = get_electrode_indices(nwbf, electrode_ids)

            electrode_table_region = nwbf.create_electrode_table_region(
                elect_ind, "filtered electrode table"
            )
            eseries_name = "filtered data"
            es = pynwb.ecephys.ElectricalSeries(
                name=eseries_name,
                data=np.empty(tuple(output_shape_list), dtype=data_dtype),
                electrodes=electrode_table_region,
                timestamps=np.empty(output_shape_list[time_axis]),
            )
            # Add the electrical series to the scratch area
            nwbf.add_scratch(es)
            io.write(nwbf)

            # reload the NWB file to get the h5py objects for the data and the timestamps
            with pynwb.NWBHDF5IO(
                path=analysis_file_abs_path, mode="a", load_namespaces=True
            ) as io:
                nwbf = io.read()
                es = nwbf.objects[es.object_id]
                filtered_data = es.data
                new_timestamps = es.timestamps
                indices = np.array(indices, ndmin=2)
                # Filter and write the output dataset
                ts_offset = 0

                print("Filtering data")
                for ii, (start, stop) in enumerate(indices):
                    # calculate the size of the timestamps and the data and determine whether they
                    # can be loaded into < 90% of available RAM
                    mem = psutil.virtual_memory()
                    interval_samples = stop - start
                    if (
                        interval_samples * (timestamp_size + n_electrodes * data_size)
                        < 0.9 * mem.available
                    ):
                        print(f"Interval {ii}: loading data into memory")
                        timestamps = np.asarray(
                            timestamps_on_disk[start:stop], dtype=timestamp_dtype
                        )
                        if time_axis == 0:
                            data = np.asarray(
                                data_on_disk[start:stop, :], dtype=data_dtype
                            )
                        else:
                            data = np.asarray(
                                data_on_disk[:, start:stop], dtype=data_dtype
                            )
                        extracted_ts = timestamps[0::decimation]
                        new_timestamps[
                            ts_offset : ts_offset + len(extracted_ts)
                        ] = extracted_ts
                        ts_offset += len(extracted_ts)
                        # filter the data
                        gsp.filter_data_fir(
                            data,
                            filter_coeff,
                            axis=time_axis,
                            input_index_bounds=[0, interval_samples - 1],
                            output_index_bounds=[
                                filter_delay,
                                filter_delay + stop - start,
                            ],
                            ds=decimation,
                            input_dim_restrictions=input_dim_restrictions,
                            outarray=filtered_data,
                            output_offset=output_offsets[ii],
                        )
                    else:
                        print(f"Interval {ii}: leaving data on disk")
                        data = data_on_disk
                        timestamps = timestamps_on_disk
                        extracted_ts = timestamps[start:stop:decimation]
                        new_timestamps[
                            ts_offset : ts_offset + len(extracted_ts)
                        ] = extracted_ts
                        ts_offset += len(extracted_ts)
                        # filter the data
                        gsp.filter_data_fir(
                            data,
                            filter_coeff,
                            axis=time_axis,
                            input_index_bounds=[start, stop],
                            output_index_bounds=[
                                filter_delay,
                                filter_delay + stop - start,
                            ],
                            ds=decimation,
                            input_dim_restrictions=input_dim_restrictions,
                            outarray=filtered_data,
                            output_offset=output_offsets[ii],
                        )

                start_end = [new_timestamps[0], new_timestamps[-1]]

                io.write(nwbf)

        return es.object_id, start_end

    def filter_data(
        self, timestamps, data, filter_coeff, valid_times, electrodes, decimation
    ):
        """
        :param timestamps: numpy array with list of timestamps for data
        :param data: original data array
        :param filter_coeff: numpy array with filter coefficients for FIR filter
        :param valid_times: 2D numpy array with start and stop times of intervals to be filtered
        :param electrodes: list of electrodes to filter
        :param decimation: decimation factor
        :return: filtered_data, timestamps
        """
        gsp = _import_ghostipy()

        n_dim = len(data.shape)
        n_samples = len(timestamps)
        time_axis = 0 if data.shape[0] == n_samples else 1
        electrode_axis = 1 - time_axis
        input_dim_restrictions = [None] * n_dim
        input_dim_restrictions[electrode_axis] = np.s_[electrodes]

        indices = []
        output_shape_list = [0] * n_dim
        output_shape_list[electrode_axis] = len(electrodes)
        output_offsets = [0]

        filter_delay = self.calc_filter_delay(filter_coeff)
        for a_start, a_stop in valid_times:
            if a_start < timestamps[0]:
                print(
                    f"Interval start time {a_start} is smaller than first timestamp "
                    f"{timestamps[0]}, using first timestamp instead"
                )
                a_start = timestamps[0]
            if a_stop > timestamps[-1]:
                print(
                    f"Interval stop time {a_stop} is larger than last timestamp "
                    f"{timestamps[-1]}, using last timestamp instead"
                )
                a_stop = timestamps[-1]
            frm, to = np.searchsorted(timestamps, (a_start, a_stop))
            if to > n_samples:
                to = n_samples

            indices.append((frm, to))
            shape, dtype = gsp.filter_data_fir(
                data,
                filter_coeff,
                axis=time_axis,
                input_index_bounds=[frm, to],
                output_index_bounds=[filter_delay, filter_delay + to - frm],
                describe_dims=True,
                ds=decimation,
                input_dim_restrictions=input_dim_restrictions,
            )
            output_offsets.append(output_offsets[-1] + shape[time_axis])
            output_shape_list[time_axis] += shape[time_axis]

        # create the dataset and the timestamps array
        filtered_data = np.empty(tuple(output_shape_list), dtype=data.dtype)

        new_timestamps = np.empty((output_shape_list[time_axis],), timestamps.dtype)

        indices = np.array(indices, ndmin=2)

        # Filter  the output dataset
        ts_offset = 0

        for ii, (start, stop) in enumerate(indices):
            extracted_ts = timestamps[start:stop:decimation]

            # print(f"Diffs {np.diff(extracted_ts)}")
            new_timestamps[ts_offset : ts_offset + len(extracted_ts)] = extracted_ts
            ts_offset += len(extracted_ts)

            # finally ready to filter data!
            gsp.filter_data_fir(
                data,
                filter_coeff,
                axis=time_axis,
                input_index_bounds=[start, stop],
                output_index_bounds=[filter_delay, filter_delay + stop - start],
                ds=decimation,
                input_dim_restrictions=input_dim_restrictions,
                outarray=filtered_data,
                output_offset=output_offsets[ii],
            )

        return filtered_data, new_timestamps

    def calc_filter_delay(self, filter_coeff):
        """
        :param filter_coeff:
        :return: filter delay
        """
        return (len(filter_coeff) - 1) // 2

    def create_standard_filters(self):
        """Add standard filters to the Filter table including
        0-400 Hz low pass for continuous raw data -> LFP
        """
        self.add_filter(
            "LFP 0-400 Hz",
            20000,
            "lowpass",
            [400, 425],
            "standard LFP filter for 20 KHz data",
        )
        self.add_filter(
            "LFP 0-400 Hz",
            30000,
            "lowpass",
            [400, 425],
            "standard LFP filter for 20 KHz data",
        )

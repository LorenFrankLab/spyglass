import warnings

import datajoint as dj
import ndx_franklab_novela
import numpy as np
import pandas as pd
import pynwb
import matplotlib.pyplot as plt
import psutil
import scipy.signal as signal

from ..utils.nwb_helper_fn import get_electrode_indices

from .common_device import Probe  # noqa: F401
from .common_filter import FirFilter
from .common_interval import (
    IntervalList,
    interval_list_censor,  # noqa: F401
    interval_list_contains_ind,
    interval_list_intersect,
)
from .common_nwbfile import AnalysisNwbfile, Nwbfile
from .common_region import BrainRegion  # noqa: F401
from .common_session import Session  # noqa: F401
from ..utils.dj_helper_fn import fetch_nwb  # dj_replace
from ..utils.nwb_helper_fn import (
    estimate_sampling_rate,
    get_data_interface,
    get_electrode_indices,
    get_nwb_file,
    get_valid_intervals,
    get_config,
)

def _import_ghostipy():
    try:
        import ghostipy as gsp

        return gsp
    except ImportError as e:
        raise ImportError(
            "You must install ghostipy to use filtering methods. Please note that to install ghostipy on "
            "an Mac M1, you must first install pyfftw from conda-forge."
        ) from e

schema = dj.schema("lfp_v1")

@schema
class LFPInterval(dj.Manual):
    definition = """
    -> Session
    lfp_interval_name: varchar(200)
    ---
    lfp_interval: longblob # numpy array of [start time, end_time]
    """

@schema
class LFPGroup(dj.Manual):
    definition = """
    # Set of electrodes to filter for LFP
    -> Session
    lfp_group_name: varchar(200)
    ---
    sort_reference_electrode_id = -1: int  # the electrode to use for reference. -1: no reference, -2: common median
    """

    class LFPElectrode(dj.Part):
        definition = """
        -> LFPGroup
        -> Electrode
        """

    def set_lfp_electrodes(self, nwb_file_name, electrode_list):
        """Removes all electrodes for the specified nwb file and then adds back the electrodes in the list

        Parameters
        ----------
        nwb_file_name : str
            The name of the nwb file for the desired session
        electrode_list : list
            list of electrodes to be used for LFP

        """
        # remove the session and then recreate the session and Electrode list
        (LFPSelection() & {"nwb_file_name": nwb_file_name}).delete()
        # check to see if the user allowed the deletion
        if len((LFPSelection() & {"nwb_file_name": nwb_file_name}).fetch()) == 0:
            LFPSelection().insert1({"nwb_file_name": nwb_file_name})

            # TODO: do this in a better way
            all_electrodes = (Electrode() & {"nwb_file_name": nwb_file_name}).fetch(
                as_dict=True
            )
            primary_key = Electrode.primary_key
            for e in all_electrodes:
                # create a dictionary so we can insert new elects
                if e["electrode_id"] in electrode_list:
                    lfpelectdict = {k: v for k, v in e.items() if k in primary_key}
                    LFPSelection().LFPElectrode.insert1(lfpelectdict, replace=True)


@schema
class LFP(dj.Imported):
    definition = """
    -> LFPSelection
    ---
    -> IntervalList             # the valid intervals for the data
    -> FirFilter                # the filter used for the data
    -> AnalysisNwbfile          # the name of the nwb file with the lfp data
    lfp_object_id: varchar(40)  # the NWB object ID for loading this object from the file
    lfp_sampling_rate: float    # the sampling rate, in HZ
    """

    def make(self, key):
        # get the NWB object with the data; FIX: change to fetch with additional infrastructure
        rawdata = Raw().nwb_object(key)
        sampling_rate, interval_list_name = (Raw() & key).fetch1(
            "sampling_rate", "interval_list_name"
        )
        sampling_rate = int(np.round(sampling_rate))

        valid_times = (
            IntervalList()
            & {
                "nwb_file_name": key["nwb_file_name"],
                "interval_list_name": interval_list_name,
            }
        ).fetch1("valid_times")
        # keep only the intervals > 1 second long
        min_interval_length = 1.0
        valid = []
        for count, interval in enumerate(valid_times):
            if interval[1] - interval[0] > min_interval_length:
                valid.append(count)
        valid_times = valid_times[valid]
        print(
            f"LFP: found {len(valid)} of {count+1} intervals > {min_interval_length} sec long."
        )

        # target 1 KHz sampling rate
        decimation = sampling_rate // 1000

        # get the LFP filter that matches the raw data
        filter = (
            FirFilter()
            & {"filter_name": "LFP 0-400 Hz"}
            & {"filter_sampling_rate": sampling_rate}
        ).fetch(as_dict=True)

        # there should only be one filter that matches, so we take the first of the dictionaries
        key["filter_name"] = filter[0]["filter_name"]
        key["filter_sampling_rate"] = filter[0]["filter_sampling_rate"]

        filter_coeff = filter[0]["filter_coeff"]
        if len(filter_coeff) == 0:
            print(
                f"Error in LFP: no filter found with data sampling rate of {sampling_rate}"
            )
            return None
        # get the list of selected LFP Channels from LFPElectrode
        electrode_keys = (LFPSelection.LFPElectrode & key).fetch("KEY")
        electrode_id_list = list(k["electrode_id"] for k in electrode_keys)
        electrode_id_list.sort()

        lfp_file_name = AnalysisNwbfile().create(key["nwb_file_name"])

        lfp_file_abspath = AnalysisNwbfile().get_abs_path(lfp_file_name)
        lfp_object_id, timestamp_interval = FirFilter().filter_data_nwb(
            lfp_file_abspath,
            rawdata,
            filter_coeff,
            valid_times,
            electrode_id_list,
            decimation,
        )

        # now that the LFP is filtered and in the file, add the file to the AnalysisNwbfile table
        AnalysisNwbfile().add(key["nwb_file_name"], lfp_file_name)

        key["analysis_file_name"] = lfp_file_name
        key["lfp_object_id"] = lfp_object_id
        key["lfp_sampling_rate"] = sampling_rate // decimation

        # finally, we need to censor the valid times to account for the downsampling
        lfp_valid_times = interval_list_censor(valid_times, timestamp_interval)
        # add an interval list for the LFP valid times, skipping duplicates
        key["interval_list_name"] = "lfp valid times"
        IntervalList.insert1(
            {
                "nwb_file_name": key["nwb_file_name"],
                "interval_list_name": key["interval_list_name"],
                "valid_times": lfp_valid_times,
            },
            replace=True,
        )
        self.insert1(key)

    def nwb_object(self, key):
        # return the NWB object in the raw NWB file
        lfp_file_name = (LFP() & {"nwb_file_name": key["nwb_file_name"]}).fetch1(
            "analysis_file_name"
        )
        lfp_file_abspath = AnalysisNwbfile().get_abs_path(lfp_file_name)
        lfp_nwbf = get_nwb_file(lfp_file_abspath)
        # get the object id
        nwb_object_id = (self & {"analysis_file_name": lfp_file_name}).fetch1(
            "lfp_object_id"
        )
        return lfp_nwbf.objects[nwb_object_id]

    def fetch_nwb(self, *attrs, **kwargs):
        return fetch_nwb(
            self, (AnalysisNwbfile, "analysis_file_abs_path"), *attrs, **kwargs
        )

    def fetch1_dataframe(self, *attrs, **kwargs):
        nwb_lfp = self.fetch_nwb()[0]
        return pd.DataFrame(
            nwb_lfp["lfp"].data, index=pd.Index(nwb_lfp["lfp"].timestamps, name="time")
        )




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


@schema
class LFPBandSelection(dj.Manual):
    definition = """
    -> LFP
    -> FirFilter                   # the filter to use for the data
    -> IntervalList.proj(target_interval_list_name='interval_list_name')  # the original set of times to be filtered
    lfp_band_sampling_rate: int    # the sampling rate for this band
    ---
    min_interval_len = 1: float  # the minimum length of a valid interval to filter
    """

    class LFPBandElectrode(dj.Part):
        definition = """
        -> LFPBandSelection
        -> LFPSelection.LFPElectrode  # the LFP electrode to be filtered
        reference_elect_id = -1: int  # the reference electrode to use; -1 for no reference
        ---
        """

    def set_lfp_band_electrodes(
        self,
        nwb_file_name,
        electrode_list,
        filter_name,
        interval_list_name,
        reference_electrode_list,
        lfp_band_sampling_rate,
    ):
        """
        Adds an entry for each electrode in the electrode_list with the specified filter, interval_list, and
        reference electrode.
        Also removes any entries that have the same filter, interval list and reference electrode but are not
        in the electrode_list.
        :param nwb_file_name: string - the name of the nwb file for the desired session
        :param electrode_list: list of LFP electrodes to be filtered
        :param filter_name: the name of the filter (from the FirFilter schema)
        :param interval_name: the name of the interval list (from the IntervalList schema)
        :param reference_electrode_list: A single electrode id corresponding to the reference to use for all
        electrodes or a list with one element per entry in the electrode_list
        :param lfp_band_sampling_rate: The output sampling rate to be used for the filtered data; must be an
        integer divisor of the LFP sampling rate
        :return: none
        """
        # Error checks on parameters
        # electrode_list
        query = LFPSelection().LFPElectrode() & {"nwb_file_name": nwb_file_name}
        available_electrodes = query.fetch("electrode_id")
        if not np.all(np.isin(electrode_list, available_electrodes)):
            raise ValueError(
                "All elements in electrode_list must be valid electrode_ids in the LFPSelection table"
            )
        # sampling rate
        lfp_sampling_rate = (LFP() & {"nwb_file_name": nwb_file_name}).fetch1(
            "lfp_sampling_rate"
        )
        decimation = lfp_sampling_rate // lfp_band_sampling_rate
        if lfp_sampling_rate // decimation != lfp_band_sampling_rate:
            raise ValueError(
                f"lfp_band_sampling rate {lfp_band_sampling_rate} is not an integer divisor of lfp "
                f"samping rate {lfp_sampling_rate}"
            )
        # filter
        query = FirFilter() & {
            "filter_name": filter_name,
            "filter_sampling_rate": lfp_sampling_rate,
        }
        if not query:
            raise ValueError(
                f"filter {filter_name}, sampling rate {lfp_sampling_rate} is not in the FirFilter table"
            )
        # interval_list
        query = IntervalList() & {
            "nwb_file_name": nwb_file_name,
            "interval_name": interval_list_name,
        }
        if not query:
            raise ValueError(
                f"interval list {interval_list_name} is not in the IntervalList table; the list must be "
                "added before this function is called"
            )
        # reference_electrode_list
        if len(reference_electrode_list) != 1 and len(reference_electrode_list) != len(
            electrode_list
        ):
            raise ValueError(
                "reference_electrode_list must contain either 1 or len(electrode_list) elements"
            )
        # add a -1 element to the list to allow for the no reference option
        available_electrodes = np.append(available_electrodes, [-1])
        if not np.all(np.isin(reference_electrode_list, available_electrodes)):
            raise ValueError(
                "All elements in reference_electrode_list must be valid electrode_ids in the LFPSelection "
                "table"
            )

        # make a list of all the references
        ref_list = np.zeros((len(electrode_list),))
        ref_list[:] = reference_electrode_list

        key = dict()
        key["nwb_file_name"] = nwb_file_name
        key["filter_name"] = filter_name
        key["filter_sampling_rate"] = lfp_sampling_rate
        key["target_interval_list_name"] = interval_list_name
        key["lfp_band_sampling_rate"] = lfp_sampling_rate // decimation
        # insert an entry into the main LFPBandSelectionTable
        self.insert1(key, skip_duplicates=True)

        # get all of the current entries and delete any that are not in the list
        elect_id, ref_id = (self.LFPBandElectrode() & key).fetch(
            "electrode_id", "reference_elect_id"
        )
        for e, r in zip(elect_id, ref_id):
            if not len(np.where((electrode_list == e) & (ref_list == r))[0]):
                key["electrode_id"] = e
                key["reference_elect_id"] = r
                (self.LFPBandElectrode() & key).delete()

        # iterate through all of the new elements and add them
        for e, r in zip(electrode_list, ref_list):
            key["electrode_id"] = e
            query = Electrode & {"nwb_file_name": nwb_file_name, "electrode_id": e}
            key["electrode_group_name"] = query.fetch1("electrode_group_name")
            key["reference_elect_id"] = r
            self.LFPBandElectrode().insert1(key, skip_duplicates=True)


@schema
class LFPBand(dj.Computed):
    definition = """
    -> LFPBandSelection
    ---
    -> AnalysisNwbfile
    -> IntervalList
    filtered_data_object_id: varchar(40)  # the NWB object ID for loading this object from the file
    """

    def make(self, key):
        # get the NWB object with the lfp data; FIX: change to fetch with additional infrastructure
        lfp_object = (LFP() & {"nwb_file_name": key["nwb_file_name"]}).fetch_nwb()[0][
            "lfp"
        ]

        # get the electrodes to be filtered and their references
        lfp_band_elect_id, lfp_band_ref_id = (
            LFPBandSelection().LFPBandElectrode() & key
        ).fetch("electrode_id", "reference_elect_id")

        # sort the electrodes to make sure they are in ascending order
        lfp_band_elect_id = np.asarray(lfp_band_elect_id)
        lfp_band_ref_id = np.asarray(lfp_band_ref_id)
        lfp_sort_order = np.argsort(lfp_band_elect_id)
        lfp_band_elect_id = lfp_band_elect_id[lfp_sort_order]
        lfp_band_ref_id = lfp_band_ref_id[lfp_sort_order]

        lfp_sampling_rate = (LFP() & {"nwb_file_name": key["nwb_file_name"]}).fetch1(
            "lfp_sampling_rate"
        )
        interval_list_name, lfp_band_sampling_rate = (LFPBandSelection() & key).fetch1(
            "target_interval_list_name", "lfp_band_sampling_rate"
        )
        valid_times = (
            IntervalList()
            & {
                "nwb_file_name": key["nwb_file_name"],
                "interval_list_name": interval_list_name,
            }
        ).fetch1("valid_times")
        # the valid_times for this interval may be slightly beyond the valid times for the lfp itself,
        # so we have to intersect the two
        lfp_interval_list = (LFP() & {"nwb_file_name": key["nwb_file_name"]}).fetch1(
            "interval_list_name"
        )
        lfp_valid_times = (
            IntervalList()
            & {
                "nwb_file_name": key["nwb_file_name"],
                "interval_list_name": lfp_interval_list,
            }
        ).fetch1("valid_times")
        min_length = (LFPBandSelection & key).fetch1("min_interval_len")
        lfp_band_valid_times = interval_list_intersect(
            valid_times, lfp_valid_times, min_length=min_length
        )

        filter_name, filter_sampling_rate, lfp_band_sampling_rate = (
            LFPBandSelection() & key
        ).fetch1("filter_name", "filter_sampling_rate", "lfp_band_sampling_rate")

        decimation = int(lfp_sampling_rate) // lfp_band_sampling_rate

        # load in the timestamps
        timestamps = np.asarray(lfp_object.timestamps)
        # get the indices of the first timestamp and the last timestamp that are within the valid times
        included_indices = interval_list_contains_ind(lfp_band_valid_times, timestamps)
        # pad the indices by 1 on each side to avoid message in filter_data
        if included_indices[0] > 0:
            included_indices[0] -= 1
        if included_indices[-1] != len(timestamps) - 1:
            included_indices[-1] += 1

        timestamps = timestamps[included_indices[0] : included_indices[-1]]

        # load all the data to speed filtering
        lfp_data = np.asarray(
            lfp_object.data[included_indices[0] : included_indices[-1], :],
            dtype=type(lfp_object.data[0][0]),
        )

        # get the indices of the electrodes to be filtered and the references
        lfp_band_elect_index = get_electrode_indices(lfp_object, lfp_band_elect_id)
        lfp_band_ref_index = get_electrode_indices(lfp_object, lfp_band_ref_id)

        # subtract off the references for the selected channels
        for index, elect_index in enumerate(lfp_band_elect_index):
            if lfp_band_ref_id[index] != -1:
                lfp_data[:, elect_index] = (
                    lfp_data[:, elect_index] - lfp_data[:, lfp_band_ref_index[index]]
                )

        # get the LFP filter that matches the raw data
        filter = (
            FirFilter()
            & {"filter_name": filter_name}
            & {"filter_sampling_rate": filter_sampling_rate}
        ).fetch(as_dict=True)
        if len(filter) == 0:
            raise ValueError(
                f"Filter {filter_name} and sampling_rate {lfp_band_sampling_rate} does not exit in the "
                "FirFilter table"
            )

        filter_coeff = filter[0]["filter_coeff"]
        if len(filter_coeff) == 0:
            print(
                f"Error in LFPBand: no filter found with data sampling rate of {lfp_band_sampling_rate}"
            )
            return None

        # create the analysis nwb file to store the results.
        lfp_band_file_name = AnalysisNwbfile().create(key["nwb_file_name"])
        lfp_band_file_abspath = AnalysisNwbfile().get_abs_path(lfp_band_file_name)
        # filter the data and write to an the nwb file
        filtered_data, new_timestamps = FirFilter().filter_data(
            timestamps,
            lfp_data,
            filter_coeff,
            lfp_band_valid_times,
            lfp_band_elect_index,
            decimation,
        )

        # now that the LFP is filtered, we create an electrical series for it and add it to the file
        with pynwb.NWBHDF5IO(
            path=lfp_band_file_abspath, mode="a", load_namespaces=True
        ) as io:
            nwbf = io.read()
            # get the indices of the electrodes in the electrode table of the file to get the right values
            elect_index = get_electrode_indices(nwbf, lfp_band_elect_id)
            electrode_table_region = nwbf.create_electrode_table_region(
                elect_index, "filtered electrode table"
            )
            eseries_name = "filtered data"
            # TODO: use datatype of data
            es = pynwb.ecephys.ElectricalSeries(
                name=eseries_name,
                data=filtered_data,
                electrodes=electrode_table_region,
                timestamps=new_timestamps,
            )
            # Add the electrical series to the scratch area
            nwbf.add_scratch(es)
            io.write(nwbf)
            filtered_data_object_id = es.object_id
        #
        # add the file to the AnalysisNwbfile table
        AnalysisNwbfile().add(key["nwb_file_name"], lfp_band_file_name)
        key["analysis_file_name"] = lfp_band_file_name
        key["filtered_data_object_id"] = filtered_data_object_id

        # finally, we need to censor the valid times to account for the downsampling if this is the first time we've
        # downsampled these data
        key["interval_list_name"] = (
            interval_list_name + " lfp band " + str(lfp_band_sampling_rate) + "Hz"
        )
        tmp_valid_times = (
            IntervalList
            & {
                "nwb_file_name": key["nwb_file_name"],
                "interval_list_name": key["interval_list_name"],
            }
        ).fetch("valid_times")
        if len(tmp_valid_times) == 0:
            lfp_band_valid_times = interval_list_censor(
                lfp_band_valid_times, new_timestamps
            )
            # add an interval list for the LFP valid times
            IntervalList.insert1(
                {
                    "nwb_file_name": key["nwb_file_name"],
                    "interval_list_name": key["interval_list_name"],
                    "valid_times": lfp_band_valid_times,
                }
            )
        else:
            # check that the valid times are the same
            assert np.isclose(
                tmp_valid_times[0], lfp_band_valid_times
            ).all(), "previously saved lfp band times do not match current times"

        self.insert1(key)

    def fetch_nwb(self, *attrs, **kwargs):
        return fetch_nwb(
            self, (AnalysisNwbfile, "analysis_file_abs_path"), *attrs, **kwargs
        )

    def fetch1_dataframe(self, *attrs, **kwargs):
        filtered_nwb = self.fetch_nwb()[0]
        return pd.DataFrame(
            filtered_nwb["filtered_data"].data,
            index=pd.Index(filtered_nwb["filtered_data"].timestamps, name="time"),
        )

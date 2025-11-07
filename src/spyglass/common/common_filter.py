# code to define filters that can be applied to continuous time data
import warnings
from typing import Union

import datajoint as dj
import matplotlib.pyplot as plt
import numpy as np
import psutil
import pynwb
import scipy.signal as signal

from spyglass.utils import SpyglassMixin, logger
from spyglass.utils.nwb_helper_fn import get_electrode_indices

schema = dj.schema("common_filter")


def _import_ghostipy():
    try:
        import ghostipy as gsp

        return gsp
    except (ImportError, ModuleNotFoundError) as e:
        raise ImportError(
            "You must install ghostipy to use filtering methods. Please note "
            "that to install ghostipy on an Mac M1, you must first install "
            "pyfftw from conda-forge."
        ) from e


@schema
class FirFilterParameters(SpyglassMixin, dj.Manual):
    """Filter parameters for filtering continuous time data.

    Attributes
    ----------
    filter_name: str
        The name of the filter.
    filter_sampling_rate: int
        The sampling rate of the filter.
    filter_type: str
        The type of filter ('lowpass', 'highpass', or 'bandpass').
    filter_low_stop: float
        The lowest frequency for the stop band for low filter.
    filter_low_pass: float
        The lowest frequency for the pass band of low filter.
    filter_high_pass: float
        The highest frequency for the pass band of high filter.
    filter_high_stop: float
        The highest frequency for the stop band of high filter.
    filter_comments: str
        Comments about the filter.
    filter_band_edges: np.ndarray
        A numpy array of filter coefficients.
    """

    definition = """
    filter_name: varchar(80)           # descriptive name of this filter
    filter_sampling_rate: int          # sampling rate for this filter
    ---
    filter_type: enum("lowpass", "highpass", "bandpass")
    filter_low_stop = 0: float     # lowest freq for stop band for low filt
    filter_low_pass = 0: float     # lowest freq for pass band of low filt
    filter_high_pass = 0: float    # highest freq for pass band for high filt
    filter_high_stop = 0: float    # highest freq for stop band of high filt
    filter_comments: varchar(2000) # comments about the filter
    filter_band_edges: blob        # numpy array of filter bands
                                   # redundant with individual parameters
    filter_coeff: longblob         # numpy array of filter coefficients
    """

    def add_filter(
        self,
        filter_name: str,
        fs: float,
        filter_type: str,
        band_edges: list,
        comments: str = "",
    ) -> None:
        """Add filter to the Filter table.

        Parameters
        ----------
        filter_name: str
            The name of the filter.
        fs: float
            The filter sampling rate.
        filter_type: str
            The type of the filter ('lowpass', 'highpass', or 'bandpass').
        band_edges: List[float]
            The band edges for the filter.
        comments: str, optional)
            Additional comments for the filter. Default "".

        Returns
        -------
        None
            Returns None if there is an error in the filter type or band
            frequencies.

        Raises
        ------
        Exception:
            Raises an exception if an unexpected filter type is encountered.
        """
        VALID_FILTERS = {"lowpass": 2, "highpass": 2, "bandpass": 4}
        FILTER_ERR = "Error in Filter.add_filter: "
        FILTER_N_ERR = FILTER_ERR + "filter {} requires {} band_frequencies."

        # add an FIR bandpass filter of the specified type.
        # band_edges should be as follows:
        #   low pass : [high_pass high_stop]
        #   high pass: [low stop low pass]
        #   band pass: [low_stop low_pass high_pass high_stop].
        if filter_type not in VALID_FILTERS:
            logger.error(
                FILTER_ERR
                + f"{filter_type} not valid type: {VALID_FILTERS.keys()}"
            )
            return None

        if not len(band_edges) == VALID_FILTERS[filter_type]:
            logger.error(
                FILTER_N_ERR.format(filter_name, VALID_FILTERS[filter_type])
            )
            return None

        gsp = _import_ghostipy()
        TRANS_SPLINE = 2  # transition spline will be quadratic

        if filter_type != "bandpass":
            transition_width = band_edges[1] - band_edges[0]

        else:
            # transition width is mean of left and right transition regions
            transition_width = (
                (band_edges[1] - band_edges[0])
                + (band_edges[3] - band_edges[2])
            ) / 2.0

        numtaps = gsp.estimate_taps(fs, transition_width)
        filterdict = {
            "filter_type": filter_type,
            "filter_name": filter_name,
            "filter_sampling_rate": fs,
            "filter_comments": comments,
            "filter_low_stop": 0,
            "filter_low_pass": 0,
            "filter_high_pass": 0,
            "filter_high_stop": 0,
            "filter_band_edges": np.asarray(band_edges),
        }

        # set the desired frequency response
        if filter_type == "lowpass":
            desired = [1, 0]
            pass_stop_dict = {
                "filter_high_pass": band_edges[0],
                "filter_high_stop": band_edges[1],
            }
        elif filter_type == "highpass":
            desired = [0, 1]
            pass_stop_dict = {
                "filter_low_stop": band_edges[0],
                "filter_low_pass": band_edges[1],
            }
        else:
            desired = [0, 1, 1, 0]
            pass_stop_dict = {
                "filter_low_stop": band_edges[0],
                "filter_low_pass": band_edges[1],
                "filter_high_pass": band_edges[2],
                "filter_high_stop": band_edges[3],
            }

        # create 1d array for coefficients
        filterdict.update(
            {
                **pass_stop_dict,
                "filter_coeff": np.array(
                    gsp.firdesign(
                        numtaps, band_edges, desired, fs=fs, p=TRANS_SPLINE
                    ),
                    ndmin=1,
                ),
            }
        )

        self.insert1(filterdict, skip_duplicates=True)

    def _filter_restrict(self, filter_name, fs):
        return (
            self & {"filter_name": filter_name} & {"filter_sampling_rate": fs}
        ).fetch1()

    def plot_magnitude(self, filter_name, fs, return_fig=False):
        """Plot the magnitude of the frequency response of the filter."""
        filter_dict = self._filter_restrict(filter_name, fs)
        plt.figure()
        w, h = signal.freqz(filter_dict["filter_coeff"], worN=65536)
        magnitude = 20 * np.log10(np.abs(h))
        plt.plot(w / np.pi * fs / 2, magnitude)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.title("Frequency Response")
        plt.xlim(0, np.max(filter_dict["filter_band_edges"] * 2))
        plt.ylim(np.min(magnitude), -1 * np.min(magnitude) * 0.1)
        plt.grid(True)
        if return_fig:
            return plt.gcf()

    def plot_fir_filter(self, filter_name, fs, return_fig=False):
        """Plot the filter."""
        filter_dict = self._filter_restrict(filter_name, fs)
        plt.figure()
        plt.clf()
        plt.plot(filter_dict["filter_coeff"], "k")
        plt.xlabel("Coefficient")
        plt.ylabel("Magnitude")
        plt.title("Filter Taps")
        plt.grid(True)
        if return_fig:
            return plt.gcf()

    def filter_delay(self, filter_name, fs):
        """Return the filter delay for the specified filter."""
        return self.calc_filter_delay(
            self._filter_restrict(filter_name, fs)["filter_coeff"]
        )

    def _time_bound_check(self, start, stop, all, nsamples):
        timestamp_warn = "Interval time warning: "
        if start < all[0]:
            warnings.warn(
                timestamp_warn
                + "start time smaller than first timestamp, "
                + f"substituting first: {start} < {all[0]}"
            )
            start = all[0]

        if stop > all[-1]:
            logger.warning(
                timestamp_warn
                + "stop time larger than last timestamp, "
                + f"substituting last: {stop} < {all[-1]}"
            )
            stop = all[-1]

        frm, to = np.searchsorted(all, (start, stop))
        to = min(to, nsamples)
        return frm, to

    def filter_data_nwb(
        self,
        analysis_file_abs_path: str,
        eseries: pynwb.ecephys.ElectricalSeries,
        filter_coeff: np.ndarray,
        valid_times: np.ndarray,
        electrode_ids: list,
        decimation: int,
        description: str = "filtered data",
        data_type: Union[None, str] = None,
    ):
        """
        Filter data from an NWB electrical series using the ghostipy package,
        and save the result as a new electrical series in the analysis NWB file.

        Parameters
        ----------
        analysis_file_abs_path : str
            Full path to the analysis NWB file.
        eseries : pynwb.ecephys.ElectricalSeries
            Electrical series with data to be filtered.
        filter_coeff : np.ndarray
            Array with filter coefficients for FIR filter.
        valid_times : np.ndarray
            Array with start and stop times of intervals to be filtered.
        electrode_ids : list
            List of electrode IDs to filter.
        decimation : int
            Decimation factor.
        description : str
            Description of the filtered data.
        data_type : Union[None, str]
            Type of data (e.g., "LFP").

        Returns
        -------
        tuple
            The NWB object ID of the filtered data and a list containing the
            first and last timestamps.
        """
        # Note: type -> data_type to avoid conflict with builtin type
        # All existing refs to this func use positional args, so no need to
        # adjust elsewhere, but low probability of issues with custom scripts

        MEM_USE_LIMIT = 0.9  # % of RAM use permitted

        gsp = _import_ghostipy()

        data_on_disk = eseries.data
        timestamps_on_disk = eseries.timestamps

        n_samples = len(timestamps_on_disk)
        time_axis = 0 if data_on_disk.shape[0] == n_samples else 1
        electrode_axis = 1 - time_axis

        n_electrodes = data_on_disk.shape[electrode_axis]
        input_dim_restrictions = [None] * len(data_on_disk.shape)

        # Get input dimension restrictions
        input_dim_restrictions[electrode_axis] = np.s_[
            get_electrode_indices(eseries, electrode_ids)
        ]

        indices = []
        output_shape_list = [0] * len(data_on_disk.shape)
        output_shape_list[electrode_axis] = len(electrode_ids)
        data_dtype = data_on_disk[0][0].dtype

        filter_delay = self.calc_filter_delay(filter_coeff)

        output_offsets = [0]

        for a_start, a_stop in valid_times:
            frm, to = self._time_bound_check(
                a_start, a_stop, timestamps_on_disk, n_samples
            )

            indices.append((frm, to))

            shape, _ = gsp.filter_data_fir(
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

        # Create dynamic table region and electrode series, write/close file
        with pynwb.NWBHDF5IO(
            path=analysis_file_abs_path, mode="a", load_namespaces=True
        ) as io:
            nwbf = io.read()

            # get the indices of the electrodes in the electrode table
            elect_ind = get_electrode_indices(nwbf, electrode_ids)

            electrode_table_region = nwbf.create_electrode_table_region(
                elect_ind, "filtered electrode table"
            )
            es = pynwb.ecephys.ElectricalSeries(
                name="filtered data",
                data=np.empty(tuple(output_shape_list), dtype=data_dtype),
                electrodes=electrode_table_region,
                timestamps=np.empty(output_shape_list[time_axis]),
                description=description,
            )
            if data_type == "LFP":
                ecephys_module = nwbf.create_processing_module(
                    name="ecephys", description=description
                )
                ecephys_module.add(pynwb.ecephys.LFP(electrical_series=es))
            else:
                nwbf.add_scratch(es)

            io.write(nwbf)

        # Reload NWB file to get h5py objects for data/timestamps
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

            logger.info("Filtering data")
            for ii, (start, stop) in enumerate(indices):
                # Calc size of timestamps + data, check if < 90% of RAM
                interval_samples = stop - start
                req_mem = interval_samples * (
                    timestamps_on_disk[0].itemsize
                    + n_electrodes * data_on_disk[0][0].itemsize
                )
                if req_mem < MEM_USE_LIMIT * psutil.virtual_memory().available:
                    logger.info(f"Interval {ii}: loading data into memory")
                    timestamps = np.asarray(
                        timestamps_on_disk[start:stop],
                        dtype=timestamps_on_disk[0].dtype,
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
                    input_index_bounds = [0, interval_samples - 1]

                else:
                    logger.info(f"Interval {ii}: leaving data on disk")
                    data = data_on_disk
                    timestamps = timestamps_on_disk
                    extracted_ts = timestamps[start:stop:decimation]
                    new_timestamps[
                        ts_offset : ts_offset + len(extracted_ts)
                    ] = extracted_ts
                    ts_offset += len(extracted_ts)
                    input_index_bounds = [start, stop]

                # filter the data
                gsp.filter_data_fir(
                    data,
                    filter_coeff,
                    axis=time_axis,
                    input_index_bounds=input_index_bounds,
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
        self,
        timestamps,
        data,
        filter_coeff,
        valid_times,
        electrodes,
        decimation,
    ):
        """
        Parameters
        ----------
        timestamps: numpy array
            List of timestamps for data
        data:
            original data array
        filter_coeff: numpy array
            Filter coefficients for FIR filter
        valid_times: 2D numpy array
            Start and stop times of intervals to be filtered
        electrodes: list
            Electrodes to filter
        decimation:
            decimation factor

        Return
        ------
        filtered_data, timestamps
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
            frm, to = self._time_bound_check(
                a_start, a_stop, timestamps, n_samples
            )
            if np.isclose(frm, to, rtol=0, atol=1e-8):
                continue
            indices.append((frm, to))

            shape, _ = gsp.filter_data_fir(
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

        new_timestamps = np.empty(
            (output_shape_list[time_axis],), timestamps.dtype
        )

        indices = np.array(indices, ndmin=2)

        # Filter  the output dataset
        ts_offset = 0

        for ii, (start, stop) in enumerate(indices):
            extracted_ts = timestamps[start:stop:decimation]

            new_timestamps[ts_offset : ts_offset + len(extracted_ts)] = (
                extracted_ts
            )
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
        Parameters
        ----------
        filter_coeff: numpy array

        Return
        ------
        filter delay: int
        """
        return (len(filter_coeff) - 1) // 2

    def create_standard_filters(self):
        """Add standard filters to the Filter table

        Includes 0-400 Hz low pass for continuous raw data -> LFP
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
            "standard LFP filter for 30 KHz data",
        )

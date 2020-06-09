#code to define filters that can be applied to continuous time data
import datajoint as dj
import pynwb
import scipy.signal as signal
import numpy as np
import ghostipy as gsp
import matplotlib.pyplot as plt
import uuid
import h5py as h5
from hdmf.common.table import DynamicTableRegion

schema = dj.schema('common_filter')


@schema
class FirFilter(dj.Manual):
    definition = """                                                                             
    filter_name: varchar(80)    # descriptive name of this filter
    filter_sampling_rate: int       # sampling rate for this filter
    ---
    filter_type: enum('lowpass', 'highpass', 'bandpass')
    filter_low_stop=0: float         # lowest frequency for stop band for low frequency side of filter
    filter_low_pass=0: float         # lowest frequency for pass band of low frequency side of filter
    filter_high_pass=0: float         # highest frequency for pass band for high frequency side of filter
    filter_high_stop=0: float         # highest frequency for stop band of high frequency side of filter
    filter_comments: varchar(255)   # comments about the filter
    filter_band_edges: blob         # numpy array containing the filter bands (redundant with individual parameters)
    filter_coeff: blob               # numpy array containing the filter coefficients 
    """
    def __init__(self, *args):
        # do your custom stuff here
        super().__init__(*args)  # call the base implementation

    def zpk(self):
        # return the zeros, poles, and gain for the filter
        return signal.tf2zpk(self.filter_coeff, 1)

    def add_filter(self, filter_name, fs, filter_type, band_edges, comments=''):
        # add an FIR bandpass filter of the specified type ('lowpass', 'highpass', or 'bandpass').
        # band_edges should be as follows:
        #   low pass filter: [high_pass high_stop]
        #   high pass filter: [low stop low pass]
        #   band pass filter: [low_stop low_pass high_pass high_stop].
        if filter_type not in ['lowpass', 'highpass', 'bandpass']:
            print('Error in Filter.add_filter: filter type {} is not ''lowpass'', ''highpass'' or '
                  '''bandpass'''.format(filter_type))
            return None

        p = 2  # transition spline will be quadratic
        if filter_type == 'lowpass' or filter_type == 'highpass':
            # check that two frequencies were passed in and that they are in the right order
            if len(band_edges) != 2:
                print('Error in Filter.add_filter: lowpass and highpass filter requires two band_frequencies')
                return None
            tw = band_edges[1] - band_edges[0]

        elif filter_type == 'bandpass':
            if len(band_edges) != 4:
                print('Error in Filter.add_filter: bandpass filter requires four band_frequencies.')
                return None
            # the transition width is the mean of the widths of left and right transition regions
            tw = ((band_edges[1] - band_edges[0]) + (band_edges[3] - band_edges[2])) / 2.0

        numtaps = gsp.estimate_taps(fs, tw)
        filterdict = dict()
        filterdict['filter_name'] = filter_name
        filterdict['filter_sampling_rate'] = fs
        filterdict['filter_comments'] = comments

        # set the desired frequency response
        if filter_type == 'lowpass':
            desired = [1, 0]
            filterdict['filter_low_stop'] =  0
            filterdict['filter_low_pass'] =  0
            filterdict['filter_high_pass'] = band_edges[0]
            filterdict['filter_high_stop'] = band_edges[1]
        elif filter_type == 'highpass':
            desired = [0, 1]
            filterdict['filter_low_stop'] = band_edges[0]
            filterdict['filter_low_pass'] = band_edges[1]
            filterdict['filter_high_pass'] = 0
            filterdict['filter_high_stop'] = 0
        else:
            desired = [0, 1, 1, 0]
            filterdict['filter_low_stop'] = band_edges[0]
            filterdict['filter_low_pass'] = band_edges[1]
            filterdict['filter_high_pass'] = band_edges[2]
            filterdict['filter_high_stop'] = band_edges[3]

        filterdict['filter_band_edges'] = np.asarray(band_edges)
        # create 1d array for coefficients
        filterdict['filter_coeff'] = np.array(gsp.firdesign(numtaps, band_edges, desired, fs=fs, p=p), ndmin=1)
        # add this filter to the table
        self.insert1(filterdict, replace="True")

    def plot_magnitude(self, filter_name, fs):
        filter = (self & {'filter_name': filter_name} & {'filter_sampling_rate' : fs}).fetch(as_dict=True)
        f = filter[0]
        plt.figure()
        w, h = signal.freqz(filter[0]['filter_coeff'], worN=65536)
        magnitude = 20 * np.log10(np.abs(h))
        plt.plot(w / np.pi * fs / 2, magnitude)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.title('Frequency Response')
        plt.xlim(0, np.max(f['filter_coeffand_edges'] * 2))
        plt.ylim(np.min(magnitude), -1 * np.min(magnitude) * .1)
        plt.grid(True)

    def plot_fir_filter(self, filter_name, fs):
        filter = (self & {'filter_name': filter_name} & {'filter_sampling_rate' : fs}).fetch(as_dict=True)
        f = filter[0]
        plt.figure()
        plt.clf()
        b = plt.plot(f['filter_coeff'], 'k' )
        plt.xlabel('Coefficient')
        plt.ylabel('Magnitude')
        plt.title('Filter Taps')
        plt.grid(True)

    def filter_delay(self, filter_name, fs):
        # return the filter delay
        filter = (self & {'filter_name': filter_name} & {'filter_sampling_rate': fs}).fetch(as_dict=True)
        f = filter[0]
        return calc_filter_delay(filter['filter_coeff'])

    def filter_data_nwb(self, nwb_file_name, timestamps, data, filter_coeff, valid_times, electrodes,
                        decimation):
        """
        :param nwb_file_name: str   name of previously created nwb file where filtered data should be stored
        :param timestamps: numpy array with list of timestamps for data
        :param data: original data array
        :param filter_coeff: numpy array with filter coefficients for FIR filter
        :param valid_times: 2D numpy array with start and stop times of intervals to be filtered
        :param electrodes: list of electrodes to filter
        :param decimation: decimation factor
        :return: The NWB object id of the filtered data

        This function takes data and timestamps from an NWB electrical series and filters them using the ghostipy
        package, saving the result as a new electricalseries in the nwb_file_name, which should have previously been
        created and linked to the original NWB file using common_session.LinkedNwbfile.create()
        """

        n_dim = len(data.shape)
        n_samples = len(timestamps)
        # find the
        time_axis = 0 if data.shape[0] == n_samples else 1
        electrode_axis = 1 - time_axis
        input_dim_restrictions = [None] * n_dim
        input_dim_restrictions[electrode_axis] = np.s_[electrodes]

        indices = []
        output_shape_list = [0] * n_dim
        output_shape_list[electrode_axis] = len(electrodes)
        output_offsets = [0]

        filter_delay = calc_filter_delay(filter_coeff)
        for a_start, a_stop in valid_times:
            frm, to = np.searchsorted(timestamps, (a_start, a_stop))
            if to > n_samples:
                to = n_samples

            indices.append((frm, to))
            shape, dtype = gsp.filter_data_fir(data,
                                               filter_coeff,
                                               axis=time_axis,
                                               input_index_bounds=[frm, to],
                                               output_index_bounds=[filter_delay, filter_delay + to - frm],
                                               describe_dims=True,
                                               ds=decimation,
                                               input_dim_restrictions=input_dim_restrictions)
            print(f'dtype = {dtype}, {data}')
            output_offsets.append(output_offsets[-1] + shape[time_axis])
            output_shape_list[time_axis] += shape[time_axis]

        # open the nwb file to create the dynamic table region and electrode series, then write and close the file
        with pynwb.NWBHDF5IO(nwb_file_name, "a") as io:
            nwbf=io.read()

        electrode_table_region = nwbf.create_electrode_table_region(electrodes, 'filtered electrode table')
        # FIX: name needs to be unique
        eseries_name = 'filt_test'
        es = pynwb.ecephys.ElectricalSeries(name='filt test', data=np.empty(tuple(output_shape_list), dtype='i2'),
                                            electrodes=electrode_table_region, timestamps= np.empty(0))
        # check to see if there is already an ephys processing module, and if not, add one
        #if len(nwbf.processing) == 0 or 'lfp' not in nwbf.processing:
        #    nwbf.create_processing_module('lfp', 'filtered data')
        #nwbf.processing['lfp'].add(es)
        nwbf.add_acquisition(es)
        with pynwb.NWBHDF5IO(nwb_file_name, "a") as io:
            io.write(nwbf)

        # reload the NWB file to get the h5py objects for the data and the timestamps
        with pynwb.NWBHDF5IO(nwb_file_name, "a") as io:
            nwbf = io.read()

        es = nwbf.processing['ephys'].get_container(eseries_name)
        filtered_data = es.data
        new_timestamps = es.timestamps

        indices = np.array(indices, ndmin=2)

        # Filter and write the output dataset
        ts_offset = 0

        for ii, (start, stop) in enumerate(indices):
            extracted_ts = timestamps[start:stop:decimation]

            print(f"Diffs {np.diff(extracted_ts)}")
            new_timestamps[ts_offset:ts_offset + len(extracted_ts)] = extracted_ts
            ts_offset += len(extracted_ts)

            # finally ready to filter data!
            gsp.filter_data_fir(data,
                                filter_coeff,
                                axis=time_axis,
                                input_index_bounds=[start, stop],
                                output_index_bounds=[filter_delay, filter_delay + stop - start],
                                ds=ds,
                                input_dim_restrictions=input_dim_restrictions,
                                outarray=filtered_data,
                                output_offset=output_offsets[ii])

        with pynwb.NWBHDF5IO(nwb_file_name, "a") as io:
            io.write(nwbf)

        # Get the object ID for the filtered data and add an entry to the
        #return nwbf.

    def filter_data_hdf5(self, file_path, timestamps, data, filter_coeff, valid_times, electrodes,
                        decimation):
        """
        :param file_path: str   location for saving the timestamps and filtered data
        :param timestamps: numpy array with list of timestamps for data
        :param data: original data array
        :param filter_coeff: numpy array with filter coefficients for FIR filter
        :param valid_times: 2D numpy array with start and stop times of intervals to be filtered
        :param electrodes: list of electrodes to filter
        :param decimation: decimation factor
        :return: hdf5_file_name - the file name for the hdf5 file with the filtered data and timestamps in it

        This function takes data and timestamps from an NWB electrical series and filters them using the ghostipy
        package, saving the result as a new electricalseries in the nwb_file_name, which should have previously been
        created and linked to the original NWB file using common_session.LinkedNwbfile.create()
        """

        n_dim = len(data.shape)
        n_samples = len(timestamps)
        # find the
        time_axis = 0 if data.shape[0] == n_samples else 1
        electrode_axis = 1 - time_axis
        input_dim_restrictions = [None] * n_dim
        input_dim_restrictions[electrode_axis] = np.s_[electrodes]

        indices = []
        output_shape_list = [0] * n_dim
        output_shape_list[electrode_axis] = len(electrodes)
        output_offsets = [0]

        filter_delay = calc_filter_delay(filter_coeff)
        for a_start, a_stop in valid_times:
            frm, to = np.searchsorted(timestamps, (a_start, a_stop))
            if to > n_samples:
                to = n_samples

            indices.append((frm, to))
            shape, dtype = gsp.filter_data_fir(data,
                                               filter_coeff,
                                               axis=time_axis,
                                               input_index_bounds=[frm, to],
                                               output_index_bounds=[filter_delay, filter_delay + to - frm],
                                               describe_dims=True,
                                               ds=decimation,
                                               input_dim_restrictions=input_dim_restrictions)
            output_offsets.append(output_offsets[-1] + shape[time_axis])
            output_shape_list[time_axis] += shape[time_axis]

        # create a unique filename for these data
        new_filepath = file_path +  uuid.uuid4().hex + '.hdf5'
        with h5.File(new_filepath, 'w') as outfile:

            filtered_data = outfile.create_dataset('filtered_data',
                                          shape=tuple(output_shape_list),
                                          dtype=data.dtype)
            filtered_data.attrs['electrode_ids'] = electrodes
            new_timestamps = outfile.create_dataset('timestamps',
                                      shape=(output_shape_list[time_axis], ),
                                      dtype=timestamps.dtype)

            indices = np.array(indices, ndmin=2)

            # Filter and write the output dataset
            ts_offset = 0

            for ii, (start, stop) in enumerate(indices):
                extracted_ts = timestamps[start:stop:decimation]

                #print(f"Diffs {np.diff(extracted_ts)}")
                new_timestamps[ts_offset:ts_offset + len(extracted_ts)] = extracted_ts
                ts_offset += len(extracted_ts)

                # filter data
                gsp.filter_data_fir(data,
                                    filter_coeff,
                                    axis=time_axis,
                                    input_index_bounds=[start, stop],
                                    output_index_bounds=[filter_delay, filter_delay + stop - start],
                                    ds=decimation,
                                    input_dim_restrictions=input_dim_restrictions,
                                    outarray=filtered_data,
                                    output_offset=output_offsets[ii])
            return new_filepath



    def filter_data(self, timestamps, data, filter_coeff, valid_times, electrodes,
                        decimation):
        """
        :param timestamps: numpy array with list of timestamps for data
        :param data: original data array
        :param filter_coeff: numpy array with filter coefficients for FIR filter
        :param valid_times: 2D numpy array with start and stop times of intervals to be filtered
        :param electrodes: list of electrodes to filter
        :param decimation: decimation factor
        :return: filtered_data, timestamps
        """

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

        filter_delay = calc_filter_delay(filter_coeff)
        for a_start, a_stop in valid_times:
            frm, to = np.searchsorted(timestamps, (a_start, a_stop))
            if to > n_samples:
                to = n_samples

            indices.append((frm, to))
            shape, dtype = gsp.filter_data_fir(data,
                                               filter_coeff,
                                               axis=time_axis,
                                               input_index_bounds=[frm, to],
                                               output_index_bounds=[filter_delay, filter_delay + to - frm],
                                               describe_dims=True,
                                               ds=decimation,
                                               input_dim_restrictions=input_dim_restrictions)
            output_offsets.append(output_offsets[-1] + shape[time_axis])
            output_shape_list[time_axis] += shape[time_axis]

        # create the dataset and the timestamps array
        filtered_data = np.empty(tuple(output_shape_list), dtype=dtype)
        new_timestamps = np.empty((output_shape_list[time_axis],), timestamps.dtype)

        indices = np.array(indices, ndmin=2)

        # Filter  the output dataset
        ts_offset = 0

        for ii, (start, stop) in enumerate(indices):
            extracted_ts = timestamps[start:stop:decimation]

            print(f"Diffs {np.diff(extracted_ts)}")
            new_timestamps[ts_offset:ts_offset + len(extracted_ts)] = extracted_ts
            ts_offset += len(extracted_ts)

            # finally ready to filter data!
            gsp.filter_data_fir(data,
                                filter_coeff,
                                axis=time_axis,
                                input_index_bounds=[start, stop],
                                output_index_bounds=[filter_delay, filter_delay + stop - start],
                                ds=ds,
                                input_dim_restrictions=input_dim_restrictions,
                                outarray=filtered_data,
                                output_offset=output_offsets[ii])

        return filtered_data, new_timestamps

def calc_filter_delay(filter_coeff):
    """
    :param filter_coeff:
    :return: filter delay
    """
    return (len(filter_coeff) - 1) // 2



def create_standard_filters():
    """ Add standard filters to the Filter table including
    0-400 Hz low pass for continuous raw data -> LFP
    """
    FirFilter().add_filter('LFP 0-400 Hz', 20000, 'lowpass', [400, 425], 'standard LFP filter for 20 KHz data')
    FirFilter().add_filter('LFP 0-400 Hz', 30000, 'lowpass', [400, 425], 'standard LFP filter for 20 KHz data')
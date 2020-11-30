#code to define filters that can be applied to continuous time data
import datajoint as dj
import pynwb
import scipy.signal as signal
import numpy as np
import ghostipy as gsp
import matplotlib.pyplot as plt
import uuid
import h5py as h5
from .nwb_helper_fn import get_electrode_indices
from .common_nwbfile import AnalysisNwbfile

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
        
        else:
            raise Exception(f'Unexpected filter type: {filter_type}')

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
        self.insert1(filterdict, skip_duplicates="True")

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
        return self.calc_filter_delay(filter['filter_coeff'])

    def filter_data_nwb(self, analysis_file_abs_path, eseries, filter_coeff, valid_times, electrode_ids,
                        decimation):
        """
        :param analysis_nwb_file_name: str   full path to previously created analysis nwb file where filtered data should be stored. This also has the name of the original NWB file where the data will be taken from
        :param eseries: electrical series with data to be filtered
        :param filter_coeff: numpy array with filter coefficients for FIR filter
        :param valid_times: 2D numpy array with start and stop times of intervals to be filtered
        :param electrode_ids: list of electrode_ids to filter
        :param decimation: int decimation factor 
        :return: The NWB object id of the filtered data

        This function takes data and timestamps from an NWB electrical series and filters them using the ghostipy
        package, saving the result as a new electricalseries in the nwb_file_name, which should have previously been
        created and linked to the original NWB file using common_session.AnalysisNwbfile.create()
        """
        data = eseries.data
        timestamps = eseries.timestamps
        n_dim = len(data.shape)
        n_samples = len(timestamps)
        # find the
        time_axis = 0 if data.shape[0] == n_samples else 1
        electrode_axis = 1 - time_axis
        input_dim_restrictions = [None] * n_dim

        # to get the input dimension restrictions we need to look at the electrode table for the eseries and get the indices from that
        input_dim_restrictions[electrode_axis] = np.s_[get_electrode_indices(eseries, electrode_ids)]

        indices = []
        output_shape_list = [0] * n_dim
        output_shape_list[electrode_axis] = len(electrode_ids)
        output_offsets = [0]

        output_dtype = type(data[0][0])

        filter_delay = self.calc_filter_delay(filter_coeff)
        for a_start, a_stop in valid_times:
            if a_start < timestamps[0]:
                raise ValueError('Interval start time %f is smaller than first timestamp %f' % (a_start, timestamps[0]))
            if a_stop > timestamps[-1]:
                raise ValueError('Interval stop time %f is larger than last timestamp %f' % (a_stop, timestamps[-1]))
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
            #print(f'dtype = {dtype}, {data}')
            output_offsets.append(output_offsets[-1] + shape[time_axis])
            #TODO: remove int() when fixed:
            output_shape_list[time_axis] += shape[time_axis]

        # open the nwb file to create the dynamic table region and electrode series, then write and close the file
        with pynwb.NWBHDF5IO(path=analysis_file_abs_path, mode="a") as io:
            nwbf=io.read()
            # get the indices of the electrodes in the electrode table
            elect_ind = get_electrode_indices(nwbf, electrode_ids)

            electrode_table_region = nwbf.create_electrode_table_region(elect_ind, 'filtered electrode table')
            eseries_name = 'filtered data'
            # TODO: use datatype of data
            es = pynwb.ecephys.ElectricalSeries(name=eseries_name, 
                                                data=np.empty(tuple(output_shape_list), 
                                                dtype=output_dtype),
                                                electrodes=electrode_table_region, 
                                                timestamps= np.empty(output_shape_list[time_axis]))
            # Add the electrical series to the scratch area
            nwbf.add_scratch(es)
            io.write(nwbf)

        # reload the NWB file to get the h5py objects for the data and the timestamps
        with pynwb.NWBHDF5IO(path=analysis_file_abs_path, mode="a") as io:
            nwbf = io.read()
            es = nwbf.objects[es.object_id]
            filtered_data = es.data
            new_timestamps = es.timestamps

            indices = np.array(indices, ndmin=2)
            # Filter and write the output dataset
            ts_offset = 0

            for ii, (start, stop) in enumerate(indices):
                extracted_ts = timestamps[start:stop:decimation]
                new_timestamps[ts_offset:ts_offset + len(extracted_ts)] = extracted_ts
                ts_offset += len(extracted_ts)
                # filter the data
                gsp.filter_data_fir(data,
                                    filter_coeff,
                                    axis=time_axis,
                                    input_index_bounds=[start, stop],
                                    output_index_bounds=[filter_delay, filter_delay + stop - start],
                                    ds=decimation,
                                    input_dim_restrictions=input_dim_restrictions,
                                    outarray=filtered_data,
                                    output_offset=output_offsets[ii])
            io.write(nwbf)
        # TODO: add the Analysis file to kachery
        #AnalysisNwbfile().add_to_kachery(analysis_file_abs_path)
        # return the object ID for the filtered data
        return es.object_id

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

        filter_delay = self.calc_filter_delay(filter_coeff)
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
        filtered_data = np.empty(tuple(output_shape_list), dtype=data.dtype)

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
                                ds=decimation,
                                input_dim_restrictions=input_dim_restrictions,
                                outarray=filtered_data,
                                output_offset=output_offsets[ii])

        return filtered_data, new_timestamps

    def calc_filter_delay(self, filter_coeff):
        """
        :param filter_coeff:
        :return: filter delay
        """
        return (len(filter_coeff) - 1) // 2


    def create_standard_filters(self):
        """ Add standard filters to the Filter table including
        0-400 Hz low pass for continuous raw data -> LFP
        """
        self.add_filter('LFP 0-400 Hz', 20000, 'lowpass', [400, 425], 'standard LFP filter for 20 KHz data')
        self.add_filter('LFP 0-400 Hz', 30000, 'lowpass', [400, 425], 'standard LFP filter for 20 KHz data')
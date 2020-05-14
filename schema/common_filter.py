#code to define filters that can be applied to continuous time data
import datajoint as dj
import pynwb
import scipy.signal as signal
import numpy as np
import ghostipy as gsp
import matplotlib.pyplot as plt
from hdmf.common.table import DynamicTableRegion

schema = dj.schema('common_filter')


@schema
class Filter(dj.Manual):
    definition = """                                                                             
    filter_name: varchar(80)    # descriptive name of this filter
    filter_sampling_rate: int       # sampling rate for this filter
    ---
    filter_type: enum('lowpass', 'highpass', 'bandpass')
    filter_low_stop=0: float         # highest frequency for stop band for high pass side of filter
    filter_low_pass=0: float         # lowest frequency for pass band of high pass side of filter
    filter_high_pass=0: float         # highest frequency for pass band for low pass side of filter
    filter_high_stop=0: float         # lowest frequency for stop band of low pass side of filter
    filter_comments: varchar(255)   # comments about the filter
    filter_band_edges: blob         # numpy array containing the filter bands (redundant with individual parameters)
    filter_b: blob                  # numpy array containing the filter numerator 
    filter_a: blob                  # numpy array containing filter denominator                                                   
    """

    def zpk(self):
        # return the zeros, poles, and gain for the filter
        return signal.tf2zpk(self.filter_b, self.filter_a)

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
            filterdict['filter_low_stop'] = -1
            filterdict['filter_low_pass'] = -1
            filterdict['filter_high_pass'] = band_edges[0]
            filterdict['filter_high_stop'] = band_edges[1]
        elif filter_type == 'highpass':
            desired = [0, 1]
            filterdict['filter_low_stop'] = band_edges[0]
            filterdict['filter_low_pass'] = band_edges[1]
            filterdict['filter_high_pass'] = -1
            filterdict['filter_high_stop'] = -1
        else:
            desired = [0, 1, 1, 0]
            filterdict['filter_low_stop'] = band_edges[0]
            filterdict['filter_low_pass'] = band_edges[1]
            filterdict['filter_high_pass'] = band_edges[2]
            filterdict['filter_high_stop'] = band_edges[3]

        filterdict['filter_band_edges'] = np.asarray(band_edges)
        filterdict['filter_b'] = gsp.firdesign(numtaps, band_edges, desired, fs=fs, p=p)
        filterdict['filter_a'] = np.asarray([1])
        # add this filter to the table
        self.insert1(filterdict, replace="True")

    def plot_magnitude(self, filter_name, fs):
        filter = (self & {'filter_name': filter_name} & {'filter_sampling_rate' : fs}).fetch(as_dict=True)
        f = filter[0]
        plt.figure()
        w, h = signal.freqz(filter[0]['filter_b'], worN=65536)
        magnitude = 20 * np.log10(np.abs(h))
        plt.plot(w / np.pi * fs / 2, magnitude)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.title('Frequency Response')
        plt.xlim(0, np.max(f['filter_band_edges'] * 2))
        plt.ylim(np.min(magnitude), -1 * np.min(magnitude) * .1)
        plt.grid(True)

    def plot_fir_filter(self, filter_name, fs):
        filter = (self & {'filter_name': filter_name} & {'filter_sampling_rate' : fs}).fetch(as_dict=True)
        f = filter[0]
        plt.figure()
        plt.clf()
        b = plt.plot(f['filter_b'], 'k' )
        plt.xlabel('Coefficient')
        plt.ylabel('Magnitude')
        plt.title('Filter Taps')
        plt.grid(True)

    def filter_delay(self, filter_name, fs):
        # return the filter delay
        filter = (self & {'filter_name': filter_name} & {'filter_sampling_rate': fs}).fetch(as_dict=True)
        f = filter[0]
        return calc_filter_delay(filter['filter_b'])

    def filter_data_nwb(nwb_file_name, timestamps, data, fs_orig, filter_b, interval_list, electrodes, decimation):
        """
        :param nwb_file_name: str   name of previously created nwb file where filtered data should be stored
        :param timestamps: numpy array with list of timestamps for data
        :param data: original data array
        :param fs_orig: sampling frequency of data
        :param filter_coeff: numpy array with filter coefficients for FIR filter
        :param interval_list: 2D numpy array with start and stop times of intervals to be filtered
        :param electrodes: list of electrodes to filter
        :param decimation: decimation factor
        :return: file handle to the new nwb file with the data saved in an electricalseries object
        """

        n_dim = len(data.shape)
        n_samples = len(timestamps)
        electrode_axis = int(np.where(data.shape == n_samples))
        # assume 2d array, so the time_axis is the non-electrode axis
        time_axis = 1 - electrode_axis
        input_dim_restrictions = [None] * n_dim
        input_dim_restrictions[electrode_axis] = np.s_[electrodes]

        indices = []
        output_shape_list = [0] * n_dim
        output_shape_list[electrode_axis] = len(electrodes)
        output_offsets = [0]

        filter_delay = calc_filter_delay(filter_b)

        for a_start, a_stop in interval_list:
            frm, to = np.searchsorted(timestamps, (a_start, a_stop))
            if to > n_samples:
                to = n_samples

            indices.append((frm, to))
            shape, dtype = gsp.filter_data_fir(data,
                                               filter_b,
                                               axis=time_axis,
                                               input_index_bounds=[frm, to],
                                               output_index_bounds=[filter_delay, filter_delay + to - frm],
                                               describe_dims=True,
                                               ds=decimation,
                                               input_dim_restrictions=input_dim_restrictions)
            output_offsets.append(output_offsets[-1] + shape[time_axis])
            output_shape_list[time_axis] += shape[time_axis]

        # open the nwb file to create the dynamic table region and electrode series, then write and close the file
        try:
            io = pynwb.NWBHDF5IO(nwb_file_name, mode='r+')
            nwbf = io.read()
        except:
            print('Error in Filter.filter_data_nwb: nwbfile {} cannot be opened for writing\n'.format(
                nwb_file_name))
            return

        table_region = DynamicTableRegion('selected electrodes', electrodes, 'filtered electrode table',
                                table=nwbf.ElectrodeTable)
        es = pynwb.ecephys.ElectricalSeries(name='filt test', data=np.empty(tuple(output_shape_list), dtype='i2'),
                                            electrodes=table_region, timestamps= np.empty(0))
        nwbf.processing['ecephys'].add(es)
        io.write()
        io.close()

        # reload the NWB file to get the h5py objects for the data and the timestamps
        io = pynwb.NWBHDF5IO(nwb_file_name, mode='r+')
        nwbf = io.read()

         #es = nwb.processing[]


        indices = np.array(indices, ndmin=2)
        new_epoch_data = np.zeros(indices.shape)

        # Now let's set up our output dataset
        ts_offset = 0
        new_filepath = '/home/jchu/DataHDD/install_07-09-2017_0200_0400_sd09-run.hdf5'


        for ii, (start, stop) in enumerate(indices):
            extracted_ts = ts[start:stop:ds]

            print(f"Diffs {np.diff(extracted_ts)}")
            new_ts[ts_offset:ts_offset + len(extracted_ts)] = extracted_ts
            ts_offset += len(extracted_ts)

            # finally ready to filter data!
            gsp.filter_data_fir(infile['chdata'],
                                fg_filter,
                                axis=time_axis,
                                input_index_bounds=[start, stop],
                                output_index_bounds=[filter_delay, filter_delay + stop - start],
                                ds=ds,
                                input_dim_restrictions=input_dim_restrictions,
                                outarray=outfile['fast_gamma_data'],
                                output_offset=output_offsets[ii])


def calc_filter_delay(filter_b):
    """
    :param filter_b:
    :return: filter delay
    """
    return (len(filter_b) - 1) // 2



def create_standard_filters():
    """ Add standard filters to the Filter table including
    0-400 Hz low pass for continuous raw data -> LFP
    """
    Filter().add_filter('LFP 0-400 Hz', 20000, 'lowpass', [400, 425], 'standard LFP filter for 20 KHz data')
    Filter().add_filter('LFP 0-400 Hz', 30000, 'lowpass', [400, 425], 'standard LFP filter for 20 KHz data')
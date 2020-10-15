

import os
data_dir = '/Users/loren/data/nwb_builder_test_data'  # CHANGE ME

os.environ['NWB_DATAJOINT_BASE_DIR'] = data_dir
os.environ['KACHERY_STORAGE_DIR'] = os.path.join(data_dir, 'kachery-storage')
os.environ['SPIKE_SORTING_STORAGE_DIR'] = os.path.join(data_dir, 'spikesorting')

import numpy as np
import pynwb
import os

# DataJoint and DataJoint schema
import nwb_datajoint as nd
import datajoint as dj
from ndx_franklab_novela import Probe

def main():
    nwb_file_name = (nd.common.Session() & {'session_id': 'beans_01'}).fetch1('nwb_file_name')
    # ### Set the sort grouping by shank
    

    #nd.common.SortGroup().set_group_by_shank(nwb_file_name)
    #nd.common.SortGroup()
    #nd.common.SpikeSorter().insert_from_spikeinterface()
    #nd.common.SpikeSorterParameters().insert_from_spikeinterface()

    # ### create a 'franklab_mountainsort' parameter set
    p = (nd.common.SpikeSorterParameters() & {'sorter_name': 'mountainsort4', 'parameter_set_name' : 'default'}).fetch1()
    param = p['parameter_dict']
    param['adjacency_radius'] = 100
    param['curation'] = False
    param['num_workers'] = 1
    param['verbose'] = False
    param['clip_size'] = 30
    param['noise_overlap_threshold'] = 0

    #nd.common.SpikeSorterParameters().insert1({'sorter_name': 'mountainsort4', 'parameter_set_name' : 'franklab_mountainsort_20KHz', 'parameter_dict' : param}, skip_duplicates='True')
    

    param = p['parameter_dict']

    # ### Create a set of spike sorting parameters for sorting group 1

    # create a 10 second test intervals for debugging
    s1 = (nd.common.IntervalList() & {'interval_list_name' : '01_s1'}).fetch1('valid_times')
    a = s1[0][0]
    b = a + 100
    t = np.asarray([[a,b]])
    #t = np.vstack((t, np.asarray([[a+120,b+120]])))
    #nd.common.SortIntervalList().insert1({'nwb_file_name' : nwb_file_name, 'sort_interval_list_name' : 'test', 'sort_intervals' : t}, replace='True')

    # create the sorting waveform parameters table
    n_noise_waveforms = 1000 # the number of random noise waveforms to save
    save_all_waveforms='False'
    waveform_param_dict = st.postprocessing.get_waveforms_params()
    waveform_param_dict['grouping_property'] = 'group'
    # set the window to half of the clip size before and half after
    waveform_param_dict['ms_before'] = .75
    waveform_param_dict['ms_after'] = .75
    waveform_param_dict['dtype'] = 'i2'
    waveform_param_dict['verbose'] = False
    waveform_param_dict['max_spikes_per_unit'] = 1000
    nd.common.SpikeSortingWaveformParameters.insert1({'waveform_parameters_name' : 'franklab default', 'n_noise_waveforms' : n_noise_waveforms, 
                                                    'save_all_waveforms': save_all_waveforms, 'waveform_parameter_dict' : waveform_param_dict})

    sort_group_id = 1
    key = dict()
    key['nwb_file_name'] = nwb_file_name
    key['sort_group_id'] = sort_group_id
    key['sorter_name'] = 'mountainsort4'
    key['parameter_set_name'] = 'franklab_mountainsort_20KHz'
    key['waveform_parameters_name'] = 'franklab default'
    key['interval_list_name'] = '01_s1'
    key['sort_interval_list_name'] = 'test'
    nd.common.SpikeSortingParameters().insert1(key, skip_duplicates='True')


    # ### run the sort - this can take some time
    print('sorting')
    nd.common.SpikeSorting().populate()


    # 
    # ### Example: Retrieve the spike trains:
    # Note that these spikes are all noise; this is for demonstration purposes only.

    sorting = (nd.common.SpikeSorting & {'nwb_file_name' : nwb_file_name, 'sort_group_id' : sort_group_id}).fetch()
    key = {'nwb_file_name' : nwb_file_name, 'sort_group_id' : sort_group_id}
    units = (nd.common.SpikeSorting & key).fetch_nwb()[0]['units'].to_dataframe()
    #print(units)


if __name__ == "__main__":
    main()

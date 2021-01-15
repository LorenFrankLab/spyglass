

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
#import datajoint as dj
from ndx_franklab_novela import Probe

nwb_file_name = (nd.common.Session() & {'session_id': 'beans_01'}).fetch1('nwb_file_name')



nd.common.SpikeSorting().populate()


# 
# ### Example: Retrieve the spike trains:
# Note that these spikes are all noise; this is for demonstration purposes only.

sorting = (nd.common.SpikeSorting & {'nwb_file_name' : nwb_file_name, 'sort_group_id' : sort_group_id}).fetch()
key = {'nwb_file_name' : nwb_file_name, 'sort_group_id' : sort_group_id}
units = (nd.common.SpikeSorting & key).fetch_nwb()[0]['units'].to_dataframe()
units


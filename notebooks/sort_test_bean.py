from pathlib import Path
import os

import numpy as np
import pynwb
import datajoint as dj
import nwb_datajoint as nd
from ndx_franklab_novela import Probe

import spiketoolkit as st

import warnings
warnings.simplefilter('ignore')

data_dir = Path('/stelmo/nwb') # CHANGE ME TO THE BASE DIRECTORY FOR DATA STORAGE ON YOUR SYSTEM

os.environ['NWB_DATAJOINT_BASE_DIR'] = str(data_dir)
os.environ['KACHERY_STORAGE_DIR'] = str(data_dir / 'kachery-storage')
os.environ['SPIKE_SORTING_STORAGE_DIR'] = str(data_dir / 'spikesorting')

# define the file names
nwb_file_name = 'beans20190718.nwb'
filename, file_extension = os.path.splitext(nwb_file_name)
nwb_file_name2 = filename + '_' + file_extension

nd.common.SpikeSorting().populate()
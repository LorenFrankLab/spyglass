import os
from pathlib import Path

# CHANGE ME TO THE BASE DIRECTORY FOR DATA STORAGE ON YOUR SYSTEM
# data_dir = Path('/Users/loren/data/nwb_builder_test_data')
data_dir = Path('/mnt/c/Users/Ryan/Documents/NWB_Data/Frank Lab Data/')
# data_dir = Path('/stelmo/nwb/')

os.environ['NWB_DATAJOINT_BASE_DIR'] = str(data_dir)
os.environ['KACHERY_STORAGE_DIR'] = str(data_dir / 'kachery-storage')
os.environ['DJ_SUPPORT_FILEPATH_MANAGEMENT'] = 'TRUE'

raw_dir = data_dir / 'raw'
analysis_dir = data_dir / 'analysis'

import datajoint as dj
dj.config['database.host'] = 'localhost'  # CHANGE THESE AS NEEDED
dj.config['database.user'] = 'root'
dj.config['database.password'] = 'tutorial'
dj.config['stores'] = {
  'raw': {
    'protocol': 'file',
    'location': str(raw_dir),
    'stage': str(raw_dir)
  },
  'analysis': {
    'protocol': 'file',
    'location': str(analysis_dir),
    'stage': str(analysis_dir)
  }
}

import nwb_datajoint as nd

nd.insert_sessions(['beans20190718-trim.nwb'])

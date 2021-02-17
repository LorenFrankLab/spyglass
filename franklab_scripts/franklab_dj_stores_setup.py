# Set the locations of the external file stores for datajoint

import datajoint as dj
import os
import pathlib

assert os.getenv('NWB_DATAJOINT_BASE_DIR') is not None, 'environment variable NWB_DATAJOINT_BASE_DIR must be set'

data_dir = pathlib.Path(os.environ['NWB_DATAJOINT_BASE_DIR'])
raw_dir = data_dir / 'raw'
analysis_dir = data_dir / 'analysis'

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

dj.config.save_global()

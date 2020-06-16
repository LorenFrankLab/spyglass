#!/usr/bin/env python

import os

# Configure datajoint
import datajoint as dj
dj.config["enable_python_native_blobs"] = True

# Import nwb_datajoint
import nwb_datajoint as nwb_dj

def main():
  # Set the base directory and kachery storage directory
  os.environ['NWB_DATAJOINT_BASE_DIR'] = '/data'
  os.environ['KACHERY_STORAGE_DIR'] = '/data/kachery-storage'

  # Import the sessions
  nwb_raw_file_name = 'nwb_builder_test_data/beans20190718.nwb'
  nwb_dj.insert_sessions([nwb_raw_file_name]) 

if __name__ == '__main__':
    main()
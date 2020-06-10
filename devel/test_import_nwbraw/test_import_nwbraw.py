#!/usr/bin/env python

import pynwb
import os

from ndx_franklab_novela import probe

os.environ['DJ_SUPPORT_FILEPATH_MANAGEMENT'] = 'TRUE'
import datajoint as dj
dj.config["enable_python_native_blobs"] = True
if 'DJ_DATABASE_USER' in os.environ:
  dj.config['database.user'] = os.environ['DJ_DATABASE_USER']
if 'DJ_DATABASE_PASSWORD' in os.environ:
  dj.config['database.password'] = os.environ['DJ_DATABASE_PASSWORD']

import nwb_datajoint as nwb_dj

def main():
  nwb_raw_file_name = '/data/nwb_builder_test_data/beans20190718.nwb'
  # nwb_raw_file_name = '/data/nwb_builder_test_data/beans20190718ex.nwb'

  nwb_dj.NWBPopulate([nwb_raw_file_name]) 

if __name__ == '__main__':
    main()
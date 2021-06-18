#!/usr/bin/python

import os
import pathlib
import sys

import datajoint as dj


def set_configuration(user_name):
    # define the hostname and port to connect
    dj.config['database.host'] = 'lmf-db.cin.ucsf.edu'
    dj.config['database.port'] = 3306
    dj.config['enable_python_native_blobs'] = True
    dj.config['database.use_tls'] = True
    dj.config['database.user'] = user_name

    # change password; this will prompt you to type in a new password
    dj.set_password()

    # next, set up external stores
    # (read about them here: https://docs.datajoint.io/python/admin/5-blob-config.html)

    assert os.getenv(
        'NWB_DATAJOINT_BASE_DIR') is not None, 'environment variable NWB_DATAJOINT_BASE_DIR must be set'

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

    # finally, save these settings
    dj.config.save_global()


if __name__ == "__main__":
    set_configuration(sys.argv[1])

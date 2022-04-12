#!/usr/bin/python

import os
import pathlib
import sys
import yaml
import tempfile

import datajoint as dj

def generate_config_yaml(filename: str):
    """Generate a default yaml configuration file for the Frank laboratory for the specified user_name

    Parameters
    ----------
    filename : str
        The name of the file to generate
    """
    config = {}
       # define the hostname and port to connect
    config['database.host'] = 'lmf-db.cin.ucsf.edu'
    config['database.port'] = 3306
    config['enable_python_native_blobs'] = True
    config['database.use_tls'] = True

    # next, set up external stores
    # (read about them here: https://docs.datajoint.io/python/admin/5-blob-config.html)

    assert os.getenv('SPYGLASS_BASE_DIR') is not None, 'environment variable SPYGLASS_BASE_DIR must be set'

    data_dir = pathlib.Path(os.environ['SPYGLASS_BASE_DIR'])
    raw_dir = data_dir / 'raw'
    analysis_dir = data_dir / 'analysis'

    config['stores'] = {
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
    with open(filename, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


def set_configuration(user_name:str, file_name:str = None):
    """Sets the dj.config parameters for the specified user. Allows for specification of a yaml file with the defaults; 
    if no file is specified, it will write one using the generate_config_yaml function.

    Parameters
    ----------
    user_name : str
        The user_name to set configuration options for
    file_name : str, optional
        A yaml file with the configuration information, by default None
    """
    if file_name is None:
        #create a named temporary file and write the default config to it
        config_file = tempfile.NamedTemporaryFile("r+")
        generate_config_yaml(config_file.name)
        config_file.seek(0)
    else:
        config_file = open(file_name)
   
    # now read in the config file
    config = yaml.full_load(config_file)

    # copy the elements of config to dj.config
    for item in config:
        dj.config[item] = config[item]

    # set the users password
    dj.set_password()
    
    # finally, save these settings
    dj.config.save_global()


if __name__ == "__main__":
    set_configuration(sys.argv[1])

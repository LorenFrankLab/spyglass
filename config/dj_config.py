#!/usr/bin/python

import os
import pathlib
import sys
import tempfile

import datajoint as dj
import yaml


def generate_config_yaml(filename: str, **kwargs):
    """Generate a yaml configuration file for the specified user_name. By default this

    Parameters
    ----------
    filename : str
        The name of the file to generate
    **kwargs: list of parameters names and values that can include
        database_host : host name of system running mysql (default lmf-db.cin.ucsf.edu)
        database_port : port number for mysql server (default 3306)
        database_use_tls : True or False (default True to use tls encryption)
        raw
    """
    config = {}
    # define the hostname and port to connect
    print("printing kwargs")
    print(kwargs)
    config["database.host"] = (
        kwargs["database_host"] if "database_host" in kwargs else "lmf-db.cin.ucsf.edu"
    )
    config["database.port"] = (
        kwargs["database_port"] if "database_port" in kwargs else 3306
    )
    config["database.use_tls"] = (
        kwargs["database_use_tls"] if "database.use_tls" in kwargs else True
    )
    config["filepath_checksum_size_limit"] = 1 * 1024**3
    config["enable_python_native_blobs"] = True

    # next, set up external stores
    # (read about them here: https://docs.datajoint.io/python/admin/5-blob-config.html)

    assert (
        os.getenv("SPYGLASS_BASE_DIR") is not None
    ), "environment variable SPYGLASS_BASE_DIR must be set"

    data_dir = pathlib.Path(os.environ["SPYGLASS_BASE_DIR"])
    raw_dir = data_dir / "raw"
    analysis_dir = data_dir / "analysis"

    config["stores"] = {
        "raw": {"protocol": "file", "location": str(raw_dir), "stage": str(raw_dir)},
        "analysis": {
            "protocol": "file",
            "location": str(analysis_dir),
            "stage": str(analysis_dir),
        },
    }
    with open(filename, "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


def set_configuration(user_name: str, file_name: str = None):
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
        # create a named temporary file and write the default config to it
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

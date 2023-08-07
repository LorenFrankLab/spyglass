#!/usr/bin/python

import os
import sys
import tempfile

import datajoint as dj
import yaml
import json
import warnings

from pymysql.err import OperationalError


def generate_config(filename: str = None, **kwargs):
    """Generate a datajoint configuration file.

    Parameters
    ----------
    filename : str
        The name of the file to generate. Must be either yaml or json
    **kwargs: list of parameters names and values that can include
        base_dir : SPYGLASS_BASE_DIR
        database_user : user name of system running mysql
        database_host : mysql host name (default lmf-db.cin.ucsf.edu)
        database_port : port number for mysql server (default 3306)
        database_use_tls : Default True. Use TLS encryption.
    """
    # TODO: merge with existing spyglass.settings.py

    base_dir = os.environ.get("SPYGLASS_BASE_DIR") or kwargs.get("base_dir")
    if not base_dir:
        raise ValueError(
            "Please set base directory environment variable SPYGLASS_BASE_DIR"
        )

    base_dir = os.path.abspath(base_dir)
    if not os.path.exists(base_dir):
        warnings.warn(f"Base dir does not exist on this machine: {base_dir}")

    raw_dir = os.path.join(base_dir, "raw")
    analysis_dir = os.path.join(base_dir, "analysis")

    config = {
        "database.host": kwargs.get("database_host", "lmf-db.cin.ucsf.edu"),
        "database.user": kwargs.get("database_user"),
        "database.port": kwargs.get("database_port", 3306),
        "database.use_tls": kwargs.get("database_use_tls", True),
        "filepath_checksum_size_limit": 1 * 1024**3,
        "enable_python_native_blobs": True,
        "stores": {
            "raw": {
                "protocol": "file",
                "location": raw_dir,
                "stage": raw_dir,
            },
            "analysis": {
                "protocol": "file",
                "location": analysis_dir,
                "stage": analysis_dir,
            },
        },
        "custom": {"spyglass_dirs": {"base": base_dir}},
    }
    if not kwargs.get("database_user"):
        # Adding then removing if empty retains order to make easier to read
        config.pop("database.user")

    if not filename:
        filename = "dj_local_config.json"
    if os.path(filename).exists():
        warnings.warn(f"File already exists: {filename}")
    else:
        with open(filename, "w") as outfile:
            if filename.endswith("json"):
                json.dump(config, outfile, indent=2)
            else:
                yaml.dump(config, outfile, default_flow_style=False)

    return config


def set_configuration(config: dict):
    """Sets the dj.config parameters.

    Parameters
    ----------
    config : dict
        Datajoint config as dictionary
    """
    # copy the elements of config to dj.config
    for key, value in config.items():
        dj.config[key] = value

    dj.set_password()  # set the users password
    dj.config.save_global()  # save these settings


def main(*args):
    user_name, base_dir, outfile = args + (None,) * (3 - len(args))

    config = generate_config(
        outfile, database_user=user_name, base_dir=base_dir
    )
    try:
        set_configuration(config)
    except OperationalError as e:
        warnings.warn(f"Database connections issues: {e}")


if __name__ == "__main__":
    main(*sys.argv[1:])

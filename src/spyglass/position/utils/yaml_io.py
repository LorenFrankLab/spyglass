"""Thin YAML helpers used across the position pipeline.

Consolidates the repeated ``yaml.safe_load`` / ``yaml.safe_dump`` patterns
from train.py, video.py, and tool_strategies.py into one importable location.
"""

from pathlib import Path
from typing import Union

import yaml


def load_yaml(path: Union[str, Path]) -> dict:
    """Read a YAML file and return its contents as a dict.

    Parameters
    ----------
    path : str or Path
        Path to the YAML file.

    Returns
    -------
    dict
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


def dump_yaml(path: Union[str, Path], data: dict) -> None:
    """Write *data* to a YAML file at *path*.

    Parameters
    ----------
    path : str or Path
        Destination file path.  Parent directory must already exist.
    data : dict
        Data to serialise.
    """
    with open(path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True)

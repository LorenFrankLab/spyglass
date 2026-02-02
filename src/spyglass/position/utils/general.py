"""General utilities for position processing.

This module contains general-purpose helper functions used across
the position pipeline. These are re-exported from the parent utils.py
module for backwards compatibility.
"""

import inspect
import os
from pathlib import Path


def get_param_names(func):
    """Get parameter names for a function signature.

    Parameters
    ----------
    func : callable
        Function to inspect

    Returns
    -------
    list
        List of parameter names

    Examples
    --------
    >>> def my_func(a, b, c=3):
    ...     pass
    >>> get_param_names(my_func)
    ['a', 'b', 'c']
    """
    if not callable(func):
        return []  # If called when func is not imported
    return list(inspect.signature(func).parameters)


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to remove special characters.

    Parameters
    ----------
    filename : str
        Original filename

    Returns
    -------
    str
        Sanitized filename with special characters replaced

    Examples
    --------
    >>> sanitize_filename("my file, name & stuff.txt")
    'my_file-_name_and_stuff_txt'
    """
    char_map = {
        " ": "_",
        ".": "_",
        ",": "-",
        "&": "and",
        "'": "",
    }
    return "".join([char_map.get(c, c) for c in filename])


def get_most_recent_file(path: Path, ext: str = "") -> Path:
    """Get the most recent file of a given extension in a directory.

    Parameters
    ----------
    path : Path
        Path to the directory containing the files
    ext : str, optional
        File extension to filter by (e.g., ".h5", ".csv"). Default is "",
        which returns the most recent file regardless of extension.

    Returns
    -------
    Path
        Path to the most recently modified file

    Raises
    ------
    FileNotFoundError
        If no files with the given extension are found

    Examples
    --------
    >>> from pathlib import Path
    >>> get_most_recent_file(Path("/path/to/dir"), ext=".h5")
    PosixPath('/path/to/dir/latest_file.h5')
    """
    eval_files = list(Path(path).glob(f"*{ext}"))
    if not eval_files:
        raise FileNotFoundError(f"No {ext} files found in directory: {path}")

    eval_latest, max_modified_time = None, 0
    for eval_path in eval_files:
        modified_time = os.path.getmtime(eval_path)
        if modified_time > max_modified_time:
            eval_latest = eval_path
            max_modified_time = modified_time

    return eval_latest

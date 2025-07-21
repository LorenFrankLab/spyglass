import inspect
import os
from pathlib import Path


def get_param_names(func):
    """Get parameter names for a function signature."""
    if not callable(func):
        return []  # If called when func is not imported
    return list(inspect.signature(func).parameters)


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to remove special characters"""
    char_map = {
        " ": "_",
        ".": "_",
        ",": "-",
        "&": "and",
        "'": "",
    }
    return "".join([char_map.get(c, c) for c in filename])


def get_most_recent_file(path: Path, ext: str = "") -> Path:
    """Get the most recent file of a given extension in a directory

    Parameters
    ----------
    path : Path
        Path to the directory containing the files
    ext : str
        File extension to filter by (e.g., ".h5", ".csv"). Default is "",
        which returns the most recent file regardless of extension.
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

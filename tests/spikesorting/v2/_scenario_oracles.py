"""Independent input-artifact readers for declared integration scenarios."""

from pathlib import Path

import numpy as np


def _raw_acquisition_clock(nwb_source):
    """Read a fixture's raw clock from an absolute path or ingested filename."""
    import h5py

    path = Path(nwb_source)
    if not path.is_absolute():
        from spyglass.settings import raw_dir

        path = Path(raw_dir) / path
    with h5py.File(path, "r") as handle:
        series = handle["acquisition/e-series"]
        object_id = series.attrs["object_id"]
        if isinstance(object_id, bytes):
            object_id = object_id.decode()
        if "timestamps" in series:
            clock = np.asarray(series["timestamps"][:], dtype=np.float64)
        else:
            start = float(series["starting_time"][()])
            rate = float(series["starting_time"].attrs["rate"])
            clock = start + np.arange(series["data"].shape[0]) / rate
    return clock, str(object_id)

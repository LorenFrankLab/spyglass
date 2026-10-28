"""Compression of analysis-file objects written through AnalysisMixin."""

from datetime import datetime

import h5py
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def frame():
    """A DataFrame with enough rows for compression to show."""
    return pd.DataFrame(
        {"a": np.arange(4096, dtype="float64"), "b": np.zeros(4096)}
    )


def test_table_columns_compressed(frame):
    """Every column of a converted DataFrame is wrapped for compression."""
    from hdmf.backends.hdf5.h5_utils import H5DataIO
    from hdmf.common import DynamicTable

    from spyglass.utils.h5py_helper_fn import GZIP_LEVEL
    from spyglass.utils.mixins.analysis import AnalysisMixin

    table = DynamicTable.from_dataframe(name="pandas_table", df=frame)
    AnalysisMixin._compress_table_columns(table)

    for column in (*table.columns, table.id):
        assert isinstance(column.data, H5DataIO), column.name
        assert column.data.io_settings["compression"] == "gzip"
        assert column.data.io_settings["compression_opts"] == GZIP_LEVEL


def test_table_columns_compressed_on_disk(frame, tmp_path):
    """The filter survives the write, not just the in-memory wrapper."""
    from hdmf.common import DynamicTable
    from pynwb import NWBHDF5IO, NWBFile

    from spyglass.utils.mixins.analysis import AnalysisMixin

    table = DynamicTable.from_dataframe(name="pandas_table", df=frame)
    AnalysisMixin._compress_table_columns(table)

    nwbf = NWBFile(
        session_description="s",
        identifier="i",
        session_start_time=datetime.now().astimezone(),
    )
    nwbf.add_scratch(table)
    path = tmp_path / "analysis.nwb"
    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwbf)

    with h5py.File(path, "r") as f:
        for col in ("a", "b", "id"):
            ds = f[f"scratch/pandas_table/{col}"]
            assert ds.compression == "gzip", f"{col} written uncompressed"


def test_scratch_array_still_compressed():
    """The ndarray branch keeps its constructor wrapping.

    ``set_data_io`` is a silent no-op on ScratchData, so the two branches
    cannot share one mechanism.
    """
    from hdmf.backends.hdf5.h5_utils import H5DataIO
    from pynwb.core import ScratchData

    from spyglass.utils.h5py_helper_fn import compressed_data

    scratch = ScratchData(
        name="numpy_array",
        data=compressed_data(np.arange(4096)),
        description="d",
    )
    assert isinstance(scratch.data, H5DataIO)

    plain = ScratchData(name="x", data=np.arange(4096), description="d")
    plain.set_data_io(H5DataIO, {"compression": "gzip"})
    assert not isinstance(plain.data, H5DataIO), (
        "set_data_io now works on ScratchData; the constructor wrapping in "
        "_add_nwb_object_to_open_nwb could be simplified"
    )

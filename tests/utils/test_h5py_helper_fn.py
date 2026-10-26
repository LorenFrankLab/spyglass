"""Tests for HDF5 helpers."""

import h5py
import numpy as np
import pytest


@pytest.fixture
def table_missing_id(tmp_path):
    """A DynamicTable group with a column but no 'id'."""
    path = tmp_path / "table.nwb"
    with h5py.File(path, "w") as f:
        grp = f.create_group("tbl")
        grp.attrs["neurodata_type"] = "DynamicTable"
        grp.create_dataset("a", data=np.arange(64))
    return path


def test_add_id_column_uses_shared_level(table_missing_id):
    """The added column is compressed at the one shared level."""
    from spyglass.utils.h5py_helper_fn import (
        GZIP_LEVEL,
        add_id_column_to_table,
    )

    with h5py.File(table_missing_id, "a") as f:
        add_id_column_to_table(f, "tbl")

    with h5py.File(table_missing_id, "r") as f:
        id_ds = f["tbl/id"]
        assert id_ds.compression == "gzip"
        assert id_ds.compression_opts == GZIP_LEVEL
        np.testing.assert_array_equal(id_ds[:], np.arange(64))
        assert id_ds.attrs["neurodata_type"] == "ElementIdentifiers"


def test_add_id_column_is_idempotent(table_missing_id):
    """A table that already has an 'id' is left alone."""
    from spyglass.utils.h5py_helper_fn import add_id_column_to_table

    with h5py.File(table_missing_id, "a") as f:
        add_id_column_to_table(f, "tbl")
        before = f["tbl/id"].attrs["object_id"]
        add_id_column_to_table(f, "tbl")
        assert f["tbl/id"].attrs["object_id"] == before


def test_add_id_column_rejects_non_table(tmp_path):
    """A group that is not a DynamicTable raises."""
    from spyglass.utils.h5py_helper_fn import add_id_column_to_table

    path = tmp_path / "plain.nwb"
    with h5py.File(path, "w") as f:
        f.create_group("grp")

    with h5py.File(path, "a") as f:
        with pytest.raises(ValueError, match="not a DynamicTable"):
            add_id_column_to_table(f, "grp")
        with pytest.raises(ValueError, match="does not exist"):
            add_id_column_to_table(f, "absent")

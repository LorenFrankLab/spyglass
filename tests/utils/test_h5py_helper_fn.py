"""File-only checks for HDF5 dataset conversion."""

import h5py
import numpy as np
import pytest

from spyglass.utils.h5py_helper_fn import convert_dataset_type

pytestmark = pytest.mark.unit


@pytest.fixture(scope="session", autouse=True)
def mini_insert():
    """These file-only tests need no repository sample-data ingestion."""
    yield


@pytest.mark.parametrize("max_size", [8, None])
def test_conversion_preserves_compression_resize_limits_and_fill(
    tmp_path, max_size
):
    path = tmp_path / "compressed.h5"
    with h5py.File(path, "w") as file:
        group = file.create_group("electrodes")
        dataset = group.create_dataset(
            "rel_x",
            data=np.array([-11, 0, 12], dtype=np.int16),
            chunks=(2,),
            maxshape=(max_size,),
            compression="gzip",
            compression_opts=5,
            shuffle=True,
            fletcher32=True,
            fillvalue=-91,
        )
        dataset.attrs.update(
            neurodata_type="VectorData", namespace="hdmf-common", object_id="x"
        )
        convert_dataset_type(group, "rel_x", "float64")

    with h5py.File(path, "r+") as file:
        dataset = file["electrodes/rel_x"]
        assert dataset.dtype == np.float64
        np.testing.assert_array_equal(dataset[:], [-11, 0, 12])
        assert dataset.chunks == (2,)
        assert dataset.maxshape == (max_size,)
        assert dataset.compression == "gzip"
        assert dataset.compression_opts == 5
        assert dataset.shuffle is True
        assert dataset.fletcher32 is True
        assert dataset.fillvalue == -91
        assert dict(dataset.attrs) == {
            "neurodata_type": "VectorData",
            "namespace": "hdmf-common",
            "object_id": "x",
        }
        dataset.resize((6,))
        np.testing.assert_array_equal(dataset[:], [-11, 0, 12, -91, -91, -91])


@pytest.mark.parametrize("values", [np.float32(1.25), np.array([1.25, -2.5])])
def test_conversion_keeps_scalar_and_contiguous_layout(tmp_path, values):
    path = tmp_path / "unchunked.h5"
    values = np.asarray(values, dtype=np.float32)
    with h5py.File(path, "w") as file:
        dataset = file.create_dataset("data", data=values)
        dataset.attrs["description"] = "unchunked source"
        convert_dataset_type(file, "/data", "float64")

    with h5py.File(path, "r") as file:
        dataset = file["data"]
        assert dataset.dtype == np.float64
        assert dataset.shape == values.shape
        assert dataset.maxshape == values.shape
        assert dataset.chunks is None
        assert dataset.id.get_create_plist().get_layout() == h5py.h5d.CONTIGUOUS
        assert dataset.attrs["description"] == "unchunked source"
        np.testing.assert_array_equal(dataset[()], values)


def test_conversion_preserves_scaleoffset_filter(tmp_path):
    path = tmp_path / "scaleoffset.h5"
    with h5py.File(path, "w") as file:
        file.create_dataset(
            "data",
            data=np.array([0.125, -0.25, 1.5], dtype=np.float32),
            chunks=(3,),
            maxshape=(None,),
            scaleoffset=4,
            compression="gzip",
            fillvalue=-7,
        )
        convert_dataset_type(file, "data", "float64")

    with h5py.File(path, "r") as file:
        dataset = file["data"]
        assert dataset.dtype == np.float64
        assert dataset.scaleoffset == 4
        assert dataset.compression == "gzip"
        assert dataset.maxshape == (None,)
        assert dataset.fillvalue == -7
        np.testing.assert_array_equal(dataset[:], [0.125, -0.25, 1.5])


def test_invalid_conversion_leaves_original_dataset(tmp_path):
    with h5py.File(tmp_path / "invalid.h5", "w") as file:
        original = file.create_dataset(
            "data", data=np.array([1, 2], dtype="i2")
        )
        with pytest.raises(TypeError):
            convert_dataset_type(file, "data", "invalid-dtype")
        assert file["data"].id == original.id
        assert file["data"].dtype == np.dtype("i2")
        np.testing.assert_array_equal(file["data"][:], [1, 2])

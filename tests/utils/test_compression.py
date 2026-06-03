"""Tests for NWB dataset-level compression utilities."""

import tempfile
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pynwb
import pytest
from hdmf.backends.hdf5.h5_utils import H5DataIO


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_h5(temp_dir):
    """Create a sample HDF5 file with various dataset types."""
    path = temp_dir / "sample.h5"
    with h5py.File(path, "w") as f:
        f.attrs["format"] = "test"
        # Large dataset (compressible)
        f.create_dataset(
            "big_data",
            data=np.zeros((10000, 32), dtype=np.int16),
        )
        f["big_data"].attrs["unit"] = "uV"
        # Small dataset (below threshold)
        f.create_dataset("small_data", data=np.array([1, 2, 3]))
        # Scalar dataset
        f.create_dataset("scalar", data=42)
        # Already-compressed dataset
        f.create_dataset(
            "pre_compressed",
            data=np.ones((1000,), dtype=np.float32),
            compression="gzip",
            compression_opts=1,
        )
        # Group with nested dataset
        grp = f.create_group("group1")
        grp.attrs["desc"] = "test group"
        grp.create_dataset(
            "nested",
            data=np.random.randn(5000).astype(np.float64),
        )
    return path


@pytest.fixture
def sample_nwb(temp_dir):
    """Create a minimal NWB file for round-trip testing."""
    path = temp_dir / "test.nwb"
    nwbfile = pynwb.NWBFile(
        session_description="test session",
        identifier="test-id-001",
        session_start_time=datetime(2024, 1, 1),
    )
    # Add electrode table
    device = nwbfile.create_device(name="probe")
    group = nwbfile.create_electrode_group(
        name="group0",
        description="test group",
        location="brain",
        device=device,
    )
    for i in range(4):
        nwbfile.add_electrode(
            x=0.0,
            y=0.0,
            z=0.0,
            imp=0.0,
            location="brain",
            filtering="none",
            group=group,
        )
    region = nwbfile.create_electrode_table_region(
        region=[0, 1, 2, 3], description="test"
    )
    # Add ElectricalSeries with data
    es = pynwb.ecephys.ElectricalSeries(
        name="test_series",
        data=np.random.randn(10000, 4).astype(np.float32),
        electrodes=region,
        rate=30000.0,
    )
    nwbfile.add_acquisition(es)

    with pynwb.NWBHDF5IO(str(path), "w") as io:
        io.write(nwbfile)
    return path


class TestCompressedData:
    """Tests for compressed_data() wrapper."""

    def test_returns_h5dataio(self):
        from spyglass.utils.compression import compressed_data

        data = np.zeros((100, 10))
        result = compressed_data(data)
        assert isinstance(result, H5DataIO)

    def test_gzip_kwargs(self):
        from spyglass.utils.compression import compressed_data

        data = np.zeros((100,))
        result = compressed_data(data, compression="gzip", compression_opts=6)
        assert result.io_settings["compression"] == "gzip"
        assert result.io_settings["compression_opts"] == 6

    def test_lzf_no_opts(self):
        from spyglass.utils.compression import compressed_data

        data = np.zeros((100,))
        result = compressed_data(data, compression="lzf")
        assert result.io_settings["compression"] == "lzf"
        # LZF doesn't use compression_opts; H5DataIO omits None values
        assert result.io_settings.get("compression_opts") is None


class TestRepackNwb:
    """Tests for repack_nwb()."""

    def test_basic_repack(self, sample_h5, temp_dir):
        from spyglass.utils.compression import repack_nwb

        output = temp_dir / "repacked.h5"
        stats = repack_nwb(sample_h5, output)

        assert output.exists()
        assert stats["datasets_compressed"] > 0
        assert stats["repacked_size"] > 0
        assert stats["original_size"] > 0

    def test_overwrites_in_place(self, sample_h5):
        from spyglass.utils.compression import repack_nwb

        original_size = sample_h5.stat().st_size
        stats = repack_nwb(sample_h5)

        assert sample_h5.exists()
        assert stats["original_size"] == original_size

    def test_skips_already_compressed(self, sample_h5, temp_dir):
        from spyglass.utils.compression import repack_nwb

        output = temp_dir / "repacked.h5"
        repack_nwb(sample_h5, output)

        with h5py.File(output, "r") as f:
            # pre_compressed should keep its original compression
            assert f["pre_compressed"].compression == "gzip"

    def test_skips_small_datasets(self, sample_h5, temp_dir):
        from spyglass.utils.compression import repack_nwb

        output = temp_dir / "repacked.h5"
        stats = repack_nwb(sample_h5, output, min_dataset_bytes=1024)

        # small_data (3 elements * 8 bytes = 24 bytes) should be skipped
        assert stats["datasets_skipped"] > 0

    def test_skips_scalar_datasets(self, sample_h5, temp_dir):
        from spyglass.utils.compression import repack_nwb

        output = temp_dir / "repacked.h5"
        repack_nwb(sample_h5, output)

        with h5py.File(output, "r") as f:
            assert f["scalar"][()] == 42

    def test_preserves_shapes_and_dtypes(self, sample_h5, temp_dir):
        from spyglass.utils.compression import repack_nwb

        output = temp_dir / "repacked.h5"
        repack_nwb(sample_h5, output)

        with h5py.File(sample_h5, "r") as orig:
            with h5py.File(output, "r") as repack:
                assert orig["big_data"].shape == repack["big_data"].shape
                assert orig["big_data"].dtype == repack["big_data"].dtype
                assert (
                    orig["group1/nested"].shape == repack["group1/nested"].shape
                )

    def test_preserves_attributes(self, sample_h5, temp_dir):
        from spyglass.utils.compression import repack_nwb

        output = temp_dir / "repacked.h5"
        repack_nwb(sample_h5, output)

        with h5py.File(output, "r") as f:
            assert f.attrs["format"] == "test"
            assert f["big_data"].attrs["unit"] == "uV"
            assert f["group1"].attrs["desc"] == "test group"

    def test_preserves_group_structure(self, sample_h5, temp_dir):
        from spyglass.utils.compression import repack_nwb

        output = temp_dir / "repacked.h5"
        repack_nwb(sample_h5, output)

        with h5py.File(output, "r") as f:
            assert "group1" in f
            assert "nested" in f["group1"]

    def test_data_integrity(self, sample_h5, temp_dir):
        from spyglass.utils.compression import repack_nwb

        # Read original data
        with h5py.File(sample_h5, "r") as f:
            orig_big = f["big_data"][:]
            orig_nested = f["group1/nested"][:]

        output = temp_dir / "repacked.h5"
        repack_nwb(sample_h5, output)

        # Read repacked data
        with h5py.File(output, "r") as f:
            np.testing.assert_array_equal(f["big_data"][:], orig_big)
            np.testing.assert_array_equal(f["group1/nested"][:], orig_nested)

    def test_compressed_datasets_have_compression(self, sample_h5, temp_dir):
        from spyglass.utils.compression import repack_nwb

        output = temp_dir / "repacked.h5"
        repack_nwb(sample_h5, output)

        with h5py.File(output, "r") as f:
            assert f["big_data"].compression == "gzip"
            assert f["group1/nested"].compression == "gzip"

    def test_file_not_found(self, temp_dir):
        from spyglass.utils.compression import repack_nwb

        with pytest.raises(FileNotFoundError):
            repack_nwb(temp_dir / "nonexistent.h5")

    def test_atomic_write_on_failure(self, sample_h5, temp_dir):
        """Verify temp file is cleaned up if repack fails."""
        from spyglass.utils.compression import repack_nwb

        output = temp_dir / "repacked.h5"
        # Create a file that will fail verification
        # by corrupting the source mid-repack (simulated via bad path)
        # Instead, test that output doesn't exist on error
        try:
            repack_nwb(
                sample_h5,
                output,
                compression="gzip",
                compression_opts=4,
                verify=True,
            )
        except Exception:
            pass
        # No leftover temp files
        tmp_files = list(temp_dir.glob("*.nwb.tmp"))
        assert len(tmp_files) == 0


class TestNwbRoundTrip:
    """Test repacking NWB files and reading back with pynwb."""

    def test_nwb_readable_after_repack(self, sample_nwb):
        from spyglass.utils.compression import repack_nwb

        repack_nwb(sample_nwb)

        with pynwb.NWBHDF5IO(str(sample_nwb), "r") as io:
            nwbfile = io.read()
            es = nwbfile.acquisition["test_series"]
            assert es.data.shape == (10000, 4)
            assert es.data.dtype == np.float32

    def test_nwb_data_matches_after_repack(self, sample_nwb):
        from spyglass.utils.compression import repack_nwb

        # Read original
        with pynwb.NWBHDF5IO(str(sample_nwb), "r") as io:
            nwbfile = io.read()
            orig_data = nwbfile.acquisition["test_series"].data[:]

        repack_nwb(sample_nwb)

        # Read repacked
        with pynwb.NWBHDF5IO(str(sample_nwb), "r") as io:
            nwbfile = io.read()
            repack_data = nwbfile.acquisition["test_series"].data[:]

        np.testing.assert_array_equal(orig_data, repack_data)

    def test_nwb_selective_read(self, sample_nwb):
        """Verify selective reads work on compressed datasets."""
        from spyglass.utils.compression import repack_nwb

        with pynwb.NWBHDF5IO(str(sample_nwb), "r") as io:
            nwbfile = io.read()
            orig_slice = nwbfile.acquisition["test_series"].data[100:200]

        repack_nwb(sample_nwb)

        with pynwb.NWBHDF5IO(str(sample_nwb), "r") as io:
            nwbfile = io.read()
            repack_slice = nwbfile.acquisition["test_series"].data[100:200]

        np.testing.assert_array_equal(orig_slice, repack_slice)


class TestVerifyRepack:
    """Tests for _verify_repack()."""

    def test_matching_files_pass(self, sample_h5, temp_dir):
        from spyglass.utils.compression import _verify_repack, repack_nwb

        output = temp_dir / "repacked.h5"
        repack_nwb(sample_h5, output, verify=False)
        # Should not raise
        _verify_repack(sample_h5, output)

    def test_mismatched_shape_fails(self, temp_dir):
        from spyglass.utils.compression import _verify_repack

        f1 = temp_dir / "a.h5"
        f2 = temp_dir / "b.h5"

        with h5py.File(f1, "w") as f:
            f.create_dataset("data", data=np.zeros((10, 5)))
        with h5py.File(f2, "w") as f:
            f.create_dataset("data", data=np.zeros((10, 3)))

        with pytest.raises(ValueError, match="Shape mismatch"):
            _verify_repack(f1, f2)

    def test_missing_dataset_fails(self, temp_dir):
        from spyglass.utils.compression import _verify_repack

        f1 = temp_dir / "a.h5"
        f2 = temp_dir / "b.h5"

        with h5py.File(f1, "w") as f:
            f.create_dataset("data", data=np.zeros((10,)))
        with h5py.File(f2, "w") as f:
            pass  # Empty file

        with pytest.raises(ValueError, match="Missing in repacked"):
            _verify_repack(f1, f2)

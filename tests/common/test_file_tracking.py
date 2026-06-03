"""Unit tests for dataset-level compression file tracking.

These tests cover the compression tracking tables and utilities that
don't require full database access.
"""

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_h5(temp_dir):
    """Create a sample HDF5 file for testing."""
    path = temp_dir / "test.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=np.zeros((10000, 16), dtype=np.int16))
        f.create_dataset("timestamps", data=np.arange(10000.0))
        f.create_dataset("small", data=np.array([1, 2, 3]))
    return path


class TestRepackIntegrity:
    """Tests for repack round-trip integrity."""

    def test_repack_preserves_data(self, sample_h5, temp_dir):
        from spyglass.utils.compression import repack_nwb

        with h5py.File(sample_h5, "r") as f:
            orig_data = f["data"][:]
            orig_ts = f["timestamps"][:]

        output = temp_dir / "repacked.h5"
        repack_nwb(sample_h5, output)

        with h5py.File(output, "r") as f:
            np.testing.assert_array_equal(f["data"][:], orig_data)
            np.testing.assert_array_equal(f["timestamps"][:], orig_ts)

    def test_repack_applies_compression(self, sample_h5, temp_dir):
        from spyglass.utils.compression import repack_nwb

        output = temp_dir / "repacked.h5"
        stats = repack_nwb(sample_h5, output)

        assert stats["datasets_compressed"] >= 2  # data + timestamps
        with h5py.File(output, "r") as f:
            assert f["data"].compression == "gzip"
            assert f["timestamps"].compression == "gzip"

    def test_repack_in_place(self, sample_h5):
        from spyglass.utils.compression import repack_nwb

        orig_size = sample_h5.stat().st_size
        stats = repack_nwb(sample_h5)

        assert sample_h5.exists()
        assert stats["original_size"] == orig_size
        assert stats["repacked_size"] > 0

    def test_repack_stats_keys(self, sample_h5, temp_dir):
        from spyglass.utils.compression import repack_nwb

        output = temp_dir / "repacked.h5"
        stats = repack_nwb(sample_h5, output)

        expected_keys = {
            "original_size",
            "repacked_size",
            "compression_ratio",
            "datasets_compressed",
            "datasets_skipped",
        }
        assert set(stats.keys()) == expected_keys

    def test_cleanup_on_failure(self, temp_dir):
        """No temp files left after failed repack."""
        from spyglass.utils.compression import repack_nwb

        fake_path = temp_dir / "nonexistent.h5"
        with pytest.raises(FileNotFoundError):
            repack_nwb(fake_path)

        tmp_files = list(temp_dir.glob("*.nwb.tmp"))
        assert len(tmp_files) == 0

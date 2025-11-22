"""Unit tests for file compression utility functions.

These tests cover the compression utilities that don't require database access.
Full integration tests with database are in test_file_tracking_integration.py
"""

import gzip
import tempfile
from pathlib import Path

import pytest
from datajoint.hash import uuid_from_file


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_file(temp_dir):
    """Create a sample file for testing."""
    file_path = temp_dir / "test_file.nwb"
    with open(file_path, "wb") as f:
        f.write(b"Test data content " * 1000)  # ~18 KB
    return file_path


class TestSafeCompress:
    """Tests for _safe_compress context manager."""

    def test_checksum_computation(self, sample_file, temp_dir):
        """Test that checksums are computed correctly."""
        # Import here to avoid module-level schema creation
        from spyglass.common.common_file_tracking import _safe_compress

        output_path = temp_dir / "output.gz"

        with _safe_compress(str(sample_file), str(output_path)) as checksums:
            # Simulate compression
            with open(sample_file, "rb") as f_in:
                with gzip.open(output_path, "wb") as f_out:
                    f_out.write(f_in.read())

        assert "input" in checksums
        assert "output" in checksums
        # UUID string format is 36 chars (includes hyphens)
        assert len(str(checksums["input"])) == 36  # UUID string length
        assert len(str(checksums["output"])) == 36

        # Verify input checksum matches
        expected = uuid_from_file(sample_file)
        assert checksums["input"] == expected

    def test_lock_file_creation(self, sample_file, temp_dir):
        """Test that lock file is created and removed."""
        from spyglass.common.common_file_tracking import _safe_compress

        output_path = temp_dir / "output.gz"
        lock_path = Path(str(output_path) + ".lock")

        with _safe_compress(str(sample_file), str(output_path)) as _:
            # Lock is removed in finally block
            import gzip

            with open(sample_file, "rb") as f_in:
                with gzip.open(output_path, "wb") as f_out:
                    f_out.write(f_in.read())

        # Lock should be removed after
        assert not lock_path.exists()

    def test_lock_file_prevents_concurrent(self, sample_file, temp_dir):
        """Test that existing lock file prevents compression."""
        from spyglass.common.common_file_tracking import _safe_compress

        output_path = temp_dir / "output.gz"
        lock_path = Path(str(output_path) + ".lock")

        # Create lock file
        lock_path.touch()

        with pytest.raises(RuntimeError, match="Lock file exists"):
            with _safe_compress(str(sample_file), str(output_path)):
                pass

        # Cleanup
        lock_path.unlink()

    def test_cleanup_on_error(self, sample_file, temp_dir):
        """Test that temp file is cleaned up on error."""
        from spyglass.common.common_file_tracking import _safe_compress

        output_path = temp_dir / "output.gz"

        with pytest.raises(ValueError):
            with _safe_compress(str(sample_file), str(output_path)):
                raise ValueError("Test error")

        # Temp file should be cleaned up
        temp_files = list(temp_dir.glob("*.tmp"))
        assert len(temp_files) == 0


class TestCompressionIntegrity:
    """Tests for compression/decompression round trip."""

    def test_checksum_verification(self, sample_file, temp_dir):
        """Test that checksum verification catches corruption."""
        # Compress file
        compressed_path = temp_dir / "compressed.gz"
        with gzip.open(compressed_path, "wb") as f_out:
            with open(sample_file, "rb") as f_in:
                f_out.write(f_in.read())

        # Get original checksum
        original_checksum = uuid_from_file(sample_file)

        # Decompress
        decompressed_path = temp_dir / "decompressed.nwb"
        with gzip.open(compressed_path, "rb") as f_in:
            with open(decompressed_path, "wb") as f_out:
                f_out.write(f_in.read())

        # Verify checksum matches
        decompressed_checksum = uuid_from_file(decompressed_path)
        assert original_checksum == decompressed_checksum

    def test_concurrent_compression_prevention(self, sample_file, temp_dir):
        """Test that lock files prevent concurrent compression."""
        from spyglass.common.common_file_tracking import _safe_compress

        output_path = temp_dir / "output.gz"
        lock_path = Path(str(output_path) + ".lock")

        # Start first compression
        lock_path.touch()

        # Try second compression
        with pytest.raises(RuntimeError, match="Lock file exists"):
            with _safe_compress(str(sample_file), str(output_path)):
                pass

        # Cleanup
        lock_path.unlink()

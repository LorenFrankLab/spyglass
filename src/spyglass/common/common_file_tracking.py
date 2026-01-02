"""File compression tracking for Spyglass NWB files.

Provides compression/decompression with transparent access, leveraging
existing Nwbfile infrastructure and externals table for metadata.
"""

import gzip
import lzma
import os
import shutil
import tempfile
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

import datajoint as dj
import pynwb

from spyglass.common.common_nwbfile import Nwbfile
from spyglass.common.common_nwbfile import schema as nwbfile_schema
from spyglass.utils import logger

schema = dj.schema("common_file_compression")


# ============================================================================
# Compression Algorithm Classes
# ============================================================================


class CompressionAlgorithm(ABC):
    """Abstract base class for compression algorithms.

    Subclasses must define:
    - name: str - Algorithm identifier (e.g., 'gzip', 'lzma')
    - suffix: str - File extension (e.g., '.gz', '.xz')
    - compress(input_path, output_path, **kwargs) - Compression implementation
    - decompress(input_path, output_path) - Decompression implementation

    Algorithms auto-register on class definition via __init_subclass__.

    Examples
    --------
    >>> # Get algorithm instance
    >>> algo = CompressionAlgorithm.get("gzip")
    >>> algo.compress("input.nwb", "output.gz", compresslevel=6)
    >>> algo.decompress("output.gz", "restored.nwb")
    """

    # Class-level registry mapping algorithm name to instance
    _registry = {}

    # Subclasses must define these
    name = None
    suffix = None

    def __init_subclass__(cls, **kwargs):
        """Auto-register algorithm when subclass is defined."""
        super().__init_subclass__(**kwargs)
        if cls.name is not None:
            CompressionAlgorithm._registry[cls.name] = cls()

    @classmethod
    def get(cls, algorithm: str):
        """Get algorithm instance by name.

        Parameters
        ----------
        algorithm : str
            Algorithm name (e.g., 'gzip', 'lzma')

        Returns
        -------
        CompressionAlgorithm
            Algorithm instance

        Raises
        ------
        ValueError
            If algorithm not found in registry
        """
        if algorithm not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Unsupported algorithm: {algorithm}. "
                f"Available: {available}"
            )
        return cls._registry[algorithm]

    @classmethod
    def get_all(cls):
        """Get all registered algorithms.

        Returns
        -------
        dict
            Mapping of algorithm name to instance
        """
        return cls._registry.copy()

    @abstractmethod
    def compress(self, input_path, output_path, **kwargs):
        """Compress a file.

        Parameters
        ----------
        input_path : str or Path
            Path to input file
        output_path : str or Path
            Path to output compressed file
        **kwargs
            Algorithm-specific parameters
        """

    @abstractmethod
    def decompress(self, input_path, output_path):
        """Decompress a file.

        Parameters
        ----------
        input_path : str or Path
            Path to compressed file
        output_path : str or Path
            Path to output decompressed file
        """


class GzipAlgorithm(CompressionAlgorithm):
    """Gzip compression algorithm."""

    name = "gzip"
    suffix = ".gz"

    def compress(self, input_path, output_path, **kwargs):
        """Compress file using gzip.

        Parameters
        ----------
        input_path : str or Path
            Path to input file
        output_path : str or Path
            Path to output file
        **kwargs
            Passed to gzip.open (e.g., compresslevel=6)
        """
        with open(input_path, "rb") as f_in:
            with gzip.open(output_path, "wb", **kwargs) as f_out:
                shutil.copyfileobj(f_in, f_out)

    def decompress(self, input_path, output_path):
        """Decompress gzip file.

        Parameters
        ----------
        input_path : str or Path
            Path to compressed file
        output_path : str or Path
            Path to output file
        """
        with gzip.open(input_path, "rb") as f_in:
            with open(output_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)


class LzmaAlgorithm(CompressionAlgorithm):
    """LZMA/XZ compression algorithm."""

    name = "lzma"
    suffix = ".xz"

    def compress(self, input_path, output_path, **kwargs):
        """Compress file using lzma.

        Parameters
        ----------
        input_path : str or Path
            Path to input file
        output_path : str or Path
            Path to output file
        **kwargs
            Passed to lzma.open (e.g., preset=6)
        """
        with open(input_path, "rb") as f_in:
            with lzma.open(output_path, "wb", **kwargs) as f_out:
                shutil.copyfileobj(f_in, f_out)

    def decompress(self, input_path, output_path):
        """Decompress lzma file.

        Parameters
        ----------
        input_path : str or Path
            Path to compressed file
        output_path : str or Path
            Path to output file
        """
        with lzma.open(input_path, "rb") as f_in:
            with open(output_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)


@schema
class CompressionParams(dj.Lookup):
    """Supported compression algorithms and parameters.

    Maps algorithm names to CompressionAlgorithm instances with configurable
    parameters. Adding new algorithms requires defining a new CompressionAlgorithm
    subclass with compress() and decompress() methods.
    """

    definition = """
    param_id: int unsigned auto_increment  # Unique parameter set ID
    ---
    algorithm: varchar(32)      # Algorithm name (gzip, lzma, zstd, etc.)
    kwargs: blob                # Algorithm parameters as dict
    description: varchar(127)   # Human-readable description
    """

    contents = [
        (1, "gzip", {"compresslevel": 1}, "Gzip level 1 (fastest)"),
        (2, "gzip", {"compresslevel": 4}, "Gzip level 4 (balanced)"),
        (3, "gzip", {"compresslevel": 6}, "Gzip level 6 (default)"),
        (4, "gzip", {"compresslevel": 9}, "Gzip level 9 (best)"),
        (5, "lzma", {"preset": 1}, "LZMA preset 1 (fastest)"),
        (6, "lzma", {"preset": 6}, "LZMA preset 6 (default)"),
        (7, "lzma", {"preset": 9}, "LZMA preset 9 (best)"),
    ]

    default_param_id = 3  # gzip level 6

    def insert1(self, *args, **kwargs):
        """Require admin privileges to add compression algorithms."""
        from spyglass.common import LabMember

        LabMember().check_admin_privilege(
            "Admin permissions required to add compression algorithms"
        )
        super().insert1(*args, **kwargs)


@schema
class CompressedNwbfile(dj.Manual):
    """Tracks compressed NWB files with transparent decompression.

    Uses externals table for file size and checksum metadata.
    """

    definition = """
    -> Nwbfile                # Foreign key to existing Nwbfile table
    ---
    is_compressed: bool       # Whether file is currently compressed
    is_deleted=0: bool  # Whether original file has been deleted
    compressed_path: varchar(255)      # Path to compressed file (.gz)
    -> CompressionParams               # Compression algorithm used
    compressed_size_bytes: bigint unsigned  # Size of compressed file
    compression_ratio: float            # Ratio (original/compressed)
    compressed_time=CURRENT_TIMESTAMP: timestamp  # When compressed
    """

    _cache = {}  # {nwb_file_name: temp_path}

    def make(self, key, param_id=None):
        """Compress an NWB file (standard DataJoint make function).

        Parameters
        ----------
        key : dict
            Must contain 'nwb_file_name'
        param_id : int, optional
            Compression parameter set ID from CompressionParams
            (default: uses default_param_id)
        """
        nwb_file_name = key["nwb_file_name"]

        # Get original file info from Nwbfile and externals
        if not (Nwbfile & key):
            raise ValueError(f"NWB file not found: {nwb_file_name}")

        original_path = (Nwbfile & key).get_abs_path()

        if not os.path.exists(original_path):
            raise FileNotFoundError(f"File not found: {original_path}")

        # Check if already compressed
        if self & key:
            logger.warning(
                f"{nwb_file_name} already tracked in compression table"
            )
            return

        # Get original size from externals table
        try:
            # Access the external store 'raw' which tracks file metadata
            ext_store = nwbfile_schema.external["raw"]

            # Query the external tracking table for file size
            ext_table = ext_store.tracker
            file_meta = (ext_table & {"filepath": original_path}).fetch1()
            original_size = file_meta["size"]
        except (KeyError, AttributeError, Exception) as e:
            # Fallback to filesystem if externals table query fails
            logger.warning(
                f"Could not fetch size from externals table ({e}), "
                "using filesystem"
            )
            original_size = os.path.getsize(original_path)

        # Get compression parameters
        if param_id is None:
            param_id = CompressionParams().default_param_id

        params_query = CompressionParams & {"param_id": param_id}
        if not params_query:
            raise ValueError(
                f"Parameter ID not found: {param_id}. "
                f"Available IDs: {CompressionParams.fetch('param_id')}"
            )

        params = params_query.fetch1()
        algorithm = params["algorithm"]
        kwargs = params["kwargs"]
        description = params["description"]

        # Get compression algorithm instance
        algo = CompressionAlgorithm.get(algorithm)
        compressed_path = str(original_path) + algo.suffix

        logger.info(
            f"Compressing {nwb_file_name} with {description} "
            f"(param_id={param_id})..."
        )

        # Use _safe_compress to create compressed file atomically
        with _safe_compress(original_path, compressed_path) as temp_path:
            # Compress using the algorithm's compress method with kwargs
            algo.compress(original_path, temp_path, **kwargs)

        compressed_size = os.path.getsize(compressed_path)
        ratio = original_size / compressed_size if compressed_size > 0 else 0

        # Insert into table
        self.insert1(
            {
                "nwb_file_name": nwb_file_name,
                "is_compressed": True,
                "compressed_path": compressed_path,
                "param_id": param_id,
                "compressed_size_bytes": compressed_size,
                "compression_ratio": ratio,
            }
        )

    def decompress(self, nwb_file_name):
        """Decompress file and update is_compressed flag.

        Uses the algorithm-specific decompression function based on the
        compression parameters stored in the table.

        Parameters
        ----------
        nwb_file_name : str
            NWB file name

        Returns
        -------
        str
            Path to original (decompressed) file
        """
        key = {"nwb_file_name": nwb_file_name}

        # Check if in compression table
        if not (self & key):
            logger.warning(f"{nwb_file_name} not in compression table")
            return (Nwbfile & key).get_abs_path()

        metadata = (self & key).fetch1()

        # If already decompressed, return original path
        if not metadata["is_compressed"]:
            logger.info(f"{nwb_file_name} already decompressed")
            return (Nwbfile & key).get_abs_path()

        compressed_path = metadata["compressed_path"]
        original_path = (Nwbfile & key).get_abs_path()
        param_id = metadata["param_id"]

        if not os.path.exists(compressed_path):
            raise FileNotFoundError(
                f"Compressed file not found: {compressed_path}"
            )

        # Get algorithm from params
        params = (CompressionParams & {"param_id": param_id}).fetch1()
        algorithm = params["algorithm"]

        # Get algorithm instance
        algo = CompressionAlgorithm.get(algorithm)

        logger.info(f"Decompressing {nwb_file_name} ({algorithm})...")

        # Decompress and time it
        start_time = time.time()
        algo.decompress(compressed_path, original_path)
        decompress_time_ms = int((time.time() - start_time) * 1000)

        # Update is_compressed flag
        (self & key)._update("is_compressed", False)

        logger.info(f"Decompressed {nwb_file_name} in {decompress_time_ms}ms")

        return original_path

    def delete_files(self, age_days=7, dry_run=True, check_recent_access=False):
        """Delete original files for compressed entries older than specified age.

        Optionally checks AccessLog to avoid deleting recently accessed files.

        Parameters
        ----------
        age_days : int
            Minimum age in days for files to delete (default: 7)
        dry_run : bool
            If True, report what would be deleted without deleting (default: True)
        check_recent_access : bool
            If True, skip files accessed within age_days (default: False)

        Returns
        -------
        list
            List of files deleted (or would be deleted if dry_run=True)
        """
        cutoff_date = datetime.now() - timedelta(days=age_days)

        # Find compressed files older than cutoff and not deleted
        compressed_entries = (
            self
            & "is_compressed = 1"
            & "is_deleted = 0"
            & f'compressed_time < "{cutoff_date}"'
        ).fetch(as_dict=True)

        deleted = []
        skipped = []

        for entry in compressed_entries:
            nwb_file_name = entry["nwb_file_name"]

            # Optionally check if file was accessed recently
            if check_recent_access:
                recent_access = (
                    Nwbfile.AccessLog
                    & {"nwb_file_name": nwb_file_name}
                    & f'access_time >= "{cutoff_date}"'
                )
                if recent_access:
                    skipped.append(nwb_file_name)
                    logger.info(
                        f"Skipping {nwb_file_name} - "
                        f"accessed within last {age_days} days"
                    )
                    continue

            original_path = Nwbfile().get_abs_path(nwb_file_name)

            if os.path.exists(original_path):
                size_gb = os.path.getsize(original_path) / 1e9

                if dry_run:
                    logger.info(
                        f"[DRY RUN] Would delete {nwb_file_name} "
                        f"({size_gb:.2f} GB, compressed {age_days}+ days ago)"
                    )
                    deleted.append(nwb_file_name)
                else:
                    os.unlink(original_path)
                    # Mark as deleted in table
                    (self & {"nwb_file_name": nwb_file_name})._update(
                        "is_deleted", True
                    )
                    logger.info(
                        f"Deleted {nwb_file_name} "
                        f"({size_gb:.2f} GB, compressed {age_days}+ days ago)"
                    )
                    deleted.append(nwb_file_name)

        if dry_run:
            logger.info(
                f"[DRY RUN] Would delete {len(deleted)} files"
                + (
                    f", skipped {len(skipped)} recently accessed"
                    if skipped
                    else ""
                )
                + ", run with dry_run=False to actually delete"
            )
        else:
            logger.info(
                f"Deleted {len(deleted)} original files"
                + (
                    f", skipped {len(skipped)} recently accessed"
                    if skipped
                    else ""
                )
            )

        return deleted

    def get_stats(self):
        """Get compression statistics.

        Retrieves original file sizes from externals table and compressed
        sizes from this table.

        Returns
        -------
        dict
            Summary statistics
        """
        if not self:
            return {
                "total_files": 0,
                "total_original_gb": 0,
                "total_compressed_gb": 0,
                "total_saved_gb": 0,
                "avg_ratio": 0,
                "currently_compressed": 0,
            }

        entries = self.fetch(as_dict=True)

        total_orig = 0
        total_comp = 0
        ratios = []
        currently_compressed = 0

        for entry in entries:
            # Get original size from file system
            nwb_file_name = entry["nwb_file_name"]
            original_path = Nwbfile().get_abs_path(nwb_file_name)

            if os.path.exists(original_path):
                orig_size = os.path.getsize(original_path)
            else:
                # If original deleted, calculate from compressed size and ratio
                orig_size = (
                    entry["compressed_size_bytes"] * entry["compression_ratio"]
                )

            total_orig += orig_size
            total_comp += entry["compressed_size_bytes"]
            ratios.append(entry["compression_ratio"])

            if entry["is_compressed"]:
                currently_compressed += 1

        avg_ratio = sum(ratios) / len(ratios) if ratios else 0

        return {
            "total_files": len(self),
            "total_original_gb": total_orig / 1e9,
            "total_compressed_gb": total_comp / 1e9,
            "total_saved_gb": (total_orig - total_comp) / 1e9,
            "avg_ratio": avg_ratio,
            "currently_compressed": currently_compressed,
        }


# ============================================================================
# Internal Utilities
# ============================================================================


@contextmanager
def _safe_compress(input_path, output_path):
    """Context manager for safe compression with locks and temp files.

    Checksums are managed by DataJoint's externals table.

    Yields the temp file path to write compressed data to.
    """
    output_path = Path(output_path)
    lock_path = Path(str(output_path) + ".lock")

    if lock_path.exists():
        raise RuntimeError(f"Another file lock exists: {lock_path}")

    lock_path.touch()
    temp_fd, temp_output = tempfile.mkstemp(
        suffix=".tmp", dir=output_path.parent
    )
    os.close(temp_fd)
    temp_path = Path(temp_output)

    try:
        # Yield temp path for writing
        yield temp_path

        # Atomically move temp file to final location
        temp_path.rename(output_path)

    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise

    finally:
        if lock_path.exists():
            lock_path.unlink()

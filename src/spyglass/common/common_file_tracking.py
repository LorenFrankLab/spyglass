"""NWB dataset-level compression tracking for Spyglass.

Tracks repacked NWB files that have had HDF5 dataset-level compression
applied via `repack_nwb()`. Dataset-level compression is transparent to
readers — pynwb/h5py decompress only the chunks actually read.
"""

import os
import time

import datajoint as dj

from spyglass.common.common_nwbfile import Nwbfile
from spyglass.utils import logger
from spyglass.utils.compression import repack_nwb

schema = dj.schema("common_file_compression")


@schema
class CompressionParams(dj.Lookup):
    """Supported dataset-level compression algorithms and parameters.

    Maps param_id to HDF5 compression filter settings used by
    `repack_nwb()` and `compressed_data()`.
    """

    definition = """
    param_id: int unsigned auto_increment
    ---
    compression: varchar(16)    # HDF5 filter name (gzip, lzf)
    compression_opts: int unsigned  # Filter level (gzip: 1-9, lzf: 0)
    description: varchar(127)   # Human-readable description
    """

    contents = [
        (1, "gzip", 1, "Gzip level 1 (fastest)"),
        (2, "gzip", 4, "Gzip level 4 (balanced)"),
        (3, "gzip", 6, "Gzip level 6 (default)"),
        (4, "gzip", 9, "Gzip level 9 (best ratio)"),
        (5, "lzf", 0, "LZF (fast, moderate ratio)"),
    ]

    default_param_id = 2  # gzip level 4

    def insert1(self, *args, **kwargs):
        """Require admin privileges to add compression parameters."""
        from spyglass.common import LabMember

        LabMember().check_admin_privilege(
            "Admin permissions required to add compression parameters"
        )
        super().insert1(*args, **kwargs)


@schema
class CompressedNwbfile(dj.Manual):
    """Tracks NWB files repacked with dataset-level compression.

    Each entry records that an NWB file has been repacked in-place,
    replacing the original with a version where large datasets have
    HDF5 compression applied. The repacked file is readable by pynwb
    without any special handling.
    """

    definition = """
    -> Nwbfile
    ---
    is_repacked: bool                      # Whether file is currently repacked
    -> CompressionParams
    original_size_bytes: bigint unsigned    # Size before repacking
    repacked_size_bytes: bigint unsigned    # Size after repacking
    compression_ratio: float               # original / repacked
    datasets_compressed: int unsigned       # Number of datasets compressed
    datasets_skipped: int unsigned          # Number of datasets skipped
    repacked_time=CURRENT_TIMESTAMP: timestamp  # When repacked
    """

    def make(self, key, param_id=None):
        """Repack an NWB file with dataset-level compression.

        Parameters
        ----------
        key : dict
            Must contain 'nwb_file_name'.
        param_id : int, optional
            Compression parameter set ID from CompressionParams.
            Default: CompressionParams.default_param_id.
        """
        nwb_file_name = key["nwb_file_name"]

        if not (Nwbfile & key):
            raise ValueError(f"NWB file not found: {nwb_file_name}")

        original_path = (Nwbfile & key).get_abs_path()

        if not os.path.exists(original_path):
            raise FileNotFoundError(f"File not found: {original_path}")

        if self & key:
            logger.warning(
                f"{nwb_file_name} already tracked in compression table"
            )
            return

        if param_id is None:
            param_id = CompressionParams().default_param_id

        params_query = CompressionParams & {"param_id": param_id}
        if not params_query:
            raise ValueError(
                f"Parameter ID not found: {param_id}. "
                f"Available: {CompressionParams.fetch('param_id')}"
            )

        params = params_query.fetch1()
        compression = params["compression"]
        compression_opts = params["compression_opts"]
        description = params["description"]

        logger.info(f"Repacking {nwb_file_name} with {description}...")

        start = time.time()
        stats = repack_nwb(
            input_path=original_path,
            compression=compression,
            compression_opts=compression_opts,
            verify=True,
        )
        elapsed = time.time() - start

        logger.info(
            f"Repacked {nwb_file_name} in {elapsed:.1f}s: "
            f"{stats['compression_ratio']:.2f}x ratio, "
            f"{stats['datasets_compressed']} datasets compressed"
        )

        self.insert1(
            {
                "nwb_file_name": nwb_file_name,
                "is_repacked": True,
                "param_id": param_id,
                "original_size_bytes": stats["original_size"],
                "repacked_size_bytes": stats["repacked_size"],
                "compression_ratio": stats["compression_ratio"],
                "datasets_compressed": stats["datasets_compressed"],
                "datasets_skipped": stats["datasets_skipped"],
            }
        )

    def get_stats(self):
        """Get compression statistics across all tracked files.

        Returns
        -------
        dict
            Summary: total_files, total_original_gb, total_repacked_gb,
            total_saved_gb, avg_ratio, currently_repacked.
        """
        if not self:
            return {
                "total_files": 0,
                "total_original_gb": 0,
                "total_repacked_gb": 0,
                "total_saved_gb": 0,
                "avg_ratio": 0,
                "currently_repacked": 0,
            }

        entries = self.fetch(as_dict=True)

        total_orig = sum(e["original_size_bytes"] for e in entries)
        total_repack = sum(e["repacked_size_bytes"] for e in entries)
        ratios = [e["compression_ratio"] for e in entries]
        currently_repacked = sum(1 for e in entries if e["is_repacked"])

        return {
            "total_files": len(entries),
            "total_original_gb": total_orig / 1e9,
            "total_repacked_gb": total_repack / 1e9,
            "total_saved_gb": (total_orig - total_repack) / 1e9,
            "avg_ratio": sum(ratios) / len(ratios),
            "currently_repacked": currently_repacked,
        }


@schema
class NwbRepackJob(dj.Manual):
    """Tracks archive-wide compression campaign progress.

    Temporary table for coordinating batch repacking of existing
    NWB files across multiple workers.
    """

    definition = """
    -> Nwbfile
    ---
    status: enum('pending','in_progress','completed','failed','skipped')
    original_size_bytes=null: bigint unsigned
    repacked_size_bytes=null: bigint unsigned
    compression_ratio=null: float
    error_message='': varchar(512)
    started_at=null: timestamp
    completed_at=null: timestamp
    worker_id='': varchar(64)
    """

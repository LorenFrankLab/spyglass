"""Dataset-level compression tracking for Spyglass NWB files.

Tracks NWB files repacked in-place via:

    h5repack -f GZIP={level} -m 1  src.nwb  tmp.nwb

The -m 1 flag is required — without it, zero-shape datasets (video
placeholders) are silently dropped. h5repack -f NONE returns rc=1
even on success; verification gates on pynwb readability and dataset
count, not on rc.
"""

import os
import subprocess
import tempfile
import warnings
from pathlib import Path

import datajoint as dj
import h5py

from spyglass.common.common_nwbfile import Nwbfile
from spyglass.utils import logger

schema = dj.schema("common_file_compression")


@schema
class CompressionParams(dj.Lookup):
    """h5repack compression parameters.

    Each row maps param_id to the GZIP level passed to
    `h5repack -f GZIP={level} -m 1`. Only gzip is included;
    lzf support in h5repack is not universally available.
    """

    definition = """
    param_id: int unsigned
    ---
    level: tinyint unsigned   # GZIP level (1-9)
    description: varchar(64)
    """

    contents = [
        (1, 1, "GZIP level 1 — fastest, lowest ratio"),
        (2, 4, "GZIP level 4 — balanced"),
        (3, 6, "GZIP level 6 — benchmarked default"),
        (4, 9, "GZIP level 9 — best ratio, slowest"),
    ]

    default_param_id = 3  # level 6 — matches benchmarks


@schema
class CompressedNwbfile(dj.Manual):
    """Tracks NWB files repacked with h5repack dataset-level compression.

    After repacking, the file is readable by pynwb/h5py without any
    decompression step; only the chunks actually accessed are decompressed.
    """

    definition = """
    -> Nwbfile
    ---
    -> CompressionParams
    original_size_bytes: bigint unsigned
    repacked_size_bytes: bigint unsigned
    compression_ratio: float
    datasets_total: int unsigned
    datasets_compressed: int unsigned   # datasets that gained compression
    repacked_at=CURRENT_TIMESTAMP: timestamp
    """

    def make(self, key, param_id=None):
        """Repack a raw NWB file with h5repack and record the result.

        Parameters
        ----------
        key : dict
            Must contain 'nwb_file_name'.
        param_id : int, optional
            Row from CompressionParams. Default: default_param_id (level 6).
        """
        if self & key:
            logger.warning(
                f"{key['nwb_file_name']} already in CompressedNwbfile"
            )
            return

        if param_id is None:
            param_id = CompressionParams.default_param_id

        params = (CompressionParams & {"param_id": param_id}).fetch1()
        level = params["level"]

        src = Path(Nwbfile.get_abs_path(key["nwb_file_name"]))
        if not src.exists():
            raise FileNotFoundError(src)

        original_size = src.stat().st_size
        n_datasets = _count_datasets(src)

        tmp_fd, tmp_str = tempfile.mkstemp(suffix=".nwb.tmp", dir=src.parent)
        os.close(tmp_fd)
        tmp = Path(tmp_str)

        try:
            _h5repack(src, tmp, level)
            _verify(src, tmp, n_datasets)
            n_compressed = _count_compressed_datasets(tmp)
            repacked_size = tmp.stat().st_size
            tmp.rename(src)  # atomic on POSIX within same filesystem
        except Exception:
            if tmp.exists():
                tmp.unlink()
            raise

        # Update DataJoint externals table with new hash and size
        from spyglass.utils.dj_helper_fn import _resolve_external_table

        _resolve_external_table(str(src), src.name, location="raw")

        ratio = original_size / repacked_size if repacked_size else 0.0
        self.insert1(
            {
                **key,
                "param_id": param_id,
                "original_size_bytes": original_size,
                "repacked_size_bytes": repacked_size,
                "compression_ratio": round(ratio, 4),
                "datasets_total": n_datasets,
                "datasets_compressed": n_compressed,
            }
        )

        logger.info(
            f"Repacked {src.name}: {ratio:.2f}x ratio, "
            f"{n_compressed}/{n_datasets} datasets compressed"
        )


# ---------------------------------------------------------------------------
# h5repack helpers
# ---------------------------------------------------------------------------


def _h5repack(src: Path, dst: Path, level: int) -> None:
    """Run h5repack -f GZIP={level} -m 1 src dst.

    Raises subprocess.CalledProcessError if rc != 0.
    -m 1 preserves zero-shape datasets that would otherwise be
    silently dropped.
    """
    result = subprocess.run(
        ["h5repack", "-f", f"GZIP={level}", "-m", "1", str(src), str(dst)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            result.args,
            result.stdout,
            result.stderr,
        )


def _verify(src: Path, dst: Path, expected_count: int) -> None:
    """Verify dst is a valid repacked copy of src.

    Checks:
    1. Dataset count matches (detects silent drops).
    2. pynwb can open and read the file (warnings tolerated, errors fail).

    Raises ValueError if either check fails.
    """
    n_dst = _count_datasets(dst)
    if n_dst != expected_count:
        raise ValueError(
            f"Dataset count mismatch: src={expected_count} dst={n_dst}"
        )

    from pynwb import NWBHDF5IO

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with NWBHDF5IO(str(dst), "r", load_namespaces=True) as io:
                io.read()
    except Exception as exc:
        raise ValueError(f"Repacked file not readable by pynwb: {exc}") from exc


def _count_datasets(path: Path) -> int:
    """Count all HDF5 datasets in a file."""
    n = [0]
    with h5py.File(str(path), "r") as f:
        f.visititems(
            lambda _, obj: (
                n.__setitem__(0, n[0] + 1)
                if isinstance(obj, h5py.Dataset)
                else None
            )
        )
    return n[0]


def _count_compressed_datasets(path: Path) -> int:
    """Count datasets that have a compression filter applied."""
    n = [0]
    with h5py.File(str(path), "r") as f:
        f.visititems(
            lambda _, obj: (
                n.__setitem__(0, n[0] + 1)
                if isinstance(obj, h5py.Dataset) and obj.compression
                else None
            )
        )
    return n[0]

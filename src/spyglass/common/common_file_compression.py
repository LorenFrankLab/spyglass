"""Dataset-level compression tracking for Spyglass NWB files.

Tracks NWB files repacked in-place via:

    h5repack -f GZIP=6 -m 1  src.nwb  tmp.nwb

The -m 1 flag is required — without it, zero-shape datasets (video
placeholders) are silently dropped. h5repack -f NONE returns rc=1
even on success; verification gates on pynwb readability and dataset
count, not on rc.

Nwbfile always stores the copy ({session}_.nwb). CompressedNwbfile is
keyed by Nwbfile and has a File part table with one row per physical
file — 'copy' ({session}_.nwb) and 'raw' ({session}.nwb).
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

_DEFAULT_GZIP_LEVEL = 6  # benchmarked default


@schema
class CompressedNwbfile(dj.Manual):
    """Session-level record that both NWB files in a pair have been repacked.

    Keyed by Nwbfile (always the copy, {session}_.nwb). The File part
    table holds per-file stats for both 'copy' and 'raw'.
    """

    definition = """
    -> Nwbfile
    ---
    repacked_at=CURRENT_TIMESTAMP: timestamp
    """

    class File(dj.Part):
        """Per-file compression stats for one file in the session pair."""

        definition = """
        -> master
        file_role: enum('copy', 'raw')
        ---
        gzip_level: tinyint unsigned # level passed to h5repack; 0=pre-existing
        original_size_bytes: bigint unsigned
        repacked_size_bytes: bigint unsigned
        compression_ratio: float
        datasets_total: int unsigned
        datasets_compressed: int unsigned
        """

    def make(self, key, gzip_level=_DEFAULT_GZIP_LEVEL):
        """Repack both files in a session pair and record results.

        Requires admin privileges — modifies source files in place.

        Parameters
        ----------
        key : dict
            Must contain 'nwb_file_name' for the copy ({session}_.nwb).
        gzip_level : int, optional
            GZIP compression level (1-9). Default: 6 (benchmarked).
        """
        from spyglass.common import LabMember

        LabMember().check_admin_privilege(
            "Admin privileges required to repack NWB files"
        )

        if self & key:
            logger.warning(
                f"{key['nwb_file_name']} already in CompressedNwbfile"
            )
            return

        copy_name = key["nwb_file_name"]
        raw_name = copy_name[:-5] + ".nwb"  # strip trailing _ before .nwb

        copy_stats = _repack_file(Nwbfile.get_abs_path(copy_name), gzip_level)
        raw_stats = _repack_file(Nwbfile.get_abs_path(raw_name), gzip_level)

        self.insert1(key)
        self.File.insert(
            [
                {**key, "file_role": "copy", **copy_stats},
                {**key, "file_role": "raw", **raw_stats},
            ]
        )

        logger.info(
            f"Repacked {copy_name}: "
            f"copy {copy_stats['compression_ratio']:.2f}x, "
            f"raw {raw_stats['compression_ratio']:.2f}x"
        )


# ---------------------------------------------------------------------------
# h5repack helpers
# ---------------------------------------------------------------------------


def _repack_file(path: str, gzip_level: int) -> dict:
    """Repack a single file and return per-file stats for File part table.

    If the file is already fully compressed, records it without repacking
    (gzip_level=0 in the returned stats).

    Parameters
    ----------
    path : str
        Absolute path to the NWB file.
    gzip_level : int
        GZIP level to pass to h5repack.

    Returns
    -------
    dict
        Keys: gzip_level, original_size_bytes, repacked_size_bytes,
        compression_ratio, datasets_total, datasets_compressed.
    """
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(src)

    original_size = src.stat().st_size
    n_datasets = _count_datasets(path=src)
    n_already_compressed = _count_compressed_datasets(path=src)

    shared = {
        "original_size_bytes": original_size,
        "datasets_total": n_datasets,
    }

    if n_already_compressed == n_datasets:
        logger.info(
            f"{src.name} already fully compressed, recording without repacking"
        )
        return {
            "gzip_level": 0,
            "repacked_size_bytes": original_size,
            "compression_ratio": 1.0,
            "datasets_compressed": n_already_compressed,
            **shared,
        }

    tmp_fd, tmp_str = tempfile.mkstemp(suffix=".nwb.tmp", dir=src.parent)
    os.close(tmp_fd)
    tmp = Path(tmp_str)

    try:
        _h5repack(src=src, dst=tmp, level=gzip_level)
        _verify(src=src, dst=tmp, expected_count=n_datasets)
        n_compressed = _count_compressed_datasets(path=tmp)
        repacked_size = tmp.stat().st_size
        tmp.rename(src)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise

    from spyglass.utils.dj_helper_fn import _resolve_external_table

    _resolve_external_table(
        filepath=str(src), file_name=src.name, location="raw"
    )

    ratio = original_size / repacked_size if repacked_size else 0.0

    return {
        "gzip_level": gzip_level,
        "repacked_size_bytes": repacked_size,
        "compression_ratio": round(ratio, 4),
        "datasets_compressed": n_compressed,
        **shared,
    }


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

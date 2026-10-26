"""Dataset-level compression tracking for Spyglass NWB files.

Tracks NWB files repacked in-place via:

    h5repack -f GZIP=6 -m 1  src.nwb  tmp.nwb

The -m 1 flag is required — without it, zero-shape datasets (video
placeholders) are silently dropped. A repack is accepted only when the
source and the repacked file hash identically.

Nwbfile always stores the copy ({session}_.nwb). CompressedNwbfile
holds one row per physical file — 'copy' ({session}_.nwb) and 'raw'
({session}.nwb).
"""

import os
import subprocess
import tempfile
from pathlib import Path
from shutil import which

import datajoint as dj
import h5py

from spyglass.common.common_nwbfile import Nwbfile
from spyglass.utils import SpyglassMixin, logger
from spyglass.utils.nwb_hash import NwbfileHasher

schema = dj.schema("common_file_compression")

_DEFAULT_GZIP_LEVEL = 6  # benchmarked default


@schema
class CompressedNwbfile(SpyglassMixin, dj.Computed):
    """Per-file record of an in-place repack.

    One row per physical file: the 'copy' ({session}_.nwb) that Nwbfile
    keys on, and the 'raw' ({session}.nwb) beside it.
    """

    definition = """
    -> Nwbfile
    file_role: enum('copy', 'raw')
    ---
    repacked_at=CURRENT_TIMESTAMP: timestamp
    gzip_level=null: tinyint unsigned # null if already compressed, not repacked
    original_size_bytes: bigint unsigned
    repacked_size_bytes: bigint unsigned
    """

    @property
    def compression_ratio(self):
        """Per-file ratio of original to repacked size."""
        return self.proj(
            compression_ratio="original_size_bytes / repacked_size_bytes"
        )

    @property
    def space_saved(self):
        """Bytes before and after, summed over the restricted files."""
        return dj.U().aggr(
            self,
            original_bytes="SUM(original_size_bytes)",
            repacked_bytes="SUM(repacked_size_bytes)",
        )

    def populate(self, *restrictions, **kwargs):
        """Repack the files of each restricted session.

        Parameters
        ----------
        *restrictions : Any
            At least one restriction is required. Repacking rewrites files
            in place, so an unrestricted call would sweep the whole corpus.
        **kwargs : Any
            Passed to ``dj.Computed.populate``.
        """
        if not restrictions:
            raise ValueError(
                "CompressedNwbfile.populate requires a restriction. Repacking "
                + "rewrites files in place; an unrestricted call would sweep "
                + "every session."
            )
        return super().populate(*restrictions, **kwargs)

    def make_fetch(self, key) -> list:
        """Resolve the paths of the session's two NWB files.

        Parameters
        ----------
        key : dict
            Primary key of Nwbfile, naming the copy ({session}_.nwb).

        Returns
        -------
        list
            Single-item list holding {file_role: absolute path}.
        """
        from spyglass.common import LabMember

        LabMember().check_admin_privilege(
            "Admin privileges required to repack NWB files"
        )

        copy_name = key["nwb_file_name"]
        raw_name = copy_name[:-5] + ".nwb"  # strip trailing _ before .nwb

        return [
            {
                "copy": Nwbfile.get_abs_path(copy_name),
                "raw": Nwbfile.get_abs_path(raw_name),
            }
        ]

    def make_compute(self, key, paths: dict) -> list:
        """Repack each file to a temporary sibling and verify it.

        Runs outside the insert transaction: a repack takes hours on a large
        file, and nothing is moved into place until every file has passed.

        Parameters
        ----------
        key : dict
            Primary key of Nwbfile.
        paths : dict
            {file_role: absolute path}, from make_fetch.

        Returns
        -------
        list
            Single-item list holding {file_role: (tmp or None, stats)}.
        """
        staged = {}
        try:
            for role, path in paths.items():
                staged[role] = _stage_repack(Path(path), _DEFAULT_GZIP_LEVEL)
        except Exception:
            for tmp, _ in staged.values():
                if tmp is not None:
                    tmp.unlink(missing_ok=True)
            raise

        return [staged]

    def make_insert(self, key, staged: dict) -> None:
        """Move each repacked file into place and record what it saved.

        Parameters
        ----------
        key : dict
            Primary key of Nwbfile.
        staged : dict
            {file_role: (tmp or None, stats)}, from make_compute.
        """
        rows = []
        for role, (tmp, stats) in staged.items():
            if tmp is not None:
                _commit_repack(tmp=tmp, src=Path(stats.pop("_path")))
            else:
                stats.pop("_path")
            rows.append({**key, "file_role": role, **stats})

        self.insert(rows)

        saved = sum(
            r["original_size_bytes"] - r["repacked_size_bytes"] for r in rows
        )
        logger.info(
            f"Repacked {key['nwb_file_name']}: {saved / 1e9:.2f} GB saved"
        )


# ---------------------------------------------------------------------------
# h5repack helpers
# ---------------------------------------------------------------------------


def _stage_repack(src: Path, gzip_level: int) -> tuple:
    """Repack a file to a verified temporary sibling, leaving it in place.

    Nothing is moved over the source; see ``_commit_repack``.

    Parameters
    ----------
    src : Path
        NWB file to repack.
    gzip_level : int
        GZIP level to pass to h5repack.

    Returns
    -------
    tuple
        (tmp, stats). ``tmp`` is None when the file is already compressed
        and nothing was repacked. ``stats`` carries the row's size columns
        plus ``_path``, the source to move onto.
    """
    if not src.exists():
        raise FileNotFoundError(src)

    original_size = src.stat().st_size
    n_datasets, n_compressed = _dataset_counts(src)

    if n_compressed == n_datasets:
        logger.info(
            f"{src.name} already fully compressed, recording without repacking"
        )
        return None, {
            "_path": str(src),
            "gzip_level": None,
            "original_size_bytes": original_size,
            "repacked_size_bytes": original_size,
        }

    tmp_fd, tmp_str = tempfile.mkstemp(suffix=".nwb.tmp", dir=src.parent)
    os.close(tmp_fd)
    tmp = Path(tmp_str)

    try:
        _h5repack(src=src, dst=tmp, level=gzip_level)
        _verify(src=src, dst=tmp)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    return tmp, {
        "_path": str(src),
        "gzip_level": gzip_level,
        "original_size_bytes": original_size,
        "repacked_size_bytes": tmp.stat().st_size,
    }


def _commit_repack(tmp: Path, src: Path) -> None:
    """Move a staged repack over its source and refresh the stored checksum.

    Parameters
    ----------
    tmp : Path
        Verified repack output, from ``_stage_repack``.
    src : Path
        File to replace.
    """
    from spyglass.utils.dj_helper_fn import _resolve_external_table

    _replace_preserving_mode(tmp=tmp, src=src)
    _resolve_external_table(
        filepath=str(src), file_name=src.name, location="raw"
    )


def _replace_preserving_mode(tmp: Path, src: Path) -> None:
    """Move a repacked file over its source, keeping the source's mode.

    Parameters
    ----------
    tmp : Path
        Repacked file to move into place.
    src : Path
        File to replace. Must exist.
    """
    # mkstemp creates the repack output 0600, so a plain rename would leave a
    # raw file on shared storage readable only by whoever ran the repack.
    os.chmod(tmp, src.stat().st_mode)
    tmp.rename(src)


def _h5repack(src: Path, dst: Path, level: int) -> None:
    """Run h5repack -f GZIP={level} -m 1 src dst.

    Raises subprocess.CalledProcessError if rc != 0.
    -m 1 preserves zero-shape datasets that would otherwise be
    silently dropped.
    """
    if which("h5repack") is None:
        raise RuntimeError(
            "h5repack not found on PATH. It ships with the HDF5 tools: "
            "`conda install -c conda-forge hdf5` or `apt install hdf5-tools`."
        )

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


def _verify(src: Path, dst: Path) -> None:
    """Raise if a repacked file does not match its source.

    Parameters
    ----------
    src : Path
        Source file, before repacking.
    dst : Path
        Repacked file.

    Raises
    ------
    ValueError
        If the two files do not hash identically.
    """
    # Default precision, so a repack is held to the same standard as recompute.
    src_hash = NwbfileHasher(src).hash
    dst_hash = NwbfileHasher(dst).hash

    if src_hash != dst_hash:
        raise ValueError(
            f"Repacked file does not match its source: {src.name}\n"
            + f"\tsrc: {src_hash}\n"
            + f"\tdst: {dst_hash}"
        )


def _dataset_counts(path: Path) -> tuple[int, int]:
    """Count a file's datasets, and how many carry a compression filter.

    Parameters
    ----------
    path : Path
        HDF5 file to walk.

    Returns
    -------
    tuple[int, int]
        Total dataset count, and the count with a compression filter.
    """
    total = compressed = 0

    def visit(_, obj):
        nonlocal total, compressed
        if isinstance(obj, h5py.Dataset):
            total += 1
            compressed += bool(obj.compression)

    with h5py.File(str(path), "r") as f:
        f.visititems(visit)

    return total, compressed

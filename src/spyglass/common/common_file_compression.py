"""Dataset-level compression tracking for Spyglass NWB files.

Repacks files in place via `h5repack`, with the filters named by
`h5py_helper_fn.GZIP_LEVEL` and `SHUFFLE`. A repack is accepted only when
the source and the repacked file hash identically.

-m 1 preserves zero-shape datasets, such as video placeholders.

Nwbfile keys on the copy ({session}_.nwb); CompressedNwbfile holds a row for
it and one for the raw file ({session}.nwb) beside it.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from shutil import disk_usage, which

import datajoint as dj
import h5py

from spyglass.common.common_nwbfile import Nwbfile
from spyglass.utils import SpyglassMixin, logger
from spyglass.utils.dj_helper_fn import bytes_to_human_readable
from spyglass.utils.h5py_helper_fn import GZIP_LEVEL, SHUFFLE
from spyglass.utils.nwb_hash import NwbfileHasher
from spyglass.utils.nwb_helper_fn import get_nwb_raw_filename

schema = dj.schema("common_file_compression")


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

        return [
            {
                "copy": Nwbfile.get_abs_path(copy_name),
                "raw": Nwbfile.get_abs_path(get_nwb_raw_filename(copy_name)),
            }
        ]

    def make_compute(self, key, paths: dict) -> list:
        """Repack each file to a temporary sibling and verify it.

        Runs outside the insert transaction. Nothing is moved into place
        until every file of the session has passed.

        Parameters
        ----------
        key : dict
            Primary key of Nwbfile.
        paths : dict
            {file_role: absolute path}, from make_fetch.

        Returns
        -------
        list
            Single-item list holding {file_role: (tmp or None, src, stats)}.
        """
        staged = {}
        try:
            for role, path in paths.items():
                staged[role] = _stage_repack(Path(path), GZIP_LEVEL)
        except Exception:
            for tmp, _, _ in staged.values():
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
            {file_role: (tmp or None, src, stats)}, from make_compute.
        """
        rows = []
        for role, (tmp, src, stats) in staged.items():
            if tmp is not None:
                _commit_repack(tmp=tmp, src=src)
            rows.append({**key, "file_role": role, **stats})

        self.insert(rows)

        saved = sum(
            r["original_size_bytes"] - r["repacked_size_bytes"] for r in rows
        )
        logger.info(
            f"Repacked {key['nwb_file_name']}: "
            + f"{bytes_to_human_readable(saved)} saved"
        )


# ---------------------------------------------------------------------------
# h5repack helpers
# ---------------------------------------------------------------------------


def _assert_space_to_stage(src: Path) -> None:
    """Raise if the source's directory cannot hold a second copy of it.

    A repack writes its output beside the source before replacing it. The
    worst case, a file that does not compress, needs the source's size again.

    Parameters
    ----------
    src : Path
        File about to be repacked.

    Raises
    ------
    RuntimeError
        If free space is below the source's size.
    """
    needed = src.stat().st_size
    free = disk_usage(src.parent).free

    if free < needed:
        raise RuntimeError(
            f"Not enough space to repack {src.name}: needs "
            + f"{bytes_to_human_readable(needed)}, "
            + f"{bytes_to_human_readable(free)} free in {src.parent}"
        )


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
        (tmp, src, stats). ``tmp`` is None when the file is already
        compressed and nothing was repacked; ``stats`` holds the row's
        size columns.
    """
    if not src.exists():
        raise FileNotFoundError(src)

    original_size = src.stat().st_size

    if not _has_uncompressed_dataset(src):
        logger.info(
            f"{src.name} already fully compressed, recording without repacking"
        )
        return (
            None,
            src,
            {
                "gzip_level": None,
                "original_size_bytes": original_size,
                "repacked_size_bytes": original_size,
            },
        )

    _assert_space_to_stage(src)

    tmp_fd, tmp_str = tempfile.mkstemp(suffix=".nwb.tmp", dir=src.parent)
    os.close(tmp_fd)
    tmp = Path(tmp_str)

    try:
        _h5repack(src=src, dst=tmp, level=gzip_level)
        _verify(src=src, dst=tmp)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    return (
        tmp,
        src,
        {
            "gzip_level": gzip_level,
            "original_size_bytes": original_size,
            "repacked_size_bytes": tmp.stat().st_size,
        },
    )


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
    # mkstemp creates the output 0600, not the source's mode.
    os.chmod(tmp, src.stat().st_mode)
    tmp.rename(src)


def _h5repack(src: Path, dst: Path, level: int) -> None:
    """Run h5repack over src into dst, with the configured filters.

    Raises subprocess.CalledProcessError if rc != 0.
    -m 1 preserves zero-shape datasets.
    """
    if which("h5repack") is None:
        raise RuntimeError(
            "h5repack not found on PATH. It ships with the HDF5 tools: "
            "`conda install -c conda-forge hdf5` or `apt install hdf5-tools`."
        )

    # SHUF must precede GZIP: h5repack applies filters in the order given.
    filters = ["-f", "SHUF"] if SHUFFLE else []
    filters += ["-f", f"GZIP={level}"]

    result = subprocess.run(
        ["h5repack", *filters, "-m", "1", str(src), str(dst)],
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


def _has_uncompressed_dataset(path: Path) -> bool:
    """Whether any dataset in a file lacks a compression filter.

    Parameters
    ----------
    path : Path
        HDF5 file to walk.

    Returns
    -------
    bool
        True if at least one dataset is uncompressed. False for a file with
        no datasets at all.
    """

    def visit(_, obj):
        if isinstance(obj, h5py.Dataset) and not obj.compression:
            return True  # a non-None return stops the walk

    with h5py.File(str(path), "r") as f:
        return bool(f.visititems(visit))

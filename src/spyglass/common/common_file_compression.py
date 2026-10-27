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
from dataclasses import dataclass
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

        # Only the copy is registered in Nwbfile. stage() raises if the
        # raw file is not on disk.
        return [
            {
                "copy": Nwbfile.get_abs_path(copy_name),
                "raw": Nwbfile.get_abs_path(
                    get_nwb_raw_filename(copy_name), new_file=True
                ),
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
            Single-item list holding {file_role: _StagedRepack}.
        """
        staged = {}
        try:
            for role, path in paths.items():
                staged[role] = _StagedRepack.stage(Path(path), GZIP_LEVEL)
        except Exception:
            for item in staged.values():
                item.discard()
            raise

        return [staged]

    def make_insert(self, key, staged: dict) -> None:
        """Move each repacked file into place and record what it saved.

        Parameters
        ----------
        key : dict
            Primary key of Nwbfile.
        staged : dict
            {file_role: _StagedRepack}, from make_compute.

        Notes
        -----
        Each original is set aside until every file is in place and the rows
        are inserted. A failure in between puts them all back.
        """
        rows = []
        try:
            for role, item in staged.items():
                item.commit()
                rows.append({**key, "file_role": role, **item.stats})

            self.insert(rows)
        except Exception:
            for item in reversed(list(staged.values())):
                item.restore()
            raise

        for item in staged.values():
            item.release()

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


def _fingerprint(src: Path) -> tuple:
    """Identify a file's current contents cheaply.

    Parameters
    ----------
    src : Path
        File to describe.

    Returns
    -------
    tuple
        (inode, size, mtime_ns). Any ordinary write changes at least one.
    """
    st = src.stat()
    return (st.st_ino, st.st_size, st.st_mtime_ns)


@dataclass
class _StagedRepack:
    """One file repacked beside itself, waiting to be moved into place.

    Every method is a no-op for a file that needed no repacking.
    """

    src: Path
    stats: dict
    tmp: Path = None  # None when every eligible dataset was compressed
    staged_at: tuple = None
    kept: Path = None  # the original, once commit has set it aside

    @classmethod
    def stage(cls, src: Path, gzip_level: int) -> "_StagedRepack":
        """Repack a file to a verified sibling, leaving the source in place.

        Parameters
        ----------
        src : Path
            NWB file to repack.
        gzip_level : int
            GZIP level to pass to h5repack.

        Returns
        -------
        _StagedRepack
            With ``tmp`` None when the file needed no repacking.
        """
        if not src.exists():
            raise FileNotFoundError(src)

        original_size = src.stat().st_size

        if not _has_uncompressed_dataset(src):
            logger.info(
                f"{src.name} already fully compressed, "
                "recording without repacking"
            )
            return cls(
                src=src,
                staged_at=_fingerprint(src),
                stats={
                    "gzip_level": None,
                    "original_size_bytes": original_size,
                    "repacked_size_bytes": original_size,
                },
            )

        _assert_space_to_stage(src)

        # One staging path per source, so a repack refused for a fixable
        # reason survives here for the operator to finish by hand.
        tmp = src.with_name(src.name + ".repack.tmp")
        tmp.unlink(missing_ok=True)

        try:
            _h5repack(src=src, dst=tmp, level=gzip_level)
            _verify(src=src, dst=tmp)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

        return cls(
            src=src,
            tmp=tmp,
            staged_at=_fingerprint(src),
            stats={
                "gzip_level": gzip_level,
                "original_size_bytes": original_size,
                "repacked_size_bytes": tmp.stat().st_size,
            },
        )

    def discard(self) -> None:
        """Remove the staged file, before anything has been moved."""
        if self.tmp is not None:
            self.tmp.unlink(missing_ok=True)

    def commit(self) -> None:
        """Set the original aside and put the repack in its place.

        Raises
        ------
        RuntimeError
            If the source changed since staging. The staged file describes
            the older contents, so replacing would discard the newer write.
        """
        from spyglass.utils.dj_helper_fn import _resolve_external_table

        if self.tmp is None:
            return

        # Hours pass between staging and here, and writers such as
        # make_file_obj_id_unique touch the source.
        if _fingerprint(self.src) != self.staged_at:
            raise RuntimeError(
                f"{self.src.name} changed while it was being repacked, so "
                "the staged copy is stale and was not used. Left "
                f"unchanged.\n"
                f"\tat staging: {self.staged_at}\n"
                f"\tnow:        {_fingerprint(self.src)}\n"
                "The next repack rebuilds it from the current contents."
            )

        stat_src = self.src.stat()  # read before the move; src will not be

        kept = self.src.with_name(self.src.name + ".orig")
        kept.unlink(missing_ok=True)
        self.src.rename(kept)
        self.kept = kept

        try:
            _replace_preserving_mode(
                tmp=self.tmp, src=self.src, stat_src=stat_src
            )
            _resolve_external_table(
                filepath=str(self.src),
                file_name=self.src.name,
                location="raw",
            )
        except Exception:
            self.restore()
            raise

    def restore(self) -> None:
        """Put the set-aside original back, undoing ``commit``."""
        if self.kept is None:
            return
        _restore_original(kept=self.kept, src=self.src)
        self.kept = None

    def release(self) -> None:
        """Drop the set-aside original, once its row is inserted."""
        if self.kept is None:
            return
        self.kept.unlink(missing_ok=True)
        self.kept = None


def _restore_original(kept: Path, src: Path) -> None:
    """Put a set-aside original back, replacing anything at its path.

    Parameters
    ----------
    kept : Path
        The ``{source}.orig`` written by ``commit``.
    src : Path
        Path the original came from.
    """
    if not kept.exists():
        return
    src.unlink(missing_ok=True)
    kept.rename(src)


def _replace_preserving_mode(
    tmp: Path, src: Path, stat_src: os.stat_result = None
) -> None:
    """Move a repacked file over its source, keeping mode and ownership.

    Parameters
    ----------
    tmp : Path
        Repacked file to move into place.
    src : Path
        Path to replace.
    stat_src : os.stat_result, optional
        The source's attributes, for a caller that has already moved it
        aside. Read from ``src`` when omitted.

    Raises
    ------
    PermissionError
        If the source belongs to another user or group and this process
        cannot reassign the replacement to match. The source is left in
        place and the repacked file kept for the operator to finish.
    """
    stat_src = stat_src or src.stat()

    # The staged file belongs to whoever ran the repack, and only root may
    # give a file away.
    if (os.geteuid(), os.getegid()) != (stat_src.st_uid, stat_src.st_gid):
        try:
            os.chown(tmp, stat_src.st_uid, stat_src.st_gid)
        except PermissionError as err:
            raise PermissionError(
                f"Cannot preserve ownership of {src.name}: owned by "
                f"uid={stat_src.st_uid} gid={stat_src.st_gid}, running as "
                f"uid={os.geteuid()} gid={os.getegid()}.\n"
                f"The verified repack is kept at {tmp}. Finish it with:\n"
                f"    sudo chown {stat_src.st_uid}:{stat_src.st_gid} {tmp}\n"
                f"    sudo chmod {stat_src.st_mode & 0o7777:o} {tmp}\n"
                f"    sudo mv {tmp} {src}\n"
                "Re-running the repack discards it and starts over."
            ) from err

    # The staged file is created 0600, not the source's mode.
    os.chmod(tmp, stat_src.st_mode)
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

    Scalars and zero-length datasets are ignored: a filter needs a chunked
    layout, so they cannot carry one and h5repack -m 1 leaves them alone.

    Parameters
    ----------
    path : Path
        HDF5 file to walk.

    Returns
    -------
    bool
        True if at least one eligible dataset is uncompressed. False for a
        file with no eligible datasets.
    """

    def visit(_, obj):
        if (
            isinstance(obj, h5py.Dataset)
            and obj.shape
            and all(d > 0 for d in obj.shape)
            and not obj.compression
        ):
            return True  # a non-None return stops the walk

    with h5py.File(str(path), "r") as f:
        return bool(f.visititems(visit))

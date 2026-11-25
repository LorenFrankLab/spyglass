"""Opt-in path tracking for tables that store a path as a varchar."""

import os
import pwd
import stat as stat_mod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from functools import cached_property
from hashlib import md5
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from spyglass.utils.mixins.base import BaseMixin

STATES = (
    "ok",  # present and whole
    "partial",  # directory holds files but no marker
    "empty",  # directory present and emptied
    "missing",  # absent, containing directory present
    "parent_missing",  # containing directory absent too
    "broken_link",  # symlink outlived its target
    "not_a_dir",  # a file where a directory is declared
    "denied",  # unreadable, so presence is unknown
)

# 'denied' is not a loss: it says the scan could not look.
LOST_STATES = tuple(s for s in STATES if s not in ("ok", "denied"))

_TRACKING_CLASSES: List[type] = []


def path_hash(path: str) -> str:
    """Return the md5 of a path, used as a fixed-width key."""
    return md5(str(path).encode()).hexdigest()


@dataclass(frozen=True)
class TrackedPath:
    """One path-bearing attribute of a table.

    Parameters
    ----------
    attr : str
        Attribute holding the path. Values must be absolute.
    kind : str, optional
        'dir' or 'file'. Default 'dir'.
    markers : tuple of str, optional
        Files that mark a directory whole. A directory holding files but
        none of these is 'partial'. Directories only.
    exclude : str, optional
        Name of a method on the table returning the keys to skip, for
        paths whose absence is intended.
    """

    attr: str
    kind: str = "dir"
    markers: Tuple[str, ...] = field(default=())
    exclude: Optional[str] = None

    def __post_init__(self):
        if self.kind not in ("dir", "file"):
            raise ValueError(f"kind must be 'dir' or 'file': {self.kind}")
        if self.markers and self.kind != "dir":
            raise ValueError(f"markers need kind='dir': {self.attr}")


class PathScanner:
    """Classify paths on disk against one `TrackedPath` declaration.

    Parameters
    ----------
    tracked : TrackedPath
        Declaration supplying `kind` and `markers`.
    """

    # Per-path ceiling on the recursive walk
    WALK_CAP = 50_000

    def __init__(self, tracked: TrackedPath):
        self.kind = tracked.kind
        self.markers = tracked.markers

    def _lstat(self, path: str) -> Tuple[Optional[os.stat_result], str]:
        """Stat a path without following symlinks.

        Returns
        -------
        tuple
            `(stat_result, "")`, or `(None, state)` naming why it failed.
        """
        try:
            return os.lstat(path), ""
        except FileNotFoundError:
            return None, "missing"
        except OSError:  # PermissionError, NotADirectoryError, ELOOP
            return None, "denied"

    def _owner(self, st: os.stat_result) -> str:
        """Return the unix owner name, or the bare uid when unresolvable."""
        try:
            return pwd.getpwuid(st.st_uid).pw_name
        except KeyError:
            return str(st.st_uid)

    def _walk(self, path: str) -> Tuple[int, int]:
        """Return the file count and total bytes under a directory."""
        n_files = size = 0
        for root, _, files in os.walk(path):
            for name in files:
                st, _ = self._lstat(os.path.join(root, name))
                if st is None:
                    continue
                n_files += 1
                size += st.st_size
                if n_files >= self.WALK_CAP:
                    return n_files, size
        return n_files, size

    def _is_whole(self, path: str) -> bool:
        """Return whether a directory holds one of its markers."""
        return not self.markers or any(
            os.path.isfile(os.path.join(path, name)) for name in self.markers
        )

    def scan(self, path: str) -> Dict:
        """Classify one path.

        Parameters
        ----------
        path : str
            Absolute path.

        Returns
        -------
        dict
            `state`, `n_files`, `size_bytes`, `mtime`, `owner`.
        """
        st, failure = self._lstat(path)

        if st is None:
            state = failure
            if failure == "missing" and not os.path.isdir(
                os.path.dirname(path)
            ):
                state = "parent_missing"
            return dict(
                state=state, n_files=0, size_bytes=0, mtime=None, owner=None
            )

        found = dict(
            n_files=0,
            size_bytes=0,
            mtime=datetime.fromtimestamp(st.st_mtime),
            owner=self._owner(st),
        )

        if stat_mod.S_ISLNK(st.st_mode) and not os.path.exists(path):
            return dict(found, state="broken_link")

        if self.kind == "file":
            return dict(found, state="ok", n_files=1, size_bytes=st.st_size)

        if not os.path.isdir(path):
            return dict(found, state="not_a_dir", size_bytes=st.st_size)

        n_files, size = self._walk(path)

        if n_files == 0:
            state = "empty"
        elif not self._is_whole(path):
            state = "partial"
        else:
            state = "ok"

        return dict(found, state=state, n_files=n_files, size_bytes=size)


class PathTrackingMixin(BaseMixin):
    """Record which of a table's tracked paths are absent from disk.

    Declare alongside `SpyglassMixin`, with the paths the table owns::

        class SpikeSorting(PathTrackingMixin, SpyglassMixin, dj.Computed):
            _tracked_paths = (TrackedPath("sorting_path", kind="dir"),)
    """

    _tracked_paths: Tuple[TrackedPath, ...] = ()

    @cached_property
    def _run_tbl(self):
        """Lazy import of CheckMissingRun, which would cycle at module load."""
        from spyglass.common.common_file_tracking import CheckMissingRun

        return CheckMissingRun

    @cached_property
    def _missing_tbl(self):
        """Lazy import of MissingPath, which would cycle at module load."""
        from spyglass.common.common_file_tracking import MissingPath

        return MissingPath

    def __init_subclass__(cls, **kwargs):
        """Register any subclass that declares tracked paths."""
        super().__init_subclass__(**kwargs)
        # `cls.__dict__`, not `cls._tracked_paths`: the inherited value would
        # register every subclass of a declaring table, part tables included
        if cls.__dict__.get("_tracked_paths"):
            _TRACKING_CLASSES.append(cls)

    @classmethod
    def tracking_tables(cls) -> List:
        """Return an instance of every table declaring tracked paths.

        A table whose module was never imported is absent, which
        `CheckMissingRun.Coverage` reports as uncovered.

        Returns
        -------
        tables : list
            One instance per registered class.
        """
        return [tbl_cls() for tbl_cls in _TRACKING_CLASSES]

    def _excluded_keys(self, tracked: TrackedPath) -> set:
        """Return the key tuples this attribute's exclusion skips."""
        if tracked.exclude is None:
            return set()
        return getattr(self, tracked.exclude)()

    def check_missing(
        self,
        run_id: Optional[int] = None,
        max_workers: int = 16,
        verbose: bool = True,
    ) -> Dict[str, int]:
        """Record tracked paths of this table that are absent from disk.

        Parameters
        ----------
        run_id : int, optional
            Run to attribute the findings to. A new run is opened and closed
            when omitted.
        max_workers : int, optional
            Threads used to stat paths. Default 16.
        verbose : bool, optional
            Show a progress bar per attribute. Default True.

        Returns
        -------
        counts : dict
            Rows recorded, per attribute.
        """
        if not self._tracked_paths:
            return {}

        own_run = run_id is None
        if own_run:
            run_id = self._run_tbl().open()

        counts = {}
        try:
            for tracked in self._tracked_paths:
                counts[tracked.attr] = self._check_one(
                    tracked, run_id, max_workers, verbose
                )
        finally:
            if own_run:
                self._run_tbl().close(run_id)

        return counts

    def _attr_key(self, tracked: TrackedPath) -> dict:
        """Return the `MissingPath` restriction for one tracked attribute."""
        return dict(full_table_name=self.full_table_name, attr=tracked.attr)

    def refresh_missing(
        self, max_workers: int = 16, verbose: bool = True
    ) -> int:
        """Drop recorded paths of this table that are present again.

        Re-stats only what is already recorded, so it costs a fraction of a
        full `check_missing`. Recovery is a rare, announced event, such as a
        restore from backup, so it is triggered rather than run every scan.

        Parameters
        ----------
        max_workers : int, optional
            Threads used to stat paths. Default 16.
        verbose : bool, optional
            Show a progress bar per attribute. Default True.

        Returns
        -------
        n_recovered : int
            Rows dropped because the path is present.
        """
        n_recovered = 0

        for tracked in self._tracked_paths:
            recorded = self._missing_tbl() & self._attr_key(tracked)
            rows = recorded.fetch("path_hash", "path", as_dict=True)
            if not rows:
                continue

            scanner = PathScanner(tracked)
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                scans = tqdm(
                    pool.map(scanner.scan, [r["path"] for r in rows]),
                    total=len(rows),
                    desc=f"refresh {self.__class__.__name__}.{tracked.attr}",
                    disable=not verbose,
                )
                back = [
                    dict(self._attr_key(tracked), path_hash=row["path_hash"])
                    for row, scan in zip(rows, scans)
                    if scan["state"] == "ok"
                ]

            if back:
                (self._missing_tbl() & back).delete_quick()
                n_recovered += len(back)

        return n_recovered

    def _check_one(self, tracked, run_id, max_workers, verbose=False) -> int:
        """Scan one tracked attribute and record what is absent."""
        pk = list(self.primary_key)
        rows = self.fetch(*pk, tracked.attr, as_dict=True)
        excluded = self._excluded_keys(tracked)
        scanner = PathScanner(tracked)

        with_path = [
            (tuple(r[k] for k in pk), r[tracked.attr])
            for r in rows
            if r[tracked.attr]
        ]
        todo = [(key, path) for key, path in with_path if key not in excluded]

        found = []

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            # `map` keeps input order, so results still zip with their keys
            scans = tqdm(
                pool.map(scanner.scan, [path for _, path in todo]),
                total=len(todo),
                desc=f"{self.__class__.__name__}.{tracked.attr}",
                disable=not verbose,
            )
            for (key, path), scan in zip(todo, scans):
                if scan["state"] == "ok":
                    continue
                # Named rather than splatted from `scan`, which also returns
                # the file count and size it classifies a directory by
                found.append(
                    dict(
                        full_table_name=self.full_table_name,
                        attr=tracked.attr,
                        path_hash=path_hash(path),
                        path=path,
                        source_key=dict(zip(pk, key)),
                        state=scan["state"],
                        run_id=run_id,
                    )
                )

        # One row per absent path, not per entry referencing it. Entries can
        # share a path: 441 of 2441 DLC video_path values repeat, so 165
        # absent references are 146 absent files. Counting references here
        # would report a number the table cannot reproduce.
        by_path = {finding["path_hash"]: finding for finding in found}

        self._missing_tbl().insert(list(by_path.values()), skip_duplicates=True)
        self._run_tbl.Coverage().insert1(
            dict(
                run_id=run_id,
                full_table_name=self.full_table_name,
                attr=tracked.attr,
                n_checked=len(todo),
                n_missing=len(by_path),
                n_excluded=len(with_path) - len(todo),
                completed=1,
            ),
            skip_duplicates=True,
        )

        return len(by_path)

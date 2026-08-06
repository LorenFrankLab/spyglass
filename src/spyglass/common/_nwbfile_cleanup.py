"""Internal filesystem-cleanup models for ``common_nwbfile``.

This module is deliberately stdlib-only so the primary DataJoint schema can
define its tables near the top of ``common_nwbfile.py`` without coupling these
value objects to schema initialization.
"""

import math
import os
import stat as stat_module
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Set, Tuple, Union


def _check_number(
    name: str,
    value: Union[int, float],
    *,
    minimum: float,
    maximum: Optional[float] = None,
) -> float:
    """Validate a numeric safety limit.

    Parameters
    ----------
    name : str
        Parameter name, used in the error message.
    value : int or float
        Supplied value.
    minimum : float
        Inclusive lower bound.
    maximum : float, optional
        Inclusive upper bound. Omit for an unbounded limit.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If the value is not a finite number within bounds. ``bool`` is
        rejected explicitly because it is an ``int`` subclass, so
        ``max_delete_fraction=True`` would otherwise read as 1.0. NaN and
        inf are rejected rather than coerced: under NaN every comparison is
        False and the guard silently vanishes.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    if value < minimum or (maximum is not None and value > maximum):
        bound = (
            f"in [{minimum}, {maximum}]"
            if maximum is not None
            else f">= {minimum}"
        )
        raise ValueError(f"{name} must be {bound}, got {value!r}")
    return float(value)


@dataclass(frozen=True)
class TargetSnapshot:
    """Identity of a real analysis file at scan time.

    Attributes
    ----------
    real_path : pathlib.Path
        Fully resolved path of the file.
    dev, ino : int
        Device and inode, used to prove the file was not swapped.
    size : int
        Size in bytes; 0 marks an empty analysis file.
    mtime_ns, ctime_ns : int
        Nanosecond timestamps, used by the age gate.
    mode : int
        Raw ``st_mode``; must be a regular file to be deletable.
    """

    real_path: Path
    dev: int
    ino: int
    size: int
    mtime_ns: int
    ctime_ns: int
    mode: int

    @property
    def newest_ns(self) -> int:
        """Newest of mtime/ctime, the conservative age basis."""
        return max(self.mtime_ns, self.ctime_ns)

    @property
    def is_regular(self) -> bool:
        """True when the target is a regular file (not fifo/socket/device)."""
        return stat_module.S_ISREG(self.mode)


@dataclass(frozen=True)
class AccessSnapshot:
    """One path under analysis_dir through which a target was reached.

    Attributes
    ----------
    access_path : pathlib.Path
        Path as found by the directory walk.
    is_link : bool
        True when ``access_path`` is itself a symlink.
    raw_link_target : str or None
        ``os.readlink`` value when ``is_link``, else None. Compared at
        deletion time so a re-pointed link is not followed.
    dev, ino : int
        ``lstat`` device and inode of the access path itself.
    mtime_ns, ctime_ns : int
        ``lstat`` nanosecond timestamps of the access path itself.
    """

    access_path: Path
    is_link: bool
    raw_link_target: Optional[str]
    dev: int
    ino: int
    mtime_ns: int
    ctime_ns: int

    def __post_init__(self):
        """Reject an access whose link flag and raw target disagree.

        ``raw_link_target`` is the ``os.readlink`` value of a symlink and is
        None for a non-link; ``os.readlink`` never returns an empty string,
        so the two are equivalent. Enforcing it stops the type from lying
        about itself -- a link with no recorded target, or a non-link
        carrying one, could otherwise skew the deletion-time re-pointed-link
        check.
        """
        if self.is_link != (self.raw_link_target is not None):
            raise ValueError(
                f"AccessSnapshot for {self.access_path}: is_link="
                f"{self.is_link} disagrees with raw_link_target="
                f"{self.raw_link_target!r}"
            )

    @classmethod
    def from_path(cls, access_path: Path) -> "AccessSnapshot":
        """Snapshot the leaf identity and raw link target at ``access_path``.

        ``lstat`` deliberately inspects the access itself rather than following
        a symlink. Any filesystem error propagates to the caller, which owns the
        policy for logging, skipping, or aborting.
        """
        access_path = Path(access_path)
        lst = os.lstat(access_path)
        is_link = stat_module.S_ISLNK(lst.st_mode)
        raw_target = os.readlink(access_path) if is_link else None
        return cls(
            access_path=access_path,
            is_link=is_link,
            raw_link_target=raw_target,
            dev=lst.st_dev,
            ino=lst.st_ino,
            mtime_ns=lst.st_mtime_ns,
            ctime_ns=lst.st_ctime_ns,
        )

    @property
    def newest_ns(self) -> int:
        """Newest of mtime/ctime for this access path."""
        return max(self.mtime_ns, self.ctime_ns)


@dataclass(frozen=True)
class CleanupCandidate:
    """A real analysis file plus every in-tree path that reaches it.

    A broken symlink has access snapshots but no target snapshot; there is
    no target to snapshot. ``broken`` is therefore the absence of a target
    record, not a flag beside meaningless fields.
    """

    real_path: Path
    target: Optional[TargetSnapshot]
    accesses: Tuple[AccessSnapshot, ...]

    def __post_init__(self):
        """Reject a candidate with no access paths.

        ``newest_ns`` is undefined without at least one access, and an
        empty-``accesses`` candidate slips through the structural preflight
        (its ``*.nwb`` loop is vacuous) only to raise from ``max()`` partway
        through the deletion loop -- after earlier candidates have already
        been unlinked. Enforcing it here also covers the dry-run path, which
        returns before the preflight runs.
        """
        if not self.accesses:
            raise ValueError(
                f"CleanupCandidate for {self.real_path} has no access paths; "
                "a candidate must be reachable from within the analysis "
                "directory"
            )
        if self.target is not None and self.target.real_path != self.real_path:
            raise ValueError(
                f"CleanupCandidate.real_path {self.real_path} does not match "
                f"its target snapshot path {self.target.real_path}"
            )

    @property
    def broken(self) -> bool:
        """True when this candidate is a dangling symlink."""
        return self.target is None

    @property
    def newest_ns(self) -> int:
        """Newest timestamp across the target and every access alias.

        A fresh link to an old target may be work awaiting registration, so
        eligibility requires everything to be old enough.
        """
        stamps = [access.newest_ns for access in self.accesses]
        if self.target is not None:
            stamps.append(self.target.newest_ns)
        return max(stamps)


@dataclass
class _TrackedFileState:
    """Lexical and physical identities fetched from analysis externals.

    ``resolved_paths`` preserves the existing behavior for missing targets,
    while the identity sets recognize live files and leaf symlinks reached
    through case variants, hard links, or mount aliases.
    """

    resolved_paths: Set[Path]
    target_identities: Set[Tuple[int, int]]
    access_identities: Set[Tuple[int, int]]

    def matches(self, candidate: CleanupCandidate) -> bool:
        """Return whether any tracked entry identifies this candidate."""
        if candidate.real_path in self.resolved_paths:
            return True
        if (
            candidate.target is not None
            and (
                candidate.target.dev,
                candidate.target.ino,
            )
            in self.target_identities
        ):
            return True
        return any(
            (access.dev, access.ino) in self.access_identities
            for access in candidate.accesses
        )


@dataclass(frozen=True)
class _PhysicalRoot:
    """Resolved directory root plus its filesystem identity."""

    name: str
    path: Path
    dev: int
    ino: int


@dataclass(frozen=True)
class CleanupPlan:
    """Filesystem cleanup plan for analysis NWB files.

    Attributes
    ----------
    scanned_files : set of pathlib.Path
        Resolved real paths of every ``*.nwb`` entry reached by the scan.
        A leaf symlink contributes the path it resolves to, which may lie
        outside the analysis directory. Alternate mount spellings can still
        name the same filesystem object, so "resolved" does not imply a unique
        canonical spelling.
    tracked_files : set of pathlib.Path
        Resolved paths currently referenced by DataJoint external stores.
    files_to_delete : set of pathlib.Path
        Resolved target paths selected as deletion candidates. Applying the
        plan may additionally unlink authorizing leaf symlinks; final
        validation may also refuse a candidate that appears in this set.
    empty_files : set of pathlib.Path
        Empty (0-byte) analysis files selected for deletion.
    untracked_files : set of pathlib.Path
        Non-empty files selected because no external store references them.
    candidates : dict of pathlib.Path to CleanupCandidate
        Deletion candidates keyed by resolved real path, carrying the
        identity snapshots re-verified before any unlink.
    deferred_recent_files : set of pathlib.Path
        Candidates held back because they are newer than the age limit.
    broken_links : set of pathlib.Path
        Resolved targets of dangling ``*.nwb`` symlinks.
    """

    scanned_files: Set[Path]
    tracked_files: Set[Path]
    files_to_delete: Set[Path]
    empty_files: Set[Path]
    untracked_files: Set[Path]
    candidates: Dict[Path, CleanupCandidate]
    deferred_recent_files: Set[Path]
    broken_links: Set[Path]

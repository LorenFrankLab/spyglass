"""Internal filesystem-cleanup models for ``common_nwbfile``.

This module is deliberately stdlib-only so the primary DataJoint schema can
define its tables near the top of ``common_nwbfile.py`` without coupling these
value objects to schema initialization.

Contents
--------
- ``_check_number`` : validate a numeric safety limit (finite, in-bounds).
- ``TargetSnapshot`` : identity of a real analysis file at scan time.
- ``AccessSnapshot`` : one in-tree path (regular or symlink) reaching a target.
- ``CleanupCandidate`` : a real target plus every access alias that reaches it,
  and the age-eligibility check over all of them.
- ``_TrackedFileState`` : resolved paths and filesystem identities of files the
  DataJoint external stores currently track.
- ``_PhysicalRoot`` : a resolved directory root plus its filesystem identity.
- ``CleanupPlan`` : the classified filesystem-cleanup plan for analysis files,
  plus its ``validate()`` safety-limit check.
- ``CleanupPlanner`` : builds a ``CleanupPlan`` from injected walk, snapshot,
  and tracked-state callables (schema-agnostic, no DataJoint import).
- ``CleanupExecutor`` : validates and unlinks a plan's candidates -- the
  destructive act loop and every pre-delete safety guard -- via injected
  tracking/registry/protected-root/leaf-validation callables.
"""

import logging
import math
import os
import stat as stat_module
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Set, Tuple, Union


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
        """Reject a structurally invalid candidate.

        Two invariants are enforced at construction. First, ``accesses`` must
        be non-empty: ``newest_ns`` is undefined without at least one access,
        and an empty-``accesses`` candidate slips through the structural
        preflight (its ``*.nwb`` loop is vacuous) only to raise from ``max()``
        partway through the deletion loop -- after earlier candidates have
        already been unlinked. Enforcing it here also covers the dry-run path,
        which returns before the preflight runs. Second, a present ``target``
        must describe this candidate's own ``real_path``; a mismatch is an
        internal-consistency bug caught here rather than at act time.
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

    def is_old_enough(self, *, now_ns: int, min_file_age_hours: float) -> bool:
        """Return True when every part of this candidate is old enough.

        Exactly ``min_file_age_hours`` old is eligible. A future timestamp
        yields a negative age and is therefore deferred. A non-positive
        ``min_file_age_hours`` disables the gate.
        """
        if min_file_age_hours <= 0:
            return True
        threshold_ns = int(min_file_age_hours * 3600 * 10**9)
        return (now_ns - self.newest_ns) >= threshold_ns


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

    def validate(
        self,
        *,
        max_delete_fraction: float = 0.9,
        max_delete_to_tracked_ratio: float = 10.0,
    ) -> Tuple[bool, Optional[str]]:
        """Check this plan against the destructive-cleanup safety limits.

        The two limits are validated here as well as at the ``cleanup()``
        entry point, so a direct call cannot smuggle a ``NaN``/``inf``/``bool``
        limit past the comparisons -- under ``NaN`` every comparison is False
        and the guard would silently vanish.

        Returns
        -------
        tuple[bool, str or None]
            ``(True, None)`` when the plan is safe to apply, otherwise
            ``(False, reason)`` where ``reason`` explains the refusal. Callers
            decide whether to raise (real run) or warn (dry run).

        Raises
        ------
        ValueError
            If either limit is non-finite, a bool, or outside its bounds.
        """
        max_delete_fraction = _check_number(
            "max_delete_fraction", max_delete_fraction, minimum=0, maximum=1
        )
        max_delete_to_tracked_ratio = _check_number(
            "max_delete_to_tracked_ratio",
            max_delete_to_tracked_ratio,
            minimum=0,
        )

        scanned_count = len(self.scanned_files)
        delete_count = len(self.files_to_delete)

        if delete_count == 0:
            return True, None

        # Denominator is what this sweep was entitled to act on: the
        # deletions plus the scanned files it recognized as tracked.
        # Age-deferred files are excluded -- folding them in would dilute the
        # fraction (e.g. 89 deletions against 1 tracked file is 98.9% and
        # refused, but adding 10 deferred files to the denominator reads as
        # 89% and passes).
        local_tracked = self.scanned_files & self.tracked_files
        tracked_count = len(local_tracked)
        eligible_count = delete_count + tracked_count

        if tracked_count == 0:
            return False, (
                "Analysis cleanup would delete "
                f"{delete_count} files after scanning {scanned_count} files, "
                "but no tracked analysis files were found. Refusing "
                "destructive cleanup; run with dry_run=True and verify the "
                "configured analysis directory."
            )

        delete_fraction = delete_count / max(eligible_count, 1)
        if delete_fraction > max_delete_fraction:
            return False, (
                "Analysis cleanup would delete "
                f"{delete_count}/{eligible_count} eligible analysis files "
                f"({delete_fraction:.1%}), above the safety limit "
                f"{max_delete_fraction:.1%}. Refusing destructive cleanup; "
                "run with dry_run=True and verify the cleanup plan."
            )

        delete_ratio = delete_count / tracked_count
        if delete_ratio > max_delete_to_tracked_ratio:
            return False, (
                "Analysis cleanup would delete "
                f"{delete_count} files with only {tracked_count} tracked "
                f"analysis files ({delete_ratio:.1f}x), above the safety "
                f"limit {max_delete_to_tracked_ratio:.1f}x. Refusing "
                "destructive cleanup; run with dry_run=True and verify the "
                "configured analysis directory."
            )

        return True, None

    def structural_problems(self) -> List[str]:
        """Per-candidate structural-integrity problems (a pure check).

        Returns human-readable problems; an empty list means the candidate
        records are internally consistent. Checked at act time rather than in
        ``__post_init__`` because the plan's ``Set``/``Dict`` fields are
        mutable -- ``frozen=True`` blocks only attribute rebinding, so a plan
        can be forged by post-construction mutation that ``__post_init__``
        would never see. Callers pair this with the separate ``candidates``-
        keys/``files_to_delete`` consistency check.
        """
        problems: List[str] = []
        for key_path, candidate in self.candidates.items():
            if key_path != candidate.real_path:
                problems.append(
                    f"plan key {key_path} does not match candidate "
                    f"{candidate.real_path}"
                )
            if (
                candidate.target is not None
                and candidate.target.real_path != candidate.real_path
            ):
                problems.append(
                    f"{candidate.real_path}: target snapshot names a "
                    f"different path {candidate.target.real_path}"
                )
            # The scanner only ever yields *.nwb entries, so any other
            # suffix means the plan did not come from it. Enforcing the
            # invariant here stops a forged plan using, say,
            # analysis/voucher.txt -> /outside/victim.nwb as authority.
            for access in candidate.accesses:
                # Exactly the scanner's predicate, case-sensitive: a
                # `voucher.NWB` cannot come from the scan, so accepting it
                # here would let a forged plan authorize deletion.
                if not access.access_path.name.endswith(".nwb"):
                    problems.append(
                        f"{candidate.real_path}: access path "
                        f"{access.access_path} is not a *.nwb entry"
                    )
        return problems


class CleanupPlanner:
    """Build a :class:`CleanupPlan` from filesystem and tracking state.

    The planner is DataJoint-import-free: the directory walk, per-entry
    snapshotting, and tracked-state lookup are all injected callables, so it
    unit-tests with fakes and never imports the DataJoint schema. ``cleanup``
    wires the real ``AnalysisNwbfile`` methods in, which keeps their existing
    monkeypatch seams -- notably ``_walk_analysis_files`` -- effective, and
    keeps the ``tqdm`` progress wrapper on the schema side so this module stays
    stdlib-only.

    Parameters
    ----------
    walker : callable
        ``() -> Iterator[Path]`` yielding every scanned access path.
    snapshotter : callable
        ``(Path) -> (real_path, target_or_None, access) or None`` for one
        entry; None means the entry was skipped (vanished, unreadable).
    tracking_state_fn : callable
        ``(custom_tables) -> _TrackedFileState`` for the current tracked set.
    logger : logging.Logger
        Used to warn about a skipped non-regular target.
    min_file_age_hours : float, optional
        Candidates newer than this are deferred. Defaults to 24.0.
    now_ns : int, optional
        Injected clock in nanoseconds. Defaults to ``time.time_ns()``.
    """

    def __init__(
        self,
        *,
        walker: Callable[[], Iterator[Path]],
        snapshotter: Callable[
            [Path],
            Optional[Tuple[Path, Optional[TargetSnapshot], AccessSnapshot]],
        ],
        tracking_state_fn: Callable[[list], _TrackedFileState],
        logger: logging.Logger,
        min_file_age_hours: float = 24.0,
        now_ns: Optional[int] = None,
    ):
        self._walker = walker
        self._snapshotter = snapshotter
        self._tracking_state_fn = tracking_state_fn
        self._logger = logger
        self._min_file_age_hours = min_file_age_hours
        self._now_ns = now_ns

    def build(self, custom_tables: list) -> CleanupPlan:
        """Scan the analysis directory and classify every ``*.nwb`` entry.

        A file tracked through any alias wins over empty and broken
        classification; a non-regular target is skipped with a warning; a
        candidate newer than the age gate is deferred.
        """
        now_ns = time.time_ns() if self._now_ns is None else self._now_ns

        tracked_state = self._tracking_state_fn(custom_tables)
        tracked = set(tracked_state.resolved_paths)

        targets: Dict[Path, Optional[TargetSnapshot]] = {}
        accesses: Dict[Path, List[AccessSnapshot]] = {}
        for access_path in self._walker():
            snapped = self._snapshotter(access_path)
            if snapped is None:
                continue
            real_path, target, access = snapped
            accesses.setdefault(real_path, []).append(access)
            if target is not None:
                targets[real_path] = target
            else:
                targets.setdefault(real_path, None)

        scanned = set(accesses)
        candidates: Dict[Path, CleanupCandidate] = {}
        empty: Set[Path] = set()
        untracked: Set[Path] = set()
        broken: Set[Path] = set()
        deferred: Set[Path] = set()

        for real_path, target in targets.items():
            candidate = CleanupCandidate(
                real_path=real_path,
                target=target,
                accesses=tuple(accesses[real_path]),
            )
            if tracked_state.matches(candidate):
                # CleanupPlan's safety denominator intersects scanned and
                # tracked PATHS. Preserve that contract when tracking matched
                # through filesystem identity rather than path spelling.
                tracked.add(real_path)
                continue  # tracked wins over empty and over broken
            if target is not None and not target.is_regular:
                self._logger.warning(
                    f"Skipping non-regular analysis path: {real_path}"
                )
                continue
            if not candidate.is_old_enough(
                now_ns=now_ns,
                min_file_age_hours=self._min_file_age_hours,
            ):
                deferred.add(real_path)
                continue
            candidates[real_path] = candidate
            if candidate.broken:
                broken.add(real_path)
            elif target.size == 0:
                empty.add(real_path)
            else:
                untracked.add(real_path)

        return CleanupPlan(
            scanned_files=scanned,
            tracked_files=tracked,
            files_to_delete=set(candidates),
            empty_files=empty,
            untracked_files=untracked,
            candidates=candidates,
            deferred_recent_files=deferred,
            broken_links=broken,
        )

    @staticmethod
    def _walk_analysis_files(analysis_dir) -> Iterator[Path]:
        """Yield every ``*.nwb`` entry under ``analysis_dir``.

        Directory symlinks are not followed, matching prior behavior. A
        walk error is re-raised rather than skipped: a destructive sweep
        must not act on a partial view of the tree.

        Yields
        ------
        pathlib.Path
            Path as found by the walk, which may itself be a symlink.
        """

        def _reraise(err: OSError) -> None:
            raise err

        scan_root = str(Path(analysis_dir).expanduser())
        for dirpath, _dirnames, filenames in os.walk(
            scan_root, followlinks=False, onerror=_reraise
        ):
            for fname in filenames:
                if fname.endswith(".nwb"):
                    yield Path(dirpath) / fname

    @staticmethod
    def _snapshot_entry(
        access_path: Path, logger: logging.Logger
    ) -> Optional[Tuple[Path, Optional[TargetSnapshot], AccessSnapshot]]:
        """Snapshot one scanned entry.

        Returns
        -------
        tuple or None
            ``(real_path, target_or_None, access_snapshot)``, or None when the
            entry cannot be snapshotted -- it vanished mid-scan, is unreadable,
            or is unresolvable -- and should be skipped.

        Notes
        -----
        Per-entry errors are logged and skipped, not raised: an entry that
        cannot be stat'd never becomes a deletion candidate, so skipping can
        only under-clean. Errors from the directory *walk* remain fatal --
        there a partial view means unseen files, which would skew the
        deletion-limit denominators.
        """
        try:
            access = AccessSnapshot.from_path(access_path)
        except OSError as err:
            # Skip, don't abort (the Notes above cover why per-entry skips are
            # safe): aborting on one bad symlink -- a loop, a 0700 parent --
            # would wedge the periodic sweep and every later maintenance phase.
            logger.warning(f"Skipping unreadable analysis entry: {err}")
            return None

        is_link = access.is_link
        real_path = Path(os.path.realpath(access_path))

        try:
            st = os.stat(access_path)  # follows symlinks
        except FileNotFoundError:
            if is_link:
                return real_path, None, access  # broken link
            return None  # regular entry vanished mid-scan
        except OSError as err:
            # Symlink loop (ELOOP), unreadable component, or I/O error.
            # Skip rather than abort, for the reason above.
            logger.warning(f"Skipping unresolvable analysis entry: {err}")
            return None

        target = TargetSnapshot(
            real_path=real_path,
            dev=st.st_dev,
            ino=st.st_ino,
            size=st.st_size,
            mtime_ns=st.st_mtime_ns,
            ctime_ns=st.st_ctime_ns,
            mode=st.st_mode,
        )
        return real_path, target, access


class CleanupExecutor:
    """Validate and unlink a :class:`CleanupPlan`'s deletion candidates.

    DataJoint-import-free like :class:`CleanupPlanner`: the tracked-state refresh,
    registry refresh, protected-root snapshot, and per-leaf re-validation are
    injected callables, so the destructive act loop -- and every safety guard
    it runs -- lives here rather than on the DataJoint schema, while remaining
    unit-testable with fakes.

    Parameters
    ----------
    plan : CleanupPlan
        The plan to apply.
    analysis_dir : str or pathlib.Path
        Configured analysis directory; deletions inside it are local, targets
        outside it are external (cross-volume) deletions.
    tracking_refresher : callable
        ``(known_tables) -> _TrackedFileState``; re-read once per candidate.
    registry_refresher : callable
        ``(known_tables) -> list`` monotonic custom-table accumulator; re-read
        once per candidate so a table declared mid-pass is never forgotten.
    managed_roots_fn : callable
        ``() -> list[_PhysicalRoot]``; the protected stores, snapshotted once
        before the first unlink.
    access_validator : callable
        ``(access, *, analysis_root) -> bool``; re-verifies one leaf symlink
        immediately before it is unlinked.
    logger : logging.Logger
        Cleanup logger for skip/audit messages.
    min_file_age_hours : float, optional
        Act-time age recheck threshold. Defaults to 24.0.
    now_ns : int, optional
        Injected clock for the act-time recheck. Defaults to
        ``time.time_ns()``.
    """

    def __init__(
        self,
        plan: CleanupPlan,
        *,
        analysis_dir: Union[str, Path],
        tracking_refresher: Callable[[list], _TrackedFileState],
        registry_refresher: Callable[[list], list],
        managed_roots_fn: Callable[[], List[_PhysicalRoot]],
        access_validator: Callable[..., bool],
        logger: logging.Logger,
        min_file_age_hours: float = 24.0,
        now_ns: Optional[int] = None,
    ):
        self._plan = plan
        self._analysis_dir = analysis_dir
        self._tracking_refresher = tracking_refresher
        self._registry_refresher = registry_refresher
        self._managed_roots_fn = managed_roots_fn
        self._access_validator = access_validator
        self._logger = logger
        self._min_file_age_hours = min_file_age_hours
        self._now_ns = now_ns

    def execute(
        self, custom_tables: list, dry_run: bool = True
    ) -> Tuple[Set[Path], Set[Path]]:
        """Apply the plan: preview when ``dry_run``, else validate and unlink.

        Returns ``(candidate_target_paths, tracked_files)``. In a real run the
        whole plan is structurally validated before any unlink; then every
        candidate is re-checked (tracking, age, alias authority, target
        identity, protected-root) immediately before its target and authorizing
        leaves are removed. Any per-candidate unlink failure is raised at the
        end so later analysis-storage work fails closed.
        """
        plan = self._plan
        logger = self._logger
        min_file_age_hours = self._min_file_age_hours
        now_ns = self._now_ns

        if plan.deferred_recent_files:
            logger.info(
                f"  {len(plan.deferred_recent_files)} untracked files "
                f"deferred because they are newer than {min_file_age_hours} "
                "hours. Use min_file_age_hours=0 only for intentional "
                "immediate cleanup."
            )

        if dry_run:
            # Planned candidate LOGICAL bytes: the sum of target sizes for
            # candidates that have a target (broken links have none and add
            # 0). This is a planning figure, not guaranteed reclaimed bytes --
            # act-time re-validation may refuse candidates, and hard-linked
            # targets share physical bytes on disk.
            planned_bytes = sum(
                candidate.target.size
                for candidate in plan.candidates.values()
                if candidate.target is not None
            )
            logger.info(
                f"  {len(plan.files_to_delete)} untracked or empty analysis "
                f"files ({len(plan.untracked_files)} untracked, "
                f"{len(plan.empty_files)} empty, "
                f"{len(plan.broken_links)} broken links); "
                f"{planned_bytes} planned candidate logical bytes"
            )
            return plan.files_to_delete, plan.tracked_files

        analysis_root = Path(self._analysis_dir).expanduser().resolve()
        act_now_ns = time.time_ns() if now_ns is None else now_ns
        failures = []

        # ---- PREFLIGHT: validate the WHOLE plan before any unlink ----
        # A structurally malformed plan must not be partially executed:
        # collecting per-entry problems while still deleting the valid
        # entries would let a forged plan do real damage before it failed.
        if set(plan.candidates) != plan.files_to_delete:
            raise RuntimeError(
                "Analysis file deletion failed: cleanup plan is "
                "inconsistent, candidate keys do not match files_to_delete. "
                "Refusing to act on a malformed plan."
            )
        # Per-candidate structural integrity is the plan's own contract; the
        # keys/files_to_delete consistency above is checked here because it has
        # a distinct refusal message.
        problems = plan.structural_problems()
        if problems:
            raise RuntimeError(
                "Analysis file deletion failed: cleanup plan is malformed; "
                "refusing to delete anything:\n  " + "\n  ".join(problems)
            )

        # Snapshot every protected root before the first unlink. A missing or
        # unreadable configured store must disable the whole destructive pass,
        # not fail only after earlier candidates have already been removed.
        managed_roots = self._managed_roots_fn() if plan.candidates else []

        # ---- ACT ----
        # Accumulator, not the original snapshot: a table can appear in the
        # registry while one candidate is processed and be gone before the
        # next. Re-unioning against `custom_tables` each time would forget
        # it and delete its tracked files, defeating the "never dropped"
        # guarantee the injected self._registry_refresher (wired to
        # AnalysisNwbfile._current_custom_tables) provides within a call.
        known_tables = list(custom_tables)
        deleted_external = []
        for candidate in plan.candidates.values():
            # Tracking and registry membership are re-read for EVERY
            # candidate, not once up front: a file can be registered through
            # a new alias, or a whole custom table declared, while an earlier
            # candidate is being deleted. Resolving every current external
            # path and comparing filesystem identities is intentionally
            # conservative; an exact query on scan-time names cannot discover
            # a new alias to this target. The deferred cleanup/writer lease
            # would let this be hoisted or indexed safely.
            known_tables = self._registry_refresher(known_tables)
            if self._tracking_refresher(known_tables).matches(candidate):
                logger.warning(
                    f"Skipping {candidate.real_path}: became tracked since "
                    "the scan"
                )
                continue
            # Age is re-checked at act time against the frozen scan-time
            # snapshot, so with a later clock it can only read older: this
            # cannot newly defer a candidate that already passed planning, but
            # it does honor a min_file_age_hours raised between building a
            # plan and applying it. A file whose bytes actually changed during
            # a long scan is caught instead by _candidate_still_matches, which
            # refuses any candidate whose target or alias timestamps moved.
            if not candidate.is_old_enough(
                now_ns=act_now_ns,
                min_file_age_hours=min_file_age_hours,
            ):
                logger.warning(
                    f"Skipping {candidate.real_path}: newer than "
                    f"{min_file_age_hours}h at deletion time"
                )
                continue
            in_root = self._candidate_still_matches(
                candidate,
                analysis_root=analysis_root,
                managed_roots=managed_roots,
            )
            if not in_root:
                continue
            # Validation and unlink cannot be one atomic filesystem operation.
            # The caller must exclude concurrent writers from eligible paths;
            # a shared cleanup/writer lease is the tracked follow-up.
            try:
                if not candidate.broken:
                    is_external = not candidate.real_path.is_relative_to(
                        analysis_root
                    )
                    candidate.real_path.unlink()
                    if is_external:
                        # Record only after unlink succeeds. Emit an immediate
                        # durable audit entry as well as the end-of-run summary:
                        # a later registry/DB failure must not erase evidence
                        # of an irreversible cross-volume deletion.
                        record = (candidate.real_path, candidate.target.size)
                        deleted_external.append(record)
                        logger.warning(
                            "Deleted external analysis target "
                            f"{record[0]} ({record[1]} bytes)"
                        )
                # Each leaf is re-checked immediately before its own
                # unlink, never batched: a link approved earlier in the
                # pass could be replaced by a regular file that a blind
                # unlink would then destroy. unlink() operates on the leaf
                # and never follows it, so removal order does not matter
                # even for chained aliases.
                for access in in_root:
                    if not access.is_link:
                        continue
                    if not self._access_validator(
                        access, analysis_root=analysis_root
                    ):
                        logger.warning(
                            f"Skipping link {access.access_path}: changed "
                            "since it was validated"
                        )
                        continue
                    access.access_path.unlink()
            except OSError as e:
                failures.append(f"{candidate.real_path}: {e}")

        if deleted_external:
            total = sum(size for _, size in deleted_external)
            roots = sorted({str(p.parent) for p, _ in deleted_external})
            logger.warning(
                f"  {len(deleted_external)} symlink targets deleted OUTSIDE "
                f"the analysis directory ({total} bytes reclaimed). An "
                "in-root *.nwb symlink authorizes deletion of the "
                "non-protected external target it resolved to during "
                "final validation. "
                f"Roots touched: {roots}"
            )
            for path, size in sorted(deleted_external):
                logger.info(f"    deleted external {path} ({size} bytes)")

        if failures:
            # Raise rather than log: cleanup() runs database cleanup after
            # this, and the maintenance driver gates later analysis-storage
            # phases on a failure escaping here. A logged-and-swallowed
            # error would let both proceed with storage in an unknown state.
            raise RuntimeError(
                f"Analysis file deletion failed for {len(failures)} "
                "candidates; refusing to continue with analysis storage in "
                "an unknown state:\n  " + "\n  ".join(failures)
            )

        return plan.files_to_delete, plan.tracked_files

    @staticmethod
    def _access_still_matches(
        access: AccessSnapshot, *, analysis_root: Path
    ) -> bool:
        """Re-verify one leaf symlink immediately before unlinking it.

        Proves the leaf is still the entry that was scanned -- same type,
        inode, timestamps, raw target, and still inside the analysis root.
        Its canonical destination was checked while the whole alias chain
        still existed, during the final validation before target deletion.
        It cannot be checked here after the target has intentionally been
        removed; requiring resolution at this point would strand chained
        aliases.
        """
        try:
            lst = os.lstat(access.access_path)
        except (OSError, RuntimeError):
            return False
        if not stat_module.S_ISLNK(lst.st_mode):
            return False
        if (lst.st_dev, lst.st_ino) != (access.dev, access.ino):
            return False
        if (lst.st_mtime_ns, lst.st_ctime_ns) != (
            access.mtime_ns,
            access.ctime_ns,
        ):
            return False
        try:
            if os.readlink(access.access_path) != access.raw_link_target:
                return False
        except (OSError, RuntimeError):
            return False
        # Still inside the analysis root, resolving the PARENT so the link
        # itself is not followed.
        try:
            location = (
                access.access_path.parent.resolve() / access.access_path.name
            )
        except (OSError, RuntimeError):
            return False
        return location.is_relative_to(analysis_root)

    @staticmethod
    def _containing_managed_root(
        path: Path, roots: List[_PhysicalRoot]
    ) -> Optional[_PhysicalRoot]:
        """Return the protected root physically containing ``path``.

        Lexical containment is the common fast path. When spellings differ,
        walk the target's parents once and compare each directory identity
        with all configured roots. This recognizes case aliases and alternate
        names of a configured root itself without confusing an outside hard
        link to a protected file with a path inside the store.
        """
        if not roots:
            return None

        for root in roots:
            if path.is_relative_to(root.path):
                return root

        roots_by_identity = {(root.dev, root.ino): root for root in roots}
        current = path.parent
        while True:
            try:
                current_stat = os.stat(current)
            except OSError as err:
                raise RuntimeError(
                    "Cannot inspect ancestry of analysis target "
                    f"{path}; refusing analysis cleanup"
                ) from err
            if root := roots_by_identity.get(
                (current_stat.st_dev, current_stat.st_ino)
            ):
                return root
            parent = current.parent
            if parent == current:
                return None
            current = parent

    def _candidate_still_matches(
        self,
        candidate: CleanupCandidate,
        *,
        analysis_root: Path,
        managed_roots: List[_PhysicalRoot],
    ) -> Optional[List[AccessSnapshot]]:
        """Re-verify a candidate during final pre-delete validation.

        Returns
        -------
        list of AccessSnapshot or None
            The in-root accesses that may be unlinked, or None (with a
            warning) when anything changed since the scan, when a
            non-regular target is involved, when the target has no in-root
            access authorizing it, or when the target belongs to another
            protected Spyglass store.

        Notes
        -----
        The returned list is what the caller unlinks. Accesses outside the
        analysis root are never returned, so an extra out-of-root access on
        an otherwise valid candidate cannot be removed.
        """
        real_path = candidate.real_path

        def _refuse(reason: str):
            self._logger.warning(f"Skipping {real_path}: {reason}")
            return None

        # An in-root *.nwb access IS the deletion authority: a target is
        # deletable on any volume, so at least one access must live inside
        # analysis_root. Validate both the scanned leaf identity and its live
        # canonical destination while every alias in a possible chain still
        # exists. The later leaf unlink check cannot resolve the destination
        # because the target has intentionally been removed by then.
        in_root = []
        for access in candidate.accesses:
            try:
                lst = os.lstat(access.access_path)
            except OSError:
                return _refuse(f"access path {access.access_path} unreadable")
            if stat_module.S_ISLNK(lst.st_mode) != access.is_link:
                return _refuse(f"{access.access_path} changed link type")
            if (lst.st_dev, lst.st_ino) != (access.dev, access.ino):
                return _refuse(f"{access.access_path} identity changed")
            # Alias timestamps are part of identity. Without this,
            # `os.utime(link, follow_symlinks=False)` makes a link fresh
            # without changing dev/ino, so the age gate -- which reads the
            # scan-time snapshot -- would still authorize deletion.
            if (lst.st_mtime_ns, lst.st_ctime_ns) != (
                access.mtime_ns,
                access.ctime_ns,
            ):
                return _refuse(
                    f"{access.access_path} timestamps changed since the scan"
                )
            # Resolve the PARENT, not the link itself -- resolving the link
            # would follow it to the target.
            try:
                location = (
                    access.access_path.parent.resolve()
                    / access.access_path.name
                )
            except (OSError, RuntimeError):
                return _refuse(
                    f"{access.access_path} parent could not be resolved"
                )
            if not location.is_relative_to(analysis_root):
                continue
            if access.is_link:
                try:
                    raw = os.readlink(access.access_path)
                except OSError:
                    return _refuse(f"{access.access_path} readlink failed")
                if raw != access.raw_link_target:
                    return _refuse(f"{access.access_path} was re-pointed")
            try:
                live_target = Path(os.path.realpath(access.access_path))
            except (OSError, RuntimeError):
                return _refuse(f"{access.access_path} could not be resolved")
            if live_target != real_path:
                return _refuse(
                    f"{access.access_path} no longer resolves to the "
                    "planned target"
                )
            in_root.append(access)

        if not in_root:
            return _refuse("no access path inside the analysis directory")

        # A target belonging to another Spyglass store is never ours to
        # delete, however it was reached. `tracked` only knows the analysis
        # store, so without this a symlink into the raw store would make a
        # non-recomputable acquisition file look untracked.
        managed = None
        if not candidate.broken:
            managed = self._containing_managed_root(real_path, managed_roots)
        if managed is not None:
            return _refuse(
                "target belongs to another Spyglass store "
                f"({managed.name}: {managed.path}); analysis cleanup does "
                "not delete from it"
            )

        # Target identity is verified for EVERY live target, in-root or not,
        # because every live target may be deleted. The access checks above
        # separately prove that a current in-root *.nwb path still resolves
        # to this exact target, including through intermediate symlinks.
        if not candidate.broken:
            try:
                st = os.stat(real_path)
                lst_target = os.lstat(real_path)
            except OSError:
                return _refuse("target unreadable")
            if not stat_module.S_ISREG(lst_target.st_mode):
                return _refuse("target is not a regular file")
            if (st.st_dev, st.st_ino) != (
                lst_target.st_dev,
                lst_target.st_ino,
            ):
                return _refuse("target stat/lstat disagree")
            snap = candidate.target
            if (st.st_dev, st.st_ino) != (snap.dev, snap.ino):
                return _refuse("target identity changed since the scan")
            if st.st_size != snap.size:
                return _refuse("target size changed since the scan")
            if (st.st_mtime_ns, st.st_ctime_ns) != (
                snap.mtime_ns,
                snap.ctime_ns,
            ):
                return _refuse("target timestamps changed since the scan")
            if st.st_mode != snap.mode:
                return _refuse("target mode changed since the scan")
        else:
            try:
                os.stat(real_path)
            except FileNotFoundError:
                pass  # still broken, as planned
            except OSError:
                return _refuse("broken-link target unreadable")
            else:
                return _refuse("broken link now resolves")

        return in_root

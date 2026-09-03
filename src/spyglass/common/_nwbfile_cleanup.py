"""Small, schema-independent analysis-file cleanup plan.

The cleanup contract intentionally matches the rest of Spyglass: take one
snapshot of database paths and the filesystem, classify the set difference,
then unlink it.  A leaf ``*.nwb`` symlink inside the configured analysis
directory is a managed pointer, so its recorded target may be deleted even
when that target is on another volume.

``FileSnapshot`` records one scanned leaf and ``CleanupPlan`` owns the scan,
classification, aggregate validation, and unlink pass.
"""

import logging
import math
import os
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional, Set, Tuple, Union

_HOUR_NS = 60 * 60 * 1_000_000_000


def _validate_number(
    name: str,
    value: Union[int, float],
    *,
    minimum: float,
    maximum: Optional[float] = None,
) -> None:
    """Validate that ``value`` is finite and within the requested bounds."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    if value < minimum or (maximum is not None and value > maximum):
        bounds = (
            f"in [{minimum}, {maximum}]"
            if maximum is not None
            else f">= {minimum}"
        )
        raise ValueError(f"{name} must be {bounds}, got {value!r}")


@dataclass(frozen=True, slots=True)
class CleanupPolicy:
    """Validated, immutable deletion-safety limits for one cleanup run.

    Construct this once -- before any insert-blocking trigger is acquired --
    so an out-of-bounds or non-finite argument raises immediately, without
    leaving triggers installed. Because the values are validated here,
    ``CleanupPlan`` applies them at the deletion boundary without repeating
    the raw numeric checks.
    """

    max_delete_fraction: float = 0.9
    max_delete_to_tracked_ratio: float = 10.0
    min_file_age_hours: float = 24.0

    def __post_init__(self) -> None:
        _validate_number(
            "max_delete_fraction",
            self.max_delete_fraction,
            minimum=0,
            maximum=1,
        )
        _validate_number(
            "max_delete_to_tracked_ratio",
            self.max_delete_to_tracked_ratio,
            minimum=0,
        )
        _validate_number(
            "min_file_age_hours",
            self.min_file_age_hours,
            minimum=0,
        )


@dataclass(frozen=True, slots=True)
class FileSnapshot:
    """One ``*.nwb`` path and the target it named during the scan."""

    path: Path
    resolved_path: Path
    size: int
    mtime_ns: int
    is_link: bool
    broken: bool

    @classmethod
    def from_path(cls, path: Path) -> Optional["FileSnapshot"]:
        """Describe a regular file or dangling leaf symlink.

        Non-regular targets (directories, sockets, devices, and FIFOs) are
        outside the cleanup contract and return ``None``.
        """
        path = Path(path)
        is_link = path.is_symlink()
        # ``Path.resolve()`` raises on a leaf symlink loop on Python 3.10.
        # ``realpath`` keeps its non-strict behavior in that case, giving the
        # broken leaf a stable key so cleanup can unlink it normally.
        resolved = Path(os.path.realpath(path))

        if is_link and not path.exists():
            leaf_stat = path.lstat()
            return cls(
                path=path,
                resolved_path=resolved,
                size=0,
                mtime_ns=leaf_stat.st_mtime_ns,
                is_link=True,
                broken=True,
            )

        target_stat = path.stat()
        if not stat.S_ISREG(target_stat.st_mode):
            return None
        return cls(
            path=path,
            resolved_path=resolved,
            size=target_stat.st_size,
            mtime_ns=target_stat.st_mtime_ns,
            is_link=is_link,
            broken=False,
        )


class CleanupPlan:
    """Scan, classify, validate, and apply one cleanup snapshot.

    This is deliberately a trust-the-disk plan.  It does not retain inode
    identities, re-fetch database state per candidate, or attempt to detect
    changes between scan and unlink.  Callers must not overlap cleanup with
    analysis-file writers.
    """

    def __init__(
        self,
        analysis_dir: Union[str, Path],
        tracked_files: Set[Path],
        *,
        logger: logging.Logger,
        policy: CleanupPolicy,
        now_ns: Optional[int] = None,
    ) -> None:
        self.analysis_dir = Path(analysis_dir).expanduser().resolve()
        self.tracked_files = frozenset(
            Path(os.path.realpath(Path(path).expanduser()))
            for path in tracked_files
        )
        self.logger = logger
        self.policy = policy
        self.now_ns = time.time_ns() if now_ns is None else now_ns

        self.scanned_files: frozenset[Path] = frozenset()
        self.empty_files: Set[Path] = set()
        self.untracked_files: Set[Path] = set()
        self.deferred_recent_files: Set[Path] = set()
        self.broken_links: Set[Path] = set()
        self._candidates: Dict[Path, Tuple[FileSnapshot, ...]] = {}
        self._scan()

    def _scan(self) -> None:
        grouped: Dict[Path, list[FileSnapshot]] = {}
        for path in self.walk_analysis_files(self.analysis_dir, self.logger):
            try:
                snapshot = FileSnapshot.from_path(path)
            except (OSError, RuntimeError) as error:
                self.logger.warning(
                    f"Skipping analysis path that cannot be inspected "
                    f"{path}: {error}"
                )
                continue
            if snapshot is None:
                self.logger.warning(f"Skipping non-file analysis path: {path}")
                continue
            grouped.setdefault(snapshot.resolved_path, []).append(snapshot)

        self.scanned_files = frozenset(grouped)
        min_file_age_hours = self.policy.min_file_age_hours
        minimum_age_ns = int(min_file_age_hours * _HOUR_NS)
        for resolved_path, snapshots in grouped.items():
            if resolved_path in self.tracked_files:
                continue  # Tracked wins, including for empty or dangling files.

            newest_mtime_ns = max(item.mtime_ns for item in snapshots)
            if (
                min_file_age_hours > 0
                and self.now_ns - newest_mtime_ns < minimum_age_ns
            ):
                self.deferred_recent_files.add(resolved_path)
                continue

            candidate = tuple(snapshots)
            self._candidates[resolved_path] = candidate

            if all(item.broken for item in candidate):
                self.broken_links.add(resolved_path)
            elif max(item.size for item in candidate) == 0:
                self.empty_files.add(resolved_path)
            else:
                self.untracked_files.add(resolved_path)

    @staticmethod
    def walk_analysis_files(
        analysis_dir: Path, logger: logging.Logger
    ) -> Iterator[Path]:
        """Yield ``*.nwb`` leaves within the analysis tree.

        Directory symlinks are NOT traversed. Cleanup deletes files, so it
        must not be able to follow a symlinked subdirectory out of
        ``analysis_dir`` into an unrelated store (a raw/recording root, another
        user's tree) and delete untracked ``*.nwb`` there -- there is no
        protected-store denylist to stop it. Leaf ``*.nwb`` symlinks (managed
        pointers, including to another volume) are still yielded: ``os.walk``
        lists file symlinks in ``files`` regardless of ``followlinks``, and a
        symlinked ``analysis_dir`` root is still scanned because ``os.walk``
        descends into its top directory either way. Because symlinked
        subdirectories are never followed, no cycle can arise and no
        cycle-detection is needed.
        """

        def warn(error):
            logger.warning(f"Skipping analysis directory: {error}")

        for root, _directories, files in os.walk(
            analysis_dir,
            followlinks=False,
            onerror=warn,
        ):
            for filename in files:
                if filename.endswith(".nwb"):
                    yield Path(root) / filename

    @property
    def files_to_delete(self) -> Set[Path]:
        """Candidate targets captured by this plan."""
        return set(self._candidates)

    @property
    def candidate_bytes(self) -> int:
        """Logical bytes recorded for unique candidate target groups."""
        return sum(
            max(item.size for item in snapshots)
            for snapshots in self._candidates.values()
        )

    def validate(self) -> Tuple[bool, Optional[str]]:
        """Check the inexpensive aggregate deletion backstops.

        Uses this plan's already-validated ``CleanupPolicy`` limits; the raw
        numeric checks ran once when the policy was constructed.
        """
        max_delete_fraction = self.policy.max_delete_fraction
        max_delete_to_tracked_ratio = self.policy.max_delete_to_tracked_ratio

        # Validate the private candidate mapping consumed by ``execute()``,
        # rather than a public reporting set that a caller could mutate.
        delete_count = len(self._candidates)
        if delete_count == 0:
            return True, None

        local_tracked = self.scanned_files & self.tracked_files
        tracked_count = len(local_tracked)
        if tracked_count == 0:
            return False, (
                f"Analysis cleanup would delete {delete_count} files, but no "
                "tracked analysis files were found. Refusing destructive "
                "cleanup; run with dry_run=True and verify the configured "
                "analysis directory."
            )

        eligible_count = delete_count + tracked_count
        delete_fraction = delete_count / eligible_count
        if delete_fraction > max_delete_fraction:
            return False, (
                "Analysis cleanup would delete "
                f"{delete_count}/{eligible_count} eligible analysis files "
                f"({delete_fraction:.1%}), above the safety limit "
                f"{max_delete_fraction:.1%}."
            )

        delete_ratio = delete_count / tracked_count
        if delete_ratio > max_delete_to_tracked_ratio:
            return False, (
                f"Analysis cleanup would delete {delete_count} files with "
                f"only {tracked_count} tracked files ({delete_ratio:.1f}x), "
                "above the safety limit "
                f"{max_delete_to_tracked_ratio:.1f}x."
            )
        return True, None

    def execute(
        self,
        *,
        dry_run: bool = True,
    ) -> None:
        """Report or unlink the candidates captured by this plan.

        Destructive execution validates the plan against this plan's
        ``CleanupPolicy`` immediately before unlinking, so callers cannot
        accidentally bypass the aggregate deletion limits. A dry run still
        reports a plan that would be refused, but warns that destructive
        execution would not proceed.

        The recorded target is used instead of resolving a symlink again at
        execution time.  This keeps execution deterministic without adding
        inode, timestamp, or per-candidate database revalidation machinery.
        Ordinary unlink failures are logged and cleanup continues, matching
        existing Spyglass cleanup routines.
        """
        plan_ok, plan_error = self.validate()
        if not plan_ok:
            if dry_run:
                self.logger.warning(
                    f"Cleanup plan would be refused: {plan_error}"
                )
            else:
                raise RuntimeError(plan_error)

        if self.deferred_recent_files:
            self.logger.info(
                f"  {len(self.deferred_recent_files)} untracked files "
                f"deferred because they are newer than "
                f"{self.policy.min_file_age_hours} hours"
            )
        self.logger.info(
            f"  {len(self._candidates)} untracked or empty analysis "
            f"files ({self.candidate_bytes} logical candidate bytes)"
        )
        if dry_run:
            return

        for target, snapshots in sorted(
            self._candidates.items(), key=lambda item: str(item[0])
        ):
            if all(item.broken for item in snapshots):
                for snapshot in snapshots:
                    self._unlink(snapshot.path)
                continue

            if not self._unlink(target):
                continue
            for snapshot in snapshots:
                if snapshot.is_link and snapshot.path != target:
                    self._unlink(snapshot.path)

    def _unlink(self, path: Path) -> bool:
        try:
            path.unlink(missing_ok=True)
        except OSError as error:
            self.logger.error(f"Error deleting file {path}: {error}")
            return False
        return True

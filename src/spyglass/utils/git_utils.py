"""Spyglass installation inspector.

Detects whether the running Spyglass installation is a clean official release
or a local/editable/modified copy ("dirty"), and caches the result for the
lifetime of the process.

The result is used by ``UserEnvironment`` to populate ``dirty_path`` and
``spyglass_commit``, and by the warning system to advise non-admin users.

Detection logic
---------------
1. ``is_dev``  — ``".dev"`` present in ``__version__`` (set by
   vcs-versioning; always True for editable installs on non-release commits).
2. ``commit_hash`` — ``git rev-parse --short HEAD`` when running from a clone,
   falling back to the ``+g<hash>`` suffix in ``__version__``.  ``None`` for
   standard pip installs of a release.
3. ``install_path`` — the root of the spyglass clone this package runs from,
   identified by ``<root>/src/spyglass/__init__.py`` resolving to
   ``spyglass.__file__``; ``None`` otherwise, including an install inside an
   unrelated git repo.
4. ``has_local_changes`` — the install deviates from a commit on the official
   remote, by any of: ``git diff --name-only HEAD -- src/`` returning output
   (scoped to ``src/``, so doc edits, notebook changes and ``.claude/``
   artefacts are intentionally ignored); HEAD being contained by no remote
   branch, i.e. committed locally and never pushed; or ``__version__``
   carrying hatch-vcs's ``.dYYYYMMDD`` suffix, left by a wheel built from a
   dirty tree, which has no repo to inspect.
5. ``is_stale`` — HEAD's commit date is more than ``WARN_STALE_DAYS`` days
   old, i.e. the clone has not fetched upstream in months.  Being a handful
   of commits behind does not flag; only long-neglected clones do.
6. ``is_official`` — ``not has_local_changes and not is_stale``.  Note this
   is independent of ``is_dev``: a clone tracking ``master`` that is current
   and unmodified is treated as official even though it is a dev build.
"""

import os
import re
from datetime import date, datetime
from json import dumps as json_dumps
from json import loads as json_loads
from functools import cached_property
from pathlib import Path
from subprocess import run as sub_run
from typing import Optional

from spyglass.utils import logger

WARN_DIRTY_ENV_DAYS = 30  # permitted days for locally edited spyglass
WARN_STALE_DAYS = 90  # permitted days behind LorenFrankLab/master

DIRTY_MSG = (
    "Your Spyglass repository is not on an official commit. Please ensure you "
    "are running Spyglass from a clean repository to avoid issues with "
    "reproducibility.\n\n"
    "If this is an important change, please open a pull request to merge your "
    "changes into the main branch. If not, please reset your repository to an "
    "official commit.\n\n"
    "If you need assistance, please contact the support team. We will flag "
    "this issue for discussion if it is not resolved within {days} days."
)

STALE_MSG = (
    "Spyglass is being run from an older commit. Please consider updating to "
    "the latest version to ensure you have the latest features and bug fixes."
)


class _DirtyMarker:
    """The local "this clone has logged a dirty env" record.

    Scoped per (DataJoint user, conda env) under ``temp_dir``. Using the DB
    user — not the OS login — keeps markers from clashing when many users share
    one ``temp_dir`` (and even one OS account), and aligns the marker with how
    dirty installs are keyed for warnings and emails (``env_id`` prefix). The
    DB user is read from ``dj.config`` (no ``spyglass.common`` import), so the
    clean-install fast path stays lightweight.

    Content is a JSON object: ``commit`` the hash first flagged, ``first_seen``
    the date the episode began, and ``last_logged`` the date the database last
    heard about it. Every method is best-effort — a marker is a cache, and its
    loss must never raise.
    """

    @cached_property
    def path(self) -> Optional[Path]:
        """Marker path, resolved once per process.

        ``None`` when ``temp_dir`` is unavailable. Neither the DataJoint user
        nor ``temp_dir`` changes within a session, so this is cached rather
        than rebuilt on each read and write.
        """
        try:
            import datajoint as dj

            from spyglass.settings import temp_dir
        except Exception:
            return None
        if not temp_dir:
            return None
        dj_user = dj.config.get("database.user", "") or "unknown"
        conda_env = os.environ.get("CONDA_DEFAULT_ENV", "base")
        return Path(temp_dir) / f".spyglass_git_status_{dj_user}_{conda_env}"

    def exists(self) -> bool:
        """Return ``True`` if the marker is present."""
        return bool(self.path and self.path.exists())

    def read(self) -> dict:
        """Return the recorded state, or an empty dict."""
        if not self.exists():
            return {}
        try:
            return json_loads(self.path.read_text() or "{}")
        except Exception as exc:
            logger.debug(f"could not read dirty marker: {exc}")
            return {}

    def write(self, **fields) -> None:
        """Merge fields into the recorded state.

        Merging rather than replacing, so a caller that knows one fact need
        not read, update and hand back the whole record. Use :meth:`clear` to
        drop the record entirely.
        """
        if self.path is None:
            return
        state = self.read()
        state.update(fields)
        try:
            self.path.write_text(json_dumps(state))
        except Exception as exc:
            logger.debug(f"could not write dirty marker: {exc}")

    def clear(self) -> None:
        """Remove the marker."""
        if self.path is None:
            return
        try:
            self.path.unlink(missing_ok=True)
        except Exception as exc:
            logger.debug(f"could not clear dirty marker: {exc}")


class SpyglassInstall:
    """Inspects the Spyglass installation this process is running from.

        Holds the repository root once discovered, so the git queries below read it
        rather than taking it as an argument apiece, and caches the resulting
        summary for the lifetime of the instance.

    The module-level :data:`spyglass_install` is the one instance normal code uses.
        Constructing the class re-inspects from scratch, which is what makes the
        inspection testable without reaching into module state.
    """

    _SUBPROCESS_KWARGS = dict(capture_output=True, text=True, timeout=10)

    _VCS_HASH_RE = re.compile(r"\+g([0-9a-f]+)$")  # commit ``g+<hash>`` marker
    _VCS_DIRTY_RE = re.compile(r"\.d\d{8}$")  # commit ``.dYYYYMMDD`` suffix

    def __init__(self) -> None:
        self.warned = False  # a dirty/stale warning has been emitted

    # ── Inputs ────────────────────────────────────────────────────────────────

    @cached_property
    def marker(self) -> "_DirtyMarker":
        """This clone's local dirty record."""
        return _DirtyMarker()

    @cached_property
    def _version(self) -> str:
        """deferred so import order does not matter."""
        from spyglass import __version__

        return __version__

    @cached_property
    def install_path(self) -> Optional[str]:
        """Root of the spyglass clone this package runs from, if any.

        A candidate is accepted only when its ``src/spyglass/__init__.py``
        resolves to ``spyglass.__file__``: a venv inside an unrelated repo
        (uv's default ``.venv``, or ``conda create --prefix ./env``) would
        otherwise adopt that repo's commit and dirty state.
        """
        try:
            import spyglass

            pkg_init = Path(spyglass.__file__).resolve()
            pkg_path = pkg_init.parent  # …/spyglass/
            for candidate in [pkg_path, *pkg_path.parents]:
                if not (candidate / ".git").exists():
                    continue
                clone_init = candidate / "src" / "spyglass" / "__init__.py"
                if clone_init.exists() and clone_init.resolve() == pkg_init:
                    return str(candidate)
                return None  # the first .git decides, either way
        except Exception as exc:
            logger.debug(f"could not locate install path: {exc}")
        return None

    # ── Git queries ───────────────────────────────────────────────────────────

    def _git(self, *args: str) -> Optional[str]:
        """Run git in the install path, returning stdout or ``None``.

        ``None`` means "no answer": git missing, a non-zero exit, or no clone
        to run in. Callers decide what that implies, and all of them fail safe
        by not flagging the install.
        """
        if self.install_path is None:
            return None
        try:
            result = sub_run(
                ["git", "-C", self.install_path, *args],
                **self._SUBPROCESS_KWARGS,
            )
            if result.returncode == 0:
                return result.stdout
            logger.debug(
                f"git {args[0]} exited {result.returncode}: "
                + result.stderr.strip()
            )
        except FileNotFoundError:
            logger.debug(f"git not found; skipping {args[0]}")
        except Exception as exc:
            logger.debug(f"git {args[0]} failed: {exc}")
        return None

    def short_hash(self) -> Optional[str]:
        """Return the short HEAD hash, or ``None``."""
        out = self._git("rev-parse", "--short", "HEAD")
        return out.strip() or None if out is not None else None

    def head_is_published(self) -> bool:
        """Return ``True`` if HEAD is contained by some remote branch.

        ``git diff`` is empty for a commit that exists only locally, so
        committing local work would otherwise read as clean. Reads the
        remote-tracking refs already on disk, so no network is needed and a
        clone that has never fetched reports nothing contained.

        ``True`` when git gives no answer, so a failure does not flag.
        """
        out = self._git("branch", "-r", "--contains", "HEAD")
        return True if out is None else bool(out.strip())

    def has_src_changes(self) -> bool:
        """Return ``True`` if tracked files under ``src/`` differ from HEAD.

        Covers staged and unstaged modifications. Untracked files and changes
        outside ``src/`` are intentionally excluded, so doc edits, notebook
        changes and ``.claude/`` artefacts do not flag an install.
        """
        out = self._git("diff", "--name-only", "HEAD", "--", "src/")
        return bool(out and out.strip())

    def head_is_stale(self) -> bool:
        """Return ``True`` if HEAD is older than ``WARN_STALE_DAYS``.

        Committer date as a no-network proxy for "months behind upstream": a
        clone that fetches regularly keeps a recent HEAD, a long-neglected one
        does not.
        """
        out = self._git("log", "-1", "--format=%ct", "HEAD")
        if not (out and out.strip()):
            return False
        try:
            head_date = datetime.fromtimestamp(int(out.strip()))
        except Exception as exc:
            logger.debug(f"unreadable HEAD date: {exc}")
            return False
        return (datetime.now() - head_date).days > WARN_STALE_DAYS

    # ── Summary ───────────────────────────────────────────────────────────────

    @cached_property
    def is_dev(self) -> bool:
        """``".dev"`` appears in ``__version__`` — a non-release build."""
        return ".dev" in self._version

    @cached_property
    def commit_hash(self) -> Optional[str]:
        """Short hash of the running code, or ``None`` for a clean release.

        HEAD wins over ``__version__``: ``_version.py`` is written at install
        and does not move with a ``git pull``. A wheel built from a dirty tree
        carries a ``.dYYYYMMDD`` suffix, which is also why the version may hold
        no parseable hash.
        """
        from_version = None
        if m := self._VCS_HASH_RE.search(self._version):
            from_version = m.group(1)
        if self.install_path is None:
            return from_version
        return self.short_hash() or from_version

    @cached_property
    def has_local_changes(self) -> bool:
        """The install deviates from a commit on the official remote.

        Any of: modified sources under ``src/``; a HEAD contained by no remote
        branch, i.e. committed locally and never pushed; or a version carrying
        hatch-vcs's ``.dYYYYMMDD`` suffix, left by a wheel built from a dirty
        tree. Only the last is detectable without a clone.
        """
        if self._VCS_DIRTY_RE.search(self._version):
            return True
        if self.install_path is None:
            return False
        return self.has_src_changes() or not self.head_is_published()

    @cached_property
    def is_stale(self) -> bool:
        """HEAD is more than ``WARN_STALE_DAYS`` old. ``False`` without a clone."""
        return False if self.install_path is None else self.head_is_stale()

    @cached_property
    def is_official(self) -> bool:
        """Neither dirty nor stale.

        Independent of :attr:`is_dev`: a clean, current clone tracking
        ``master`` is official even though it is a dev build.
        """
        return not self.has_local_changes and not self.is_stale

    # ── Warning and logging ───────────────────────────────────────────────────

    def warn(self) -> None:
        """Warn when running a non-official Spyglass install.

        Touches no database: the git check and the countdown both read local
        state, so an import never opens a connection, stalls behind a VPN, or
        prompts for a password. The record is written later by :meth:`log`, on
        the session's first insert or populate.

        Two messages:

        * **Dirty** (``has_local_changes``): the countdown warning. Days are
          counted from the marker's ``first_seen``, written the first time this
          clone was seen dirty.
        * **Stale** (``is_stale``, no local changes): the softer "older commit"
          advisory, with no countdown — an older official commit is
          reproducible.

        Admin users are not exempted here, because that check is a query. They
        are exempted from the database record and the escalation email.
        """
        if self.has_local_changes:
            logger.warning(DIRTY_MSG.format(days=self._days_remaining()))
        if self.is_stale:
            logger.warning(STALE_MSG)

    def _days_remaining(self) -> int:
        """Days left before this dirty install is flagged for discussion.

        Counted from the marker's ``first_seen``, so the clock survives a new
        session. A commit made while dirty keeps the original date: the user
        has been dirty throughout, and the countdown should not restart every
        time they commit.
        """
        marker = self.marker.read()
        today = date.today()
        first_seen = marker.get("first_seen")
        commit = marker.get("commit")

        if not first_seen or commit != self.commit_hash:
            self.marker.write(
                commit=self.commit_hash,
                first_seen=first_seen or today.isoformat(),
            )

        try:
            days = (today - date.fromisoformat(first_seen)).days
        except Exception:  # unreadable date — start the clock over
            days = 0
        return max(0, WARN_DIRTY_ENV_DAYS - days)

    def warn_once(self) -> None:
        """Emit the warning at most once per process.

        Called from the import-time hook in ``spyglass/__init__.py``. Fully
        guarded — a warning must never break an import.
        """
        if self.warned:
            return
        self.warned = True  # set first: a failure must not retry all session
        try:
            self.warn()
        except Exception as exc:
            logger.debug(f"install warning failed: {exc}")


# The install this process is running from. Construction is free — every fact
# is a cached property, read on first use — so this costs nothing at import.
spyglass_install = SpyglassInstall()

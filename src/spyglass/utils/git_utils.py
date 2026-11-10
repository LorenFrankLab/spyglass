"""Spyglass installation inspector.

Detects whether the running Spyglass installation is a clean official release
or a local/editable/modified copy ("dirty"), and caches the result for the
lifetime of the process.

The result is used by ``UserEnvironment`` to populate ``dirty_path`` and
``spyglass_commit``, and by the warning system to advise non-admin users.

Reported facts
--------------
``is_dev``
    ``".dev"`` present in ``__version__``, set by vcs-versioning.
``commit_hash``
    ``git rev-parse --short HEAD`` in a clone, else the ``+g<hash>`` suffix of
    ``__version__``. ``None`` for a release install.
``install_path``
    Root of the spyglass clone this package runs from, identified by
    ``<root>/src/spyglass/__init__.py`` resolving to ``spyglass.__file__``.
    ``None`` for any other layout, including an install inside an unrelated
    git repo.
``has_local_changes``
    The install deviates from a commit on the official remote. In a clone,
    either ``git diff --name-only HEAD -- src/`` returns output, scoped to
    ``src/`` so doc and notebook edits are ignored, or HEAD is not reachable
    from :data:`OFFICIAL_REPO`, whether never committed, never pushed, or
    pushed only to a fork. Without a clone, the one signal is ``__version__``
    carrying hatch-vcs's ``.dYYYYMMDD`` suffix, left by a wheel built from a
    dirty tree.
``commit_date``
    Committer date of HEAD. ``None`` outside a clone. Stored on the
    ``UserEnvironment`` row, so staleness is also measurable from the
    database.
``is_stale``
    ``commit_date`` is more than ``WARN_STALE_DAYS`` days old. A few commits
    behind does not flag.
``is_official``
    Neither dirty nor stale. Independent of ``is_dev``: a current, unmodified
    clone on any branch of the official remote is official.
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
WARN_STALE_DAYS = 90  # permitted days behind the official remote
WARN_ADMIN_TRUST_DAYS = 7  # days a cached admin check exempts a user

OFFICIAL_REPO = "LorenFrankLab/spyglass"  # matched against remote fetch URLs

DIRTY_MSG = (
    "Your Spyglass repository is not on an official commit. Please ensure you "
    "are running Spyglass from a clean repository to avoid issues with "
    "reproducibility.\n\n"
    "If you have not fetched from github.com/" + OFFICIAL_REPO + " recently, "
    "do that and import again: this check reads the branches you have "
    "fetched, so stale references can flag a clean repository. "
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

    Scoped per (DataJoint user, conda env) under ``temp_dir``. Keying on the
    DataJoint user rather than the OS login matches how dirty installs are
    keyed for warnings and emails, by ``env_id`` prefix, and keeps markers
    distinct where many users share one ``temp_dir`` or one OS account. The
    user is read from ``dj.config``, so the clean-install fast path needs no
    ``spyglass.common`` import.

    Content is a JSON object: ``commit`` the hash first flagged, ``first_seen``
    the date the episode began, ``last_logged`` the date the database last
    heard about it, and ``is_admin``/``admin_checked`` the cached admin check
    and its date. A marker is a cache, so every method is best-effort and
    losing one never raises.
    """

    @cached_property
    def path(self) -> Optional[Path]:
        """Marker path, resolved once per process.

        ``None`` when ``temp_dir`` is unavailable. Cached: neither the
        DataJoint user nor ``temp_dir`` changes within a session.
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

        Merges, so a caller that knows one fact need not hand back the whole
        record. :meth:`clear` drops the record.
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

    Holds the repository root once discovered, so the git queries read it
    rather than taking it as an argument apiece, and caches each fact it
    reports for the lifetime of the instance.

    The module-level :data:`spyglass_install` is the instance normal code uses.
    Constructing the class inspects afresh.
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
        """``spyglass.__version__``, read lazily so import order is free."""
        from spyglass import __version__

        return __version__

    @cached_property
    def _remotes(self) -> Optional[str]:
        """``git remote -v`` output, or ``None`` when git gave no answer.

        ``None`` separates a clone where git could not be run from a clone
        that ran git and has no official remote. The two are treated
        differently in :meth:`head_is_published`.
        """
        return self._git("remote", "-v")

    @cached_property
    def official_remote(self) -> Optional[str]:
        """Name of the remote pointing at :data:`OFFICIAL_REPO`, if any.

        Matched on the URL rather than the name, which is a local convention:
        a clone may call it ``sg``, ``upstream``, ``origin`` or nothing at all,
        and may hold several forks alongside it. Accepts both ``https://`` and
        ``git@`` forms, with or without a trailing ``.git``, and compares
        case-insensitively, since GitHub resolves ``lorenfranklab/spyglass``
        to the same repository.
        """
        out = self._remotes
        if not out:
            return None
        official = OFFICIAL_REPO.lower()
        for line in out.splitlines():
            parts = line.split()
            if len(parts) < 3 or parts[2] != "(fetch)":
                continue
            url = parts[1].lower().removesuffix("/").removesuffix(".git")
            if url.endswith(official):
                return parts[0]
        return None

    @cached_property
    def install_path(self) -> Optional[str]:
        """Root of the spyglass clone this package runs from, if any.

        A candidate is accepted only when its ``src/spyglass/__init__.py``
        resolves to ``spyglass.__file__``, so a venv inside an unrelated repo
        (uv's default ``.venv``, or ``conda create --prefix ./env``) is
        rejected rather than read as a spyglass clone.
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

        ``None`` means no answer: git missing, a non-zero exit, or no clone to
        run in. Each caller treats that as "not flagged".
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
        """Return ``True`` if HEAD is obtainable from the official remote.

        The test is whether anyone can fetch this commit from
        :data:`OFFICIAL_REPO`, not whether it has been reviewed onto a default
        branch: a clean checkout of an official feature branch is obtainable,
        while a push to a personal fork leaves the code where no one else can
        get it. ``git diff`` is empty for a commit that exists only locally, so
        this is what catches committed-but-unpushed work.

        Reads the remote-tracking refs already on disk, needing no network, so
        the answer is only as current as the last fetch. A branch deleted
        upstream still counts until ``git fetch --prune`` runs.

        ``False`` when git reports no official remote, since nothing in such a
        clone can be reached from the official repository. ``True`` whenever
        git gives no answer at all, or when no official ref has been fetched,
        so neither an unusable git nor a shallow clone flags.
        """
        if self._remotes is None:  # git unusable: unknown, do not flag
            return True
        remote = self.official_remote
        if remote is None:  # git worked and found no official remote
            return False
        if not self._git("for-each-ref", f"refs/remotes/{remote}/"):
            return True  # nothing fetched to compare against; do not flag
        out = self._git(
            "branch", "-r", "--contains", "HEAD", "--list", f"{remote}/*"
        )
        return True if out is None else bool(out.strip())

    def has_src_changes(self) -> bool:
        """Return ``True`` if tracked files under ``src/`` differ from HEAD.

        Covers staged and unstaged modifications. Scoped to ``src/``, so
        untracked files, doc edits and notebook changes do not flag an
        install.
        """
        out = self._git("diff", "--name-only", "HEAD", "--", "src/")
        return bool(out and out.strip())

    @cached_property
    def commit_date(self) -> Optional[date]:
        """Committer date of HEAD, or ``None`` outside a readable clone.

        Stored on the ``UserEnvironment`` row so staleness can be measured
        from the database, where a bare commit hash carries no date.
        """
        if self.install_path is None:
            return None
        out = self._git("log", "-1", "--format=%ct", "HEAD")
        if not (out and out.strip()):
            return None
        try:
            return datetime.fromtimestamp(int(out.strip())).date()
        except Exception as exc:
            logger.debug(f"unreadable HEAD date: {exc}")
            return None

    def head_is_stale(self) -> bool:
        """Return ``True`` if HEAD is older than ``WARN_STALE_DAYS``.

        Committer date as a no-network proxy for months behind upstream: a
        clone that fetches regularly keeps a recent HEAD.
        """
        head_date = self.commit_date
        if head_date is None:
            return False
        return (date.today() - head_date).days > WARN_STALE_DAYS

    # ── Summary ───────────────────────────────────────────────────────────────

    @cached_property
    def is_dev(self) -> bool:
        """``".dev"`` in ``__version__``, marking a non-release build."""
        return ".dev" in self._version

    @cached_property
    def commit_hash(self) -> Optional[str]:
        """Short hash of the running code, or ``None`` for a clean release.

        HEAD wins over ``__version__``, which hatch-vcs writes at install and
        a ``git pull`` does not move. A ``.dYYYYMMDD`` suffix on the version,
        left by a wheel built from a dirty tree, also blocks the hash parse.
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

        In a clone, git is the authority: modified sources under ``src/``, or a
        HEAD that is not reachable from :data:`OFFICIAL_REPO`. Without one, the
        only trace is hatch-vcs's ``.dYYYYMMDD`` suffix on ``__version__``,
        left by a wheel built from a dirty tree.
        """
        if self.install_path is None:
            # A wheel: the .dYYYYMMDD suffix is the only dirty signal. In a
            # clone, _version.py is not rewritten when the tree is cleaned, so
            # the suffix outlives the edits that caused it. Ask git instead.
            return bool(self._VCS_DIRTY_RE.search(self._version))
        return self.has_src_changes() or not self.head_is_published()

    @cached_property
    def is_stale(self) -> bool:
        """HEAD is more than ``WARN_STALE_DAYS`` old. ``False`` without a clone."""
        return self.head_is_stale()

    @cached_property
    def is_official(self) -> bool:
        """Neither dirty nor stale.

        Independent of :attr:`is_dev`: a clean, current clone on any branch of
        the official remote is official though it is a dev build.
        """
        return not self.has_local_changes and not self.is_stale

    # ── Warning and logging ───────────────────────────────────────────────────

    def warn(self) -> None:
        """Warn when running a non-official Spyglass install.

        Touches no database: the git check and the countdown both read local
        state, so an import opens no connection. The record is written on the
        session's first insert or populate, by
        :meth:`~spyglass.common.common_user.UserEnvironment.log_install`.

        Two messages:

        * **Dirty** (``has_local_changes``): the countdown warning. Days are
          counted from the marker's ``first_seen``, written the first time this
          clone was seen dirty.
        * **Stale** (``is_stale``, no local changes): the softer "older
          commit" advisory, with no countdown, since an older official commit
          is reproducible.

        Admins are exempt from both, since they run modified code
        deliberately, but only once :meth:`cached_is_admin` has an answer: the
        check itself is a query and cannot run here. An admin's first flagged
        session is therefore warned, and its first insert caches the exemption
        for the sessions after it.
        """
        if self.cached_is_admin:
            return

        if self.has_local_changes:
            logger.warning(DIRTY_MSG.format(days=self._days_remaining()))
        if self.is_stale:
            logger.warning(STALE_MSG)

    @property
    def cached_is_admin(self) -> Optional[bool]:
        """Whether this user is a lab admin, from the local marker.

        ``LabMember().user_is_admin`` is a query, so the warning cannot ask it
        and reads the answer :meth:`log_install` last cached instead. Not a
        ``cached_property``: the answer can be written during the session that
        reads it.

        Returns
        -------
        bool or None
            ``None`` when no check is cached or the cached one has aged past
            ``WARN_ADMIN_TRUST_DAYS``, so a change in a user's role takes
            effect within that window rather than lasting the life of the
            marker.
        """
        marker = self.marker.read()
        if "is_admin" not in marker:
            return None
        try:
            checked = date.fromisoformat(marker.get("admin_checked", ""))
        except ValueError:  # unreadable date: treat as never checked
            return None
        if (date.today() - checked).days > WARN_ADMIN_TRUST_DAYS:
            return None
        return bool(marker["is_admin"])

    def _days_remaining(self) -> int:
        """Days left before this dirty install is flagged for discussion.

        Counted from the marker's ``first_seen``, so the clock survives a new
        session. A commit made while dirty keeps the original date, so the
        countdown does not restart each time the user commits.
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
        except Exception:  # unreadable date: start the clock over
            days = 0
        return max(0, WARN_DIRTY_ENV_DAYS - days)

    def warn_once(self) -> None:
        """Emit the warning at most once per process.

        Called from the import-time hook in ``spyglass/__init__.py``, fully
        guarded so a warning cannot break an import.
        """
        if self.warned:
            return
        self.warned = True  # set first: a failure must not retry all session
        try:
            self.warn()
        except Exception as exc:
            logger.debug(f"install warning failed: {exc}")


# The install this process is running from. Every fact is a cached property,
# read on first use, so construction costs nothing at import.
spyglass_install = SpyglassInstall()

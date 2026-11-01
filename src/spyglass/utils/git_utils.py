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

_SUBPROCESS_KWARGS = dict(capture_output=True, text=True, timeout=10)

# Days before a flagged dirty install triggers an admin notification.
WARN_DIRTY_ENV_DAYS = 30

# A clean install whose HEAD commit is older than this many days is treated as
# "stale" (months behind upstream).  Being a few commits behind does not flag;
# only an install that has not fetched upstream in a long time does.
WARN_STALE_DAYS = 90

# Set once per process after the first decisive dirty/stale warning so that the
# import-time and lazy (this_env) callers do not double-warn.
_warned = False

# Matches the ``+g<hash>`` VCS suffix appended by setuptools-scm / hatch-vcs.
_VCS_HASH_RE = re.compile(r"\+g([0-9a-f]+)$")

# Matches the `.dYYYYMMDD` suffix hatch-vcs appends when the tree it built from
# had uncommitted changes.
_VCS_DIRTY_RE = re.compile(r"\.d\d{8}$")


class _InstallInfoCache:
    """Singleton that computes Spyglass install info once per process.

    Using ``cached_property`` rather than a module-level mutable avoids the
    ``global`` keyword and makes tests straightforward — create a fresh
    instance to reset the cache rather than reaching into module internals.
    """

    @cached_property
    def info(self) -> dict:
        """Compute and cache the install-info dict."""
        return _compute_install_info()


# Module-level singleton — info computed on first access, then cached.
_install_info = _InstallInfoCache()


def _dirty_marker_path() -> Optional[Path]:
    """Path to the local "this clone has logged a dirty env" marker.

    Scoped per (DataJoint user, conda env) under ``temp_dir``. Using the DB
    user — not the OS login — keeps markers from clashing when many users share
    one ``temp_dir`` (and even one OS account), and aligns the marker with how
    dirty installs are keyed for warnings/emails (``env_id`` prefix). The DB
    user is read from ``dj.config`` (no ``spyglass.common`` import), so the
    clean-install fast path stays lightweight. ``None`` when ``temp_dir`` is
    unavailable.
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
    return Path(temp_dir) / f".spyglass_dirty_{dj_user}_{conda_env}"


def _dirty_marker_exists() -> bool:
    """Return ``True`` if the local dirty marker is present."""
    p = _dirty_marker_path()
    return bool(p and p.exists())


def _read_dirty_marker() -> dict:
    """Return the recorded dirty state for this clone, or an empty dict.

    Keys, when present: ``commit`` the hash first flagged, ``first_seen`` the
    date this episode was first noticed locally, and ``last_logged`` the date
    the database last heard about it.
    """
    p = _dirty_marker_path()
    if p is None or not p.exists():
        return {}
    try:
        return json_loads(p.read_text() or "{}")
    except Exception as exc:
        logger.debug(f"could not read dirty marker: {exc}")
        return {}


def _set_dirty_marker(state: Optional[dict] = None) -> None:
    """Write the local dirty marker (best-effort)."""
    p = _dirty_marker_path()
    if p is None:
        return
    try:
        p.write_text(json_dumps(state or {}))
    except Exception as exc:
        logger.debug(f"could not write dirty marker: {exc}")


def _clear_dirty_marker() -> None:
    """Remove the local dirty marker (best-effort)."""
    p = _dirty_marker_path()
    if p is None:
        return
    try:
        p.unlink(missing_ok=True)
    except Exception as exc:
        logger.debug(f"could not clear dirty marker: {exc}")


def warn_if_dirty(install_info: dict) -> None:
    """Warn when running a non-official Spyglass install.

    Touches no database: the git check and the countdown both read local state,
    so an import never opens a connection, stalls behind a VPN, or prompts for
    a password. The database record is written later, by
    :func:`log_install`, on the session's first insert or populate.

    Two messages:

    * **Dirty** (``has_local_changes``): the countdown warning. Days are
      counted from the local marker's ``first_seen``, written the first time
      this clone was seen dirty.
    * **Stale** (``is_stale``, no local changes): the softer "older commit"
      advisory, with no countdown — an older official commit is reproducible.

    Admin users are not exempted here, because that check is a query. They are
    exempted from the database record and the escalation email.

    Parameters
    ----------
    install_info : dict
        The dict returned by :func:`get_install_info`.
    """
    if install_info.get("has_local_changes"):
        marker = _read_dirty_marker()
        today = date.today()
        first_seen = marker.get("first_seen")
        commit = install_info.get("commit_hash")

        if not first_seen or marker.get("commit") != commit:
            # A new episode, or the same episode at a new commit. Keep the
            # original date in the second case: the user has been dirty
            # throughout, so the clock should not restart on every commit.
            first_seen = first_seen or today.isoformat()
            marker.update(commit=commit, first_seen=first_seen)
            _set_dirty_marker(marker)

        try:
            days = (today - date.fromisoformat(first_seen)).days
        except Exception:  # unreadable date — start the clock over
            days = 0
        remaining = max(0, WARN_DIRTY_ENV_DAYS - days)

        logger.warning(
            "Your Spyglass repository is not on an official commit. "
            "Please ensure you are running Spyglass from a clean "
            "repository to avoid issues with reproducibility.\n\n"
            "If this is an important change, please open a pull request "
            "to merge your changes into the main branch. If not, please "
            "reset your repository to an official commit.\n\n"
            "If you need assistance, please contact the support team. "
            "We will flag this issue for discussion if it is not "
            f"resolved within {remaining} days."
        )
    elif install_info.get("is_stale"):
        logger.warning(
            "Spyglass is being run from an older commit. Please consider "
            "updating to the latest version to ensure you have the latest "
            "features and bug fixes."
        )


def _warn_once() -> None:
    """Run :func:`warn_if_dirty` at most once per process.

    Called from the import-time hook in ``spyglass/__init__.py``. Fully
    guarded — a warning must never break an import.
    """
    global _warned
    if _warned:
        return
    _warned = True  # set first: a failure here should not retry all session
    try:
        warn_if_dirty(get_install_info())
    except Exception as exc:
        logger.debug(f"_warn_once failed: {exc}")


def log_install() -> None:
    """Record a dirty install in ``UserEnvironment``.

    Called from the mixin on the first insert or populate, rather than at
    import: by then a connection exists, so this adds no import-time cost and
    nothing at all in a process that never writes.

    The caller owns the once-per-session gate — see
    ``BaseMixin._logged_install_mode``, which must be set *before* calling this,
    since the insert below re-enters the mixin hook.

    Writes at most once per day per (user, environment). A dirty state already
    on record needs no more than a ``last_seen`` refresh, which the activity
    window resolves in days, and the marker's ``last_logged`` date bounds it
    without a query.

    A clean install with a marker is the resolution case: the full row is
    logged so the escalation clock stops, and the marker is cleared.

    All ``spyglass.common`` imports are deferred inside the body to avoid a
    circular dependency (``common/`` already imports from ``utils/``). Fully
    guarded: a user without write permission sees nothing.
    """
    info = get_install_info()
    flagged = info.get("has_local_changes") or info.get("is_stale")
    marker = _read_dirty_marker()

    # Fast path: a clean install that was never flagged here has nothing to
    # record and no resolution to log — no import, no query.
    if not flagged and not marker:
        return

    today = date.today().isoformat()
    if flagged and marker.get("last_logged") == today:
        return  # already on record today

    try:
        from spyglass.common.common_lab import LabMember
        from spyglass.common.common_user import UserEnvironment
    except Exception as exc:
        logger.debug(f"dirty env logging deferred import failed: {exc}")
        return

    try:
        if LabMember().user_is_admin:
            return
    except Exception as exc:
        logger.debug(f"admin check failed: {exc}")  # DB down — log anyway

    try:
        key = UserEnvironment().insert_current_env()
    except Exception as exc:
        logger.debug(f"dirty env logging failed: {exc}")
        return

    if info.get("has_local_changes"):
        marker["last_logged"] = today
        marker["first_seen"] = (
            _row_first_seen(UserEnvironment, key) or marker["first_seen"]
        )
        _set_dirty_marker(marker)
    else:  # resolution recorded; future clean sessions take the fast path
        _clear_dirty_marker()


def _row_first_seen(user_env_tbl, key: dict) -> Optional[str]:
    """Return the ISO date a logged dirty row first appeared, if readable.

    The database is the shared record, so the warning's countdown defers to it
    and the marker becomes a cache rather than a second clock. Without this,
    clearing ``temp_dir`` — a cache, and cleared as one — restarts the local
    countdown while the escalation email keeps counting, so a user could be
    emailed on the same day they were told they had the full window left.

    Returns ``None`` when there is nothing to read, leaving the caller's local
    date in place: the first warning of a new episode fires before any row
    exists.
    """
    env_id = (key or {}).get("env_id")
    if not env_id:
        return None
    try:
        stamp = (user_env_tbl() & f'env_id="{env_id}"').fetch1("timestamp")
        return stamp.date().isoformat()
    except Exception as exc:
        logger.debug(f"could not read logged dirty date: {exc}")
        return None


def get_install_info() -> dict:
    """Return a dict describing the current Spyglass installation.

    Keys
    ----
    is_dev : bool
        ``True`` when ``".dev"`` appears in ``__version__``, indicating a
        non-release (editable or pre-release) build.
    commit_hash : str or None
        Short git hash from ``git rev-parse`` when running from a local clone,
        or extracted from ``__version__`` (``+g<hash>`` suffix) otherwise.
        ``None`` for clean pip-installed releases. HEAD takes precedence
        because ``_version.py`` is written at install and does not move with a
        ``git pull``.
    install_path : str or None
        Absolute path to the spyglass clone this package runs from, accepted
        only when ``<root>/src/spyglass/__init__.py`` resolves to
        ``spyglass.__file__``.  ``None`` for site-packages installs and for an
        install that merely sits inside someone else's repo.
    has_local_changes : bool
        ``True`` when the install deviates from a commit on the official
        remote: modified sources under ``src/``, a HEAD on no remote branch,
        or a version carrying the ``.dYYYYMMDD`` suffix of a wheel built from
        a dirty tree. Only the last is detectable without a clone, so this is
        otherwise ``False`` when ``install_path`` is ``None``.
    is_stale : bool
        ``True`` when HEAD's commit date is more than ``WARN_STALE_DAYS`` days
        old (the clone is months behind upstream).  Always ``False`` when
        ``install_path`` is ``None``.
    is_official : bool
        ``True`` only when both ``has_local_changes`` and ``is_stale`` are
        ``False`` — i.e. a clean clone that is reasonably current (or a clean
        pip-installed release).  Independent of ``is_dev``.

    Notes
    -----
    Result is cached on the module-level ``_install_info`` singleton via
    ``cached_property``; subsequent calls return the same dict with no
    subprocess overhead.
    """
    return _install_info.info


def _compute_install_info() -> dict:
    """Compute install info — called exactly once per process."""
    from spyglass import __version__

    is_dev = ".dev" in __version__

    # ── commit_hash: HEAD when in a clone, else the version string ───────────
    commit_hash: Optional[str] = None
    if m := _VCS_HASH_RE.search(__version__):
        commit_hash = m.group(1)

    # A wheel built from a dirty tree carries hatch-vcs's `.dYYYYMMDD` suffix.
    # It has no repo to inspect, so this is the only trace of the edits.
    built_dirty = bool(_VCS_DIRTY_RE.search(__version__))

    # ── install_path: the spyglass clone we are running from, if any ─────────
    install_path: Optional[str] = None
    try:
        import spyglass

        pkg_init = Path(spyglass.__file__).resolve()  # …/src/spyglass/__init__
        pkg_path = pkg_init.parent  # …/spyglass/
        for candidate in [pkg_path, *pkg_path.parents]:
            if not (candidate / ".git").exists():
                continue
            # The first .git decides, and it only counts when this spyglass
            # lives in it. A venv inside an unrelated repo (uv's default
            # `.venv`, or `conda create --prefix ./env`) would otherwise adopt
            # that repo's commit and dirty state.
            clone_init = candidate / "src" / "spyglass" / "__init__.py"
            if clone_init.exists() and clone_init.resolve() == pkg_init:
                install_path = str(candidate)
            break
    except Exception as exc:
        logger.debug(f"could not locate install path: {exc}")

    # ── has_local_changes / is_stale: mostly need a git repo to inspect ──────
    has_local_changes = built_dirty  # knowable without one
    is_stale = False
    if install_path is not None:
        # Use HEAD. __version__ does not change after `git pull`, so use it
        # only when git fails.
        commit_hash = _git_short_hash(install_path) or commit_hash

        has_local_changes = (
            has_local_changes
            or _git_has_src_changes(install_path)
            or not _git_head_is_published(install_path)
        )
        is_stale = _git_head_is_stale(install_path)

    return {
        "is_dev": is_dev,
        "commit_hash": commit_hash,
        "install_path": install_path,
        "has_local_changes": has_local_changes,
        "is_stale": is_stale,
        "is_official": not has_local_changes and not is_stale,
    }


def _git_short_hash(repo_root: str) -> Optional[str]:
    """Return the short HEAD hash, or ``None`` if git is unavailable."""
    try:
        result = sub_run(
            ["git", "-C", repo_root, "rev-parse", "--short", "HEAD"],
            **_SUBPROCESS_KWARGS,
        )
        if result.returncode == 0:
            return result.stdout.strip() or None
    except Exception:
        pass
    return None


def _git_head_is_published(repo_root: str) -> bool:
    """Return ``True`` if HEAD is contained by some remote branch.

    ``git diff`` is empty for a commit that exists only locally, so committing
    local work would otherwise read as clean. ``git branch -r --contains``
    needs no network — it reads the remote-tracking refs already on disk, so a
    clone that has never fetched reports nothing contained and is treated as
    unpublished.

    Returns ``True`` on any error, so a git failure does not flag an install.
    """
    try:
        result = sub_run(
            ["git", "-C", repo_root, "branch", "-r", "--contains", "HEAD"],
            **_SUBPROCESS_KWARGS,
        )
        if result.returncode == 0:
            return bool(result.stdout.strip())
        logger.debug(
            f"git branch exited {result.returncode}: {result.stderr.strip()}"
        )
    except FileNotFoundError:
        logger.debug("git not found; assuming HEAD is published")
    except Exception as exc:
        logger.debug(f"git branch failed: {exc}")
    return True


def _git_has_src_changes(repo_root: str) -> bool:
    """Return ``True`` if tracked files under ``src/`` differ from HEAD.

    Uses ``git diff --name-only HEAD -- src/`` which covers both staged and
    unstaged modifications to tracked files.  Untracked files and changes
    outside ``src/`` are intentionally excluded.
    """
    try:
        result = sub_run(
            [
                "git",
                "-C",
                repo_root,
                "diff",
                "--name-only",
                "HEAD",
                "--",
                "src/",
            ],
            **_SUBPROCESS_KWARGS,
        )
        if result.returncode == 0:
            return bool(result.stdout.strip())
        logger.debug(
            f"git diff exited {result.returncode}: {result.stderr.strip()}"
        )
    except FileNotFoundError:
        logger.debug("git not found; assuming no local changes")
    except Exception as exc:
        logger.debug(f"git diff failed: {exc}")
    return False


def _git_head_is_stale(repo_root: str) -> bool:
    """Return ``True`` if HEAD's commit date is older than ``WARN_STALE_DAYS``.

    Uses ``git log -1 --format=%ct HEAD`` (committer date, Unix epoch) as a
    no-network proxy for "months behind upstream": a clone that fetches
    regularly keeps a recent HEAD, while a long-neglected one does not.
    Returns ``False`` (fail-safe — do not flag) on any error.
    """
    try:
        result = sub_run(
            ["git", "-C", repo_root, "log", "-1", "--format=%ct", "HEAD"],
            **_SUBPROCESS_KWARGS,
        )
        if result.returncode != 0:
            logger.debug(
                f"git log exited {result.returncode}: {result.stderr.strip()}"
            )
            return False
        stamp = result.stdout.strip()
        if not stamp:
            return False
        head_date = datetime.fromtimestamp(int(stamp))
        return (datetime.now() - head_date).days > WARN_STALE_DAYS
    except FileNotFoundError:
        logger.debug("git not found; assuming not stale")
    except Exception as exc:
        logger.debug(f"git log failed: {exc}")
    return False

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
2. ``commit_hash`` — parsed from the ``+g<hash>`` suffix in ``__version__``
   before falling back to ``git rev-parse --short HEAD``.  ``None`` for
   standard pip installs of a release.
3. ``install_path`` — the repo root if a ``.git`` directory is found as an
   ancestor of ``spyglass.__file__``; ``None`` for site-packages installs.
   Only set when the package is running from a local clone.
4. ``has_local_changes`` — ``git diff --name-only HEAD -- src/`` returns
   non-empty output.  Scoped to ``src/`` so doc edits, notebook changes, and
   ``.claude/`` artefacts are intentionally ignored.
5. ``is_stale`` — HEAD's commit date is more than ``WARN_STALE_DAYS`` days
   old, i.e. the clone has not fetched upstream in months.  Being a handful
   of commits behind does not flag; only long-neglected clones do.
6. ``is_official`` — ``not has_local_changes and not is_stale``.  Note this
   is independent of ``is_dev``: a clone tracking ``master`` that is current
   and unmodified is treated as official even though it is a dev build.
"""

import os
import re
from datetime import datetime
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


def _set_dirty_marker() -> None:
    """Create the local dirty marker (best-effort)."""
    p = _dirty_marker_path()
    if p is None:
        return
    try:
        p.touch()
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


def warn_if_dirty(install_info: dict) -> bool:
    """Warn non-admin users when running a non-official Spyglass install.

    Emits a ``logger.warning`` for two cases:

    * **Dirty** (``has_local_changes=True``): full countdown warning.
    * **Stale** (``is_stale=True``, no local changes): softer "older commit"
      advisory for clones that are months behind upstream.

    Logging behaviour (to keep the escalation clock honest while minimising
    writes):

    * **Flagged** install: log the current env to ``UserEnvironment``
      (idempotent) and set a local marker, so the 30-day clock starts at first
      warning rather than only at recompute. The countdown is derived from the
      logged timestamp.
    * **Clean** install: only if the local marker is present (this clone was
      flagged here before) and the user's most-recent log is dirty within twice
      the warn window, log the (clean) env to record the *resolution* — which
      lets :meth:`UserEnvironment.dirty_installs` drop the now-stale dirty row.
      A clean install with no marker skips the ``common`` import and DB entirely.

    Returns silently for official installs and admin users.  All
    ``spyglass.common`` imports are deferred inside the function body to avoid
    a circular dependency (``common/`` already imports from ``utils/``).

    Parameters
    ----------
    install_info : dict
        The dict returned by :func:`get_install_info`.

    Returns
    -------
    bool
        ``True`` when a decision was reached (official, admin, or a warning
        emitted); ``False`` when the call could not proceed because
        ``spyglass.common`` / the DB was not importable yet — signalling the
        caller it may retry later.
    """
    flagged = install_info.get("has_local_changes") or install_info.get(
        "is_stale"
    )

    # Fast path: a clean install that has never logged a dirty env from this
    # clone (no marker) needs no warning and has no resolution to record —
    # skip the common import / DB entirely.
    if not flagged and not _dirty_marker_exists():
        return True

    # Deferred imports — common/ imports utils/, so top-level imports here
    # would create a circular dependency.
    try:
        import datajoint as dj

        from spyglass.common.common_lab import LabMember
        from spyglass.common.common_user import (
            DIRTY_DAYS_SQL,
            UserEnvironment,
        )
    except Exception as exc:
        logger.debug(f"warn_if_dirty deferred import failed: {exc}")
        return False  # not ready — let the caller retry later

    try:
        if LabMember().user_is_admin:
            return True
    except Exception as exc:
        logger.debug(f"admin check failed: {exc}")  # DB down — warn anyway

    dj_user = dj.config.get("database.user", "")
    user_restr = f"env_id LIKE '{dj_user}_%'"

    if not flagged:
        # Clean now, but the marker says this clone was flagged before. Record
        # a resolution (log the clean env) when the most-recent log is dirty
        # and within 2x the warn window, so dirty_installs() drops the stale
        # dirty row. Clear the marker either way so future clean imports stay
        # on the fast path.
        try:
            rows = (
                (UserEnvironment & user_restr)
                .proj("dirty_path", age_days=DIRTY_DAYS_SQL)
                .fetch("dirty_path", "age_days", as_dict=True)
            )
            if rows:  # most-recent log == smallest age (DATEDIFF from NOW)
                newest = min(rows, key=lambda r: r["age_days"])
                if (
                    newest["dirty_path"] is not None
                    and newest["age_days"] <= 2 * WARN_DIRTY_ENV_DAYS
                ):
                    UserEnvironment().insert_current_env()
        except Exception as exc:
            logger.debug(f"resolution logging failed: {exc}")
        _clear_dirty_marker()
        return True

    # Flagged: log the current env so the 30-day clock runs (idempotent via
    # matching_env_id; disk-cached conda export; called directly, not via
    # this_env, to avoid recursing into _warn_once) and mark this clone dirty
    # so a later clean import records the resolution.
    try:
        UserEnvironment().insert_current_env()
        _set_dirty_marker()
    except Exception as exc:
        logger.debug(f"dirty env logging failed: {exc}")

    # Days since this user's dirty install was first logged. Computed DB-side
    # (shared DIRTY_DAYS_SQL) so it agrees with the admin notification and is
    # not skewed by client-vs-server timezone. Oldest row → largest count.
    days = 0
    try:
        dirty = UserEnvironment & user_restr & "dirty_path IS NOT NULL"
        if dirty:
            day_vals = dirty.proj(age_days=DIRTY_DAYS_SQL).fetch("age_days")
            days = int(max(day_vals)) if len(day_vals) else 0
    except Exception as exc:
        logger.debug(f"dirty day-count failed: {exc}")

    N = max(0, WARN_DIRTY_ENV_DAYS - days)

    if install_info.get("has_local_changes"):
        logger.warning(
            "Your Spyglass repository is not on an official commit. "
            "Please ensure you are running Spyglass from a clean "
            "repository to avoid issues with reproducibility.\n\n"
            "If this is an important change, please open a pull request "
            "to merge your changes into the main branch. If not, please "
            "reset your repository to an official commit.\n\n"
            "If you need assistance, please contact the support team. "
            "We will flag this issue for discussion if it is not "
            f"resolved within {N} days."
        )
    elif install_info.get("is_stale"):
        logger.warning(
            "Spyglass is being run from an older commit. Please consider "
            "updating to the latest version to ensure you have the latest "
            "features and bug fixes."
        )

    return True


def _warn_once() -> None:
    """Run :func:`warn_if_dirty` at most once per process.

    Used by both the import-time hook (``spyglass/__init__.py``) and the lazy
    path (``UserEnvironment.this_env``).  The flag is only set when
    ``warn_if_dirty`` reports a decisive result, so an early import-time call
    that finds the DB not yet ready will be retried by the lazy path.
    """
    global _warned
    if _warned:
        return
    try:
        if warn_if_dirty(get_install_info()):
            _warned = True
    except Exception as exc:
        logger.debug(f"_warn_once failed: {exc}")


def get_install_info() -> dict:
    """Return a dict describing the current Spyglass installation.

    Keys
    ----
    is_dev : bool
        ``True`` when ``".dev"`` appears in ``__version__``, indicating a
        non-release (editable or pre-release) build.
    commit_hash : str or None
        Short git hash extracted from ``__version__`` (``+g<hash>`` suffix),
        or obtained via ``git rev-parse`` when running from a local clone.
        ``None`` for clean pip-installed releases.
    install_path : str or None
        Absolute path to the repository root when running from a local clone
        (i.e. a ``.git`` directory is found as an ancestor of
        ``spyglass.__file__``).  ``None`` for site-packages installs.
    has_local_changes : bool
        ``True`` when ``git diff --name-only HEAD -- src/`` produces output,
        meaning tracked source files have been modified relative to HEAD.
        Always ``False`` when ``install_path`` is ``None``.
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

    # ── commit_hash: prefer version string, fall back to git ─────────────────
    commit_hash: Optional[str] = None
    if m := _VCS_HASH_RE.search(__version__):
        commit_hash = m.group(1)

    # ── install_path: walk up from spyglass package looking for .git ─────────
    install_path: Optional[str] = None
    try:
        import spyglass

        pkg_path = Path(spyglass.__file__).resolve().parent  # …/spyglass/
        for candidate in [pkg_path, *pkg_path.parents]:
            if (candidate / ".git").exists():
                install_path = str(candidate)
                break
    except Exception as exc:
        logger.debug(f"could not locate install path: {exc}")

    # ── has_local_changes / is_stale: only meaningful inside a git repo ──────
    has_local_changes = False
    is_stale = False
    if install_path is not None:
        # Obtain commit_hash from git if not already parsed from __version__
        if commit_hash is None:
            commit_hash = _git_short_hash(install_path)

        has_local_changes = _git_has_src_changes(install_path)
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

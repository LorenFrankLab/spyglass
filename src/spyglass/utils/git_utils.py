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
5. ``is_official`` — ``not is_dev and not has_local_changes``.
"""

import re
from functools import cached_property
from pathlib import Path
from subprocess import run as sub_run
from typing import Optional

from spyglass.utils import logger

_SUBPROCESS_KWARGS = dict(capture_output=True, text=True, timeout=10)

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
    is_official : bool
        ``True`` only when both ``is_dev`` and ``has_local_changes`` are
        ``False`` — i.e. an unmodified, tagged release install.

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
    except Exception:
        pass

    # ── has_local_changes: only meaningful inside a git repo ─────────────────
    has_local_changes = False
    if install_path is not None:
        # Obtain commit_hash from git if not already parsed from __version__
        if commit_hash is None:
            commit_hash = _git_short_hash(install_path)

        has_local_changes = _git_has_src_changes(install_path)

    return {
        "is_dev": is_dev,
        "commit_hash": commit_hash,
        "install_path": install_path,
        "has_local_changes": has_local_changes,
        "is_official": not is_dev and not has_local_changes,
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

"""Shared conda-environment cache for Spyglass.

Avoids redundant ``conda env export`` / ``pip freeze`` subprocess calls by
writing the raw subprocess outputs to a per-(user, conda-env) YAML file in
``temp_dir``.  The singleton ``_env_cache`` exported from this module is the
single source of truth.

Cache-file format
-----------------
``{temp_dir}/.spyglass_conda_{os_user}_{md5(sys.prefix)[:8]}.yaml``

Contains two top-level keys::

    conda: <raw conda env export dict: channels + dependencies>
    pip:   <raw pip freeze list>

The merge of these two raw artifacts (conflict resolution, editable-install
detection) is performed **in-process on demand** via ``get(with_pip=True)``
and is never written back to disk.

Who uses which mode
-------------------
``get()`` (conda-only, default)
    ``AnalysisMixin._logged_env_info`` and ``SQLDumpHelper._export_conda_env``
    receive the raw ``conda env export`` output, with no pip processing.

``get(with_pip=True)`` (merged)
    ``UserEnvironment``. Adds conflict resolution, pip-only package
    discovery, and editable-install detection.

Cache staleness
---------------
The cache is stale when any signal that moves with the environment is at least
as new as the cache file: ``conda-meta/history`` for conda transactions, and
the site-packages directories for pip, which adds or removes a ``*.dist-info``
directory on install, upgrade, uninstall and reinstall.

The guarantee is that the cache is as accurate as re-running the export. Edits
to source inside an editable install move no mtime, but they change no
``pip freeze`` line either.

Atomic writes
-------------
``save()`` writes to ``{cache_path}.{pid}.tmp`` then calls ``os.replace()``,
a single POSIX ``rename`` syscall.  Last writer wins; both produce valid YAML.
"""

import getpass
import os
import re
import sysconfig
from functools import cached_property
from hashlib import md5
from pathlib import Path
from pprint import pprint
from subprocess import run as sub_run
from sys import executable as sys_executable
from sys import prefix as sys_prefix
from typing import Optional, Tuple, Union

import yaml

from spyglass.utils import logger

SUBPROCESS_KWARGS = dict(capture_output=True, text=True, timeout=60)

# ── Sentinel so get() can distinguish "not yet computed" from "{}" ──────────
_NOT_COMPUTED = object()

# ── Module-level regex constants (compiled once per process) ─────────────────

_IGNORED_CONDA_SOURCE = re.compile(
    r"[/\\](?:feedstock_root|croot|conda-bld|bld|builder)[/\\]"
)

_BASIC_DEP_FORMAT = re.compile(
    r"""^
        (?P<pkg>[\w\d.-]+)          # package name
        ==                          # separator
        (?P<ver>
            \d+(?:\.\d+)*           # major.minor.patch
            (?:                     # optional suffix
                (?P<alpha>a\d+)   |
                (?P<beta>b\d+)    |
                (?P<rc>rc\d+)     |
                (?P<post>\.post\d+) |
                (?P<dev>dev\d*)   |
                (?P<dfsg>\+dfsg)  |
                (?P<ds>\+ds)      |
                (?P<ubuntu>\+ubuntu\d+(?:\.\d+)*)
            )?
        )
    $""",
    re.VERBOSE,
)

_COMMENT_INSTALL = re.compile(
    r"^\s*#.*?\((?P<pkg>[A-Za-z0-9_.-]+)==(?P<ver>[^)]+)\)$"
)

_EDITABLE_PATH = re.compile(r"^-e[ ]+(?P<path>\S+)$")


# ── Pure helper (no instance state needed) ───────────────────────────────────


def _delim_split(
    line: Optional[str], as_dict: bool = False
) -> Union[Tuple[str, str], dict]:
    """Split a dependency string on the first recognised delimiter.

    Recognised delimiters (tried in order): ``==``, `` @ ``, ``=``.
    Returns a ``(name, value)`` tuple, or a single-entry dict when
    ``as_dict=True``.  A list input is processed element-wise.
    """
    if isinstance(line, list):
        parts = [_delim_split(row) for row in line]
        return {p[0]: p[1] for p in parts} if as_dict else parts

    if line is None or not isinstance(line, str):
        ret: Tuple[str, Optional[str]] = ("", None)
    else:
        for delim in ("==", " @ ", "="):
            if delim in line:
                ret = tuple(line.split(delim, maxsplit=1))  # type: ignore[assignment]
                break
        else:
            ret = (line, "")

    return {ret[0]: ret[1]} if as_dict else ret


# ── Main class ───────────────────────────────────────────────────────────────


class CondaEnvCache:
    """Per-(user, conda-env) cache for conda environment data.

    Attributes
    ----------
    has_editable : bool
        ``True`` when a dependency is installed editable. Recompute distrusts
        those, because an editable package's version string says nothing about
        its contents. Spyglass is exempt: its upstream is known, so an edited
        clone is detected through git. Only meaningful after a
        ``get(with_pip=True)`` call.
    """

    def __init__(self) -> None:
        # Raw subprocess outputs, written to disk
        self._cached_conda = _NOT_COMPUTED  # raw conda env export dict
        self._cached_pip = _NOT_COMPUTED  # raw pip freeze list

        # Merged result, computed in-process and never written to disk
        self._merged = _NOT_COMPUTED

        self.has_editable: bool = False

        # Merge state, repopulated on each _merge_conda_pip() call
        self._conda_pip_dict: dict = {}
        self._pip_custom: dict = {}
        self._freeze_comments: dict = {}
        self._conda_conflicts: dict = {}

    # ── Path helpers (lazy, computed once per process) ───────────────────────

    @cached_property
    def _cache_path(self) -> Optional[Path]:
        """Scoped cache file path, or ``None`` when ``temp_dir`` is unavailable.

        Keyed on a digest of ``sys.prefix`` rather than the conda env name, so
        two hosts whose environments share a name but not a prefix do not share
        one cache file on a common ``temp_dir``.
        """
        from spyglass.settings import temp_dir  # deferred to avoid import cycle

        if not temp_dir:
            return None

        os_user = getpass.getuser()
        prefix_key = md5(sys_prefix.encode()).hexdigest()[:8]
        return Path(temp_dir) / f".spyglass_conda_{os_user}_{prefix_key}.yaml"

    @cached_property
    def _history_path(self) -> Optional[Path]:
        """``conda-meta/history`` path, or ``None`` if ``CONDA_PREFIX`` is unset."""
        prefix = os.environ.get("CONDA_PREFIX")
        if not prefix:
            return None
        candidate = Path(prefix) / "conda-meta" / "history"
        return candidate if candidate.exists() else None

    @cached_property
    def _site_packages_paths(self) -> Tuple[Path, ...]:
        """Existing site-packages directories for this interpreter.

        ``purelib`` and ``platlib`` are the same directory on most platforms
        and differ on some; both are checked, deduplicated.
        """
        paths = {
            sysconfig.get_paths().get(key) for key in ("purelib", "platlib")
        }
        return tuple(p for p in (Path(x) for x in paths if x) if p.is_dir())

    # ── Cache lifecycle ───────────────────────────────────────────────────────

    def is_stale(self) -> bool:
        """Return ``True`` when the cache file is absent or the env has changed.

        Compares the cache file against every signal that moves when the
        environment changes: ``conda-meta/history`` for conda transactions, and
        the site-packages directories for pip, which adds or removes a
        ``*.dist-info`` directory on install, upgrade, uninstall and
        reinstall. An equal mtime counts as stale, since filesystem timestamp
        granularity can place a change in the same tick as the write.

        A clock difference between a ``temp_dir`` and a site-packages on
        separate filesystems can mask a change.
        """
        if self._cache_path is None or not self._cache_path.exists():
            return True

        sources = [
            p
            for p in (self._history_path, *self._site_packages_paths)
            if p is not None
        ]
        if not sources:  # nothing to compare against: assume changed
            return True

        cache_mtime = self._cache_path.stat().st_mtime
        return any(p.stat().st_mtime >= cache_mtime for p in sources)

    def load(self) -> bool:
        """Read raw conda and pip data from the disk cache.

        Populates ``_cached_conda`` from the YAML file. ``_cached_pip`` is set
        only when the file actually carries a ``pip`` list; an absent/``None``
        ``pip`` key leaves it as ``_NOT_COMPUTED`` so a ``with_pip`` caller will
        run ``pip freeze`` rather than mistaking a conda-only cache for an
        environment with no pip packages. Returns ``True`` on success,
        ``False`` on any error or missing ``conda`` key.
        """
        try:
            with self._cache_path.open("r") as fh:
                data = yaml.safe_load(fh)
            if not isinstance(data, dict) or "conda" not in data:
                return False
            self._cached_conda = data["conda"]
            if data.get("pip") is not None:
                self._cached_pip = data["pip"]
            return True
        except Exception:
            return False

    def save(self) -> None:
        """Atomically write raw conda and pip data to the cache file.

        Writes to a PID-scoped temp file first, then renames: a single POSIX
        ``rename`` syscall, atomic even under concurrent writers.
        Only the raw subprocess outputs are persisted; the merged result is
        always recomputed in-process.

        The ``pip`` key is written only when ``pip freeze`` has actually run.
        A conda-only fetch persists ``{conda}`` alone so a later ``with_pip``
        reader knows pip is still missing rather than treating an empty list as
        "no pip packages".
        """
        if self._cache_path is None or self._cached_conda is _NOT_COMPUTED:
            return
        data = {"conda": self._cached_conda}
        if self._cached_pip is not _NOT_COMPUTED:
            data["pip"] = self._cached_pip
        tmp = self._cache_path.with_suffix(f".{os.getpid()}.tmp")
        try:
            with tmp.open("w") as fh:
                yaml.dump(data, fh)
            os.replace(tmp, self._cache_path)
        except Exception:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass

    def _touch(self) -> None:
        """Mark the cache file as still in use.

        ``cleanup.py`` deletes anything in ``temp_dir`` untouched for seven
        days, and a cache hit does not rewrite the file, so a stable
        environment would have its cache swept weekly and re-export on the
        next session.

        Safe only here, after :meth:`is_stale` has confirmed the file is newer
        than every staleness signal: bumping it to now preserves that ordering,
        and any later change moves a signal past the new mtime. Touching
        before the check would hide a change.
        """
        if self._cache_path is None:
            return
        try:
            self._cache_path.touch()
        except Exception as exc:
            logger.debug(f"could not touch env cache: {exc}")

    def preload(self) -> None:
        """Warm the in-process cache from disk if the disk cache is fresh.

        When the disk cache is stale or absent this is a no-op; the subprocess
        runs lazily on the first ``get()`` call instead.  Useful in tests that
        need a synchronous warm-up.
        """
        if self._cached_conda is not _NOT_COMPUTED:
            return
        if not self.is_stale():
            self.load()

    # ── Public API ────────────────────────────────────────────────────────────

    def get(self, with_pip: bool = False) -> dict:
        """Return the conda environment dict.

        Parameters
        ----------
        with_pip : bool, optional
            Default ``False``.

            * ``False``: the raw ``conda env export`` output
              (``channels`` + ``dependencies`` from conda's perspective).
              This is what ``AnalysisMixin`` and ``SQLDumpHelper`` use; it
              matches their pre-cache behaviour exactly.

            * ``True``: additionally run ``pip freeze``, resolve conda/pip
              version conflicts (pip wins), fold pip-only packages into the
              dependency list, and collect editable / non-standard installs
              into a ``custom`` section.  Pass ``True`` only when an accurate
              pip-inclusive picture is needed (``UserEnvironment``).

        ``pip freeze`` is only ever run when ``with_pip=True``; conda-only
        callers never pay that cost (nor its failure/noise) on a cold cache.
        """
        self._ensure_raw(need_pip=with_pip)
        if not with_pip:
            return (
                self._cached_conda
                if self._cached_conda is not _NOT_COMPUTED
                else {}
            )
        return self._get_merged()

    # ── Internal: raw-data layer ──────────────────────────────────────────────

    def _ensure_raw(self, need_pip: bool = False) -> None:
        """Ensure raw data is populated, fetching ``pip freeze`` only if asked.

        Order of precedence for each piece (conda always; pip only when
        ``need_pip``):
        1. Already in-process: no I/O.
        2. Fresh disk cache: one file read.
        3. Subprocess: ``conda env export`` and, when ``need_pip``,
           ``pip freeze``.

        Disk is (re)written only when something was fetched. A conda-only
        fetch persists ``{conda}`` alone (no ``pip`` key); a later
        ``with_pip`` caller, in this process or another, sees ``_cached_pip``
        absent, runs ``pip freeze``, and rewrites the complete file. A partial
        cache is never written, so no reader mistakes one for "no pip
        packages".
        """
        fetched = False
        if self._cached_conda is _NOT_COMPUTED:
            if not self.is_stale() and self.load():
                self._touch()  # loaded from disk; keep it off the sweep
            else:
                self._fetch_conda()
                fetched = True
        if need_pip and self._cached_pip is _NOT_COMPUTED:
            self._fetch_pip()
            fetched = True
        if fetched:
            self.save()

    def _fetch_conda(self) -> None:
        """Run ``conda env export``, storing the raw conda dict.

        Resets all state first so the method is safe to call more than once
        (e.g. in tests).
        """
        self._reset_state()

        conda_result = sub_run(["conda", "env", "export"], **SUBPROCESS_KWARGS)
        if conda_result.returncode != 0:
            logger.error(  # pragma: no cover
                "Failed to retrieve the Conda environment. "
                "Recompute feature disabled."
            )
            return  # pragma: no cover

        self._cached_conda = {
            k: v
            for k, v in yaml.safe_load(conda_result.stdout).items()
            if k not in ("name", "prefix")
        }

    def _fetch_pip(self) -> None:
        """Run ``pip freeze``, storing the raw pip list.

        Augments an already-fetched or disk-loaded conda export with pip data,
        leaving conda state alone.
        """
        # Use the running interpreter's pip, not whatever is first on PATH:
        # PATH may hold another env's pip, or none at all when the env's
        # python is invoked by absolute path without activation.
        pip_result = sub_run(
            [sys_executable, "-m", "pip", "freeze"], **SUBPROCESS_KWARGS
        )
        if pip_result.returncode != 0:
            logger.error(  # pragma: no cover
                "Failed to retrieve the pip environment. "
                "Recompute feature disabled."
            )
            return  # pragma: no cover

        self._cached_pip = [
            line.strip() for line in pip_result.stdout.splitlines()
        ]

    def _reset_state(self) -> None:
        self._cached_conda = _NOT_COMPUTED
        self._cached_pip = _NOT_COMPUTED
        self._merged = _NOT_COMPUTED
        self.has_editable = False
        self._conda_pip_dict = {}
        self._pip_custom = {}
        self._freeze_comments = {}
        self._conda_conflicts = {}

    # ── Internal: merge layer (UserEnvironment only) ──────────────────────────

    def _get_merged(self) -> dict:
        """Return the merged conda+pip dict, computing it in-process if needed."""
        if self._merged is not _NOT_COMPUTED:
            return self._merged  # type: ignore[return-value]
        self._merged = self._merge_conda_pip()
        return self._merged

    def _merge_conda_pip(self) -> dict:
        """Merge raw conda export and pip freeze into a unified env dict.

        * Pip-only packages (not tracked by conda) are added to the pip
          sub-list under ``dependencies``.
        * When conda and pip disagree on a version, pip wins and the
          conflict is recorded for logging.
        * Editable and non-standard installs are collected into ``custom``.

        Returns an empty dict if either raw input is unavailable.
        """
        if self._cached_conda is _NOT_COMPUTED or not self._cached_conda:
            return {}

        # Reset merge state so re-runs are safe
        self.has_editable = False
        self._conda_pip_dict = {}
        self._pip_custom = {}
        self._freeze_comments = {}
        self._conda_conflicts = {}

        conda_export = dict(
            self._cached_conda
        )  # shallow copy; the cache must not be mutated
        conda_deps: list = list(conda_export.get("dependencies", []))

        # Extract the pip sub-list that conda embeds inside dependencies.
        # Both copies above are shallow, so `dep` is the dict held by
        # `_cached_conda`. Read it; a `pop` here empties the cached export's
        # pip entry for every later conda-only `get()`.
        for i, dep in enumerate(conda_deps):
            if isinstance(dep, dict):
                conda_deps.pop(i)  # drop from the copy only
                for entry in dep.get("pip", []):
                    pkg = entry.split("==", maxsplit=1)[0]
                    self._conda_pip_dict[pkg] = entry
                break
        conda_export["dependencies"] = conda_deps

        # Walk pip freeze, extending / overriding _conda_pip_dict
        pip_list = (
            self._cached_pip if self._cached_pip is not _NOT_COMPUTED else []
        )
        for line in pip_list:
            if not self._parse_pip_line(line):
                return {}  # parse error already logged

        self._warn_if_custom_or_conflict()

        # Reassemble the dependencies list with the merged pip sub-list
        conda_deps = conda_export.pop("dependencies", [])
        conda_deps.append({"pip": list(self._conda_pip_dict.values())})
        conda_export["dependencies"] = conda_deps

        if self._pip_custom:
            conda_export["custom"] = self._pip_custom

        return conda_export

    def _parse_pip_line(self, line: str) -> bool:
        """Classify and record one line from ``pip freeze`` output.

        Returns ``True`` on success, ``False`` if the line cannot be parsed
        (which signals the caller to abort and disable the recompute feature).

        Raises
        ------
        ValueError
            If an editable path is encountered without a preceding comment line.
        """
        # ── Prebuilt conda packages (trusted source paths) ────────────────────
        if _IGNORED_CONDA_SOURCE.search(line):
            return True

        # ── Standard ``name==version`` format ────────────────────────────────
        if match := _BASIC_DEP_FORMAT.match(line):
            package, pip_version = match.group("pkg", "ver")
            conda_line = self._conda_pip_dict.get(package)
            _, conda_version = _delim_split(conda_line)

            if line == conda_line:  # conda and pip agree exactly
                return True

            if conda_version is None:  # pip-only package
                logger.debug(f"Pip-only package  : {line}")
                self._conda_pip_dict[package.lower()] = line.lower()
                return True

            # Conflicting versions: record, and let pip win
            logger.debug(f"Conda/pip conflict: {line}")
            self._conda_conflicts[package] = [pip_version, conda_version]
            self._conda_pip_dict[package] = line
            return True

        # ── ``name @ file://…`` or ``name @ git+…`` ──────────────────────────
        if " @ " in line and ("file://" in line or "git+" in line):
            dep_name, _ = _delim_split(line)
            self._pip_custom[dep_name] = line
            logger.debug(f"Custom pip install: {dep_name}")
            return True

        # ── ``-e git+https://…#egg=name`` (editable from git URL) ────────────
        if line.startswith("-e git+") and "#egg=" in line:
            _, package = line.split("#egg=", maxsplit=1)
            package = package.replace("_", "-")
            logger.debug(f"Editable from git : {package}")
            self._pip_custom[package] = line
            if "spyglass" not in package:
                self.has_editable = True
            return True

        # ── Comment line preceding an editable path ───────────────────────────
        if match := _COMMENT_INSTALL.match(line):
            pkg = match.group("pkg").replace("_", "-")
            self._freeze_comments[pkg] = match.group("ver")
            return True

        # ── ``-e /local/path`` (editable install from local path) ─────────────
        if match := _EDITABLE_PATH.match(line):
            path = match.group("path")
            path_name = Path(path).name.replace("_", "-")
            comment = self._freeze_comments.get(path_name)
            if comment is None:
                raise ValueError(
                    f"Editable path found without a preceding comment: {line}"
                )
            logger.info(f"Editable from path: {line}")
            self._pip_custom[path_name] = f"{comment} @ {path}"
            if "spyglass" not in path_name and "jsonschema" not in path_name:
                self.has_editable = True
            return True

        logger.error(f"Failed to parse: {line}\nRecompute feature disabled.")
        return False

    def _warn_if_custom_or_conflict(self) -> None:
        """Log warnings for non-standard pip installs and conda/pip conflicts.

        Two packages are exempt: spyglass, often editable so updates can be
        fetched without reinstalling, and jsonschema, which conda installs from
        a temp directory.
        """
        pip_custom_no_spy = {
            k: v
            for k, v in self._pip_custom.items()
            if "spyglass" not in k and "jsonschema" not in k
        }
        if pip_custom_no_spy:
            logger.warning(
                "Custom pip installs found. "
                "Recompute feature may not work as expected."
            )
            pprint(pip_custom_no_spy, indent=4)

        if self._conda_conflicts:
            logger.warning(  # pragma: no cover
                "Conda/pip conflicts in the environment.\n"
                "\tRecompute feature may not work as expected.\n"
                "\tUse `pip uninstall pkg` to defer to conda.\n"
                "\tConflicts as Package : [pip_version, conda_version]"
            )
            pprint(self._conda_conflicts, indent=4)  # pragma: no cover


# ── Module-level singleton ────────────────────────────────────────────────────

_env_cache = CondaEnvCache()

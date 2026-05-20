"""Shared conda-environment cache for Spyglass.

Avoids redundant ``conda env export`` / ``pip freeze`` subprocess calls by
writing the parsed result to a per-(user, conda-env) YAML file in
``temp_dir``.  The singleton ``_env_cache`` exported from this module is the
single source of truth; all three callers that previously shelled out
independently (``UserEnvironment``, ``AnalysisMixin``, ``SQLDumpHelper``)
will delegate to it after Phase 6.

Cache-file path
---------------
``{temp_dir}/.spyglass_conda_{os_user}_{conda_env}.yaml``

``os_user``   — ``getpass.getuser()``   (OS login name, not the DB user)
``conda_env`` — ``$CONDA_DEFAULT_ENV``  (defaults to ``"base"``)

Two users sharing the same ``temp_dir`` get separate files; two conda
environments active at once for the same user also get separate files.

Staleness
---------
``conda-meta/history`` is updated by every ``conda install`` / ``remove``.
The cache is stale when that file is newer than the cache file, or when
``$CONDA_PREFIX`` is absent (pip-only environments).

Atomic writes
-------------
``save()`` writes to ``{cache_path}.{pid}.tmp`` then calls ``os.replace()``,
a single POSIX ``rename`` syscall.  Last writer wins; both produce valid YAML.
"""

import getpass
import os
import re
from functools import cached_property
from pathlib import Path
from pprint import pprint
from subprocess import run as sub_run
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
    """Per-(user, conda-env) cache for the fully-parsed conda environment.

    Attributes
    ----------
    has_editable : bool
        Set to ``True`` during ``_fetch_from_subprocess`` if any non-spyglass,
        non-jsonschema editable install is found.  Retained on the instance so
        ``UserEnvironment.insert_current_env`` can read it after Phase 6.
    """

    def __init__(self) -> None:
        self._cached_result = _NOT_COMPUTED
        self.has_editable: bool = False

        # Parsing state — reset at the top of every _fetch_from_subprocess call
        self._conda_export: dict = {}
        self._conda_pip_dict: dict = {}
        self._pip_freeze: list = []
        self._pip_custom: dict = {}
        self._freeze_comments: dict = {}
        self._conda_conflicts: dict = {}

    # ── Path helpers (lazy, computed once per process) ───────────────────────

    @cached_property
    def _cache_path(self) -> Optional[Path]:
        """Scoped cache file path, or ``None`` when ``temp_dir`` is unavailable."""
        from spyglass.settings import temp_dir  # deferred to avoid import cycle

        if not temp_dir:
            return None

        os_user = getpass.getuser()
        conda_env = os.environ.get("CONDA_DEFAULT_ENV", "base")
        return Path(temp_dir) / f".spyglass_conda_{os_user}_{conda_env}.yaml"

    @cached_property
    def _history_path(self) -> Optional[Path]:
        """``conda-meta/history`` path, or ``None`` if ``CONDA_PREFIX`` is unset."""
        prefix = os.environ.get("CONDA_PREFIX")
        if not prefix:
            return None
        candidate = Path(prefix) / "conda-meta" / "history"
        return candidate if candidate.exists() else None

    # ── Cache lifecycle ───────────────────────────────────────────────────────

    def is_stale(self) -> bool:
        """Return ``True`` when the cache file is absent or older than conda history."""
        if self._cache_path is None or not self._cache_path.exists():
            return True
        if self._history_path is None:
            return True
        return (
            self._history_path.stat().st_mtime
            > self._cache_path.stat().st_mtime
        )

    def load(self) -> Optional[dict]:
        """Deserialise the YAML cache file; return ``None`` on any error."""
        try:
            with self._cache_path.open("r") as fh:
                return yaml.safe_load(fh)
        except Exception:
            return None

    def save(self, env_dict: dict) -> None:
        """Atomically write *env_dict* to the cache file.

        Writes to a PID-scoped temp file first, then renames — a single POSIX
        ``rename`` syscall that is atomic even under concurrent writers.
        """
        if self._cache_path is None:
            return
        tmp = self._cache_path.with_suffix(f".{os.getpid()}.tmp")
        try:
            with tmp.open("w") as fh:
                yaml.dump(env_dict, fh)
            os.replace(tmp, self._cache_path)
        except Exception:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass

    def get(self) -> dict:
        """Return the fully-parsed conda environment dict.

        Order of precedence:
        1. In-process cache (``_cached_result``) — fastest, no I/O.
        2. On-disk YAML cache when not stale — one file read.
        3. Fresh subprocess call — slowest; result saved to disk.
        """
        if self._cached_result is not _NOT_COMPUTED:
            return self._cached_result  # type: ignore[return-value]

        if not self.is_stale():
            loaded = self.load()
            if loaded is not None:
                self._cached_result = loaded
                return loaded

        result = self._fetch_from_subprocess()
        if result:
            self.save(result)
        self._cached_result = result
        return result

    # ── Subprocess fetch + parsing ────────────────────────────────────────────

    def _fetch_from_subprocess(self) -> dict:
        """Run ``conda env export`` + ``pip freeze`` and return the parsed dict.

        The returned structure matches ``UserEnvironment.env``:
        - ``channels``, ``dependencies`` (conda + pip merged), ``custom``
          (editable / non-standard pip installs), ``raw pip freeze``.

        Resets all parsing-state attributes before running so the method is
        safe to call more than once on the same instance (e.g. in tests).
        """
        self._reset_state()

        # ── conda env export ─────────────────────────────────────────────────
        conda_result = sub_run(["conda", "env", "export"], **SUBPROCESS_KWARGS)
        if conda_result.returncode != 0:
            logger.error(  # pragma: no cover
                "Failed to retrieve the Conda environment. "
                "Recompute feature disabled."
            )
            return {}  # pragma: no cover

        self._conda_export = {
            k: v
            for k, v in yaml.safe_load(conda_result.stdout).items()
            if k not in ("name", "prefix")
        }

        # Separate the pip sub-list that conda embed inside dependencies
        conda_pip_list: list = []
        conda_deps: list = self._conda_export.get("dependencies", [])
        for i, dep in enumerate(conda_deps):
            if isinstance(dep, dict):
                pip_obj = conda_deps.pop(i)
                conda_pip_list = pip_obj.pop("pip", [])
                break
        self._conda_export["dependencies"] = conda_deps

        if conda_pip_list:
            self._conda_pip_dict = {
                entry.split("==", maxsplit=1)[0]: entry
                for entry in conda_pip_list
            }

        # ── pip freeze ───────────────────────────────────────────────────────
        pip_result = sub_run(["pip", "freeze"], **SUBPROCESS_KWARGS)
        if pip_result.returncode != 0:
            logger.error(  # pragma: no cover
                "Failed to retrieve the pip environment. "
                "Recompute feature disabled."
            )
            return {}  # pragma: no cover

        self._pip_freeze = [
            line.strip() for line in pip_result.stdout.splitlines()
        ]

        for line in self._pip_freeze:
            if not self._parse_pip_line(line):
                return {}  # parse error already logged inside

        self._warn_if_custom_or_conflict()

        # ── Assemble final dict ───────────────────────────────────────────────
        conda_deps = self._conda_export.pop("dependencies", [])
        conda_deps.append({"pip": list(self._conda_pip_dict.values())})
        self._conda_export["dependencies"] = conda_deps

        if self._pip_custom:
            self._conda_export["custom"] = self._pip_custom
        self._conda_export["raw pip freeze"] = self._pip_freeze

        return self._conda_export

    def _reset_state(self) -> None:
        self.has_editable = False
        self._conda_export = {}
        self._conda_pip_dict = {}
        self._pip_freeze = []
        self._pip_custom = {}
        self._freeze_comments = {}
        self._conda_conflicts = {}

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

            # Conflicting versions — record and let pip win
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
        """Log warnings for non-standard pip installs and conda/pip conflicts."""
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

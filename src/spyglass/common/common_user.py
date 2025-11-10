import re
from functools import cached_property
from hashlib import md5
from json import dumps as json_dumps
from os import environ as os_environ
from pathlib import Path
from pprint import pprint
from subprocess import run as sub_run
from typing import Dict, List, Optional, Tuple, Union

import datajoint as dj
import yaml

from spyglass.utils import SpyglassMixin, logger

schema = dj.schema("common_user")

DEFAULT_ENV_ID = (
    dj.config["database.user"]
    + "_"
    + os_environ.get("CONDA_DEFAULT_ENV", "base")
    + "_00"
)
SUBPROCESS_KWARGS = dict(capture_output=True, text=True, timeout=60)


@schema
class UserEnvironment(SpyglassMixin, dj.Manual):
    definition = """ # User conda env. Default ID is User_CondaEnv_00
    env_id: varchar(127)  # Unique environment identifier
    ---
    env_hash: char(32)  # MD5 hash of the environment
    env: blob  # Conda environment
    timestamp=CURRENT_TIMESTAMP: timestamp  # Automatic timestamp
    UNIQUE INDEX (env_hash)
    has_editable=0: bool  # Whether the environment has editable installs
    """

    # Note: this tables establishes the convention of an environment ID that
    # substrings {user}_{env_name}_{num} where num is a two-digit number.
    # Substringing isn't ideal, but it simplifies downstream inherited keys.

    _has_editable = False  # Flag to track if the env has editable installs
    _conda_export = dict()  # Cached conda export
    _conda_pip_dict = dict()  # Pip dependencies from conda export
    _pip_freeze = list()  # Cached pip freeze output
    _pip_custom = dict()  # Custom pip installs from the environment
    _freeze_comments = dict()  # Comments from pip freeze
    _conda_conflicts = dict()  # Conda and pip conflicts in the environment

    def _get_conda_export(self) -> Tuple[dict, dict]:
        """Fetch the current Conda environment export.

        - Runs `conda env export` to get the current environment.
        - Parses the output to extract dependencies, drop name and prefix.
        - Extracts pip dependencies if present, converts them to a dict.

        Returns
        -------
        conda_export : dict
            The exported Conda environment as a dictionary.
        conda_pip_dict : dict
            A dictionary of pip dependencies: Dict[name, package==version]
        """
        conda_export = sub_run(["conda", "env", "export"], **SUBPROCESS_KWARGS)
        if conda_export.returncode != 0:
            logger.error(  # pragma: no cover
                "Failed to retrieve the Conda environment. "
                + "Recompute feature disabled."
            )
            return None  # pragma: no cover

        self._conda_export = {
            k: v
            for k, v in yaml.safe_load(conda_export.stdout).items()
            if k not in ["name", "prefix"]  # Exclude name and prefix
        }

        conda_pip_list = []
        conda_deps = self._conda_export.get("dependencies", [])
        for i, conda_dep in enumerate(conda_deps):
            if isinstance(conda_dep, dict):  # extract pip dependencies
                pip_obj = conda_deps.pop(i)  # remove from list
                conda_pip_list = pip_obj.pop("pip", None)
                break

        self._conda_export["dependencies"] = conda_deps

        if conda_pip_list:  # convert to dict
            self._conda_pip_dict = {
                dep.split("==", maxsplit=1)[0]: dep for dep in conda_pip_list
            }

    def _get_pip_freeze(self) -> List[str]:
        """Fetch the pip freeze output."""
        ret = sub_run(["pip", "freeze"], **SUBPROCESS_KWARGS)
        if ret.returncode != 0:
            logger.error(  # pragma: no cover
                "Failed to retrieve the pip environment. "
                + "Recompute feature disabled."
            )
            return None  # pragma: no cover

        self._pip_freeze = [line.strip() for line in ret.stdout.splitlines()]

    @property
    def _ignored_conda_source(self):
        """Regex pattern for conda-managed packages from accepted sources.

        Expects each of these sources to be a directory in the path.
        """
        sources = "feedstock_root|croot|conda-bld|bld|builder"
        return re.compile(rf"[/\\](?:{sources})[/\\]")

    @property
    def _basic_dep_format(self):
        """Regex patterns for dependency formats.

        Allowable formats with suffixes: a1, b2, rc3, .post4, dev5, etc.
        Examples:
        - package==1.2.3
        - package==1.2.3a1 (alpha)
        - package==1.2.3b2 (beta)
        - package==1.2.3rc3 (release candidate)
        - package==1.2.3.post4 (post-release)
        - package==1.2.3.dev5 (developmental release)
        - package==1.2.3+dfsg (Debian Free Software Guidelines)
        - package==1.2.3+ds (Debian Source)
        - package==1.2.3+ubuntu1.2 (Ubuntu-specific)
        """
        return re.compile(  # allows suffixes like a1, dev, post1, etc.
            r"""^
                (?P<pkg>[\w\d.-]+)                  # Package
                ==                                  # `==` separator
                (?P<ver>                            # Version
                    \d+(?:\.\d+)*                   # Major.Minor.Patch
                    (?:                             # Suffixes
                        (?P<alpha>a\d+)|            # Alpha release (a1, a2, ...)
                        (?P<beta>b\d+)|             # Beta release (b1, b2, ...)
                        (?P<rc>rc\d+)|              # Release candidate (rc1, rc2, ...)
                        (?P<post>\.post\d+)|        # Post-release (.post1, .post2, ...)
                        (?P<dev>dev\d*)|            # Developmental release (dev, dev1, ...)
                        (?P<dfsg>\+dfsg)|           # Debian Free Software Guidelines (+dfsg)
                        (?P<ds>\+ds)|               # Debian Source (+ds)
                        (?P<ubuntu>\+ubuntu\d+(?:\.\d+)*) # Ubuntu-specific (+ubuntu1.2)
                    )
                ?)
            $""",
            re.VERBOSE,
        )

    @property
    def _comment_install(self):
        """Regex pattern for custom pip installs in comments."""
        return re.compile(
            r"^\s*#.*?\((?P<pkg>[A-Za-z0-9_.-]+)==(?P<ver>[^)]+)\)$"
        )

    def _warn_if_custom_or_conflict(self):
        pip_custom_no_spy = {
            k: "".join(v)
            for k, v in self._pip_custom.items()
            if "spyglass" not in k
        }
        if pip_custom_no_spy:
            logger.warning(
                "Custom pip installs found. "
                + "Recompute feature may not work as expected."
            )
            pprint(pip_custom_no_spy, indent=4)

        if self._conda_conflicts:
            logger.warning(  # pragma: no cover
                "Conda/pip conflicts in the environment.\n"
                + "\tRecompute feature may not work as expected.\n"
                + "\tUse `pip uninstall pkg` to defer to conda.\n"
                + "\tConflicts as Package : [pip_version, conda_version]"
            )
            pprint(self._conda_conflicts, indent=4)

    def _delim_split(
        self, line: str = None, as_dict=False
    ) -> Union[Tuple[str, str], dict]:

        if isinstance(line, list):
            ret = [self._delim_split(row) for row in line]
            return {i[0]: i[1] for i in ret} if as_dict else ret

        if line is None or not isinstance(line, str):
            ret = "", None
        else:
            for delim in ["==", " @ ", "="]:
                if delim in line:
                    ret = line.split(delim, maxsplit=1)
                    break
            else:
                ret = line, ""

        return {ret[0]: ret[1]} if as_dict else tuple(ret)

    def _parse_pip_line(self, line: str) -> bool:
        """Parse a line from pip freeze output.

        This method handles various formats of pip freeze output, including:
        - Prebuilt conda packages in paths like `feedstock_root/`
        - Basic dependency formats like `package==version`, checking conflicts
        - Custom pip installs with `@` syntax, capturing local file paths
        - Editable installs with `-e git+` syntax, capturing package names

        Parameters
        ----------
        line : str
            A line from the pip freeze output.

        Returns
        -------
        bool
            True if the line was successfully parsed and handled.

        Raises
        ------
        ValueError
            If the line is an editable path is found without preceding comment.
        """
        # ------------------- Ignore conda-managed packages -------------------
        if self._ignored_conda_source.search(line):
            return True  # ignore file-based conda-managed packages

        if match := self._basic_dep_format.match(line):
            package, pip_version = match.group("pkg", "ver")
            conda_line = self._conda_pip_dict.get(package, None)
            _, conda_version = self._delim_split(conda_line)

            # --------------- if pip and conda agree, do nothing --------------
            if line == conda_line:  # same package, same version
                return True  # no conflict, same version

            # ------- if conda version is none, set pip version in conda ------
            if conda_version is None:  # append pip-only package
                logger.debug(f"Pip-only package  : {line}")
                self._conda_pip_dict[package.lower()] = line.lower()
                return True  # successfully parsed basic dependency

            # --- if conflicting versions, log conflict, overwrite with pip ---
            logger.info(f"Conda/pip conflict: {line}")
            self._conda_conflicts[package] = [pip_version, conda_version]
            self._conda_pip_dict[package] = line
            return True  # successfully parsed basic dependency conflict

        # ------- if filepath or git path, capture as custom pip install -------
        if " @ " in line and ("file://" in line or "git+" in line):
            # capture local file paths
            dep_name, path = self._delim_split(line)
            self._pip_custom[dep_name] = line
            logger.debug(f"Custom pip install: {dep_name}")
            return True  # likely versioned custom install, not 'editable'

        # ------ if editable install from git, track and flag as editable ------
        if line.startswith("-e git+") and "#egg=" in line:
            url, package = line.split("#egg=", maxsplit=1)  # pragma: no cover
            # conda convention uses dashes
            logger.debug(f"Editable from git : {package}")
            self._pip_custom[package.replace("_", "-")] = line
            if "spyglass" not in package:  # allow spyglass editable from git
                self._has_editable = True
            return True  # editable install, no version info

        # ------- if comment, then editable, track and flag as editable -------
        if match := self._comment_install.match(line):
            pkg = match.group("pkg").replace("_", "-")
            self._freeze_comments[pkg] = match.group("ver")
            return True  # comment install, successfully parsed

        if match := re.compile(r"^-e[ ]+(?P<path>\S+)$").match(line):
            path = match.group("path")  # editable path
            path_name = Path(path).name.replace("_", "-")
            comment = self._freeze_comments.get(path_name, None)
            if comment is None:
                raise ValueError(
                    f"Editable path found w/o a preceding comment: {line}"
                )
            logger.info(f"Editable from path: {line}")
            self._pip_custom[path_name] = f"{comment} @ {path}"
            self._has_editable = True
            return True  # editable install, from path with comment

        logger.error(f"Failed to parse: {line}\nRecompute feature disabled.")
        return False  # unrecognized line format

    @cached_property
    def env(self) -> dict:
        """Fetch the current Conda environment as a string.

        This method retrieves the current Conda environment, parses it, and
        augments with conflicts and editable installs from pip freeze.

        It handles the following:
        - `a==1.2.3` basic format, checking for a different pip version
        - `a @ file://path/to/package` if trusted conda source, ignore. If
           user path, add considered editablens install.
        - `a @ git+https://` assumed to be a git hash, permitted
        - `-e git+https://...#egg=package` treated as editable install

        Returns
        -------
        dict
            The Conda environment as a dictionary, including pip dependencies.
        """

        # ---------------- Start with Conda environment export ----------------
        _ = self._get_conda_export()
        if not self._conda_export:
            return ""  # pragma: no cover

        # ------------------------ Fetch pip editables ------------------------
        _ = self._get_pip_freeze()
        if not self._pip_freeze:
            return ""  # pragma: no cover

        for line in self._pip_freeze:
            parsed = self._parse_pip_line(line)
            if not parsed:
                return ""  # pragma: no cover

        _ = self._warn_if_custom_or_conflict()

        # ------- Augment 'pip' section, add custom and freeze section -------
        conda_deps = self._conda_export.pop("dependencies", [])
        conda_deps.append({"pip": list(self._conda_pip_dict.values())})
        self._conda_export["dependencies"] = conda_deps

        if self._pip_custom:  # additional sections ignored on `env create -f`
            self._conda_export["custom"] = self._pip_custom
        self._conda_export["raw pip freeze"] = self._pip_freeze

        return self._conda_export

    def parse_env_dict(self, env: Optional[dict] = None) -> dict:
        """Convert the environment string to a dictionary."""
        env = env or self.env

        deps = dict()
        if not env or not isinstance(env, dict):
            logger.error(f"Invalid env {type(env)}: {env}")
            return deps

        for dep in env.get("dependencies", []):
            if isinstance(dep, str):  # split conda by '='
                deps.update(self._delim_split(dep, as_dict=True))
            elif isinstance(dep, dict):
                for pip_dep in dep.values():
                    deps.update(self._delim_split(pip_dep, as_dict=True))

        return deps

    @cached_property
    def env_hash(self) -> str:
        """Compute an MD5 hash of the environment dictionary."""
        env_json = json_dumps(self.parse_env_dict(self.env), sort_keys=True)
        return md5(env_json.encode()).hexdigest()

    @cached_property
    def matching_env_id(self) -> Optional[str]:
        """Return the env_id that matches the current environment's hash."""
        matches = self & f'env_hash="{self.env_hash}"'
        return matches.fetch1("env_id") if matches else None

    @cached_property
    def this_env(self) -> Optional[dict]:
        """Return the environment key. Cached to avoid rerunning insert."""
        if self.matching_env_id:
            return {"env_id": self.matching_env_id}
        del self.matching_env_id  # clear the cached property
        return self.insert_current_env()

    def _increment_id(self, env_id: str) -> str:
        """Increment the environment ID.

        Users are likely to update their environments without careful
        record-keeping. This method increments the numeric suffix of the
        environment ID, if present, to avoid overwriting existing records.

        Substringing the environment ID is not ideal for precise record-keeping,
        but it simplifies the downstream inherited keys.
        """
        if not self & f'env_id="{env_id}"':
            return env_id

        # Extract the base ID and any existing numeric suffix
        base_match = re.match(r"^(.*?)(?:_(\d{2}))?$", env_id)
        base_id = base_match.group(1) if base_match else env_id
        # Check for existing IDs with the same base
        used_ids = (self & f"env_id LIKE '{base_id}%'").fetch("env_id")
        suffixes = [
            int(match.group(1))  # extract the numeric suffix
            for match in (re.search(r"_(\d{2})$", eid) for eid in used_ids)
            if match
        ]  # take the max of the suffixes and increment, or start at 1
        next_int = (max(suffixes) + 1) if suffixes else 1

        return f"{base_id}_{next_int:02d}"

    def insert_current_env(self, env_id=DEFAULT_ENV_ID) -> dict:
        """Insert the current environment into the table."""

        if not self.env:  # if conda dump fails, warned above
            return {"env_id": None}

        if self.matching_env_id:  # if env is already stored
            return {"env_id": self.matching_env_id}

        self.insert1(
            {  # if current env not stored, but name taken, increment
                "env_id": self._increment_id(env_id),
                "env_hash": self.env_hash,
                "env": self.env,
                "has_editable": self._has_editable,
            },
            skip_duplicates=False,
        )
        del self.matching_env_id  # clear the cached property
        return {"env_id": env_id}

    def write_env_yaml(
        self,
        env_id: Optional[str] = DEFAULT_ENV_ID,
        dest_path: Optional[str] = None,
    ) -> None:
        """Write the environment to a YAML file."""
        query = self & f'env_id="{env_id}"'
        if not query:
            logger.error(f"No environment found with env_id '{env_id}'.")
            return

        fetched_env, has_editable = query.fetch1("env", "has_editable")
        if has_editable:
            logger.warning(
                "Environment has editable installs. "
                + "Recompute feature may not work as expected.\n\t"
                + "Review 'custom' section in the YAML file."
            )

        env_dict = dict(name=env_id, **fetched_env)
        dest_dir = Path(dest_path) if dest_path else Path.cwd()
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / f"{env_id}.yaml"

        with open(dest_file, "w") as yaml_file:
            yaml_file.write(yaml.dump(env_dict))

        logger.info(f"Environment written to {dest_file.resolve()}")

    def has_matching_env(
        self,
        env_id: Optional[str] = DEFAULT_ENV_ID,
        relevant_deps: Optional[List[str]] = None,
        show_diffs: Optional[bool] = True,
    ) -> bool:
        """Check if env_id matches the current env, list discrepancies.

        Note: Developers working on recompute pipelines may find that they only
        want to gatekeep recompute attempts based on a subset of dependencies.
        This method allows for that by specifying a list of relevant_deps.

        Parameters
        ----------
        env_id : str, optional
            The environment ID to compare against. Default to DEFAULT_ENV_ID.
        relevant_deps : List[str], optional
            A list of dependencies to compare. Default to None, meaning compare
            all dependencies.
        show_diffs : bool, optional
            Show discrepancies between the current and stored environments.
            Default to True.
        """
        query = self & f'env_id="{env_id}"'
        if not query:
            logger.error(f"No environment found with env_id '{env_id}'.")
            return False

        stored_hash, stored_env = query.fetch1("env_hash", "env")

        if stored_hash == self.env_hash:
            return True

        mismatches = []
        this_env = self.parse_env_dict(self.env)
        prev_env = self.parse_env_dict(stored_env)
        for key in relevant_deps or prev_env:
            if this_env.get(key) != prev_env.get(key):
                mismatches.append(
                    f"  - {key}: {prev_env[key]} -> {this_env.get(key)}"
                )

        this_set = set(this_env.keys())
        prev_set = set(prev_env.keys())
        relevant_set = set(relevant_deps) if relevant_deps else prev_set

        if not relevant_set.issubset(this_set):
            diff = relevant_set - this_set
            mismatches.append(f"  - Missing deps: {diff}")

        if mismatches and show_diffs:
            logger.error(
                f"Current env does not match {env_id}: (stored vs current)"
                + "\n".join(mismatches)
            )

        return not bool(mismatches)  # True if no mismatches

    def get_dep_version(
        self, dep_name: str, env_id: Optional[str] = None
    ) -> Union[List[Dict[str, str]], Dict[str, str]]:
        """Get the version of a specific dependency in the environment."""
        if not env_id and len(self) > 1:
            ret = dict()
            for env_id in self.fetch("env_id"):
                ret.update(self.get_dep_version(dep_name, env_id))
            return ret

        restr = f"env_id='{env_id}'" if env_id else True
        query = self & restr
        if not query:
            logger.error(f"No environment found with env_id '{env_id}'.")
            return ""

        env_id, env = query.fetch1("env_id", "env")
        dependencies = self.parse_env_dict(env)

        return {env_id: dependencies.get(dep_name, "")}

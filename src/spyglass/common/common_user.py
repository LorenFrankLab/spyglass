import re
from pprint import pprint
from functools import cached_property
from hashlib import md5
from json import dumps as json_dumps
from os import environ as os_environ
from pathlib import Path
from subprocess import run as sub_run
from typing import Dict, List, Optional, Union


import datajoint as dj
import yaml

from spyglass.utils import logger

schema = dj.schema("common_user")

DEFAULT_ENV_ID = (
    dj.config["database.user"]
    + "_"
    + os_environ.get("CONDA_DEFAULT_ENV", "base")
    + "_00"
)
SUBPROCESS_KWARGS = dict(capture_output=True, text=True, timeout=60)


@schema
class UserEnvironment(dj.Manual):
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

    @cached_property
    def env(self) -> Union[dict, str]:
        """Fetch the current Conda environment as a string."""

        # ---------------- Start with Conda environment export ----------------
        conda_export = sub_run(["conda", "env", "export"], **SUBPROCESS_KWARGS)
        if conda_export.returncode != 0:
            logger.error(  # pragma: no cover
                "Failed to retrieve the Conda environment. "
                + "Recompute feature disabled."
            )
            return ""  # pragma: no cover

        ret = {
            k: v
            for k, v in yaml.safe_load(conda_export.stdout).items()
            if k not in ["name", "prefix"]
        }

        # ------------------------ Fetch pip editables ------------------------
        pip_freeze = sub_run(["pip", "freeze"], **SUBPROCESS_KWARGS)
        if pip_freeze.returncode != 0:
            logger.error(
                "Failed to retrieve the pip environment. "
                + "Recompute feature disabled."
            )
            return ""

        # Ignore basic a==0.0.0 pip installs
        basic_format = re.compile(  # allows suffixes like a1, dev, post1, etc.
            r"[\w\d_.-]+==\d+(?:\.\d+)*(?:(?:a|b|rc|\.post)[\d]+|dev[\d]*)?$"
        )
        # Ignore conda-managed packages from accepted sources
        sources = "feedstock_root|croot|conda-bld"  # accepts each in path
        ignored_conda_source = re.compile(rf"[/\\](?:{sources})[/\\]")

        # Capture custom pip installs: 1 `# package==version`, 2 `-e path`
        comment_install = re.compile(
            r"^\s*#.*?\((?P<package>[A-Za-z0-9_.-]+)==(?P<version>[^)]+)\)$"
        )

        pip_custom = dict()
        freeze_comments = dict()
        for line in pip_freeze.stdout.splitlines():
            line = line.strip()
            if basic_format.match(line) or ignored_conda_source.search(line):
                continue  # ignore file-based conda-managed packages
            if " @ " in line and ("file://" in line or "git+" in line):
                # capture local file paths
                dep_name, path = line.split(" @ ", maxsplit=1)
                pip_custom[dep_name] = path
                continue
            if line.startswith("-e git+") and "#egg=" in line:
                url, package = line.split(
                    "#egg=", maxsplit=1
                )  # pragma: no cover
                # conda convention uses dashes
                pip_custom[package.replace("_", "-")] = url.split("git+")[-1]
                self._has_editable = True
                continue
            if match := comment_install.match(line):
                freeze_comments[match.group("package").replace("_", "-")] = (
                    match.group("version")
                )
                continue
            if match := re.compile(r"^-e[ ]+(?P<path>\S+)$").match(line):
                path = match.group("path")  # editable path
                path_name = Path(path).name.replace("_", "-")
                if path_name not in freeze_comments:
                    raise ValueError(
                        f"Editable path found w/o a preceding comment: {line}"
                    )
                pip_custom[path_name] = f"{freeze_comments[path_name]} @ {path}"
                self._has_editable = True
                continue
            raise ValueError
            logger.error(
                f"Failed to parse: {line}\nRecompute feature disabled."
            )
            return

        if pip_custom:
            logger.warning(
                "Custom pip installs found in the environment.\n"
                + "\tRecompute feature may not work as expected.\n"
            )
            pprint(pip_custom)

        # ------- Augment the conda environment with custom pip installs -------
        conda_pips = []  # Find the element in the dependency list that is pip
        for conda_dep in ret.get("dependencies", []):
            if isinstance(conda_dep, dict):
                conda_pips = conda_dep.get("pip", None)
                break

        for i, pip_dep in enumerate(conda_pips):  # append custom pip installs
            dep_name, version = pip_dep.split("==", maxsplit=1)
            if dep_name in pip_custom:
                pip_path = pip_custom.pop(dep_name)  # remove from list
                conda_pips[i] = f"{dep_name}=={version} @ {pip_path}"

        if pip_custom:  # add any remaining custom pip installs
            for dep_name, pip_path in pip_custom.items():
                # needs `==` for parsing later
                conda_pips.append(f"{dep_name}==None @ {pip_path}")

        return ret

    def parse_env_dict(self, env: Optional[dict] = None) -> dict:
        """Convert the environment string to a dictionary."""
        dependencies = dict()
        if not env or not isinstance(env, dict):
            if env != "":  # only warn if it's not the no-conda empty string
                logger.error(f"Invalid env {type(env)}: {env}")
            return dependencies
        for dep in env.get("dependencies", []):
            if isinstance(dep, str):  # split conda by '='
                pip_dep, val = dep.split("=", maxsplit=1)
                dependencies[pip_dep] = val
            elif isinstance(dep, dict) and "pip" in dep:
                for pip_dep in dep["pip"]:  # split pip by '=='
                    pip_key, pip_val = pip_dep.split("==", maxsplit=1)
                    dependencies.setdefault(pip_key, pip_val)  # no overwrite
        return dependencies

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

        env_dict = dict(query.fetch1("env"), name=env_id)

        dest_dir = Path(dest_path) if dest_path else Path.cwd()

        with open(dest_dir / env_id, "w") as yaml_file:
            yaml_file.write(yaml.dump(env_dict))

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

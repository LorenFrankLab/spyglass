import datajoint as dj
import re
from os import environ as os_environ
from typing import List
from functools import cached_property
from hashlib import md5
from json import dumps as json_dumps
from subprocess import run as sub_run
from spyglass.utils import logger
from yaml import safe_load as yaml_load

schema = dj.schema("cbroz_user")  # TODO: common_user

DEFAULT_ENV_ID = (
    dj.config["database.user"]
    + "_"
    + os_environ.get("CONDA_DEFAULT_ENV", "base")
    + "_00"
)


@schema
class UserEnvironment(dj.Manual):
    definition = """ # User conda env. Default ID is User_CondaEnv_00
    env_id: varchar(32)  # Unique environment identifier
    ---
    env_hash: char(32)  # MD5 hash of the environment
    env: blob  # Full Conda environment stored as a dictionary
    timestamp=CURRENT_TIMESTAMP: timestamp  # Automatic timestamp
    UNIQUE INDEX (env_hash)
    """

    # Note: this tables establishes the convention of an environment ID that
    # substrings {user}_{env_name}_{num} where num is a two-digit number.
    # Substringing isn't ideal, but it simplifies downstream inherited keys.

    @cached_property
    def current_env(self) -> dict:
        """Fetch the current Conda environment as a dictionary."""
        result = sub_run(
            ["conda", "env", "export"], capture_output=True, text=True
        )
        if result.returncode != 0:
            logger.error("Failed to retrieve the Conda environment.")
            return {}

        parsed = yaml_load(result.stdout)

        dependencies = dict()
        for dep in parsed.get("dependencies", []):
            if isinstance(dep, str):
                pip_dep, val = dep.split("=", maxsplit=1)
                dependencies[pip_dep] = val
            elif isinstance(dep, dict) and "pip" in dep:
                for pip_dep in dep["pip"]:
                    pip_key, pip_val = pip_dep.split("==", maxsplit=1)
                    dependencies.setdefault(pip_key, pip_val)  # no overide

        return dependencies

    @cached_property
    def env_hash(self) -> str:
        """Compute an MD5 hash of the environment dictionary."""
        env_json = json_dumps(self.current_env, sort_keys=True)
        return md5(env_json.encode()).hexdigest()

    def _increment_id(self, env_id: str) -> str:
        """Increment the environment ID."""
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

        if not self.current_env:  # if conda dump fails
            logger.error("Failed to retrieve the current environment.")
            return

        if self.matching_env_id:  # if env is already stored
            logger.info(f"Env stored as '{self.matching_env_id}'")
            return {"env_id": self.matching_env_id}

        self.insert1(
            {  # if current env not stored, but name taken, increment
                "env_id": self._increment_id(env_id),
                "env_hash": self.env_hash,
                "env": self.current_env,
            }
        )
        del self.matching_env_id  # clear the cached property
        return {"env_id": env_id}

    def compare_env(
        self,
        env_id: str = DEFAULT_ENV_ID,
        relevant_deps: List[str] = None,
        show_diffs=True,
    ):
        """Check if env_id matches the current env, list discrepancies."""
        query = self & f'env_id="{env_id}"'
        if not query:
            logger.error(f"No environment found with env_id '{env_id}'.")
            return False

        stored_hash, stored_env = query.fetch1("env_hash", "env")

        if stored_hash == self.env_hash:
            return True

        mismatches = []
        for key in relevant_deps or stored_env:
            if stored_env.get(key) != self.current_env.get(key):
                mismatches.append(
                    f"  - {key}: {stored_env[key]} != {env.get(key)}"
                )

        relevant_set = set(relevant_deps) if relevant_deps else set()
        current_set = set(self.current_env.keys())
        stored_set = set(stored_env.keys())

        if not relevant_set.issubset(current_set):
            diff = relevant_set - current_set
            mismatches.append(f"  - Missing current deps: {diff}")
        if not relevant_set.issubset(stored_set):
            diff = relevant_set - stored_set
            mismatches.append(f"  - Missing stored: {diff}")

        if mismatches and show_diffs:
            logger.error(
                f"Current env does not match {env_id}: (stored vs current)"
                + "\n".join(mismatches)
            )

        return not bool(mismatches)  # True if no mismatches

    @cached_property
    def matching_env_id(self):
        """Return the env_id that matches the current environment's hash."""
        matches = self & f'env_hash="{self.env_hash}"'
        return matches.fetch1("env_id") if matches else None

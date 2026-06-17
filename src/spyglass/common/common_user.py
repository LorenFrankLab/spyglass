import re
from functools import cached_property
from hashlib import md5
from json import dumps as json_dumps
from os import environ as os_environ
from pathlib import Path
from typing import Dict, List, Optional, Union

import datajoint as dj
import yaml

from spyglass.utils import SpyglassMixin, logger
from spyglass.utils.env_cache import _delim_split, _env_cache
from spyglass.utils.git_utils import (
    WARN_DIRTY_ENV_DAYS,
    _warn_once,
    get_install_info,
)

schema = dj.schema("common_user")

DEFAULT_ENV_ID = (
    dj.config["database.user"]
    + "_"
    + os_environ.get("CONDA_DEFAULT_ENV", "base")
    + "_00"
)

# Single source of truth for "days since a dirty row was first logged",
# computed DB-side (server NOW()) so the warning countdown and the admin
# notification agree and neither is skewed by client-vs-server timezone.
DIRTY_DAYS_SQL = "DATEDIFF(NOW(), `timestamp`)"


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
    dirty_path=null: varchar(255)  # Path of local Spyglass clone, if any
    spyglass_commit=null: varchar(40)  # Short git hash of the Spyglass install
    """

    # Note: this tables establishes the convention of an environment ID that
    # substrings {user}_{env_name}_{num} where num is a two-digit number.
    # Substringing isn't ideal, but it simplifies downstream inherited keys.

    @cached_property
    def env(self) -> dict:
        """Return the current conda environment as a merged dict.

        Delegates to the shared :data:`_env_cache` singleton and requests
        the full pip-merged view (``with_pip=True``) so that pip-only
        packages, conda/pip version conflicts, and editable installs are all
        reflected.  The subprocess runs at most once per process.

        Returns
        -------
        dict
            Keys: ``channels``, ``dependencies`` (conda + merged pip
            sub-list), and optionally ``custom`` (editable / non-standard
            installs).  Empty dict on subprocess failure.
        """
        return _env_cache.get(with_pip=True)

    def parse_env_dict(self, env: Optional[dict] = None) -> dict:
        """Convert the environment string to a dictionary."""
        env = env or self.env

        deps = dict()
        if not env or not isinstance(env, dict):
            logger.error(f"Invalid env {type(env)}: {env}")
            return deps

        for dep in env.get("dependencies", []):
            if isinstance(dep, str):  # split conda by '='
                deps.update(_delim_split(dep, as_dict=True))
            elif isinstance(dep, dict):
                for pip_dep in dep.values():
                    deps.update(_delim_split(pip_dep, as_dict=True))

        return deps

    @cached_property
    def env_hash(self) -> str:
        """Compute an MD5 hash of the environment dictionary."""
        env_dict = self.parse_env_dict(self.env)
        commit = get_install_info().get("commit_hash")
        if commit:
            env_dict["spyglass_commit"] = commit
        env_json = json_dumps(env_dict, sort_keys=True)
        return md5(env_json.encode()).hexdigest()

    @cached_property
    def matching_env_id(self) -> Optional[str]:
        """Return the env_id that matches the current environment's hash."""
        matches = self & f'env_hash="{self.env_hash}"'
        return matches.fetch1("env_id") if matches else None

    @cached_property
    def this_env(self) -> Optional[dict]:
        """Return the environment key. Cached to avoid rerunning insert."""
        _warn_once()
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

        info = get_install_info()
        # Only flag the install path as "dirty" when the clone is actually
        # modified or months behind upstream — a clean, current local clone
        # (even an editable dev build) must not trigger escalation emails.
        flagged = info.get("has_local_changes") or info.get("is_stale")
        self.insert1(
            {  # if current env not stored, but name taken, increment
                "env_id": self._increment_id(env_id),
                "env_hash": self.env_hash,
                "env": self.env,
                "has_editable": _env_cache.has_editable,
                "dirty_path": info.get("install_path") if flagged else None,
                "spyglass_commit": info.get("commit_hash"),
            },
            skip_duplicates=False,
        )
        del self.matching_env_id  # clear the cached property
        return {"env_id": env_id}

    def dirty_installs(self) -> "dj.expression.QueryExpression":
        """Return dirty-install entries with days_since_warn.

        Excludes env_id bases where the user has a more recent clean install
        (``dirty_path IS NULL``), indicating the issue was self-resolved.

        Returns a DataJoint query expression.  Callers needing a DataFrame
        can call ``.fetch(format='frame')`` on the result.
        """
        dirty = self & "dirty_path IS NOT NULL"

        if dirty and (clean := self & "dirty_path IS NULL"):
            dirty_rows = dirty.fetch("env_id", "timestamp", as_dict=True)
            clean_rows = clean.fetch("env_id", "timestamp", as_dict=True)

            clean_by_base: dict = {}
            for row in clean_rows:
                m = re.match(r"^(.*?)(?:_\d{2})?$", row["env_id"])
                base = m.group(1) if m else row["env_id"]
                t = row["timestamp"]
                if base not in clean_by_base or t > clean_by_base[base]:
                    clean_by_base[base] = t

            excluded = []
            for row in dirty_rows:
                m = re.match(r"^(.*?)(?:_\d{2})?$", row["env_id"])
                base = m.group(1) if m else row["env_id"]
                clean_ts = clean_by_base.get(base)
                if clean_ts and clean_ts > row["timestamp"]:
                    excluded.append(row["env_id"])

            if excluded:
                ids = ", ".join(f'"{eid}"' for eid in excluded)
                dirty = dirty & f"env_id NOT IN ({ids})"

        return dirty.proj(..., days_since_warn=DIRTY_DAYS_SQL)

    def write_dirty_notifications(self, out_path: str = None) -> None:
        """Write overdue dirty-install rows to out_path (or stdout if None).

        Produces one tab-separated line per user where ``days_since_warn``
        exceeds ``WARN_DIRTY_ENV_DAYS``:

            user_email\\tdays_since_warn\\tspyglass_commit\\tdirty_path

        Called by ``maintenance_scripts/cleanup.py``; output is consumed by
        ``run_jobs.sh`` which sends emails via ``email_utils.sh``.  Users with
        no ``LabMember.LabMemberInfo`` email record are skipped with a message
        to stderr.

        Parameters
        ----------
        out_path : str, optional
            Destination file path.  Writes to stdout when ``None``.
        """
        import sys

        from spyglass.common.common_lab import LabMember

        overdue = [
            row
            for row in self.dirty_installs().fetch(as_dict=True)
            if row["days_since_warn"] >= WARN_DIRTY_ENV_DAYS
        ]
        if not overdue:
            return

        member_rows = LabMember.LabMemberInfo.fetch(
            "datajoint_user_name", "google_user_name", as_dict=True
        )
        email_by_dj_user = {
            m["datajoint_user_name"]: m["google_user_name"]
            for m in member_rows
            if m["datajoint_user_name"]
        }
        known_users = sorted(email_by_dj_user, key=len, reverse=True)

        out = open(out_path, "w") if out_path else sys.stdout
        try:
            for row in overdue:
                env_id = row["env_id"]
                dj_user = next(
                    (u for u in known_users if env_id.startswith(u + "_")),
                    None,
                )
                user_email = email_by_dj_user.get(dj_user) if dj_user else None
                if user_email is None:
                    print(
                        f"No LabMemberInfo email for '{env_id}' — skipping.",
                        file=sys.stderr,
                    )
                    continue
                print(
                    "\t".join(
                        [
                            user_email,
                            str(row["days_since_warn"]),
                            str(row["spyglass_commit"] or ""),
                            str(row["dirty_path"] or ""),
                        ]
                    ),
                    file=out,
                )
        finally:
            if out_path:
                out.close()

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

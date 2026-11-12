import re
from functools import cached_property
from hashlib import md5
from json import dumps as json_dumps
from os import environ as os_environ
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Union

import datajoint as dj
import yaml

from spyglass.utils import SpyglassMixin, logger
from spyglass.utils.env_cache import _delim_split, _env_cache
from spyglass.utils.git_utils import (
    WARN_DIRTY_ENV_DAYS,
    WARN_STALE_DAYS,
    spyglass_install,
)

schema = dj.schema("common_user")

DEFAULT_ENV_ID = (
    dj.config["database.user"]
    + "_"
    + os_environ.get("CONDA_DEFAULT_ENV", "base")
    + "_00"
)

# Server NOW(), so the warning countdown and the email cannot disagree
DIRTY_DAYS_SQL = "DATEDIFF(NOW(), `timestamp`)"

# The database counterpart of SpyglassInstall.is_stale, for any row
STALE_DAYS_SQL = "DATEDIFF(NOW(), `commit_date`)"


@schema
class UserEnvironment(SpyglassMixin, dj.Manual):
    definition = """ # User conda env. Default ID is User_CondaEnv_00
    env_id: varchar(127)  # Unique environment identifier
    ---
    env_hash=null: char(32)  # MD5 of env; null on dirty, and nulls never clash
    env=null: blob  # Conda environment; null on a dirty install
    timestamp=CURRENT_TIMESTAMP: timestamp  # When this row first appeared
    UNIQUE INDEX (env_hash)
    has_editable=null: bool  # 1 yes, 0 tested and no, null not tested
    dirty_path=null: varchar(255)  # Path of local Spyglass clone, if any
    spyglass_commit=null: varchar(40)  # Short git hash of the Spyglass install
    commit_date=null: date  # Committer date of spyglass_commit; null off a clone
    last_seen=CURRENT_TIMESTAMP: timestamp  # Most recent session using this row
    resolved_at=null: timestamp  # When this dirty state was cleaned up
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
        """Compute an MD5 hash of the environment dictionary.

        The Spyglass commit hash is folded in when present. For an editable /
        local clone the conda+pip export does not capture the exact source
        revision (an editable install shows as ``-e ...`` with no version), so
        the commit is the only thing separating two environments whose exports
        match but whose code differs. Including it keeps recompute reproducible,
        at one row per commit on a dev install. A clean release has
        ``commit_hash=None``, leaving its hash and existing rows untouched.

        A new row means a new ``env_id``, and ``RecordingRecomputeSelection``
        keys on ``env_id``, so a dev clone gets a fresh set of recompute
        selections per commit. Recompute is an admin operation, so this is
        acceptable.
        """
        env_dict = self.parse_env_dict(self.env)
        commit = spyglass_install.commit_hash
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
        """Return the environment key. Cached to avoid rerunning insert.

        A dirty install never matches a stored row, however well its
        environment hashes: the hash covers the commit, not the edits on top
        of it, so a clean row logged at this commit describes different code.
        """
        spyglass_install.warn_once()
        if spyglass_install.has_local_changes:
            return self.insert_current_env()  # partial row, never a match
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
        escaped = base_id.replace("_", "\\_")
        used_ids = (self & f"env_id LIKE '{escaped}%'").fetch("env_id")
        suffixes = [
            int(match.group(1))  # extract the numeric suffix
            for match in (re.search(r"_(\d{2})$", eid) for eid in used_ids)
            if match
        ]  # take the max of the suffixes and increment, or start at 1
        next_int = (max(suffixes) + 1) if suffixes else 1

        return f"{base_id}_{next_int:02d}"

    def insert_current_env(self, env_id=DEFAULT_ENV_ID) -> dict:
        """Log the current environment, returning its key.

        A dirty install takes the partial path: the row records the commit, the
        clone path and the clock, and no ``conda env export``, ``pip freeze`` or
        hashing runs. A clean install takes the full path and is stored for
        recompute.

        The insert hook runs first, before an id is chosen, because
        ``insert1`` below re-enters it.

        Parameters
        ----------
        env_id : str, optional
            Preferred environment id. A numeric suffix is added when taken.
            A flagged install whose hook has just logged the environment
            returns that row's id instead, as it does for any environment
            already on record.

        Returns
        -------
        dict
            ``{"env_id": <id>}``, or ``{"env_id": None}`` when the environment
            could not be captured.
        """
        self._log_install_once()

        if spyglass_install.has_local_changes:
            return self._log_dirty_env(env_id)

        # Stop the clock on any open dirty episode before any return
        self._mark_dirty_resolved()

        if not self.env:  # if conda dump fails, warned above
            return {"env_id": None}

        if self.matching_env_id:  # if env is already stored
            # `stale_installs` ages a row out on `last_seen`, so refresh it
            self.connection.query(
                f"UPDATE {self.full_table_name} "
                + "SET `last_seen` = NOW() WHERE `env_id` = %s",
                args=(self.matching_env_id,),
            )
            return {"env_id": self.matching_env_id}

        # Capture the incremented id once: the row and the return must agree
        new_env_id = self._increment_id(env_id)
        self.insert1(
            {
                "env_id": new_env_id,
                "env_hash": self.env_hash,
                "env": self.env,
                "has_editable": _env_cache.has_editable,
                "dirty_path": None,
                "spyglass_commit": spyglass_install.commit_hash,
                "commit_date": spyglass_install.commit_date,
            },
            skip_duplicates=False,
        )
        del self.matching_env_id  # clear the cached property
        return {"env_id": new_env_id}

    def _log_dirty_env(self, env_id: str) -> dict:
        """Record a dirty install as a partial row, or refresh the existing one.

        Writes no environment and runs no subprocess: a dirty install is not
        worth storing, so only the identity and the clock are kept.

        One row per dirty *episode*, not per session. An episode already on
        record has its ``last_seen`` refreshed and keeps its original
        ``timestamp``, so the 30-day countdown runs from when the user first
        went dirty. An episode following a resolution finds no unresolved row
        and starts fresh, rather than inheriting a months-old date and emailing
        the user immediately.

        Parameters
        ----------
        env_id : str
            Preferred environment id. A numeric suffix is added when taken.

        Returns
        -------
        dict
            ``{"env_id": <id>}`` for the row that now represents this episode.
        """
        dj_user = dj.config.get("database.user", "")
        # `_` is a LIKE wildcard: unescaped, `ann_%` matches `anna_base_00`
        mine = f"env_id LIKE '{dj_user}\\_%'"
        episode = self.unresolved_dirty() & mine

        if episode:
            # Oldest first: `fetch` has no order; the oldest dates the episode
            existing = episode.fetch("env_id", order_by="timestamp ASC")[0]
            # `timestamp` is the clock and stays put
            self.connection.query(
                f"UPDATE {self.full_table_name} "
                + "SET `last_seen` = NOW(), "
                + "`spyglass_commit` = %s, `dirty_path` = %s, "
                + "`commit_date` = %s "
                + "WHERE `env_id` = %s",
                args=(
                    spyglass_install.commit_hash,
                    spyglass_install.install_path,
                    spyglass_install.commit_date,
                    existing,
                ),
            )
            return {"env_id": existing}

        new_env_id = self._increment_id(env_id)
        self.insert1(
            {
                "env_id": new_env_id,
                "env_hash": None,
                "env": None,
                "has_editable": None,  # untested: no pip freeze ran
                "dirty_path": spyglass_install.install_path,
                "spyglass_commit": spyglass_install.commit_hash,
                "commit_date": spyglass_install.commit_date,
            },
            skip_duplicates=False,
        )
        return {"env_id": new_env_id}

    def is_trusted(self, restriction=True) -> bool:
        """Whether a logged environment may be assumed to reproduce its files.

        Recompute uses this to decide whether an attempt can be logged as
        passing at creation. Three states disqualify an environment:

        * a dirty install (``env_hash`` null), which records no environment;
        * an untested environment (``has_editable`` null), where no
          ``pip freeze`` ran and editable installs are unknown;
        * an editable dependency (``has_editable`` true), whose version string
          says nothing about its contents.

        Parameters
        ----------
        restriction : dict or str, optional
            Restriction selecting one row. Default all, which requires the
            table to hold exactly one.

        Returns
        -------
        bool
            ``True`` only when the row records a fully examined, clean
            environment.
        """
        has_editable, env_hash = (self & restriction).fetch1(
            "has_editable", "env_hash"
        )
        if env_hash is None or has_editable is None:
            return False
        return not has_editable

    def log_install(self) -> None:
        """Record a dirty Spyglass install, once per session.

        Called from the mixin on the session's first insert or populate rather
        than at import: by then a connection exists, so this adds no
        import-time cost and nothing at all in a process that never writes.
        The once-per-session gate belongs to the caller,
        ``BaseMixin._logged_install_mode``, and must be set before this runs,
        because the insert below re-enters the mixin hook.

        Writes at most once a day per environment. A dirty state already on
        record needs no more than a ``last_seen`` refresh, which the activity
        window resolves in days, and the marker's ``last_logged`` date bounds
        it without a query.

        A clean, current install with a marker is the resolution case: the
        full row is logged so the escalation clock stops, and the marker is
        cleared. A clean but stale install is still flagged, so it keeps its
        marker as the daily gate.

        An admin is exempt from the record and the email. The check is a query,
        so its answer is cached in the marker, which both spares the query on
        later sessions and lets the import-time warning honour the exemption
        without a connection of its own. A check that could not run is not
        cached.

        Fully guarded, so a user without write permission sees nothing.
        """
        sg_install = spyglass_install
        flagged = sg_install.has_local_changes or sg_install.is_stale
        marker = sg_install.marker.read()

        # Fast path: nothing to record and no resolution to log
        if not flagged and not marker:
            return

        today = date.today().isoformat()
        if flagged and marker.get("last_logged") == today:
            return  # already on record today

        is_admin = sg_install.cached_is_admin
        if is_admin is None:  # not cached, or cached too long ago
            from spyglass.common.common_lab import LabMember

            answered = True
            try:
                is_admin = LabMember().user_is_admin
            except Exception as exc:
                logger.debug(f"admin check failed: {exc}")  # DB down
                is_admin = False  # treat as non-admin for this session only
                answered = False
            # Cache only an answered check that silences a warning
            if flagged and answered:
                sg_install.marker.write(is_admin=is_admin, admin_checked=today)

        if is_admin:
            if not flagged:  # resolved: drop the marker, restore fast path
                sg_install.marker.clear()
            return

        try:
            key = self.insert_current_env()
        except Exception as exc:
            logger.debug(f"dirty env logging failed: {exc}")
            return

        if sg_install.has_local_changes:
            # The row's date, else the warning's, else today
            sg_install.marker.write(
                last_logged=today,
                first_seen=(
                    self._row_first_seen(key)
                    or marker.get("first_seen")
                    or today
                ),
            )
        elif sg_install.is_stale:
            # Still flagged: the marker is the daily gate
            sg_install.marker.write(last_logged=today)
        else:  # resolution recorded; future clean sessions take the fast path
            sg_install.marker.clear()

    def _row_first_seen(self, key: dict) -> Optional[str]:
        """Return the ISO date a logged dirty row first appeared, if readable.

        The database is the shared record, so the warning's countdown defers to
        it, making the marker a cache of the database rather than a second
        clock. ``temp_dir`` is a cache and gets cleared, which would restart a
        local-only countdown while the escalation email kept counting.

        Returns ``None`` when there is nothing to read, leaving the caller's
        local date in place: the first warning of an episode fires before any
        row exists.
        """
        env_id = (key or {}).get("env_id")
        if not env_id:
            return None
        try:
            stamp = (self & f'env_id="{env_id}"').fetch1("timestamp")
            return stamp.date().isoformat()
        except Exception as exc:
            logger.debug(f"could not read logged dirty date: {exc}")
            return None

    def unresolved_dirty(self) -> "dj.expression.QueryExpression":
        """Return partial rows whose dirty state has not been cleaned up.

        Resolution is stamped on the row when the same user next logs a full
        row, rather than inferred by comparing timestamps. MySQL ``timestamp``
        holds whole seconds, so edits discarded in the second they were logged
        produce two rows no comparison can order.

        This is the single definition of "still dirty", shared by the warning
        countdown, the notification query, and the dedupe that keeps a session
        from adding a second row for a dirty state already on record.
        """
        return self & "env_hash IS NULL" & "resolved_at IS NULL"

    def _mark_dirty_resolved(self) -> None:
        """Stamp this user's unresolved partial rows as cleaned up.

        Called when a full row is logged: the user is back on a reproducible
        install, so the escalation clock for their dirty episode stops. Uses
        server ``NOW()``, matching :data:`DIRTY_DAYS_SQL`.

        Guarded: it runs on the way to every full row, including on an
        instance whose table predates the release-note ``alter()``, where
        restricting on ``resolved_at`` raises and would take the caller's
        populate with it.
        """
        dj_user = dj.config.get("database.user", "")
        try:
            open_rows = self.unresolved_dirty() & f"env_id LIKE '{dj_user}\\_%'"
            for env_id in open_rows.fetch("env_id"):
                self.connection.query(
                    f"UPDATE {self.full_table_name} "
                    + "SET `resolved_at` = NOW() "
                    + f"WHERE `env_id` = '{env_id}'"
                )
        except Exception as exc:
            logger.debug(f"could not resolve dirty rows: {exc}")

    def dirty_installs(self) -> "dj.expression.QueryExpression":
        """Return active, unresolved dirty installs with days_since_warn.

        A partial row is included when it is:

        * **Unresolved**: ``resolved_at`` is null; see
          :meth:`unresolved_dirty`, and
        * **Active**: seen within twice ``WARN_DIRTY_ENV_DAYS``. The window
          tests ``last_seen``, which each session refreshes, so a user who is
          dirty and still working stays in scope while an abandoned clone ages
          out.

        ``days_since_warn`` is measured from ``timestamp``, the first
        appearance of this dirty state, so the countdown does not restart when
        a user opens a new session.

        Returns a DataJoint query expression.  Callers needing a DataFrame
        can call ``.fetch(format='frame')`` on the result.
        """
        cutoff = 2 * WARN_DIRTY_ENV_DAYS
        active = f"DATEDIFF(NOW(), `last_seen`) <= {cutoff}"

        return (self.unresolved_dirty() & active).proj(
            ..., days_since_warn=DIRTY_DAYS_SQL
        )

    def stale_installs(self) -> "dj.expression.QueryExpression":
        """Return active full rows whose commit is older than the stale window.

        The database counterpart of
        :attr:`~spyglass.utils.git_utils.SpyglassInstall.is_stale`, which can
        only answer for the clone it runs in. ``commit_date`` is stored so the
        same question can be asked of every logged environment.

        A row is included when it is:

        * **Dated**: ``commit_date`` is not null, so a release install or a
          clone git could not read is out of scope;
        * **Reproducible**: ``env_hash`` is not null. A dirty row is excluded
          because it runs a shorter clock already, through
          :meth:`dirty_installs`;
        * **Active**: seen within twice ``WARN_STALE_DAYS``, on ``last_seen``,
          matching how :meth:`dirty_installs` ages out an abandoned clone.

        ``days_stale`` is measured from ``commit_date``, so it reports the age
        of the code rather than of the row.

        Returns a DataJoint query expression.  Callers needing a DataFrame
        can call ``.fetch(format='frame')`` on the result.
        """
        active = f"DATEDIFF(NOW(), `last_seen`) <= {2 * WARN_STALE_DAYS}"
        stale = f"{STALE_DAYS_SQL} > {WARN_STALE_DAYS}"

        return (
            self
            & "commit_date IS NOT NULL"
            & "env_hash IS NOT NULL"
            & active
            & stale
        ).proj(..., days_stale=STALE_DAYS_SQL)

    def write_dirty_notifications(self, out_path: str = None) -> None:
        """Write overdue dirty-install rows to out_path (or stdout if None).

        One tab-separated line per user, not per row. ``run_jobs.sh`` sends
        one email per line on each cron run, so a user with several flagged
        clones, or one clone logged at several commits, gets a single line
        naming all of them:

            user_email\\tdays_since_warn\\tspyglass_commit\\tdirty_path

        ``days_since_warn`` is the largest across that user's rows, i.e. their
        oldest unresolved episode, and the two trailing fields are
        space-separated when there is more than one. Only users whose
        ``days_since_warn`` exceeds ``WARN_DIRTY_ENV_DAYS`` appear.

        Admins are excluded. ``log_install`` exempts them, but
        ``insert_current_env`` has no admin check and recompute reaches it
        through ``RecordingRecomputeSelection.env_dict`` on every spike
        sorting make, so an admin on a dirty clone accumulates partial rows
        that no clean session of theirs resolves.

        An absent commit or clone path is written as ``unknown`` rather than
        left empty: a dirty wheel has neither, and ``run_jobs.sh`` reads these
        lines with ``IFS=$'\\t'``, where bash collapses adjacent tabs and
        shifts every later field left.

        Called by ``maintenance_scripts/cleanup.py``; output is consumed by
        ``run_jobs.sh`` which sends emails via ``script_utils.sh``.  A user with
        no ``LabMember.LabMemberInfo`` email record is reported once to stderr.

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
            "datajoint_user_name", "google_user_name", "admin", as_dict=True
        )
        email_by_dj_user = {
            m["datajoint_user_name"]: m["google_user_name"]
            for m in member_rows
            if m["datajoint_user_name"]
        }
        admins = {m["datajoint_user_name"] for m in member_rows if m["admin"]}
        known_users = sorted(email_by_dj_user, key=len, reverse=True)

        by_email: Dict[str, dict] = {}
        unmatched = set()
        for row in overdue:
            env_id = row["env_id"]
            dj_user = next(
                (u for u in known_users if env_id.startswith(u + "_")),
                None,
            )
            if dj_user in admins:  # exempt, and not an unmatched user
                continue
            user_email = email_by_dj_user.get(dj_user) if dj_user else None
            if user_email is None:
                unmatched.add(dj_user or env_id)
                continue

            entry = by_email.setdefault(user_email, {"days": 0, "clones": {}})
            entry["days"] = max(entry["days"], row["days_since_warn"])
            # Name each clone once, at its newest commit
            path = row["dirty_path"] or "unknown"
            age = row["days_since_warn"]
            seen = entry["clones"].get(path)
            if seen is None or age < seen[0]:
                commit = row["spyglass_commit"] or "unknown"
                entry["clones"][path] = (age, commit)

        for name in sorted(unmatched):
            print(
                f"No LabMemberInfo email for '{name}'. Skipping.",
                file=sys.stderr,
            )

        out = open(out_path, "w") if out_path else sys.stdout
        try:
            for user_email, entry in sorted(by_email.items()):
                paths = sorted(entry["clones"])
                commits = [entry["clones"][p][1] for p in paths]
                print(
                    "\t".join(
                        [
                            user_email,
                            str(entry["days"]),
                            " ".join(commits),
                            " ".join(paths),
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
        if fetched_env is None:
            logger.error(
                f"Environment '{env_id}' was logged from a dirty install, "
                + "so it holds no environment to write."
            )
            return

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

        if stored_hash is None:  # partial row: a dirty install stored no env
            logger.error(
                f"Environment '{env_id}' was logged from a dirty install, "
                + "so it holds no environment to compare."
            )
            return False  # an empty prev_env matches everything

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
        if env is None:  # partial row: parse_env_dict would read the live env
            return {env_id: ""}

        dependencies = self.parse_env_dict(env)

        return {env_id: dependencies.get(dep_name, "")}

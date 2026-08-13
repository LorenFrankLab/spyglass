import os
import re
import stat as stat_module
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple, Union

import datajoint as dj
import h5py
import numpy as np
import pandas as pd
import pynwb
import spikeinterface as si
from hdmf.common import DynamicTable
from pynwb.core import ScratchData
from tqdm import tqdm

from spyglass import __version__ as sg_version

# CleanupCandidate is re-exported (constructed as common_nwbfile.CleanupCandidate
# in tests) even though this module's own code no longer references it directly.
from spyglass.common._nwbfile_cleanup import (
    AccessSnapshot,
    CleanupCandidate,
    CleanupExecutor,
    CleanupPlan,
    CleanupPlanner,
    TargetSnapshot,
    _check_number,
    _PhysicalRoot,
    _TrackedFileState,
)
from spyglass.settings import analysis_dir, config, raw_dir
from spyglass.utils import SpyglassAnalysis, SpyglassMixin, logger
from spyglass.utils.dj_helper_fn import get_child_tables
from spyglass.utils.nwb_hash import NwbfileHasher
from spyglass.utils.nwb_helper_fn import get_electrode_indices, get_nwb_file

# A trigger is a DB object that is automatically executed when INSERT occurs
SQL_TRIGGER_QUERY = """
SELECT COUNT(*)
FROM information_schema.TRIGGERS
WHERE TRIGGER_SCHEMA = '{database}'
  AND TRIGGER_NAME = '{trigger}'
"""

SQL_BLOCK_TEMPLATE = """
CREATE TRIGGER {database}.{trigger}
BEFORE INSERT ON {table}
FOR EACH ROW
BEGIN
  SIGNAL SQLSTATE '45000'
    SET MESSAGE_TEXT = 'Inserts disabled during maintenance: {table}';
END
"""

schema = dj.schema("common_nwbfile")


@schema
class Nwbfile(SpyglassMixin, dj.Manual):
    definition = """
    # Table for holding the NWB files.
    nwb_file_name: varchar(64)   # name of the NWB file
    ---
    nwb_file_abs_path: filepath@raw
    INDEX (nwb_file_abs_path)
    """
    # NOTE the INDEX above is implicit from filepath@... above but needs to be
    # explicit so that alter() can work

    # NOTE: See #630, #664. Excessive key length.

    @classmethod
    def insert_from_relative_file_name(cls, nwb_file_name: str) -> None:
        """Insert a new session from an existing NWB file.

        Parameters
        ----------
        nwb_file_name : str
            The relative path to the NWB file.
        """
        nwb_file_abs_path = Nwbfile.get_abs_path(nwb_file_name, new_file=True)

        if not Path(nwb_file_abs_path).exists():
            raise FileNotFoundError(f"File not found: {nwb_file_abs_path}")

        cls.insert1(
            dict(
                nwb_file_name=nwb_file_name, nwb_file_abs_path=nwb_file_abs_path
            ),
            skip_duplicates=True,
        )

    def fetch_nwb(self):
        return [
            get_nwb_file(self.get_abs_path(file))
            for file in self.fetch("nwb_file_name")
        ]

    @classmethod
    def get_abs_path(
        cls, nwb_file_name: str, new_file: bool = False, **kwargs
    ) -> str:
        """Return absolute path for a stored raw NWB file given file name.

        The SPYGLASS_BASE_DIR must be set, either as an environment or part of
        dj.config['custom']. See spyglass.settings.load_config

        Parameters
        ----------
        nwb_file_name : str
            The name of an NWB file that has been inserted into the Nwbfile()
            table. May be file substring. May include % wildcard(s).
        new_file : bool, optional
            Adding a new file to Nwbfile table. Defaults to False.

        Returns
        -------
        nwb_file_abspath : str
            The absolute path for the given file name.
        """
        file_path = raw_dir + "/" + nwb_file_name
        if new_file:
            return file_path

        query = cls & {"nwb_file_name": nwb_file_name}
        if len(query) != 1:
            raise ValueError(
                f"Could not find 1 entry for {nwb_file_name}:\n{query}"
            )

        return file_path

    @staticmethod
    def add_to_lock(nwb_file_name: str) -> None:
        """Add the specified NWB file to the list of locked items.

        The NWB_LOCK_FILE environment variable must be set to the path of the
        lock file, listing locked NWB files.

        Parameters
        ----------
        nwb_file_name : str
            The name of an NWB file in the Nwbfile table.
        """
        if not (Nwbfile() & {"nwb_file_name": nwb_file_name}):
            raise FileNotFoundError(
                f"File not found in Nwbfile table. Cannot lock {nwb_file_name}"
            )

        with open(os.getenv("NWB_LOCK_FILE"), "a+") as lock_file:
            lock_file.write(f"{nwb_file_name}\n")

    @staticmethod
    def cleanup(delete_files: bool = False) -> None:
        """Remove the filepath entries for NWB files that are not in use.

        This does not delete the files themselves unless delete_files=True is
        specified. Run this after deleting the Nwbfile() entries themselves.
        """
        schema.external["raw"].delete(delete_external_files=delete_files)


@schema
class AnalysisRegistry(dj.Manual):
    """Central registry tracking all custom AnalysisNwbfile tables.

    This table maintains a record of all team-specific AnalysisNwbfile tables
    to enable coordinated cleanup, export operations, and cross-table queries.
    Tables are auto-registered when declared via SpyglassAnalysis mixin.

    Key Methods:
        get_class(prefix) - Get AnalysisNwbfile class for a specific team prefix
        get_all_classes() - Get all registered AnalysisNwbfile class objects
        get_tracked_files() - Get all files tracked across all custom tables
        clear_cache() - Clear the class cache (useful for testing)

    Usage:
        from spyglass.common import AnalysisRegistry

        # View all registered tables
        AnalysisRegistry().fetch()

        # Get a specific team's table
        MyTeamAnalysis = AnalysisRegistry().get_class("myteam")

        # Get an instance
        my_team_analysis = MyTeamAnalysis()

        # Use enhanced helper methods
        my_team_analysis.get_prefix()  # 'myteam'
    """

    definition = """
    full_table_name: varchar(128)  # full table name of the analysis
    ---
    created_at = CURRENT_TIMESTAMP: timestamp  # when registered
    created_by : varchar(32)                   # who registered
    """

    # Class-level cache for dynamic table classes
    _class_cache: dict = {}

    def insert1(self, key: Union[str, dict], **kwargs) -> None:
        """Auto-add created_by if not provided.

        Parameters
        ----------
        key : str or dict
            The full_table_name as a string or a dict with the key
            'full_table_name'.
        kwargs : additional arguments to pass to insert1.
        """

        if isinstance(key, str):
            key = {"full_table_name": key}

        if query := self & key:
            logger.debug(f"Entry already exists: {key['full_table_name']}")
            return query

        if "created_by" not in key:
            key["created_by"] = dj.config["database.user"]

        super().insert1(key, **kwargs)

    # ---------------- Spawn analysis classes from table names ----------------

    @staticmethod
    def _parse_table_name(
        full_table_name: str,
    ) -> tuple[str, str, str, str]:
        """Parse full table name into components.

        Extracts database, table name, prefix, and suffix from a full name.

        Parameters
        ----------
        full_table_name : str
            Full table name in format `database`.`table_name`
            where database follows {prefix}_{suffix} convention.

        Returns
        -------
        database : str
            The database name (e.g., "testuser_nwbfile")
        table_name : str
            The table name (e.g., "analysis_nwbfile")
        prefix : str
            The database prefix before last underscore (e.g., "testuser")
        suffix : str
            The database suffix after last underscore (e.g., "nwbfile")

        Example
        -------
        >>> table_name = "`user_nwbfile`.`analysis_nwbfile`"
        >>> AnalysisRegistry._parse_table_name(table_name)
        ('user_nwbfile', 'analysis_nwbfile', 'user', 'nwbfile')
        """
        # Remove backticks and split into database and table
        database, table_name = full_table_name.replace("`", "").split(".")
        # Split database into prefix and suffix at last underscore
        prefix, suffix = database.rsplit("_", 1)

        return database, table_name, prefix, suffix

    def _is_valid_entry(
        self, full_table_name: str, raise_err: bool = True
    ) -> bool:
        """Check if the given table name corresponds to a valid SpyglassAnalysis.

        Parameters
        ----------
        full_table_name : str
            The full table name.
        raise_err : bool, optional
            If True, raise ValueError on invalid table name.

        Returns
        -------
        is_valid : bool
            True if the table is a valid SpyglassAnalysis, False otherwise.


        """
        database, table_name, prefix, suffix = self._parse_table_name(
            full_table_name
        )

        err = None

        if table_name != "analysis_nwbfile":
            err = f"Table name must be 'analysis_nwbfile': {table_name}"

        if suffix != "nwbfile":
            err = f"Database suffix must be 'nwbfile': {suffix}"

        # Validate prefix (alphanumeric and underscore only)
        if not re.match(r"^[a-z0-9_]+$", prefix, re.IGNORECASE):
            err = f"Invalid prefix format: {prefix}"

        if raise_err and err is not None:
            raise ValueError(err)

        return err is None

    @classmethod
    def _create_class(cls, full_name: str) -> type:
        """Create an enhanced custom analysis table class.

        Returns a class with:
        - All AnalysisMixin methods
        - Cached for reuse
        - Helper methods for common operations
        - Better repr and documentation

        Parameters
        ----------
        full_name : str
            Full table name (e.g., "`myteam_nwbfile`.`analysis_nwbfile`")

        Returns
        -------
        type
            Enhanced table class with helper methods
        """
        database, table_name, prefix, _ = cls._parse_table_name(full_name)
        camel_name = dj.utils.to_camel_case(table_name)

        if (
            database not in dj.list_schemas()
            or table_name not in dj.Schema(database).list_tables()
        ):
            raise dj.errors.MissingTableError(
                f"Cannot create class for missing table: {full_name}. "
                "Ensure the schema is created and you have permissions to it."
                f"with dj.list_schemas(); dj.Schema({database}).list_tables()"
            )

        class AnalysisMeta(type):
            """Metaclass for custom AnalysisNwbfile classes."""

            def __repr__(cls):
                """Enhanced class repr showing prefix."""
                return (
                    f"<class '{cls.__name__}'"
                    + f"prefix='{cls._analysis_prefix}'>"
                )

        class EnhancedAnalysisNwbfile(
            SpyglassAnalysis, dj.FreeTable, metaclass=AnalysisMeta
        ):
            f"""Custom AnalysisNwbfile table for {prefix}.

            Automatically created by AnalysisRegistry for schema {full_name}.
            Provides same functionality as common AnalysisNwbfile but with
            isolated database locks.
            """

            full_table_name = full_name
            _analysis_prefix = prefix

            def __init__(self):
                # Always pass connection and table name to FreeTable
                super().__init__(conn=dj.conn(), full_table_name=full_name)

            def __repr__(self) -> str:
                """Enhanced repr showing custom table info."""

                return (
                    f"<{camel_name} (custom '{prefix}' analysis table)>\n"
                    + super().__repr__()
                )

        # Set the class name dynamically
        EnhancedAnalysisNwbfile.__name__ = camel_name
        EnhancedAnalysisNwbfile.__qualname__ = camel_name
        EnhancedAnalysisNwbfile.__module__ = (
            f"spyglass.common.common_nwbfile[{prefix}]"
        )

        return EnhancedAnalysisNwbfile

    def _get_tbl_from_name(self, full_name: str) -> type:
        """Return cached or create enhanced table class.

        Now uses caching and creates enhanced classes with helper methods.

        Parameters
        ----------
        full_name : str
            The full table name.

        Returns
        -------
        type
            Enhanced table class with caching
        """
        # Check cache first
        if full_name in self._class_cache:
            return self._class_cache[full_name]

        # Create enhanced class
        cls = self._create_class(full_name)

        # Cache it
        self._class_cache[full_name] = cls

        return cls

    def get_class(self, key: Union[str, Dict]) -> Optional[type]:
        """Return the class object for the given full_table_name, uninitialized.

        Parameters
        ----------
        key : str or dict
            The prefix or full_table_name as a string or a dict with the key
            'full_table_name'.

        Returns
        -------
        class_obj : type or None
            The class object for the given full_table_name, or None.
        """
        if isinstance(key, str) and "analysis_nwbfile" not in key:
            key = f"`{key}_nwbfile`.`analysis_nwbfile`"
        if isinstance(key, str):
            key = {"full_table_name": key}

        # TODO: Add common case to table on registry declaration
        common_map = {
            Nwbfile().full_table_name: Nwbfile,
            AnalysisNwbfile().full_table_name: AnalysisNwbfile,
        }

        if key["full_table_name"] in common_map:
            return common_map[key["full_table_name"]]

        if not (self & key):
            logger.warning(f"Entry not found: {key['full_table_name']}")
            return None

        return self._get_tbl_from_name(key["full_table_name"])

    @property
    def all_classes(self) -> List[SpyglassAnalysis]:
        """Return all registered analysis table class objects, initialized.

        Returns
        -------
        class_objs : list of dj.FreeTable
            A list of all registered analysis table class objects.
        """
        return [
            self._get_tbl_from_name(key["full_table_name"])()
            for key in self.fetch(as_dict=True)
            if self._is_valid_entry(key["full_table_name"], raise_err=False)
        ]

    @classmethod
    def clear_cache(cls):
        """Clear the class cache.

        Useful for testing or when custom tables are modified.
        After clearing, the next call to get_class() will recreate
        the class objects.

        Examples
        --------
        >>> AnalysisRegistry.clear_cache()
        >>> # Next get_class() call will create fresh class instances
        """
        cls._class_cache.clear()

    def get_externals(
        self, store: str = "analysis"
    ) -> List[dj.external.ExternalTable]:
        """Return external table objects for all registered analysis schemas.

        External tables are used to manage externally stored files (e.g., on S3).
        Each custom AnalysisNwbfile schema has a corresponding external table
        named `{prefix}_nwbfile`.`~external_analysis`.

        Used for updating externals after file surgeries or migrations.

        Parameters
        ----------
        store : str, optional
            The external store name to use. Default is "analysis".
            This should match a configured store in dj.config['stores'].

        Returns
        -------
        externals : list of dj.external.ExternalTable
            A list of ExternalTable objects for all registered schemas.

        Example
        -------
        >>> from spyglass.common import AnalysisRegistry
        >>> registry = AnalysisRegistry()
        >>> externals = registry.get_externals()
        >>> for ext in externals:
        ...     print(ext.database)
        """
        ExtTable = dj.external.ExternalTable
        ext_kwargs = dict(connection=dj.conn(), store=store)

        # Get unique database prefixes from registered tables
        databases = set(
            [
                self._parse_table_name(tbl_name)[0]
                for tbl_name in self.fetch("full_table_name")
                if self._is_valid_entry(tbl_name, raise_err=False)
            ]
        )

        return [  # Create ExternalTable for each database
            ExtTable(**ext_kwargs, database=database)
            for database in sorted(databases)
        ]

    # ------------------ Blocking inserts during maintenance ------------------

    def _get_block_info(self, table: str) -> str:
        """Parse table name to get database and trigger name."""
        _ = self._is_valid_entry(table, raise_err=True)

        # Extract database and prefix using helper method
        database, _, prefix, _ = self._parse_table_name(table)

        return database, f"{prefix}_block_inserts"

    def _block_exists(self, table: str) -> bool:
        """Check if a block trigger exists for the given table.

        Parameters
        ----------
        table : str
            The full table name of the analysis table to check.

        Returns
        -------
        exists : bool
            True if the block trigger exists, False otherwise.
        """
        database, trigger = self._get_block_info(table)
        kwargs = dict(database=database, trigger=trigger, table=table)

        result = dj.conn().query(SQL_TRIGGER_QUERY.format(**kwargs))
        return result.fetchone()[0] > 0

    def _block_single_table(
        self, table: str, dry_run: bool = False
    ) -> Optional[str]:
        """Block new inserts into a single analysis table.

        Parameters
        ----------
        table : str
            The full table name of the analysis table to block.
        dry_run : bool, optional
            If True, log blocking without making changes. Defaults to False.

        Returns
        -------
        error_msg : str or None
            A message when the table could not be blocked -- trigger creation
            failed, or a blocker already exists -- otherwise None.
        """
        try:
            database, trigger = self._get_block_info(table)
            kwargs = dict(database=database, trigger=trigger, table=table)

            if self._block_exists(table):
                return (
                    f"Failed to block {table}: blocking trigger already "
                    "exists; another cleanup may be active or the trigger "
                    "may be stale"
                )

            if dry_run:
                logger.info(f"Dry run: would block inserts into {table}")
                return None

            # Create trigger
            dj.conn().query(SQL_BLOCK_TEMPLATE.format(**kwargs))

        except Exception as e:
            return f"Failed to block {table}: {e}"

        return None

    def block_new_inserts(self, dry_run: bool = False) -> None:
        """Block new inserts into all registered analysis tables.

        Creates BEFORE INSERT triggers on all registered custom analysis tables
        to prevent data modifications during maintenance operations. Refuses
        to adopt an existing trigger because it may belong to another cleanup.

        Parameters
        ----------
        dry_run : bool, optional
            If True, log blocking without making changes. Defaults to False.

        Raises
        ------
        RuntimeError
            If blocker inspection or trigger creation fails, or if any blocker
            already exists. A partial creation failure may leave some tables
            blocked because triggers do not carry per-run ownership.
        """
        # Freeze one deterministic acquisition order. Concurrent acquisitions
        # with the same stable registry snapshot then contend on the same first
        # trigger rather than each creating a different prefix of the set.
        tables = sorted(set(self.fetch("full_table_name")))

        # Inspect every table before any DDL. A pre-existing trigger is either
        # owned by an active cleanup or stale after a failed one; adopting it
        # would let this call later remove a trigger it did not create.
        existing = []
        for table in tables:
            try:
                if self._block_exists(table):
                    existing.append(table)
            except Exception as err:
                raise RuntimeError(
                    f"Failed to inspect insert blocker for {table}; refusing "
                    f"cleanup before creating any triggers: {err}"
                ) from err

        if existing:
            blocked = "\n".join(f"  - {table}" for table in existing)
            raise RuntimeError(
                "Refusing cleanup because insert-blocking triggers already "
                f"exist for:\n{blocked}\nAnother cleanup may be active, or "
                "the triggers may be stale. Confirm that no cleanup is active "
                "and inspect the blockers before removing them. Use "
                "AnalysisRegistry().unblock_new_inserts() only after confirming "
                "every blocker is stale; that helper removes all registered "
                "analysis blocking triggers."
            )

        for table in tables:
            error = self._block_single_table(table, dry_run=dry_run)
            if error is not None:
                # Raise instead of continuing: given the deterministic order
                # above, a concurrent run has claimed this same first trigger,
                # so the rest of the loop would only create triggers to undo.
                raise RuntimeError(
                    f"Failed to block 1 table(s):\n{error}\nSome analysis "
                    "tables may remain blocked. Another cleanup may be active, "
                    "or a trigger may be stale. Confirm that no cleanup is "
                    "active, then inspect the blocking triggers before using "
                    "AnalysisRegistry().unblock_new_inserts(). That helper "
                    "removes all registered analysis blocking triggers; use "
                    "it only after confirming every such trigger is stale."
                )

    def unblock_new_inserts(self) -> None:
        """Unblock new inserts into all registered analysis tables.

        Removes BEFORE INSERT triggers from all registered custom analysis
        tables, re-enabling normal insert operations.

        Raises
        ------
        RuntimeError
            If any trigger removal fails.
        """
        errors = []

        for table in self.fetch("full_table_name"):
            try:
                database, trigger = self._get_block_info(table)
                if not self._block_exists(table):
                    continue
                dj.conn().query(f"DROP TRIGGER {database}.{trigger};")
            except Exception as e:
                errors.append(f"Failed to unblock {table}: {e}")

        if errors:
            raise RuntimeError(
                f"Failed to unblock {len(errors)} table(s):\n"
                + "\n".join(errors)
            )


@schema
class AnalysisNwbfile(SpyglassAnalysis, dj.Manual):
    definition = """
    # Table for NWB files that contain results of analysis.
    analysis_file_name: varchar(64)                # name of the file
    ---
    -> Nwbfile                                     # name of the parent NWB file. Used for naming and metadata copy
    analysis_file_abs_path: filepath@analysis      # the full path to the file
    analysis_file_description = "": varchar(2000)  # an optional description of this analysis
    analysis_parameters = NULL: blob               # additional relevant parameters. Currently used only for analyses
                                                   # that span multiple NWB files
    INDEX (analysis_file_abs_path)
    """
    # NOTE the INDEX above is implicit from filepath@...
    # above but needs to be explicit so that alter() can work

    # See #630, #664. Excessive key length.

    def _walk_analysis_files(self) -> Iterator[Path]:
        """Yield every ``*.nwb`` entry under the analysis directory.

        Directory symlinks are not followed, matching prior behavior. A
        walk error is re-raised rather than skipped: a destructive sweep
        must not act on a partial view of the tree.

        Yields
        ------
        pathlib.Path
            Path as found by the walk, which may itself be a symlink.
        """

        def _reraise(err: OSError) -> None:
            raise err

        scan_root = str(Path(self._analysis_dir).expanduser())
        for dirpath, _dirnames, filenames in os.walk(
            scan_root, followlinks=False, onerror=_reraise
        ):
            for fname in filenames:
                if fname.endswith(".nwb"):
                    yield Path(dirpath) / fname

    def _snapshot_entry(
        self, access_path: Path
    ) -> Optional[Tuple[Path, Optional[TargetSnapshot], AccessSnapshot]]:
        """Snapshot one scanned entry.

        Returns
        -------
        tuple or None
            ``(real_path, target_or_None, access_snapshot)``, or None when the
            entry cannot be snapshotted -- it vanished mid-scan, is unreadable,
            or is unresolvable -- and should be skipped.

        Notes
        -----
        Per-entry errors are logged and skipped, not raised: an entry that
        cannot be stat'd never becomes a deletion candidate, so skipping can
        only under-clean. Errors from the directory *walk* remain fatal --
        there a partial view means unseen files, which would skew the
        deletion-limit denominators.
        """
        try:
            access = AccessSnapshot.from_path(access_path)
        except OSError as err:
            # Skip, don't abort (the Notes above cover why per-entry skips are
            # safe): aborting on one bad symlink -- a loop, a 0700 parent --
            # would wedge the periodic sweep and every later maintenance phase.
            logger.warning(f"Skipping unreadable analysis entry: {err}")
            return None

        is_link = access.is_link
        real_path = Path(os.path.realpath(access_path))

        try:
            st = os.stat(access_path)  # follows symlinks
        except FileNotFoundError:
            if is_link:
                return real_path, None, access  # broken link
            return None  # regular entry vanished mid-scan
        except OSError as err:
            # Symlink loop (ELOOP), unreadable component, or I/O error.
            # Skip rather than abort, for the reason above.
            logger.warning(f"Skipping unresolvable analysis entry: {err}")
            return None

        target = TargetSnapshot(
            real_path=real_path,
            dev=st.st_dev,
            ino=st.st_ino,
            size=st.st_size,
            mtime_ns=st.st_mtime_ns,
            ctime_ns=st.st_ctime_ns,
            mode=st.st_mode,
        )
        return real_path, target, access

    def _build_untracked_file_plan(
        self,
        custom_tables: List[SpyglassAnalysis],
        *,
        min_file_age_hours: float = 24.0,
        now_ns: Optional[int] = None,
    ) -> CleanupPlan:
        """Build a cleanup plan for untracked or empty analysis NWB files.

        Parameters
        ----------
        custom_tables : list
            Custom analysis table instances whose tracked files count as
            tracked here.
        min_file_age_hours : float, optional
            Candidates whose target or any access alias is newer than this
            are deferred. Defaults to 24.0. Pass 0 to disable.
        now_ns : int, optional
            Injected clock in nanoseconds. Defaults to ``time.time_ns()``.
            Tests must inject: ``os.utime`` cannot backdate ``ctime``, so a
            real file always reads as new under ``max(mtime, ctime)``.

        Returns
        -------
        CleanupPlan
        """
        planner = CleanupPlanner(
            # _walk_analysis_files is called through the bound method so its
            # test monkeypatch still applies; the tqdm wrapper stays here to
            # keep CleanupPlanner stdlib-only.
            walker=lambda: tqdm(
                self._walk_analysis_files(),
                desc="Scanning analysis files  ",  # extra spaces for alignment
            ),
            snapshotter=self._snapshot_entry,
            tracking_state_fn=self._current_tracked_state,
            logger=logger,
            min_file_age_hours=min_file_age_hours,
            now_ns=now_ns,
        )
        return planner.build(custom_tables)

    @staticmethod
    def _validate_cleanup_plan(
        plan: CleanupPlan,
        *,
        max_delete_fraction: float = 0.9,
        max_delete_to_tracked_ratio: float = 10.0,
    ) -> tuple[bool, str | None]:
        """Delegate to :meth:`CleanupPlan.validate`.

        Retained as a thin wrapper so existing callers -- and tests that call
        ``AnalysisNwbfile._validate_cleanup_plan(plan, ...)`` directly -- keep
        working while the policy check itself lives on ``CleanupPlan``.
        """
        return plan.validate(
            max_delete_fraction=max_delete_fraction,
            max_delete_to_tracked_ratio=max_delete_to_tracked_ratio,
        )

    @staticmethod
    def _current_custom_tables(
        snapshot: List[SpyglassAnalysis],
    ) -> List[SpyglassAnalysis]:
        """Re-read registry membership, unioned with the initial snapshot.

        The insert-blocking triggers only cover tables that existed when the
        triggers were installed, so a custom table declared mid-cleanup is
        invisible to them and its files would otherwise read as untracked.

        A refresh failure is **fatal**, not a fallback: falling back to the
        snapshot is safe only if the snapshot is a superset of live
        membership, which is exactly false in the case this refresh exists
        to catch. Aborting leaves files in place; guessing deletes them.

        The result is the union of the snapshot and live registry classes,
        deduplicated by ``full_table_name``, so a table can only ever be
        added to the tracked set, never dropped from it.
        """
        live = list(AnalysisRegistry().all_classes)  # let errors propagate
        merged = {}
        for tbl in list(snapshot) + live:
            merged[tbl.full_table_name] = tbl
        return list(merged.values())

    @staticmethod
    def _other_managed_roots() -> List[_PhysicalRoot]:
        """Spyglass stores that analysis cleanup must never delete from.

        ``tracked`` is built from the *analysis* external store only, so a
        file belonging to another store reads as untracked. Without this,
        a stray ``ln -s $SPYGLASS_RAW_DIR/sub-x.nwb analysis/copy.nwb``
        would let the weekly sweep delete a raw acquisition file, which is
        not recomputable. Every non-analysis Spyglass store is protected,
        not only the acquisition stores: the analysis registry cannot
        establish ownership of any of them, so each is refused regardless of
        whether its contents are recomputable. Checked by path containment
        rather than by unioning the other stores' externals, so it costs no
        queries.

        Every configured root is resolved and stat'd. Its directory identity
        lets the containment check recognize case variants and alternate
        names of the configured root itself (including a whole-root bind
        mount), while still allowing a separate hard link outside the store
        to be unlinked. Any resolution or inspection failure aborts cleanup.
        A missing configured spelling can represent an unavailable mount while
        the same storage remains reachable through another alias; silently
        omitting it would disable the corresponding guard.
        """
        excluded = {"SPYGLASS_BASE_DIR", "SPYGLASS_ANALYSIS_DIR"}
        configured_roots = sorted(
            (key, value)
            for key, value in config.items()
            if isinstance(key, str)
            and key.endswith("_DIR")
            and key not in excluded
            and value
        )

        roots = []
        for setting_name, setting in configured_roots:
            store_name = setting_name.removesuffix("_DIR").lower()
            if store_name.startswith("spyglass_"):
                store_name = store_name.removeprefix("spyglass_")
            try:
                path = Path(setting).expanduser().resolve()
                root_stat = os.stat(path)
                if not stat_module.S_ISDIR(root_stat.st_mode):
                    raise NotADirectoryError(
                        f"protected store root is not a directory: {path}"
                    )
                roots.append(
                    _PhysicalRoot(
                        name=store_name,
                        path=path,
                        dev=root_stat.st_dev,
                        ino=root_stat.st_ino,
                    )
                )
            except (OSError, RuntimeError, TypeError, ValueError) as err:
                raise RuntimeError(
                    "Cannot resolve protected Spyglass "
                    f"{store_name} store root {setting!r}; refusing "
                    "analysis cleanup"
                ) from err
        return roots

    def _current_tracked_state(
        self, custom_tables: List[SpyglassAnalysis]
    ) -> _TrackedFileState:
        """Re-fetch paths and identities for every tracked analysis file.

        A candidate is keyed by its resolved target, while DataJoint stores
        the access path used for registration. A file can therefore become
        tracked after the scan through a *new* alias that is absent from the
        candidate's access snapshots. Exact restrictions on those snapshots
        miss that registration and can delete its target. Resolved paths
        preserve the behavior for missing entries; target and leaf identities
        additionally recognize hard links, case variants, and mount aliases.

        Registry and external-table errors deliberately propagate. Failing
        closed may under-clean; falling back to stale tracking can delete a
        newly registered file. Missing paths are normal stale external state,
        so they retain their resolved-path fallback without an identity.
        """
        tracked = _TrackedFileState(set(), set(), set())
        for tbl in [self, *custom_tables]:
            for _, path in tbl._ext_tbl.fetch_external_paths():
                access_path = Path(path).expanduser()
                try:
                    resolved = access_path.resolve()
                except (OSError, RuntimeError) as err:
                    raise RuntimeError(
                        "Cannot resolve tracked analysis path "
                        f"{access_path}; refusing analysis cleanup"
                    ) from err
                tracked.resolved_paths.add(resolved)

                try:
                    access_stat = os.lstat(access_path)
                except FileNotFoundError:
                    continue
                except OSError as err:
                    raise RuntimeError(
                        "Cannot inspect tracked analysis path "
                        f"{access_path}; refusing analysis cleanup"
                    ) from err
                tracked.access_identities.add(
                    (access_stat.st_dev, access_stat.st_ino)
                )

                try:
                    target_stat = os.stat(access_path)
                except FileNotFoundError:
                    # A dangling tracked symlink has a live leaf identity but
                    # no target identity. A path that vanished between lstat
                    # and stat is likewise safe to treat as missing.
                    continue
                except OSError as err:
                    raise RuntimeError(
                        "Cannot inspect target of tracked analysis path "
                        f"{access_path}; refusing analysis cleanup"
                    ) from err
                tracked.target_identities.add(
                    (target_stat.st_dev, target_stat.st_ino)
                )
        return tracked

    @staticmethod
    def _access_still_matches(
        access: AccessSnapshot, *, analysis_root: Path
    ) -> bool:
        """Re-verify one leaf symlink immediately before unlinking it.

        Proves the leaf is still the entry that was scanned -- same type,
        inode, timestamps, raw target, and still inside the analysis root.
        Its canonical destination was checked while the whole alias chain
        still existed, during the final validation before target deletion.
        It cannot be checked here after the target has intentionally been
        removed; requiring resolution at this point would strand chained
        aliases.
        """
        try:
            lst = os.lstat(access.access_path)
        except (OSError, RuntimeError):
            return False
        if not stat_module.S_ISLNK(lst.st_mode):
            return False
        if (lst.st_dev, lst.st_ino) != (access.dev, access.ino):
            return False
        if (lst.st_mtime_ns, lst.st_ctime_ns) != (
            access.mtime_ns,
            access.ctime_ns,
        ):
            return False
        try:
            if os.readlink(access.access_path) != access.raw_link_target:
                return False
        except (OSError, RuntimeError):
            return False
        # Still inside the analysis root, resolving the PARENT so the link
        # itself is not followed.
        try:
            location = (
                access.access_path.parent.resolve() / access.access_path.name
            )
        except (OSError, RuntimeError):
            return False
        return location.is_relative_to(analysis_root)

    def _remove_untracked_files(
        self,
        custom_tables: List[SpyglassAnalysis],
        dry_run: bool = True,
        plan: CleanupPlan | None = None,
        *,
        min_file_age_hours: float = 24.0,
        now_ns: Optional[int] = None,
    ) -> tuple[Set[Path], Set[Path]]:
        """Remove analysis files that are empty (0 bytes) or not tracked.

        WARNING: This function makes `analysis_dir` a privileged directory.
        An in-root ``*.nwb`` symlink authorizes deletion of its resolved target
        on another volume when that target is not tracked in ANY schema
        externals. Both the live alias destination and target identity are
        re-verified during the final validation before deletion. Validation
        and unlink are separate filesystem operations; callers must prevent
        concurrent writers from mutating eligible paths during cleanup.

        NOTE: Subprocess would be faster, but this prioritizes cross-platform
        compatibility.

        Parameters
        ----------
        dry_run : bool, optional
            If True, return the plan's resolved candidate target paths.
            Defaults to True. This is not an exact unlink manifest: protected-
            store and per-candidate re-checks run only on the real deletion
            path, so some candidates may ultimately be refused, while applying
            an accepted symlink candidate also unlinks its authorizing leaf
            symlink even though that access path is not in the returned set.
        custom_tables : list
            List of custom analysis table instances to check for tracked files.
        plan : CleanupPlan, optional
            Precomputed cleanup plan. If omitted, the directory is scanned.
        min_file_age_hours : float, optional
            Minimum age of the target and every access alias for a candidate to
            be eligible; younger candidates are deferred. Defaults to 24.0. Pass
            0 to disable. Re-checked at act time.
        now_ns : int, optional
            Injected clock in nanoseconds for the act-time age recheck. Defaults
            to ``time.time_ns()``.

        Returns
        -------
        tuple[Set[Path], Set[Path]]
            ``(candidate_target_paths, tracked_files)``. The first set is the
            plan's candidate targets in both modes: an over-approximation of
            what a real run deletes (candidates can still be refused at act
            time) and an under-approximation of the leaves unlinked (an accepted
            symlink candidate also removes its authorizing leaf).
        """

        if plan is None:
            plan = self._build_untracked_file_plan(
                custom_tables,
                min_file_age_hours=min_file_age_hours,
                now_ns=now_ns,
            )

        # Wire the DataJoint-coupled pieces into the schema-agnostic executor:
        # tracked-state and registry refreshes, the protected-root snapshot,
        # and per-leaf re-validation. _other_managed_roots and
        # _access_still_matches stay methods here so their test seams hold.
        executor = CleanupExecutor(
            plan,
            analysis_dir=self._analysis_dir,
            tracking_refresher=self._current_tracked_state,
            registry_refresher=self._current_custom_tables,
            managed_roots_fn=self._other_managed_roots,
            access_validator=self._access_still_matches,
            logger=logger,
            min_file_age_hours=min_file_age_hours,
            now_ns=now_ns,
        )
        return executor.execute(custom_tables, dry_run=dry_run)

    def _cleanup_custom_table(
        self,
        analysis_tbl: SpyglassAnalysis,
        common_orphans: dj.expression.QueryExpression,
        dry_run: bool,
        table_num: int,
        num_tables: int,
    ) -> dj.expression.QueryExpression:
        """Clean up a single custom analysis table.

        Parameters
        ----------
        analysis_tbl : SpyglassAnalysis
            The custom analysis table to clean up.
        common_orphans : dj.expression.QueryExpression
            The common orphans to update with valid entries.
        dry_run : bool
            If True, only report what would be deleted.
        table_num : int
            Current table number for logging.
        num_tables : int
            Total number of tables for logging.

        Returns
        -------
        dj.expression.QueryExpression
            Updated common orphans with valid entries removed.
        """
        prefix = analysis_tbl.database.split("_")[0]

        # Delete orphans from this analysis table
        orphans = analysis_tbl.delete_orphans(dry_run=dry_run, safemode=False)
        n_orphans = len(orphans) if orphans is not None else 0

        # Clean up this table's external entries
        unused = analysis_tbl.cleanup_external(
            dry_run=dry_run, delete_external_files=True
        )
        self._info_msg(
            f"  [{table_num}/{num_tables}] {prefix}: {n_orphans} orphans, "
            + f"{len(unused)} unused externals"
        )

        # Remove valid entries from common orphans
        if bool(analysis_tbl):
            common_orphans -= analysis_tbl.proj()

        return common_orphans

    def cleanup(
        self,
        dry_run: bool = False,
        *,
        max_delete_fraction: float = 0.9,
        max_delete_to_tracked_ratio: float = 10.0,
        min_file_age_hours: float = 24.0,
    ) -> None:
        """Clean up common and all custom AnalysisNwbfile tables.

        Removes orphaned analysis files across both common and custom tables.
        A file is considered orphaned if it has no downstream foreign key
        references. This method coordinates cleanup across all registered
        custom AnalysisNwbfile tables to prevent premature deletion.

        Process:
            1. Discover custom tables and snapshot filesystem/tracking state.
            2. Validate the complete untracked-file deletion plan.
            3. Re-check tracking, age, alias authority, and target identity;
               then delete eligible filesystem candidates.
            4. For each custom analysis table:
               a. Delete orphaned entries (no downstream references)
               b. Clean up unused external file entries
               c. Remove valid entries from common orphan list
            5. Delete remaining common orphans.
            6. Clean up common external entries without deleting their files.

        Example:
            from spyglass.common import AnalysisNwbfile

            # Run cleanup across all tables
            AnalysisNwbfile().cleanup(dry_run=False)

        Note:
            This is a destructive operation. Ensure you have backups before
            running cleanup on production databases. File deletions cannot
            be undone.

        See Also:
            docs/src/ForDevelopers/Management.md for detailed cleanup guide.

        Parameters
        ----------
        dry_run : bool
            If True, perform a non-destructive dry run: log and report all
            cleanup actions without deleting database entries or files.
            If False, apply the cleanup changes, including deleting orphaned
            entries and associated files.
        max_delete_fraction : float
            Maximum fraction of eligible analysis NWB files that may be
            deleted by filesystem cleanup. The eligible set is the planned
            deletions plus scanned files recognized as tracked; age-deferred
            files are excluded. Set high by default (0.9) so it catches a
            catastrophically misconfigured analysis directory rather than
            routine large cleanups. Defaults to 0.9.
        max_delete_to_tracked_ratio : float
            Maximum ratio of filesystem cleanup deletions to tracked analysis
            files found in the scan. At the default ``max_delete_fraction``
            this limit cannot bind: both guards divide by the same tracked
            count T, and ``delete / (delete + T) <= 0.9`` forces
            ``delete / T <= 9``, so any plan that clears the fraction limit is
            already within the ratio limit. It becomes the operative guard
            only when ``max_delete_fraction`` is raised above 10/11 (~0.909).
            This limit applies only to
            filesystem deletion of untracked or empty analysis NWB files, not
            to orphan row deletion. Defaults to 10.0.
        min_file_age_hours : float
            Untracked files newer than this are deferred to the next cleanup
            rather than deleted, protecting work that exists on disk but is
            not yet registered -- notably a file written to another volume
            and symlinked in before its row is inserted. Defaults to 24.0.
            Pass 0 only for intentional immediate cleanup.

        Raises
        ------
        ValueError
            If a numeric safety limit is non-finite or outside its bounds.
        RuntimeError
            If insert blocking or unblocking fails, a destructive plan is
            refused or malformed, or a filesystem deletion fails. Registry,
            external-table, and filesystem inspection errors also propagate
            so destructive cleanup fails closed.
        """
        # Validate before anything happens. An unvalidated NaN or inf would
        # make every comparison False and silently disable the guard.
        max_delete_fraction = _check_number(
            "max_delete_fraction", max_delete_fraction, minimum=0, maximum=1
        )
        max_delete_to_tracked_ratio = _check_number(
            "max_delete_to_tracked_ratio",
            max_delete_to_tracked_ratio,
            minimum=0,
        )
        min_file_age_hours = _check_number(
            "min_file_age_hours", min_file_age_hours, minimum=0
        )

        heading = "============== Analysis Cleanup "
        suffix = "(Dry Run) ==============" if dry_run else "=============="
        self._info_msg(heading + suffix)

        registry = AnalysisRegistry()
        # Stays OUTSIDE the try. Moving it inside would let a partial
        # acquisition fall into `finally: unblock_new_inserts()`, which
        # drops EVERY trigger including ones owned by a concurrent run.
        # Full ownership tracking needs a cleanup lease (follow-up).
        registry.block_new_inserts(dry_run=dry_run)

        # An explicit flag, not sys.exc_info(): that returns the exception
        # being handled ANYWHERE up the calling stack, so a caller shaped
        # `except Exception: cleanup()` would make it non-None even when the
        # body succeeded -- silently downgrading an unblock failure that
        # leaves inserts blocked database-wide.
        body_failed = False
        try:
            # Inside the try: a throw from get_orphans() previously landed
            # between block and try, leaving insert triggers installed
            # database-wide with no unblock.
            custom_tables = list(registry.all_classes)
            num_tables = len(custom_tables) + 1  # +1 for common table
            common_orphans = self.get_orphans().proj()

            untracked_file_plan = self._build_untracked_file_plan(
                custom_tables, min_file_age_hours=min_file_age_hours
            )
            plan_ok, plan_err = self._validate_cleanup_plan(
                untracked_file_plan,
                max_delete_fraction=max_delete_fraction,
                max_delete_to_tracked_ratio=max_delete_to_tracked_ratio,
            )
            if not plan_ok:
                # Dry-run previews must surface refusal; real runs must abort.
                if dry_run:
                    logger.warning(f"Cleanup plan would be refused: {plan_err}")
                else:
                    raise RuntimeError(plan_err)

            # Delete files BEFORE database cleanup, shortening the
            # validate-to-act window for the filesystem sweep. Files newly
            # orphaned by this run's row deletion are caught on the next
            # invocation. Note this deferral applies only to the untracked
            # sweep: custom-table externals are still deleted below.
            _ = self._remove_untracked_files(
                custom_tables,
                dry_run=dry_run,
                plan=untracked_file_plan,
                min_file_age_hours=min_file_age_hours,
            )

            # Process each custom analysis table.
            # Subtract valid entries from common_orphans
            for i, analysis_tbl in enumerate(custom_tables, start=1):
                common_orphans = self._cleanup_custom_table(
                    analysis_tbl, common_orphans, dry_run, i, num_tables
                )

            # Delete remaining common orphans
            n_orphans = len(common_orphans)

            if bool(common_orphans) and not dry_run:
                common_orphans.delete_quick()

            # Clean up common external table entries
            unused = self.cleanup_external(
                dry_run=dry_run, delete_external_files=False
            )

            self._info_msg(
                f"  [{num_tables}/{num_tables}] common: {n_orphans} "
                f"orphans, {len(unused)} unused externals"
            )

        except BaseException:
            body_failed = True
            raise

        finally:
            if not dry_run:
                try:
                    registry.unblock_new_inserts()
                except Exception as unblock_err:
                    # A failed unblock halts ALL inserts across the database
                    # until manually cleared, so this must be loud regardless
                    # of whether another exception is already propagating.
                    logger.critical(
                        "Failed to unblock inserts after cleanup: "
                        f"{unblock_err}. Analysis inserts remain BLOCKED "
                        "database-wide until restored; run "
                        "AnalysisRegistry().unblock_new_inserts() manually."
                    )
                    # Re-raise only when the body itself succeeded;
                    # otherwise we would mask the original cleanup error
                    # (the critical log above is the signal).
                    if not body_failed:
                        raise

    def check_all_files(
        self, resolve_tables: bool = False, verbose: bool = False
    ) -> dict:
        """Check files across all analysis tables for issues.

        Iterates through common and all custom AnalysisNwbfile tables,
        checking file existence and readability. Populates AnalysisFileIssues
        with any problems found. This monitoring operation does not delete
        files, but it does write issue rows and can be run independently of
        cleanup at different frequencies.

        Parameters
        ----------
        resolve_tables : bool, optional
            After all issues are collected, populate the table field for each
            issue by querying downstream child tables. More efficient than
            per-table resolution since children are fetched once per analysis
            table across all newly inserted issues. Default False.

        Returns
        -------
        results : dict
            Dictionary mapping table names to issue counts

        Example
        -------
        >>> from spyglass.common import AnalysisNwbfile
        >>> results = AnalysisNwbfile().check_all_files(resolve_tables=True)
        >>> print(f"Total issues: {sum(results.values())}")

        See Also
        --------
        AnalysisFileIssues : Table that stores detected issues
        AnalysisFileIssues.resolve_table_refs : Populate table field on demand
        """
        from spyglass.common.common_file_tracking import AnalysisFileIssues

        self._info_msg("Checking analysis files across all tables")
        registry = AnalysisRegistry()

        # Include common table + all custom tables
        analysis_tables = [self] + list(registry.all_classes)
        num_tables = len(analysis_tables)

        results = {}
        file_checker = AnalysisFileIssues()

        # B: Fetch recompute-deleted files once for all tables
        deleted_files = file_checker._get_recompute_deleted()

        for i, analysis_tbl in enumerate(analysis_tables, start=1):
            tbl_name = analysis_tbl.full_table_name
            self._info_msg(f"  [{i}/{num_tables}] Checking {tbl_name} files")

            issue_count = file_checker.check_files(
                analysis_tbl, deleted_files=deleted_files, verbose=verbose
            )
            results[tbl_name] = issue_count

            if issue_count > 0:
                logger.warning(f"    Found {issue_count} file issues")

        total_issues = sum(results.values())
        self._info_msg(f"File check complete: {total_issues} issues found")

        if resolve_tables and total_issues > 0:
            self._info_msg("Resolving downstream table references for issues")
            file_checker.resolve_table_refs()

        return results

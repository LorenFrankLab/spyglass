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
from spyglass.common._nwbfile_cleanup import (
    AccessSnapshot,
    CleanupCandidate,
    CleanupPlan,
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
            An error message if blocking fails, otherwise None.
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
                # Stop immediately. During concurrent acquisition over a
                # stable registry, the runs collide on the same earliest
                # trigger instead of continuing into disjoint prefixes.
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
            ``(real_path, target_or_None, access_snapshot)``, or None when
            the entry vanished mid-scan and should be skipped.

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
            # Per-ENTRY errors skip; only WALK errors are fatal. An entry
            # that cannot be stat'd never becomes a candidate, so skipping
            # can only under-clean, never over-delete -- whereas aborting
            # lets one bad symlink (a loop, a 0700 parent) wedge the weekly
            # sweep, and with it every later maintenance phase.
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

    @staticmethod
    def _is_old_enough(
        candidate: CleanupCandidate,
        *,
        now_ns: int,
        min_file_age_hours: float,
    ) -> bool:
        """Return True when every part of a candidate is old enough.

        Exactly ``min_file_age_hours`` old is eligible. A future timestamp
        yields a negative age and is therefore deferred.
        """
        if min_file_age_hours <= 0:
            return True
        threshold_ns = int(min_file_age_hours * 3600 * 10**9)
        return (now_ns - candidate.newest_ns) >= threshold_ns

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
        now_ns = time.time_ns() if now_ns is None else now_ns

        tracked_state = self._current_tracked_state(custom_tables)
        tracked = set(tracked_state.resolved_paths)

        targets: Dict[Path, Optional[TargetSnapshot]] = {}
        accesses: Dict[Path, List[AccessSnapshot]] = {}
        for access_path in tqdm(
            self._walk_analysis_files(),
            desc="Scanning analysis files  ",  # extra spaces for alignment
        ):
            snapped = self._snapshot_entry(access_path)
            if snapped is None:
                continue
            real_path, target, access = snapped
            accesses.setdefault(real_path, []).append(access)
            if target is not None:
                targets[real_path] = target
            else:
                targets.setdefault(real_path, None)

        scanned = set(accesses)
        candidates: Dict[Path, CleanupCandidate] = {}
        empty: Set[Path] = set()
        untracked: Set[Path] = set()
        broken: Set[Path] = set()
        deferred: Set[Path] = set()

        for real_path, target in targets.items():
            candidate = CleanupCandidate(
                real_path=real_path,
                target=target,
                accesses=tuple(accesses[real_path]),
            )
            if tracked_state.matches(candidate):
                # CleanupPlan's safety denominator intersects scanned and
                # tracked PATHS. Preserve that contract when tracking matched
                # through filesystem identity rather than path spelling.
                tracked.add(real_path)
                continue  # tracked wins over empty and over broken
            if target is not None and not target.is_regular:
                logger.warning(
                    f"Skipping non-regular analysis path: {real_path}"
                )
                continue
            if not self._is_old_enough(
                candidate,
                now_ns=now_ns,
                min_file_age_hours=min_file_age_hours,
            ):
                deferred.add(real_path)
                continue
            candidates[real_path] = candidate
            if candidate.broken:
                broken.add(real_path)
            elif target.size == 0:
                empty.add(real_path)
            else:
                untracked.add(real_path)

        return CleanupPlan(
            scanned_files=scanned,
            tracked_files=tracked,
            files_to_delete=set(candidates),
            empty_files=empty,
            untracked_files=untracked,
            candidates=candidates,
            deferred_recent_files=deferred,
            broken_links=broken,
        )

    @staticmethod
    def _validate_cleanup_plan(
        plan: CleanupPlan,
        *,
        max_delete_fraction: float = 0.9,
        max_delete_to_tracked_ratio: float = 10.0,
    ) -> tuple[bool, str | None]:
        """Check a cleanup plan against destructive-cleanup safety limits.

        Returns
        -------
        tuple[bool, str | None]
            ``(True, None)`` when the plan is safe to apply, otherwise
            ``(False, reason)`` where ``reason`` explains the refusal. Callers
            decide whether to raise (real run) or warn (dry run).
        """
        scanned_count = len(plan.scanned_files)
        delete_count = len(plan.files_to_delete)

        if delete_count == 0:
            return True, None

        # Denominator is what this sweep was entitled to act on: the
        # deletions plus the scanned files it recognized as tracked.
        # Age-deferred files are excluded -- including them would let a plan
        # that deletes 89 of 90 eligible files read as 89%.
        local_tracked = plan.scanned_files & plan.tracked_files
        tracked_count = len(local_tracked)
        eligible_count = delete_count + tracked_count

        if tracked_count == 0:
            return False, (
                "Analysis cleanup would delete "
                f"{delete_count} files after scanning {scanned_count} files, "
                "but no tracked analysis files were found. Refusing "
                "destructive cleanup; run with dry_run=True and verify the "
                "configured analysis directory."
            )

        delete_fraction = delete_count / max(eligible_count, 1)
        if delete_fraction > max_delete_fraction:
            return False, (
                "Analysis cleanup would delete "
                f"{delete_count}/{eligible_count} eligible analysis files "
                f"({delete_fraction:.1%}), above the safety limit "
                f"{max_delete_fraction:.1%}. Refusing destructive cleanup; "
                "run with dry_run=True and verify the cleanup plan."
            )

        delete_ratio = delete_count / tracked_count
        if delete_ratio > max_delete_to_tracked_ratio:
            return False, (
                "Analysis cleanup would delete "
                f"{delete_count} files with only {tracked_count} tracked "
                f"analysis files ({delete_ratio:.1f}x), above the safety "
                f"limit {max_delete_to_tracked_ratio:.1f}x. Refusing "
                "destructive cleanup; run with dry_run=True and verify the "
                "configured analysis directory."
            )

        return True, None

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
    def _containing_managed_root(
        path: Path, roots: List[_PhysicalRoot]
    ) -> Optional[_PhysicalRoot]:
        """Return the protected root physically containing ``path``.

        Lexical containment is the common fast path. When spellings differ,
        walk the target's parents once and compare each directory identity
        with all configured roots. This recognizes case aliases and alternate
        names of a configured root itself without confusing an outside hard
        link to a protected file with a path inside the store.
        """
        if not roots:
            return None

        for root in roots:
            if path.is_relative_to(root.path):
                return root

        roots_by_identity = {(root.dev, root.ino): root for root in roots}
        current = path.parent
        while True:
            try:
                current_stat = os.stat(current)
            except OSError as err:
                raise RuntimeError(
                    "Cannot inspect ancestry of analysis target "
                    f"{path}; refusing analysis cleanup"
                ) from err
            if root := roots_by_identity.get(
                (current_stat.st_dev, current_stat.st_ino)
            ):
                return root
            parent = current.parent
            if parent == current:
                return None
            current = parent

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

    def _candidate_still_matches(
        self,
        candidate: CleanupCandidate,
        *,
        analysis_root: Path,
        managed_roots: List[_PhysicalRoot],
    ) -> Optional[List[AccessSnapshot]]:
        """Re-verify a candidate during final pre-delete validation.

        Returns
        -------
        list of AccessSnapshot or None
            The in-root accesses that may be unlinked, or None (with a
            warning) when anything changed since the scan, when a
            non-regular target is involved, when the target has no in-root
            access authorizing it, or when the target belongs to another
            protected Spyglass store.

        Notes
        -----
        The returned list is what the caller unlinks. Accesses outside the
        analysis root are never returned, so an extra out-of-root access on
        an otherwise valid candidate cannot be removed.
        """
        real_path = candidate.real_path

        def _refuse(reason: str):
            logger.warning(f"Skipping {real_path}: {reason}")
            return None

        # An in-root *.nwb access IS the deletion authority: a target is
        # deletable on any volume, so at least one access must live inside
        # analysis_root. Validate both the scanned leaf identity and its live
        # canonical destination while every alias in a possible chain still
        # exists. The later leaf unlink check cannot resolve the destination
        # because the target has intentionally been removed by then.
        in_root = []
        for access in candidate.accesses:
            try:
                lst = os.lstat(access.access_path)
            except OSError:
                return _refuse(f"access path {access.access_path} unreadable")
            if stat_module.S_ISLNK(lst.st_mode) != access.is_link:
                return _refuse(f"{access.access_path} changed link type")
            if (lst.st_dev, lst.st_ino) != (access.dev, access.ino):
                return _refuse(f"{access.access_path} identity changed")
            # Alias timestamps are part of identity. Without this,
            # `os.utime(link, follow_symlinks=False)` makes a link fresh
            # without changing dev/ino, so the age gate -- which reads the
            # scan-time snapshot -- would still authorize deletion.
            if (lst.st_mtime_ns, lst.st_ctime_ns) != (
                access.mtime_ns,
                access.ctime_ns,
            ):
                return _refuse(
                    f"{access.access_path} timestamps changed since the scan"
                )
            # Resolve the PARENT, not the link itself -- resolving the link
            # would follow it to the target.
            try:
                location = (
                    access.access_path.parent.resolve()
                    / access.access_path.name
                )
            except (OSError, RuntimeError):
                return _refuse(
                    f"{access.access_path} parent could not be resolved"
                )
            if not location.is_relative_to(analysis_root):
                continue
            if access.is_link:
                try:
                    raw = os.readlink(access.access_path)
                except OSError:
                    return _refuse(f"{access.access_path} readlink failed")
                if raw != access.raw_link_target:
                    return _refuse(f"{access.access_path} was re-pointed")
            try:
                live_target = Path(os.path.realpath(access.access_path))
            except (OSError, RuntimeError):
                return _refuse(f"{access.access_path} could not be resolved")
            if live_target != real_path:
                return _refuse(
                    f"{access.access_path} no longer resolves to the "
                    "planned target"
                )
            in_root.append(access)

        if not in_root:
            return _refuse("no access path inside the analysis directory")

        # A target belonging to another Spyglass store is never ours to
        # delete, however it was reached. `tracked` only knows the analysis
        # store, so without this a symlink into the raw store would make a
        # non-recomputable acquisition file look untracked.
        managed = None
        if not candidate.broken:
            managed = self._containing_managed_root(real_path, managed_roots)
        if managed is not None:
            return _refuse(
                "target belongs to another Spyglass store "
                f"({managed.name}: {managed.path}); analysis cleanup does "
                "not delete from it"
            )

        # Target identity is verified for EVERY live target, in-root or not,
        # because every live target may be deleted. The access checks above
        # separately prove that a current in-root *.nwb path still resolves
        # to this exact target, including through intermediate symlinks.
        if not candidate.broken:
            try:
                st = os.stat(real_path)
                lst_target = os.lstat(real_path)
            except OSError:
                return _refuse("target unreadable")
            if not stat_module.S_ISREG(lst_target.st_mode):
                return _refuse("target is not a regular file")
            if (st.st_dev, st.st_ino) != (
                lst_target.st_dev,
                lst_target.st_ino,
            ):
                return _refuse("target stat/lstat disagree")
            snap = candidate.target
            if (st.st_dev, st.st_ino) != (snap.dev, snap.ino):
                return _refuse("target identity changed since the scan")
            if st.st_size != snap.size:
                return _refuse("target size changed since the scan")
            if (st.st_mtime_ns, st.st_ctime_ns) != (
                snap.mtime_ns,
                snap.ctime_ns,
            ):
                return _refuse("target timestamps changed since the scan")
            if st.st_mode != snap.mode:
                return _refuse("target mode changed since the scan")
        else:
            try:
                os.stat(real_path)
            except FileNotFoundError:
                pass  # still broken, as planned
            except OSError:
                return _refuse("broken-link target unreadable")
            else:
                return _refuse("broken link now resolves")

        return in_root

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

        Returns
        -------
        tuple[Set[Path], Set[Path]]
            ``(candidate_target_paths, tracked_files)``. In dry-run mode the
            first set describes target candidates, not every leaf path that a
            real run may unlink.
        """

        if plan is None:
            plan = self._build_untracked_file_plan(
                custom_tables,
                min_file_age_hours=min_file_age_hours,
                now_ns=now_ns,
            )

        if plan.deferred_recent_files:
            logger.info(
                f"  {len(plan.deferred_recent_files)} untracked files "
                f"deferred because they are newer than {min_file_age_hours} "
                "hours. Use min_file_age_hours=0 only for intentional "
                "immediate cleanup."
            )

        if dry_run:
            logger.info(
                f"  {len(plan.files_to_delete)} untracked or empty analysis "
                f"files ({len(plan.untracked_files)} untracked, "
                f"{len(plan.empty_files)} empty, "
                f"{len(plan.broken_links)} broken links)"
            )
            return plan.files_to_delete, plan.tracked_files

        analysis_root = Path(self._analysis_dir).expanduser().resolve()
        act_now_ns = time.time_ns() if now_ns is None else now_ns
        failures = []

        # ---- PREFLIGHT: validate the WHOLE plan before any unlink ----
        # A structurally malformed plan must not be partially executed:
        # collecting per-entry problems while still deleting the valid
        # entries would let a forged plan do real damage before it failed.
        if set(plan.candidates) != plan.files_to_delete:
            raise RuntimeError(
                "Analysis file deletion failed: cleanup plan is "
                "inconsistent, candidate keys do not match files_to_delete. "
                "Refusing to act on a malformed plan."
            )
        structural = []
        for key_path, candidate in plan.candidates.items():
            if key_path != candidate.real_path:
                structural.append(
                    f"plan key {key_path} does not match candidate "
                    f"{candidate.real_path}"
                )
            if (
                candidate.target is not None
                and candidate.target.real_path != candidate.real_path
            ):
                structural.append(
                    f"{candidate.real_path}: target snapshot names a "
                    f"different path {candidate.target.real_path}"
                )
            # The scanner only ever yields *.nwb entries, so any other
            # suffix means the plan did not come from it. Enforcing the
            # invariant here stops a forged plan using, say,
            # analysis/voucher.txt -> /outside/victim.nwb as authority.
            for access in candidate.accesses:
                # Exactly the scanner's predicate, case-sensitive: a
                # `voucher.NWB` cannot come from the scan, so accepting it
                # here would let a forged plan authorize deletion.
                if not access.access_path.name.endswith(".nwb"):
                    structural.append(
                        f"{candidate.real_path}: access path "
                        f"{access.access_path} is not a *.nwb entry"
                    )
        if structural:
            raise RuntimeError(
                "Analysis file deletion failed: cleanup plan is malformed; "
                "refusing to delete anything:\n  " + "\n  ".join(structural)
            )

        # Snapshot every protected root before the first unlink. A missing or
        # unreadable configured store must disable the whole destructive pass,
        # not fail only after earlier candidates have already been removed.
        managed_roots = self._other_managed_roots() if plan.candidates else []

        # ---- ACT ----
        # Accumulator, not the original snapshot: a table can appear in the
        # registry while one candidate is processed and be gone before the
        # next. Re-unioning against `custom_tables` each time would forget
        # it and delete its tracked files, defeating the "never dropped"
        # guarantee _current_custom_tables provides within a single call.
        known_tables = list(custom_tables)
        deleted_external = []
        for candidate in plan.candidates.values():
            # Tracking and registry membership are re-read for EVERY
            # candidate, not once up front: a file can be registered through
            # a new alias, or a whole custom table declared, while an earlier
            # candidate is being deleted. Resolving every current external
            # path and comparing filesystem identities is intentionally
            # conservative; an exact query on scan-time names cannot discover
            # a new alias to this target. The deferred cleanup/writer lease
            # would let this be hoisted or indexed safely.
            known_tables = self._current_custom_tables(known_tables)
            if self._current_tracked_state(known_tables).matches(candidate):
                logger.warning(
                    f"Skipping {candidate.real_path}: became tracked since "
                    "the scan"
                )
                continue
            # Age is re-checked at act time against the frozen scan-time
            # snapshot, so with a later clock it can only read older: this
            # cannot newly defer a candidate that already passed planning, but
            # it does honor a min_file_age_hours raised between building a
            # plan and applying it. A file whose bytes actually changed during
            # a long scan is caught instead by _candidate_still_matches, which
            # refuses any candidate whose target or alias timestamps moved.
            if not self._is_old_enough(
                candidate,
                now_ns=act_now_ns,
                min_file_age_hours=min_file_age_hours,
            ):
                logger.warning(
                    f"Skipping {candidate.real_path}: newer than "
                    f"{min_file_age_hours}h at deletion time"
                )
                continue
            in_root = self._candidate_still_matches(
                candidate,
                analysis_root=analysis_root,
                managed_roots=managed_roots,
            )
            if not in_root:
                continue
            # Validation and unlink cannot be one atomic filesystem operation.
            # The caller must exclude concurrent writers from eligible paths;
            # a shared cleanup/writer lease is the tracked follow-up.
            try:
                if not candidate.broken:
                    is_external = not candidate.real_path.is_relative_to(
                        analysis_root
                    )
                    candidate.real_path.unlink()
                    if is_external:
                        # Record only after unlink succeeds. Emit an immediate
                        # durable audit entry as well as the end-of-run summary:
                        # a later registry/DB failure must not erase evidence
                        # of an irreversible cross-volume deletion.
                        record = (candidate.real_path, candidate.target.size)
                        deleted_external.append(record)
                        logger.warning(
                            "Deleted external analysis target "
                            f"{record[0]} ({record[1]} bytes)"
                        )
                # Each leaf is re-checked immediately before its own
                # unlink, never batched: a link approved earlier in the
                # pass could be replaced by a regular file that a blind
                # unlink would then destroy. unlink() operates on the leaf
                # and never follows it, so removal order does not matter
                # even for chained aliases.
                for access in in_root:
                    if not access.is_link:
                        continue
                    if not self._access_still_matches(
                        access, analysis_root=analysis_root
                    ):
                        logger.warning(
                            f"Skipping link {access.access_path}: changed "
                            "since it was validated"
                        )
                        continue
                    access.access_path.unlink()
            except OSError as e:
                failures.append(f"{candidate.real_path}: {e}")

        if deleted_external:
            total = sum(size for _, size in deleted_external)
            roots = sorted({str(p.parent) for p, _ in deleted_external})
            logger.warning(
                f"  {len(deleted_external)} symlink targets deleted OUTSIDE "
                f"the analysis directory ({total} bytes reclaimed). An "
                "in-root *.nwb symlink authorizes deletion of the "
                "non-protected external target it resolved to during "
                "final validation. "
                f"Roots touched: {roots}"
            )
            for path, size in sorted(deleted_external):
                logger.info(f"    deleted external {path} ({size} bytes)")

        if failures:
            # Raise rather than log: cleanup() runs database cleanup after
            # this, and the maintenance driver gates later analysis-storage
            # phases on a failure escaping here. A logged-and-swallowed
            # error would let both proceed with storage in an unknown state.
            raise RuntimeError(
                f"Analysis file deletion failed for {len(failures)} "
                "candidates; refusing to continue with analysis storage in "
                "an unknown state:\n  " + "\n  ".join(failures)
            )

        return plan.files_to_delete, plan.tracked_files

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

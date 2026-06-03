import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

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
from spyglass.settings import analysis_dir, raw_dir
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


@dataclass(frozen=True)
class CleanupPlan:
    """Filesystem cleanup plan for analysis NWB files.

    Attributes
    ----------
    scanned_files : set of pathlib.Path
        Analysis ``*.nwb`` paths found under the configured analysis directory.
    tracked_files : set of pathlib.Path
        Analysis paths currently referenced by DataJoint external stores.
    files_to_delete : set of pathlib.Path
        Files selected for filesystem deletion.
    empty_files : set of pathlib.Path
        Empty (0-byte) analysis files selected for deletion.
    untracked_files : set of pathlib.Path
        Non-empty files selected because no external store references them.
    """

    scanned_files: Set[Path]
    tracked_files: Set[Path]
    files_to_delete: Set[Path]
    empty_files: Set[Path]
    untracked_files: Set[Path]


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

            _full_table_name = full_name
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
        if dry_run:
            logger.info(f"Dry run: would block inserts into {table}")
            return None

        try:
            database, trigger = self._get_block_info(table)
            kwargs = dict(database=database, trigger=trigger, table=table)

            if self._block_exists(table):
                return

            # Create trigger
            dj.conn().query(SQL_BLOCK_TEMPLATE.format(**kwargs))

        except Exception as e:
            return f"Failed to block {table}: {e}"

        return None

    def block_new_inserts(self, dry_run: bool = False) -> None:
        """Block new inserts into all registered analysis tables.

        Creates BEFORE INSERT triggers on all registered custom analysis tables
        to prevent data modifications during maintenance operations.

        Parameters
        ----------
        dry_run : bool, optional
            If True, log blocking without making changes. Defaults to False.

        Raises
        ------
        RuntimeError
            If any trigger creation fails.
        """
        errors = []
        for table in self.fetch("full_table_name"):
            error = self._block_single_table(table, dry_run=dry_run)
            if error is not None:
                errors.append(error)

        if errors:
            raise RuntimeError(
                f"Failed to block {len(errors)} table(s):\n" + "\n".join(errors)
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

    def _build_untracked_file_plan(
        self, custom_tables: List[SpyglassAnalysis]
    ) -> CleanupPlan:
        """Build a cleanup plan for untracked or empty analysis NWB files."""

        def paths_from_external(tbl) -> Set[Path]:
            return {
                Path(fp[1]).expanduser().resolve()
                for fp in tbl._ext_tbl.fetch_external_paths()
            }

        tracked = paths_from_external(self)
        for tbl in custom_tables:
            tracked.update(paths_from_external(tbl))

        scanned = set()
        empty = set()
        untracked = set()
        for path in tqdm(
            Path(self._analysis_dir).rglob("*.nwb"),
            desc="Scanning analysis files  ",  # Note extra spaces for alignment
        ):
            # rglob("*.nwb") only yields files; skip symlinks so a stray
            # symlink under analysis_dir can't add its target (potentially
            # outside analysis_dir) to the deletion plan.
            if path.is_symlink():
                try:
                    target = os.readlink(path)
                except OSError:
                    target = "<unreadable>"
                logger.warning(
                    f"Skipping symlink in analysis dir: {path} -> {target}"
                )
                continue
            resolved_path = path.expanduser().resolve()
            scanned.add(resolved_path)
            if path.stat().st_size == 0:
                empty.add(resolved_path)
            elif resolved_path not in tracked:
                untracked.add(resolved_path)

        return CleanupPlan(
            scanned_files=scanned,
            tracked_files=tracked,
            files_to_delete=empty | untracked,
            empty_files=empty,
            untracked_files=untracked,
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
        tracked_count = len(plan.tracked_files)
        delete_count = len(plan.files_to_delete)

        if delete_count == 0:
            return True, None

        if tracked_count == 0:
            return False, (
                "Analysis cleanup would delete "
                f"{delete_count} files after scanning {scanned_count} files, "
                "but no tracked analysis files were found. Refusing "
                "destructive cleanup; run with dry_run=True and verify the "
                "configured analysis directory."
            )

        delete_fraction = delete_count / max(scanned_count, 1)
        if delete_fraction > max_delete_fraction:
            return False, (
                "Analysis cleanup would delete "
                f"{delete_count}/{scanned_count} scanned analysis files "
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

    def _remove_untracked_files(
        self,
        custom_tables: List[SpyglassAnalysis],
        dry_run: bool = True,
        plan: CleanupPlan | None = None,
    ) -> tuple[Set[Path], Set[Path]]:
        """Remove analysis files that are empty (0 bytes) or not tracked.

        WARNING: This function makes `analysis_dir` a privileged directory and
        will delete files in it that are not tracked in ANY schema externals.

        NOTE: Subprocess would be faster, but this prioritizes cross-platform
        compatibility.

        Parameters
        ----------
        dry_run : bool, optional
            If True, return the files that would be deleted. Defaults to True.
        custom_tables : list
            List of custom analysis table instances to check for tracked files.
        plan : CleanupPlan, optional
            Precomputed cleanup plan. If omitted, the directory is scanned.

        Returns
        -------
        tuple[Set[Path], Set[Path]]
            (files_to_delete, tracked_files)
        """

        plan = plan or self._build_untracked_file_plan(custom_tables)

        if dry_run:
            logger.info(
                f"  {len(plan.files_to_delete)} untracked or empty analysis "
                f"files ({len(plan.untracked_files)} untracked, "
                f"{len(plan.empty_files)} empty)"
            )
            return plan.files_to_delete, plan.tracked_files

        # files_to_delete holds resolved paths, and Path.resolve() follows
        # symlinks — a symlink inside analysis_dir resolves to its target.
        # Skip any path that resolves outside analysis_dir so cleanup cannot
        # delete data elsewhere via a symlink placed in the analysis directory.
        analysis_root = Path(self._analysis_dir).expanduser().resolve()
        for path in plan.files_to_delete:
            if not path.is_relative_to(analysis_root):
                logger.warning(
                    f"Skipping deletion outside analysis dir: {path}"
                )
                continue
            try:
                path.unlink()
            except OSError as e:
                self._logger.error(f"Error deleting file {path}: {e}")

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
        max_delete_fraction: float = 0.9,
        max_delete_to_tracked_ratio: float = 10.0,
    ) -> None:
        """Clean up common and all custom AnalysisNwbfile tables.

        Removes orphaned analysis files across both common and custom tables.
        A file is considered orphaned if it has no downstream foreign key
        references. This method coordinates cleanup across all registered
        custom AnalysisNwbfile tables to prevent premature deletion.

        Process:
            1. For each custom analysis table:
               a. Delete orphaned entries (no downstream references)
               b. Clean up unused external file entries
               c. Remove valid entries from common orphan list
            2. Delete remaining common orphans
            3. Clean up common external entries
            4. Delete empty files (0 bytes)

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
            Maximum fraction of scanned analysis NWB files that may be deleted
            by filesystem cleanup. Set high by default (0.9) so it only
            catches a catastrophically misconfigured analysis directory (one
            where the sweep would wipe nearly everything), not routine large
            cleanups. Defaults to 0.9.
        max_delete_to_tracked_ratio : float
            Maximum ratio of filesystem cleanup deletions to tracked analysis
            files. Set high by default (10.0) so it only flags scans where
            untracked-file count dwarfs tracked-file count by an order of
            magnitude — a strong signal that the analysis directory is mixed
            with non-analysis data or a wrong path was supplied. This limit
            applies only to filesystem deletion of untracked or empty analysis
            NWB files, not to orphan row deletion. Defaults to 10.0.
        """
        heading = "============== Analysis Cleanup "
        suffix = "(Dry Run) ==============" if dry_run else "=============="
        self._info_msg(heading + suffix)

        registry = AnalysisRegistry()
        registry.block_new_inserts(dry_run=dry_run)

        # Get all custom tables first so we can check their tracked files
        custom_tables = list(registry.all_classes)
        num_tables = len(custom_tables) + 1  # +1 for common table
        common_orphans = self.get_orphans().proj()

        try:
            untracked_file_plan = self._build_untracked_file_plan(custom_tables)
            plan_ok, plan_err = self._validate_cleanup_plan(
                untracked_file_plan,
                max_delete_fraction=max_delete_fraction,
                max_delete_to_tracked_ratio=max_delete_to_tracked_ratio,
            )
            if not plan_ok:
                # Dry-run previews must surface refusal; real runs must abort.
                if dry_run:
                    self._logger.warning(
                        f"Cleanup plan would be refused: {plan_err}"
                    )
                else:
                    raise RuntimeError(plan_err)

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

            # Reuse the pre-pass plan rather than rescanning: rescanning here
            # would bypass the validate-before-act guard already applied to
            # this plan. Files newly orphaned by this run's orphan deletion
            # are caught on the next cleanup invocation.
            _ = self._remove_untracked_files(
                custom_tables, dry_run=dry_run, plan=untracked_file_plan
            )

        finally:
            if not dry_run:
                # Capture the outer try-block exception (if any) BEFORE the
                # inner try: sys.exc_info() inside the inner except returns
                # the inner exception, not the outer one. We only want to
                # re-raise an unblock failure when no other exception is
                # already propagating from the cleanup body.
                cleanup_exc = sys.exc_info()[1]
                try:
                    registry.unblock_new_inserts()
                except Exception as unblock_err:
                    self._logger.error(
                        f"Failed to unblock inserts after cleanup: "
                        f"{unblock_err}"
                    )
                    if cleanup_exc is None:
                        raise

    def check_all_files(self, resolve_tables: bool = False) -> dict:
        """Check files across all analysis tables for issues.

        Iterates through common and all custom AnalysisNwbfile tables,
        checking file existence and readability. Populates AnalysisFileIssues
        table with any problems found. This is a read-only monitoring operation
        that can be run independently of cleanup at different frequencies.

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
                analysis_tbl, deleted_files=deleted_files
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

import os
import random
import re
import string
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Set, Union
from uuid import uuid4

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
        get_all_classes() - Get all registered AnalysisNwbfile classes
        get_tracked_files() - Get all files tracked across all custom tables

    Usage:
        from spyglass.common import AnalysisRegistry

        # View all registered tables
        AnalysisRegistry().fetch()

        # Get a specific team's table
        MyTeamAnalysis = AnalysisRegistry().get_class("myteam")
    """

    definition = """
    full_table_name: varchar(128)  # full table name of the analysis
    ---
    created_at = CURRENT_TIMESTAMP: timestamp  # when registered
    created_by : varchar(32)                   # who registered
    """

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

        if self & dict(full_table_name=key["full_table_name"]):
            logger.debug(f"Entry already exists: {key['full_table_name']}")
            return

        if "created_by" not in key:
            key["created_by"] = dj.config["database.user"]

        super().insert1(key, **kwargs)

    # ---------------- Spawn analysis classes from table names ----------------

    @staticmethod
    def _parse_table_name(
        full_table_name: str,
    ) -> tuple[str, str, str, str]:
        """Parse full table name into components.

        Extracts database, table name, prefix, and suffix from a full table name.

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

    def _get_tbl_from_name(self, full_name: str) -> SpyglassAnalysis:
        """Return the DataJoint table object for the given full_table_name.

        Parameters
        ----------
        full_name : str
            The full table name.

        Returns
        -------
        table_obj : dj.FreeTable
            The DataJoint table object for the given full_table_name.
        """
        camel_name = dj.utils.to_camel_case(full_name.split(".")[-1])

        return type(
            camel_name,
            (SpyglassAnalysis, dj.FreeTable),
            {
                "__init__": lambda self: dj.FreeTable.__init__(
                    self, dj.conn(), full_name
                )
            },
        )

    def get_class(self, key: Union[str, Dict]) -> Optional[type]:
        """Return the class object for the given full_table_name, uninitialized.

        Parameters
        ----------
        key : str or dict
            The full_table_name as a string or a dict with the key
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

    def _block_single_table(self, table: str) -> Optional[str]:
        """Block new inserts into a single analysis table.

        Parameters
        ----------
        table : str
            The full table name of the analysis table to block.

        Returns
        -------
        error_msg : str or None
            An error message if blocking fails, otherwise None.
        """
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

    def block_new_inserts(self) -> None:
        """Block new inserts into all registered analysis tables.

        Creates BEFORE INSERT triggers on all registered custom analysis tables
        to prevent data modifications during maintenance operations.

        Raises
        ------
        RuntimeError
            If any trigger creation fails.
        """
        errors = []
        for table in self.fetch("full_table_name"):
            error = self._block_single_table(table)
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

    def _remove_untracked_files(
        self, custom_tables: List[SpyglassAnalysis], dry_run: bool = True
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
            If None, only checks common table. Defaults to None.

        Returns
        -------
        tuple[Set[Path], Set[Path]]
            (files_to_delete, all_files_scanned) - The second set can be reused
            to avoid re-scanning the directory.
        """

        def paths_from_external(tbl) -> Set[Path]:
            return set([fp[1] for fp in tbl._ext_tbl.fetch_external_paths()])

        # Collect tracked files from common table, then custom tables
        tracked = paths_from_external(self)
        for tbl in custom_tables:
            tracked.update(paths_from_external(tbl))

        to_delete = set()
        for path in tqdm(
            Path(self._analysis_dir).rglob("*.nwb"),
            desc="Scanning analysis files",
        ):
            is_empty_file = path.is_file() and path.stat().st_size == 0
            is_empty_dir = path.is_dir() and not any(path.iterdir())
            if is_empty_file or is_empty_dir:
                to_delete.add(path)
            elif path not in tracked:
                to_delete.add(path)

        if dry_run:
            return to_delete, tracked

        for path in to_delete:
            try:
                path.unlink()
            except Exception as e:
                self._logger.error(f"Error deleting file {path}: {e}")

        return to_delete, tracked

    def cleanup(self) -> None:
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
            AnalysisNwbfile().cleanup()

        Note:
            This is a destructive operation. Ensure you have backups before
            running cleanup on production databases. File deletions cannot
            be undone.

        See Also:
            docs/src/ForDevelopers/Management.md for detailed cleanup guide.
        """
        logger.info("Analysis cleanup")
        registry = AnalysisRegistry()
        registry.block_new_inserts()

        # Get all custom tables first so we can check their tracked files
        custom_tables = list(registry.all_classes)
        num_tables = len(custom_tables)
        common_orphans = self.get_orphans().proj()

        try:  # Long try block ensures that we always unblock inserts after
            for i, analysis_tbl in enumerate(custom_tables, start=1):
                tbl_name = analysis_tbl.full_table_name
                logger.info(f"  [{i}/{num_tables}] Processing {tbl_name}")

                # Step 1a: Get orphans from this analysis table
                _ = analysis_tbl.delete_orphans(dry_run=False, safemode=False)

                # Step 1b: Clean up this table's external entries
                _ = analysis_tbl.cleanup_external()

                # Step 1c: Remove valid entries from common orphans.
                if bool(analysis_tbl):
                    common_orphans -= analysis_tbl.proj()

            # Step 3: Delete remaining common orphans.
            if bool(common_orphans):
                common_orphans.delete_quick()

            # Step 4: Clean up common external table entries
            _ = self.cleanup_external()

            # Remove untracked files, checking both common and custom tables
            _ = self._remove_untracked_files(custom_tables, dry_run=False)

        finally:
            registry.unblock_new_inserts()

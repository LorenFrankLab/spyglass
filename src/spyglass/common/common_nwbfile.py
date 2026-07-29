import math
import os
import re
import stat as stat_module
import sys
import time
from dataclasses import dataclass
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


def _check_number(
    name: str,
    value: Union[int, float],
    *,
    minimum: float,
    maximum: Optional[float] = None,
) -> float:
    """Validate a numeric safety limit.

    Parameters
    ----------
    name : str
        Parameter name, used in the error message.
    value : int or float
        Supplied value.
    minimum : float
        Inclusive lower bound.
    maximum : float, optional
        Inclusive upper bound. Omit for an unbounded limit.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If the value is not a finite number within bounds. ``bool`` is
        rejected explicitly because it is an ``int`` subclass, so
        ``max_delete_fraction=True`` would otherwise read as 1.0. NaN and
        inf are rejected rather than coerced: under NaN every comparison is
        False and the guard silently vanishes.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    if value < minimum or (maximum is not None and value > maximum):
        bound = (
            f"in [{minimum}, {maximum}]"
            if maximum is not None
            else f">= {minimum}"
        )
        raise ValueError(f"{name} must be {bound}, got {value!r}")
    return float(value)


@dataclass(frozen=True)
class TargetSnapshot:
    """Identity of a real analysis file at scan time.

    Attributes
    ----------
    real_path : pathlib.Path
        Fully resolved path of the file.
    dev, ino : int
        Device and inode, used to prove the file was not swapped.
    size : int
        Size in bytes; 0 marks an empty analysis file.
    mtime_ns, ctime_ns : int
        Nanosecond timestamps, used by the age gate.
    mode : int
        Raw ``st_mode``; must be a regular file to be deletable.
    """

    real_path: Path
    dev: int
    ino: int
    size: int
    mtime_ns: int
    ctime_ns: int
    mode: int

    @property
    def newest_ns(self) -> int:
        """Newest of mtime/ctime, the conservative age basis."""
        return max(self.mtime_ns, self.ctime_ns)

    @property
    def is_regular(self) -> bool:
        """True when the target is a regular file (not fifo/socket/device)."""
        return stat_module.S_ISREG(self.mode)


@dataclass(frozen=True)
class AccessSnapshot:
    """One path under analysis_dir through which a target was reached.

    Attributes
    ----------
    access_path : pathlib.Path
        Path as found by the directory walk.
    is_link : bool
        True when ``access_path`` is itself a symlink.
    raw_link_target : str or None
        ``os.readlink`` value when ``is_link``, else None. Compared at
        deletion time so a re-pointed link is not followed.
    dev, ino : int
        ``lstat`` device and inode of the access path itself.
    mtime_ns, ctime_ns : int
        ``lstat`` nanosecond timestamps of the access path itself.
    """

    access_path: Path
    is_link: bool
    raw_link_target: Optional[str]
    dev: int
    ino: int
    mtime_ns: int
    ctime_ns: int

    @property
    def newest_ns(self) -> int:
        """Newest of mtime/ctime for this access path."""
        return max(self.mtime_ns, self.ctime_ns)


@dataclass(frozen=True)
class CleanupCandidate:
    """A real analysis file plus every in-tree path that reaches it.

    A broken symlink has access snapshots but no target snapshot; there is
    no target to snapshot. ``broken`` is therefore the absence of a target
    record, not a flag beside meaningless fields.
    """

    real_path: Path
    target: Optional[TargetSnapshot]
    accesses: Tuple[AccessSnapshot, ...]

    @property
    def broken(self) -> bool:
        """True when this candidate is a dangling symlink."""
        return self.target is None

    @property
    def newest_ns(self) -> int:
        """Newest timestamp across the target and every access alias.

        A fresh link to an old target may be work awaiting registration, so
        eligibility requires everything to be old enough.
        """
        stamps = [access.newest_ns for access in self.accesses]
        if self.target is not None:
            stamps.append(self.target.newest_ns)
        return max(stamps)


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
    candidates : dict of pathlib.Path to CleanupCandidate
        Deletion candidates keyed by resolved real path, carrying the
        identity snapshots re-verified before any unlink.
    deferred_recent_files : set of pathlib.Path
        Candidates held back because they are newer than the age limit.
    broken_links : set of pathlib.Path
        Resolved targets of dangling ``*.nwb`` symlinks.
    """

    scanned_files: Set[Path]
    tracked_files: Set[Path]
    files_to_delete: Set[Path]
    empty_files: Set[Path]
    untracked_files: Set[Path]
    candidates: Dict[Path, CleanupCandidate]
    deferred_recent_files: Set[Path]
    broken_links: Set[Path]


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

    def _snapshot_entry(self, access_path: Path):
        """Snapshot one scanned entry.

        Returns
        -------
        tuple or None
            ``(real_path, target_or_None, access_snapshot)``, or None when
            the entry vanished mid-scan and should be skipped.

        Raises
        ------
        OSError
            On permission, symlink-loop, or I/O errors. A destructive sweep
            fails closed rather than guessing.
        """
        try:
            lst = os.lstat(access_path)
        except FileNotFoundError:
            return None  # vanished between walk and stat

        is_link = stat_module.S_ISLNK(lst.st_mode)
        raw_target = os.readlink(access_path) if is_link else None
        access = AccessSnapshot(
            access_path=access_path,
            is_link=is_link,
            raw_link_target=raw_target,
            dev=lst.st_dev,
            ino=lst.st_ino,
            mtime_ns=lst.st_mtime_ns,
            ctime_ns=lst.st_ctime_ns,
        )
        real_path = Path(os.path.realpath(access_path))

        try:
            st = os.stat(access_path)  # follows symlinks
        except FileNotFoundError:
            if is_link:
                return real_path, None, access  # broken link
            return None  # regular entry vanished mid-scan

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

        def paths_from_external(tbl) -> Set[Path]:
            return {
                Path(fp[1]).expanduser().resolve()
                for fp in tbl._ext_tbl.fetch_external_paths()
            }

        tracked = paths_from_external(self)
        for tbl in custom_tables:
            tracked.update(paths_from_external(tbl))

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
            if real_path in tracked:
                continue  # tracked wins over empty and over broken
            if target is not None and not target.is_regular:
                logger.warning(
                    f"Skipping non-regular analysis path: {real_path}"
                )
                continue
            candidate = CleanupCandidate(
                real_path=real_path,
                target=target,
                accesses=tuple(accesses[real_path]),
            )
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

    def _current_tracked_paths(
        self, custom_tables: List[SpyglassAnalysis]
    ) -> Set[Path]:
        """Re-fetch tracked analysis paths for an act-time check."""
        tracked = {
            Path(fp[1]).expanduser().resolve()
            for fp in self._ext_tbl.fetch_external_paths()
        }
        for tbl in custom_tables:
            tracked.update(
                Path(fp[1]).expanduser().resolve()
                for fp in tbl._ext_tbl.fetch_external_paths()
            )
        return tracked

    @staticmethod
    def _access_still_matches(
        access: AccessSnapshot, *, real_path: Path, analysis_root: Path
    ) -> bool:
        """Re-verify one leaf symlink immediately before unlinking it.

        Deleting the target opens a window in which the link could be
        replaced; unlinking blindly would remove the replacement.

        Leaf inode and raw link text are not sufficient on their own: an
        intermediate directory symlink can be re-pointed without touching
        either, so the canonical destination, containment, and timestamps
        are all re-checked here.
        """
        try:
            lst = os.lstat(access.access_path)
        except OSError:
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
        except OSError:
            return False
        # Still inside the analysis root, resolving the PARENT so the link
        # itself is not followed.
        try:
            location = (
                access.access_path.parent.resolve() / access.access_path.name
            )
        except OSError:
            return False
        if not location.is_relative_to(analysis_root):
            return False
        # And still aliasing the candidate we just deleted. The target is
        # gone by now, so realpath resolves non-strictly to the same path.
        return Path(os.path.realpath(access.access_path)) == real_path

    def _candidate_still_matches(
        self, candidate: CleanupCandidate, *, analysis_root: Path
    ) -> Optional[List[AccessSnapshot]]:
        """Re-verify a candidate immediately before deleting it.

        Returns
        -------
        list of AccessSnapshot or None
            The in-root accesses that may be unlinked, or None (with a
            warning) when anything changed since the scan, when a
            non-regular target is involved, or when the target has no
            in-root access authorizing it.

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
            location = (
                access.access_path.parent.resolve() / access.access_path.name
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
            # Canonical equality is required for EVERY access, not just
            # symlinks. Without it an unrelated regular file inside the
            # analysis root could vouch for an arbitrary outside target in
            # a structurally consistent but forged plan.
            if Path(os.path.realpath(access.access_path)) != real_path:
                continue
            in_root.append(access)

        # Applies to broken candidates too: a forged broken plan naming an
        # outside symlink must not authorize unlinking it.
        if not in_root:
            return _refuse(
                "no in-root access currently resolving to it authorizes "
                "deletion"
            )

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
                if access.access_path.name[-4:].lower() != ".nwb":
                    structural.append(
                        f"{candidate.real_path}: access path "
                        f"{access.access_path} is not a *.nwb entry"
                    )
        if structural:
            raise RuntimeError(
                "Analysis file deletion failed: cleanup plan is malformed; "
                "refusing to delete anything:\n  " + "\n  ".join(structural)
            )

        # ---- ACT ----
        # Accumulator, not the original snapshot: a table can appear in the
        # registry while one candidate is processed and be gone before the
        # next. Re-unioning against `custom_tables` each time would forget
        # it and delete its tracked files, defeating the "never dropped"
        # guarantee _current_custom_tables provides within a single call.
        known_tables = list(custom_tables)
        for candidate in plan.candidates.values():
            # Tracking and registry membership are re-read for EVERY
            # candidate, not once up front: a file can be registered, or a
            # whole custom table declared, while an earlier candidate is
            # being deleted. Costs one fetch per candidate; the candidate
            # set is the untracked minority, and correctness wins here.
            # The deferred cleanup lease would let this be hoisted again.
            known_tables = self._current_custom_tables(known_tables)
            if candidate.real_path in self._current_tracked_paths(known_tables):
                logger.warning(
                    f"Skipping {candidate.real_path}: became tracked since "
                    "the scan"
                )
                continue
            # Age is re-checked at act time, not only during planning: a
            # long scan may have started before the file was written.
            # _candidate_still_matches additionally refuses any candidate
            # whose alias timestamps moved, so a touched link cannot slip
            # through on a stale snapshot.
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
                candidate, analysis_root=analysis_root
            )
            if not in_root:
                continue
            try:
                if not candidate.broken:
                    candidate.real_path.unlink()
                # Re-verify EVERY link as a group before unlinking any of
                # them. Aliases can chain (a.nwb -> b.nwb -> target): once
                # b is removed, a no longer resolves to the target, so
                # validating and unlinking in one pass would reject a and
                # leave it dangling, with the outcome depending on access
                # order. Validating first makes the result order-independent.
                # Only in-root accesses validated earlier are considered, so
                # an extra out-of-root access is never unlinked.
                to_unlink = []
                for access in in_root:
                    if not access.is_link:
                        continue
                    if not self._access_still_matches(
                        access,
                        real_path=candidate.real_path,
                        analysis_root=analysis_root,
                    ):
                        logger.warning(
                            f"Skipping link {access.access_path}: changed "
                            "after target deletion"
                        )
                        continue
                    to_unlink.append(access)
                for access in to_unlink:
                    access.access_path.unlink()
            except OSError as e:
                failures.append(f"{candidate.real_path}: {e}")

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
        max_delete_fraction: float = 0.9,
        max_delete_to_tracked_ratio: float = 10.0,
        *,
        min_file_age_hours: float = 24.0,
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
            files found in the scan. At the default ``max_delete_fraction``
            this limit cannot bind: every scanned file that is kept is
            tracked, so the fraction limit already caps the ratio at 9. It
            becomes the operative guard only when ``max_delete_fraction`` is
            raised above 10/11 (~0.909). This limit applies only to
            filesystem deletion of untracked or empty analysis NWB files, not
            to orphan row deletion. Defaults to 10.0.
        min_file_age_hours : float
            Untracked files newer than this are deferred to the next cleanup
            rather than deleted, protecting work that exists on disk but is
            not yet registered -- notably a file written to another volume
            and symlinked in before its row is inserted. Defaults to 24.0.
            Pass 0 only for intentional immediate cleanup.
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
                    self._logger.warning(
                        f"Cleanup plan would be refused: {plan_err}"
                    )
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
                    # A failed unblock halts ALL inserts across the database
                    # until manually cleared, so this must be loud regardless
                    # of whether another exception is already propagating.
                    self._logger.critical(
                        "Failed to unblock inserts after cleanup: "
                        f"{unblock_err}. Analysis inserts remain BLOCKED "
                        "database-wide until restored; run "
                        "AnalysisRegistry().unblock_new_inserts() manually."
                    )
                    # Re-raise only when no other exception is already
                    # propagating; otherwise we would mask the original
                    # cleanup error (the critical log above is the signal).
                    if cleanup_exc is None:
                        raise

    def check_all_files(
        self, resolve_tables: bool = False, verbose: bool = False
    ) -> dict:
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

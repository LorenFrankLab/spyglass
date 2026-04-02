from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import datajoint as dj
from datajoint.hash import uuid_from_file
from datajoint.logging import logger as dj_logger
from tqdm import tqdm

from spyglass.common.common_nwbfile import AnalysisRegistry
from spyglass.settings import test_mode
from spyglass.utils import SpyglassAnalysis, logger

schema = dj.Schema("common_file_tracking")


@schema
class AnalysisFileIssues(dj.Manual):
    """Track analysis files with issues across all analysis tables.

    This Manual table stores only problematic files (missing, unreadable,
    or checksum errors) from both common and custom AnalysisNwbfile tables.
    It's designed to work with AnalysisRegistry to support custom analysis
    tables.

    Key Methods
    -----------
    check1_file(analysis_tbl, analysis_file_name) :
        Check a single file for issues
    check_files(analysis_tbl, verbose=False) :
        Check all files in an analysis table
    show_downstream(restriction=True) :
        Show downstream tables affected by file issues

    Usage
    -----
    Typically called via AnalysisNwbfile.check_all_files():

    >>> from spyglass.common import AnalysisNwbfile
    >>> results = AnalysisNwbfile().check_all_files()
    >>> # View issues
    >>> from spyglass.common.common_file_tracking import AnalysisFileIssues
    >>> AnalysisFileIssues().show_downstream()

    Notes
    -----
    - Only inserts entries for files with issues (keeps table lean)
    - Supports both common and custom analysis tables via AnalysisRegistry
    - Tracks source table via full_table_name field
    """

    definition = """
    # Tracks analysis files with issues (missing, unreadable, checksum errors)
    full_table_name: varchar(128)  # source analysis table
    analysis_file_name: varchar(64)  # name of the problematic file
    ---
    on_disk : bool  # whether the analysis file exists on disk
    can_read : bool  # whether the analysis file is readable
    issue : varchar(255)  # description of the issue
    table=NULL : varchar(64)  # name of the downstream table that refs the file
    """

    # NOTE: exists and readable are SQL keywords. Use on_disk and can_read

    not_exist = dict(on_disk=False, can_read=False)
    checksum = dict(on_disk=True, can_read=False)

    @staticmethod
    def _get_recompute_deleted() -> set:
        """Return analysis file names intentionally deleted by recompute.

        Queries the v1 spike sorting recompute table for files that were
        successfully recomputed and then deleted (deleted=1).

        Returns
        -------
        deleted : set of str
            Analysis file names that were intentionally deleted.
        """
        deleted = set()

        try:
            from spyglass.spikesorting.v1.recompute import (
                RecordingRecompute as V1RecordingRecompute,
            )

            deleted.update(
                (V1RecordingRecompute().with_names & "deleted=1").fetch(
                    "analysis_file_name"
                )
            )
        except Exception as e:
            logger.warning(f"Could not fetch v1 recompute deleted files: {e}")

        return deleted

    @staticmethod
    def _batch_resolve_paths(analysis_tbl, file_names: list) -> tuple:
        """Fetch all file paths and stored hashes in a single external table query.

        Parameters
        ----------
        analysis_tbl : SpyglassAnalysis
            The analysis table whose external table is queried.
        file_names : list of str
            All analysis file names to resolve.

        Returns
        -------
        path_map : dict
            Mapping of analysis_file_name -> absolute path string.
        hash_map : dict
            Mapping of analysis_file_name -> stored contents_hash UUID.
            Only includes files where contents_hash is not NULL.
        """
        analysis_dir = Path(analysis_tbl._analysis_dir)

        path_map = {}
        hash_map = {}
        for row in analysis_tbl._ext_tbl.fetch(
            "filepath", "contents_hash", as_dict=True
        ):
            fname = Path(row["filepath"]).name
            path_map[fname] = str(analysis_dir / row["filepath"])
            if row["contents_hash"] is not None:
                hash_map[fname] = row["contents_hash"]

        # Fall back to generated subdirectory path for unregistered files
        for fname in file_names:
            if fname not in path_map:
                subdir = fname[: fname.rfind("_")]
                path_map[fname] = str(analysis_dir / subdir / fname)

        return path_map, hash_map

    @staticmethod
    def _check_path(
        fname: str,
        analysis_file_name: str,
        full_table_name: str,
        deleted_files: set,
        hash_map: dict,
    ) -> dict:
        """Check file existence and checksum. Thread-safe; makes no DB calls.

        Existence is checked via Path.exists(). If a stored hash is available
        in hash_map, the file's hash is computed locally (I/O-only) and
        compared against the stored value.

        Parameters
        ----------
        fname : str
            Absolute path of the file to check.
        analysis_file_name : str
            Name of the analysis file (used as the insert key).
        full_table_name : str
            Full table name of the source analysis table.
        deleted_files : set
            File names intentionally deleted by recompute (skip these).
        hash_map : dict
            Pre-fetched mapping of analysis_file_name -> stored contents_hash.

        Returns
        -------
        insert_dict : dict or None
            Dict ready for insertion into AnalysisFileIssues, or None if OK.
        """
        if analysis_file_name in deleted_files:
            return None

        key = dict(
            full_table_name=full_table_name,
            analysis_file_name=analysis_file_name,
        )

        if not Path(fname).exists():
            return dict(
                key,
                on_disk=False,
                can_read=False,
                issue=f"path not found: {fname}",
            )

        stored_hash = hash_map.get(analysis_file_name)
        if stored_hash is not None:
            try:
                current_hash = uuid_from_file(fname)
            except OSError as exc:
                return dict(
                    key,
                    on_disk=True,
                    can_read=False,
                    issue=f"failed to hash file {fname}: {exc}",
                )
            if current_hash != stored_hash:
                return dict(
                    key,
                    on_disk=True,
                    can_read=False,
                    issue=f"checksum mismatch: {fname}",
                )

        return None

    def get_tbl(
        self, analysis_tbl: SpyglassAnalysis, analysis_file_name: str
    ) -> str:
        """Find which child table contains the file.

        Parameters
        ----------
        analysis_tbl : AnalysisNwbfile
            The specific analysis table instance to search within
        analysis_file_name : str
            Name of the analysis file to find

        Returns
        -------
        table_name : str
            Full table name of the child table that contains the file
        """
        # Query only this table's children for the file
        ret = [
            c.full_table_name
            for c in analysis_tbl.children(as_objects=True)
            if c & dict(analysis_file_name=analysis_file_name)
            if not c.full_table_name.endswith("_log")
        ]

        if len(ret) != 1:
            raise ValueError(
                f"{len(ret)} tables for {analysis_file_name} in "
                f"{analysis_tbl.full_table_name}: {ret}"
            )
        return ret[0]

    def resolve_table_refs(self, restriction=True) -> int:
        """Populate the table field for issues where it is currently NULL.

        For each source analysis table, fetches its child tables once, then
        batch-queries each child against all unresolved file names — one query
        per child rather than one per (child × file).

        Parameters
        ----------
        restriction : bool or dict, optional
            Limit which AnalysisFileIssues rows are updated. Default True
            (all rows with table IS NULL).

        Returns
        -------
        updated : int
            Number of rows whose table field was populated.
        """
        entries = (self & restriction & "`table` IS NULL").fetch(as_dict=True)
        if not entries:
            return 0

        # Group unresolved file names by their source analysis table
        by_table = {}
        for entry in entries:
            by_table.setdefault(entry["full_table_name"], []).append(
                entry["analysis_file_name"]
            )

        updated = 0
        for table_name, file_names in by_table.items():
            analysis_cls = AnalysisRegistry().get_class(table_name)
            if not analysis_cls:
                continue

            children = [
                c
                for c in analysis_cls().children(as_objects=True)
                if not c.full_table_name.endswith("_log")
            ]

            # One batch query per child instead of one per (child × file)
            file_keys = [dict(analysis_file_name=f) for f in file_names]
            for child in children:
                matched = (child & file_keys).fetch("analysis_file_name")
                for fname in matched:
                    self.update1(
                        dict(
                            full_table_name=table_name,
                            analysis_file_name=fname,
                            table=child.full_table_name,
                        )
                    )
                    updated += 1

        return updated

    def check1_file(self, analysis_tbl, analysis_file_name, deleted_files=None):
        """Check a single file and insert only if there's an issue.

        Parameters
        ----------
        analysis_tbl : AnalysisNwbfile
            The analysis table instance containing the file
        analysis_file_name : str
            Name of the analysis file to check
        deleted_files : set, optional
            Set of analysis file names intentionally deleted by recompute.
            Files in this set are skipped even if missing from disk.
            If None, computed via _get_recompute_deleted().

        Returns
        -------
        issue_found : bool
            True if an issue was found and inserted, False if file is OK
        """
        prev_level = dj_logger.level
        dj_logger.setLevel("ERROR")

        if deleted_files is None:
            deleted_files = self._get_recompute_deleted()

        issue_found = False
        insert = None
        key = {
            "full_table_name": analysis_tbl.full_table_name,
            "analysis_file_name": analysis_file_name,
        }

        try:
            fname = analysis_tbl.get_abs_path(analysis_file_name)
            # Check if file exists on disk
            if not Path(fname).exists():
                if analysis_file_name not in deleted_files:
                    insert = dict(
                        key, **self.not_exist, issue=f"path not found: {fname}"
                    )
                    issue_found = True
        except FileNotFoundError as e:
            if analysis_file_name not in deleted_files:
                insert = dict(key, **self.not_exist, issue=e.args[0])
                issue_found = True
        except dj.DataJointError as e:
            insert = dict(
                key,
                **self.checksum,
                issue=e.args[0],
                table=self.get_tbl(analysis_tbl, analysis_file_name),
            )
            issue_found = True

        if insert:
            self.insert1(insert, skip_duplicates=True)

        dj_logger.setLevel(prev_level)
        return issue_found

    def check_files(
        self,
        analysis_tbl: SpyglassAnalysis,
        verbose: bool = False,
        deleted_files: set = None,
        resolve_tables: bool = False,
        limit: int = None,
    ) -> int:
        """Check all files in an analysis table.

        Uses batch path resolution and a thread pool to parallelize existence
        and checksum checks. Optionally resolves downstream table references
        for new issues via resolve_table_refs().

        Parameters
        ----------
        analysis_tbl : SpyglassAnalysis
            The analysis table to check.
        verbose : bool, optional
            Show a tqdm progress bar. Default False.
        deleted_files : set, optional
            Pre-computed set of files deleted by recompute. If None, fetched
            via _get_recompute_deleted(). Pass from check_all_files to avoid
            re-fetching once per table.
        resolve_tables : bool, optional
            After inserting issues, populate the table field by querying
            downstream child tables. Default False.
        limit : int, optional
            Cap the number of files checked. Useful for debugging. Default None
            (check all files).

        Returns
        -------
        issue_count : int
            Number of new file issues found and inserted.
        """
        if not analysis_tbl:
            return 0

        if deleted_files is None:
            deleted_files = self._get_recompute_deleted()

        all_files = analysis_tbl.fetch("analysis_file_name")
        if not len(all_files):
            return 0

        # A: One external-table query resolves all paths and fetches stored hashes
        path_map, hash_map = self._batch_resolve_paths(analysis_tbl, all_files)

        # E: Skip files already recorded as issues (avoid redundant checks)
        known_issues = set(
            (self & {"full_table_name": analysis_tbl.full_table_name}).fetch(
                "analysis_file_name"
            )
        )
        to_check = [f for f in all_files if f not in known_issues]
        if limit is not None:
            to_check = to_check[:limit]

        full_table_name = analysis_tbl.full_table_name

        # C: Parallel existence checks — I/O-bound, benefits from threading
        issues = []
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self._check_path,
                    path_map[f],
                    f,
                    full_table_name,
                    deleted_files,
                    hash_map,
                )
                for f in to_check
            ]
            for future in tqdm(
                as_completed(futures), total=len(futures), disable=not verbose
            ):
                result = future.result()
                if result:
                    issues.append(result)

        # D: One batch insert instead of one insert1() per issue
        inserted_count = 0
        if issues:
            # Count how many of these issues already exist before insertion
            pre_existing = len(self & issues)
            self.insert(issues, skip_duplicates=True)
            # Count again after insertion to see how many are now present
            post_existing = len(self & issues)
            inserted_count = post_existing - pre_existing
            if resolve_tables:
                self.resolve_table_refs({"full_table_name": full_table_name})

        return inserted_count

    def show_downstream(self, restriction=True):
        """Show downstream tables affected by file issues.

        Parameters
        ----------
        restriction : bool or dict
            Restriction to apply when fetching issues

        Returns
        -------
        affected_tables : list or query
            Downstream tables that reference files with issues
        """
        entries = (self & "can_read=0" & restriction).fetch("KEY", as_dict=True)
        if not entries:
            if not test_mode:
                logger.info("No issues found.")
            return []

        # Get unique analysis tables from entries
        analysis_tables_names = set(e["full_table_name"] for e in entries)

        # For each analysis table, check which children are affected
        ret = []
        for table_name in analysis_tables_names:
            analysis_cls = AnalysisRegistry().get_class(table_name)
            if not analysis_cls:
                continue

            analysis_tbl = analysis_cls()
            children = analysis_tbl.children(as_objects=True)

            # Check which children have the problematic entries
            for child in children:
                if child & entries:
                    ret.append(child & entries)

        if not ret and not test_mode:
            logger.info("No issues found.")

        return ret or []

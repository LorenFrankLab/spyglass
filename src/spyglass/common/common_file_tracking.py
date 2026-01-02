from pathlib import Path

import datajoint as dj
from datajoint.logging import logger as dj_logger
from tqdm import tqdm

from spyglass.common import AnalysisRegistry
from spyglass.utils import SpyglassAnalysis

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
        ]

        if len(ret) != 1:
            raise ValueError(
                f"{len(ret)} tables for {analysis_file_name} in "
                f"{analysis_tbl.full_table_name}: {ret}"
            )
        return ret[0]

    def check1_file(self, analysis_tbl, analysis_file_name):
        """Check a single file and insert only if there's an issue.

        Parameters
        ----------
        analysis_tbl : AnalysisNwbfile
            The analysis table instance containing the file
        analysis_file_name : str
            Name of the analysis file to check

        Returns
        -------
        issue_found : bool
            True if an issue was found and inserted, False if file is OK
        """
        prev_level = dj_logger.level
        dj_logger.setLevel("ERROR")

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
                insert = dict(
                    key, **self.not_exist, issue=f"path not found: {fname}"
                )
                issue_found = True
        except FileNotFoundError as e:
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
        self, analysis_tbl: SpyglassAnalysis, verbose: bool = False
    ) -> int:
        """Check all files in an analysis table.

        Parameters
        ----------
        analysis_tbl : type
            The analysis table class

        Returns
        -------
        issue_count : int
            Number of files with issues found
        """
        if not analysis_tbl:
            return 0

        issue_count = 0
        all_files = analysis_tbl.fetch("analysis_file_name")
        for f_name in tqdm(all_files, disable=not verbose):
            if self.check1_file(analysis_tbl, f_name):
                issue_count += 1

        return issue_count

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
            print("No issues found.")
            return

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

        if not ret:
            print("No issues found.")
            return
        return ret if len(ret) > 1 else ret[0]

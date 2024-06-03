"""A schema to store the usage of advanced Spyglass features.

Records show usage of features such as cautious delete and fault-permitting
insert, which will be used to
determine which features are used, how often, and by whom. This will help
plan future development of Spyglass.
"""

from pathlib import Path
from typing import List, Union

import datajoint as dj
from datajoint import FreeTable
from datajoint import config as dj_config
from pynwb import NWBHDF5IO

from spyglass.common.common_nwbfile import AnalysisNwbfile, Nwbfile
from spyglass.settings import export_dir
from spyglass.utils import SpyglassMixin, logger
from spyglass.utils.dj_graph import RestrGraph
from spyglass.utils.dj_helper_fn import (
    make_file_obj_id_unique,
    unique_dicts,
    update_analysis_for_dandi_standard,
)

schema = dj.schema("common_usage")


@schema
class CautiousDelete(dj.Manual):
    definition = """
    id: int auto_increment
    ---
    dj_user: varchar(64)
    duration: float
    origin: varchar(64)
    restriction: varchar(255)
    merge_deletes = null: blob
    """


@schema
class InsertError(dj.Manual):
    definition = """
    id: int auto_increment
    ---
    dj_user: varchar(64)
    connection_id: int     # MySQL CONNECTION_ID()
    nwb_file_name: varchar(64)
    table: varchar(64)
    error_type: varchar(64)
    error_message: varchar(255)
    error_raw = null: blob
    """


@schema
class ExportSelection(SpyglassMixin, dj.Manual):
    definition = """
    export_id: int auto_increment
    ---
    paper_id: varchar(32)
    analysis_id: varchar(32)
    spyglass_version: varchar(16)
    time=CURRENT_TIMESTAMP: timestamp
    unique index (paper_id, analysis_id)
    """

    class Table(SpyglassMixin, dj.Part):
        definition = """
        -> master
        table_id: int
        ---
        table_name: varchar(128)
        restriction: varchar(2048)
        """

        def insert1(self, key, **kwargs):
            key = self._auto_increment(key, pk="table_id")
            super().insert1(key, **kwargs)

        def insert(self, keys: List[dict], **kwargs):
            if not isinstance(keys[0], dict):
                raise TypeError("Pass Table Keys as list of dict")
            keys = [self._auto_increment(k, pk="table_id") for k in keys]
            super().insert(keys, **kwargs)

    class File(SpyglassMixin, dj.Part):
        definition = """
        -> master
        -> AnalysisNwbfile
        """
        # Note: only tracks AnalysisNwbfile. list_file_paths also grabs Nwbfile.

    def insert1_return_pk(self, key: dict, **kwargs) -> int:
        """Custom insert to return export_id."""
        status = "Resuming"
        if not (query := self & key):
            key = self._auto_increment(key, pk="export_id")
            super().insert1(key, **kwargs)
            status = "Starting"
        export_id = query.fetch1("export_id")
        export_key = {"export_id": export_id}
        if query := (Export & export_key):
            query.super_delete(warn=False)
        logger.info(f"{status} {export_key}")
        return export_id

    def start_export(self, paper_id, analysis_id) -> None:
        """Start logging a new export."""
        self._start_export(paper_id, analysis_id)

    def stop_export(self, **kwargs) -> None:
        """Stop logging the current export."""
        self._stop_export()

    # NOTE: These helpers could be moved to table below, but I think
    #       end users may want to use them to check what's in the export log
    #       before actually exporting anything, which is more associated with
    #       Selection

    def list_file_paths(self, key: dict) -> list[str]:
        """Return a list of unique file paths for a given restriction/key.

        Note: This list reflects files fetched during the export process. For
        upstream files, use RestrGraph.file_paths.
        """
        file_table = self * self.File & key
        analysis_fp = [
            AnalysisNwbfile().get_abs_path(fname)
            for fname in file_table.fetch("analysis_file_name")
        ]
        nwbfile_fp = [
            Nwbfile().get_abs_path(fname)
            for fname in (AnalysisNwbfile * file_table).fetch("nwb_file_name")
        ]
        return [{"file_path": p} for p in list({*analysis_fp, *nwbfile_fp})]

    def get_restr_graph(self, key: dict, verbose=False) -> RestrGraph:
        """Return a RestrGraph for a restriction/key's tables/restrictions.

        Ignores duplicate entries.

        Parameters
        ----------
        key : dict
            Any valid restriction key for ExportSelection.Table
        verbose : bool, optional
            Turn on RestrGraph verbosity. Default False.
        """
        leaves = unique_dicts(
            (self * self.Table & key).fetch(
                "table_name", "restriction", as_dict=True
            )
        )
        return RestrGraph(seed_table=self, leaves=leaves, verbose=verbose)

    def preview_tables(self, **kwargs) -> list[dj.FreeTable]:
        """Return a list of restricted FreeTables for a given restriction/key.

        Useful for checking what will be exported.
        """
        return self.get_restr_graph(kwargs).leaf_ft

    def _max_export_id(self, paper_id: str, return_all=False) -> int:
        """Return last export associated with a given paper id.

        Used to populate Export table."""
        if isinstance(paper_id, dict):
            paper_id = paper_id.get("paper_id")
        if not (query := self & {"paper_id": paper_id}):
            return None
        all_export_ids = query.fetch("export_id")
        return all_export_ids if return_all else max(all_export_ids)

    def paper_export_id(self, paper_id: str) -> dict:
        """Return the maximum export_id for a paper, used to populate Export."""
        return {"export_id": self._max_export_id(paper_id)}


@schema
class Export(SpyglassMixin, dj.Computed):
    definition = """
    -> ExportSelection
    ---
    paper_id: varchar(32)
    """

    # In order to get a many-to-one relationship btwn Selection and Export,
    # we ignore all but the last export_id. If more exports are added above,
    # generating a new output will overwrite the old ones.

    class Table(SpyglassMixin, dj.Part):
        definition = """
        -> master
        table_id: int
        ---
        table_name: varchar(128)
        restriction: mediumblob
        unique index (export_id, table_name)
        """

    class File(SpyglassMixin, dj.Part):
        definition = """
        -> master
        file_id: int
        ---
        file_path: varchar(255)
        """

    def populate_paper(self, paper_id: Union[str, dict]):
        if isinstance(paper_id, dict):
            paper_id = paper_id.get("paper_id")
        self.populate(ExportSelection().paper_export_id(paper_id))

    def make(self, key):
        query = ExportSelection & key
        paper_key = query.fetch("paper_id", as_dict=True)[0]

        # Null insertion if export_id is not the maximum for the paper
        all_export_ids = query._max_export_id(paper_key, return_all=True)
        max_export_id = max(all_export_ids)
        if key.get("export_id") != max_export_id:
            logger.info(
                f"Skipping export_id {key['export_id']}, use {max_export_id}"
            )
            self.insert1(key)
            return

        # If lesser ids are present, delete parts yielding null entries
        processed_ids = set(
            list(self.Table.fetch("export_id"))
            + list(self.File.fetch("export_id"))
        )
        if overlap := set(all_export_ids) - {max_export_id} & processed_ids:
            logger.info(f"Overwriting export_ids {overlap}")
            for export_id in overlap:
                id_dict = {"export_id": export_id}
                (self.Table & id_dict).delete_quick()
                (self.Table & id_dict).delete_quick()

        restr_graph = query.get_restr_graph(paper_key)
        file_paths = unique_dicts(  # Original plus upstream files
            query.list_file_paths(paper_key) + restr_graph.file_paths
        )

        table_inserts = [
            {**key, **rd, "table_id": i}
            for i, rd in enumerate(restr_graph.as_dict)
        ]
        file_inserts = [
            {**key, **fp, "file_id": i} for i, fp in enumerate(file_paths)
        ]

        # Writes but does not run mysqldump. Assumes single version per paper.
        version_key = query.fetch("spyglass_version", as_dict=True)[0]
        self.write_export(
            free_tables=restr_graph.all_ft, **paper_key, **version_key
        )

        self.insert1({**key, **paper_key})
        self.Table().insert(table_inserts)
        self.File().insert(file_inserts)

    def _get_credentials(self):
        """Get credentials for database connection."""
        return {
            "user": dj_config["database.user"],
            "password": dj_config["database.password"],
            "host": dj_config["database.host"],
        }

    def _write_sql_cnf(self):
        """Write SQL cnf file to avoid password prompt."""
        cnf_path = Path("~/.my.cnf").expanduser()

        if cnf_path.exists():
            return

        template = "[client]\nuser={user}\npassword={password}\nhost={host}\n"

        with open(str(cnf_path), "w") as file:
            file.write(template.format(**self._get_credentials()))
        cnf_path.chmod(0o600)

    def _bash_escape(self, s):
        """Escape restriction string for bash."""
        s = s.strip()

        replace_map = {
            "WHERE ": "",  # Remove preceding WHERE of dj.where_clause
            "  ": " ",  # Squash double spaces
            "( (": "((",  # Squash double parens
            ") )": ")",
            '"': "'",  # Replace double quotes with single
            "`": "",  # Remove backticks
            " AND ": " \\\n\tAND ",  # Add newline and tab for readability
            " OR ": " \\\n\tOR  ",  # OR extra space to align with AND
            ")AND(": ") \\\n\tAND (",
            ")OR(": ") \\\n\tOR  (",
            "#": "\\#",
        }
        for old, new in replace_map.items():
            s = s.replace(old, new)
        if s.startswith("(((") and s.endswith(")))"):
            s = s[2:-2]  # Remove extra parens for readability
        return s

    def _cmd_prefix(self, docker_id=None):
        """Get prefix for mysqldump command. Includes docker exec if needed."""
        if not docker_id:
            return "mysqldump "
        return (
            f"docker exec -i {docker_id} \\\n\tmysqldump "
            + "-u {user} --password={password} \\\n\t".format(
                **self._get_credentials()
            )
        )

    def _write_mysqldump(
        self,
        free_tables: List[FreeTable],
        paper_id: str,
        docker_id=None,
        spyglass_version=None,
    ):
        """Write mysqlmdump.sh script to export data.

        Parameters
        ----------
        paper_id : str
            Paper ID to use for export file names
        docker_id : str, optional
            Docker container ID to export from. Default None
        spyglass_version : str, optional
            Spyglass version to include in export. Default None
        """
        paper_dir = Path(export_dir) / paper_id if not docker_id else Path(".")
        paper_dir.mkdir(exist_ok=True)

        dump_script = paper_dir / f"_ExportSQL_{paper_id}.sh"
        dump_content = paper_dir / f"_Populate_{paper_id}.sql"

        prefix = self._cmd_prefix(docker_id)
        version = (  # Include spyglass version as comment in dump
            "echo '--'\n"
            + f"echo '-- SPYGLASS VERSION: {spyglass_version} --'\n"
            + "echo '--'\n\n"
            if spyglass_version
            else ""
        )
        create_cmd = (
            "echo 'CREATE DATABASE IF NOT EXISTS {database}; "
            + "USE {database};'\n\n"
        )
        dump_cmd = prefix + '{database} {table} --where="\\\n\t{where}"\n\n'

        tables_by_db = sorted(free_tables, key=lambda x: x.full_table_name)

        with open(dump_script, "w") as file:
            file.write(
                "#!/bin/bash\n\n"
                + f"exec > {dump_content}\n\n"  # Redirect output to sql file
                + f"{version}"  # Include spyglass version as comment
            )

            prev_db = None
            for table in tables_by_db:
                if not (where := table.where_clause()):
                    continue
                where = self._bash_escape(where)
                database, table_name = (
                    table.full_table_name.replace("`", "")
                    .replace("#", "\\#")
                    .split(".")
                )
                if database != prev_db:
                    file.write(create_cmd.format(database=database))
                    prev_db = database
                file.write(
                    dump_cmd.format(
                        database=database, table=table_name, where=where
                    )
                )
        logger.info(f"Export script written to {dump_script}")

    def write_export(
        self,
        free_tables: List[FreeTable],
        paper_id: str,
        docker_id=None,
        spyglass_version=None,
    ):
        """Write export bash script for all tables in graph.

        Also writes a user-specific .my.cnf file to avoid password prompt.

        Parameters
        ----------
        free_tables : List[FreeTable]
            List of restricted FreeTables to export
        paper_id : str
            Paper ID to use for export file names
        docker_id : str, optional
            Docker container ID to export from. Default None
        spyglass_version : str, optional
            Spyglass version to include in export. Default None
        """
        self._write_sql_cnf()
        self._write_mysqldump(
            free_tables, paper_id, docker_id, spyglass_version
        )

        # TODO: export conda env

    def prepare_files_for_export(self, key, **kwargs):
        """Resolve common known errors to make a set of analysis
        files dandi compliant

        Parameters
        ----------
        key : dict
            restriction for a single entry of the Export table
        """
        key = (self & key).fetch1("KEY")
        self._make_fileset_ids_unique(key)
        file_list = (self.File() & key).fetch("file_path")
        for file in file_list:
            update_analysis_for_dandi_standard(file, **kwargs)

    def _make_fileset_ids_unique(self, key):
        """Make the object_id of each nwb in a dataset unique"""
        key = (self & key).fetch1("KEY")
        file_list = (self.File() & key).fetch("file_path")
        unique_object_ids = []
        for file_path in file_list:
            with NWBHDF5IO(file_path, "r") as io:
                nwb = io.read()
                object_id = nwb.object_id
            if object_id not in unique_object_ids:
                unique_object_ids.append(object_id)
            else:
                new_id = make_file_obj_id_unique(file_path)
                unique_object_ids.append(new_id)

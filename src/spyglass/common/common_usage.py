"""A schema to store the usage of advanced Spyglass features.

Records show usage of features such as cautious delete and fault-permitting
insert, which will be used to
determine which features are used, how often, and by whom. This will help
plan future development of Spyglass.
"""

from typing import Union

import datajoint as dj

from spyglass.common.common_nwbfile import AnalysisNwbfile, Nwbfile
from spyglass.utils import SpyglassMixin, logger
from spyglass.utils.dj_graph import RestrGraph
from spyglass.utils.dj_helper_fn import unique_dicts

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
        table_name: varchar(64)
        restriction: varchar(2048)
        """

        def insert1(self, key, **kwargs):
            key = self._auto_increment(key, pk="table_id")
            super().insert1(key, **kwargs)

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
            super().insert1(key, **kwargs)
            status = "Starting"
        export_id = query.fetch1("export_id")
        export_key = {"export_id": export_id}
        if query := (Export & export_key):
            query.super_delete(warn=False)
        logger.info(f"{status} {export_key}")
        return export_id

    def start_export(self, paper_id, analysis_id) -> None:
        self._start_export(paper_id, analysis_id)

    def stop_export(self) -> None:
        self._stop_export()

    # NOTE: These helpers could be moved to table below, but I think
    #       end users may want to use them to check what's in the export log
    #       before actually exporting anything, which is more associated with
    #       Selection

    def list_file_paths(self, key: dict) -> list[str]:
        file_table = self.File & key
        analysis_fp = [
            AnalysisNwbfile().get_abs_path(fname)
            for fname in file_table.fetch("analysis_file_name")
        ]
        nwbfile_fp = [
            Nwbfile().get_abs_path(fname)
            for fname in (AnalysisNwbfile * file_table).fetch("nwb_file_name")
        ]
        return [{"file_path": p} for p in list({*analysis_fp, *nwbfile_fp})]

    def get_restr_graph(self, key: dict) -> RestrGraph:
        leaves = unique_dicts(
            (self.Table & key).fetch("table_name", "restriction", as_dict=True)
        )
        return RestrGraph(seed_table=self, leaves=leaves, verbose=True)

    def preview_tables(self, key: dict) -> list[dj.FreeTable]:
        return self.get_restr_graph(key).leaf_ft

    def _min_export_id(self, paper_id: str) -> int:
        """Return all export_ids for a paper."""
        if isinstance(paper_id, dict):
            paper_id = paper_id.get("paper_id")
        if not (query := self & {"paper_id": paper_id}):
            return None
        return min(query.fetch("export_id"))

    def paper_export_id(self, paper_id: str) -> dict:
        """Return the minimum export_id for a paper, used to populate Export."""
        return {"export_id": self._min_export_id(paper_id)}


@schema
class Export(SpyglassMixin, dj.Computed):
    definition = """
    -> ExportSelection
    """

    # In order to get a many-to-one relationship btwn Selection and Export,
    # we ignore all but the first export_id.

    class Table(SpyglassMixin, dj.Part):
        definition = """
        -> master
        table_id: int
        ---
        table_name: varchar(64)
        restriction: varchar(2048)
        unique index (table_name)
        """

    class File(SpyglassMixin, dj.Part):
        definition = """
        -> master
        file_id: int
        ---
        file_path: varchar(255)
        """
        # What's needed? full path? relative path?

    def populate_paper(self, paper_id: Union[str, dict]):
        if isinstance(paper_id, dict):
            paper_id = paper_id.get("paper_id")
        self.populate(ExportSelection().paper_export_id(paper_id))

    def make(self, key):
        query = ExportSelection & key
        paper_key = query.fetch("paper_id", as_dict=True)[0]

        # Null insertion if export_id is not the minimum for the paper
        min_export_id = query._min_export_id(paper_key)
        if key.get("export_id") != min_export_id:
            logger.info(
                f"Skipping export_id {key['export_id']}, use {min_export_id}"
            )
            self.insert1(key)
            return

        restr_graph = query.get_restr_graph(paper_key)
        file_paths = query.list_file_paths(paper_key)

        table_inserts = [
            {**key, **rd, "table_id": i}
            for i, rd in enumerate(restr_graph.as_dict)
        ]
        file_inserts = [
            {**key, **fp, "file_id": i} for i, fp in enumerate(file_paths)
        ]

        # Writes but does not run mysqldump. Assumes single version per paper.
        version_key = query.fetch("spyglass_version", as_dict=True)[0]
        restr_graph.write_export(**paper_key, **version_key)

        self.insert1(key)
        self.Table().insert(table_inserts)  # TODO: Duplicate error??
        self.File().insert(file_inserts)

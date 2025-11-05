"""A schema to store the usage of advanced Spyglass features.

Records show usage of features such as cautious delete and fault-permitting
insert, which will be used to
determine which features are used, how often, and by whom. This will help
plan future development of Spyglass.
"""

from typing import List, Union

import datajoint as dj
from pynwb import NWBHDF5IO

from spyglass.common.common_nwbfile import AnalysisNwbfile, Nwbfile
from spyglass.settings import test_mode
from spyglass.utils import SpyglassMixin, SpyglassMixinPart, logger
from spyglass.utils.dj_graph import RestrGraph
from spyglass.utils.dj_helper_fn import (
    make_file_obj_id_unique,
    unique_dicts,
    update_analysis_for_dandi_standard,
)
from spyglass.utils.nwb_helper_fn import get_linked_nwbs
from spyglass.utils.sql_helper_fn import SQLDumpHelper

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
class ActivityLog(dj.Manual):
    """A log of suspected low-use features worth deprecating."""

    definition = """
    id: int auto_increment
    ---
    function: varchar(64)
    dj_user: varchar(64)
    timestamp=CURRENT_TIMESTAMP: timestamp
    """

    @classmethod
    def deprecate_log(cls, name, alt=None, warning=True) -> None:
        """Log a deprecation warning for a feature.

        Parameters
        ----------
        name : str
            The name of the feature to deprecate.
        alt : str, optional
            What to use instead. Default no such message.
        warning : bool, optional
            Whether to log a warning. Default is True.
        """
        if warning:
            msg = f"\n\tUse {alt} instead" if alt else ""
            logger.warning(
                f"DEPRECATION scheduled for Spyglass 0.6.0: {name}{msg}"
            )
        cls.insert1(
            dict(dj_user=dj.config["database.user"], function=name[:64])
        )


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

    class Table(SpyglassMixinPart):
        definition = """
        -> master
        table_id: int
        ---
        table_name: varchar(128)
        restriction: varchar(2048)
        """

        def insert1(self, key, **kwargs):
            """Override insert1 to auto-increment table_id."""
            key = self._auto_increment(key, pk="table_id")
            super().insert1(key, **kwargs)

        def insert(self, keys: List[dict], **kwargs):
            """Override insert to auto-increment table_id."""
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
            safemode = False if test_mode else None  # No prompt in tests
            query.super_delete(warn=False, safemode=safemode)
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

    def _list_raw_files(self, key: dict) -> list[str]:
        """Return a list of unique nwb file names for a given restriction/key."""
        file_table = self * self.File & key
        return list(
            {
                *AnalysisNwbfile.join(file_table, log_export=False).fetch(
                    "nwb_file_name"
                )
            }
        )

    def _list_analysis_files(self, key: dict) -> list[str]:
        """Return a list of unique analysis file names for a given restriction/key."""
        file_table = self * self.File & key
        return list(file_table.fetch("analysis_file_name"))

    def list_file_paths(self, key: dict, as_dict=True) -> list[str]:
        """Return a list of unique file paths for a given restriction/key.

        Note: This list reflects files fetched during the export process. For
        upstream files, use RestrGraph.file_paths.

        Parameters
        ----------
        key : dict
            Any valid restriction key for ExportSelection.Table
        as_dict : bool, optional
            Return as a list of dicts: [{'file_path': x}]. Default True.
            If False, returns a list of strings without key.
        """
        unique_fp = {
            *[
                AnalysisNwbfile().get_abs_path(p)
                for p in self._list_analysis_files(key)
            ],
            *[Nwbfile().get_abs_path(p) for p in self._list_raw_files(key)],
        }

        return [{"file_path": p} for p in unique_fp] if as_dict else unique_fp

    @property
    def _externals(self) -> dj.external.ExternalMapping:
        """Return the external mapping for the common_n schema."""
        return dj.external.ExternalMapping(schema=AnalysisNwbfile)

    def _add_externals_to_restr_graph(
        self, restr_graph: RestrGraph, key: dict
    ) -> RestrGraph:
        """Add external tables to a RestrGraph for a given restriction/key.

        Tables added as nodes with restrictions based on file paths. Names
        added to visited set to appear in restr_ft obj passed to SQLDumpHelper.

        This process adds files explicitly listed in the ExportSelection.File
        by the logging process. A separate RestrGraph process, cascade_files, is
        used to track all tables with fk-ref to file tables, and cascade up to
        externals.

        Parameters
        ----------
        restr_graph : RestrGraph
            A RestrGraph object to add external tables to.
        key : dict
            Any valid restriction key for ExportSelection.Table

        Returns
        -------
        restr_graph : RestrGraph
            The updated RestrGraph
        """
        # only add items if found respective file types
        if raw_files := self._list_raw_files(key):
            raw_tbl = self._externals["raw"]
            raw_name = raw_tbl.full_table_name
            raw_restr = "filepath in ('" + "','".join(raw_files) + "')"
            restr_graph.graph.add_node(raw_name, ft=raw_tbl, restr=raw_restr)
            restr_graph.visited.add(raw_name)

        if analysis_files := self._list_analysis_files(key):
            analysis_tbl = self._externals["analysis"]
            analysis_name = analysis_tbl.full_table_name
            # to avoid issues with analysis subdir, we use REGEXP
            # this is slow, but we're only doing this once, and future-proof
            analysis_restr = (
                "filepath REGEXP '" + "|".join(analysis_files) + "'"
            )
            restr_graph.graph.add_node(
                analysis_name, ft=analysis_tbl, restr=analysis_restr
            )
            restr_graph.visited.add(analysis_name)

        return restr_graph

    def get_restr_graph(
        self, key: dict, verbose=False, cascade=True
    ) -> RestrGraph:
        """Return a RestrGraph for a restriction/key's tables/restrictions.

        Ignores duplicate entries.

        Parameters
        ----------
        key : dict
            Any valid restriction key for ExportSelection.Table
        verbose : bool, optional
            Turn on RestrGraph verbosity. Default False.
        cascade : bool, optional
            Propagate restrictions to upstream tables. Default True.
        """
        leaves = unique_dicts(
            (self * self.Table & key).fetch(
                "table_name", "restriction", as_dict=True
            )
        )

        restr_graph = RestrGraph(
            seed_table=self,
            leaves=leaves,
            verbose=verbose,
            cascade=False,
            include_files=True,
        )
        restr_graph = self._add_externals_to_restr_graph(restr_graph, key)

        if cascade:
            restr_graph.cascade()

        return restr_graph

    def preview_tables(self, **kwargs) -> list[dj.FreeTable]:
        """Return a list of restricted FreeTables for a given restriction/key.

        Useful for checking what will be exported.
        """
        kwargs["cascade"] = False
        return self.get_restr_graph(kwargs).leaf_ft

    def show_all_tables(self, **kwargs) -> list[dj.FreeTable]:
        """Return a list of all FreeTables for a given restriction/key.

        Useful for checking what will be exported.
        """
        kwargs["cascade"] = True
        return self.get_restr_graph(kwargs).restr_ft

    def _max_export_id(self, paper_id: str, return_all=False) -> int:
        """Return last export associated with a given paper id.

        Used to populate Export table."""
        if isinstance(paper_id, dict):
            paper_id = paper_id.get("paper_id")
        if not (query := self & {"paper_id": paper_id}):
            return None
        all_export_ids = query.fetch("export_id")
        return all_export_ids if return_all else max(all_export_ids)

    def paper_export_id(self, paper_id: str, return_all=False) -> dict:
        """Return the maximum export_id for a paper, used to populate Export."""
        if not return_all:
            return {"export_id": self._max_export_id(paper_id)}
        return [{"export_id": id} for id in self._max_export_id(paper_id, True)]


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
        """Populate Export for a given paper_id."""
        self.load_shared_schemas()
        if isinstance(paper_id, dict):
            paper_id = paper_id.get("paper_id")
        self.populate(ExportSelection().paper_export_id(paper_id))

    def make(self, key):
        """Populate Export table with the latest export for a given paper."""
        paper_key = (ExportSelection & key).fetch("paper_id", as_dict=True)[0]
        query = ExportSelection & paper_key

        # Null insertion if export_id is not the maximum for the paper
        all_export_ids = ExportSelection()._max_export_id(paper_key, True)
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

        logger.debug(f"Building restr graph for export {key['export_id']}")
        restr_graph = ExportSelection().get_restr_graph(paper_key)

        # Original plus upstream files
        logger.debug("Collecting file paths from export selection")
        file_paths = {
            *query.list_file_paths(paper_key, as_dict=False),
            *restr_graph.file_paths,
        }
        logger.debug(f"Found {len(file_paths)} total files to export")

        # Check for linked nwb objects and add them to the export
        unlinked_files = set()
        for file in file_paths:
            if not (links := get_linked_nwbs(file)):
                unlinked_files.add(file)
                continue
            logger.warning(
                "Dandi not yet supported for linked nwb objects "
                + f"excluding {file} from export "
                + f" and including {links} instead"
            )
            unlinked_files.update(links)
        file_paths = unlinked_files

        table_count = len(restr_graph.as_dict)
        logger.debug(f"Preparing {table_count} table entries for export")
        table_inserts = [
            {**key, **rd, "table_id": i}
            for i, rd in enumerate(restr_graph.as_dict)
        ]

        file_count = len(file_paths)
        logger.debug(f"Preparing {file_count} file entries for export")
        file_inserts = [
            {**key, "file_path": fp, "file_id": i}
            for i, fp in enumerate(file_paths)
        ]

        version_ids = query.fetch("spyglass_version")
        if len(set(version_ids)) > 1:
            raise ValueError(
                "Multiple versions in ExportSelection\n"
                + "Please rerun all analyses with the same version"
            )
        self.compare_versions(
            version_ids[0],
            msg="Must use same Spyglass version for analysis and export",
        )

        logger.debug("Writing MySQL dump for export")
        sql_helper = SQLDumpHelper(**paper_key, spyglass_version=version_ids[0])
        sql_helper.write_mysqldump(free_tables=restr_graph.restr_ft)

        logger.debug("Inserting export metadata into database")
        self.insert1({**key, **paper_key})
        self.Table().insert(table_inserts)
        self.File().insert(file_inserts)

        logger.info(
            f"Export {key['export_id']} completed successfully: "
            f"{table_count} tables, {file_count} files"
        )

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

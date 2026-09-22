"""A schema to store the usage of advanced Spyglass features.

Records show usage of features such as cautious delete and fault-permitting
insert, which will be used to
determine which features are used, how often, and by whom. This will help
plan future development of Spyglass.
"""

from functools import partial
from multiprocessing import Pool, cpu_count
from typing import List, Union

import datajoint as dj
from datajoint.condition import make_condition
from pynwb import NWBHDF5IO
from tqdm import tqdm

from spyglass.common.common_nwbfile import AnalysisNwbfile, Nwbfile
from spyglass.settings import debug_mode, test_mode
from spyglass.utils import SpyglassMixin, SpyglassMixinPart, logger
from spyglass.utils.dandi_file_updates import update_analysis_for_dandi_standard
from spyglass.utils.dj_graph import RestrGraph
from spyglass.utils.dj_helper_fn import make_file_obj_id_unique
from spyglass.utils.nwb_helper_fn import get_linked_nwbs
from spyglass.utils.sql_helper_fn import SQLDumpHelper

schema = dj.schema("common_usage")

_warned_functions: set = set()


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
class IngestionPlanLog(SpyglassMixin, dj.Manual):
    """A file's ingestion plan, staged at entry granularity.

    A staging area for *incomplete* ingestion, never a second source of
    truth. One live plan per file, updated in place across attempts. On full
    success every entry is migrated to its real table and its blob cleared,
    leaving the hashes behind as provenance.

    `nwb_hash` is recorded for provenance only and is never a validity gate.
    The expected loop is ingest → read the report → edit the file → retry, so
    the file hash differs on every attempt by construction; gating freshness
    on it would discard the whole plan exactly when it is most useful.
    Validity is per entry, via the read-set object digests.
    """

    definition = """
    nwb_file_name: varchar(64)
    ---
    verdict: varchar(16)                 # no_op|all_new|partial_new|conflict
    status = "open": enum("open", "complete")
    attempt = 1: int                     # how many times this file was planned
    nwb_hash = NULL: varchar(32)         # provenance only, never a gate
    spyglass_version = NULL: varchar(32)
    dj_user: varchar(64)
    timestamp = CURRENT_TIMESTAMP: timestamp
    """

    class Entry(SpyglassMixinPart):
        """One prospective row, with the two hashes that classify it.

        `key_hash` is the entry's identity, so a re-plan updates the row it
        already has rather than appending a duplicate -- which is what lets
        a second attempt stage N+M where the first staged N.

        `blob_hash` covers the whole serialized entry, so change detection
        needs no byte-comparison of arrays, datetimes or nested dicts.

        Two hashes because they answer different questions. Same key, same
        blob: already staged, unchanged. Same key, different blob:
        divergence, not novelty. One hash cannot tell those apart.

        `table_name` is a plain string, not a foreign key: prospective
        entries routinely name tables whose rows do not exist yet.
        """

        definition = """
        -> master
        table_name: varchar(128)
        key_hash: varchar(32)            # of the primary key: stable identity
        ---
        state: enum("planned","blocked","failed","exists","conflict","inserted")
        blob_hash = NULL: varchar(32)    # of the whole entry: change detection
        entry_blob = NULL: longblob      # cleared once migrated
        problem_code = NULL: varchar(64)
        message = NULL: varchar(255)
        """

    class Problem(SpyglassMixinPart):
        """A file-level problem, belonging to no single entry."""

        definition = """
        -> master
        problem_id: int
        ---
        table_name = "": varchar(128)
        severity: varchar(16)
        code: varchar(64)
        message = "": varchar(255)
        suggested_revision = NULL: blob
        error_raw = NULL: blob
        """

    # Per entry, and transient: blobs are cleared on success. Sized against
    # the fattest single row seen in production (IntervalList.valid_times,
    # 50 KB max), with headroom. Over the cap an entry is staged as hashes
    # and a problem only, and marked so it is re-parsed rather than trusted:
    # silent degradation would read as a cache hit.
    _entry_blob_cap = 1 << 20  # 1 MiB

    def stage(self, plan) -> dict:
        """Record a plan, updating the entries it already holds.

        Parameters
        ----------
        plan : IngestionPlan
            The dataclass from `spyglass.data_import.ingestion_plan`, as
            returned by `plan_nwbfile`.

        Returns
        -------
        dict
            The master key of the staged plan.
        """
        from spyglass.data_import.ingestion_plan import BLOCKING

        master_key = {"nwb_file_name": plan.nwb_file_name}
        existing = self & master_key
        attempt = (existing.fetch1("attempt") + 1) if existing else 1

        master = dict(
            master_key,
            verdict=plan.verdict,
            status="open",
            attempt=attempt,
            nwb_hash=plan.nwb_hash,
            spyglass_version=plan.spyglass_version,
            dj_user=dj.config["database.user"],
        )

        rows, problems = [], []
        for table_plan in plan.table_plans:
            # A table that could not be parsed, or was skipped because a
            # parent failed, stages its entries in that state rather than as
            # ready-to-insert. `exists` and `conflict` are per-entry
            # judgements the plan does not carry at entry granularity; they
            # are set by the insert pass, which checks each row anyway.
            state = {"failed": "failed", "skipped": "blocked"}.get(
                table_plan.status, "planned"
            )
            for target, entries in table_plan.entries:
                name = getattr(target, "full_table_name", str(target))
                for entry in entries:
                    rows.append(
                        self._entry_row(master_key, target, name, entry, state)
                    )

            for problem in table_plan.problems:
                problems.append(problem)
        problems.extend(plan.fatal)

        problem_rows = [
            dict(
                master_key,
                problem_id=index,
                table_name=problem.table or "",
                severity=problem.severity,
                code=problem.code,
                message=(problem.message or "")[:255],
                suggested_revision=problem.suggested_revision,
                error_raw=getattr(problem, "traceback", None),
            )
            for index, problem in enumerate(problems)
        ]

        with self._safe_context():
            # Replace rather than append: an entry keeps its identity across
            # attempts, so re-planning updates what is already staged.
            (self.Entry & master_key).delete_quick()
            (self.Problem & master_key).delete_quick()
            existing.delete_quick()
            self.insert1(master)
            self.Entry.insert(rows)
            self.Problem.insert(problem_rows)

        blocked = sum(1 for p in problems if p.severity in BLOCKING)
        logger.info(
            f"Staged plan for {plan.nwb_file_name}: attempt {attempt}, "
            + f"{len(rows)} entries, {blocked} blocking problems"
        )
        return master_key

    def mark_inserted(
        self, plan, inserted, existing=(), complete: bool = False
    ) -> None:
        """Record which staged entries made it into their real tables.

        Clears `entry_blob` for those entries: keeping a payload for a row
        that now exists in its own table would make the log a second copy of
        the data, which is the failure this design exists to avoid. The
        hashes stay, as provenance.

        Parameters
        ----------
        plan : IngestionPlan
            The plan that was inserted.
        inserted : list of (str, table, rows)
            What `insert_plan` actually wrote.
        existing : list of (str, table, rows), optional
            What it found already stored and so did not write. These are
            recorded as `exists` and lose their payload too: the invariant
            is that no blob is kept for an entry present in its own table,
            and an entry that was skipped is no less present than one just
            written.
        complete : bool, optional
            Whether every entry is now stored, closing the plan. Default
            False.
        """
        from datajoint.hash import key_hash

        from spyglass.data_import.ingestion_plan import row_key

        master_key = {"nwb_file_name": plan.nwb_file_name}
        if not (self & master_key):  # never staged; nothing to record
            return

        def identify(table_name, target, rows):
            for row in rows:
                try:
                    identity = key_hash(dict(row_key(target, row)))
                except Exception:
                    identity = key_hash(row)
                yield {
                    **master_key,
                    "table_name": table_name,
                    "key_hash": identity,
                }

        done = [
            (entry_key, state)
            for state, group in (("inserted", inserted), ("exists", existing))
            for table_name, target, rows in group
            for entry_key in identify(table_name, target, rows)
        ]

        # One transaction with the state change: a crash between the data
        # write and this would leave a stale blob claiming work already done.
        with self._safe_context():
            for entry_key, state in done:
                if not (self.Entry & entry_key):
                    continue  # not staged, nothing to migrate
                # update1 on the table itself: DataJoint refuses it on a
                # restricted query, and the key is already complete.
                self.Entry.update1(
                    {**entry_key, "state": state, "entry_blob": None}
                )
            if complete:
                self.update1({**master_key, "status": "complete"})

        migrated = sum(1 for _, state in done if state == "inserted")
        logger.info(
            f"{plan.nwb_file_name}: {migrated} entries migrated, "
            + f"{len(done) - migrated} already stored"
            + (", plan complete" if complete else "")
        )

    def _entry_row(
        self,
        master_key: dict,
        target,
        table_name: str,
        entry: dict,
        state: str = "planned",
    ):
        """Build one staged row, hashing its key and its whole payload.

        The two hashes must cover different things or the pair is useless:
        `key_hash` over the primary key alone is what makes an entry
        re-identifiable across attempts, and `blob_hash` over the whole entry
        is what separates "already staged, unchanged" from "staged, and the
        plan now disagrees with it".

        `target` is the live table, so its primary key is known here without
        resolving `table_name` back to a class. A plan rebuilt from storage
        carries name-only targets; those fall back to hashing the whole
        entry, which still detects change but cannot tell divergence from
        novelty.

        Parameters
        ----------
        master_key : dict
            The plan this entry belongs to.
        target : dj.Table
            The table the entry is destined for.
        table_name : str
            Its full table name, as stored.
        entry : dict
            The prospective row.
        state : str, optional
            How this entry stands, from its table's plan. Default
            `planned`.
        """
        from datajoint.hash import key_hash

        from spyglass.data_import.ingestion_plan import entry_digest, row_key

        try:
            # DataJoint's own key hash, as JobTable uses for the same job.
            # Sound here because primary keys are scalars, where `str` is
            # faithful, and `row_key` has already coerced them to the
            # column's declared type.
            primary = key_hash(dict(row_key(target, entry)))
        except Exception:  # name-only target, from a rebuilt plan
            primary = key_hash(entry)

        blob = dict(entry)
        row = dict(
            master_key,
            table_name=table_name,
            key_hash=primary,
            # Not key_hash: it would abbreviate a large array and miss an
            # edit inside it. See entry_digest.
            blob_hash=entry_digest(blob),
            state=state,
        )

        packed = dj.blob.pack(blob)
        if len(packed) > self._entry_blob_cap:
            # Hashes and a problem only. Not stageable, so a later attempt
            # re-parses it rather than trusting a payload we did not keep.
            logger.warning(
                f"{table_name}: entry of {len(packed)} bytes exceeds the "
                + f"{self._entry_blob_cap} byte staging cap; storing hashes "
                + "only, it will be re-parsed"
            )
            return dict(
                row,
                state="failed",
                problem_code="entry_too_large",
                message=f"{len(packed)} bytes over the staging cap",
            )

        return dict(row, entry_blob=blob)


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
    def deprecate_log(cls, name, alt=None, warning=True, doc=None) -> None:
        """Log a deprecation warning for a feature.

        Parameters
        ----------
        name : str
            The name of the feature to deprecate.
        alt : str, optional
            Exact replacement call to display. Default no such message.
        warning : bool, optional
            Whether to log a warning. Default is True.
        doc : str, optional
            URL of the migration guide. Default no such message.
        """
        if warning and name not in _warned_functions:
            _warned_functions.add(name)
            msg = f"DEPRECATION scheduled for Spyglass 0.7.0: {name}"
            if alt:
                msg += f"\n\tUse instead: {alt}"
            if doc:
                msg += f"\n\tMigration guide: {doc}"
            logger.warning(msg)
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
        self._info_msg(f"{status} {export_key}")
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

    def _list_raw_files(
        self, key: dict, included_nwb_files: list[str] = None
    ) -> list[str]:
        """Return a list of unique nwb file names for a given restriction/key.

        If included_nwb_files is provided, only returns raw files
        that are in that list.

        Parameters
        ----------
        key : dict
            Any valid restriction key for ExportSelection.Table
        included_nwb_files : list, optional
            A whitelist of nwb files to include in the export. Default None applies
            no whitelist restriction.

        Returns
        -------
        list[str]
            List of unique nwb file names.
        """
        file_table = self * self.File & key
        files = list(
            {
                *AnalysisNwbfile.join(file_table, log_export=False).fetch(
                    "nwb_file_name"
                )
            }
        )
        if included_nwb_files is None:
            return files
        return [x for x in files if x in included_nwb_files]

    def _list_analysis_files(
        self, key: dict, included_nwb_files: list[str] = None
    ) -> list[str]:
        """Return a list of unique analysis file names for a given restriction/key.
        If included_nwb_files is provided, only returns analysis files
        that are derivatives of those raw files.

        Parameters
        ----------
        key : dict
            Any valid restriction key for ExportSelection.Table
        included_nwb_files : list, optional
            A whitelist of nwb files to include in the export. Default None applies
            no whitelist restriction.
        Returns
        -------
        list[str]
            List of unique analysis file names.

        """
        file_table = self * self.File & key
        files = list(file_table.fetch("analysis_file_name"))
        if included_nwb_files is None:
            return files
        return [
            x
            for x in files
            if any(
                [
                    nwb_file_name.split("_.nwb")[0] in x
                    for nwb_file_name in included_nwb_files
                ]
            )
        ]

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
        self,
        restr_graph: RestrGraph,
        key: dict,
        raw_files=None,
        analysis_files=None,
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
        raw_files : list, optional
            A list of raw nwb file names to add. Default None, which retrieves
            from ExportSelection._list_raw_files.
        analysis_files : list, optional
            A list of analysis nwb file names to add. Default None, which retrieves
            from ExportSelection._list_analysis_files.

        Returns
        -------
        restr_graph : RestrGraph
            The updated RestrGraph
        """
        if raw_files is None:
            raw_files = self._list_raw_files(key)
        if analysis_files is None:
            analysis_files = self._list_analysis_files(key)

        # only add items if found respective file types
        if raw_files:
            raw_tbl = self._externals["raw"]
            raw_name = raw_tbl.full_table_name
            raw_restr = "filepath in ('" + "','".join(raw_files) + "')"
            restr_graph.graph.add_node(raw_name, ft=raw_tbl, restr=raw_restr)
            restr_graph.visited.add(raw_name)

        if analysis_files:
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
        self, key: dict, verbose=False, cascade=True, included_nwb_files=None
    ) -> RestrGraph:
        """Return a RestrGraph for a restriction/key's tables/restrictions.

        Restriction graph limits to entries stemming from the raw nwb_files
        listed in included_nwb_files, if provided.

        Ignores duplicate entries.

        Parameters
        ----------
        key : dict
            Any valid restriction key for ExportSelection.Table
        verbose : bool, optional
            Turn on RestrGraph verbosity. Default False.
        cascade : bool, optional
            Propagate restrictions to upstream tables. Default True.
        included_nwb_files : list, optional
            A whitelist of nwb files to include in the export. Default None applies
            no whitelist restriction.
        """
        selection_tables = self * self.Table & key
        tracked_tables = set(selection_tables.fetch("table_name"))
        leaves = []
        # Condense to single restriction per table (OR of all restrictions).
        # Large performance boost for large exports with many logged entries
        for table_name in tracked_tables:
            restr_list = (selection_tables & dict(table_name=table_name)).fetch(
                "restriction"
            )
            restriction = make_condition(
                dj.FreeTable(dj.conn(), table_name), restr_list, set()
            )
            leaves.append(
                {"table_name": table_name, "restriction": restriction}
            )

        restr_graph = RestrGraph(
            seed_table=self,
            leaves=leaves,
            verbose=verbose,
            cascade=False,
            include_files=True,
        )

        if included_nwb_files is None:
            restr_graph = self._add_externals_to_restr_graph(restr_graph, key)
            if cascade:
                restr_graph.cascade()
            return restr_graph

        # Restrict the graph to only include entries stemming from the
        # included nwb files
        logger.debug("Generating restriction graph of included nwb files")
        nwb_restr = make_condition(
            Nwbfile(),
            [f"nwb_file_name = '{f}'" for f in included_nwb_files],
            set(),
        )
        whitelist_graph = RestrGraph(
            seed_table=Nwbfile,
            leaves={
                "table_name": Nwbfile.full_table_name,
                "restriction": nwb_restr,
            },
            verbose=verbose,
            cascade=True,
            include_files=True,
            direction="down",
        )
        logger.debug("Intersecting with export restriction graph")
        restr_graph = restr_graph & whitelist_graph
        raw_files_to_add = self._list_raw_files(key, included_nwb_files)
        analysis_files_to_add = self._list_analysis_files(
            key, included_nwb_files
        )
        restr_graph = self._add_externals_to_restr_graph(
            restr_graph,
            key,
            raw_files=raw_files_to_add,
            analysis_files=analysis_files_to_add,
        )

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
    included_nwb_file_names = null: mediumblob   # list of nwb files included in export
    """

    _nwb_whitelist_paper_cache = dict()
    _n_file_link_processes = 1

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

    def populate_paper(
        self,
        paper_id: Union[str, dict],
        included_nwb_files=None,
        n_processes=1,
    ):
        """Populate Export for a given paper_id.

        Parameters
        ----------
        paper_id : str or dict
            The paper_id to populate Export for. If dict, must contain key "paper_id".
        included_nwb_files : list, optional
            A whitelist of nwb files to include in the export. Default None applies
            no whitelist restriction.
        n_processes : int, optional
            The number of processes to use for checking linked nwb files.
            Default 1 (no multiprocessing).
        """
        self.load_shared_schemas()
        if isinstance(paper_id, dict):
            paper_id = paper_id.get("paper_id")

        self._nwb_whitelist_paper_cache[paper_id] = included_nwb_files
        if n_processes < 1:
            n_processes = 1
        elif n_processes > cpu_count():
            n_processes = cpu_count()
        self._n_file_link_processes = n_processes

        self.populate(
            {
                **ExportSelection().paper_export_id(paper_id),
            }
        )

    def make(self, key):
        """Populate Export table with the latest export for a given paper."""
        logger.debug(f"Populating Export for {key}")
        paper_key = (ExportSelection & key).fetch("paper_id", as_dict=True)[0]
        paper_id = paper_key["paper_id"]
        query = ExportSelection & paper_key

        included_nwb_files = self._nwb_whitelist_paper_cache.get(paper_id, None)

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

        logger.debug(f"Building restr graph for {key['export_id']}")
        restr_graph = ExportSelection().get_restr_graph(
            paper_key, included_nwb_files=included_nwb_files, verbose=debug_mode
        )
        # Original plus upstream files
        logger.debug("Collecting file paths from export selection")
        file_paths = {
            *query.list_file_paths(paper_key, as_dict=False),
            *restr_graph.file_paths,
        }
        logger.debug(f"Found {len(file_paths)} total files to export")
        if included_nwb_files:
            # Limit to derivatives of the included nwb files
            file_paths = {
                f
                for f in file_paths
                if any(
                    [
                        nwb_file_name.split("_.nwb")[0] in f
                        for nwb_file_name in included_nwb_files
                    ]
                )
            }

        unlinked_files = set()
        if self._n_file_link_processes == 1:
            for file in tqdm(
                file_paths,
                desc="Checking linked nwb files",
                disable=test_mode,
            ):
                unlinked_files.update(get_unlinked_files(file))
        else:
            with Pool(processes=self._n_file_link_processes) as pool:
                results = list(
                    tqdm(
                        pool.map(get_unlinked_files, file_paths),
                        total=len(file_paths),
                        desc="Checking linked nwb files",
                    )
                )
            for files in results:
                unlinked_files.update(files)
        file_paths = unlinked_files

        restr_graph.enforce_restr_strings()  # ensure all restr are strings

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

    def prepare_files_for_export(self, key, n_processes=1, **kwargs):
        """Resolve common known errors to make a set of analysis
        files dandi compliant

        Parameters
        ----------
        key : dict
            restriction for a single entry of the Export table
        """
        key = (self & key).fetch1("KEY")
        file_list = (self.File() & key).fetch("file_path")

        if n_processes == 1:
            self._make_fileset_ids_unique(key)
            for file in file_list:
                update_analysis_for_dandi_standard(file, **kwargs)
            return
        with Pool(processes=n_processes) as pool:
            list(pool.imap_unordered(make_file_obj_id_unique, file_list))
            update_fn = partial(update_analysis_for_dandi_standard, **kwargs)
            list(pool.imap_unordered(update_fn, file_list))

    def _make_fileset_ids_unique(self, key, n_processes=1):
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


def get_unlinked_files(file_path):
    if not (links := get_linked_nwbs(file_path)):
        return {file_path}
    logger.warning(
        "Dandi not yet supported for linked nwb objects "
        + f"excluding {file_path} from export "
        + f" and including {links} instead"
    )
    return set(links)


error_schema = dj.schema("common_export_error_log")


@error_schema
class ExportErrorLog(dj.Manual):
    definition = """
    file: varchar(255)  # file being processed
    source: varchar(255)  # source of the error (e.g., table name or function)
    ---
    """

    @staticmethod
    def _logger_warning(key):
        logger.warning(
            f"Logging export error for file: {key.get('file', 'unknown')}"
            + f" from source: {key.get('source', 'unknown')}"
        )

    def insert1(self, key, **kwargs):
        """Insert a new entry into the ExportErrorLog table.

        Parameters
        ----------
        key : dict
            Dictionary containing the primary key fields for the table.
        **kwargs : dict
            Additional keyword arguments for non-primary key fields.
        """
        self._logger_warning(key)
        super().insert1(key, **kwargs)

from atexit import register as exit_register
from atexit import unregister as exit_unregister
from contextlib import nullcontext
from functools import cached_property
from inspect import stack as inspect_stack
from os import environ
from time import time
from typing import Dict, List, Union

import datajoint as dj
from datajoint.condition import make_condition
from datajoint.errors import DataJointError
from datajoint.expression import QueryExpression
from datajoint.logging import logger as dj_logger
from datajoint.table import Table
from datajoint.utils import get_master, to_camel_case, user_choice
from networkx import NetworkXError
from pymysql.err import DataError

from spyglass.utils.database_settings import SHARED_MODULES
from spyglass.utils.dj_helper_fn import (  # NonDaemonPool,
    NonDaemonPool,
    fetch_nwb,
    get_nwb_table,
    populate_pass_function,
)
from spyglass.utils.dj_merge_tables import Merge, is_merge_table
from spyglass.utils.logging import logger

try:
    import pynapple  # noqa F401
except ImportError:
    pynapple = None

EXPORT_ENV_VAR = "SPYGLASS_EXPORT_ID"


class SpyglassMixin:
    """Mixin for Spyglass DataJoint tables.

    Provides methods for fetching NWBFile objects and checking user permission
    prior to deleting. As a mixin class, all Spyglass tables can inherit custom
    methods from a central location.

    Methods
    -------
    fetch_nwb(*attrs, **kwargs)
        Fetch NWBFile object from relevant table. Uses either a foreign key to
        a NWBFile table (including AnalysisNwbfile) or a _nwb_table attribute to
        determine which table to use.
    delte_downstream_merge(restriction=None, dry_run=True, reload_cache=False)
        Delete downstream merge table entries associated with restriction.
        Requires caching of merge tables and links, which is slow on first call.
        `restriction` can be set to a string to restrict the delete. `dry_run`
        can be set to False to commit the delete. `reload_cache` can be set to
        True to reload the merge cache.
    ddp(*args, **kwargs)
        Alias for delete_downstream_parts
    cautious_delete(force_permission=False, *args, **kwargs)
        Check user permissions before deleting table rows. Permission is granted
        to users listed as admin in LabMember table or to users on a team with
        with the Session experimenter(s). If the table where the delete is
        executed cannot be linked to a Session, a warning is logged and the
        delete continues. If the Session has no experimenter, or if the user is
        not on a team with the Session experimenter(s), a PermissionError is
        raised. `force_permission` can be set to True to bypass permission check.
    cdel(*args, **kwargs)
        Alias for cautious_delete.
    """

    # _nwb_table = None # NWBFile table class, defined at the table level

    # pks for delete permission check, assumed to be one field for each
    _session_pk = None  # Session primary key. Mixin is ambivalent to Session pk
    _member_pk = None  # LabMember primary key. Mixin ambivalent table structure

    _banned_search_tables = set()  # Tables to avoid in restrict_by
    _parallel_make = False  # Tables that use parallel processing in make

    def __init__(self, *args, **kwargs):
        """Initialize SpyglassMixin.

        Checks that schema prefix is in SHARED_MODULES.
        """
        if self.is_declared:
            return
        if self.database and self.database.split("_")[0] not in [
            *SHARED_MODULES,
            dj.config["database.user"],
            "temp",
            "test",
        ]:
            logger.error(
                f"Schema prefix not in SHARED_MODULES: {self.database}"
            )
        if is_merge_table(self) and not isinstance(self, Merge):
            raise TypeError(
                "Table definition matches Merge but does not inherit class: "
                + self.full_table_name
            )

    # -------------------------- Misc helper methods --------------------------

    @property
    def camel_name(self):
        """Return table name in camel case."""
        return to_camel_case(self.table_name)

    def _auto_increment(self, key, pk, *args, **kwargs):
        """Auto-increment primary key."""
        if not key.get(pk):
            key[pk] = (dj.U().aggr(self, n=f"max({pk})").fetch1("n") or 0) + 1
        return key

    def file_like(self, name=None, **kwargs):
        """Convenience method for wildcard search on file name fields."""
        if not name:
            return self
        attr = None
        for field in self.heading.names:
            if "file" in field:
                attr = field
                break
        if not attr:
            logger.error(f"No file_like field found in {self.full_table_name}")
            return
        return self & f"{attr} LIKE '%{name}%'"

    def find_insert_fail(self, key):
        """Find which parent table is causing an IntergrityError on insert."""
        for parent in self.parents(as_objects=True):
            parent_key = {
                k: v for k, v in key.items() if k in parent.heading.names
            }
            parent_name = to_camel_case(parent.table_name)
            if query := parent & parent_key:
                logger.info(f"{parent_name}:\n{query}")
            else:
                logger.info(f"{parent_name}: MISSING")

    @classmethod
    def _safe_context(cls):
        """Return transaction if not already in one."""
        return (
            cls.connection.transaction
            if not cls.connection.in_transaction
            else nullcontext()
        )

    # ------------------------------- fetch_nwb -------------------------------

    @cached_property
    def _nwb_table_tuple(self) -> tuple:
        """NWBFile table class.

        Used to determine fetch_nwb behavior. Also used in Merge.fetch_nwb.
        Implemented as a cached_property to avoid circular imports."""
        from spyglass.common.common_nwbfile import (
            AnalysisNwbfile,
            Nwbfile,
        )  # noqa F401

        table_dict = {
            AnalysisNwbfile: "analysis_file_abs_path",
            Nwbfile: "nwb_file_abs_path",
        }

        resolved = getattr(self, "_nwb_table", None) or (
            AnalysisNwbfile
            if "-> AnalysisNwbfile" in self.definition
            else Nwbfile if "-> Nwbfile" in self.definition else None
        )

        if not resolved:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not have a "
                "(Analysis)Nwbfile foreign key or _nwb_table attribute."
            )

        return (
            resolved,
            table_dict[resolved],
        )

    def fetch_nwb(self, *attrs, **kwargs):
        """Fetch NWBFile object from relevant table.

        Implementing class must have a foreign key reference to Nwbfile or
        AnalysisNwbfile (i.e., "-> (Analysis)Nwbfile" in definition)
        or a _nwb_table attribute. If both are present, the attribute takes
        precedence.

        Additional logic support Export table logging.
        """
        table, tbl_attr = self._nwb_table_tuple

        if self.export_id and "analysis" in tbl_attr:
            tbl_pk = "analysis_file_name"
            fnames = (self * table).fetch(tbl_pk)
            logger.debug(
                f"Export {self.export_id}: fetch_nwb {self.table_name}, {fnames}"
            )
            self._export_table.File.insert(
                [
                    {"export_id": self.export_id, tbl_pk: fname}
                    for fname in fnames
                ],
                skip_duplicates=True,
            )
            self._export_table.Table.insert1(
                dict(
                    export_id=self.export_id,
                    table_name=self.full_table_name,
                    restriction=make_condition(self, self.restriction, set()),
                ),
                skip_duplicates=True,
            )

        return fetch_nwb(self, self._nwb_table_tuple, *attrs, **kwargs)

    def fetch_pynapple(self, *attrs, **kwargs):
        """Get a pynapple object from the given DataJoint query.

        Parameters
        ----------
        *attrs : list
            Attributes from normal DataJoint fetch call.
        **kwargs : dict
            Keyword arguments from normal DataJoint fetch call.

        Returns
        -------
        pynapple_objects : list of pynapple objects
            List of dicts containing pynapple objects.

        Raises
        ------
        ImportError
            If pynapple is not installed.

        """
        if pynapple is None:
            raise ImportError("Pynapple is not installed.")

        nwb_files, file_path_fn = get_nwb_table(
            self,
            self._nwb_table_tuple[0],
            self._nwb_table_tuple[1],
            *attrs,
            **kwargs,
        )

        return [
            pynapple.load_file(file_path_fn(file_name))
            for file_name in nwb_files
        ]

    # ------------------------ delete_downstream_parts ------------------------

    def _import_part_masters(self):
        """Import tables that may constrain a RestrGraph. See #1002"""
        from spyglass.common.common_ripple import (
            RippleLFPSelection,
        )  # noqa F401
        from spyglass.decoding.decoding_merge import DecodingOutput  # noqa F401
        from spyglass.decoding.v0.clusterless import (  # noqa F401
            UnitMarksIndicatorSelection,
        )
        from spyglass.decoding.v0.sorted_spikes import (  # noqa F401
            SortedSpikesIndicatorSelection,
        )
        from spyglass.decoding.v1.core import PositionGroup  # noqa F401
        from spyglass.lfp.analysis.v1 import LFPBandSelection  # noqa F401
        from spyglass.lfp.lfp_merge import LFPOutput  # noqa F401
        from spyglass.linearization.merge import (  # noqa F401
            LinearizedPositionOutput,
            LinearizedPositionV1,
        )
        from spyglass.mua.v1.mua import MuaEventsV1  # noqa F401
        from spyglass.position.position_merge import PositionOutput  # noqa F401
        from spyglass.ripple.v1.ripple import RippleTimesV1  # noqa F401
        from spyglass.spikesorting.analysis.v1.group import (  # noqa F401
            SortedSpikesGroup,
        )
        from spyglass.spikesorting.spikesorting_merge import (  # noqa F401
            SpikeSortingOutput,
        )
        from spyglass.spikesorting.v0.figurl_views import (  # noqa F401
            SpikeSortingRecordingView,
        )

        _ = (
            DecodingOutput(),
            LFPBandSelection(),
            LFPOutput(),
            LinearizedPositionOutput(),
            LinearizedPositionV1(),
            MuaEventsV1(),
            PositionGroup(),
            PositionOutput(),
            RippleLFPSelection(),
            RippleTimesV1(),
            SortedSpikesGroup(),
            SortedSpikesIndicatorSelection(),
            SpikeSortingOutput(),
            SpikeSortingRecordingView(),
            UnitMarksIndicatorSelection(),
        )

    @cached_property
    def _part_masters(self) -> set:
        """Set of master tables downstream of self.

        Cache of masters in self.descendants(as_objects=True) with another
        foreign key reference in the part. Used for delete_downstream_parts.
        """
        self.connection.dependencies.load()
        part_masters = set()

        def search_descendants(parent):
            for desc in parent.descendants(as_objects=True):
                if (  # Check if has master, is part
                    not (master := get_master(desc.full_table_name))
                    # has other non-master parent
                    or not set(desc.parents()) - set([master])
                    or master in part_masters  # already in cache
                ):
                    continue
                if master not in part_masters:
                    part_masters.add(master)
                    search_descendants(dj.FreeTable(self.connection, master))

        try:
            _ = search_descendants(self)
        except NetworkXError:
            try:  # Attempt to import missing table
                self._import_part_masters()
                _ = search_descendants(self)
            except NetworkXError as e:
                table_name = "".join(e.args[0].split("`")[1:4])
                raise ValueError(f"Please import {table_name} and try again.")

        logger.info(
            f"Building part-parent cache for {self.camel_name}.\n\t"
            + f"Found {len(part_masters)} downstream part tables"
        )

        return part_masters

    def _commit_downstream_delete(self, down_fts, start=None, **kwargs):
        """
        Commit delete of downstream parts via down_fts. Logs with _log_delete.

        Used by both delete_downstream_parts and cautious_delete.
        """
        start = start or time()

        safemode = (
            dj.config.get("safemode", True)
            if kwargs.get("safemode") is None
            else kwargs["safemode"]
        )
        _ = kwargs.pop("safemode", None)

        ran_deletes = True
        if down_fts:
            for down_ft in down_fts:
                dj_logger.info(
                    f"Spyglass: Deleting {len(down_ft)} rows from "
                    + f"{down_ft.full_table_name}"
                )
            if (
                self._test_mode
                or not safemode
                or user_choice("Commit deletes?", default="no") == "yes"
            ):
                for down_ft in down_fts:  # safemode off b/c already checked
                    down_ft.delete(safemode=False, **kwargs)
            else:
                logger.info("Delete aborted.")
                ran_deletes = False

        self._log_delete(start, del_blob=down_fts if ran_deletes else None)

        return ran_deletes

    def delete_downstream_parts(
        self,
        restriction: str = None,
        dry_run: bool = True,
        reload_cache: bool = False,
        disable_warning: bool = False,
        return_graph: bool = False,
        verbose: bool = False,
        **kwargs,
    ) -> List[dj.FreeTable]:
        """Delete downstream merge table entries associated with restriction.

        Requires caching of merge tables and links, which is slow on first call.

        Parameters
        ----------
        restriction : str, optional
            Restriction to apply to merge tables. Default None. Will attempt to
            use table restriction if None.
        dry_run : bool, optional
            If True, return list of merge part entries to be deleted. Default
            True.
        reload_cache : bool, optional
            If True, reload merge cache. Default False.
        disable_warning : bool, optional
            If True, do not warn if no merge tables found. Default False.
        return_graph: bool, optional
            If True, return RestrGraph object used to identify downstream
            tables. Default False, return list of part FreeTables.
            True. If False, return dictionary of merge tables and their joins.
        verbose : bool, optional
            If True, call RestrGraph with verbose=True. Default False.
        **kwargs : Any
            Passed to datajoint.table.Table.delete.
        """
        from spyglass.utils.dj_graph import RestrGraph  # noqa F401

        start = time()

        if reload_cache:
            _ = self.__dict__.pop("_part_masters", None)

        _ = self._part_masters  # load cache before loading graph
        restriction = restriction or self.restriction or True

        restr_graph = RestrGraph(
            seed_table=self,
            leaves={self.full_table_name: restriction},
            direction="down",
            cascade=True,
            verbose=verbose,
        )

        if return_graph:
            return restr_graph

        down_fts = restr_graph.ft_from_list(
            self._part_masters, sort_reverse=False
        )

        if not down_fts and not disable_warning:
            logger.warning(
                f"No part deletes found w/ {self.camel_name} & "
                + f"{restriction}.\n\tIf this is unexpected, try importing "
                + " Merge table(s) and running with `reload_cache`."
            )

        if dry_run:
            return down_fts

        self._commit_downstream_delete(down_fts, start, **kwargs)

    def ddp(
        self, *args, **kwargs
    ) -> Union[List[QueryExpression], Dict[str, List[QueryExpression]]]:
        """Alias for delete_downstream_parts."""
        return self.delete_downstream_parts(*args, **kwargs)

    # ---------------------------- cautious_delete ----------------------------

    @cached_property
    def _delete_deps(self) -> List[Table]:
        """List of tables required for delete permission and orphan checks.

        LabMember, LabTeam, and Session are required for delete permission.
        common_nwbfile.schema.external is required for deleting orphaned
        external files. IntervalList is required for deleting orphaned interval
        lists.

        Used to delay import of tables until needed, avoiding circular imports.
        Each of these tables inheits SpyglassMixin.
        """
        from spyglass.common import (  # noqa F401
            IntervalList,
            LabMember,
            LabTeam,
            Session,
        )
        from spyglass.common.common_nwbfile import schema  # noqa F401

        self._session_pk = Session.primary_key[0]
        self._member_pk = LabMember.primary_key[0]
        return [LabMember, LabTeam, Session, schema.external, IntervalList]

    def _get_exp_summary(self):
        """Get summary of experimenters for session(s), including NULL.

        Parameters
        ----------
        sess_link : datajoint.expression.QueryExpression
            Join of table link with Session table.

        Returns
        -------
        str
            Summary of experimenters for session(s).
        """

        Session = self._delete_deps[2]
        SesExp = Session.Experimenter

        # Not called in delete permission check, only bare _get_exp_summary
        if self._member_pk in self.heading.names:
            return self * SesExp

        empty_pk = {self._member_pk: "NULL"}
        format = dj.U(self._session_pk, self._member_pk)

        restr = self.restriction or True
        sess_link = self._session_connection.cascade(restr, direction="up")

        exp_missing = format & (sess_link - SesExp).proj(**empty_pk)
        exp_present = format & (sess_link * SesExp - exp_missing).proj()

        return exp_missing + exp_present

    @cached_property
    def _session_connection(self):
        """Path from Session table to self. False if no connection found."""
        from spyglass.utils.dj_graph import TableChain  # noqa F401

        connection = TableChain(parent=self._delete_deps[2], child=self)
        return connection if connection.has_link else False

    @cached_property
    def _test_mode(self) -> bool:
        """Return True if in test mode.

        Avoids circular import. Prevents prompt on delete."""
        from spyglass.settings import test_mode

        return test_mode

    def _check_delete_permission(self) -> None:
        """Check user name against lab team assoc. w/ self -> Session.

        Returns
        -------
        None
            Permission granted.

        Raises
        ------
        PermissionError
            Permission denied because (a) Session has no experimenter, or (b)
            user is not on a team with Session experimenter(s).
        """
        LabMember, LabTeam, Session, _, _ = self._delete_deps

        dj_user = dj.config["database.user"]
        if dj_user in LabMember().admin:  # bypass permission check for admin
            return

        if (
            not self._session_connection  # Table has no session
            or self._member_pk in self.heading.names  # Table has experimenter
        ):
            logger.warn(  # Permit delete if no session connection
                "Could not find lab team associated with "
                + f"{self.__class__.__name__}."
                + "\nBe careful not to delete others' data."
            )
            return

        sess_summary = self._get_exp_summary()
        experimenters = sess_summary.fetch(self._member_pk)
        if None in experimenters:
            raise PermissionError(
                "Please ensure all Sessions have an experimenter in "
                + f"SessionExperimenter:\n{sess_summary}"
            )

        user_name = LabMember().get_djuser_name(dj_user)
        for experimenter in set(experimenters):
            # Check once with cache, if fails, reload and check again
            # On eval as set, reload will only be called once
            if user_name not in LabTeam().get_team_members(
                experimenter
            ) and user_name not in LabTeam().get_team_members(
                experimenter, reload=True
            ):
                sess_w_exp = sess_summary & {self._member_pk: experimenter}
                raise PermissionError(
                    f"User '{user_name}' is not on a team with '{experimenter}'"
                    + ", an experimenter for session(s):\n"
                    + f"{sess_w_exp}"
                )
        logger.info(f"Queueing delete for session(s):\n{sess_summary}")

    @cached_property
    def _cautious_del_tbl(self):
        """Temporary inclusion for usage tracking."""
        from spyglass.common.common_usage import CautiousDelete

        return CautiousDelete()

    def _log_delete(self, start, del_blob=None, super_delete=False):
        """Log use of cautious_delete."""
        safe_insert = dict(
            duration=time() - start,
            dj_user=dj.config["database.user"],
            origin=self.full_table_name,
        )
        restr_str = "Super delete: " if super_delete else ""
        restr_str += "".join(self.restriction) if self.restriction else "None"
        try:
            self._cautious_del_tbl.insert1(
                dict(
                    **safe_insert,
                    restriction=restr_str[:255],
                    merge_deletes=del_blob,
                )
            )
        except (DataJointError, DataError):
            self._cautious_del_tbl.insert1(
                dict(**safe_insert, restriction="Unknown")
            )

    # TODO: Intercept datajoint delete confirmation prompt for merge deletes
    def cautious_delete(
        self, force_permission: bool = False, dry_run=False, *args, **kwargs
    ):
        """Permission check, then delete potential orphans and table rows.

        Permission is granted to users listed as admin in LabMember table or to
        users on a team with with the Session experimenter(s). If the table
        cannot be linked to Session, a warning is logged and the delete
        continues. If the Session has no experimenter, or if the user is not on
        a team with the Session experimenter(s), a PermissionError is raised.

        Potential downstream orphans are deleted first. These are master tables
        whose parts have foreign keys to descendants of self. Then, rows from
        self are deleted. Last, Nwbfile and IntervalList externals are deleted.

        Parameters
        ----------
        force_permission : bool, optional
            Bypass permission check. Default False.
        dry_run : bool, optional
            Default False. If True, return items to be deleted as
            Tuple[Upstream, Downstream, externals['raw'], externals['analysis']]
            If False, delete items.
        *args, **kwargs : Any
            Passed to datajoint.table.Table.delete.
        """
        start = time()
        external, IntervalList = self._delete_deps[3], self._delete_deps[4]

        if not force_permission or dry_run:
            self._check_delete_permission()

        down_fts = self.delete_downstream_parts(
            dry_run=True,
            disable_warning=True,
        )

        if dry_run:
            return (
                down_fts,
                IntervalList(),  # cleanup func relies on downstream deletes
                external["raw"].unused(),
                external["analysis"].unused(),
            )

        if not self._commit_downstream_delete(down_fts, start=start, **kwargs):
            return  # Abort delete based on user input

        super().delete(*args, **kwargs)  # Confirmation here

        for ext_type in ["raw", "analysis"]:
            external[ext_type].delete(
                delete_external_files=True, display_progress=False
            )

        _ = IntervalList().nightly_cleanup(dry_run=False)

        self._log_delete(start=start, del_blob=down_fts)

    def cdel(self, *args, **kwargs):
        """Alias for cautious_delete."""
        return self.cautious_delete(*args, **kwargs)

    def delete(self, *args, **kwargs):
        """Alias for cautious_delete, overwrites datajoint.table.Table.delete"""
        self.cautious_delete(*args, **kwargs)

    def super_delete(self, warn=True, *args, **kwargs):
        """Alias for datajoint.table.Table.delete."""
        if warn:
            logger.warning("!! Bypassing cautious_delete !!")
            self._log_delete(start=time(), super_delete=True)
        super().delete(*args, **kwargs)

    # -------------------------- non-daemon populate --------------------------
    def populate(self, *restrictions, **kwargs):
        """Populate table in parallel.

        Supersedes datajoint.table.Table.populate for classes with that
        spawn processes in their make function
        """

        # Pass through to super if not parallel in the make function or only a single process
        processes = kwargs.pop("processes", 1)
        if processes == 1 or not self._parallel_make:
            return super().populate(*restrictions, **kwargs)

        # If parallel in both make and populate, use non-daemon processes
        # Get keys to populate
        keys = (self._jobs_to_do(restrictions) - self.target).fetch(
            "KEY", limit=kwargs.get("limit", None)
        )
        # package the call list
        call_list = [(type(self), key, kwargs) for key in keys]

        # Create a pool of non-daemon processes to populate a single entry each
        pool = NonDaemonPool(processes=processes)
        try:
            pool.map(populate_pass_function, call_list)
        except Exception as e:
            raise e
        finally:
            pool.close()
            pool.terminate()

    # ------------------------------- Export Log -------------------------------

    @cached_property
    def _spyglass_version(self):
        """Get Spyglass version."""
        from spyglass import __version__ as sg_version

        return ".".join(sg_version.split(".")[:3])  # Major.Minor.Patch

    @cached_property
    def _export_table(self):
        """Lazy load export selection table."""
        from spyglass.common.common_usage import ExportSelection

        return ExportSelection()

    @property
    def export_id(self):
        """ID of export in progress.

        NOTE: User of an env variable to store export_id may not be thread safe.
        Exports must be run in sequence, not parallel.
        """

        return int(environ.get(EXPORT_ENV_VAR, 0))

    @export_id.setter
    def export_id(self, value):
        """Set ID of export using `table.export_id = X` notation."""
        if self.export_id != 0 and self.export_id != value:
            raise RuntimeError("Export already in progress.")
        environ[EXPORT_ENV_VAR] = str(value)
        exit_register(self._export_id_cleanup)  # End export on exit

    @export_id.deleter
    def export_id(self):
        """Delete ID of export using `del table.export_id` notation."""
        self._export_id_cleanup()

    def _export_id_cleanup(self):
        """Cleanup export ID."""
        if environ.get(EXPORT_ENV_VAR):
            del environ[EXPORT_ENV_VAR]
        exit_unregister(self._export_id_cleanup)  # Remove exit hook

    def _start_export(self, paper_id, analysis_id):
        """Start export process."""
        if self.export_id:
            logger.info(f"Export {self.export_id} in progress. Starting new.")
            self._stop_export(warn=False)

        self.export_id = self._export_table.insert1_return_pk(
            dict(
                paper_id=paper_id,
                analysis_id=analysis_id,
                spyglass_version=self._spyglass_version,
            )
        )

    def _stop_export(self, warn=True):
        """End export process."""
        if not self.export_id and warn:
            logger.warning("Export not in progress.")
        del self.export_id

    def _log_fetch(self, *args, **kwargs):
        """Log fetch for export."""
        if not self.export_id or self.database == "common_usage":
            return

        banned = [
            "head",  # Prevents on Table().head() call
            "tail",  # Prevents on Table().tail() call
            "preview",  # Prevents on Table() call
            "_repr_html_",  # Prevents on Table() call in notebook
            "cautious_delete",  # Prevents add on permission check during delete
            "get_abs_path",  # Assumes that fetch_nwb will catch file/table
        ]
        called = [i.function for i in inspect_stack()]
        if set(banned) & set(called):  # if called by any in banned, return
            return

        logger.debug(f"Export {self.export_id}: fetch()   {self.table_name}")

        restr = self.restriction or True
        limit = kwargs.get("limit")
        offset = kwargs.get("offset")
        if limit or offset:  # Use result as restr if limit/offset
            restr = self.restrict(restr).fetch(
                log_fetch=False, as_dict=True, limit=limit, offset=offset
            )

        restr_str = make_condition(self, restr, set())

        if isinstance(restr_str, str) and len(restr_str) > 2048:
            raise RuntimeError(
                "Export cannot handle restrictions > 2048.\n\t"
                + "If required, please open an issue on GitHub.\n\t"
                + f"Restriction: {restr_str}"
            )
        self._export_table.Table.insert1(
            dict(
                export_id=self.export_id,
                table_name=self.full_table_name,
                restriction=restr_str,
            ),
            skip_duplicates=True,
        )

    def fetch(self, *args, log_fetch=True, **kwargs):
        """Log fetch for export."""
        ret = super().fetch(*args, **kwargs)
        if log_fetch:
            self._log_fetch(*args, **kwargs)
        return ret

    def fetch1(self, *args, log_fetch=True, **kwargs):
        """Log fetch1 for export."""
        ret = super().fetch1(*args, **kwargs)
        if log_fetch:
            self._log_fetch(*args, **kwargs)
        return ret

    # ------------------------------ Restrict by ------------------------------

    def __lshift__(self, restriction) -> QueryExpression:
        """Restriction by upstream operator e.g. ``q1 << q2``.

        Returns
        -------
        QueryExpression
            A restricted copy of the query expression using the nearest upstream
            table for which the restriction is valid.
        """
        return self.restrict_by(restriction, direction="up")

    def __rshift__(self, restriction) -> QueryExpression:
        """Restriction by downstream operator e.g. ``q1 >> q2``.

        Returns
        -------
        QueryExpression
            A restricted copy of the query expression using the nearest upstream
            table for which the restriction is valid.
        """
        return self.restrict_by(restriction, direction="down")

    def _ensure_names(self, tables) -> List[str]:
        """Ensure table is a string in a list."""
        if not isinstance(tables, (list, tuple, set)):
            tables = [tables]
        for table in tables:
            return [getattr(table, "full_table_name", table) for t in tables]

    def ban_search_table(self, table):
        """Ban table from search in restrict_by."""
        self._banned_search_tables.update(self._ensure_names(table))

    def unban_search_table(self, table):
        """Unban table from search in restrict_by."""
        self._banned_search_tables.difference_update(self._ensure_names(table))

    def see_banned_tables(self):
        """Print banned tables."""
        logger.info(f"Banned tables: {self._banned_search_tables}")

    def restrict_by(
        self,
        restriction: str = True,
        direction: str = "up",
        return_graph: bool = False,
        verbose: bool = False,
        **kwargs,
    ) -> QueryExpression:
        """Restrict self based on up/downstream table.

        If fails to restrict table, the shortest path may not have been correct.
        If there's a different path that should be taken, ban unwanted tables.

        >>> my_table = MyTable() # must be instantced
        >>> my_table.ban_search_table(UnwantedTable1)
        >>> my_table.ban_search_table([UnwantedTable2, UnwantedTable3])
        >>> my_table.unban_search_table(UnwantedTable3)
        >>> my_table.see_banned_tables()
        >>>
        >>> my_table << my_restriction

        Parameters
        ----------
        restriction : str
            Restriction to apply to the some table up/downstream of self.
        direction : str, optional
            Direction to search for valid restriction. Default 'up'.
        return_graph : bool, optional
            If True, return FindKeyGraph object. Default False, returns
            restricted version of present table.
        verbose : bool, optional
            If True, print verbose output. Default False.

        Returns
        -------
        Union[QueryExpression, TableChain]
            Restricted version of present table or TableChain object. If
            return_graph, use all_ft attribute to see all tables in cascade.
        """
        from spyglass.utils.dj_graph import TableChain  # noqa: F401

        if restriction is True:
            return self

        try:
            ret = self.restrict(restriction)  # Save time trying first
            if len(ret) < len(self):
                # If it actually restricts, if not it might by a dict that
                # is not a valid restriction, returned as True
                logger.warning("Restriction valid for this table. Using as is.")
                return ret
        except DataJointError:
            pass  # Could avoid try/except if assert_join_compatible return bool
            logger.debug("Restriction not valid. Attempting to cascade.")

        if direction == "up":
            parent, child = None, self
        elif direction == "down":
            parent, child = self, None
        else:
            raise ValueError("Direction must be 'up' or 'down'.")

        graph = TableChain(
            parent=parent,
            child=child,
            direction=direction,
            search_restr=restriction,
            banned_tables=list(self._banned_search_tables),
            cascade=True,
            verbose=verbose,
            **kwargs,
        )

        if not graph.found_restr:
            return None

        if return_graph:
            return graph

        ret = self & graph._get_restr(self.full_table_name)
        warn_text = (
            f" after restrict with path: {graph.path_str}\n\t "
            + "See `help(YourTable.restrict_by)`"
        )
        if len(ret) == len(self):
            logger.warning("Same length" + warn_text)
        elif len(ret) == 0:
            logger.warning("No entries" + warn_text)

        return ret


class SpyglassMixinPart(SpyglassMixin, dj.Part):
    """
    A part table for Spyglass Group tables. Assists in propagating
    delete calls from upstreeam tables to downstream tables.
    """

    def delete(self, *args, **kwargs):
        """Delete master and part entries."""
        restriction = self.restriction or True  # for (tbl & restr).delete()
        (self.master & restriction).delete(*args, **kwargs)

from atexit import register as exit_register
from atexit import unregister as exit_unregister
from collections import OrderedDict
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
from datajoint.utils import get_master, user_choice
from networkx import NetworkXError
from pymysql.err import DataError

from spyglass.utils.database_settings import SHARED_MODULES
from spyglass.utils.dj_chains import TableChain, TableChains
from spyglass.utils.dj_helper_fn import fetch_nwb, get_nwb_table
from spyglass.utils.dj_merge_tables import RESERVED_PRIMARY_KEY as MERGE_PK
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
    ddm(*args, **kwargs)
        Alias for delete_downstream_merge.
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

    # ------------------------ delete_downstream_merge ------------------------

    @cached_property
    def _merge_tables(self) -> Dict[str, dj.FreeTable]:
        """Dict of merge tables downstream of self: {full_table_name: FreeTable}.

        Cache of items in parents of self.descendants(as_objects=True). Both
        descendant and parent must have the reserved primary key 'merge_id'.
        """
        self.connection.dependencies.load()
        merge_tables = {}
        visited = set()

        def search_descendants(parent):
            # TODO: Add check that parents are in the graph. If not, raise error
            #       asking user to import the table.
            # TODO: Make a `is_merge_table` helper, and check for false
            #       positives in the mixin init.
            for desc in parent.descendants(as_objects=True):
                if (
                    MERGE_PK not in desc.heading.names
                    or not (master_name := get_master(desc.full_table_name))
                    or master_name in merge_tables
                ):
                    continue
                master_ft = dj.FreeTable(self.connection, master_name)
                if is_merge_table(master_ft):
                    merge_tables[master_name] = master_ft
                if master_name not in visited:
                    visited.add(master_name)
                    search_descendants(master_ft)

        try:
            _ = search_descendants(self)
        except NetworkXError as e:
            table_name = "".join(e.args[0].split("`")[1:4])
            raise ValueError(f"Please import {table_name} and try again.")

        logger.info(
            f"Building merge cache for {self.table_name}.\n\t"
            + f"Found {len(merge_tables)} downstream merge tables"
        )

        return merge_tables

    @cached_property
    def _merge_chains(self) -> OrderedDict[str, List[dj.FreeTable]]:
        """Dict of chains to merges downstream of self

        Format: {full_table_name: TableChains}.

        For each merge table found in _merge_tables, find the path from self to
        merge via merge parts. If the path is valid, add it to the dict. Cache
        prevents need to recompute whenever delete_downstream_merge is called
        with a new restriction. To recompute, add `reload_cache=True` to
        delete_downstream_merge call.
        """
        merge_chains = {}
        for name, merge_table in self._merge_tables.items():
            chains = TableChains(self, merge_table, connection=self.connection)
            if len(chains):
                merge_chains[name] = chains

        # This is ordered by max_len of chain from self to merge, which assumes
        # that the merge table with the longest chain is the most downstream.
        # A more sophisticated approach would order by length from self to
        # each merge part independently, but this is a good first approximation.
        return OrderedDict(
            sorted(
                merge_chains.items(), key=lambda x: x[1].max_len, reverse=True
            )
        )

    def _get_chain(self, substring) -> TableChains:
        """Return chain from self to merge table with substring in name."""
        for name, chain in self._merge_chains.items():
            if substring.lower() in name:
                return chain
        raise ValueError(f"No chain found with '{substring}' in name.")

    def _commit_merge_deletes(
        self, merge_join_dict: Dict[str, List[QueryExpression]], **kwargs
    ) -> None:
        """Commit merge deletes.

        Parameters
        ----------
        merge_join_dict : Dict[str, List[QueryExpression]]
            Dictionary of merge tables and their joins. Uses 'merge_id' primary
            key to restrict delete.

        Extracted for use in cautious_delete and delete_downstream_merge."""
        for table_name, part_restr in merge_join_dict.items():
            table = self._merge_tables[table_name]
            keys = [part.fetch(MERGE_PK, as_dict=True) for part in part_restr]
            (table & keys).delete(**kwargs)

    def delete_downstream_merge(
        self,
        restriction: str = None,
        dry_run: bool = True,
        reload_cache: bool = False,
        disable_warning: bool = False,
        return_parts: bool = True,
        **kwargs,
    ) -> Union[List[QueryExpression], Dict[str, List[QueryExpression]]]:
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
        return_parts : bool, optional
            If True, return list of merge part entries to be deleted. Default
            True. If False, return dictionary of merge tables and their joins.
        **kwargs : Any
            Passed to datajoint.table.Table.delete.
        """
        if reload_cache:
            del self._merge_tables
            del self._merge_chains

        restriction = restriction or self.restriction or True

        merge_join_dict = {}
        for name, chain in self._merge_chains.items():
            join = chain.join(restriction)
            if join:
                merge_join_dict[name] = join

        if not merge_join_dict and not disable_warning:
            logger.warning(
                f"No merge deletes found w/ {self.table_name} & "
                + f"{restriction}.\n\tIf this is unexpected, try importing "
                + " Merge table(s) and running with `reload_cache`."
            )

        if dry_run:
            return merge_join_dict.values() if return_parts else merge_join_dict

        self._commit_merge_deletes(merge_join_dict, **kwargs)

    def ddm(
        self,
        restriction: str = None,
        dry_run: bool = True,
        reload_cache: bool = False,
        disable_warning: bool = False,
        return_parts: bool = True,
        *args,
        **kwargs,
    ) -> Union[List[QueryExpression], Dict[str, List[QueryExpression]]]:
        """Alias for delete_downstream_merge."""
        return self.delete_downstream_merge(
            restriction=restriction,
            dry_run=dry_run,
            reload_cache=reload_cache,
            disable_warning=disable_warning,
            return_parts=return_parts,
            *args,
            **kwargs,
        )

    # ---------------------------- cautious_delete ----------------------------

    @cached_property
    def _delete_deps(self) -> List[Table]:
        """List of tables required for delete permission check.

        LabMember, LabTeam, and Session are required for delete permission.

        Used to delay import of tables until needed, avoiding circular imports.
        Each of these tables inheits SpyglassMixin.
        """
        from spyglass.common import LabMember, LabTeam, Session  # noqa F401

        self._session_pk = Session.primary_key[0]
        self._member_pk = LabMember.primary_key[0]
        return [LabMember, LabTeam, Session]

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

        Session = self._delete_deps[-1]
        SesExp = Session.Experimenter

        # Not called in delete permission check, only bare _get_exp_summary
        if self._member_pk in self.heading.names:
            return self * SesExp

        empty_pk = {self._member_pk: "NULL"}
        format = dj.U(self._session_pk, self._member_pk)

        restr = self.restriction or True
        sess_link = self._session_connection.join(restr, reverse_order=True)

        exp_missing = format & (sess_link - SesExp).proj(**empty_pk)
        exp_present = format & (sess_link * SesExp - exp_missing).proj()

        return exp_missing + exp_present

    @cached_property
    def _session_connection(self) -> Union[TableChain, bool]:
        """Path from Session table to self. False if no connection found."""
        connection = TableChain(parent=self._delete_deps[-1], child=self)
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
        LabMember, LabTeam, Session = self._delete_deps

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
    def _usage_table(self):
        """Temporary inclusion for usage tracking."""
        from spyglass.common.common_usage import CautiousDelete

        return CautiousDelete()

    def _log_delete(self, start, merge_deletes=None, super_delete=False):
        """Log use of cautious_delete."""
        if isinstance(merge_deletes, QueryExpression):
            merge_deletes = merge_deletes.fetch(as_dict=True)
        safe_insert = dict(
            duration=time() - start,
            dj_user=dj.config["database.user"],
            origin=self.full_table_name,
        )
        restr_str = "Super delete: " if super_delete else ""
        restr_str += "".join(self.restriction) if self.restriction else "None"
        try:
            self._usage_table.insert1(
                dict(
                    **safe_insert,
                    restriction=restr_str[:255],
                    merge_deletes=merge_deletes,
                )
            )
        except (DataJointError, DataError):
            self._usage_table.insert1(
                dict(**safe_insert, restriction="Unknown")
            )

    # TODO: Intercept datajoint delete confirmation prompt for merge deletes
    def cautious_delete(self, force_permission: bool = False, *args, **kwargs):
        """Delete table rows after checking user permission.

        Permission is granted to users listed as admin in LabMember table or to
        users on a team with with the Session experimenter(s). If the table
        cannot be linked to Session, a warning is logged and the delete
        continues. If the Session has no experimenter, or if the user is not on
        a team with the Session experimenter(s), a PermissionError is raised.

        Parameters
        ----------
        force_permission : bool, optional
            Bypass permission check. Default False.
        *args, **kwargs : Any
            Passed to datajoint.table.Table.delete.
        """
        start = time()

        if not force_permission:
            self._check_delete_permission()

        merge_deletes = self.delete_downstream_merge(
            dry_run=True,
            disable_warning=True,
            return_parts=False,
        )

        safemode = (
            dj.config.get("safemode", True)
            if kwargs.get("safemode") is None
            else kwargs["safemode"]
        )

        if merge_deletes:
            for table, content in merge_deletes.items():
                count = sum([len(part) for part in content])
                dj_logger.info(f"Merge: Deleting {count} rows from {table}")
            if (
                not self._test_mode
                or not safemode
                or user_choice("Commit deletes?", default="no") == "yes"
            ):
                self._commit_merge_deletes(merge_deletes, **kwargs)
            else:
                logger.info("Delete aborted.")
                self._log_delete(start)
                return

        super().delete(*args, **kwargs)  # Additional confirm here

        self._log_delete(start=start, merge_deletes=merge_deletes)

    def cdel(self, force_permission=False, *args, **kwargs):
        """Alias for cautious_delete."""
        self.cautious_delete(force_permission=force_permission, *args, **kwargs)

    def delete(self, force_permission=False, *args, **kwargs):
        """Alias for cautious_delete, overwrites datajoint.table.Table.delete"""
        self.cautious_delete(force_permission=force_permission, *args, **kwargs)

    def super_delete(self, warn=True, *args, **kwargs):
        """Alias for datajoint.table.Table.delete."""
        if warn:
            logger.warning("!! Bypassing cautious_delete !!")
            self._log_delete(start=time(), super_delete=True)
        super().delete(*args, **kwargs)

    # ------------------------------- Export Log -------------------------------

    @cached_property
    def _spyglass_version(self):
        """Get Spyglass version from dj.config."""
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

    # ------------------------- Other helper methods -------------------------

    def _auto_increment(self, key, pk, *args, **kwargs):
        """Auto-increment primary key."""
        if not key.get(pk):
            key[pk] = (dj.U().aggr(self, n=f"max({pk})").fetch1("n") or 0) + 1
        return key

    def file_like(self, name=None, **kwargs):
        """Convenience method for wildcard search on file name fields."""
        if not name:
            return self & True
        attr = None
        for field in self.heading.names:
            if "file" in field:
                attr = field
                break
        if not attr:
            logger.error(f"No file-like field found in {self.full_table_name}")
            return
        return self & f"{attr} LIKE '%{name}%'"

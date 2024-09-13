from atexit import register as exit_register
from atexit import unregister as exit_unregister
from contextlib import nullcontext
from functools import cached_property
from inspect import stack as inspect_stack
from os import environ
from re import match as re_match
from time import time
from typing import List

import datajoint as dj
from datajoint.condition import make_condition
from datajoint.errors import DataJointError
from datajoint.expression import QueryExpression
from datajoint.table import Table
from datajoint.utils import to_camel_case
from packaging.version import parse as version_parse
from pandas import DataFrame
from pymysql.err import DataError

from spyglass.utils.database_settings import SHARED_MODULES
from spyglass.utils.dj_helper_fn import (
    NonDaemonPool,
    ensure_names,
    fetch_nwb,
    get_nwb_table,
    populate_pass_function,
)
from spyglass.utils.dj_merge_tables import Merge, is_merge_table
from spyglass.utils.logging import logger

try:
    import pynapple  # noqa F401
except (ImportError, ModuleNotFoundError):
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
    cautious_delete(force_permission=False, *args, **kwargs)
        Check user permissions before deleting table rows. Permission is granted
        to users listed as admin in LabMember table or to users on a team with
        with the Session experimenter(s). If the table where the delete is
        executed cannot be linked to a Session, a warning is logged and the
        delete continues. If the Session has no experimenter, or if the user is
        not on a team with the Session experimenter(s), a PermissionError is
        raised. `force_permission` can be set to True to bypass permission check.
    """

    # _nwb_table = None # NWBFile table class, defined at the table level

    # pks for delete permission check, assumed to be one field for each
    _session_pk = None  # Session primary key. Mixin is ambivalent to Session pk
    _member_pk = None  # LabMember primary key. Mixin ambivalent table structure

    _banned_search_tables = set()  # Tables to avoid in restrict_by
    _parallel_make = False  # Tables that use parallel processing in make

    _use_transaction = True  # Use transaction in populate.

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
        rets = []
        for parent in self.parents(as_objects=True):
            parent_key = {
                k: v for k, v in key.items() if k in parent.heading.names
            }
            parent_name = to_camel_case(parent.table_name)
            if query := parent & parent_key:
                rets.append(f"{parent_name}:\n{query}")
            else:
                rets.append(f"{parent_name}: MISSING")
        logger.info("\n".join(rets))

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
        from spyglass.common.common_nwbfile import (  # noqa F401
            AnalysisNwbfile,
            Nwbfile,
        )

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

    def load_shared_schemas(self, additional_prefixes: list = None) -> None:
        """Load shared schemas to include in graph traversal.

        Parameters
        ----------
        additional_prefixes : list, optional
            Additional prefixes to load. Default None.
        """
        all_shared = [
            *SHARED_MODULES,
            dj.config["database.user"],
            "file",
            "sharing",
        ]

        if additional_prefixes:
            all_shared.extend(additional_prefixes)

        # Get a list of all shared schemas in spyglass
        schemas = dj.conn().query(
            "SELECT DISTINCT table_schema "  # Unique schemas
            + "FROM information_schema.key_column_usage "
            + "WHERE"
            + '    table_name not LIKE "~%%"'  # Exclude hidden
            + "    AND constraint_name='PRIMARY'"  # Only primary keys
            + "AND ("  # Only shared schemas
            + " OR ".join([f"table_schema LIKE '{s}_%%'" for s in all_shared])
            + ") "
            + "ORDER BY table_schema;"
        )

        # Load the dependencies for all shared schemas
        for schema in schemas:
            dj.schema(schema[0]).connection.dependencies.load()

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
        from spyglass.common import LabMember  # noqa F401
        from spyglass.common import IntervalList, LabTeam, Session
        from spyglass.common.common_nwbfile import schema  # noqa F401

        self._session_pk = Session.primary_key[0]
        self._member_pk = LabMember.primary_key[0]
        return [LabMember, LabTeam, Session, schema.external, IntervalList]

    @cached_property
    def _graph_deps(self) -> list:
        from spyglass.utils.dj_graph import RestrGraph  # noqa #F401
        from spyglass.utils.dj_graph import TableChain

        return [TableChain, RestrGraph]

    def _get_exp_summary(self):
        """Get summary of experimenters for session(s), including NULL.

        Parameters
        ----------
        sess_link : datajoint.expression.QueryExpression
            Join of table link with Session table.

        Returns
        -------
        Union[QueryExpression, None]
            dj.Union object Summary of experimenters for session(s). If no link
            to Session, return None.
        """
        if not self._session_connection.has_link:
            return None

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
        TableChain = self._graph_deps[0]

        return TableChain(parent=self._delete_deps[2], child=self, verbose=True)

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

        if not (sess_summary := self._get_exp_summary()):
            logger.warn(
                f"Could not find a connection from {self.camel_name} "
                + "to Session.\n Be careful not to delete others' data."
            )
            return

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

    def _log_delete(self, start, del_blob=None, super_delete=False):
        """Log use of super_delete."""
        from spyglass.common.common_usage import CautiousDelete

        safe_insert = dict(
            duration=time() - start,
            dj_user=dj.config["database.user"],
            origin=self.full_table_name,
        )
        restr_str = "Super delete: " if super_delete else ""
        restr_str += "".join(self.restriction) if self.restriction else "None"
        try:
            CautiousDelete().insert1(
                dict(
                    **safe_insert,
                    restriction=restr_str[:255],
                    merge_deletes=del_blob,
                )
            )
        except (DataJointError, DataError):
            CautiousDelete().insert1(dict(**safe_insert, restriction="Unknown"))

    @cached_property
    def _has_updated_dj_version(self):
        """Return True if DataJoint version is up to date."""
        target_dj = version_parse("0.14.2")
        ret = version_parse(dj.__version__) >= target_dj
        if not ret:
            logger.warning(f"Please update DataJoint to {target_dj} or later.")
        return ret

    def cautious_delete(
        self, force_permission: bool = False, dry_run=False, *args, **kwargs
    ):
        """Permission check, then delete potential orphans and table rows.

        Permission is granted to users listed as admin in LabMember table or to
        users on a team with with the Session experimenter(s). If the table
        cannot be linked to Session, a warning is logged and the delete
        continues. If the Session has no experimenter, or if the user is not on
        a team with the Session experimenter(s), a PermissionError is raised.

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
        if len(self) == 0:
            logger.warning(f"Table is empty. No need to delete.\n{self}")
            return

        if self._has_updated_dj_version:
            kwargs["force_masters"] = True

        external, IntervalList = self._delete_deps[3], self._delete_deps[4]

        if not force_permission or dry_run:
            self._check_delete_permission()

        if dry_run:
            return (
                IntervalList(),  # cleanup func relies on downstream deletes
                external["raw"].unused(),
                external["analysis"].unused(),
            )

        super().delete(*args, **kwargs)  # Confirmation here

        for ext_type in ["raw", "analysis"]:
            external[ext_type].delete(
                delete_external_files=True, display_progress=False
            )

        if not self._test_mode:
            _ = IntervalList().nightly_cleanup(dry_run=False)

    def delete(self, *args, **kwargs):
        """Alias for cautious_delete, overwrites datajoint.table.Table.delete"""
        self.cautious_delete(*args, **kwargs)

    def super_delete(self, warn=True, *args, **kwargs):
        """Alias for datajoint.table.Table.delete."""
        if warn:
            logger.warning("!! Bypassing cautious_delete !!")
            self._log_delete(start=time(), super_delete=True)
        super().delete(*args, **kwargs)

    # -------------------------------- populate --------------------------------

    def _hash_upstream(self, keys):
        """Hash upstream table keys for no transaction populate.

        Uses a RestrGraph to capture all upstream tables, restrict them to
        relevant entries, and hash the results. This is used to check if
        upstream tables have changed during a no-transaction populate and avoid
        the following data-integrity error:

        1. User A starts no-transaction populate.
        2. User B deletes and repopulates an upstream table, changing contents.
        3. User A finishes populate, inserting data that is now invalid.

        Parameters
        ----------
        keys : list
            List of keys for populating table.
        """
        RestrGraph = self._graph_deps[1]

        if not (parents := self.parents(as_objects=True, primary=True)):
            # Should not happen, as this is only called from populated tables
            raise RuntimeError("No upstream tables found for upstream hash.")

        leaves = {  # Restriction on each primary parent
            p.full_table_name: [
                {k: v for k, v in key.items() if k in p.heading.names}
                for key in keys
            ]
            for p in parents
        }

        return RestrGraph(seed_table=self, leaves=leaves, cascade=True).hash

    def populate(self, *restrictions, **kwargs):
        """Populate table in parallel, with or without transaction protection.

        Supersedes datajoint.table.Table.populate for classes with that
        spawn processes in their make function and always use transactions.

        `_use_transaction` class attribute can be set to False to disable
        transaction protection for a table. This is not recommended for tables
        with short processing times. A before-and-after hash check is performed
        to ensure upstream tables have not changed during populate, and may
        be a more time-consuming process. To permit the `make` to insert without
        populate, set `_allow_insert` to True.
        """
        processes = kwargs.pop("processes", 1)

        # Decide if using transaction protection
        use_transact = kwargs.pop("use_transation", None)
        if use_transact is None:  # if user does not specify, use class default
            use_transact = self._use_transaction
            if self._use_transaction is False:  # If class default is off, warn
                logger.warning(
                    "Turning off transaction protection this table by default. "
                    + "Use use_transation=True to re-enable.\n"
                    + "Read more about transactions:\n"
                    + "https://docs.datajoint.io/python/definition/05-Transactions.html\n"
                    + "https://github.com/LorenFrankLab/spyglass/issues/1030"
                )
        if use_transact is False and processes > 1:
            raise RuntimeError(
                "Must use transaction protection with parallel processing.\n"
                + "Call with use_transation=True.\n"
                + f"Table default transaction use: {self._use_transaction}"
            )

        # Get keys, needed for no-transact or multi-process w/_parallel_make
        keys = [True]
        if use_transact is False or (processes > 1 and self._parallel_make):
            keys = (self._jobs_to_do(restrictions) - self.target).fetch(
                "KEY", limit=kwargs.get("limit", None)
            )

        if use_transact is False:
            upstream_hash = self._hash_upstream(keys)
            if kwargs:  # Warn of ignoring populate kwargs, bc using `make`
                logger.warning(
                    "Ignoring kwargs when not using transaction protection."
                )

        if processes == 1 or not self._parallel_make:
            if use_transact:  # Pass single-process populate to super
                kwargs["processes"] = processes
                return super().populate(*restrictions, **kwargs)
            else:  # No transaction protection, use bare make
                for key in keys:
                    self.make(key)
                if upstream_hash != self._hash_upstream(keys):
                    (self & keys).delete(safemode=False)
                    logger.error(
                        "Upstream tables changed during non-transaction "
                        + "populate. Please try again."
                    )
                return

        # If parallel in both make and populate, use non-daemon processes
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

        ret = ".".join(sg_version.split(".")[:3])  # Ditch commit info

        if self._test_mode:
            return ret[:16] if len(ret) > 16 else ret

        if not bool(re_match(r"^\d+\.\d+\.\d+", ret)):  # Major.Minor.Patch
            raise ValueError(
                f"Spyglass version issues. Expected #.#.#, Got {ret}."
                + "Please try running `hatch build` from your spyglass dir."
            )

        return ret

    def compare_versions(
        self, version: str, other: str = None, msg: str = None
    ) -> None:
        """Compare two versions. Raise error if not equal.

        Parameters
        ----------
        version : str
            Version to compare.
        other : str, optional
            Other version to compare. Default None. Use self._spyglass_version.
        msg : str, optional
            Additional error message info. Default None.
        """
        if self._test_mode:
            return

        other = other or self._spyglass_version

        if version_parse(version) != version_parse(other):
            raise RuntimeError(
                f"Found mismatched versions: {version} vs {other}\n{msg}"
            )

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

    def ban_search_table(self, table):
        """Ban table from search in restrict_by."""
        self._banned_search_tables.update(ensure_names(table, force_list=True))

    def unban_search_table(self, table):
        """Unban table from search in restrict_by."""
        self._banned_search_tables.difference_update(
            ensure_names(table, force_list=True)
        )

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
        TableChain = self._graph_deps[0]

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

    # ------------------------------ Check locks ------------------------------

    def exec_sql_fetchall(self, query):
        """
        Execute the given query and fetch the results.    Parameters
        ----------
        query : str
            The SQL query to execute.    Returns
        -------
        list of tuples
            The results of the query.
        """
        results = dj.conn().query(query).fetchall()
        return results  # Check if performance schema is enabled

    def check_threads(self, detailed=False, all_threads=False) -> DataFrame:
        """Check for locked threads in the database.

        Parameters
        ----------
        detailed : bool, optional
            Show all columns in the metadata_locks table. Default False, show
            summary.
        all_threads : bool, optional
            Show all threads, not just those related to this table.
            Default False.


        Returns
        -------
        DataFrame
            A DataFrame containing the metadata locks.
        """
        performance__status = self.exec_sql_fetchall(
            "SHOW VARIABLES LIKE 'performance_schema';"
        )
        if performance__status[0][1] == "OFF":
            raise RuntimeError(
                "Database does not monitor threads. "
                + "Please ask you administrator to enable performance schema."
            )

        metadata_locks_query = """
        SELECT
            ml.OBJECT_SCHEMA, -- Table schema
            ml.OBJECT_NAME, -- Table name
            ml.OBJECT_TYPE, -- What is locked
            ml.LOCK_TYPE, -- Type of lock
            ml.LOCK_STATUS, -- Lock status
            ml.OWNER_THREAD_ID, -- Thread ID of the lock owner
            t.PROCESSLIST_ID, -- User connection ID
            t.PROCESSLIST_USER, -- User
            t.PROCESSLIST_HOST, -- User machine
            t.PROCESSLIST_TIME, -- Time in seconds
            t.PROCESSLIST_DB, -- Thread database
            t.PROCESSLIST_COMMAND, -- Likely Query
            t.PROCESSLIST_STATE, -- Waiting for lock, sending data, or locked
            t.PROCESSLIST_INFO -- Actual query
        FROM performance_schema.metadata_locks AS ml
        JOIN performance_schema.threads AS t
        ON ml.OWNER_THREAD_ID = t.THREAD_ID
        """

        where_clause = (
            f"WHERE ml.OBJECT_SCHEMA = '{self.database}' "
            + f"AND ml.OBJECT_NAME = '{self.table_name}'"
        )
        metadata_locks_query += ";" if all_threads else where_clause

        df = DataFrame(
            self.exec_sql_fetchall(metadata_locks_query),
            columns=[
                "Schema",  # ml.OBJECT_SCHEMA -- Table schema
                "Table Name",  # ml.OBJECT_NAME -- Table name
                "Locked",  # ml.OBJECT_TYPE -- What is locked
                "Lock Type",  # ml.LOCK_TYPE -- Type of lock
                "Lock Status",  # ml.LOCK_STATUS -- Lock status
                "Thread ID",  # ml.OWNER_THREAD_ID -- Thread ID of the lock owner
                "Connection ID",  # t.PROCESSLIST_ID -- User connection ID
                "User",  # t.PROCESSLIST_USER -- User
                "Host",  # t.PROCESSLIST_HOST -- User machine
                "Process Database",  # t.PROCESSLIST_DB -- Thread database
                "Time (s)",  # t.PROCESSLIST_TIME -- Time in seconds
                "Process",  # t.PROCESSLIST_COMMAND -- Likely Query
                "State",  # t.PROCESSLIST_STATE
                "Query",  # t.PROCESSLIST_INFO -- Actual query
            ],
        )

        df["Name"] = df["User"].apply(self._delete_deps[0]().get_djuser_name)

        keep_cols = []
        if all_threads:
            keep_cols.append("Table")
            df["Table"] = df["Schema"] + "." + df["Table Name"]
        df = df.drop(columns=["Schema", "Table Name"])

        if not detailed:
            keep_cols.extend(["Locked", "Name", "Time (s)", "Process", "State"])
            df = df[keep_cols]

        return df


class SpyglassMixinPart(SpyglassMixin, dj.Part):
    """
    A part table for Spyglass Group tables. Assists in propagating
    delete calls from upstreeam tables to downstream tables.
    """

    def delete(self, *args, **kwargs):
        """Delete master and part entries."""
        restriction = self.restriction or True  # for (tbl & restr).delete()
        (self.master & restriction).delete(*args, **kwargs)

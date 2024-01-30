from time import time
from typing import Dict, List

import datajoint as dj
import networkx as nx
from datajoint.table import logger as dj_logger
from datajoint.user_tables import Table, TableMeta
from datajoint.utils import get_master, user_choice

from spyglass.utils.database_settings import SHARED_MODULES
from spyglass.utils.dj_helper_fn import fetch_nwb
from spyglass.utils.dj_merge_tables import RESERVED_PRIMARY_KEY as MERGE_PK
from spyglass.utils.dj_merge_tables import Merge
from spyglass.utils.logging import logger


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
    cdel(*args, **kwargs)
        Alias for cautious_delete.
    delte_downstream_merge(restriction=None, dry_run=True, reload_cache=False)
        Delete downstream merge table entries associated with restricton.
        Requires caching of merge tables and links, which is slow on first call.
        `restriction` can be set to a string to restrict the delete. `dry_run`
        can be set to False to commit the delete. `reload_cache` can be set to
        True to reload the merge cache.
    ddm(*args, **kwargs)
        Alias for delete_downstream_merge.
    """

    _nwb_table_dict = {}  # Dict mapping NWBFile table to path attribute name.
    # _nwb_table = None # NWBFile table class, defined at the table level
    _nwb_table_resolved = None  # NWBFiletable class, resolved here from above
    _delete_dependencies = []  # Session, LabMember, LabTeam, delay import
    # pks for delete permission check, assumed to be on field
    _session_pk = None  # Session primary key. Mixin is ambivalent to Session pk
    _member_pk = None  # LabMember primary key. Mixin ambivalent table structure
    _merge_table_cache = {}  # Cache of merge tables downstream of self
    _merge_chains_cache = {}  # Cache of table chains to merges
    _session_connection_cache = None  # Cache of path from Session to self
    _test_mode_cache = None  # Cache of test mode setting for delete
    _usage_table_cache = None  # Temporary inclusion for usage tracking

    # ------------------------------- fetch_nwb -------------------------------

    @property
    def _table_dict(self):
        """Dict mapping NWBFile table to path attribute name.

        Used to delay import of NWBFile tables until needed, avoiding circular
        imports.
        """
        if not self._nwb_table_dict:
            from spyglass.common.common_nwbfile import (  # noqa F401
                AnalysisNwbfile,
                Nwbfile,
            )

            self._nwb_table_dict = {
                AnalysisNwbfile: "analysis_file_abs_path",
                Nwbfile: "nwb_file_abs_path",
            }
        return self._nwb_table_dict

    @property
    def _nwb_table_tuple(self):
        """NWBFile table class.

        Used to determine fetch_nwb behavior. Also used in Merge.fetch_nwb.
        Multiple copies for different purposes.

        - _nwb_table may be user-set. Don't overwrite.
        - _nwb_table_resolved is set here from either _nwb_table or definition.
        - _nwb_table_tuple is used to cache result of _nwb_table_resolved and
            return the appropriate path_attr from _table_dict above.
        """
        if not self._nwb_table_resolved:
            from spyglass.common.common_nwbfile import (  # noqa F401
                AnalysisNwbfile,
                Nwbfile,
            )

            if hasattr(self, "_nwb_table"):
                self._nwb_table_resolved = self._nwb_table

            if not hasattr(self, "_nwb_table"):
                self._nwb_table_resolved = (
                    AnalysisNwbfile
                    if "-> AnalysisNwbfile" in self.definition
                    else Nwbfile if "-> Nwbfile" in self.definition else None
                )

            if getattr(self, "_nwb_table_resolved", None) is None:
                raise NotImplementedError(
                    f"{self.__class__.__name__} does not have a "
                    "(Analysis)Nwbfile foreign key or _nwb_table attribute."
                )

        return (
            self._nwb_table_resolved,
            self._table_dict[self._nwb_table_resolved],
        )

    def fetch_nwb(self, *attrs, **kwargs):
        """Fetch NWBFile object from relevant table.

        Implementing class must have a foreign key to Nwbfile or
        AnalysisNwbfile or a _nwb_table attribute.

        A class that does not have with either '-> Nwbfile' or
        '-> AnalysisNwbfile' in its definition can use a _nwb_table attribute to
        specify which table to use.
        """
        nwb_table, path_attr = self._nwb_table_tuple

        return fetch_nwb(self, (nwb_table, path_attr), *attrs, **kwargs)

    # -------------------------------- delete ---------------------------------

    @property
    def _delete_deps(self) -> list:
        """List of tables required for delete permission check.

        Used to delay import of tables until needed, avoiding circular imports.
        """
        if not self._delete_dependencies:
            from spyglass.common import LabMember, LabTeam, Session  # noqa F401

            self._delete_dependencies = [LabMember, LabTeam, Session]
            self._session_pk = Session.primary_key[0]
            self._member_pk = LabMember.primary_key[0]
        return self._delete_dependencies

    @property
    def _test_mode(self) -> bool:
        """Return True if test mode is enabled."""
        if not self._test_mode_cache:
            from spyglass.settings import test_mode

            self._test_mode_cache = test_mode
        return self._test_mode_cache

    @property
    def _merge_tables(self) -> Dict[str, dj.FreeTable]:
        """Dict of merge tables downstream of self.

        Cache of items in parents of self.descendants(as_objects=True) that
        have a merge primary key.
        """
        if self._merge_table_cache:
            return self._merge_table_cache

        def has_merge_pk(table):
            return MERGE_PK in table.heading.names

        self.connection.dependencies.load()
        for desc in self.descendants(as_objects=True):
            if not has_merge_pk(desc):
                continue
            if not (master_name := get_master(desc.full_table_name)):
                continue
            master = dj.FreeTable(self.connection, master_name)
            if has_merge_pk(master):
                self._merge_table_cache[master_name] = master
        logger.info(
            f"Building merge cache for {self.table_name}.\n\t"
            + f"Found {len(self._merge_table_cache)} downstream merge tables"
        )

        return self._merge_table_cache

    @property
    def _merge_chains(self) -> Dict[str, List[dj.FreeTable]]:
        """Dict of merge links downstream of self.

        For each merge table found in _merge_tables, find the path from self to
        merge. If the path is valid, add it to the dict. Cache prevents need
        to recompute whenever delete_downstream_merge is called with a new
        restriction.
        """
        if self._merge_chains_cache:
            return self._merge_chains_cache

        for name, merge_table in self._merge_tables.items():
            chains = TableChains(self, merge_table, connection=self.connection)
            if len(chains):
                self._merge_chains_cache[name] = chains
        return self._merge_chains_cache

    def _commit_merge_deletes(self, merge_join_dict, **kwargs):
        """Commit merge deletes.

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
    ) -> List[dj.expression.QueryExpression]:
        """Delete downstream merge table entries associated with restricton.

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
            self._merge_table_cache = {}
            self._merge_chains_cache = {}

        restriction = restriction or self.restriction or True

        merge_join_dict = {}
        for name, chain in self._merge_chains.items():
            join = chain.join(restriction)
            if join:
                merge_join_dict[name] = join

        if not merge_join_dict and not disable_warning:
            logger.warning(
                f"No merge tables found downstream of {self.full_table_name}."
                + "\n\tIf this is unexpected, try running with `reload_cache`."
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
    ):
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
        empty_pk = {self._member_pk: "NULL"}

        format = dj.U(self._session_pk, self._member_pk)
        sess_link = self._session_connection.join(self.restriction)

        exp_missing = format & (sess_link - SesExp).proj(**empty_pk)
        exp_present = format & (sess_link * SesExp - exp_missing).proj()

        return exp_missing + exp_present

    @property
    def _session_connection(self) -> dj.expression.QueryExpression:
        """Path from Session table to self.

        None is not yet cached, False if no connection found.
        """
        if self._session_connection_cache is None:
            connection = TableChain(parent=self._delete_deps[-1], child=self)
            self._session_connection_cache = (
                connection if connection.has_link else False
            )
        return self._session_connection_cache

    def _check_delete_permission(self) -> None:
        """Check user name against lab team assoc. w/ self * Session.

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

        if not self._session_connection:
            logger.warn(  # Permit delete if no session connection
                "Could not find lab team associated with "
                + f"{self.__class__.__name__}."
                + "\nBe careful not to delete others' data."
            )
            return

        sess_summary = self._get_exp_summary()
        experimenters = sess_summary.fetch(self._member_pk)
        if None in experimenters:
            # TODO: Check if allow delete of remainder?
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

    @property
    def _usage_table(self):
        """Temporary inclusion for usage tracking."""
        if not self._usage_table_cache:
            from spyglass.common.common_usage import CautiousDelete

            self._usage_table_cache = CautiousDelete
        return self._usage_table_cache

    def _log_use(self, start, merge_deletes=None):
        """Log use of cautious_delete."""
        self._usage_table.insert1(
            dict(
                duration=time() - start,
                dj_user=dj.config["database.user"],
                origin=self.full_table_name,
                restriction=self.restriction,
                merge_deletes=merge_deletes,
            )
        )

    # Rename to `delete` when we're ready to use it
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
                self._log_use(start)
                return

        super().delete(*args, **kwargs)  # Additional confirm here

        self._log_use(start=start, merge_deletes=merge_deletes)

    def cdel(self, *args, **kwargs):
        """Alias for cautious_delete."""
        self.cautious_delete(*args, **kwargs)


class TableChains:
    """Class for representing chains from parent to Merge table via parts."""

    def __init__(self, parent, child, connection=None):
        self.parent = parent
        self.child = child
        self.connection = connection or parent.connection
        parts = child.parts(as_objects=True)
        self.part_names = [part.full_table_name for part in parts]
        self.chains = [TableChain(parent, part) for part in parts]
        self.has_link = any([chain.has_link for chain in self.chains])

    def __repr__(self):
        return "\n".join([str(chain) for chain in self.chains])

    def __len__(self):
        return len([c for c in self.chains if c.has_link])

    def join(self, restriction=None):
        restriction = restriction or self.parent.restriction or True
        joins = []
        for chain in self.chains:
            if joined := chain.join(restriction):
                joins.append(joined)
        return joins


class TableChain:
    """Class for representing a chain of tables.

    Note: Parent -> Merge should use TableChains instead.
    """

    def __init__(self, parent: Table, child: Table, connection=None):
        self._connection = connection or parent.connection
        if not self._connection.dependencies._loaded:
            self._connection.dependencies.load()

        if (  # if child is a merge table
            get_master(child.full_table_name) == ""
            and MERGE_PK in child.heading.names
        ):
            logger.error("Child is a merge table. Use TableChains instead.")

        self._link_symbol = " -> "
        self.parent = parent
        self.child = child
        self._repr = None
        self._names = None  # full table names of tables in chain
        self._objects = None  # free tables in chain
        self._has_link = child.full_table_name in parent.descendants()

    def __str__(self):
        """Return string representation of chain: parent -> child."""
        if not self._has_link:
            return "No link"
        return (
            "Chain: "
            + self.parent.table_name
            + self._link_symbol
            + self.child.table_name
        )

    def __repr__(self):
        """Return full representation of chain: parent -> {links} -> child."""
        if self._repr:
            return self._repr
        self._repr = (
            "Chain: "
            + self._link_symbol.join([t.table_name for t in self.objects])
            if self.names
            else "No link"
        )
        return self._repr

    def __len__(self):
        """Return number of tables in chain."""
        return len(self.names)

    @property
    def has_link(self) -> bool:
        """Return True if parent is linked to child.

        Cached as hidden attribute _has_link to set False if nx.NetworkXNoPath
        is raised by nx.shortest_path.
        """
        return self._has_link

    @property
    def names(self) -> List[str]:
        """Return list of full table names in chain.

        Uses networkx.shortest_path.
        """
        if not self._has_link:
            return None
        if self._names:
            return self._names
        try:
            self._names = nx.shortest_path(
                self.parent.connection.dependencies,
                self.parent.full_table_name,
                self.child.full_table_name,
            )
            return self._names
        except nx.NetworkXNoPath:
            self._has_link = False
            return None

    @property
    def objects(self) -> List[dj.FreeTable]:
        """Return list of FreeTable objects for each table in chain."""
        if not self._objects:
            self._objects = (
                [dj.FreeTable(self._connection, name) for name in self.names]
                if self.names
                else None
            )
        return self._objects

    def join(self, restricton: str = None) -> dj.expression.QueryExpression:
        """Return join of tables in chain with restriction applied to parent."""
        restriction = restricton or self.parent.restriction or True
        join = self.objects[0] & restriction
        for table in self.objects[1:]:
            join = join * table
        return join if join else None

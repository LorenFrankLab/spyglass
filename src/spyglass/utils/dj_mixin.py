from collections.abc import Iterable
from typing import Dict, List, Union

import datajoint as dj
from datajoint.table import logger as dj_logger
from datajoint.user_tables import Table, TableMeta
from datajoint.utils import get_master, user_choice

from spyglass.settings import test_mode
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
    """

    _nwb_table_dict = {}  # Dict mapping NWBFile table to path attribute name.
    # _nwb_table = None # NWBFile table class, defined at the table level
    _nwb_table_resolved = None  # NWBFiletable class, resolved here from above
    _delete_dependencies = []  # Session, LabMember, LabTeam, delay import
    _merge_delete_func = None  # delete_downstream_merge, delay import
    # pks for delete permission check, assumed to be on field
    _session_pk = None  # Session primary key. Mixin is ambivalent to Session pk
    _member_pk = None  # LabMember primary key. Mixin ambivalent table structure
    _merge_cache = {}  # Cache of merge tables downstream of self
    _merge_cache_links = {}  # Cache of merge links downstream of self
    _session_connection_cache = None  # Cache of path from Session to self

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
                    else Nwbfile
                    if "-> Nwbfile" in self.definition
                    else None
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
    def _merge_del_func(self) -> callable:
        """Callable: delete_downstream_merge function.

        Used to delay import of func until needed, avoiding circular imports.
        """
        if not self._merge_delete_func:
            from spyglass.utils.dj_merge_tables import (  # noqa F401
                delete_downstream_merge,
            )

            self._merge_delete_func = delete_downstream_merge
        return self._merge_delete_func

    @staticmethod
    def _get_instanced(*tables) -> Union[dj.user_tables.Table, list]:
        """Return instance of table(s) if not already instanced."""
        ret = []
        if not isinstance(tables, Iterable):
            tables = tuple(tables)
        for table in tables:
            if not isinstance(table, dj.user_tables.Table):
                ret.append(table())
            else:
                ret.append(table)
        return ret[0] if len(ret) == 1 else ret

    @staticmethod
    def _link_repr(parent, child, width=120):
        len_each = (width - 4) // 2
        p = parent.full_table_name[:len_each].ljust(len_each)
        c = child.full_table_name[:len_each].ljust(len_each)
        return f"{p} -> {c}"

    def _get_connection(
        self,
        child: dj.user_tables.TableMeta,
        parent: dj.user_tables.TableMeta = None,
        recurse_level: int = 4,
        visited: set = None,
    ) -> Union[List[dj.FreeTable], List[List[dj.FreeTable]]]:
        """
        Return list of tables connecting the parent and child for a valid join.

        Parameters
        ----------
        parent : dj.user_tables.TableMeta
            DataJoint table upstream in pipeline.
        child : dj.user_tables.TableMeta
            DataJoint table downstream in pipeline.
        recurse_level : int, optional
            Maximum number of recursion levels. Default is 4.
        visited : set, optional
            Set of visited tables (used internally for recursion).

        Returns
        -------
        List[dj.FreeTable] or List[List[dj.FreeTable]]
            List of paths, with each path as a list FreeTables connecting the
            parent and child for a valid join.
        """
        parent = parent or self
        parent, child = self._get_instanced(parent, child)
        visited = visited or set()
        child_is_merge = child.full_table_name in self._merge_tables

        if recurse_level < 1 or (  # if too much recursion
            not child_is_merge  # merge table ok
            and (  # already visited, outside spyglass, or no connection
                child.full_table_name in visited
                or child.full_table_name.strip("`").split("_")[0]
                not in SHARED_MODULES
                or child.full_table_name not in parent.descendants()
            )
        ):
            return []

        if child.full_table_name in parent.children():
            logger.debug(f"1-{recurse_level}:" + self._link_repr(parent, child))
            if isinstance(child, dict) or isinstance(parent, dict):
                __import__("pdb").set_trace()
            return [parent, child]

        if child_is_merge:
            ret = []
            parts = child.parts(as_objects=True)
            if not parts:
                logger.warning(f"Merge has no parts: {child.full_table_name}")
            for part in child.parts(as_objects=True):
                links = self._get_connection(
                    parent=parent,
                    child=part,
                    recurse_level=recurse_level,
                    visited=visited,
                )
                visited.add(part.full_table_name)
                if links:
                    logger.debug(
                        f"2-{recurse_level}:" + self._link_repr(parent, part)
                    )
                    ret.append(links + [child])

            return ret

        for subchild in parent.children(as_objects=True):
            links = self._get_connection(
                parent=subchild,
                child=child,
                recurse_level=recurse_level - 1,
                visited=visited,
            )
            visited.add(subchild.full_table_name)
            if links:
                logger.debug(
                    f"3-{recurse_level}:" + self._link_repr(subchild, child)
                )
                if parent.full_table_name in [l.full_table_name for l in links]:
                    return links
                else:
                    return [parent] + links

        return []

    def _join_list(
        self,
        tables: Union[List[dj.FreeTable], List[List[dj.FreeTable]]],
        restriction: str = None,
    ) -> dj.expression.QueryExpression:
        """Return join of all tables in list. Omits empty items."""
        restriction = restriction or self.restriction or True

        if not isinstance(tables[0], (list, tuple)):
            tables = [tables]
        ret = []
        for table_list in tables:
            join = table_list[0] & restriction
            for table in table_list[1:]:
                join = join * table
            if join:
                ret.append(join)
        return ret[0] if len(ret) == 1 else ret

    def _connection_repr(self, connection) -> str:
        if isinstance(connection[0], (Table, TableMeta)):
            connection = [connection]
        if not isinstance(connection[0], (list, tuple)):
            connection = [connection]
        ret = []
        for table_list in connection:
            connection_str = ""
            for table in table_list:
                if isinstance(table, str):
                    connection_str += table + " -> "
                else:
                    connection_str += table.table_name + " -> "
            ret.append(f"\n\tPath: {connection_str[:-4]}")
        return ret

    def _ensure_dependencies_loaded(self) -> None:
        """Ensure connection dependencies loaded."""
        if not self.connection.dependencies._loaded:
            self.connection.dependencies.load()

    @property
    def _merge_tables(self) -> Dict[str, dj.FreeTable]:
        """Dict of merge tables downstream of self.

        Cache of items in parents of self.descendants(as_objects=True) that
        have a merge primary key.
        """
        if self._merge_cache:
            return self._merge_cache

        def has_merge_pk(table):
            return MERGE_PK in table.heading.names

        self._ensure_dependencies_loaded()
        for desc in self.descendants(as_objects=True):
            if not has_merge_pk(desc):
                continue
            if not (master_name := get_master(desc.full_table_name)):
                continue
            master = dj.FreeTable(self.connection, master_name)
            if has_merge_pk(master):
                self._merge_cache[master_name] = master
        logger.info(f"Found {len(self._merge_cache)} merge tables")

        return self._merge_cache

    @property
    def _merge_links(self) -> Dict[str, List[dj.FreeTable]]:
        """Dict of merge links downstream of self.

        For each merge table found in _merge_tables, find the path from self to
        merge. If the path is valid, add it to the dict. Cahche prevents need
        to recompute whenever delete_downstream_merge is called with a new
        restriction.
        """
        if self._merge_cache_links:
            return self._merge_cache_links
        for name, merge_table in self._merge_tables.items():
            connection = self._get_connection(child=merge_table)
            if connection:
                self._merge_cache_links[name] = connection
        return self._merge_cache_links

    def _commit_merge_deletes(self, merge_join_dict, **kwargs):
        ret = []
        for table, selection in merge_join_dict.items():
            keys = selection.fetch(MERGE_PK, as_dict=True)
            ret.append(table & keys)  # NEEDS TESTING WITH ACTUAL DELETE
            # (table & keys).delete(**kwargs) # TODO: Run delete here
        return ret

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
            self._merge_cache = {}
            self._merge_cache_links = {}

        restriction = restriction or self.restriction or True

        merge_join_dict = {}
        for merge_name, merge_link in self._merge_links.items():
            logger.debug(self._connection_repr(merge_link))
            joined = self._join_list(merge_link, restriction=self.restriction)
            if joined:
                merge_join_dict[self._merge_tables[merge_name]] = joined

        if not merge_join_dict and not disable_warning:
            logger.warning(
                f"No merge tables found downstream of {self.full_table_name}."
                + "\n\tIf this is unexpected, try running with `reload_cache`."
            )

        if dry_run:
            return merge_join_dict
        self._commit_merge_deletes(merge_join_dict, **kwargs)

    def ddm(self, *args, **kwargs):
        """Alias for delete_downstream_merge."""
        return self.delete_downstream_merge(*args, **kwargs)

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

        format = dj.U(self._session_pk, self._member_pk)

        sess_link = self._join_list(self._session_connection)
        exp_missing = format & (sess_link - Session.Experimenter).proj(
            **{self._member_pk: "NULL"}
        )
        exp_present = (
            format & (sess_link * Session.Experimenter - exp_missing).proj()
        )
        return exp_missing + exp_present

    @property
    def _session_connection(self) -> dj.expression.QueryExpression:
        """Path from Session table to self.

        None is not yet cached, False if no connection found.
        """
        if self._session_connection_cache is None:
            self._session_connection_cache = (
                self._get_connection(parent=self._delete_deps[-1], child=self)
                or False
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
                count, name = len(content), table.full_table_name
                dj_logger.info(f"Merge: Deleting {count} rows from {name}")
            if (
                not test_mode
                or not safemode
                or user_choice("Commit deletes?", default="no") == "yes"
            ):
                self._commit_merge_deletes(merge_deletes, **kwargs)
            else:
                logger.info("Delete aborted.")
                return

        super().delete(*args, **kwargs)  # Additional confirm here

    def cdel(self, *args, **kwargs):
        """Alias for cautious_delete."""
        self.cautious_delete(*args, **kwargs)

import datajoint as dj
from datajoint.table import logger as dj_logger
from datajoint.utils import user_choice

from spyglass.utils.dj_helper_fn import fetch_nwb
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

    _nwb_table_dict = {}
    _delete_dependencies = []
    _merge_delete_func = None

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

    def fetch_nwb(self, *attrs, **kwargs):
        """Fetch NWBFile object from relevant table.

        Implementing class must have a foreign key to Nwbfile or
        AnalysisNwbfile or a _nwb_table attribute.

        A class that does not have with either '-> Nwbfile' or
        '-> AnalysisNwbfile' in its definition can use a _nwb_table attribute to
        specify which table to use.
        """
        _nwb_table_dict = self._table_dict
        analysis_table, nwb_table = _nwb_table_dict.keys()

        if not hasattr(self, "_nwb_table"):
            self._nwb_table = (
                analysis_table
                if "-> AnalysisNwbfile" in self.definition
                else nwb_table
                if "-> Nwbfile" in self.definition
                else None
            )

        if getattr(self, "_nwb_table", None) is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not have a (Analysis)Nwbfile "
                "foreign key or _nwb_table attribute."
            )

        return fetch_nwb(
            self,
            (self._nwb_table, _nwb_table_dict[self._nwb_table]),
            *attrs,
            **kwargs,
        )

    # -------------------------------- delete ---------------------------------

    @property
    def _delete_deps(self) -> list:
        """List of tables required for delete permission check.

        Used to delay import of tables until needed, avoiding circular imports.
        """
        if not self._delete_dependencies:
            from spyglass.common import LabMember, LabTeam, Session  # noqa F401

            self._delete_dependencies = [LabMember, LabTeam, Session]
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

    def _find_session(
        self,
        table: dj.user_tables.UserTable,
        Session: dj.user_tables.UserTable,
        search_limit: int = 2,
    ) -> dj.expression.QueryExpression:
        """Find Session table associated with table.

        Parameters
        ----------
        table : datajoint.user_tables.UserTable
            Table to search for Session ancestor.
        Session : datajoint.user_tables.UserTable
            Session table to search for. Passed as arg to prevent re-import.
        search_limit : int, optional
            Number of levels of children of target table to search. Default 2.

        Returns
        -------
        datajoint.expression.QueryExpression or None
            Join of table link with Session table if found, else None.
        """
        # TODO: check search_limit default is enough for any table in spyglass
        if self.full_table_name == Session.full_table_name:
            # if self is Session, return self
            return self

        elif (
            # if Session is not in ancestors of table, search children
            Session.full_table_name not in table.ancestors()
            and search_limit > 0  # prevent infinite recursion
        ):
            for child in table.children():
                table = self._find_session(child, Session, search_limit - 1)
                if table:  # table is link, will valid join to Session
                    break

        elif search_limit < 1:  # if no session ancestor found and limit reached
            return  # Err kept in parent func to centralize permission logic

        return table * Session

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

        sess = self._find_session(self, Session)
        if not sess:  # Permit delete if not linked to a session
            logger.warn(
                "Could not find lab team associated with "
                + f"{self.__class__.__name__}."
                + "\nBe careful not to delete others' data."
            )
            return

        experimenters = (sess * Session.Experimenter).fetch("lab_member_name")
        if len(experimenters) < len(sess):
            # TODO: adjust to check each session individually? Expensive but
            #       prevents against edge case of one sess with mult and another
            #       with none
            raise PermissionError(
                f"Please ensure all Sessions have an experimenter:\n{sess}"
            )

        user_name = LabMember().get_djuser_name(dj_user)
        for experimenter in set(experimenters):
            if user_name not in LabTeam().get_team_members(experimenter):
                raise PermissionError(
                    f"User '{user_name}' is not on a team with '{experimenter}'"
                    + ", an experimenter for session(s):\n"
                    + f"{sess * Session.Experimenter}"
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

        if not force_permission:
            self._check_delete_permission()

        merge_deletes = self._merge_del_func(
            self,
            restriction=self.restriction,
            dry_run=True,
            disable_warning=True,
        )

        safemode = (
            dj.config.get("safemode", True)
            if kwargs.get("safemode") is None
            else kwargs["safemode"]
        )

        if merge_deletes:
            for table, _ in merge_deletes:
                count, name = len(table), table.full_table_name
                dj_logger.info(f"Merge: Deleting {count} rows from {name}")
            if (
                not safemode
                or user_choice("Commit deletes?", default="no") == "yes"
            ):
                for merge_table, _ in merge_deletes:
                    merge_table.delete({**kwargs, "safemode": False})
            else:
                logger.info("Delete aborted.")
                return

        super().delete(*args, **kwargs)  # Additional confirm here

    def cdel(self, *args, **kwargs):
        """Alias for cautious_delete."""
        self.cautious_delete(*args, **kwargs)

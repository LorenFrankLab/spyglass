import datajoint as dj
from datajoint.table import logger as dj_logger
from datajoint.utils import user_choice

from spyglass.common.common_nwbfile import AnalysisNwbfile, Nwbfile
from spyglass.utils.dj_helper_fn import fetch_nwb
from spyglass.utils.dj_merge_tables import delete_downstream_merge


class SpyglassMixin:
    """Mixin for Spyglass DataJoint tables ."""

    # ------------------------------- fetch_nwb -------------------------------

    _nwb_table_dict = {
        AnalysisNwbfile: "analysis_file_abs_path",
        Nwbfile: "nwb_file_abs_path",
    }

    def fetch_nwb(self, *attrs, **kwargs):
        """Fetch NWBFile object from relevant table.

        Impleminting class must have a foreign key to Nwbfile or
        AnalysisNwbfile or a _nwb_table attribute.

        A class that does not have with either '-> Nwbfile' or
        '-> AnalysisNwbfile' in its definition can use a _nwb_table attribute to
        specify which table to use.
        """

        if not hasattr(self, "_nwb_table"):
            self._nwb_table = (
                AnalysisNwbfile
                if "-> AnalysisNwbfile" in self.definition
                else Nwbfile
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
            (self._nwb_table, self._nwb_table_dict[self._nwb_table]),
            *attrs,
            **kwargs,
        )

    # -------------------------------- delete ---------------------------------

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

        else:  # if no session ancestor found and limit reached
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
        from spyglass.common import LabMember, LabTeam, Session  # noqa F401

        dj_user = dj.config["database.user"]
        if dj_user in LabMember().admin:  # bypass permission check for admin
            return

        sess = self._find_session(self, Session)
        if not sess:  # Permit delete if not linked to a session
            print(
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
                    f"User {user_name} is not on a team with {experimenter}."
                )

    # Rename to `delete` when we're ready to use it
    # TODO: Intercept datajoint delete confirmation prompt for merge deletes
    def cautious_delete(self, force_permission=False, *args, **kwargs):
        """Delete table rows after checking user permission."""

        if not force_permission:
            self._check_delete_permission()

        merge_deletes = delete_downstream_merge(
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
                print("Delete aborted.")
                return

        super().delete(*args, **kwargs)  # Additional confirm here

    def cdel(self, *args, **kwargs):
        """Alias for cautious_delete."""
        self.cautious_delete(*args, **kwargs)

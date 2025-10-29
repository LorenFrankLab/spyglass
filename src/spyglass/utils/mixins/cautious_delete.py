"""Mixin to add cautious delete functionality to DataJoint tables."""

from functools import cached_property
from os import environ as os_environ
from time import time
from typing import List

import datajoint as dj
from datajoint.errors import DataJointError
from datajoint.table import Table
from packaging.version import parse as version_parse
from pymysql.err import DataError

from spyglass.utils.mixins.base import BaseMixin


class CautiousDeleteMixin(BaseMixin):
    """Mixin to add cautious delete functionality to DataJoint tables."""

    # pks for delete permission check, assumed to be one field for each
    _session_pk = None  # Session primary key. Mixin is ambivalent to Session pk
    _member_pk = None  # LabMember primary key. Mixin ambivalent table structure

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

        return TableChain(
            parent=self._delete_deps[2],
            child=self,
            banned_tables=["`common_lab`.`lab_team`"],  # See #1353
        )

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
            self._logger.warning(  # Permit delete if no session connection
                "Could not find lab team associated with "
                + f"{self.__class__.__name__}."
                + "\nBe careful not to delete others' data."
            )
            return

        if not (sess_summary := self._get_exp_summary()):
            self._logger.warning(
                f"Could not find a connection from {self.camel_name} "
                + "to Session.\n Be careful not to delete others' data."
            )
            return

        experimenters = sess_summary.fetch(self._member_pk)
        if None in experimenters:
            raise PermissionError(
                "Please ensure all Sessions have an experimenter in "
                + f"Session.Experimenter:\n{sess_summary}"
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
        self._logger.info(f"Queueing delete for session(s):\n{sess_summary}")

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
            self._logger.warning(
                f"Please update DataJoint to {target_dj} or later."
            )
        return ret

    @cached_property
    def _has_updated_sg_version(self):
        """Return True if Spyglass version is up to date."""
        if os_environ.get("SPYGLASS_UPDATED", False):
            return True

        from spyglass.common.common_version import SpyglassVersions

        return SpyglassVersions().is_up_to_date

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
            self._logger.warning(f"Table is empty. No need to delete.\n{self}")
            return

        if self._has_updated_dj_version and not isinstance(self, dj.Part):
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

        return None

    def delete(self, *args, **kwargs):
        """Alias for cautious_delete, overwrites datajoint.table.Table.delete"""
        self.cautious_delete(*args, **kwargs)

    def super_delete(self, warn=True, *args, **kwargs):
        """Alias for datajoint.table.Table.delete."""
        if warn:
            self._logger.warning("!! Bypassing cautious_delete !!")
            self._log_delete(start=time(), super_delete=True)
        super().delete(*args, **kwargs)

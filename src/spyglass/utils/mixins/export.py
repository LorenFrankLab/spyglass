from atexit import register as exit_register
from atexit import unregister as exit_unregister
from collections import defaultdict
from contextvars import ContextVar
from functools import cached_property
from inspect import stack as inspect_stack
from os import environ
from re import match as re_match
from typing import List

from datajoint.condition import AndList, Top, make_condition
from datajoint.expression import QueryExpression
from datajoint.table import Table
from packaging.version import parse as version_parse

from spyglass.utils.mixins.fetch import FetchMixin
from spyglass.utils.sql_helper_fn import bash_escape_sql

EXPORT_ENV_VAR = "SPYGLASS_EXPORT_ID"
FETCH_LOG_FLAG = ContextVar("FETCH_LOG_FLAG", default=True)


class ExportMixin(FetchMixin):
    """Mixin for DataJoint tables to support export logging.

    Uses FetchMixin._log_fetch to log fetch calls to an Export table.
    """

    _export_cache = defaultdict(set)

    # ------------------------------ Version Info -----------------------------

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

    # ------------------------------- Dependency -------------------------------

    @cached_property
    def _export_table(self):
        """Lazy load export selection table."""
        from spyglass.common.common_usage import ExportSelection

        return ExportSelection()

    # ------------------------------ ID Property ------------------------------

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
        self._export_cache = dict()
        if environ.get(EXPORT_ENV_VAR):
            del environ[EXPORT_ENV_VAR]
        exit_unregister(self._export_id_cleanup)  # Remove exit hook

    # ------------------------------- Export API -------------------------------

    def _start_export(self, paper_id, analysis_id):
        """Start export process."""
        if self.export_id:
            self._logger.info(
                f"Export {self.export_id} in progress. Starting new."
            )
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
            self._logger.warning("Export not in progress.")
        del self.export_id

    # ------------------------------- Log Fetch -------------------------------

    def _called_funcs(self):
        """Get stack trace functions."""
        ignore = {
            "__and__",  # caught by restrict
            "__mul__",  # caught by join
            "_called_funcs",  # run here
            "_log_fetch",  # run here
            "_log_fetch_nwb",  # run here
            "<module>",
            "_exec_file",
            "_pseudo_sync_runner",
            "_run_cell",
            "_run_cmd_line_code",
            "_run_with_log",
            "execfile",
            "init_code",
            "initialize",
            "inner",
            "interact",
            "launch_instance",
            "mainloop",
            "run",
            "run_ast_nodes",
            "run_cell",
            "run_cell_async",
            "run_code",
            "run_line_magic",
            "safe_execfile",
            "start",
            "start_ipython",
        }

        ret = {i.function for i in inspect_stack()} - ignore
        return ret

    def _log_fetch(self, restriction=None, *args, **kwargs):
        """Log fetch for export."""
        if (
            not self.export_id
            or self.database == "common_usage"
            or not FETCH_LOG_FLAG.get()
        ):
            return
        elif isinstance(restriction, Top):
            raise RuntimeError(
                "Cannot log fetch with Top() restriction, as it is not "
                "deterministic.\nUse a specific restriction, like a dict."
            )

        banned = [
            "head",  # Prevents on Table().head() call
            "tail",  # Prevents on Table().tail() call
            "preview",  # Prevents on Table() call
            "_repr_html_",  # Prevents on Table() call in notebook
            "cautious_delete",  # Prevents add on permission check during delete
            # "get_abs_path",  # Assumes that fetch_nwb will catch file/table
            "_check_delete_permission",  # Prevents on Table().delete()
            "delete",  # Prevents on Table().delete()
            "_load_admin",  # Prevents on permission check
        ]  # if called by any in banned, return
        if set(banned) & self._called_funcs():
            return

        restr = restriction or self.restriction or True
        limit = kwargs.get("limit")
        offset = kwargs.get("offset")
        if limit or offset:  # Use result as restr if limit/offset
            restr = self.restrict(restr).fetch(
                log_export=False, as_dict=True, limit=limit, offset=offset
            )

        restr_str = make_condition(self, restr, set())

        if restr_str is True:
            restr_str = "True"  # otherwise stored in table as '1'

        if isinstance(restr_str, str) and "SELECT" in restr_str:
            raise RuntimeError(
                "Export cannot handle subquery restrictions. Please submit a "
                + "bug report on GitHub with the code you ran and this"
                + f"restriction:\n\t{restr_str}"
            )

        if isinstance(restr_str, str) and len(restr_str) > 2048:
            raise RuntimeError(
                "Export cannot handle restrictions > 2048.\n\t"
                + "If required, please open an issue on GitHub.\n\t"
                + f"Restriction: {restr_str}"
            )

        if isinstance(restr_str, str):
            restr_str = bash_escape_sql(restr_str, add_newline=False)

        if restr_str in self._export_cache[self.full_table_name]:
            return
        self._export_cache[self.full_table_name].add(restr_str)

        self._export_table.Table.insert1(
            dict(
                export_id=self.export_id,
                table_name=self.full_table_name,
                restriction=restr_str,
            )
        )
        restr_logline = restr_str.replace("AND", "\n\tAND").replace(
            "OR", "\n\tOR"
        )
        self._logger.debug(
            f"\nTable: {self.full_table_name}\nRestr: {restr_logline}"
        )

    @property
    def _custom_analysis_parent(self):
        """Check for custom AnalysisNwbfile parent table.

        Returns
        -------
        custom_analysis_parent : str or None
            Custom AnalysisNwbfile parent table if exists, else None.
        """
        this_name = self.full_table_name

        custom_parent = [
            p
            for p in self.parents()
            if p.endswith("nwbfile`.`analysis_nwbfile`")
            and not p.startswith("`common_")
        ]
        if not custom_parent:
            return None

        if len(custom_parent) > 1:
            raise RuntimeError(
                f"Multiple AnalysisNwbfile parents found for "
                f"{this_name}: {custom_parent}"
            )

        from spyglass.common.common_nwbfile import AnalysisRegistry

        return AnalysisRegistry().get_class(custom_parent[0])()

    def _parent_copy_to_common(self, fnames: List[str] = None):
        """Copy parent custom AnalysisNwbfile entries to common.

        Also used by `sharing_kachery.py` to ensure common table integrity.

        Parameters
        ----------
        fnames : List[str], optional
            List of analysis_file_name to copy, by default None (fetches all
            from self, likely restricted).
        """
        custom_parent = self._custom_analysis_parent
        if not custom_parent:
            return

        if not fnames:
            fnames = self.fetch("analysis_file_name", log_export=False)
        f_dict = [{"analysis_file_name": fname} for fname in fnames]

        parent_name = custom_parent.full_table_name
        self._logger.debug(f"Copying parent {parent_name} entries to common")

        (custom_parent & f_dict)._copy_to_common()

    def _log_fetch_nwb(self, table, table_attr):
        """Log fetch_nwb for export table.

        For custom AnalysisNwbfile tables, copy entries to common table
        to maintain referential integrity in ExportSelection.File.
        """
        from spyglass.common.common_nwbfile import AnalysisNwbfile

        this_name = self.full_table_name
        tbl_pk = "analysis_file_name"
        fnames = self.fetch(tbl_pk, log_export=True)
        self._logger.debug(
            f"Export: fetch_nwb\nTable:{this_name},\nFiles: {fnames}"
        )

        # Check if this table is itself a custom AnalysisNwbfile table
        is_custom_analysis = (
            this_name.endswith("_nwbfile`.`analysis_nwbfile`")
            and this_name != AnalysisNwbfile().full_table_name
        )

        if is_custom_analysis:
            # Self is custom AnalysisNwbfile, copy to common
            self._logger.debug(
                f"Export: detected custom AnalysisNwbfile table {this_name}"
            )
            self._copy_to_common()
        elif custom_parent := self._custom_analysis_parent:
            self._parent_copy_to_common(fnames=fnames)

        # Insert into ExportSelection.File (FK now guaranteed valid)
        self._logger.debug(
            f"Export: inserting {len(fnames)} files from {this_name}"
        )
        self._export_table.File.insert(
            [{"export_id": self.export_id, tbl_pk: fname} for fname in fnames],
            skip_duplicates=True,
        )

        # Log fetch on common AnalysisNwbfile (entries were copied there)
        this_restr = None
        if len(fnames) == 1:
            this_restr = f"{tbl_pk} = '{fnames[0]}'"
        elif len(fnames) > 1:
            this_restr = f"{tbl_pk} IN {tuple(fnames)}"
        if this_restr:
            AnalysisNwbfile()._log_fetch(restriction=this_restr)

    def _run_join(self, **kwargs):
        """Log join for export.

        Special case to log primary keys of each table in join, avoiding
        long restriction strings.
        """
        table_list = [self]
        other = kwargs.get("other")

        if hasattr(other, "_log_fetch"):  # Check if other has mixin
            table_list.append(other)  # can other._log_fetch
        else:
            self._logger.warning(f"Cannot export log join for\n{other}")

        joined = self.proj().join(other.proj(), log_export=False)
        for table in table_list:  # log separate for unique pks
            if isinstance(table, type) and issubclass(table, Table):
                # instancing table if class
                table = table()  # adapted from dj.declare.compile_foreign_key
            for r in joined.fetch(*table.primary_key, as_dict=True):
                table._log_fetch(restriction=r)

    def _run_with_log(self, method, *args, log_export=True, **kwargs):
        """Run method, log fetch, and return result.

        Uses FETCH_LOG_FLAG to prevent multiple logs in one user call.
        """
        log_this_call = FETCH_LOG_FLAG.get()  # One log per fetch call

        if log_this_call and not self.database == "common_usage":
            FETCH_LOG_FLAG.set(False)

        try:
            ret = method(*args, **kwargs)
        finally:
            if log_this_call:
                FETCH_LOG_FLAG.set(True)

        if log_export and self.export_id and log_this_call:
            if getattr(method, "__name__", None) == "join":  # special case
                self._run_join(**kwargs)
            else:
                restr = kwargs.get("restriction")
                if isinstance(restr, QueryExpression) and getattr(
                    restr, "restriction"  # if table, try to get restriction
                ):
                    restr = restr.restriction
                self._log_fetch(restriction=restr)
            self._logger.debug(f"Export: {self._called_funcs()}")

        return ret

    def is_restr(self, restr) -> bool:
        """Check if a restriction is actually restricting."""
        return bool(restr) and restr != True and not isinstance(restr, Top)

    # -------------------------- Intercept DJ methods --------------------------

    def fetch(self, *args, log_export=True, **kwargs):
        """Log fetch for export."""
        if not self.export_id:
            return super().fetch(*args, **kwargs)
        return self._run_with_log(
            super().fetch, *args, log_export=log_export, **kwargs
        )

    def fetch1(self, *args, log_export=True, **kwargs):
        """Log fetch1 for export."""
        if not self.export_id:
            return super().fetch1(*args, **kwargs)
        return self._run_with_log(
            super().fetch1, *args, log_export=log_export, **kwargs
        )

    def restrict(self, restriction, log_export=True):  # NOTE: added param
        """Log restrict for export."""
        if not self.export_id or log_export is False:
            return super().restrict(restriction)

        log_export = "fetch_nwb" not in self._called_funcs()
        if self.is_restr(restriction) and self.is_restr(self.restriction):
            combined = AndList([restriction, self.restriction])
        else:  # Only combine if both are restricting
            combined = restriction or self.restriction
        return self._run_with_log(
            super().restrict, restriction=combined, log_export=log_export
        )

    def join(self, other, log_export=True, *args, **kwargs):
        """Log join for export.

        Join in dj_helper_func related to fetch_nwb have `log_export=False`
        because these entries are caught on the file cascade in RestrGraph.
        """
        if not self.export_id:
            return super().join(other=other, *args, **kwargs)

        return self._run_with_log(
            super().join, other=other, log_export=log_export, *args, **kwargs
        )

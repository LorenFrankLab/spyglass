from atexit import register as exit_register
from atexit import unregister as exit_unregister
from functools import cached_property
from inspect import stack as inspect_stack
from os import environ
from re import match as re_match

from datajoint.condition import make_condition
from packaging.version import parse as version_parse

from spyglass.utils.logging import logger

EXPORT_ENV_VAR = "SPYGLASS_EXPORT_ID"


class ExportMixin:

    # ---------------------------- Version Handling ----------------------------

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
        if environ.get(EXPORT_ENV_VAR):
            del environ[EXPORT_ENV_VAR]
        exit_unregister(self._export_id_cleanup)  # Remove exit hook

    # ------------------------------- Export API -------------------------------

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

    # ------------------------------- Log Fetch -------------------------------

    def _log_fetch(self, restriction=None, *args, **kwargs):
        """Log fetch for export."""
        if not self.export_id or self.database == "common_usage":
            return

        banned = [
            "head",  # Prevents on Table().head() call
            "tail",  # Prevents on Table().tail() call
            "preview",  # Prevents on Table() call
            "_repr_html_",  # Prevents on Table() call in notebook
            "cautious_delete",  # Prevents add on permission check during delete
            # "get_abs_path",  # Assumes that fetch_nwb will catch file/table
        ]
        called = [i.function for i in inspect_stack()]
        if set(banned) & set(called):  # if called by any in banned, return
            return

        logger.debug(f"Export {self.export_id}: fetch()   {self.table_name}")

        restr = restriction or self.restriction or True
        limit = kwargs.get("limit")
        offset = kwargs.get("offset")
        if limit or offset:  # Use result as restr if limit/offset
            restr = self.restrict(restr).fetch(
                log_fetch=False, as_dict=True, limit=limit, offset=offset
            )

        restr_str = make_condition(self, restr, set())

        if restr_str is True:
            restr_str = "True"  # otherwise stored in table as '1'

        __import__("pdb").set_trace()

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
            )
        )

    def _log_fetch_nwb(self, table, table_attr):
        """Log fetch_nwb for export table."""
        tbl_pk = "analysis_file_name"
        fnames = (self * table).fetch(tbl_pk)
        logger.debug(
            f"Export {self.export_id}: fetch_nwb {self.table_name}, {fnames}"
        )
        self._export_table.File.insert(
            [{"export_id": self.export_id, tbl_pk: fname} for fname in fnames],
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

    def fetch(self, *args, log_fetch=True, **kwargs):
        """Log fetch for export."""
        ret = super().fetch(*args, **kwargs)
        if log_fetch and self.export_id:
            self._log_fetch(*args, **kwargs)
        return ret

    def fetch1(self, *args, log_fetch=True, **kwargs):
        """Log fetch1 for export."""
        ret = super().fetch1(*args, **kwargs)
        if log_fetch and self.export_id:
            self._log_fetch(*args, **kwargs)
        return ret

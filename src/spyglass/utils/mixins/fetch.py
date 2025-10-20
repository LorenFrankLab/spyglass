"""Mixin class for fetching NWB files and pynapple objects."""

from functools import cached_property
from typing import Optional, Tuple, Type, Union

from spyglass.utils.dj_helper_fn import fetch_nwb, get_nwb_table
from spyglass.utils.mixins.base import BaseMixin

try:
    import pynapple  # noqa F401
except (ImportError, ModuleNotFoundError):
    pynapple = None


class FetchMixin(BaseMixin):

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

        log_export = kwargs.pop("log_export", True)
        # Check if either:
        # 1. tbl_attr contains "analysis" (table references AnalysisNwbfile)
        # 2. self is itself an AnalysisNwbfile table (custom or master)
        is_analysis_table = self.full_table_name.endswith(
            "_nwbfile`.`analysis_nwbfile`"
        )
        if (
            log_export
            and self.export_id
            and ("analysis" in tbl_attr or is_analysis_table)
        ):
            self._log_fetch_nwb(table, tbl_attr)

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

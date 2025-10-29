"""Mixin class for fetching NWB files and pynapple objects."""

from functools import cached_property

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
            AnalysisRegistry,
            Nwbfile,
        )

        attr_val = getattr(self, "_nwb_table", None)

        # ---------- Overwrite AnalysisNwbfile if using a custom one ----------
        if not attr_val:

            def get_prefix(name: str) -> str:
                return name.split("_")[0].strip("`")

            analysis_parents = [
                p for p in self.parents() if p.endswith(".`analysis_nwbfile`")
            ]
            if len(analysis_parents) > 1:
                raise ValueError(
                    f"{self.__class__.__name__} has multiple AnalysisNwbfile "
                    "parents. This is not permitted."
                )
            if (
                len(analysis_parents) == 1
                and get_prefix(analysis_parents[0]) != "common"
            ):
                this_prefix = get_prefix(analysis_parents[0])
                AnalysisNwbfile = AnalysisRegistry().get_class(this_prefix)
        # --------------------------------------------------------------------

        resolved = attr_val or (
            AnalysisNwbfile
            if "-> AnalysisNwbfile" in self.definition
            else Nwbfile if "-> Nwbfile" in self.definition else None
        )

        if not resolved:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not have a "
                "(Analysis)Nwbfile foreign key or _nwb_table attribute."
            )

        table_dict = {
            AnalysisNwbfile: "analysis_file_abs_path",
            Nwbfile: "nwb_file_abs_path",
        }

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
        is_export = log_export and self.export_id

        if is_export and ("analysis" in tbl_attr):
            self._log_fetch_nwb(table, tbl_attr)

        is_analysis_table = self.full_table_name.endswith(
            "_nwbfile`.`analysis_nwbfile`"
        )
        if is_export and is_analysis_table:
            self._copy_to_common()

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

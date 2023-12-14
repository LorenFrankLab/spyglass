from ..common.common_nwbfile import AnalysisNwbfile, Nwbfile
from .dj_helper_fn import fetch_nwb


class SpyglassMixin:
    """Mixin for Spyglass DataJoint tables ."""

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

    # def delete(self):
    #     print(f"Deleting with mixin {self.__class__.__name__}...")

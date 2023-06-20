import datajoint as dj
import pandas as pd

from spyglass.common.common_ephys import LFP as CommonLFP  # noqa: F401
from spyglass.common.common_filter import FirFilterParameters  # noqa: F401
from spyglass.common.common_interval import IntervalList  # noqa: F401
from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.lfp.v1.lfp import LFPV1, ImportedLFPV1
from spyglass.utils.dj_helper_fn import fetch_nwb
from spyglass.utils.dj_merge_tables import Merge

schema = dj.schema("lfp_merge")


@schema
class LFPOutput(Merge):
    definition = """
    merge_id: uuid
    """

    class LFPV1(dj.Part):
        definition = """
        -> master
        ---
        -> LFPV1
        """

    class ImportedLFPV1(dj.Part):
        definition = """
        -> master
        ---
        -> ImportedLFPV1
        """

    class CommonLFP(dj.Part):
        """Table to pass-through legacy LFP"""

        definition = """
        -> master
        ---
        -> CommonLFP
        """

    @staticmethod
    def get_lfp_object(key: dict):
        """Returns the lfp object corresponding to the key

        Parameters
        ----------
        key : dict
            A dictionary containing some combination of
                                    uuid,
                                    nwb_file_name,
                                    lfp_electrode_group_name,
                                    interval_list_name,
                                    fir_filter_name

        Returns
        -------
        lfp_object
            The entry or entries in the LFPOutput part table that corresponds to the key
        """
        # first check if this returns anything from the LFP table
        lfp_object = LFPOutput.LFP & key
        if lfp_object is not None:
            return LFPV1 & lfp_object.fetch("KEY")
        else:
            return ImportedLFPV1 & (LFPOutput.ImportedLFP & key).fetch("KEY")

    def fetch_nwb(self, *attrs, **kwargs):
        parts = self._merge_restrict_parts(
            restriction=self.restriction, return_empties=False
        )

        if len(parts) == 1:
            return fetch_nwb(
                parts[0],
                (AnalysisNwbfile, "analysis_file_abs_path"),
                *attrs,
                **kwargs,
            )
        else:  # TODO: needs more testing!
            raise ValueError("Multiple sources found in Merge Table")

    def fetch1_dataframe(self, *attrs, **kwargs):
        nwb_lfp = self.fetch_nwb()[0]
        return pd.DataFrame(
            nwb_lfp["lfp"].data,
            index=pd.Index(nwb_lfp["lfp"].timestamps, name="time"),
        )

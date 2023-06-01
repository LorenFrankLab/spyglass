import datajoint as dj
import pandas as pd

from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.lfp.v1.lfp import LFPV1, ImportedLFPV1
from spyglass.utils.dj_helper_fn import fetch_nwb

schema = dj.schema("lfp_merge")


@schema
class LFPOutput(dj.Manual):
    definition = """
    lfp_id: uuid
    ---
    source: varchar(40)
    version: int

    """

    class LFPV1(dj.Part):
        definition = """
        -> LFPOutput
        -> LFPV1
        ---
        -> AnalysisNwbfile
        lfp_object_id: varchar(40)
        """

        def fetch_nwb(self, *attrs, **kwargs):
            return fetch_nwb(
                self, (AnalysisNwbfile, "analysis_file_abs_path"), *attrs, **kwargs
            )

        def fetch1_dataframe(self, *attrs, **kwargs):
            nwb_lfp = self.fetch_nwb()[0]
            return pd.DataFrame(
                nwb_lfp["lfp"].data,
                index=pd.Index(nwb_lfp["lfp"].timestamps, name="time"),
            )

    class ImportedLFPV1(dj.Part):
        definition = """
        -> LFPOutput
        -> ImportedLFPV1
        ---
        -> AnalysisNwbfile
        lfp_object_id: varchar(40)
        """

        def fetch_nwb(self, *attrs, **kwargs):
            return fetch_nwb(
                self, (AnalysisNwbfile, "analysis_file_abs_path"), *attrs, **kwargs
            )

        def fetch1_dataframe(self, *attrs, **kwargs):
            nwb_lfp = self.fetch_nwb()[0]
            return pd.DataFrame(
                nwb_lfp["lfp"].data,
                index=pd.Index(nwb_lfp["lfp"].timestamps, name="time"),
            )

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
        query = LFPOutput.LFP & key
        if query is not None:
            return LFPV1 & query.fetch("KEY")
        else:
            return ImportedLFPV1 & (LFPOutput.ImportedLFP & key).fetch("KEY")

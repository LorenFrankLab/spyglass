import datajoint as dj
import pandas as pd

from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.lfp.v1.lfp import LFPV1, ImportedLFPV1
from spyglass.utils.dj_helper_fn import fetch_nwb
from spyglass.common.common_ephys import LFP as CommonLFP

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
                self,
                (AnalysisNwbfile, "analysis_file_abs_path"),
                *attrs,
                **kwargs,
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
                self,
                (AnalysisNwbfile, "analysis_file_abs_path"),
                *attrs,
                **kwargs,
            )

        def fetch1_dataframe(self, *attrs, **kwargs):
            nwb_lfp = self.fetch_nwb()[0]
            return pd.DataFrame(
                nwb_lfp["lfp"].data,
                index=pd.Index(nwb_lfp["lfp"].timestamps, name="time"),
            )

    class CommonLFP(dj.Part):
        """
        Table to pass-through legacy LFP
        """

        definition = """
        -> PositionOutput
        -> CommonLFP
        ---
        -> IntervalList             # the valid intervals for the data
        -> FirFilterParameters                # the filter used for the data
        -> AnalysisNwbfile          # the name of the nwb file with the lfp data
        lfp_object_id: varchar(40)  # the NWB object ID for loading this object from the file
        lfp_sampling_rate: float    # the sampling rate, in HZ
        """

        def fetch_nwb(self, *attrs, **kwargs):
            return fetch_nwb(
                self,
                (AnalysisNwbfile, "analysis_file_abs_path"),
                *attrs,
                **kwargs,
            )

        def fetch1_dataframe(self, *attrs, **kwargs):
            nwb_lfp = self.fetch_nwb()[0]
            return pd.DataFrame(
                nwb_lfp["lfp"].data,
                index=pd.Index(nwb_lfp["lfp"].timestamps, name="time"),
            )

import datajoint as dj
import pandas as pd

from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.lfp_band.v1.lfp_band import LFPBandV1  # noqa F401
from spyglass.utils.dj_helper_fn import fetch_nwb

schema = dj.schema("lfp_band_merge")


@schema
class LFPBandOutput(dj.Manual):
    definition = """
    lfp_band_id: uuid
    ---
    source: varchar(40)
    version: int

    """

    class LFPBandV1(dj.Part):
        definition = """
        -> LFPBandOutput
        -> LFPBandV1
        ---
        -> AnalysisNwbfile
        lfp_band_object_id: varchar(40)
        """

        def fetch_nwb(self, *attrs, **kwargs):
            return fetch_nwb(
                self,
                (AnalysisNwbfile, "analysis_file_abs_path"),
                *attrs,
                **kwargs,
            )

        def fetch1_dataframe(self, *attrs, **kwargs):
            filtered_nwb = self.fetch_nwb()[0]
            return pd.DataFrame(
                filtered_nwb["filtered_data"].data,
                index=pd.Index(
                    filtered_nwb["filtered_data"].timestamps, name="time"
                ),
            )

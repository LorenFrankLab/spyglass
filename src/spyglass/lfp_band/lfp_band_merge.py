import datajoint as dj
import pandas as pd

from spyglass.lfp_band.v1.lfp_band import LFPBandV1  # noqa F401
from spyglass.utils.dj_merge_tables import Merge

schema = dj.schema("lfp_band_merge")


@schema
class LFPBandOutput(Merge):
    definition = """
    merge_id: uuid
    ---
    source: varchar(32)
    """

    class LFPBandV1(dj.Part):
        definition = """
        -> master
        ---
        -> LFPBandV1
        """

    def fetch1_dataframe(self, *attrs, **kwargs):
        filtered_nwb = self.fetch_nwb()[0]
        return pd.DataFrame(
            filtered_nwb["lfp_band"].data,
            index=pd.Index(filtered_nwb["lfp_band"].timestamps, name="time"),
        )

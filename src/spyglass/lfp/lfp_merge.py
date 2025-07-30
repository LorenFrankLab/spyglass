import datajoint as dj
import pandas as pd

from spyglass.common.common_ephys import LFP as CommonLFP  # noqa: F401
from spyglass.common.common_filter import FirFilterParameters  # noqa: F401
from spyglass.common.common_interval import IntervalList  # noqa: F401
from spyglass.lfp.lfp_imported import ImportedLFP  # noqa: F401
from spyglass.lfp.v1.lfp import LFPV1  # noqa: F401
from spyglass.utils.dj_merge_tables import _Merge
from spyglass.utils.dj_mixin import SpyglassMixin

schema = dj.schema("lfp_merge")


@schema
class LFPOutput(_Merge, SpyglassMixin):
    definition = """
    merge_id: uuid
    ---
    source: varchar(32)
    """

    class LFPV1(SpyglassMixin, dj.Part):  # noqa: F811
        definition = """
        -> master
        ---
        -> LFPV1
        """

    class ImportedLFP(SpyglassMixin, dj.Part):  # noqa: F811
        definition = """
        -> master
        ---
        -> ImportedLFP
        """

    class CommonLFP(SpyglassMixin, dj.Part):  # noqa: F811
        """Table to pass-through legacy LFP"""

        definition = """
        -> master
        ---
        -> CommonLFP
        """

    def fetch1_dataframe(self, *attrs, **kwargs):
        """Fetch a single dataframe from the merged table."""
        # Note: `proj` below facilitates operator syntax eg Table & restrict
        _ = self.ensure_single_entry()
        nwb_lfp = self.fetch_nwb(self.proj())[0]
        return pd.DataFrame(
            nwb_lfp["lfp"].data,
            index=pd.Index(nwb_lfp["lfp"].timestamps, name="time"),
        )

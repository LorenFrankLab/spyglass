from spyglass.utils import SpyglassMixin
import datajoint as dj
from spyglass.utils.dj_merge_tables import _Merge

from spyglass.electrode_localization.v1.micro_ct import ChannelBrainLocationMicroCTV1 # noqa: F401
from spyglass.electrode_localization.v1.histology import ChannelBrainLocationHistologyV1 # noqa: F401

schema = dj.schema("electrode_localization_v1")

@schema
class ChannelBrainLocation(_Merge, SpyglassMixin):
    class HistologyV1(SpyglassMixin, dj.Part):
        definition = """
        -> master
        -> ChannelBrainLocationHistologyV1
        """"

    class MicroCTV1(SpyglassMixin, dj.Part):
        definition = """
        -> master
        -> ChannelBrainLocationMicroCTV1
        """
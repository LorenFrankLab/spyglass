"""Merge of electrode channel locations from histology and microCT pipelines."""

import datajoint as dj

from spyglass.electrode_localization.v1.histology import (  # noqa: F401
    ChannelBrainLocationHistologyV1,
)
from spyglass.electrode_localization.v1.micro_ct import (  # noqa: F401
    ChannelBrainLocationMicroCTV1,
)
from spyglass.utils import SpyglassMixin
from spyglass.utils.dj_merge_tables import _Merge

schema = dj.schema("electrode_localization_v1")


@schema
class ChannelBrainLocation(_Merge, SpyglassMixin):
    """Merge of electrode channel locations from histology and microCT pipelines.

    The master table lists each (subject, probe, channel), with parts:
      - HistologyV1: ground‐truth coordinates from manual histology alignment.
      - MicroCTV1: coregistered coordinates from the microCT volume.
    """

    class HistologyV1(SpyglassMixin, dj.Part):
        definition = """
        -> master
        -> ChannelBrainLocationHistologyV1
        """

    class MicroCTV1(SpyglassMixin, dj.Part):
        definition = """
        -> master
        -> ChannelBrainLocationMicroCTV1
        """

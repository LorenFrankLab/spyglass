import datajoint as dj

from spyglass.decoding.v1.clusterless import ClusterlessDecodingV1  # noqa: F401
from spyglass.decoding.v1.sorted_spikes import (
    SortedSpikesDecodingV1,
)  # noqa: F401
from spyglass.utils import SpyglassMixin, _Merge

schema = dj.schema("decoding_merge")


@schema
class DecodingOutput(_Merge, SpyglassMixin):
    definition = """
    merge_id: uuid
    ---
    source: varchar(32)
    """

    class ClusterlessDecodingV1(SpyglassMixin, dj.Part):
        definition = """
        -> master
        ---
        -> ClusterlessDecodingV1
        """

    class SortedSpikesDecodingV1(SpyglassMixin, dj.Part):
        definition = """
        -> master
        ---
        -> SortedSpikesDecodingV1
        """

import datajoint as dj
from spyglass.utils.dj_merge_tables import _Merge
from spyglass.decoding.v1.clusterless import ClusterlessDecodingV1
from spyglass.decoding.v1.sorted_spikes import SortedSpikesDecodingV1

schema = dj.schema("decoding_merge")


@schema
class DecodingOutput(_Merge):
    definition = """
    merge_id: uuid
    ---
    source: varchar(32)
    """

    class ClusterlessDecodingV1(dj.Part):
        definition = """
        -> master
        ---
        -> ClusterlessDecodingV1
        """

    class SortedSpikesDecodingV1(dj.Part):
        definition = """
        -> master
        ---
        -> SortedSpikesDecodingV1
        """

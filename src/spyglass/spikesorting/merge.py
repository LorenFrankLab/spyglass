import datajoint as dj

from spyglass.spikesorting.imported import ImportedSpikeSorting  # noqa: F401
from spyglass.spikesorting.spikesorting_curation import (  # noqa: F401
    CuratedSpikeSorting,
)
from spyglass.spikesorting.v1.curation import CurationV1  # noqa: F401
from spyglass.utils.dj_merge_tables import _Merge
from spyglass.utils.dj_mixin import SpyglassMixin

schema = dj.schema("spikesorting_merge")


@schema
class SpikeSortingOutput(_Merge):
    definition = """
    # Output of spike sorting pipelines.
    merge_id: uuid
    ---
    source: varchar(32)
    """

    class CurationV1(SpyglassMixin, dj.Part):  # noqa: F811
        definition = """
        -> master
        ---
        -> CurationV1
        """

    class ImportedSpikeSorting(SpyglassMixin, dj.Part):  # noqa: F811
        definition = """
        -> master
        ---
        -> ImportedSpikeSorting
        """

    class CuratedSpikeSorting(SpyglassMixin, dj.Part):  # noqa: F811
        definition = """
        -> master
        ---
        -> CuratedSpikeSorting
        """

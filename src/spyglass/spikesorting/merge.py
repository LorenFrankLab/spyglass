import datajoint as dj
import pandas as pd

from spyglass.common.common_filter import FirFilterParameters  # noqa: F401
from spyglass.common.common_interval import IntervalList  # noqa: F401
from spyglass.spikesorting.v1.artifact import ArtifactIntervalV1  # noqa: F401
from spyglass.utils.dj_merge_tables import _Merge

schema = dj.schema("spikesorting_merge")


@schema
class SpikeSortingOutput(_Merge):
    definition = """
    merge_id: uuid
    ---
    source: varchar(32)
    """

    class SpikeSortingV1(dj.Part):
        definition = """
        -> master
        ---
        -> SpikeSortingV1
        """

    class ImportedSpikeSorting(dj.Part):
        definition = """
        -> master
        ---
        -> ImportedSpikeSorting
        """

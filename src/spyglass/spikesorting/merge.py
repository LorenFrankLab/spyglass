import datajoint as dj
from datajoint.utils import to_camel_case

from spyglass.spikesorting.imported import ImportedSpikeSorting  # noqa: F401
from spyglass.spikesorting.spikesorting_curation import (  # noqa: F401
    CuratedSpikeSorting,
)
from spyglass.spikesorting.v1.curation import CurationV1  # noqa: F401
from spyglass.utils.dj_merge_tables import _Merge
from spyglass.utils.dj_mixin import SpyglassMixin

schema = dj.schema("spikesorting_merge")

source_class_dict = {
    "CuratedSpikeSorting": CuratedSpikeSorting,
    "ImportedSpikeSorting": ImportedSpikeSorting,
    "CurationV1": CurationV1,
}


@schema
class SpikeSortingOutput(_Merge, SpyglassMixin):
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

    def get_recording(cls, key):
        """get the recording associated with a spike sorting output"""
        recording_key = cls.merge_restrict(key).proj()
        query = (
            source_class_dict[
                to_camel_case(cls.merge_get_parent(key).table_name)
            ]
            & recording_key
        )
        return query.get_recording(recording_key)

    def get_sorting(cls, key):
        """get the sorting associated with a spike sorting output"""
        sorting_key = cls.merge_restrict(key).proj()
        query = (
            source_class_dict[
                to_camel_case(cls.merge_get_parent(key).table_name)
            ]
            & sorting_key
        )
        return query.get_sorting(sorting_key)

import uuid

import datajoint as dj

from spyglass.spikesorting.imported import ImportedSpikeSorting  # noqa: F401
from spyglass.spikesorting.spikesorting_curation import (  # noqa: F401
    CuratedSpikeSorting,
)
from spyglass.spikesorting.v1.curation import CurationV1  # noqa: F401
from spyglass.utils.dj_merge_tables import _Merge

schema = dj.schema("spikesorting_merge")


@schema
class SpikeSortingOutput(_Merge):
    definition = """
    # Output of spike sorting pipelines. Use `insert_from_sourcee` method to insert rows.
    merge_id: uuid
    ---
    source: varchar(32)
    """

    class CurationV1(dj.Part):  # noqa: F811
        definition = """
        -> master
        ---
        -> CurationV1
        """

    class ImportedSpikeSorting(dj.Part):  # noqa: F811
        definition = """
        -> master
        ---
        -> ImportedSpikeSorting
        """

    class CuratedSpikeSorting(dj.Part):  # noqa: F811
        definition = """
        -> master
        ---
        -> CuratedSpikeSorting
        """

    @classmethod
    def insert_from_source(cls, key, source="CurationV1"):
        """Insert a row from part table into SpikeSortingOutput

        Parameters
        ----------
        key : dict
            primary key of CurationV1 table
        Source : str (optional)
            Default is "CurationV1". Other options are "ImportedSpikeSorting"
            and "CuratedSpikeSorting". Also accepts the part-parent class.
        """
        if not isinstance(source, str):
            source = source.__name__

        insert_to = getattr(cls, source, None)
        if insert_to is None:
            raise ValueError(f"Source {source} is not a valid part table")

        key.update({"merge_id": uuid.uuid4(), "source": source})
        cls.insert1(key, ignore_extra_fields=True, skip_duplicates=True)
        insert_to.insert1(
            key,
            skip_duplicates=True,
            ignore_extra_fields=True,
        )
        return key

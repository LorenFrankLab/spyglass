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
    # Output of spike sorting pipelines. Use `insert_from_TableName` method to insert rows.
    merge_id: uuid
    ---
    source: varchar(32)
    """

    class CurationV1(dj.Part):
        definition = """
        -> master
        ---
        -> CurationV1
        """

    class ImportedSpikeSorting(dj.Part):
        definition = """
        -> master
        ---
        -> ImportedSpikeSorting
        """

    class CuratedSpikeSorting(dj.Part):
        definition = """
        -> master
        ---
        -> CuratedSpikeSorting
        """

    @classmethod
    def insert_from_CurationV1(cls, key):
        """Insert a row from CurationV1 of v1 into SpikeSortingOutput

        Parameters
        ----------
        key : dict
            primary key of CurationV1 table
        """
        key["merge_id"] = uuid.uuid4()
        key["source"] = "CurationV1"
        cls.insert1(key, ignore_extra_fields=True, skip_duplicates=True)
        cls.CurationV1.insert1(
            key,
            skip_duplicates=True,
            ignore_extra_fields=True,
        )
        return key

    @classmethod
    def insert_from_ImportedSpikeSorting(cls, key):
        """Insert a row from ImportedSpikeSorting into SpikeSortingOutput

        Parameters
        ----------
        key : dict
            primary key of ImportedSpikeSorting table
        """
        key["merge_id"] = uuid.uuid4()
        key["source"] = "ImportedSpikeSorting"
        cls.insert1(key, ignore_extra_fields=True, skip_duplicates=True)
        cls.ImportedSpikeSorting.insert1(
            key, ignore_extra_fields=True, skip_duplicates=True
        )
        return key

    @classmethod
    def insert_from_CuratedSpikeSorting(cls, key):
        """Insert a row from CuratedSpikeSorting of v0 into SpikeSortingOutput

        Parameters
        ----------
        key : dict
            primary key of CuratedSpikeSorting table
        """
        key["merge_id"] = uuid.uuid4()
        key["source"] = "CuratedSpikeSorting"
        cls.insert1(key, ignore_extra_fields=True, skip_duplicates=True)
        cls.CuratedSpikeSorting.insert1(
            key,
            skip_duplicates=True,
            ignore_extra_fields=True,
        )
        return key

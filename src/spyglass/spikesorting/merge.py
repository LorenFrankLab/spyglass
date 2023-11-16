import datajoint as dj
import uuid
from spyglass.spikesorting.v1.curation import CurationV1  # noqa: F401
from spyglass.spikesorting.imported import ImportedSpikeSorting  # noqa: F401
from spyglass.utils.dj_merge_tables import _Merge

schema = dj.schema("spikesorting_merge")


@schema
class SpikeSortingOutput(_Merge):
    definition = """
    merge_id: varchar(36)
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

    @classmethod
    def insert_from_CurationV1(cls, key):
        """Insert a row from CurationV1 into SpikeSortingOutput

        Parameters
        ----------
        key : dict
            primary key of CurationV1 table
        """
        key["merge_id"] = str(uuid.uuid4())
        key["source"] = "CurationV1"
        cls.insert1([key["merge_id"]], skip_duplicates=True)
        cls.CurationV1.insert1(key, skip_duplicates=True)
        return key

    @classmethod
    def insert_from_ImportedSpikeSorting(cls, key):
        """Insert a row from ImportedSpikeSorting into SpikeSortingOutput

        Parameters
        ----------
        key : dict
            primary key of ImportedSpikeSorting table
        """
        key["merge_id"] = str(uuid.uuid4())
        key["source"] = "ImportedSpikeSorting"
        cls.insert1([key["merge_id"]], skip_duplicates=True)
        cls.CurationV1.insert1(key, skip_duplicates=True)
        return key

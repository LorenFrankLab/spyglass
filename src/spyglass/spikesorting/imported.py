import datajoint as dj
from spyglass.common.common_session import Session

schema = dj.schema("spikesorting_imported")


@schema
class ImportedSpikeSorting(dj.Imported):
    definition = """
    -> Session
    ---
    object_id: varchar(32)
    """

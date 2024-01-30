import datajoint as dj

from spyglass.spikesorting.v1.unit_inclusion import (  # noqa: F401
    ImportedUnitInclusionV1,
)
from spyglass.utils.dj_merge_tables import _Merge
from spyglass.utils.dj_mixin import SpyglassMixin

schema = dj.schema("spikesorting_unit_inclusion_merge")


@schema
class UnitInclusionOutput(_Merge, SpyglassMixin):
    definition = """
    # Used to select units for downstream analysis.
    merge_id: uuid
    ---
    source: varchar(32)
    """

    class ImportedUnitInclusionV1(SpyglassMixin, dj.Part):  # noqa: F811
        definition = """
        -> master
        ---
        -> ImportedUnitInclusionV1
        """

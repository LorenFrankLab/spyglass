import datajoint as dj
from datajoint.utils import to_camel_case

from spyglass.spikesorting.spikesorting_merge import (
    SpikeSortingOutput,
)  # noqa: F401
from spyglass.utils.dj_mixin import SpyglassMixin

schema = dj.schema("spikesorting_v1_unit_inclusion")


@schema
class ImportedUnitInclusionV1(SpyglassMixin, dj.Manual):
    definition = """
    # Units selected from a sorting for downstream analysis
    -> SpikeSortingOutput.proj(spikesorting_merge_id="merge_id")
    unit_group_name : varchar(40)   # Name of unit group
    ---
    included_units_ind=NULL : blob  # Index of units included in this group or None for all units
    description : varchar(100)      # Description of the unit group
    """

    def insert_all_units(
        self,
        spikesorting_merge_ids: list[str],
    ):
        """Include all units for a given sorting"""
        insertion = []
        for merge_id in spikesorting_merge_ids:
            insertion.append(
                {
                    "spikesorting_merge_id": merge_id,
                    "unit_group_name": "all units",
                    "description": "all units",
                }
            )
        self.insert(insertion, skip_duplicates=True)

        from spyglass.spikesorting.unit_inclusion_merge import (
            UnitInclusionOutput,
        )

        part_name = UnitInclusionOutput._part_name(self.table_name)
        UnitInclusionOutput._merge_insert(
            insertion, part_name=part_name, skip_duplicates=True
        )

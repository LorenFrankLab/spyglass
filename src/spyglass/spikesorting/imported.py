import copy

import datajoint as dj
import pynwb
from datajoint.utils import to_camel_case

from spyglass.common.common_nwbfile import Nwbfile
from spyglass.common.common_session import Session  # noqa: F401
from spyglass.utils import SpyglassMixin, logger

schema = dj.schema("spikesorting_imported")


@schema
class ImportedSpikeSorting(SpyglassMixin, dj.Imported):
    definition = """
    -> Session
    ---
    object_id: varchar(40)
    """

    _nwb_table = Nwbfile

    def make(self, key):
        orig_key = copy.deepcopy(key)
        nwb_file_abs_path = Nwbfile.get_abs_path(key["nwb_file_name"])

        with pynwb.NWBHDF5IO(
            nwb_file_abs_path, "r", load_namespaces=True
        ) as io:
            nwbfile = io.read()
            if nwbfile.units:
                key["object_id"] = nwbfile.units.object_id
                self.insert1(key, skip_duplicates=True)
            else:
                logger.warn("No units found in NWB file")

        part_name = to_camel_case(self.table_name.split("__")[-1])
        # TODO: The next line belongs in a merge table function
        from spyglass.spikesorting.merge import SpikeSortingOutput

        SpikeSortingOutput._merge_insert(
            [orig_key], part_name=part_name, skip_duplicates=True
        )

    @classmethod
    def get_recording(cls, key):
        raise NotImplementedError(
            "Imported spike sorting does not have a `get_recording` method"
        )

    @classmethod
    def get_sorting(cls, key):
        raise NotImplementedError(
            "Imported spike sorting does not have a `get_sorting` method"
        )

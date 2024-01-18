import datajoint as dj
import pynwb

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

    def make(self, key):
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

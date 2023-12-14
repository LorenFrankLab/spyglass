import datajoint as dj
from spyglass.common.common_session import Session
from spyglass.common.common_nwbfile import Nwbfile
import pynwb

schema = dj.schema("spikesorting_imported")


@schema
class ImportedSpikeSorting(dj.Imported):
    definition = """
    -> Session
    ---
    object_id: varchar(32)
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
                print("No units found in NWB file")

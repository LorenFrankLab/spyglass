import datajoint as dj
import ndx_franklab_novela
import pynwb

from spyglass.common.common_device import DataAcquisitionDevice
from spyglass.common.common_lab import (
    Institution,
    Lab,
    LabMember,
    decompose_name,
)
from spyglass.common.common_nwbfile import Nwbfile
from spyglass.common.common_subject import Subject
from spyglass.utils import SpyglassIngestion, logger

schema = dj.schema("common_session")


@schema
class Session(SpyglassIngestion, dj.Imported):
    definition = """
    # Table for holding experimental sessions.
    # Note that each session can have multiple experimenters and data acquisition
    # devices. See DataAcquisitionDevice and Experimenter part tables below.
    -> Nwbfile
    ---
    -> [nullable] Subject
    -> [nullable] Institution
    -> [nullable] Lab
    session_id = NULL: varchar(200)
    session_description: varchar(2000)
    session_start_time: datetime
    timestamps_reference_time: datetime
    experiment_description = NULL: varchar(2000)
    """

    @property
    def _source_nwb_object_type(self):
        """The NWB object type from which this table can ingest data."""
        return pynwb.NWBFile

    @property
    def table_key_to_obj_attr(self):
        return {
            "self": {
                "institution_name": "institution",
                "lab_name": "lab",
                "session_id": "session_id",
                "session_description": "session_description",
                "session_start_time": "session_start_time",
                "timestamps_reference_time": "timestamps_reference_time",
                "experiment_description": "experiment_description",
            },
            "subject": {"subject_id": "subject_id"},
        }

    class DataAcquisitionDevice(SpyglassIngestion, dj.Part):
        definition = """
        # Part table linking Session to multiple DataAcquisitionDevice entries.
        -> Session
        -> DataAcquisitionDevice
        """

        @property
        def _source_nwb_object_type(self):
            return ndx_franklab_novela.DataAcqDevice

        @property
        def table_key_to_obj_attr(self):
            return {
                "self": {
                    "data_acquisition_device_name": "name",
                }
            }

    class Experimenter(SpyglassIngestion, dj.Part):
        definition = """
        # Part table linking Session to multiple LabMember entries.
        -> Session
        -> LabMember
        """

        @property
        def _source_nwb_object_type(self):
            return pynwb.NWBFile

        def generate_entries_from_nwb_object(
            self, nwb_obj: pynwb.NWBFile, base_key=None
        ):
            """Override to handle multiple experimenters."""
            base_key = base_key or dict()
            experimenter_list = nwb_obj.experimenter
            if not experimenter_list:
                logger.info("No experimenter metadata found for Session.\n")
                return dict()

            entries = []
            for experimenter in experimenter_list:
                _, first, last = decompose_name(experimenter)
                entries.append(
                    {
                        "lab_member_name": f"{first} {last}",
                        **base_key,
                    }
                )
            return {self: entries}

    def make(self, key):
        self.insert_from_nwbfile(**key)

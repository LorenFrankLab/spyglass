import datajoint as dj
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

    _source_nwb_object_type = pynwb.NWBFile

    def get_nwb_objects(self, nwb_file, nwb_file_name=None):
        """The file itself is the source: a session describes the whole file.

        Stated explicitly rather than leaning on the default type filter. That
        filter searches `nwb_file.objects`, so it only finds the root NWBFile
        if the file lists itself in its own object collection -- true today,
        but an implicit dependency on pynwb's internals for a table whose
        source is unambiguous. Same reasoning as PositionSource and
        RawPosition in common_behav.
        """
        return [nwb_file]

    table_key_to_obj_attr = {
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

    @property
    def with_date_str(self):
        """Project session_date_str (YYYYMMDD) from session_start_time.

        Returns a query expression with the primary key plus a computed
        ``session_date_str`` column, useful for date-based restriction::

            Session().with_date_str & {"session_date_str": "20230101"}
        """
        return (self & self.restriction).proj(
            subject_id="subject_id",
            session_date_str="DATE_FORMAT(session_start_time, '%%Y%%m%%d')",
        )

    class DataAcquisitionDevice(SpyglassIngestion, dj.Part):  # noqa: F811
        definition = """
        # Part table linking Session to multiple DataAcquisitionDevice entries.
        -> Session
        -> DataAcquisitionDevice
        """

        _source_nwb_object_type = "DataAcqDevice"

        table_key_to_obj_attr = {
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

        _source_nwb_object_type = pynwb.NWBFile

        def generate_entries_from_nwb_object(
            self, nwb_obj: pynwb.NWBFile, base_key=None
        ):
            """Override to handle multiple experimenters."""
            base_key = base_key or dict()
            experimenter_list = nwb_obj.experimenter
            if not experimenter_list:
                self._info_msg("No experimenter metadata found for Session.\n")
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

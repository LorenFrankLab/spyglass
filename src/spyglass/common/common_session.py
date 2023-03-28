import os
import datajoint as dj

from .common_device import CameraDevice, DataAcquisitionDevice, Probe
from .common_lab import Institution, Lab, LabMember
from .common_nwbfile import Nwbfile
from .common_subject import Subject
from ..utils.nwb_helper_fn import get_nwb_file, get_config

schema = dj.schema("common_session")


@schema
class Session(dj.Imported):
    definition = """
    # Table for holding experimental sessions.
    # Note that each session can have multiple experimenters and data acquisition devices. See DataAcquisitionDevice
    # and Experimenter part tables below.
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

    class DataAcquisitionDevice(dj.Part):
        definition = """
        # Part table that allows a Session to be associated with multiple DataAcquisitionDevice entries.
        -> Session
        -> DataAcquisitionDevice
        """

        # NOTE: as a Part table, it is generally advised not to delete entries directly
        # (see https://docs.datajoint.org/python/computation/03-master-part.html),
        # but you can use `delete(force=True)`.

    class Experimenter(dj.Part):
        definition = """
        # Part table that allows a Session to be associated with multiple LabMember entries.
        -> Session
        -> LabMember
        """

    def make(self, key):
        # These imports must go here to avoid cyclic dependencies
        # from .common_task import Task, TaskEpoch
        from .common_interval import IntervalList

        # from .common_ephys import Unit

        nwb_file_name = key["nwb_file_name"]
        nwb_file_abspath = Nwbfile.get_abs_path(nwb_file_name)
        nwbf = get_nwb_file(nwb_file_abspath)
        config = get_config(nwb_file_abspath)

        # certain data are not associated with a single NWB file / session because they may apply to
        # multiple sessions. these data go into dj.Manual tables.
        # e.g., a lab member may be associated with multiple experiments, so the lab member table should not
        # be dependent on (contain a primary key for) a session.

        # here, we create new entries in these dj.Manual tables based on the values read from the NWB file
        # then, they are linked to the session via fields of Session (e.g., Subject, Institution, Lab) or part
        # tables (e.g., Experimenter, DataAcquisitionDevice).

        print("Institution...")
        Institution().insert_from_nwbfile(nwbf)

        print("Lab...")
        Lab().insert_from_nwbfile(nwbf)

        print("LabMember...")
        LabMember().insert_from_nwbfile(nwbf)

        print("Subject...")
        Subject().insert_from_nwbfile(nwbf)

        print("Populate DataAcquisitionDevice...")
        DataAcquisitionDevice.insert_from_nwbfile(nwbf, config)
        print()

        print("Populate CameraDevice...")
        CameraDevice.insert_from_nwbfile(nwbf)
        print()

        print("Populate Probe...")
        Probe.insert_from_nwbfile(nwbf, config)
        print()

        if nwbf.subject is not None:
            subject_id = nwbf.subject.subject_id
        else:
            subject_id = None

        Session().insert1(
            {
                "nwb_file_name": nwb_file_name,
                "subject_id": subject_id,
                "institution_name": nwbf.institution,
                "lab_name": nwbf.lab,
                "session_id": nwbf.session_id,
                "session_description": nwbf.session_description,
                "session_start_time": nwbf.session_start_time,
                "timestamps_reference_time": nwbf.timestamps_reference_time,
                "experiment_description": nwbf.experiment_description,
            },
            skip_duplicates=True,
        )

        print("Skipping Apparatus for now...")
        # Apparatus().insert_from_nwbfile(nwbf)

        # interval lists depend on Session (as a primary key) but users may want to add these manually so this is
        # a manual table that is also populated from NWB files

        print("IntervalList...")
        IntervalList().insert_from_nwbfile(nwbf, nwb_file_name=nwb_file_name)

        # print('Unit...')
        # Unit().insert_from_nwbfile(nwbf, nwb_file_name=nwb_file_name)

        self._add_data_acquisition_device_part(nwb_file_name, nwbf, config)
        self._add_experimenter_part(nwb_file_name, nwbf)

    def _add_data_acquisition_device_part(self, nwb_file_name, nwbf, config):
        # get device names from both the NWB file and the associated config file
        device_names, _, _ = DataAcquisitionDevice.get_all_device_names(nwbf, config)

        for device_name in device_names:
            # ensure that the foreign key exists and do nothing if not
            query = DataAcquisitionDevice & {
                "data_acquisition_device_name": device_name
            }
            if len(query) == 0:
                print(
                    f"DataAcquisitionDevice with name {device_name} does not exist. "
                    "Cannot link Session with DataAcquisitionDevice in Session.DataAcquisitionDevice."
                )
                continue
            key = dict()
            key["nwb_file_name"] = nwb_file_name
            key["data_acquisition_device_name"] = device_name
            Session.DataAcquisitionDevice.insert1(key)

    def _add_experimenter_part(self, nwb_file_name, nwbf):
        if nwbf.experimenter is None:
            return

        for name in nwbf.experimenter:
            # ensure that the foreign key exists and do nothing if not
            query = LabMember & {"lab_member_name": name}
            if len(query) == 0:
                print(
                    f"LabMember with name {name} does not exist. "
                    "Cannot link Session with LabMember in Session.Experimenter."
                )
                continue

            key = dict()
            key["nwb_file_name"] = nwb_file_name
            key["lab_member_name"] = name
            Session.Experimenter.insert1(key)


@schema
class SessionGroup(dj.Manual):
    definition = """
    session_group_name: varchar(200)
    ---
    session_group_description: varchar(2000)
    """

    @staticmethod
    def add_group(
        session_group_name: str,
        session_group_description: str,
        *,
        skip_duplicates: bool = False,
    ):
        SessionGroup.insert1(
            {
                "session_group_name": session_group_name,
                "session_group_description": session_group_description,
            },
            skip_duplicates=skip_duplicates,
        )

    @staticmethod
    def update_session_group_description(
        session_group_name: str, session_group_description
    ):
        SessionGroup.update1(
            {
                "session_group_name": session_group_name,
                "session_group_description": session_group_description,
            }
        )

    @staticmethod
    def add_session_to_group(
        nwb_file_name: str, session_group_name: str, *, skip_duplicates: bool = False
    ):
        SessionGroupSession.insert1(
            {"session_group_name": session_group_name, "nwb_file_name": nwb_file_name},
            skip_duplicates=skip_duplicates,
        )

    @staticmethod
    def remove_session_from_group(nwb_file_name: str, session_group_name: str):
        query = {
            "session_group_name": session_group_name,
            "nwb_file_name": nwb_file_name,
        }
        (SessionGroupSession & query).delete()

    @staticmethod
    def delete_group(session_group_name: str):
        query = {"session_group_name": session_group_name}
        (SessionGroup & query).delete()

    @staticmethod
    def get_group_sessions(session_group_name: str):
        results = (
            SessionGroupSession & {"session_group_name": session_group_name}
        ).fetch(as_dict=True)
        return [{"nwb_file_name": result["nwb_file_name"]} for result in results]

    @staticmethod
    def create_spyglass_view(session_group_name: str):
        import figurl as fig

        FIGURL_CHANNEL = os.getenv("FIGURL_CHANNEL")
        assert FIGURL_CHANNEL, "Environment variable not set: FIGURL_CHANNEL"
        data = {"type": "spyglassview", "sessionGroupName": session_group_name}
        F = fig.Figure(view_url="gs://figurl/spyglassview-1", data=data)
        return F


# The reason this is not implemented as a dj.Part is that
# datajoint prohibits deleting from a subtable without
# also deleting the parent table.
# See: https://docs.datajoint.org/python/computation/03-master-part.html
@schema
class SessionGroupSession(dj.Manual):
    definition = """
    -> SessionGroup
    -> Session
    """

import datajoint as dj

from spyglass.common.common_device import (
    CameraDevice,
    DataAcquisitionDevice,
    Probe,
)
from spyglass.common.common_lab import Institution, Lab, LabMember
from spyglass.common.common_nwbfile import Nwbfile
from spyglass.common.common_subject import Subject
from spyglass.settings import config, debug_mode, test_mode
from spyglass.utils import SpyglassMixin, logger
from spyglass.utils.nwb_helper_fn import get_config, get_nwb_file

schema = dj.schema("common_session")


@schema
class Session(SpyglassMixin, dj.Imported):
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

    class DataAcquisitionDevice(SpyglassMixin, dj.Part):
        definition = """
        # Part table that allows a Session to be associated with multiple DataAcquisitionDevice entries.
        -> Session
        -> DataAcquisitionDevice
        """

        # NOTE: as a Part table, it is generally advised not to delete entries directly
        # (see https://docs.datajoint.org/python/computation/03-master-part.html),
        # but you can use `delete(force=True)`.

    class Experimenter(SpyglassMixin, dj.Part):
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

        # certain data are not associated with a single NWB file / session
        # because they may apply to multiple sessions. these data go into
        # dj.Manual tables. e.g., a lab member may be associated with multiple
        # experiments, so the lab member table should not be dependent on
        # (contain a primary key for) a session.

        # here, we create new entries in these dj.Manual tables based on the
        # values read from the NWB file then, they are linked to the session
        # via fields of Session (e.g., Subject, Institution, Lab) or part
        # tables (e.g., Experimenter, DataAcquisitionDevice).

        logger.info("Institution...")
        Institution().insert_from_nwbfile(nwbf)

        logger.info("Lab...")
        Lab().insert_from_nwbfile(nwbf)

        logger.info("LabMember...")
        LabMember().insert_from_nwbfile(nwbf)

        logger.info("Subject...")
        Subject().insert_from_nwbfile(nwbf)

        if not debug_mode:  # TODO: remove when demo files agree on device
            logger.info("Populate DataAcquisitionDevice...")
            DataAcquisitionDevice.insert_from_nwbfile(nwbf, config)

        logger.info("Populate CameraDevice...")
        CameraDevice.insert_from_nwbfile(nwbf)

        logger.info("Populate Probe...")
        Probe.insert_from_nwbfile(nwbf, config)

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

        logger.info("Skipping Apparatus for now...")
        # Apparatus().insert_from_nwbfile(nwbf)

        # interval lists depend on Session (as a primary key) but users may want to add these manually so this is
        # a manual table that is also populated from NWB files

        logger.info("IntervalList...")
        IntervalList().insert_from_nwbfile(nwbf, nwb_file_name=nwb_file_name)

        # logger.info('Unit...')
        # Unit().insert_from_nwbfile(nwbf, nwb_file_name=nwb_file_name)

        self._add_data_acquisition_device_part(nwb_file_name, nwbf, config)
        self._add_experimenter_part(nwb_file_name, nwbf)

    def _add_data_acquisition_device_part(self, nwb_file_name, nwbf, config):
        # get device names from both the NWB file and the associated config file
        device_names, _, _ = DataAcquisitionDevice.get_all_device_names(
            nwbf, config
        )

        for device_name in device_names:
            # ensure that the foreign key exists and do nothing if not
            query = DataAcquisitionDevice & {
                "data_acquisition_device_name": device_name
            }
            if len(query) == 0:
                logger.warn(
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
                logger.warn(
                    f"LabMember with name {name} does not exist. "
                    "Cannot link Session with LabMember in Session.Experimenter."
                )
                continue

            key = dict()
            key["nwb_file_name"] = nwb_file_name
            key["lab_member_name"] = name
            Session.Experimenter.insert1(key)


@schema
class SessionGroup(SpyglassMixin, dj.Manual):
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
        nwb_file_name: str,
        session_group_name: str,
        *,
        skip_duplicates: bool = False,
    ):
        if test_mode:
            skip_duplicates = True
        SessionGroupSession.insert1(
            {
                "session_group_name": session_group_name,
                "nwb_file_name": nwb_file_name,
            },
            skip_duplicates=skip_duplicates,
        )

    @staticmethod
    def remove_session_from_group(
        nwb_file_name: str, session_group_name: str, *args, **kwargs
    ):
        query = {
            "session_group_name": session_group_name,
            "nwb_file_name": nwb_file_name,
        }
        (SessionGroupSession & query).delete(
            force_permission=test_mode, *args, **kwargs
        )

    @staticmethod
    def delete_group(session_group_name: str, *args, **kwargs):
        query = {"session_group_name": session_group_name}
        (SessionGroup & query).delete(
            force_permission=test_mode, *args, **kwargs
        )

    @staticmethod
    def get_group_sessions(session_group_name: str):
        results = (
            SessionGroupSession & {"session_group_name": session_group_name}
        ).fetch(as_dict=True)
        return [
            {"nwb_file_name": result["nwb_file_name"]} for result in results
        ]

    @staticmethod
    def create_spyglass_view(session_group_name: str):
        import figurl as fig

        FIGURL_CHANNEL = config.get("FIGURL_CHANNEL")
        if not FIGURL_CHANNEL:
            raise ValueError("FIGURL_CHANNEL config/env variable not set")

        return fig.Figure(
            view_url="gs://figurl/spyglassview-1",
            data={
                "type": "spyglassview",
                "sessionGroupName": session_group_name,
            },
        )


# The reason this is not implemented as a dj.Part is that
# datajoint prohibits deleting from a subtable without
# also deleting the parent table.
# See: https://docs.datajoint.org/python/computation/03-master-part.html


@schema
class SessionGroupSession(SpyglassMixin, dj.Manual):
    definition = """
    -> SessionGroup
    -> Session
    """

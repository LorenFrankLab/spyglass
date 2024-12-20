import datajoint as dj

from spyglass.common.common_device import (
    CameraDevice,
    DataAcquisitionDevice,
    Probe,
)
from spyglass.common.common_lab import Institution, Lab, LabMember
from spyglass.common.common_nwbfile import Nwbfile
from spyglass.common.common_subject import Subject
from spyglass.settings import debug_mode
from spyglass.utils import SpyglassMixin, logger
from spyglass.utils.nwb_helper_fn import get_config, get_nwb_file

schema = dj.schema("common_session")


@schema
class Session(SpyglassMixin, dj.Imported):
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

    class DataAcquisitionDevice(SpyglassMixin, dj.Part):
        definition = """
        # Part table linking Session to multiple DataAcquisitionDevice entries.
        -> Session
        -> DataAcquisitionDevice
        """

        # NOTE: as a Part table, it is ill advised to delete entries directly
        # (https://docs.datajoint.org/python/computation/03-master-part.html),
        # but you can use `delete(force=True)`.

    class Experimenter(SpyglassMixin, dj.Part):
        definition = """
        # Part table linking Session to multiple LabMember entries.
        -> Session
        -> LabMember
        """

    def make(self, key):
        """Populate the Session table and others from an nwb file.

        Calls the insert_from_nwbfile method for each of the following tables:
            - Institution
            - Lab
            - LabMember
            - Subject
            - DataAcquisitionDevice
            - CameraDevice
            - Probe
            - IntervalList
        """
        # These imports must go here to avoid cyclic dependencies
        # from .common_task import Task, TaskEpoch
        from .common_interval import IntervalList

        # from .common_ephys import Unit

        nwb_file_name = key["nwb_file_name"]
        nwb_file_abspath = Nwbfile.get_abs_path(nwb_file_name)
        nwbf = get_nwb_file(nwb_file_abspath)
        config = get_config(nwb_file_abspath, calling_table=self.camel_name)

        # certain data are not associated with a single NWB file / session
        # because they may apply to multiple sessions. these data go into
        # dj.Manual tables. e.g., a lab member may be associated with multiple
        # experiments, so the lab member table should not be dependent on
        # (contain a primary key for) a session.

        # here, we create new entries in these dj.Manual tables based on the
        # values read from the NWB file then, they are linked to the session
        # via fields of Session (e.g., Subject, Institution, Lab) or part
        # tables (e.g., Experimenter, DataAcquisitionDevice).

        logger.info("Session populates Institution...")
        institution_name = Institution().insert_from_nwbfile(nwbf, config)

        logger.info("Session populates Lab...")
        lab_name = Lab().insert_from_nwbfile(nwbf, config)

        logger.info("Session populates LabMember...")
        LabMember().insert_from_nwbfile(nwbf, config)

        logger.info("Session populates Subject...")
        subject_id = Subject().insert_from_nwbfile(nwbf, config)

        if not debug_mode:  # TODO: remove when demo files agree on device
            logger.info("Session populates Populate DataAcquisitionDevice...")
            DataAcquisitionDevice.insert_from_nwbfile(nwbf, config)

        logger.info("Session populates Populate CameraDevice...")
        CameraDevice.insert_from_nwbfile(nwbf, config)

        logger.info("Session populates Populate Probe...")
        Probe.insert_from_nwbfile(nwbf, config)

        Session().insert1(
            {
                "nwb_file_name": nwb_file_name,
                "subject_id": subject_id,
                "institution_name": institution_name,
                "lab_name": lab_name,
                "session_id": nwbf.session_id,
                "session_description": nwbf.session_description,
                "session_start_time": nwbf.session_start_time,
                "timestamps_reference_time": nwbf.timestamps_reference_time,
                "experiment_description": nwbf.experiment_description,
            },
            skip_duplicates=True,
            allow_direct_insert=True,  # for populate_all_common
        )

        logger.info("Skipping Apparatus for now...")
        # Apparatus().insert_from_nwbfile(nwbf)

        # interval lists depend on Session (as a primary key) but users may
        # want to add these manually so this is a manual table that is also
        # populated from NWB files

        logger.info("Session populates IntervalList...")
        IntervalList().insert_from_nwbfile(nwbf, nwb_file_name=nwb_file_name)

        # logger.info('Unit...')
        # Unit().insert_from_nwbfile(nwbf, nwb_file_name=nwb_file_name)

        self._add_data_acquisition_device_part(nwb_file_name, nwbf, config)
        self._add_experimenter_part(nwb_file_name, nwbf, config)

    def _add_data_acquisition_device_part(self, nwb_file_name, nwbf, config={}):
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
                logger.warning(
                    "Cannot link Session with DataAcquisitionDevice.\n"
                    + f"DataAcquisitionDevice does not exist: {device_name}"
                )
                continue
            key = dict()
            key["nwb_file_name"] = nwb_file_name
            key["data_acquisition_device_name"] = device_name
            Session.DataAcquisitionDevice.insert1(key)

    def _add_experimenter_part(
        self, nwb_file_name: str, nwbf, config: dict = None
    ):
        # Use config file over nwb file
        config = config or dict()
        if members := config.get("LabMember"):
            experimenter_list = [
                member["lab_member_name"] for member in members
            ]
        elif nwbf.experimenter is not None:
            experimenter_list = nwbf.experimenter
        else:
            return

        for name in experimenter_list:
            # ensure that the foreign key exists and do nothing if not
            query = LabMember & {"lab_member_name": name}
            if len(query) == 0:
                logger.warning(
                    "Cannot link Session with LabMember. "
                    + f"LabMember does not exist: {name}"
                )
                continue

            key = dict()
            key["nwb_file_name"] = nwb_file_name
            key["lab_member_name"] = name
            Session.Experimenter.insert1(key)

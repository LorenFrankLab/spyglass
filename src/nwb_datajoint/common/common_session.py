import datajoint as dj
import warnings

from .common_device import DataAcquisitionDevice, CameraDevice, Probe
from .common_lab import Lab, Institution, LabMember
from .common_nwbfile import Nwbfile
from .common_subject import Subject
from .nwb_helper_fn import get_nwb_file

schema = dj.schema("common_session")

# TODO: figure out what to do about ExperimenterList


@schema
class Session(dj.Imported):
    definition = """
    # Table for holding experimental sessions.
    -> Nwbfile
    ---
    -> Subject
    -> Institution
    -> Lab
    session_id: varchar(80)
    session_description: varchar(80)
    session_start_time: datetime
    timestamps_reference_time: datetime
    experiment_description = NULL: varchar(80)
    """

    def make(self, key):
        # These imports must go here to avoid cyclic dependencies
        from .common_interval import IntervalList
        from .common_task import Task
        # from .common_ephys import Unit
        # TODO add Task

        nwb_file_name = key['nwb_file_name']
        nwb_file_abspath = Nwbfile.get_abs_path(nwb_file_name)
        nwbf = get_nwb_file(nwb_file_abspath)

        # certain data should not be associated with a single NWB file / session because they may apply to
        # multiple sessions. these data go into dj.Manual tables
        # e.g., a lab member may be associated with multiple experiments, so the lab member table should not
        # be dependent on (contain a primary key for) a session

        print('Institution...')
        Institution().insert_from_nwbfile(nwbf)

        print('Lab...')
        Lab().insert_from_nwbfile(nwbf)

        print('LabMember...')
        LabMember().insert_from_nwbfile(nwbf)

        print('Subject...')
        Subject().insert_from_nwbfile(nwbf)

        print('DataAcquisitionDevice...')
        DataAcquisitionDevice().insert_from_nwbfile(nwbf)

        print('CameraDevice...')
        CameraDevice().insert_from_nwbfile(nwbf)

        print('Probe...')
        Probe().insert_from_nwbfile(nwbf)

        print('Task...')
        Task().insert_from_nwbfile(nwbf)

        if nwbf.subject is not None and nwbf.subject.subject_id is not None:
            subject_id = nwbf.subject.subject_id
        else:
            subject_id = 'UNKNOWN'

        Session().insert1({
            'nwb_file_name': nwb_file_name,
            'subject_id': subject_id,
            'institution_name': nwbf.institution,
            'lab_name': nwbf.lab,
            'session_id': nwbf.session_id if nwbf.session_id is not None else 'UNKNOWN',
            'session_description': nwbf.session_description,
            'session_start_time': nwbf.session_start_time,
            'timestamps_reference_time': nwbf.timestamps_reference_time,
            'experiment_description': nwbf.experiment_description
        }, skip_duplicates=True)

        print('Skipping Apparatus for now...')
        # Apparatus().insert_from_nwbfile(nwbf)

        # interval lists depend on Session (as a primary key) but users may want to add these manually so this is
        # a manual table that is also populated from NWB files

        print('IntervalList...')
        IntervalList().insert_from_nwbfile(nwbf, nwb_file_name=nwb_file_name)

        # print('Unit...')
        # Unit().insert_from_nwbfile(nwbf, nwb_file_name=nwb_file_name)


@schema
class ExperimenterList(dj.Imported):
    definition = """
    -> Session
    """

    class Experimenter(dj.Part):
        definition = """
        -> ExperimenterList
        -> LabMember
        """

    def make(self, key):
        nwb_file_name = key['nwb_file_name']
        nwb_file_abspath = Nwbfile().get_abs_path(nwb_file_name)
        self.insert1({'nwb_file_name': nwb_file_name}, skip_duplicates=True)
        nwbf = get_nwb_file(nwb_file_abspath)

        if nwbf.experimenter is None:
            return

        for e in nwbf.experimenter:
            # check to see if the experimenter is in the lab member list, and if not add her / him
            if {'lab_member_name': e} not in LabMember():
                names = [x.strip() for x in e.split(' ')]
                labmember_dict = dict()
                labmember_dict['lab_member_name'] = e
                if len(names) == 2:
                    labmember_dict['first_name'] = names[0]
                    labmember_dict['last_name'] = names[1]
                else:
                    warnings.warn(f'Experimenter {e} does not seem to have a first and last name')
                    labmember_dict['first_name'] = 'unknown'
                    labmember_dict['last_name'] = 'unknown'
                LabMember().insert1(labmember_dict)
            # now insert the experimenter, which is a combination of the nwbfile and the name
            key = dict(
                nwb_file_name=nwb_file_name,
                lab_member_name=e
            )
            self.Experimenter().insert1(key)

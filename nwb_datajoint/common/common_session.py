import datajoint as dj
import pynwb

from .common_device import DataAcquisitionDevice, CameraDevice, Probe
from .common_lab import Lab, Institution, LabMember
from .common_nwbfile import Nwbfile
from .common_subject import Subject

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
    experiment_description: varchar(80)
    """

    def make(self, key):
        # These imports must go here to avoid cyclic dependencies
        from .common_task import Task, TaskEpoch
        from .common_interval import IntervalList
        # from .common_ephys import Unit

        nwb_file_name = key['nwb_file_name']
        nwb_file_abspath = Nwbfile.get_abs_path(nwb_file_name)
        with pynwb.NWBHDF5IO(path=nwb_file_abspath, mode='r') as io:
            nwbf = io.read()

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

            self.insert1({
                'nwb_file_name': nwb_file_name,
                'subject_id': nwbf.subject.subject_id,
                'institution_name': nwbf.institution,
                'lab_name': nwbf.lab,
                'session_id': nwbf.session_id if nwbf.session_id is not None else 'tmp_id',
                'session_description': nwbf.session_description,
                'session_start_time': nwbf.session_start_time,
                'timestamps_reference_time': nwbf.timestamps_reference_time,
                'experiment_description': nwbf.experiment_description
            }, skip_duplicates=True)

            print('Skipping Apparatus for now...')
            # Apparatus().insert_from_nwbfile(nwbf)

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
        with pynwb.NWBHDF5IO(path=nwb_file_abspath, mode='r') as io:
            nwbf = io.read()
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
                        print(
                            'Warning: experimenter {} does not seem to have a first and last name'.format(e))
                        labmember_dict['first_name'] = 'unknown'
                        labmember_dict['last_name'] = 'unknown'
                    LabMember().insert1(labmember_dict)
                # now insert the experimenter, which is a combination of the nwbfile and the name
                ExperimenterList().Experimenter().insert1({
                    'nwb_file_name': nwb_file_name,
                    'lab_member_name': e
                })

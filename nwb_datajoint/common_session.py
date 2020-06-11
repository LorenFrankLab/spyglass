import datajoint as dj

from .common_lab import LabMember

# [common_lab, common_subject, common_device]

schema = dj.schema("common_session")

@schema
class Session(dj.Imported):
    definition = """
    -> Nwbfile
    ---
    -> common_subject.Subject
    -> common_lab.Institution
    -> common_lab.Lab
    session_id: varchar(80)
    session_description: varchar(80)
    session_start_time: datetime
    timestamps_reference_time: datetime
    experiment_description: varchar(80)
    """

    class DataAcqDevice(dj.Part):
        definition = """
        -> Session
        -> common_device.Device
        """

    def insert_from_nwbfile(self, nwbf, *, nwb_file_name):
        key = dict(
            nwb_file_name=nwb_file_name
        )

        # populate the Session with information from the file
        key['subject_id'] = nwbf.subject.subject_id
        key['institution_name'] = nwbf.institution
        key['lab_name'] = nwbf.lab
        # Remove when bug fixed: session_id can be empty in current version
        key['session_id'] = nwbf.session_id
        if (key['session_id'] == None):
            key['session_id'] = 'tmp_id'
        key['session_description'] = nwbf.session_description
        key['session_start_time'] = nwbf.session_start_time
        key['experiment_description'] = nwbf.experiment_description
        key['timestamps_reference_time'] = nwbf.timestamps_reference_time
        self.insert1(key, skip_duplicates=True)

        ExperimenterList().insert1(dict(nwb_file_name=nwb_file_name))
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
            key['lab_member_name'] = e
            Experimenter().insert1(dict(
                nwb_file_name=nwb_file_name,
                lab_member_name=e
            ))

@schema
class Experimenter(dj.Imported):
    definition = """
    -> ExperimenterList
    -> common_lab.LabMember
    """

@schema
class ExperimenterList(dj.Imported):
    definition = """
    -> Session
    """
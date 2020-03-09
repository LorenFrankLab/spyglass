import datajoint as dj
import common_lab
import common_subject
import common_device

# we need to include pynwb and the franklab NWB namespace to be able to open the file
import pynwb

[common_lab, common_subject, common_device]

schema = dj.schema("common_session")

@schema
class Nwbfile(dj.Manual):
    definition = """
    nwb_file_name: varchar(80)
    ---
    """


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

    def make(self, key):
        try:
            io = pynwb.NWBHDF5IO(key['nwb_file_name'], mode='r')
            nwbf = io.read()
        except:
            print('Error in Session: nwbfile {} cannot be opened for reading\n'.format(key['nwb_file_name']))
            print(io.read())
            return

        # populate the Session with information from the file
        key['subject_id'] = nwbf.subject.subject_id
        key['institution_name'] = nwbf.institution
        key['lab_name'] = nwbf.lab
        key['session_id'] = nwbf.session_id
        key['session_description'] = nwbf.session_description
        key['session_start_time'] = nwbf.session_start_time
        key['experiment_description'] = nwbf.experiment_description
        key['timestamps_reference_time'] = nwbf.timestamps_reference_time
        self.insert1(key)

        # insert the devices
        devices = list(nwbf.devices.keys())
        for d in devices:
            Session.DataAcqDevice.insert1(dict(nwb_file_name=key['nwb_file_name'], device_name=d), skip_duplicates=True)
        io.close()

@schema
class ExperimenterList(dj.Imported):
    definition = """
    -> Session
    """
    class Experimenter(dj.Part):
        definition = """
        -> ExperimenterList
        -> common_lab.LabMember
        """

    def make(self, key):
        self.insert1(key)
        try:
            io = pynwb.NWBHDF5IO(key['nwb_file_name'], mode='r')
            nwbf = io.read()
        except:
            print('Error in Experimenter: nwbfile {} cannot be opened for reading\n'.format(key['nwb_file_name']))
            print(io.read())
            return

        for e in nwbf.experimenter:
            # check to see if the experimenter is in the lab member list, and if not add her / him
            if {'lab_member_name': e} not in common_lab.LabMember():
                names = [x.strip() for x in e.split(' ')]
                labmember_dict = dict()
                labmember_dict['lab_member_name'] = e
                if len(names) == 2:
                    labmember_dict['first_name'] = names[0]
                    labmember_dict['last_name'] = names[1]
                else:
                    print('Warning: experimenter {} does not seem to have a first and last name'.format(e))
                    labmember_dict['first_name'] = 'unknown'
                    labmember_dict['last_name'] = 'unknown'
                common_lab.LabMember.insert1(labmember_dict)
            # now insert the experimenter, which is a combination of the nwbfile and the name
            key['lab_member_name'] = e
            ExperimenterList.Experimenter.insert1(key)



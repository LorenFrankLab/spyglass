# we need to include pynwb and the franklab NWB namespace to be able to open the file
import pynwb

import datajoint as dj
import nwb_datajoint.common_device as common_device
import nwb_datajoint.common_lab as common_lab
import nwb_datajoint.common_subject as common_subject

import pynwb

# [common_lab, common_subject, common_device]

schema = dj.schema("common_session")

@schema
class Session(dj.Manual):
    definition = """
    nwb_file_name: varchar(80) # The name of the NWB file with the raw ephys data
    nwb_file_sha1: varchar(40) # SHA-1 hash of the raw NWB file    ---    
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

    def insert_from_nwb(self, nwb_file_name, nwb_file_sha1):
        key = dict(
            nwb_file_name=nwb_file_name,
            nwb_file_sha1=nwb_file_sha1
        )

        io = pynwb.NWBHDF5IO(nwb_file_name, mode='r')
        nwbf = io.read()

        devices = list(nwbf.devices.keys())
        device_dict = dict()
        for d in devices:
            # note that at present we skip the header_device from trodes rec files. We could add it in if at some
            # point we felt it was needed.
            if (d == 'data_acq_device'):
                # FIX: we need to get the right fields in the NWB device for this schema
                # device.Device.insert1(dict(device_name=d))
                device_dict['device_name'] = d
                device_dict['system'] = d.system
                device_dict['amplifier'] = d.amplifier
                device_dict['adc_circuit'] = d.circuit
                common_device.Device.insert1(device_dict, skip_duplicates=True)
        io.close()

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

        # insert the devices
        ''' Uncomment when devices correct 
        devices = list(nwbf.devices.keys())
        for d in devices:
            Session.DataAcqDevice.insert1(
                dict(nwb_file_name=key['nwb_file_name'], device_name=d), skip_duplicates=True)
        '''
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
            print('Error in Experimenter: nwbfile {} cannot be opened for reading\n'.format(
                key['nwb_file_name']))
            io.close()
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
                    print(
                        'Warning: experimenter {} does not seem to have a first and last name'.format(e))
                    labmember_dict['first_name'] = 'unknown'
                    labmember_dict['last_name'] = 'unknown'
                common_lab.LabMember.insert1(labmember_dict)
            # now insert the experimenter, which is a combination of the nwbfile and the name
            key['lab_member_name'] = e
            ExperimenterList.Experimenter.insert1(key)
        io.close()

# we need to include pynwb and the franklab NWB namespace to be able to open the file
import pynwb
import os

import datajoint as dj
import common_device
import common_lab
import common_subject
import shutil



[common_lab, common_subject, common_device]

schema = dj.schema("common_session")


@schema
class Nwbfile(dj.Manual):
    definition = """
    nwb_file_name: varchar(255) # the name of the nwb file (same as nwb_file_location)
    ---
    nwb_file_location: filepath@local  # the datajoint managed location of the NWB file
    nwb_raw_file_name: varchar(80) # The name of the NWB file with the raw ephys data
    """
    def insert_nwb_file(self, nwb_raw_file_name):
        '''
        Creates a copy of the raw NWB file without the raw data and then adds that file name to the schema while
        as the basis of analyses.
        :param nwb_raw_file_name: string - the name of the nwb file with raw data
        :return: nwb_file_name: string - the full path  of the copied field
        '''
        nwb_file_root_name, ext  = os.path.splitext(os.path.basename(nwb_raw_file_name))
        nwb_file_name = os.path.join(dj.config['stores']['local']['location'], nwb_file_root_name+'_pp.nwb')
        # TO DO: Create a copy of the NWB file, removing the electrical series object and replacing it with a link to
        # the raw file

        # TEMPORARY HACK: create a copy of the original file:
        if not os.path.exists(nwb_file_name):
            print('Copying file; this step will be removed once NWB export functionality works')
            shutil.copyfile(nwb_raw_file_name, nwb_file_name)
        key = dict()
        key['nwb_file_name'] = nwb_file_name
        key['nwb_file_location'] = nwb_file_name
        key['nwb_raw_file_name'] = nwb_raw_file_name
        self.insert1(key, skip_duplicates="True")


@schema
class LinkedNwbfile(dj.Manual):
    definition = """
    -> Nwbfile              
    linked_file_name: varchar(255) # the name of the linked file
    ---
    linked_file_location: filepath@local  # the datajoint managed location of the linked file
    """

    def __init__(self, *args):
        # do your custom stuff here
        super().__init__(*args)  # call the base implementation


    def create(self, nwb_file_name):
        '''
        Opens the input NWB file, creates a copy, writes out the copy to disk and return the name of the new file
        :param nwb_file_name:
        :return: linked_file_name - the name of the linked file
        '''
        #
        in_io  = pynwb.NWBHDF5IO(nwb_file_name, 'r')
        nwbf_in = in_io.read()
        nwbf_out = nwbf_in.copy()

        key = dict()
        key['nwb_file_name'] = nwb_file_name
        # get the current number of linked files
        #n_linked_files = len((LinkedNwbfile() & {'nwb_file_name' : nwb_file_name}).fetch())
        # name the file, adding the number of links with preceeding zeros

        # the init function is called everytime this object is accessed by DataJoint, so to set a variable we have to
        # do it here.

        n_linked_files = len(LinkedNwbfile())
        nwb_out_file_name = os.path.splitext(nwb_file_name)[0] + str(n_linked_files).zfill(8) + '.nwb'
        key['linked_file_name'] = nwb_out_file_name
        key['linked_file_location'] = nwb_out_file_name
        # write the linked file
        print(f'writing new NWB file {nwb_out_file_name}')
        with pynwb.NWBHDF5IO(nwb_out_file_name, 'a', manager=in_io.manager) as io:
            io.write(nwbf_out)

        in_io.close()
        # insert the key into the Linked File table
        self.insert1(key)
        print('inserted file')

        return nwb_out_file_name

    def get_name_without_create(self, nwb_file_name):
        '''
        Returns the name of a new NWB linked NWB file and adds it to the table but does NOT create it. This is
        currently necessary because of a bug in the HDMF library that prevents multiple appends to an NWB file and
        should be removed when that bug is fixed.
        :param nwb_file_name:
        :return: linked_file_name - the name of the file that can be linked
        '''
        key = dict()
        key['nwb_file_name'] = nwb_file_name
        n_linked_files = len(LinkedNwbfile())
        linked_file_name = os.path.splitext(nwb_file_name)[0] + str(n_linked_files).zfill(8) + '.nwb'
        key['linked_file_name'] = linked_file_name
        self.insert1(key)
        return linked_file_name


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
            print('Error in Session: nwbfile {} cannot be opened for reading\n'.format(
                key['nwb_file_name']))
            print(io.read())
            io.close()
            return

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
        self.insert1(key)

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

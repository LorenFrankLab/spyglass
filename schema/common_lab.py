# Frank lab schema: lab name and members
import datajoint as dj
import pynwb

schema = dj.schema("common_lab", locals())


@schema
class LabMember(dj.Lookup):
    definition = """
    lab_member_name: varchar(80)
    ---
    first_name: varchar(80)
    last_name: varchar(80)
    """
    def insert_from_nwb(self, nwb_file_name):
        '''
        Insert lab member information from NWB file
        :param nwb_file_name:
        :return: None
        '''
        labmember_dict = dict()
        with pynwb.NWBHDF5IO(nwb_file_name, 'r') as io:
            nwbf = io.read()
            for labmember in nwbf.experimenter:
                labmember_dict['lab_member_name'] = str(labmember)
                labmember_dict['first_name'] = str.split(labmember)[0]
                labmember_dict['last_name'] = str.split(labmember)[-1]
                self.insert1(labmember_dict, skip_duplicates="True")


@schema
class Institution(dj.Manual):
    definition = """
    institution_name: varchar(80)
    ---
    """
    def insert_from_nwb(self, nwb_file_name):
        '''
        Insert institution information from NWB file
        :param nwb_file_name:
        :return: None
        '''
        with pynwb.NWBHDF5IO(nwb_file_name, 'r') as io:
            nwbf = io.read()
            self.insert1(dict(institution_name=nwbf.institution), skip_duplicates=True)

@schema
class Lab(dj.Manual):
    definition = """
    lab_name: varchar(80)
    ---
    """
    def insert_from_nwb(self, nwb_file_name):
        '''
        Insert Lab name information from NWB file
        :param nwb_file_name:
        :return: None
        '''
        with pynwb.NWBHDF5IO(nwb_file_name, 'r') as io:
            nwbf = io.read()
            self.insert1(dict(lab_name=nwbf.lab), skip_duplicates=True)
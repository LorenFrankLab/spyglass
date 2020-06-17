import os
import datajoint as dj
import pynwb

schema = dj.schema("common_lab", locals())

import kachery as ka

# TODO: make decision about docstring -- probably use :param ...:

@schema
class Nwbfile(dj.Manual):
    definition = """
    nwb_file_name: varchar(400)
    ---
    nwb_file_sha1: varchar(40)
    """
    def insert_from_relative_file_name(self, nwb_file_name):
        """Insert a new session from an existing nwb file.

        Args:
            nwb_file_name (str): Relative path to the nwb file
        """
        nwb_file_abspath = Nwbfile.get_abs_path(nwb_file_name)
        assert os.path.exists(nwb_file_abspath), f'File does not exist: {nwb_file_abspath}'

        print('Computing SHA-1 and storing in kachery...')
        with ka.config(use_hard_links=True):
            kachery_path = ka.store_file(nwb_file_abspath)
            sha1 = ka.get_file_hash(kachery_path)
        
        self.insert1(dict(
            nwb_file_name=nwb_file_name,
            nwb_file_sha1=sha1
        ), skip_duplicates=True)
    
    @staticmethod
    def get_abs_path(nwb_file_name):
        base_dir = os.getenv('NWB_DATAJOINT_BASE_DIR', None)
        assert base_dir is not None, 'You must set NWB_DATAJOINT_BASE_DIR or provide the base_dir argument'

        nwb_file_abspath = os.path.join(base_dir, nwb_file_name)
        return nwb_file_abspath

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
        in_io  = pynwb.NWBHDF5IO(path=nwb_file_name, mode='r')
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
        with pynwb.NWBHDF5IO(path=nwb_out_file_name, mode='a', manager=in_io.manager) as io:
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


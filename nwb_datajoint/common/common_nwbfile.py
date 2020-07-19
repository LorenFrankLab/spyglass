import os
import datajoint as dj
import pynwb

schema = dj.schema("common_lab", locals())

import kachery as ka

# define the fields that should be kept in AnalysisNWBFiles
nwb_keep_fields = ('devices', 'electrode_groups', 'electrodes', 'experiment_description', 'experimenter', 'file_create_date', 'institution', 'lab', 'session_description', 'session_id', 
                   'session_start_time', 'subject', 'timestamps_reference_time')

# TODO: make decision about docstring -- probably use :param ...:

@schema
class Nwbfile(dj.Manual):
    definition = """
    nwb_file_name: varchar(255) #the name of the NWB file
    ---
    nwb_file_sha1: varchar(40) # the sha1 hash of the NWB file for kachery
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
class AnalysisNwbfile(dj.Manual):
    definition = """   
    analysis_file_name: varchar(255) # the name of the file 
    ---
    -> Nwbfile # the name of the parent NWB file. Used for naming and metadata copy
    analysis_file_description='': varchar(255) # an optional description of this analysis
    analysis_file_sha1='': varchar(40) # the sha1 hash of the NWB file for kachery
    analysis_parameters=NULL: blob # additional relevant parmeters. Currently used only for analyses that span multiple NWB files
    """

    def __init__(self, *args):
        # do your custom stuff here
        super().__init__(*args)  # call the base implementation


    def create(self, nwb_file_name):
        '''
        Opens the input NWB file, creates a copy, writes out the copy to disk and return the name of the new file
        :param nwb_file_name:
        :return: file_name - the name of the new NWB file
        '''
        #
        nwb_file_abspath = Nwbfile.get_abs_path(nwb_file_name)

        in_io  = pynwb.NWBHDF5IO(path=nwb_file_abspath, mode='r')
        nwbf_in = in_io.read()
        
        #  pop off the unnecessary elements to save space
        nwb_fields = nwbf_in.fields
        for field in nwb_fields:
            if field not in nwb_keep_fields:
                nwb_object = getattr(nwbf_in, field)
                if type(nwb_object) is dict:
                    for module in list(nwb_object.keys()):
                        nwb_object.pop(module)

        key = dict()
        key['nwb_file_name'] = nwb_file_name
        # get the current number of analysis files related to this nwb file
        n_analysis_files = len((AnalysisNwbfile() & {'parent_nwb_file': nwb_file_name}).fetch())
        # name the file, adding the number of files with preceeding zeros

        analysis_file_name = os.path.splitext(nwb_file_name)[0] + str(n_analysis_files).zfill(8) + '.nwb'
        key['analysis_file_name'] = analysis_file_name
        key['analysis_file_description'] = ''
        # write the new file
        print(f'writing new NWB file {analysis_file_name}')
        analysis_file_abspath = AnalysisNwbfile.get_abs_path(analysis_file_name)
        with pynwb.NWBHDF5IO(path=analysis_file_abspath, mode='w') as io:
            io.write(nwbf_in)

        in_io.close()
        # insert the key into the Linked File table
        self.insert1(key)
        print('inserted file')

        return analysis_file_name

    @staticmethod
    def get_abs_path(analysis_nwb_file_name):
        base_dir = os.getenv('NWB_DATAJOINT_BASE_DIR', None)
        assert base_dir is not None, 'You must set NWB_DATAJOINT_BASE_DIR or provide the base_dir argument'

        analysis_nwb_file_abspath = os.path.join(base_dir, 'analysis', analysis_nwb_file_name)
        return analysis_nwb_file_abspath

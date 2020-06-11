import os
import datajoint as dj

schema = dj.schema("common_lab", locals())

import kachery as ka

@schema
class Nwbfile(dj.Imported):
    definition = """
    nwb_file_name: varchar(80)
    ---
    nwb_file_sha1: varchar(40)
    """
    def insert_from_file_name(self, nwb_file_name, *, base_dir=None):
        '''
        Insert NWB file from file name (relative path)
        :param nwb_file_name:
        :param base_dir:
        :return: None
        '''

        if base_dir is None:
            base_dir = os.getenv('NWB_DATAJOINT_BASE_DIR', None)
        assert base_dir is not None, 'You must set NWB_DATAJOINT_BASE_DIR or provide the base_dir argument'
        nwb_file_abspath = os.path.join(base_dir, nwb_file_name)
        assert os.path.exists(nwb_file_abspath), f'File does not exist: {nwb_file_abspath}'
        assert os.getenv('KACHERY_STORAGE_DIR', None), 'You must set the KACHERY_STORAGE_DIR environment variable.'

        print('Computing SHA-1 and storing in kachery...')
        with ka.config(use_hard_links=True):
            kachery_path = ka.store_file(nwb_file_abspath)
            sha1 = ka.get_file_hash(kachery_path)
        
        self.insert1(dict(
            nwb_file_name=nwb_file_name,
            nwb_file_sha1=sha1
        ))

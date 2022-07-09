import os
import stat
import pathlib
import random
import string
from warnings import WarningMessage

import datajoint as dj
from matplotlib import projections
import pandas as pd
import numpy as np
import pynwb
import spikeinterface as si
import kachery_cloud as kcl
from kachery_cloud.TaskBackend import TaskClient

from hdmf.common import DynamicTable

from ..common.common_nwbfile import Nwbfile, AnalysisNwbfile


schema = dj.schema('sharing_kachery')

def kachery_request_upload(uri: str):
    """generate a kachery task request to upload the specified uri to the cloud

    Parameters
    ----------
    uri : str
        the uri of the requested file

    Returns when task completes     
    """
    project_id = os.environ['KACHERY_CLOUD_PROJECT'] if 'KACHERY_CLOUD_PROJECT' in os.environ else None
    task_client = TaskClient(project_id=project_id)
    print('Requesting upload task')
    task_client.request_task(
        task_type='action',
        task_name='kachery_store_shared_file.1',
        task_input={
            'uri': uri,
        }
    )
    return


def kachery_download_file(uri: str, dest:str):
    """downloads the specified uri from using kachery cloud.
    First tries to download directly, and if that fails, starts an upload request for the file and then downloads it

    Parameters
    ----------
    uri : str
        the uri of the requested file
    dest : str
        the full path for the downloaded file

    Returns 
        str
            The path to the downloaded file or None if the download was unsucessful    
    """
    fname = kcl.load_file(uri, dest=dest)
    if fname is None:
        # if we can't load the uri directly, it should be because it is not in the cloud, so we need to start a task to load it
        kcl.request_file_experimental(uri=uri, project_id=os.environ['KACHERY_CLOUD_PROJECT'])
        if not kcl.load_file(uri, dest=dest):
            print('Error: analysis file uri is in database but file cannot be downloaded')
            return False
    print('file requested')
    return True


@schema
class KacherySharingGroup(dj.Manual):
    definition = """
    group_name: varchar(200) # the name of the group we are sharing with
    ---
    description: varchar(200) # description of this group
    access_group_id = '': varchar(100) # the id for this group on http://cloud.kacheryhub.org/. Leaving this empty implies that the group is public. 
    """

@schema
class NwbfileKacherySelection(dj.Manual):
    definition="""
    -> KacherySharingGroup
    -> Nwbfile
    """

@schema
class NwbfileKachery(dj.Computed):
    definition = """
    -> NwbfileKacherySelection
    ---
    nwb_file_uri='': varchar(200)  # the uri for underscore NWB file for kachery. This is left empty for non-public files
    nwb_file_enc_uri='': varchar(200) # the encyrpted uri for the underscore NWB file for kachery. This is left empty for public files 
    """
    class LinkedFile(dj.Part):
        definition = """
        -> NwbfileKachery
        linked_file_rel_path: varchar(200) # the relative to the linked data file (assumes base path of SPYGLASS_BASE_DIR environment variable)
        ---
        linked_file_uri='': varchar(200) # the uri for the linked file
        linked_file_enc_uri='': varchar(200) # the encrypted uri for the linked file

        """
    def make(self, key):
        # note that we're assuming that the user has initialized a kachery-cloud client with kachery-cloud-init
        linked_key = key
        print(f'Linking {key["nwb_file_name"]} and storing in kachery-cloud...')
        nwb_abs_path = Nwbfile().get_abs_path(key['nwb_file_name'])
        uri = kcl.link_file(nwb_abs_path)
        access_group = (KacherySharingGroup & {'group_name' : key['group_name']}).fetch1('access_group_id')
        if access_group != '':
            key['nwb_file_enc_uri'] = kcl.encrypt_uri(uri, access_group=access_group)
        else:  
            key['nwb_file_uri'] = uri
        self.insert1(key)

        # we also need to insert the original NWB file. 
        # TODO: change this to automatically detect all linked files
        # For the moment, remove the last character ('_') and add the extension
        linked_file_path = os.path.splitext(nwb_abs_path)[0][:-1] + '.nwb'
        uri = kcl.link_file(linked_file_path)
        if access_group != '':
            linked_key['linked_file_enc_uri'] = kcl.encrypt_uri(uri, access_group=access_group)
        else:  
            linked_key['linked_file_uri'] = uri
        linked_key['linked_file_rel_path'] = str.replace(linked_file_path, os.environ['SPYGLASS_BASE_DIR'], '')
        self.LinkedFile.insert1(linked_key)

    
    @staticmethod
    def download_file(nwb_file_name: str):
        """Download the specified nwb file and associated linked files from kachery-cloud if possible

        Parameters
        ----------
        nwb_file_name : str
            The name of the nwb file
        
        Returns
        ----------
        bool
            True if the file was successfully downloaded, false otherwise
        """
        nwb_uri, nwb_enc_uri = (NwbfileKachery & {'nwb_file_name' : nwb_file_name}).fetch1('nwb_file_uri', 'nwb_file_enc_uri')
        if nwb_enc_uri != '':
            # decypt the URI
            uri = kcl.decrypt_uri(nwb_enc_uri)
        else:
            uri = nwb_uri 
        print(f'attempting to download uri {uri}')

        if not kachery_download_file(uri=uri, dest=Nwbfile.get_abs_path(nwb_file_name)):
            raise Exception(f'{Nwbfile.get_abs_path(nwb_file_name)} cannot be downloaded')
            return False
        # now download the linked file(s)
        linked_files = (NwbfileKachery.LinkedFile & {'nwb_file_name' : nwb_file_name}).fetch(as_dict=True)
        for file in linked_files:
            if file['linked_file_enc_uri'] != '':
                uri = kcl.decrypt_uri(file['linked_file_enc_uri'])
            else:
                uri = file['linked_file_uri']
            print(f'attempting to download linked file uri {uri}')
            if not kachery_download_file(uri=uri, dest=file['linked_file_path']):
                raise Exception(f'{Nwbfile.get_abs_path(nwb_file_name)} cannot be downloaded')
                return False
            
        return True

        
@schema
class AnalysisNwbfileKacherySelection(dj.Manual):
    definition="""
    -> KacherySharingGroup
    -> AnalysisNwbfile
    """

@schema
class AnalysisNwbfileKachery(dj.Computed):
    definition = """
    -> AnalysisNwbfileKacherySelection
    ---
    analysis_file_uri='': varchar(200)  # the uri of the file (for public sharing)
    analysis_file_enc_uri='': varchar(200) # the encrypted uri of the file (for private sharing)
    """

    class LinkedFile(dj.Part):
        definition = """
        -> AnalysisNwbfileKachery
        linked_file_name: varchar(200) # the name of the linked data file
        ---
        linked_file_uri='': varchar(200) # the uri for the linked file
        linked_file_enc_uri='': varchar(200) # the encrypted uri for the linked file
        """

    def make(self, key):
       # note that we're assuming that the user has initialized a kachery-cloud client with kachery-cloud-init
        linked_key = key
        print(f'Linking {key["analysis_file_name"]} and storing in kachery-cloud...')
        uri = kcl.link_file(AnalysisNwbfile().get_abs_path(key['analysis_file_name']))
        access_group = (KacherySharingGroup & {'group_name' : key['group_name']}).fetch1('access_group_id')
        if access_group != '':
            key['analysis_file_enc_uri'] = kcl.encrypt_uri(uri, access_group=access_group)
        else:  
            key['analysis_file_uri'] = uri
        self.insert1(key)
         
        # we also need to insert any linked files
        # TODO: change this to automatically detect all linked files
        #self.LinkedFile.insert1(key)

    @staticmethod
    def download_file(analysis_file_name: str):
        """Download the specified analysis file and associated linked files from kachery-cloud if possible

        Parameters
        ----------
        analysis_file_name : str
            The name of the analysis file
        
        Returns
        ----------
        bool
            True if the file was successfully downloaded, false otherwise
        """
        analysis_uri, analysis_enc_uri = (AnalysisNwbfileKachery & {'analysis_file_name' : analysis_file_name}).fetch1('analysis_file_uri', 'analysis_file_enc_uri')
        if analysis_enc_uri != '':
            # decypt the URI
            uri = kcl.decrypt_uri(analysis_enc_uri)
        else:
            uri = analysis_uri 
        print(f'attempting to download uri {uri}')

        if not kachery_download_file(uri=uri, dest=AnalysisNwbfile.get_abs_path(analysis_file_name)):
            raise Exception(f'{AnalysisNwbfile.get_abs_path(analysis_file_name)} cannot be downloaded')
            return False
        # now download the linked file(s)
        linked_files = (AnalysisNwbfileKachery.LinkedFile & {'analysis_file_name' : analysis_file_name}).fetch(as_dict=True)
        for file in linked_files:
            if file['linked_file_enc_uri'] != '':
                uri = kcl.decrypt_uri(file['linked_file_enc_uri'])
            else:
                uri = file['linked_file_uri']
            print(f'attempting to download linked file uri {uri}')
            if not kachery_download_file(uri=uri, dest=file['linked_file_path']):
                raise Exception(f'{AnalysisNwbfile.get_abs_path(analysis_file_name)} cannot be downloaded')
                return False
            
        return True

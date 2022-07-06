import os
import stat
import pathlib
import random
import string
from warnings import WarningMessage

import datajoint as dj
import pandas as pd
import numpy as np
import pynwb
import spikeinterface as si
import kachery_cloud as kcl

from hdmf.common import DynamicTable

from ..common.common_nwbfile import Nwbfile, AnalysisNwbfile


schema = dj.schema('sharing_kachery')

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
        linked_file_path: varchar(200) # the path to the linked data file
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
        key['linked_file_path'] = os.path.splitext(nwb_abs_path)[0][:-1] + '.nwb'
        uri = kcl.link_file(key['linked_file_path'])
        if access_group != '':
            linked_key['linked_file_enc_uri'] = kcl.encrypt_uri(uri, access_group=access_group)
        else:  
            linked_key['linked_file_uri'] = uri
        self.LinkedFile.insert1(key)
    
    @staticmethod
    def download_file(nwb_file_name: str):
        """Download the specified Nwbfile and associated linked files from kachery-cloud if possible

        Parameters
        ----------
        nwb_file_name : str
            The name of the NWB file
        
        Returns
        ----------
        bool
            True if the file was successfully downloaded, false otherwise
        """
        nwb_uri, nwb_enc_uri = (NwbfileKachery & {'nwb_file_name' : nwb_file_name}).fetch('nwb_file_uri', 'nwb_file_enc_uri')
        uri = nwb_uri if nwb_uri != '' else nwb_enc_uri
        if uri != '':
            if not kcl.load_file(uri, dest=nwb_file_name):
                Warning('nwb file uri is in database but file cannot be downloaded')
                return False
        else:
            Warning(f'nwb file uri for {nwb_file_name} is not in database ')
            return False

        # now download the linked file(s)
        linked_files = (NwbfileKachery.LinkedFile & {'nwb_file_name' : nwb_file_name}).fetch(as_dict=True)
        for file in linked_files:
            uri = file['linked_file_uri'] if file['linked_file_uri'] != '' else file['linked_file_enc_uri']
            if uri != '':
                if not kcl.load_file(uri, dest=file['linked_file_path']):
                    Warning('nwb linked file uri is in database but file cannot be downloaded')
                    return False
            else:
                Warning(f'nwb file uri for {nwb_file_name} is not in database ')
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
        """Download the specified analysisfile and associated linked files from kachery-cloud if possible

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
        uri = analysis_uri if analysis_uri != '' else analysis_enc_uri
        print(f'attempting to download uri {uri}')
        
        if uri != '':
            if not kcl.load_file(uri, dest=AnalysisNwbfile.get_abs_path(analysis_file_name)):
                print('Error: analysis file uri is in database but file cannot be downloaded')
                return False
        else:
            print(f'Warning: analysis file uri for {analysis_file_name} is not in database ')
            return False

        # now download the linked file(s)
        linked_files = (AnalysisNwbfileKachery.LinkedFile & {'analysis_file_name' : analysis_file_name}).fetch(as_dict=True)
        for file in linked_files:
            uri = file['linked_file_uri'] if file['linked_file_uri'] != '' else file['linked_file_enc_uri']
            if uri != '':
                if not kcl.load_file(uri, dest=file['linked_file_path']):
                    print('Error: analysis linked file uri is in database but file cannot be downloaded')
                    return False
            else:
                print(f'Warning: analysis file uri for {analysis_file_name} is not in database ')
                return False
        return True

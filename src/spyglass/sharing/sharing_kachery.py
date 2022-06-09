import os
import stat
import pathlib
import random
import string
from typing_extensions import Self
from warnings import WarningMessage

import datajoint as dj
import pandas as pd
import numpy as np
import pynwb
import spikeinterface as si
import kachery_client as kc

from hdmf.common import DynamicTable

from ..common.common_nwbfile import Nwbfile, AnalysisNwbfile


schema = dj.schema('share_kachery')

@schema
class KacheryChannel(dj.Manual):
    definition = """
    channel_name: varchar(200) # the name of the channel
    ---
    description: varchar(200) # description of this channel
    """
    def get_channel(self):
        """Returns a dictionary key for the KacheryChannel entry corresponding to the currently defined channel based on environment variables

        Returns
        -------
        dict
            dictionary key for a KacheryChannel
        """
        try:
            host = os.environ["KACHERY_DAEMON_HOST"]
            port = os.environ["KACHERY_DAEMON_PORT"]
        except:
            raise ValueError("Kachery environment variables are not set; refer to spyglass setup documents to fix.")
            
        try:
            return (self & {'host' : host, 'port': port}).fetch1("KEY")
        except:
            WarningMessage(f'Kachery channel for host {host} and port {port} must be in the KacheryChannel table')
            return None

    @staticmethod
    def set_channel(key):
        """Sets the environment variables to use the specified channel

        Parameters
        ----------
        key : dict
            key to a KacheryChannel entry
        
        returns True if successful, False otherwise
        """
        # get the host and port from the environment variables.
        host_env_var = "KACHERY_" + str(key["channel_name"]).upper() + "_HOST"
        port_env_var = "KACHERY_" + str(key["channel_name"]).upper() + "_PORT"
        try:
            os.environ["KACHERY_DAEMON_HOST"] = os.environ[host_env_var]
            os.environ["KACHERY_DAEMON_PORT"] = os.environ[port_env_var]
            return True
        except:
            print(f'Error: environment variable {host_env_var} and/or {port_env_var} not set')
            return False

    def channel_valid(self, key): 
        """Returns True if the specified channel is valid on this machine

        Parameters
        ----------
        key : dict
            key to a KacheryChannel entry
        """
        current_channel = self.get_channel()
        self.set_channel(key)
        # TODO: check whether we can access the kachery daemon
    

@schema
class NwbfileKacherySelection(dj.Manual):
    definition="""
    -> KacheryChannel
    -> Nwbfile
    """

@schema
class NwbfileKachery(dj.Computed):
    definition = """
    -> NwbfileKacherySelection
    ---
    nwb_file_uri: varchar(200)  # the uri the underscore NWB file for kachery
    nwb_raw_file_uri: varchar(200) # the uri of the original NWB file for kachery
    """

    def make(self, key):
        # set up to use the selected kachery channel 
        current_channel = KacheryChannel.get_channel()
        if key['channel_name'] != current_channel['channel_name']:
            KacheryChannel.set_channel(key)

        print(f'Linking {key["nwb_file_name"]} and storing in kachery...')
        nwb_abs_path = Nwbfile().get_abs_path(key['nwb_file_name'])
        key['nwb_file_uri'] = kc.link_file(nwb_abs_path)
        # we also need to insert the original NWB file. 
        # To do so we remove the last character ('_') and add the extension
        orig_nwb_abs_path = os.path.splitext(nwb_abs_path)[0][:-1] + '.nwb'
        key['nwb_raw_file_uri'] = kc.link_file(orig_nwb_abs_path)
        self.insert1(key)
        # reset the environment variables if they were set
        KacheryChannel.set_channel(current_channel)
        
@schema
class AnalysisNwbfileKacherySelection(dj.Manual):
    definition="""
    -> KacheryChannel
    -> AnalysisNwbfile
    """

@schema
class AnalysisNwbfileKachery(dj.Computed):
    definition = """
    -> AnalysisNwbfileKacherySelection
    ---
    analysis_file_uri: varchar(200)  # the uri of the file
    """

    class LinkedFiles(dj.Part):
        definition = """
        -> AnalysisNwbfileKachery
        linked_file_name: varchar(200) # the name of the linked data file
        ---
        linked_file_uri: varchar(200) # the uri for the linked file
        """
    def make(self, key):
        # set up to use the selected kachery channel 
        current_channel = KacheryChannel.get_channel()
        if key['channel_name'] != current_channel['channel_name']:
            KacheryChannel.set_channel(key)
        print(f'Linking {key["analysis_file_name"]} and storing in kachery...')
        key['analysis_file_uri'] = kc.link_file(AnalysisNwbfile().get_abs_path(key['analysis_file_name']))
        self.insert1(key)
        # TODO Detect and insert linked files

        # reset the environment variables 
        KacheryChannel.set_channel(current_channel)
        

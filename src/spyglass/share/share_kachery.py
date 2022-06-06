import os
import stat
import pathlib
import random
import string

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
    host:   varchar(200)    # the host name for the computer running the daemon for this channel
    port:   smallint unsigned  # the port number for the daemon on the host
    description: varchar(200) # description of this channel
    """

@schema
class NwbfileKachery(dj.Computed):
    definition = """
    -> KacheryChannel
    -> Nwbfile
    ---
    nwb_file_uri: varchar(200)  # the uri the NWB file for kachery
    """

    def make(self, key):
        # set up to use the selected kachery channel
        try:
            default_kachery_host = os.environ["KACHERY_DAEMON_HOST"]
            default_kachery_port = os.environ["KACHERY_DAEMON_PORT"]
        except:
            default_kachery_host = None
            default_kachery_port = None
        
        kachery_host, kachery_port = (KacheryChannel & {'channel_name' : key['channel_name']}).fetch1('host', 'port')
        os.environ["KACHERY_DAEMON_HOST"] = kachery_host
        os.environ["KACHERY_DAEMON_PORT"] = kachery_port
        print(f'Linking {key["nwb_file_name"]} and storing in kachery...')
        key['nwb_file_uri'] = kc.link_file(Nwbfile().get_abs_path(key['nwb_file_name']))
        self.insert1(key)
        # reset the environment variables if they were set
        if default_kachery_host is not None and default_kachery_port is not None:
            os.environ["KACHERY_DAEMON_HOST"] = default_kachery_host
            os.environ["KACHERY_DAEMON_PORT"] = default_kachery_port
        


@schema
class AnalysisNwbfileKachery(dj.Computed):
    definition = """
    -> AnalysisNwbfile
    ---
    analysis_file_uri: varchar(200)  # the uri of the file
    """
    def make(self, key):
        # set up to use the selected kachery channel
        try:
            default_kachery_host = os.environ["KACHERY_DAEMON_HOST"]
            default_kachery_port = os.environ["KACHERY_DAEMON_PORT"]
        except:
            default_kachery_host = None
            default_kachery_port = None
        
        kachery_host, kachery_port = (KacheryChannel & {'channel_name' : key['channel_name']}).fetch1('host', 'port')
        os.environ["KACHERY_DAEMON_HOST"] = kachery_host
        os.environ["KACHERY_DAEMON_PORT"] = kachery_port
        print(f'Linking {key["analysis_file_name"]} and storing in kachery...')
        key['analysis_file_uri'] = kc.link_file(AnalysisNwbfile().get_abs_path(key['analysis_file_name']))
        self.insert1(key)
        # reset the environment variables if they were set
        if default_kachery_host is not None and default_kachery_port is not None:
            os.environ["KACHERY_DAEMON_HOST"] = default_kachery_host
            os.environ["KACHERY_DAEMON_PORT"] = default_kachery_port
        
 

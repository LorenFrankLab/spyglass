from copy import Error
import json
import os
import pathlib
import re
import tempfile
import time
from pathlib import Path
import shutil

import datajoint as dj
import kachery_client as kc
import numpy as np
import pandas as pd
import pynwb
import scipy.stats as stats
import sortingview as sv
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.sorters as ss
import spikeinterface.toolkit as st
# from mountainsort4.mdaio_impl import readmda

from .common_device import Probe
from .common_lab import LabMember, LabTeam
from .common_ephys import Electrode, ElectrodeGroup, Raw
from .common_nwbfile import AnalysisNwbfile, Nwbfile
from .common_session import Session
from .dj_helper_fn import dj_replace, fetch_nwb
from .nwb_helper_fn import get_valid_intervals
from .sortingview_utils import add_to_sortingview_workspace, set_workspace_permission

si.set_global_tmp_folder(os.environ['KACHERY_TEMP_DIR'])

schema = dj.schema('common_waveforms')

@schema
class WaveformParameters(dj.Manual):
    definition = """
    waveform_parameters_name: varchar(80) # name of waveform extraction parameters
    ---
    waveform_parameter_dict: blob # a dictionary containing waveform extraction parameters
    """

@schema
class Waveforms(dj.Manual):
    definition = """
    -> SpikeSorting
    ---
    -> WaveformParameters
    """

@schema
class WaveformsComputed(dj.Computed):
    definition = """
    -> Waveforms
    ---
    -> AnalysisNwbfile
    """
    def make(self, key):
        self.insert1(key)

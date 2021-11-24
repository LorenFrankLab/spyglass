from copy import Error
import os
from pathlib import Path

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

from .common_spikesorting import SpikeSorting

si.set_global_tmp_folder(os.environ['KACHERY_TEMP_DIR'])

schema = dj.schema('common_waveforms')

@schema
class WaveformParameters(dj.Manual):
    definition = """
    list_name: varchar(80) # name of waveform extraction parameters
    ---
    params: blob # a dictionary containing waveform extraction parameters
    """
    def insert_default_params(self):
        key = {}
        self.insert1(key)
        

@schema
class WaveformSelection(dj.Manual):
    definition = """
    -> SpikeSorting
    -> WaveformParameters
    ---
    """

@schema
class Waveforms(dj.Computed):
    definition = """
    -> WaveformSelection
    ---
    -> AnalysisNwbfile
    """
    def make(self, key):
        self.insert1(key)
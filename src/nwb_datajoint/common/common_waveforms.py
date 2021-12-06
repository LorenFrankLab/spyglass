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
    params: blob # a dict of waveform extraction parameters
    """
    def insert_default_params(self):
        key = {}
        key['list_name'] = 'default'
        key['params'] = {'ms_before':1, 'ms_after':1, 'max_spikes_per_unit':1000}
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
        recording = key['recording']
        sorting = key['sorting']
        save_path = key['save_path']
        params = key['params']
        si.extract_waveforms(recording=recording, sorting=sorting, folder=save_path, **params)
        # add to nwb file
        # nwbfile = pynwb.NWBFile(...)
        # wfs = [
        #         [     # elec 1
        #             [1, 2, 3],  # spike 1, [sample 1, sample 2, sample 3]
        #             [1, 2, 3],  # spike 2
        #             [1, 2, 3],  # spike 3
        #             [1, 2, 3]   # spike 4
        #         ], [  # elec 2
        #             [1, 2, 3],  # spike 1
        #             [1, 2, 3],  # spike 2
        #             [1, 2, 3],  # spike 3
        #             [1, 2, 3]   # spike 4
        #         ], [  # elec 3
        #             [1, 2, 3],  # spike 1
        #             [1, 2, 3],  # spike 2
        #             [1, 2, 3],  # spike 3
        #             [1, 2, 3]   # spike 4
        #         ]
        # ]
        # elecs = ... # DynamicTableRegion referring to three electrodes (rows) of the electrodes table
        # nwbfile.add_unit(spike_times=[1, 2, 3], electrodes=elecs, waveforms=wfs)
        
        # key['analysis_nwb_file'] = analysis_nwb_file_path
        self.insert1(key)
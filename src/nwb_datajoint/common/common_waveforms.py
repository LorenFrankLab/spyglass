import os
from pathlib import Path
import shutil

import datajoint as dj
import sortingview as sv
import spikeinterface as si

from .common_nwbfile import AnalysisNwbfile
from .common_spikesorting import SpikeSortingRecording, SpikeSorting, Sortings

schema = dj.schema('common_waveforms')

@schema
class WaveformParameters(dj.Manual):
    definition = """
    waveform_params_name: varchar(80) # name of waveform extraction parameters
    ---
    waveform_params: blob # a dict of waveform extraction parameters
    """
    def insert_default(self):
        waveform_params_name = 'default'
        waveform_params = {'ms_before': 0.5, 'ms_after': 0.5, 'max_spikes_per_unit': 5000,
                           'n_jobs': 5, 'total_memory': '5G'}
        self.insert1([waveform_params_name, waveform_params], skip_duplicates=True) 

@schema
class WaveformSelection(dj.Manual):
    definition = """
    -> Sortings
    -> WaveformParameters
    ---
    """

@schema
class Waveforms(dj.Computed):
    definition = """
    -> WaveformSelection
    ---
    waveform_extractor_path: varchar(220)
    -> AnalysisNwbfile
    object_id: varchar(40)   # Object ID for the waveforms in NWB file
    """
    def make(self, key):
        recording_path = (SpikeSortingRecording & key).fetch1('recording_path')
        recording = si.load_extractor(recording_path)
        
        sorting_path = (Sortings & key).fetch1('sorting_path')
        sorting = si.load_extractor(sorting_path)
        
        print('Extracting waveforms...')
        waveform_params = (WaveformParameters & key).fetch1('waveform_params')
        waveform_extractor_name = self._get_waveform_extractor_name(key)
        key['waveform_extractor_path'] = str(Path(os.environ['NWB_DATAJOINT_WAVEFORMS_DIR']) / Path(waveform_extractor_name))
        if os.path.exists(key['waveform_extractor_path']):
            shutil.rmtree(key['waveform_extractor_path'])
        waveforms = si.extract_waveforms(recording=recording, 
                                         sorting=sorting, 
                                         folder=key['waveform_extractor_path'],
                                         **waveform_params)
        
        key['analysis_file_name'] = AnalysisNwbfile().create(key['nwb_file_name'])
        object_id = AnalysisNwbfile().add_units_waveforms(key['analysis_file_name'],
                                                          waveform_extractor=waveforms)
        key['object_id'] = object_id       
        AnalysisNwbfile().add(key['nwb_file_name'], key['analysis_file_name'])       
             
        self.insert1(key)
    
    def load_waveforms(self, key: dict):
        """Returns a spikeinterface waveform extractor specified by key

        Parameters
        ----------
        key : dict
            Could be an entry in Waveforms, or some other key that uniquely defines
            an entry in Waveforms

        Returns
        -------
        we : spikeinterface.WaveformExtractor
        """
        we_path = (self & key).fetch1('waveform_extractor_path')
        we = si.WaveformExtractor.load_from_folder(we_path)
        return we
    
    def fetch_nwb(self, key):
        # TODO: implement fetching waveforms from NWB
        return NotImplementedError
    
    def _get_waveform_extractor_name(self, key):
        we_name = key['recording_id'] + '_' + key['sorting_id'] + '_waveform'
        return we_name
    
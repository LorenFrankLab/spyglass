import os
from pathlib import Path

import datajoint as dj
import sortingview as sv
import spikeinterface as si

from .common_nwbfile import AnalysisNwbfile
from .common_spikesorting import SpikeSortingRecording, SpikeSorting, SortingID

schema = dj.schema('common_waveforms')

@schema
class WaveformParameters(dj.Manual):
    definition = """
    waveform_params_name: varchar(80) # name of waveform extraction parameters
    ---
    waveform_params: blob # a dict of waveform extraction parameters
    """
    def insert_default(self):
        key = {}
        key['waveform_params_name'] = 'default'
        key['waveform_params'] = {'ms_before':1, 'ms_after':1, 'max_spikes_per_unit': 2000,
                                  'n_jobs':5, 'total_memory': '5G'}
        self.insert1(key, skip_duplicates=True) 

@schema
class WaveformSelection(dj.Manual):
    definition = """
    -> SortingID
    -> WaveformParameters
    ---
    """

@schema
class Waveforms(dj.Computed):
    definition = """
    -> WaveformSelection
    ---
    -> AnalysisNwbfile
    object_id: varchar(40) # Object ID for the waveforms in NWB file
    waveform_extractor_path: varchar(220)
    """
    def make(self, key):
        waveform_extractor_name = self._get_waveform_extractor_name(key)
        key['waveform_extractor_path'] = self._get_waveform_save_path(waveform_extractor_name)
        
        params = (WaveformParameters & {'waveform_params_name': key['waveform_params_name']}).fetch1('waveform_params')

        recording_object = (SpikeSortingRecording & key).fetch1('recording_extractor_object')
        recording = sv.LabboxEphysRecordingExtractor(recording_object)
        new_recording = si.create_recording_from_old_extractor(recording)
        new_recording.annotate(is_filtered=True)
        
        sorting_object = (SortingID & key).fetch1('sorting_extractor_object')
        sorting = sv.LabboxEphysSortingExtractor(sorting_object)
        new_sorting = si.create_sorting_from_old_extractor(sorting)
        
        print('Extracting waveforms...')
        waveforms = si.extract_waveforms(recording=new_recording, 
                                         sorting=new_sorting, 
                                         folder=key['waveform_extractor_path'],
                                         **params)
        
        key['analysis_file_name'] = AnalysisNwbfile().create(key['nwb_file_name'])
        
        object_id = AnalysisNwbfile().add_units_waveforms(key['analysis_file_name'],
                                                          waveforms)
        key['object_id'] = object_id       
        AnalysisNwbfile().add(key['nwb_file_name'], key['analysis_file_name'])       
             
        self.insert1(key)
    
    def load_waveforms(self, key):
        # TODO: check if multiple entries are passed
        key = (Waveforms & key).fetch1()
        folder = key['waveform_extractor_path']
        we = si.WaveformExtractor.load_from_folder(folder)
        return we
    
    def fetch_nwb(self, key):
        # TODO: implement fetching waveforms from NWB
        return NotImplementedError
    
    def _get_waveform_extractor_name(self, key):
        key = (SpikeSorting & key).fetch1()
        sorting_name = SpikeSorting()._get_sorting_name(key)
        we_name = sorting_name + '_waveform'
        return we_name

    def _get_waveform_save_path(self, waveform_extractor_name):
        waveforms_dir = Path(os.environ['NWB_DATAJOINT_BASE_DIR']) / 'waveforms' / waveform_extractor_name
        if waveforms_dir.exists() is False:
            os.mkdir(waveforms_dir)
        return str(waveforms_dir)
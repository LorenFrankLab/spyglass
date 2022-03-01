from typing import List, Union
import datajoint as dj
import sortingview as sv
from sortingview.SpikeSortingView import create_raw_traces_plot
from sortingview.SpikeSortingView.Figure import Figure
import numpy as np
import kachery_client as kc
import os
import spikeinterface as si

from ..common.common_spikesorting import SpikeSortingRecording

schema = dj.schema('figurl_view_spike_sorting_recording')

@schema
class SpikeSortingRecordingView(dj.Computed):
    definition = """
    # Schema for storing figurl views of spike sorting recordings
    -> SpikeSortingRecording
    ---
    figurl: varchar(10000)
    """
    def make(self, key):
        # Get the SpikeSortingRecording row
        a = (SpikeSortingRecording & key).fetch1()
        sort_group_id = a['sort_group_id']
        sort_interval_name = a['sort_interval_name']
        recording_id = a['recording_id']
        recording_path = a['recording_path']

        # Load the SI recording extractor
        recording: si.BaseRecording = si.load_extractor(recording_path)
        
        # Raw traces (sample)
        # Extract the first 1 second of traces
        traces: np.array = recording.get_traces(
            start_frame=0,
            end_frame=int(recording.get_sampling_frequency() * 1)
        ).astype(np.float32)
        f1 = create_raw_traces_plot(
            traces=traces,
            start_time_sec=0,
            sampling_frequency=recording.get_sampling_frequency(),
            label='Raw traces (sample)'
        )

        # Electrode geometry
        f2 = create_electrode_geometry(recording)

        # Mountain layout
        F = create_mountain_layout(
            figures=[f1, f2],
            label=f'{recording_id}:{sort_group_id}:{sort_interval_name}'
        )

        # Insert row into table
        key2 = key
        key2['figurl'] = F.url()
        self.insert1(key2)

def create_electrode_geometry(recording: si.BaseRecording):
    channel_locations = {}
    channel_locations0 = recording.get_channel_locations()
    for ii, channel_id in enumerate(recording.get_channel_ids()):
        channel_locations[str(channel_id)] = channel_locations0[ii, :].astype(np.float32)
    data = {
        'type': 'ElectrodeGeometry',
        'channelLocations': channel_locations
    }
    return Figure(data=data, label='Electrode geometry')

def create_mountain_layout(figures: List[Figure], label: Union[str, None]=None, sorting_curation_uri: Union[str, None]=None) -> Figure:
    if label is None:
        label = 'SpikeSortingView'
    
    data = {
        'type': 'MountainLayout',
        'views': [
            {
                'type': fig0.data['type'],
                'label': fig0.label,
                'figureDataSha1': _upload_data_and_return_sha1(fig0.get_serialized_figure_data())
            }
            for fig0 in figures
        ]
    }
    if sorting_curation_uri is not None:
        data['sortingCurationUri'] = sorting_curation_uri
    return Figure(data=data, label=label)

def _upload_data_and_return_sha1(data):
    data_uri = kc.store_json(data)
    data_hash = data_uri.split('/')[2]
    kc.upload_file(data_uri, channel=os.environ['FIGURL_CHANNEL'], single_chunk=True)
    return data_hash
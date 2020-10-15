import spikeextractors as se
import labbox_ephys as le
import numpy as np
import h5py

# Prepare recording, sorting extractors using any method
recording, sorting = se.example_datasets.toy_example()

# Specify the output path
output_h5_path = 'test_snippets.h5'

# Prepare the snippets h5 file
le.prepare_snippets_h5_from_extractors(
    recording=recording,
    sorting=sorting,
    output_h5_path=output_h5_path,
    start_frame=None,
    end_frame=None,
    max_events_per_unit=1000,
    max_neighborhood_size=2
)

# Example display some contents of the file
with h5py.File(output_h5_path, 'r') as f:
    unit_ids = np.array(f.get('unit_ids'))
    sampling_frequency = np.array(f.get('sampling_frequency'))[0]
    print(f'Unit IDs: {unit_ids}')
    print(f'Sampling freq: {sampling_frequency}')
    for unit_id in unit_ids:
        unit_spike_train = np.array(f.get(f'unit_spike_trains/{unit_id}'))
        unit_waveforms = np.array(f.get(f'unit_waveforms/{unit_id}/waveforms'))
        unit_waveforms_channel_ids = np.array(f.get(f'unit_waveforms/{unit_id}/channel_ids'))
        print(f'Unit {unit_id} | Tot num events: {len(unit_spike_train)} | shape of subsampled snippets: {unit_waveforms.shape}')
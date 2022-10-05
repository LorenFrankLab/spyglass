import pynwb
import numpy as np
from scipy import interpolate

with pynwb.NWBHDF5IO('/hdd/dj/raw/sub-MS10_ses-Peter-MS10-170321-164406-concat_behavior+ecephys.nwb', mode='r', load_namespaces=True) as io:
    nwbf = io.read()
    position = nwbf.processing['behavior'].data_interfaces['SubjectPosition'].spatial_series['SpatialSeries'].data[:,:2]
    position_timestamps = nwbf.processing['behavior'].data_interfaces['SubjectPosition'].spatial_series['SpatialSeries'].timestamps[:]
    units = nwbf.units.to_dataframe()                                                                                                            
spike_times_list = units['spike_times'].to_numpy()
sampling_rate = 20000
start_time = position_timestamps[0]
# end_time = 127639440/20000
end_time = position_timestamps[-1]
bin_size = 0.002
bin_edges = np.arange(start_time, end_time, bin_size)
n_bins = int((end_time-start_time)/bin_size)
n_units = len(spike_times_list)

spike_indicator = np.zeros((n_bins, n_units))
for unit in range(n_units):
    spike_times = spike_times_list[unit]
    spike_times = spike_times[(spike_times > start_time) & (spike_times < end_time)]
    for spike in spike_times:
        spike_indicator[np.searchsorted(bin_edges, spike), unit] = 1

        
f = interpolate.interp1d(position_timestamps, position, axis=0)
position_interp = f(bin_edges[1:])
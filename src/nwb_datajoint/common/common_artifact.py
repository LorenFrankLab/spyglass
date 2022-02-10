from functools import reduce

import datajoint as dj
import numpy as np
import scipy.stats as stats
import spikeinterface as si

from .common_interval import IntervalList
from .common_spikesorting import SpikeSortingRecording
from .nwb_helper_fn import get_valid_intervals

schema = dj.schema('common_artifact')

@schema
class ArtifactDetectionParameters(dj.Manual):
    definition = """
    # Parameters for detecting artifact times within a sort group.
    artifact_params_name: varchar(200)
    ---
    artifact_params: blob  # dictionary of parameters
    """
    def insert_default(self):
        """Insert the default artifact parameters with an appropriate parameter dict.
        """
        artifact_params = {}
        artifact_params['zscore_thresh'] = None # must be None or >= 0
        artifact_params['amplitude_thresh'] = 3000 # must be None or >= 0
        artifact_params['proportion_above_thresh'] = 1.0 # all electrodes of sort group
        artifact_params['removal_window_len'] = 1.0 # in milliseconds
        self.insert1(['default', artifact_params], skip_duplicates=True)  

@schema
class ArtifactDetectionSelection(dj.Manual):
    definition = """
    # Specifies artifact detection parameters to apply to a sort group's recording.
    -> SpikeSortingRecording
    -> ArtifactDetectionParameters
    ---
    """

@schema
class ArtifactDetection(dj.Computed):
    definition = """
    # Stores artifact times and valid no-artifact times as intervals.
    -> ArtifactDetectionSelection
    ---
    artifact_times: longblob # np array of artifact times for each detected artifact
    artifact_removed_valid_times: longblob # np array of valid no-artifact start and end times (an array of intervals)
    artifact_removed_interval_list_name: varchar(200) # name of the array of detected artifact-free times
    """

    def make(self, key):
         # get the dict of artifact params associated with this artifact_params_name
        artifact_params = (ArtifactDetectionParameters & key).fetch1("artifact_params")
        
        recording_path = (SpikeSortingRecording & key).fetch1('recording_path')
        recording = si.load_extractor(recording_path) 
        
        # get artifact-related timing info
        # could call an input verification method right before this line
        artifact_removed_valid_times, artifact_times = _get_artifact_times(recording, **artifact_params)
        key['artifact_times'] = artifact_times
        key['artifact_removed_valid_times'] = artifact_removed_valid_times
        
        # set up a name for no-artifact times using recording id
        key['artifact_removed_interval_list_name'] = key['recording_id'] + '_' + key['artifact_params_name'] + '_artifact_removed_valid_times'
        
        # insert artifact times and valid times into ArtifactRemovedIntervalList with an appropriate name
        tmp_key = {}
        tmp_key['artifact_removed_interval_list_name'] = key['artifact_removed_interval_list_name']
        tmp_key['artifact_params_name'] = key['artifact_params_name']
        tmp_key['artifact_removed_valid_times'] = key['artifact_removed_valid_times']
        tmp_key['recording_id'] = key['recording_id']
        ArtifactRemovedIntervalList.insert1(tmp_key) #better to overwrite or to skip repeats? not sure?
        
        # also insert into IntervalList
        tmp_key = {}
        tmp_key['nwb_file_name'] = key['nwb_file_name']
        tmp_key['interval_list_name'] = key['artifact_removed_interval_list_name']
        tmp_key['valid_times'] = key['artifact_removed_valid_times']
        IntervalList.insert1(tmp_key, skip_duplicates=True)
        
        # insert into computed table
        self.insert1(key)

@schema
class ArtifactRemovedIntervalList(dj.Manual):
    definition = """
    # Stores intervals without detected artifacts.
    artifact_removed_interval_list_name: varchar(200)
    ---
    -> ArtifactDetectionParameters
    artifact_removed_valid_times: longblob # np array of valid no-artifact start and end times
    recording_id: varchar(12)
    """
    
def _get_artifact_times(recording, zscore_thresh=None, amplitude_thresh=None,
                        proportion_above_thresh=1.0, removal_window_len=1.0):
    """Detects times during which artifacts do and do not occur.
    Artifacts are defined as periods where the absolute value of the recording signal exceeds one
    OR both specified amplitude or zscore thresholds on the proportion of channels specified,
    with the period extended by the removal_window_len/2 ms on each side. Z-score and amplitude
    threshold values of None are ignored.

    Parameters
    ----------
    recording: si.Recording
    zscore_thresh: float, optional
        Stdev threshold for exclusion, should be >=0, defaults to None
    amplitude_thresh: float, optional
        Amplitude threshold for exclusion, should be >=0, defaults to None
    proportion_above_thresh: float, optional
        Proportion of electrodes that need to have threshold crossings, defaults to 1 
    removal_window_len: float, optional
        Width of the window in milliseconds to mask out per artifact (window/2 removed on each side of threshold crossing), defaults to 1 ms
    
    Return
    ------
    artifact_times: np.ndarray
        The timestamps contained in each of the detected artifact windows
    artifact_removed_valid_times: np.ndarray
        Intervals of valid times where artifacts were not detected
        
    Raises
    ------
    ValueError
        when amplitude or zscore thresholds are negative
    """
    
    # if both thresholds are None, we essentially skip artifract detection and
    # return an array with the times of the first and last samples of the recording
    if (amplitude_thresh is None) and (zscore_thresh is None):
        recording_interval = np.asarray([recording.get_times()[0], recording.get_times()[-1]])
        artifact_times_empty = np.asarray([])
        return recording_interval, artifact_times_empty
    
    # otherwise if amplitude or zscore thresholds are negative, raise a value error
    # doesn't make sense to use a negative threshold on an absolute signal - might be nice to tell the user?
    thresholds = [t for t in [amplitude_thresh, zscore_thresh] if t is not None]
    for t in thresholds:
        if t < 0:
            raise ValueError("Amplitude and Z-Score thresholds must be >= 0, or None")
    # also verify that proportion above makes sense/is in [0:1] inclusive?
    # if ((proportion_above_thresh < 0) or (proportion_above_thresh > 1)):
    #     raise ValueError("proportion_above_thresh must be >= 0 and <= 1")
    
    # related to ^^, if we want to verify inputs values, then i might vote for breaking that out into its own method on this class
    # and calling the input vertification method before calling this current method in the downstream computed table's make fxn

    # turn milliseconds to remove total into # samples to remove from either side of each detected artifact
    # sampling freq is in samp/sec. 1000 ms per sec. 1/2 to be removed from either side of artifact sample. 
    half_window_points = np.round(
        recording.get_sampling_frequency() * (1/1000) * removal_window_len * (1/2) )
    
    # get the data traces
    data = recording.get_traces()

    # compute the number of electrodes that have to be above threshold
    nelect_above = np.round(proportion_above_thresh * len(recording.get_channel_ids()))

    # find the artifact occurrences using one or both thresholds, across channels
    if ((amplitude_thresh is not None) and (zscore_thresh is None)):
        above_a = np.abs(data) > amplitude_thresh
        above_thresh = np.ravel(np.argwhere(np.sum(above_a, axis=0) >= nelect_above))
    elif ((amplitude_thresh is None) and (zscore_thresh is not None)):
        dataz = np.abs(stats.zscore(data, axis=1))
        above_z = dataz > zscore_thresh
        above_thresh = np.ravel(np.argwhere(np.sum(above_z, axis=0) >= nelect_above))
    else:
        above_a = np.abs(data) > amplitude_thresh
        dataz = np.abs(stats.zscore(data, axis=1))
        above_z = dataz > zscore_thresh
        above_thresh = np.ravel(np.argwhere(
            np.sum(np.logical_or(above_z, above_a), axis=0) >= nelect_above))
    
    valid_timestamps = recording.get_times()
    
    # keep track of artifact detected timestamps - this could be saved if we wanted?
    # artifact_detected_times = valid_timestamps[above_thresh] # the specific artifact times, NOT including the window around them

    # keep track of all the artifact times within each artifact removal window
    artifact_times = []
    for a in above_thresh:
        if (a - half_window_points) < 0: # make sure indices are valid for the slice
            a_times = np.copy(valid_timestamps[0:int(a + half_window_points)])
            artifact_times.append(a_times)
        else:
            a_times = np.copy(valid_timestamps[int(a - half_window_points):int(a + half_window_points)])
            artifact_times.append(a_times)
    artifact_times = np.asarray(artifact_times)
    # alternatively, we could store these as intervals instead of arrays of all the times? (using interval_list_union many times or something?)
    
    # turn all artifact detected times into -1 so valid non-artifact intervals can be easily found
    for a in above_thresh:
        if (a - half_window_points) < 0: # make sure indices are valid for the slice
            valid_timestamps[0:int(a + half_window_points)] = -1
        else:
            valid_timestamps[int(a - half_window_points):int(a + half_window_points)] = -1
    artifact_removed_valid_times = get_valid_intervals(valid_timestamps[valid_timestamps != -1], recording.get_sampling_frequency(), 1.5, 0.001)        
    
    return artifact_removed_valid_times, artifact_times

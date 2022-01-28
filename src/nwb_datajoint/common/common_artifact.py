import os
import pathlib
import time
from pathlib import Path
import shutil
import uuid
from functools import reduce

import datajoint as dj
import numpy as np
import pynwb
import scipy.stats as stats
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.sorters as ss
import spikeinterface.toolkit as st

from .common_device import Probe
from .common_lab import LabMember, LabTeam
from .common_ephys import Electrode, ElectrodeGroup, Raw
from .common_interval import (IntervalList, SortInterval,
                              interval_list_intersect, union_adjacent_index)
from .common_nwbfile import AnalysisNwbfile, Nwbfile
from .common_session import Session
from .dj_helper_fn import dj_replace, fetch_nwb
from .nwb_helper_fn import get_valid_intervals

schema = dj.schema('common_artifact')

# TODO: need to change in common_spikesorting:
#(1) always remove artifacts rather than interpolating or zeroing
#then concatenate across the gaps, just as you would between any other
#discontinuous intervals - this also does not require any parameters
#(2) note that artifact removal also occurs in finding valid times
#before concatenation and whitening in SpikeSorting def make
#(3) remove placeholder SpikeSortingArtifactDetectionParameters schema
#(4) add ArtifactRemovedIntervalList somehow as primary key in 
#SpikeSortingSelection, add as a primary dependency or the int name at least

@schema
class SpikeSortingArtifactParameters(dj.Manual):
    definition = """
    # Parameters for detecting artifact times within a sort group
    artifact_params_name: varchar(200)
    ---
    artifact_params: blob  # dictionary of parameters for get_no_artifact_times() function
    """
    def insert_default(self):
        """Insert the default artifact parameters with an appropriate parameter dict.
        """
        artifact_params = {}
        artifact_params['skip'] = True
        artifact_params['zscore_thresh'] = -1.0 
        artifact_params['amplitude_thresh'] = -1.0
        artifact_params['proportion_above_thresh'] = -1.0
        artifact_params['zero_window_len'] = 30 # 1 ms at 30 KHz, but this is of course skipped
        self.insert1(['none', artifact_params], skip_duplicates=True)

    # TODO: check whether window is in samples or time, clarify naming to make units obvious
    #understand spike interface recording
    #make this return both artifact and artifact free intervals rather than just one (==-1 and !=-1 times)
    #check - returns list or np array?
    #rename zero window len to specify that these times are being removed (not zero'd)

    def get_artifact_times(self, recording, zscore_thresh=-1.0, amplitude_thresh=-1.0,
                           proportion_above_thresh=1.0, zero_window_len=1.0, skip=True):
        """Detects times during which artifacts occur.
        Artifacts are defined as periods where the absolute amplitude of the signal exceeds one
        OR both specified amplitude or zscore thresholds on the proportion of channels specified,
        with the period extended by the zero_window/2 samples on each side. Z-score and amplitude
        threshold values <0 are ignored.

        Parameters
        ----------
        recording: si.Recording
        zscore_thresh: float, optional
            Stdev threshold for exclusion, defaults to -1.0
        amplitude_thresh: float, optional
            Amplitude threshold for exclusion, defaults to -1.0
        proportion_above_thresh: float, optional
        zero_window_len: int, optional
            the width of the window in milliseconds to zero out (window/2 on each side of threshold crossing)
        
        Return
        ------
        artifact_interval: np.ndarray
            timestamps for detected artifacts
        valid_artifact_removed_interval: np.ndarray
            timestamps for valid times where artifacts were not detected
        """

        # if no thresholds were specified, we return an array with the timestamps of the first and last samples
        if zscore_thresh <= 0 and amplitude_thresh <= 0:
            return np.asarray([[recording._timestamps[0], recording._timestamps[recording.get_num_frames()-1]]])

        half_window_points = np.round(
            recording.get_sampling_frequency() * 1000 * zero_window_len / 2)
        nelect_above = np.round(proportion_above_thresh * data.shape[0])
        # get the data traces
        data = recording.get_traces()

        # compute the number of electrodes that have to be above threshold based on the number of rows of data
        nelect_above = np.round(
            proportion_above_thresh * len(recording.get_channel_ids()))

        # apply the amplitude threshold
        above_a = np.abs(data) > amplitude_thresh

        # zscore the data and get the absolute value for thresholding
        dataz = np.abs(stats.zscore(data, axis=1))
        above_z = dataz > zscore_thresh

        above_both = np.ravel(np.argwhere(
            np.sum(np.logical_or(above_z, above_a), axis=0) >= nelect_above))
        valid_timestamps = recording._timestamps
        # for each above threshold point, set the timestamps on either side of it to -1
        for a in above_both:
            valid_timestamps[a - half_window_points:a +
                             half_window_points] = -1

        # use get_valid_intervals to find all of the resulting valid times.
        return get_valid_intervals(valid_timestamps[valid_timestamps != -1], recording.get_sampling_frequency(), 1.5, 0.001)

@schema
class SpikeSortingArtifactParametersSelection(dj.Manual):
    definition = """
    # Specifies artifact detection parameters to apply to a sort group's recording
    -> SpikeSortingArtifactParameters
    -> SpikeSortingRecording
    ---
    """

@schema
class SpikeSortingArtifactInterval(dj.Computed):
    definition = """
    # Stores artifact times and valid no-artifact times as intervals
    -> SpikeSortingArtifactParametersSelection
    ---
    artifact_detected_times: longblob # np array of artifact start and end times
    artifact_removed_valid_times: longblob # np array of valid no-artifact start and end times
    artifact_removed_interval_list_name: varchar(200) # name array of detected artifact-free times
    """

    def make(self, key):
        #fetches the parameters and the recording
        #uses get_artifact_times fxn to return the artifact and no-artifact time intervals
        #insert intervals into table - insert art times and no art times?
        #auto name based on primary keys? so it will be unique and informative
        #autoinserts subset of info (nwb file name, artifact list name, valid times)
        #into both artifact removed manual interval list (for spikesorting),
        #as well as insert into general IntervalList() for any other analyses where these times are relevant?

@schema
class SpikeSortingArtifactRemovedIntervalList(dj.Manual):
    definition = """
    # Stores intervals without detected artifacts
    -> Session
    artifact_removed_interval_list_name: varchar(200)
    ---
    artifact_removed_valid_times: longblob # np array of valid no-artifact start and end times
    """
    
    #notes to self:
    #gets used in this way: intersection of valid times, artifactremovedtimes, and SortInterval
    #data in that intersection are concatenated together, whitened, then sorted

    #this table is NOT restricted to single-sort-group-based artifact-free times
    #this table can also include artifact-free times determined across sort groups
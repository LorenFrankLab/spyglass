from functools import reduce
from typing import Union

import datajoint as dj
import numpy as np
import spikeinterface as si
from spikeinterface.core.job_tools import ChunkRecordingExecutor, ensure_n_jobs

from spyglass.common.common_interval import (
    IntervalList,
    _union_concat,
    interval_from_inds,
    interval_set_difference_inds,
)
from spyglass.spikesorting.utils import (
    _check_artifact_thresholds,
    _compute_artifact_chunk,
    _init_artifact_worker,
)
from spyglass.spikesorting.v0.spikesorting_recording import (
    SpikeSortingRecording,
)
from spyglass.utils import SpyglassMixin, logger
from spyglass.utils.nwb_helper_fn import get_valid_intervals

schema = dj.schema("spikesorting_artifact")


@schema
class ArtifactDetectionParameters(SpyglassMixin, dj.Manual):
    definition = """
    # Parameters for detecting artifact times within a sort group.
    artifact_params_name: varchar(200)
    ---
    artifact_params: blob  # dictionary of parameters
    """

    def insert_default(self):
        """Insert the default artifact parameters with an appropriate parameter dict."""
        artifact_params = {}
        artifact_params["zscore_thresh"] = None  # must be None or >= 0
        artifact_params["amplitude_thresh"] = 3000  # must be None or >= 0
        # all electrodes of sort group
        artifact_params["proportion_above_thresh"] = 1.0
        artifact_params["removal_window_ms"] = 1.0  # in milliseconds
        self.insert1(["default", artifact_params], skip_duplicates=True)

        artifact_params_none = {}
        artifact_params_none["zscore_thresh"] = None
        artifact_params_none["amplitude_thresh"] = None
        self.insert1(["none", artifact_params_none], skip_duplicates=True)


@schema
class ArtifactDetectionSelection(SpyglassMixin, dj.Manual):
    definition = """
    # Specifies artifact detection parameters to apply to a sort group's recording.
    -> SpikeSortingRecording
    -> ArtifactDetectionParameters
    ---
    custom_artifact_detection=0 : tinyint
    """


@schema
class ArtifactDetection(SpyglassMixin, dj.Computed):
    definition = """
    # Stores artifact times and valid no-artifact times as intervals.
    -> ArtifactDetectionSelection
    ---
    artifact_times: longblob # np array of artifact intervals
    artifact_removed_valid_times: longblob # np array of valid no-artifact intervals
    artifact_removed_interval_list_name: varchar(200) # name of the array of no-artifact valid time intervals
    """

    _parallel_make = True

    def make(self, key):
        """Populate the ArtifactDetection table.

        If custom_artifact_detection is set in selection table, do nothing.

        Fetches...
            - Parameters from ArtifactDetectionParameters
            - Recording from SpikeSortingRecording (loads with spikeinterface)
        Uses module-level function _get_artifact_times to detect artifacts.
        """
        if not (ArtifactDetectionSelection & key).fetch1(
            "custom_artifact_detection"
        ):
            # get the dict of artifact params associated with this artifact_params_name
            artifact_params = (ArtifactDetectionParameters & key).fetch1(
                "artifact_params"
            )

            recording_path = (SpikeSortingRecording & key).fetch1(
                "recording_path"
            )
            recording_name = SpikeSortingRecording._get_recording_name(key)
            recording = si.load_extractor(recording_path)

            job_kwargs = {
                "chunk_duration": "10s",
                "n_jobs": 4,
                "progress_bar": "True",
            }

            artifact_removed_valid_times, artifact_times = _get_artifact_times(
                recording, **artifact_params, **job_kwargs
            )

            key["artifact_times"] = artifact_times
            key["artifact_removed_valid_times"] = artifact_removed_valid_times

            # set up a name for no-artifact times using recording id
            key["artifact_removed_interval_list_name"] = (
                recording_name
                + "_"
                + key["artifact_params_name"]
                + "_artifact_removed_valid_times"
            )

            ArtifactRemovedIntervalList.insert1(key, replace=True)

            # also insert into IntervalList
            tmp_key = {}
            tmp_key["nwb_file_name"] = key["nwb_file_name"]
            tmp_key["interval_list_name"] = key[
                "artifact_removed_interval_list_name"
            ]
            tmp_key["valid_times"] = key["artifact_removed_valid_times"]
            tmp_key["pipeline"] = "spikesorting_artifact_v0"
            IntervalList.insert1(tmp_key, replace=True)

            # insert into computed table
            self.insert1(key)


@schema
class ArtifactRemovedIntervalList(SpyglassMixin, dj.Manual):
    definition = """
    # Stores intervals without detected artifacts.
    # Note that entries can come from either ArtifactDetection() or alternative artifact removal analyses.
    artifact_removed_interval_list_name: varchar(170)
    ---
    -> ArtifactDetectionSelection
    artifact_removed_valid_times: longblob
    artifact_times: longblob # np array of artifact intervals
    """

    # See #630, #664. Excessive key length.


def _get_artifact_times(
    recording: si.BaseRecording,
    zscore_thresh: Union[float, None] = None,
    amplitude_thresh: Union[float, None] = None,
    proportion_above_thresh: float = 1.0,
    removal_window_ms: float = 1.0,
    verbose: bool = False,
    **job_kwargs,
):
    """Detects times during which artifacts do and do not occur.

    Artifacts are defined as periods where the absolute value of the recording
    signal exceeds one or both specified amplitude or z-score thresholds on the
    proportion of channels specified, with the period extended by the
    removal_window_ms/2 on each side. Z-score and amplitude threshold values of
    None are ignored.

    Parameters
    ----------
    recording : si.BaseRecording
    zscore_thresh : float, optional
        Stdev threshold for exclusion, should be >=0, defaults to None
    amplitude_thresh : float, optional
        Amplitude threshold for exclusion, should be >=0, defaults to None
    proportion_above_thresh : float, optional, should be>0 and <=1
        Proportion of electrodes that need to have threshold crossings, defaults to 1
    removal_window_ms : float, optional
        Width of the window in milliseconds to mask out per artifact
        (window/2 removed on each side of threshold crossing), defaults to 1 ms

    Returns
    -------
    artifact_removed_valid_times : np.ndarray
        Intervals of valid times where artifacts were not detected, unit: seconds
    artifact_intervals : np.ndarray
        Intervals in which artifacts are detected (including removal windows), unit: seconds
    """

    num_segments = recording.get_num_segments()
    if num_segments > 1:
        valid_timestamps = np.array([])
        for segment in range(num_segments):
            valid_timestamps = np.concatenate(
                (valid_timestamps, recording.get_times(segment_index=segment))
            )
        recording = si.concatenate_recordings([recording])
    elif num_segments == 1:
        valid_timestamps = recording.get_times(0)

    # if both thresholds are None, we skip artifract detection
    if amplitude_thresh is zscore_thresh is None:
        logger.info(
            "Amplitude and zscore thresholds are both None, "
            + "skipping artifact detection"
        )
        return np.asarray(
            [[valid_timestamps[0], valid_timestamps[-1]]]
        ), np.asarray([])

    # verify threshold parameters
    (
        amplitude_thresh,
        zscore_thresh,
        proportion_above_thresh,
    ) = _check_artifact_thresholds(
        amplitude_thresh=amplitude_thresh,
        zscore_thresh=zscore_thresh,
        proportion_above_thresh=proportion_above_thresh,
    )

    # detect frames that are above threshold in parallel
    n_jobs = ensure_n_jobs(recording, n_jobs=job_kwargs.get("n_jobs", 1))
    logger.info(f"Using {n_jobs} jobs...")

    if n_jobs == 1:
        init_args = (
            recording,
            zscore_thresh,
            amplitude_thresh,
            proportion_above_thresh,
        )
    else:
        init_args = (
            recording.to_dict(),
            zscore_thresh,
            amplitude_thresh,
            proportion_above_thresh,
        )

    executor = ChunkRecordingExecutor(
        recording=recording,
        func=_compute_artifact_chunk,
        init_func=_init_artifact_worker,
        init_args=init_args,
        verbose=verbose,
        handle_returns=True,
        job_name="detect_artifact_frames",
        **job_kwargs,
    )

    artifact_frames = executor.run()
    artifact_frames = np.concatenate(artifact_frames)

    # turn ms to remove total into s to remove from either side of each
    # detected artifact
    half_removal_window_s = removal_window_ms / 2 / 1000

    if len(artifact_frames) == 0:
        recording_interval = np.asarray(
            [[valid_timestamps[0], valid_timestamps[-1]]]
        )
        artifact_times_empty = np.asarray([])
        logger.warning("No artifacts detected.")
        return recording_interval, artifact_times_empty

    # convert indices to intervals
    artifact_intervals = interval_from_inds(artifact_frames)

    # convert to seconds and pad with window
    artifact_intervals_s = np.zeros(
        (len(artifact_intervals), 2), dtype=np.float64
    )
    for interval_idx, interval in enumerate(artifact_intervals):
        artifact_intervals_s[interval_idx] = [
            valid_timestamps[interval[0]] - half_removal_window_s,
            np.minimum(
                valid_timestamps[interval[1]] + half_removal_window_s,
                valid_timestamps[-1],
            ),
        ]
    # make the artifact intervals disjoint
    if len(artifact_intervals_s) > 1:
        artifact_intervals_s = reduce(_union_concat, artifact_intervals_s)

    # convert seconds back to indices
    artifact_intervals_new = []
    for artifact_interval_s in artifact_intervals_s:
        artifact_intervals_new.append(
            np.searchsorted(valid_timestamps, artifact_interval_s)
        )

    # compute set difference between intervals (of indices)
    try:
        # if artifact_intervals_new is a list of lists then len(artifact_intervals_new[0]) is the number of intervals
        # otherwise artifact_intervals_new is a list of ints and len(artifact_intervals_new[0]) is not defined
        len(artifact_intervals_new[0])
    except TypeError:
        # convert to list of lists
        artifact_intervals_new = [artifact_intervals_new]
    artifact_removed_valid_times_ind = interval_set_difference_inds(
        [(0, len(valid_timestamps) - 1)], artifact_intervals_new
    )

    # convert back to seconds
    artifact_removed_valid_times = []
    for i in artifact_removed_valid_times_ind:
        artifact_removed_valid_times.append(
            (valid_timestamps[i[0]], valid_timestamps[i[1]])
        )

    return np.asarray(artifact_removed_valid_times), artifact_intervals_s

import warnings
from functools import reduce
from typing import Union

import datajoint as dj
import numpy as np
import scipy.stats as stats
import spikeinterface as si
from spikeinterface.core.job_tools import ChunkRecordingExecutor, ensure_n_jobs

from ..common.common_interval import (
    IntervalList,
    _union_concat,
    interval_from_inds,
    interval_list_intersect,
)
from ..utils.nwb_helper_fn import get_valid_intervals
from .spikesorting_recording import SpikeSortingRecording

schema = dj.schema("spikesorting_artifact")


@schema
class ArtifactDetectionParameters(dj.Manual):
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
class ArtifactDetectionSelection(dj.Manual):
    definition = """
    # Specifies artifact detection parameters to apply to a sort group's recording.
    -> SpikeSortingRecording
    -> ArtifactDetectionParameters
    ---
    custom_artifact_detection=0 : tinyint
    """


@schema
class ArtifactDetection(dj.Computed):
    definition = """
    # Stores artifact times and valid no-artifact times as intervals.
    -> ArtifactDetectionSelection
    ---
    artifact_times: longblob # np array of artifact intervals
    artifact_removed_valid_times: longblob # np array of valid no-artifact intervals
    artifact_removed_interval_list_name: varchar(200) # name of the array of no-artifact valid time intervals
    """

    def make(self, key):
        if not (ArtifactDetectionSelection & key).fetch1("custom_artifact_detection"):
            # get the dict of artifact params associated with this artifact_params_name
            artifact_params = (ArtifactDetectionParameters & key).fetch1(
                "artifact_params"
            )

            recording_path = (SpikeSortingRecording & key).fetch1("recording_path")
            recording_name = SpikeSortingRecording._get_recording_name(key)
            recording = si.load_extractor(recording_path)

            job_kwargs = {"chunk_duration": "10s", "n_jobs": 4, "progress_bar": "True"}

            artifact_removed_valid_times, artifact_times = _get_artifact_times(
                recording, **artifact_params, **job_kwargs
            )

            # NOTE: decided not to do this but to just create a single long segment; keep for now
            # get artifact times by segment
            # if AppendSegmentRecording, get artifact times for each segment
            # if isinstance(recording, AppendSegmentRecording):
            #     artifact_removed_valid_times = []
            #     artifact_times = []
            #     for rec in recording.recording_list:
            #         rec_valid_times, rec_artifact_times = _get_artifact_times(rec, **artifact_params)
            #         for valid_times in rec_valid_times:
            #             artifact_removed_valid_times.append(valid_times)
            #         for artifact_times in rec_artifact_times:
            #             artifact_times.append(artifact_times)
            #     artifact_removed_valid_times = np.asarray(artifact_removed_valid_times)
            #     artifact_times = np.asarray(artifact_times)
            # else:
            #     artifact_removed_valid_times, artifact_times = _get_artifact_times(recording, **artifact_params)

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
            tmp_key["interval_list_name"] = key["artifact_removed_interval_list_name"]
            tmp_key["valid_times"] = key["artifact_removed_valid_times"]
            IntervalList.insert1(tmp_key, replace=True)

            # insert into computed table
            self.insert1(key)


@schema
class ArtifactRemovedIntervalList(dj.Manual):
    definition = """
    # Stores intervals without detected artifacts.
    # Note that entries can come from either ArtifactDetection() or alternative artifact removal analyses.
    artifact_removed_interval_list_name: varchar(200)
    ---
    -> ArtifactDetectionSelection
    artifact_removed_valid_times: longblob
    artifact_times: longblob # np array of artifact intervals
    """


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
    Artifacts are defined as periods where the absolute value of the recording signal exceeds one
    or both specified amplitude or zscore thresholds on the proportion of channels specified,
    with the period extended by the removal_window_ms/2 on each side. Z-score and amplitude
    threshold values of None are ignored.

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

    if recording.get_num_segments() > 1:
        valid_timestamps = np.array([])
        for segment in range(recording.get_num_segments()):
            valid_timestamps = np.concatenate(
                (valid_timestamps, recording.get_times(segment_index=segment))
            )
        recording = si.concatenate_recordings([recording])
    elif recording.get_num_segments() == 1:
        valid_timestamps = recording.get_times(0)

    # if both thresholds are None, we skip artifract detection
    if (amplitude_thresh is None) and (zscore_thresh is None):
        recording_interval = np.asarray([valid_timestamps[0], valid_timestamps[-1]])
        artifact_times_empty = np.asarray([])
        print(
            "Amplitude and zscore thresholds are both None, skipping artifact detection"
        )
        return recording_interval, artifact_times_empty

    # verify threshold parameters
    (
        amplitude_thresh,
        zscore_thresh,
        proportion_above_thresh,
    ) = _check_artifact_thresholds(
        amplitude_thresh, zscore_thresh, proportion_above_thresh
    )

    # detect frames that are above threshold in parallel
    n_jobs = ensure_n_jobs(recording, n_jobs=job_kwargs.get("n_jobs", 1))
    print(f"using {n_jobs} jobs...")

    func = _compute_artifact_chunk
    init_func = _init_artifact_worker
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
        recording,
        func,
        init_func,
        init_args,
        verbose=verbose,
        handle_returns=True,
        job_name="detect_artifact_frames",
        **job_kwargs,
    )
    artifact_frames = executor.run()
    artifact_frames = np.concatenate(artifact_frames)

    # turn ms to remove total into s to remove from either side of each detected artifact
    half_removal_window_s = removal_window_ms / 1000 * 0.5

    if len(artifact_frames) == 0:
        recording_interval = np.asarray([[valid_timestamps[0], valid_timestamps[-1]]])
        artifact_times_empty = np.asarray([])
        print("No artifacts detected.")
        return recording_interval, artifact_times_empty

    artifact_intervals = interval_from_inds(artifact_frames)

    artifact_intervals_s = np.zeros((len(artifact_intervals), 2), dtype=np.float64)
    for interval_idx, interval in enumerate(artifact_intervals):
        artifact_intervals_s[interval_idx] = [
            valid_timestamps[interval[0]] - half_removal_window_s,
            valid_timestamps[interval[1]] + half_removal_window_s,
        ]
    artifact_intervals_s = reduce(_union_concat, artifact_intervals_s)

    valid_intervals = get_valid_intervals(
        valid_timestamps, recording.get_sampling_frequency(), 1.5, 0.000001
    )
    artifact_removed_valid_times = interval_list_intersect(
        valid_intervals, artifact_intervals_s
    )

    return artifact_removed_valid_times, artifact_intervals_s


def _init_artifact_worker(
    recording, zscore_thresh=None, amplitude_thresh=None, proportion_above_thresh=1.0
):
    # create a local dict per worker
    worker_ctx = {}
    if isinstance(recording, dict):
        worker_ctx["recording"] = si.load_extractor(recording)
    else:
        worker_ctx["recording"] = recording
    worker_ctx["zscore_thresh"] = zscore_thresh
    worker_ctx["amplitude_thresh"] = amplitude_thresh
    worker_ctx["proportion_above_thresh"] = proportion_above_thresh
    return worker_ctx


def _compute_artifact_chunk(segment_index, start_frame, end_frame, worker_ctx):
    recording = worker_ctx["recording"]
    zscore_thresh = worker_ctx["zscore_thresh"]
    amplitude_thresh = worker_ctx["amplitude_thresh"]
    proportion_above_thresh = worker_ctx["proportion_above_thresh"]
    # compute the number of electrodes that have to be above threshold
    nelect_above = np.ceil(proportion_above_thresh * len(recording.get_channel_ids()))

    traces = recording.get_traces(
        segment_index=segment_index, start_frame=start_frame, end_frame=end_frame
    )

    # find the artifact occurrences using one or both thresholds, across channels
    if (amplitude_thresh is not None) and (zscore_thresh is None):
        above_a = np.abs(traces) > amplitude_thresh
        above_thresh = (
            np.ravel(np.argwhere(np.sum(above_a, axis=1) >= nelect_above)) + start_frame
        )
    elif (amplitude_thresh is None) and (zscore_thresh is not None):
        dataz = np.abs(stats.zscore(traces, axis=1))
        above_z = dataz > zscore_thresh
        above_thresh = (
            np.ravel(np.argwhere(np.sum(above_z, axis=1) >= nelect_above)) + start_frame
        )
    else:
        above_a = np.abs(traces) > amplitude_thresh
        dataz = np.abs(stats.zscore(traces, axis=1))
        above_z = dataz > zscore_thresh
        above_thresh = (
            np.ravel(
                np.argwhere(
                    np.sum(np.logical_or(above_z, above_a), axis=1) >= nelect_above
                )
            )
            + start_frame
        )

    return above_thresh


def _check_artifact_thresholds(
    amplitude_thresh, zscore_thresh, proportion_above_thresh
):
    """Alerts user to likely unintended parameters. Not an exhaustive verification.

    Parameters
    ----------
    zscore_thresh: float
    amplitude_thresh: float
    proportion_above_thresh: float

    Return
    ------
    zscore_thresh: float
    amplitude_thresh: float
    proportion_above_thresh: float

    Raise
    ------
    ValueError: if signal thresholds are negative
    """
    # amplitude or zscore thresholds should be negative, as they are applied to an absolute signal
    signal_thresholds = [t for t in [amplitude_thresh, zscore_thresh] if t is not None]
    for t in signal_thresholds:
        if t < 0:
            raise ValueError("Amplitude and Z-Score thresholds must be >= 0, or None")

    # proportion_above_threshold should be in [0:1] inclusive
    if proportion_above_thresh < 0:
        warnings.warn(
            "Warning: proportion_above_thresh must be a proportion >0 and <=1."
            f" Using proportion_above_thresh = 0.01 instead of {str(proportion_above_thresh)}"
        )
        proportion_above_thresh = 0.01
    elif proportion_above_thresh > 1:
        warnings.warn(
            "Warning: proportion_above_thresh must be a proportion >0 and <=1. "
            f"Using proportion_above_thresh = 1 instead of {str(proportion_above_thresh)}"
        )
        proportion_above_thresh = 1
    return amplitude_thresh, zscore_thresh, proportion_above_thresh

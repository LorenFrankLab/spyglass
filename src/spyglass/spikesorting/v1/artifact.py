import warnings
from functools import reduce
from typing import Union, List

import datajoint as dj
import numpy as np
import scipy.stats as stats
import spikeinterface as si
import spikeinterface.extractors as se
from spikeinterface.core.job_tools import ChunkRecordingExecutor, ensure_n_jobs

from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.common.common_interval import (
    IntervalList,
    _union_concat,
    interval_from_inds,
    interval_list_complement,
)
from spyglass.spikesorting.v1.utils import generate_nwb_uuid
from spyglass.spikesorting.v1.recording import (
    SpikeSortingRecording,
    SpikeSortingRecordingSelection,
)

schema = dj.schema("spikesorting_v1_artifact")


@schema
class ArtifactDetectionParameters(dj.Lookup):
    definition = """
    # Parameter for detecting artifacts (non-neural high amplitude events)
    artifact_param_name : varchar(200)
    ---
    artifact_params : blob
    """

    contents = [
        [
            "default",
            {
                "zscore_thresh": None,
                "amplitude_thresh_uV": 3000,
                "proportion_above_thresh": 1.0,
                "removal_window_ms": 1.0,
                "chunk_duration": "10s",
                "n_jobs": 4,
                "progress_bar": "True",
            },
        ],
        [
            "none",
            {
                "zscore_thresh": None,
                "amplitude_thresh_uV": None,
                "chunk_duration": "10s",
                "n_jobs": 4,
                "progress_bar": "True",
            },
        ],
    ]

    @classmethod
    def insert_default(cls):
        cls.insert(cls.contents, skip_duplicates=True)


@schema
class ArtifactDetectionSelection(dj.Manual):
    definition = """
    # Processed recording and artifact detection parameters. Use `insert_selection` method to insert new rows.
    artifact_id: varchar(30)
    ---
    -> SpikeSortingRecording
    -> ArtifactDetectionParameters
    """

    @classmethod
    def insert_selection(cls, key: dict):
        """Insert a row into ArtifactDetectionSelection with an
        automatically generated unique artifact ID as the sole primary key.

        Parameters
        ----------
        key : dict
            primary key of SpikeSortingRecording and ArtifactDetectionParameters

        Returns
        -------
        artifact_id : str
            the unique artifact ID serving as primary key for ArtifactDetectionSelection
        """
        if len((cls & key).fetch()) > 0:
            print(
                "This row has already been inserted into ArtifactDetectionSelection."
            )
            return (cls & key).fetch1()
        key["artifact_id"] = generate_nwb_uuid(
            (SpikeSortingRecordingSelection & key).fetch1("nwb_file_name"),
            "A",
            6,
        )
        cls.insert1(key, skip_duplicates=True)
        return key


@schema
class ArtifactDetection(dj.Computed):
    definition = """
    # Detects artifacts (e.g. large transients from movement) and saves artifact-free intervals in IntervalList.
    -> ArtifactDetectionSelection
    """

    def make(self, key):
        # FETCH:
        # - artifact parameters
        # - recording analysis nwb file
        artifact_params, recording_analysis_nwb_file = (
            ArtifactDetectionParameters
            * SpikeSortingRecording
            * ArtifactDetectionSelection
            & key
        ).fetch1("artifact_params", "analysis_file_name")
        sort_interval_valid_times = (
            IntervalList
            & {
                "nwb_file_name": (
                    SpikeSortingRecordingSelection * ArtifactDetectionSelection
                    & key
                ).fetch1("nwb_file_name"),
                "interval_list_name": (
                    SpikeSortingRecordingSelection * ArtifactDetectionSelection
                    & key
                ).fetch1("interval_list_name"),
            }
        ).fetch1("valid_times")
        # DO:
        # - load recording
        recording_analysis_nwb_file_abs_path = AnalysisNwbfile.get_abs_path(
            recording_analysis_nwb_file
        )
        recording = se.read_nwb_recording(
            recording_analysis_nwb_file_abs_path, load_time_vector=True
        )

        # - detect artifacts
        artifact_removed_valid_times, _ = _get_artifact_times(
            recording,
            sort_interval_valid_times,
            **artifact_params,
        )

        # INSERT
        # - into IntervalList
        IntervalList.insert1(
            dict(
                nwb_file_name=(
                    SpikeSortingRecordingSelection * ArtifactDetectionSelection
                    & key
                ).fetch1("nwb_file_name"),
                interval_list_name=key["artifact_id"],
                valid_times=artifact_removed_valid_times,
            ),
            skip_duplicates=True,
        )
        # - into ArtifactRemovedInterval
        self.insert1(key)


def _get_artifact_times(
    recording: si.BaseRecording,
    sort_interval_valid_times: List[List],
    zscore_thresh: Union[float, None] = None,
    amplitude_thresh_uV: Union[float, None] = None,
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
    sort_interval_valid_times : List[List]
        The sort interval for the recording, unit: seconds
    zscore_thresh : float, optional
        Stdev threshold for exclusion, should be >=0, defaults to None
    amplitude_thresh_uV : float, optional
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

    valid_timestamps = recording.get_times()

    # if both thresholds are None, we skip artifract detection
    if amplitude_thresh_uV is zscore_thresh is None:
        print(
            "Amplitude and zscore thresholds are both None, "
            + "skipping artifact detection"
        )
        return np.asarray(
            [valid_timestamps[0], valid_timestamps[-1]]
        ), np.asarray([])

    # verify threshold parameters
    (
        amplitude_thresh_uV,
        zscore_thresh,
        proportion_above_thresh,
    ) = _check_artifact_thresholds(
        amplitude_thresh_uV, zscore_thresh, proportion_above_thresh
    )

    # detect frames that are above threshold in parallel
    n_jobs = ensure_n_jobs(recording, n_jobs=job_kwargs.get("n_jobs", 1))
    print(f"Using {n_jobs} jobs...")
    func = _compute_artifact_chunk
    init_func = _init_artifact_worker
    if n_jobs == 1:
        init_args = (
            recording,
            zscore_thresh,
            amplitude_thresh_uV,
            proportion_above_thresh,
        )
    else:
        init_args = (
            recording.to_dict(),
            zscore_thresh,
            amplitude_thresh_uV,
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
    half_removal_window_s = removal_window_ms / 2 / 1000

    if len(artifact_frames) == 0:
        recording_interval = np.asarray(
            [[valid_timestamps[0], valid_timestamps[-1]]]
        )
        artifact_times_empty = np.asarray([])
        print("No artifacts detected.")
        return recording_interval, artifact_times_empty

    # convert indices to intervals
    artifact_intervals = interval_from_inds(artifact_frames)

    # convert to seconds and pad with window
    artifact_intervals_s = np.zeros(
        (len(artifact_intervals), 2), dtype=np.float64
    )
    for interval_idx, interval in enumerate(artifact_intervals):
        interv_ind = [
            np.searchsorted(
                valid_timestamps,
                valid_timestamps[interval[0]] - half_removal_window_s,
            ),
            np.searchsorted(
                valid_timestamps,
                valid_timestamps[interval[1]] + half_removal_window_s,
            ),
        ]
        artifact_intervals_s[interval_idx] = [
            valid_timestamps[interv_ind[0]],
            valid_timestamps[interv_ind[1]],
        ]

    # make the artifact intervals disjoint
    artifact_intervals_s = reduce(_union_concat, artifact_intervals_s)

    # find non-artifact intervals in timestamps
    artifact_removed_valid_times = interval_list_complement(
        sort_interval_valid_times, artifact_intervals_s, min_length=1
    )
    artifact_removed_valid_times = reduce(
        _union_concat, artifact_removed_valid_times
    )

    return artifact_removed_valid_times, artifact_intervals_s


def _init_artifact_worker(
    recording,
    zscore_thresh=None,
    amplitude_thresh_uV=None,
    proportion_above_thresh=1.0,
):
    # create a local dict per worker
    worker_ctx = {}
    if isinstance(recording, dict):
        worker_ctx["recording"] = si.load_extractor(recording)
    else:
        worker_ctx["recording"] = recording
    worker_ctx["zscore_thresh"] = zscore_thresh
    worker_ctx["amplitude_thresh_uV"] = amplitude_thresh_uV
    worker_ctx["proportion_above_thresh"] = proportion_above_thresh
    return worker_ctx


def _compute_artifact_chunk(segment_index, start_frame, end_frame, worker_ctx):
    recording = worker_ctx["recording"]
    zscore_thresh = worker_ctx["zscore_thresh"]
    amplitude_thresh_uV = worker_ctx["amplitude_thresh_uV"]
    proportion_above_thresh = worker_ctx["proportion_above_thresh"]
    # compute the number of electrodes that have to be above threshold
    nelect_above = np.ceil(
        proportion_above_thresh * len(recording.get_channel_ids())
    )

    traces = recording.get_traces(
        segment_index=segment_index,
        start_frame=start_frame,
        end_frame=end_frame,
    )

    # find the artifact occurrences using one or both thresholds, across channels
    if (amplitude_thresh_uV is not None) and (zscore_thresh is None):
        above_a = np.abs(traces) > amplitude_thresh_uV
        above_thresh = (
            np.ravel(np.argwhere(np.sum(above_a, axis=1) >= nelect_above))
            + start_frame
        )
    elif (amplitude_thresh_uV is None) and (zscore_thresh is not None):
        dataz = np.abs(stats.zscore(traces, axis=1))
        above_z = dataz > zscore_thresh
        above_thresh = (
            np.ravel(np.argwhere(np.sum(above_z, axis=1) >= nelect_above))
            + start_frame
        )
    else:
        above_a = np.abs(traces) > amplitude_thresh_uV
        dataz = np.abs(stats.zscore(traces, axis=1))
        above_z = dataz > zscore_thresh
        above_thresh = (
            np.ravel(
                np.argwhere(
                    np.sum(np.logical_or(above_z, above_a), axis=1)
                    >= nelect_above
                )
            )
            + start_frame
        )

    return above_thresh


def _check_artifact_thresholds(
    amplitude_thresh_uV, zscore_thresh, proportion_above_thresh
):
    """Alerts user to likely unintended parameters. Not an exhaustive verification.

    Parameters
    ----------
    zscore_thresh: float
    amplitude_thresh_uV: float
    proportion_above_thresh: float

    Return
    ------
    zscore_thresh: float
    amplitude_thresh_uV: float
    proportion_above_thresh: float

    Raise
    ------
    ValueError: if signal thresholds are negative
    """
    # amplitude or zscore thresholds should be negative, as they are applied to an absolute signal
    signal_thresholds = [
        t for t in [amplitude_thresh_uV, zscore_thresh] if t is not None
    ]
    for t in signal_thresholds:
        if t < 0:
            raise ValueError(
                "Amplitude and Z-Score thresholds must be >= 0, or None"
            )

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
    return amplitude_thresh_uV, zscore_thresh, proportion_above_thresh


def merge_intervals(intervals):
    """Takes a list of intervals each of which is [start_time, stop_time]
    and takes union over intervals that are intersecting

    Parameters
    ----------
    intervals : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    if len(intervals) == 0:
        return []

    # Sort the intervals based on their start times
    intervals.sort(key=lambda x: x[0])

    merged = [intervals[0]]

    for i in range(1, len(intervals)):
        current_start, current_stop = intervals[i]
        last_merged_start, last_merged_stop = merged[-1]

        if current_start <= last_merged_stop:
            # Overlapping intervals, merge them
            merged[-1] = [
                last_merged_start,
                max(last_merged_stop, current_stop),
            ]
        else:
            # Non-overlapping intervals, add the current one to the list
            merged.append([current_start, current_stop])

    return np.asarray(merged)

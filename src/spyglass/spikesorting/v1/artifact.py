import uuid
from typing import Union

import datajoint as dj
import numpy as np
import spikeinterface as si
import spikeinterface.extractors as se
from spikeinterface.core.job_tools import ChunkRecordingExecutor, ensure_n_jobs

from spyglass.common.common_interval import Interval, IntervalList
from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.spikesorting.utils import (
    _check_artifact_thresholds,
    _compute_artifact_chunk,
    _init_artifact_worker,
)
from spyglass.spikesorting.v1.recording import (
    SpikeSortingRecording,
    SpikeSortingRecordingSelection,
)
from spyglass.utils import SpyglassMixin, logger

schema = dj.schema("spikesorting_v1_artifact")


@schema
class ArtifactDetectionParameters(SpyglassMixin, dj.Lookup):
    """Parameters for detecting artifacts (non-neural high amplitude events).

    Attributes
    ----------
    artifact_param_name : str
        Name of the artifact detection parameters.
    artifact_params : dict
        Dictionary of parameters for artifact detection, which may include:
        zscore_thresh: float, optional
            Stdev threshold for exclusion, should be >=0, defaults to None
        amplitude_thresh_uV: float, optional
            Amplitude threshold for exclusion, should be >=0, defaults to None
        proportion_above_thresh: float, optional, should be>0 and <=1
            Proportion of electrodes that need to have threshold crossings,
            defaults to 1
        removal_window_ms: float, optional
            Width of the window in milliseconds to mask out per artifact
            (window/2 removed on each side of threshold crossing),
            defaults to 1 ms
        chunk_duration: str, optional
            Duration of chunks to process in seconds, defaults to "10s"
        n_jobs: int, optional
            Number of parallel jobs to use, defaults to 4
        progress_bar: bool, optional
            Whether to show a progress bar, defaults to True
    """

    definition = """
    # Parameters for detecting artifacts (non-neural high amplitude events).
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
        """Insert default parameters into ArtifactDetectionParameters."""
        cls.insert(cls.contents, skip_duplicates=True)


@schema
class ArtifactDetectionSelection(SpyglassMixin, dj.Manual):
    definition = """
    # Processed recording/artifact detection parameters. See `insert_selection`.
    artifact_id: uuid
    ---
    -> SpikeSortingRecording
    -> ArtifactDetectionParameters
    """

    @classmethod
    def insert_selection(cls, key: dict):
        """Insert a row into ArtifactDetectionSelection.

        Automatically generates a unique artifact ID as the sole primary key.

        Parameters
        ----------
        key : dict
            primary key of SpikeSortingRecording and ArtifactDetectionParameters

        Returns
        -------
        artifact_id : str
            the unique artifact ID serving as primary key for
            ArtifactDetectionSelection
        """
        query = cls & key
        if query:
            logger.warning("Similar row(s) already inserted.")
            return query.fetch(as_dict=True)
        key["artifact_id"] = uuid.uuid4()
        cls.insert1(key, skip_duplicates=True)
        return key


@schema
class ArtifactDetection(SpyglassMixin, dj.Computed):
    definition = """
    # Detected artifacts (e.g. large transients from movement).
    # Intervals are stored in IntervalList with `artifact_id` as `interval_list_name`.
    -> ArtifactDetectionSelection
    """

    def make(self, key):
        """Populate ArtifactDetection with detected artifacts.

        1. Fetches...
            - Artifact parameters from ArtifactDetectionParameters
            - Recording analysis NWB file from SpikeSortingRecording
            - Valid times from IntervalList
        2. Load the recording from the NWB file with spikeinterface
        3. Detect artifacts using module-level `_get_artifact_times`
        4. Insert result into IntervalList with `artifact_id` as
            `interval_list_name`
        """
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
        ).fetch_interval()

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
                interval_list_name=str(key["artifact_id"]),
                valid_times=artifact_removed_valid_times,
                pipeline="spikesorting_artifact_v1",
            ),
            skip_duplicates=True,
        )
        # - into ArtifactRemovedInterval
        self.insert1(key)


def _get_artifact_times(
    recording: si.BaseRecording,
    sort_interval_valid_times: Interval,
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
    sort_interval_valid_times : Interval
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

    # CB: V0 checks num segments. Is that no longer necessary?

    valid_timestamps = recording.get_times()

    # if both thresholds are None, we skip artifract detection
    if amplitude_thresh_uV is zscore_thresh is None:
        logger.info(
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
        amplitude_thresh=amplitude_thresh_uV,
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

    if len(artifact_frames) == 0:
        recording_interval = np.asarray(
            [[valid_timestamps[0], valid_timestamps[-1]]]
        )
        artifact_times_empty = np.asarray([])
        logger.warning("No artifacts detected.")
        return recording_interval, artifact_times_empty

    # convert indices to intervals
    artifact_intervals_s = Interval(
        artifact_frames, from_inds=True
    ).add_removal_window(removal_window_ms, valid_timestamps)

    # find non-artifact intervals in timestamps
    artifact_removed_valid_times = sort_interval_valid_times.subtract(
        artifact_intervals_s, min_length=1
    ).union_consolidate()

    return artifact_removed_valid_times.times, artifact_intervals_s


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
    # TODO: Migrate to common_interval.py

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

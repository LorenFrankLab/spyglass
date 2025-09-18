from typing import Union

import datajoint as dj
import numpy as np
import spikeinterface as si
from spikeinterface.core.job_tools import ChunkRecordingExecutor, ensure_n_jobs

from spyglass.common.common_interval import Interval, IntervalList
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
    """Parameters for detecting artifact times within a sort group.

    Attributes
    ----------
    artifact_params_name : str
        Name of the artifact detection parameters.
    artifact_params : dict
        Dictionary of parameters for artifact detection, including:
            zscore_thresh : float or None
                Stdev threshold for exclusion, should be >=0
            amplitude_thresh : float or None
                Amplitude threshold for exclusion, should be >=0
            proportion_above_thresh : float
                Proportion of electrodes that need to have threshold crossings,
                should be >0 and <=1
            removal_window_ms : float
                Width of the window in milliseconds to mask out per artifact
    """

    definition = """
    # Parameters for detecting artifact times within a sort group.
    artifact_params_name: varchar(200)
    ---
    artifact_params: blob  # dictionary of parameters
    """

    def insert_default(self):
        """Insert the default artifact parameters."""
        params_default = {
            "zscore_thresh": 5.0,  # must be None or >= 0
            "amplitude_thresh": 3000,  # must be None or >= 0
            "proportion_above_thresh": 0.5,  # must be > 0 and <= 1
            "removal_window_ms": 1.0,  # in milliseconds
        }
        params_none = {
            "zscore_thresh": None,  # must be None or >= 0
            "amplitude_thresh": None,  # must be None or >= 0
        }
        self.insert(
            [("default", params_default), ("none", params_none)],
            skip_duplicates=True,
        )


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
        query = ArtifactDetectionParameters * ArtifactDetectionSelection & key

        if query.fetch1("custom_artifact_detection"):
            return

        # get the dict of artifact params associated with this artifact_params_name
        artifact_params = query.fetch1("artifact_params")

        recording_name = SpikeSortingRecording._get_recording_name(key)
        recording = SpikeSortingRecording().load_recording(key)

        job_kwargs = {
            "chunk_duration": "10s",
            "n_jobs": 4,
            "progress_bar": "True",
        }

        artifact_removed_valid_times, artifact_times = _get_artifact_times(
            recording, **artifact_params, **job_kwargs
        )

        # set up a name for no-artifact times using recording id
        artifact_removed_name = (
            f"{recording_name}_{key['artifact_params_name']}"
            + "_artifact_removed_valid_times"
        )

        artifact_array = getattr(artifact_times, "times", artifact_times)

        key.update(
            {
                "artifact_times": artifact_array,
                "artifact_removed_valid_times": artifact_removed_valid_times,
                "artifact_removed_interval_list_name": artifact_removed_name,
            }
        )

        interval_key = {
            "nwb_file_name": key["nwb_file_name"],
            "interval_list_name": artifact_removed_name,
            "valid_times": artifact_removed_valid_times,
            "pipeline": "spikesorting_artifact_v0",
        }

        ArtifactRemovedIntervalList.insert1(key, replace=True)
        IntervalList.insert1(interval_key, replace=True)
        # insert into computed table
        self.insert1(key)


@schema
class ArtifactRemovedIntervalList(SpyglassMixin, dj.Manual):
    definition = """
    # Stores intervals without detected artifacts.
    # Note that entries can come from either ArtifactDetection() or
    # alternative artifact removal analyses.
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

    if len(artifact_frames) == 0:
        recording_interval = np.asarray(
            [[valid_timestamps[0], valid_timestamps[-1]]]
        )
        artifact_times_empty = np.asarray([])
        logger.warning("No artifacts detected.")
        return recording_interval, artifact_times_empty

    # turn ms to remove total into s to remove from either side of each
    # detected artifact
    # generate intervals for artifact frame indices
    artifact_intervals_s = Interval(
        artifact_frames, from_inds=True
    ).add_removal_window(removal_window_ms, valid_timestamps)
    artifact_interv_idx = artifact_intervals_s.to_indices(
        valid_timestamps, as_interval=True
    )

    artifact_removed_valid_times = artifact_interv_idx.subtract(
        [(0, len(valid_timestamps) - 1)], reverse=True
    ).to_seconds(valid_timestamps)

    return np.asarray(artifact_removed_valid_times), artifact_intervals_s

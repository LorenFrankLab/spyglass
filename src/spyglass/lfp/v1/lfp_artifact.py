import datajoint as dj

from spyglass.common.common_interval import IntervalList
from spyglass.lfp.v1.lfp import LFPV1
from spyglass.lfp.v1.lfp_artifact_difference_detection import (
    difference_artifact_detector,
)
from spyglass.lfp.v1.lfp_artifact_MAD_detection import mad_artifact_detector

schema = dj.schema("lfp_v1")

MIN_LFP_INTERVAL_DURATION = 1.0  # 1 second minimum interval duration

ARTIFACT_DETECTION_ALGORITHMS = {
    "difference": difference_artifact_detector,
    "mad": mad_artifact_detector,  # MAD = median absolute deviation
}


@schema
class LFPArtifactDetectionParameters(dj.Manual):
    definition = """
    # Parameters for detecting LFP artifact times within a LFP group.
    artifact_params_name: varchar(200)
    ---
    artifact_params: blob  # dictionary of parameters
    """

    def insert_default(self):
        """Insert the default artifact parameters with an appropriate parameter dict."""
        artifact_params = {
            "artifact_detection_algorithm": "difference",
            "artifact_detection_algorithm_params": {
                "amplitude_thresh_1st": 500,  # must be None or >= 0
                "proportion_above_thresh_1st": 0.1,
                "amplitude_thresh_2nd": 1000,  # must be None or >= 0
                "proportion_above_thresh_2nd": 0.05,
                "removal_window_ms": 10.0,  # in milliseconds
                "local_window_ms": 40.0,  # in milliseconds
            },
        }

        self.insert1(
            ["default_difference", artifact_params], skip_duplicates=True
        )

        artifact_params_none = {
            "artifact_detection_algorithm": "difference",
            "artifact_detection_algorithm_params": {
                "amplitude_thresh_1st": None,  # must be None or >= 0
                "proportion_above_thresh_1st": None,
                "amplitude_thresh_2nd": None,  # must be None or >= 0
                "proportion_above_thresh_2nd": None,
                "removal_window_ms": None,  # in milliseconds
                "local_window_ms": None,  # in milliseconds
            },
        }
        self.insert1(["none", artifact_params_none], skip_duplicates=True)

        artifact_params_mad = {
            "artifact_detection_algorithm": "mad",
            "artifact_detection_algorithm_params": {
                "mad_thresh": 6.0,  # akin to z-score standard deviations if the distribution is normal
                "proportion_above_thresh": 0.1,
                "removal_window_ms": 10.0,  # in milliseconds
            },
        }
        self.insert1(["default_mad", artifact_params_mad], skip_duplicates=True)


@schema
class LFPArtifactDetectionSelection(dj.Manual):
    definition = """
    # Specifies artifact detection parameters to apply to a sort group's recording.
    -> LFPV1
    -> LFPArtifactDetectionParameters
    ---
    """


@schema
class LFPArtifactDetection(dj.Computed):
    definition = """
    # Stores artifact times and valid no-artifact times as intervals.
    -> LFPArtifactDetectionSelection
    ---
    artifact_times: longblob # np array of artifact intervals
    artifact_removed_valid_times: longblob # np array of valid no-artifact intervals
    artifact_removed_interval_list_name: varchar(200) # name of the array of no-artifact valid time intervals
    """

    def make(self, key):
        artifact_params = (
            LFPArtifactDetectionParameters
            & {"artifact_params_name": key["artifact_params_name"]}
        ).fetch1("artifact_params")

        artifact_detection_algorithm = artifact_params[
            "artifact_detection_algorithm"
        ]
        artifact_detection_params = artifact_params[
            "artifact_detection_algorithm_params"
        ]

        # get LFP data
        lfp_eseries = (LFPV1() & key).fetch_nwb()[0]["lfp"]
        sampling_frequency = (LFPV1() & key).fetch("lfp_sampling_rate")[0]

        (
            artifact_removed_valid_times,
            artifact_times,
        ) = ARTIFACT_DETECTION_ALGORITHMS[artifact_detection_algorithm](
            lfp_eseries,
            **artifact_detection_params,
            sampling_frequency=sampling_frequency,
        )

        key["artifact_times"] = artifact_times
        key["artifact_removed_valid_times"] = artifact_removed_valid_times

        # set up a name for no-artifact times using recording id
        # we need some name here for recording_name
        key["artifact_removed_interval_list_name"] = "_".join(
            [
                key["nwb_file_name"],
                key["target_interval_list_name"],
                "LFP",
                key["artifact_params_name"],
                "artifact_removed_valid_times",
            ]
        )

        LFPArtifactRemovedIntervalList.insert1(key, replace=True)

        # also insert into IntervalList
        interval_key = {
            "nwb_file_name": key["nwb_file_name"],
            "interval_list_name": key["artifact_removed_interval_list_name"],
            "valid_times": key["artifact_removed_valid_times"],
        }
        IntervalList.insert1(interval_key, replace=True)

        # insert into computed table
        self.insert1(key)


@schema
class LFPArtifactRemovedIntervalList(dj.Manual):
    definition = """
    # Stores intervals without detected artifacts.
    # Note that entries can come from either ArtifactDetection() or alternative artifact removal analyses.
    artifact_removed_interval_list_name: varchar(200)
    ---
    -> LFPArtifactDetectionSelection
    artifact_removed_valid_times: longblob
    artifact_times: longblob # np.array of artifact intervals
    """

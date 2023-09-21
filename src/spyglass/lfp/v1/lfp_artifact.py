import datajoint as dj

from spyglass.common.common_interval import IntervalList
from spyglass.lfp.v1.lfp import LFPV1
from spyglass.lfp.v1.lfp_artifact_difference_detection import (
    difference_artifact_detector,
)
from spyglass.lfp.v1.lfp_artifact_MAD_detection import mad_artifact_detector
import numpy as np
from spyglass.common import get_electrode_indices

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
        """Insert the default artifact parameters."""
        diff_params = [
            "default_difference",
            {
                "artifact_detection_algorithm": "difference",
                "artifact_detection_algorithm_params": {
                    "amplitude_thresh_1st": 500,  # must be None or >= 0
                    "proportion_above_thresh_1st": 0.1,
                    "amplitude_thresh_2nd": 1000,  # must be None or >= 0
                    "proportion_above_thresh_2nd": 0.05,
                    "removal_window_ms": 10.0,  # in milliseconds
                    "local_window_ms": 40.0,  # in milliseconds
                },
            },
        ]

        diff_ref_params = [
            "default_difference_ref",
            {
                "artifact_detection_algorithm": "difference",
                "artifact_detection_algorithm_params": {
                    "amplitude_thresh_1st": 500,  # must be None or >= 0
                    "proportion_above_thresh_1st": 0.1,
                    "amplitude_thresh_2nd": 1000,  # must be None or >= 0
                    "proportion_above_thresh_2nd": 0.05,
                    "removal_window_ms": 10.0,  # in milliseconds
                    "local_window_ms": 40.0,  # in milliseconds
                },
                "referencing": {
                    "ref_on": 1,
                    "reference_list": [0, 0, 0, 0, 0],
                    "electrode_list": [0, 0],
                },
            },
        ]

        no_params = [
            "none",
            {
                "artifact_detection_algorithm": "difference",
                "artifact_detection_algorithm_params": {
                    "amplitude_thresh_1st": None,  # must be None or >= 0
                    "proportion_above_thresh_1st": None,
                    "amplitude_thresh_2nd": None,  # must be None or >= 0
                    "proportion_above_thresh_2nd": None,
                    "removal_window_ms": None,  # in milliseconds
                    "local_window_ms": None,  # in milliseconds
                },
            },
        ]

        mad_params = [
            "default_mad",
            {
                "artifact_detection_algorithm": "mad",
                "artifact_detection_algorithm_params": {
                    # akin to z-score std dev if the distribution is normal
                    "mad_thresh": 6.0,
                    "proportion_above_thresh": 0.1,
                    "removal_window_ms": 10.0,  # in milliseconds
                },
            },
        ]

        self.insert(
            [diff_params, diff_ref_params, no_params, mad_params],
            skip_duplicates=True,
        )


@schema
class LFPArtifactDetectionSelection(dj.Manual):
    definition = """
    # Artifact detection parameters to apply to a sort group's recording.
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
    artifact_removed_valid_times: longblob # np array of no-artifact intervals
    artifact_removed_interval_list_name: varchar(200)
        # name of the array of no-artifact valid time intervals
    """

    def make(self, key):
        artifact_params = (
            LFPArtifactDetectionParameters
            & {"artifact_params_name": key["artifact_params_name"]}
        ).fetch1("artifact_params")

        algorithm = artifact_params["artifact_detection_algorithm"]
        params = artifact_params["artifact_detection_algorithm_params"]
        lfp_band_ref_id = artifact_params["referencing"]["reference_list"]

        # get LFP data
        lfp_eseries = (LFPV1 & key).fetch_nwb()[0]["lfp"]
        sampling_frequency = (LFPV1 & key).fetch("lfp_sampling_rate")[0]

        # do referencing at this step
        lfp_data = np.asarray(
            lfp_eseries.data[:, :],
            dtype=type(lfp_eseries.data[0][0]),
        )

        lfp_band_ref_index = get_electrode_indices(lfp_eseries, lfp_band_ref_id)
        lfp_band_elect_index = get_electrode_indices(
            lfp_eseries, artifact_params["referencing"]["electrode_list"]
        )

        # maybe this lfp_elec_list is supposed to be a list on indices

        for index, elect_index in enumerate(lfp_band_elect_index):
            if lfp_band_ref_id[index] == -1:
                continue
            lfp_data[:, elect_index] = (
                lfp_data[:, elect_index]
                - lfp_data[:, lfp_band_ref_index[index]]
            )

        is_diff = algorithm == "difference"
        data = lfp_data if is_diff else lfp_eseries
        ref = artifact_params["referencing"]["ref_on"] if is_diff else None

        (
            artifact_removed_valid_times,
            artifact_times,
        ) = ARTIFACT_DETECTION_ALGORITHMS[algorithm](
            data,
            **params,
            sampling_frequency=sampling_frequency,
            timestamps=lfp_eseries.timestamps if is_diff else None,
            referencing=ref,
        )

        key.update(
            dict(
                artifact_times=artifact_times,
                artifact_removed_valid_times=artifact_removed_valid_times,
                # name for no-artifact time name using recording id
                artifact_removed_interval_list_name="_".join(
                    [
                        key["nwb_file_name"],
                        key["target_interval_list_name"],
                        "LFP",
                        key["artifact_params_name"],
                        "artifact_removed_valid_times",
                    ]
                ),
            )
        )

        interval_key = {  # also insert into IntervalList
            "nwb_file_name": key["nwb_file_name"],
            "interval_list_name": key["artifact_removed_interval_list_name"],
            "valid_times": key["artifact_removed_valid_times"],
        }

        LFPArtifactRemovedIntervalList.insert1(key, replace=True)
        IntervalList.insert1(interval_key, replace=True)
        self.insert1(key)


@schema
class LFPArtifactRemovedIntervalList(dj.Manual):
    definition = """
    # Stores intervals without detected artifacts. Entries can come from either
    # ArtifactDetection() or alternative artifact removal analyses.
    artifact_removed_interval_list_name: varchar(200)
    ---
    -> LFPArtifactDetectionSelection
    artifact_removed_valid_times: longblob
    artifact_times: longblob # np.array of artifact intervals
    """

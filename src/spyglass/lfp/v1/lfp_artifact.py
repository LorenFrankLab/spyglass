import uuid

import datajoint as dj
import numpy as np

from spyglass.common import get_electrode_indices
from spyglass.common.common_interval import IntervalList
from spyglass.lfp.v1.lfp import LFPV1
from spyglass.lfp.v1.lfp_artifact_difference_detection import (
    difference_artifact_detector,
)
from spyglass.lfp.v1.lfp_artifact_MAD_detection import mad_artifact_detector
from spyglass.utils.dj_mixin import SpyglassMixin

schema = dj.schema("lfp_v1")

MIN_LFP_INTERVAL_DURATION = 1.0  # 1 second minimum interval duration

ARTIFACT_DETECTION_ALGORITHMS = {
    "difference": difference_artifact_detector,
    "mad": mad_artifact_detector,  # MAD = median absolute deviation
}


@schema
class LFPArtifactDetectionParameters(SpyglassMixin, dj.Manual):
    """Parameters for detecting LFP artifacts.

    Attributes
    ----------
    artifact_params_name : str
        Name of the artifact detection parameters.
    artifact_params : dict
        artifact_detection_algorithm: str
        artifact_detection_algorithm_params: dict
            amplitude_thresh_1st : float, optional
                Amplitude (ad units) threshold for exclusion, should be >=0,
                defaults to None
            amplitude_thresh_2nd : float, optional
                Amplitude (ad units) threshold for exclusion, should be >=0,
                defaults to None
            proportion_above_thresh_1st : float, optional,
                should be>0 and <=1. Proportion of electrodes that need to have
                threshold crossings, defaults to 1
            proportion_above_thresh_2nd : float, optional, should be>0 and <=1
                Proportion of electrodes that need to have threshold crossings,
                defaults to 1
            removal_window_ms : float, optional
                Width of the window in milliseconds to mask out per artifact
                (window/2 removed on each side of threshold crossing), defaults
                to 1 ms
            local_window_ms: float, optional
                Width of the local time window in milliseconds used for artifact
                detection. This parameter defines the duration of the local
                analysis window, defaults to None.
        referencing: dict, optional
            ref_on: bool
            reference_list: list of int
                Reference electrode IDs.
            electrode_list: list of int
                Electrode IDs to be referenced.
    """

    definition = """
    # Parameters for detecting LFP artifact times within a LFP group.
    artifact_params_name: varchar(64)
    ---
    artifact_params: blob  # dictionary of parameters
    """

    # See #630, #664. Excessive key length.

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
class LFPArtifactDetectionSelection(SpyglassMixin, dj.Manual):
    definition = """
    # Artifact detection parameters to apply to a sort group's recording.
    -> LFPV1
    -> LFPArtifactDetectionParameters
    ---
    """


@schema
class LFPArtifactDetection(SpyglassMixin, dj.Computed):
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
        """Populate the LFPArtifactDetection table with artifact times.

        1. Fetch parameters and LFP data from LFPArtifactDetectionParameters
            and LFPV1, respectively.
        2. Optionally reference the LFP data.
        3. Pass data to chosen artifact detection algorithm.
        3. Insert into LFPArtifactRemovedIntervalList, IntervalList, and
            LFPArtifactDetection.
        """
        artifact_params = (
            LFPArtifactDetectionParameters
            & {"artifact_params_name": key["artifact_params_name"]}
        ).fetch1("artifact_params")

        algorithm = artifact_params["artifact_detection_algorithm"]
        params = artifact_params["artifact_detection_algorithm_params"]

        # get LFP data
        lfp_eseries = (LFPV1 & key).fetch_nwb()[0]["lfp"]
        sampling_frequency = (LFPV1 & key).fetch("lfp_sampling_rate")[0]
        lfp_data = np.asarray(
            lfp_eseries.data[:, :],
            dtype=type(lfp_eseries.data[0][0]),
        )

        is_diff = algorithm == "difference"
        # do referencing at this step
        if "referencing" in artifact_params:
            ref = artifact_params["referencing"]["ref_on"] if is_diff else None
            lfp_band_ref_id = artifact_params["referencing"]["reference_list"]
            if artifact_params["referencing"]["ref_on"]:
                lfp_band_ref_index = get_electrode_indices(
                    lfp_eseries, lfp_band_ref_id
                )
                lfp_band_elect_index = get_electrode_indices(
                    lfp_eseries,
                    artifact_params["referencing"]["electrode_list"],
                )

                # maybe this lfp_elec_list is supposed to be a list on indices
                for index, elect_index in enumerate(lfp_band_elect_index):
                    if lfp_band_ref_id[index] == -1:
                        continue
                    lfp_data[:, elect_index] = (
                        lfp_data[:, elect_index]
                        - lfp_data[:, lfp_band_ref_index[index]]
                    )
        else:
            ref = False if is_diff else None

        data = lfp_data if is_diff else lfp_eseries

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
                artifact_removed_interval_list_name=uuid.uuid4(),
            )
        )

        interval_key = {  # also insert into IntervalList
            "nwb_file_name": key["nwb_file_name"],
            "interval_list_name": key["artifact_removed_interval_list_name"],
            "valid_times": key["artifact_removed_valid_times"],
            "pipeline": self.full_table_name,
        }

        LFPArtifactRemovedIntervalList.insert1(key)
        IntervalList.insert1(interval_key)
        self.insert1(key)


@schema
class LFPArtifactRemovedIntervalList(SpyglassMixin, dj.Manual):
    definition = """
    # Stores intervals without detected artifacts. Entries can come from either
    # ArtifactDetection() or alternative artifact removal analyses.
    artifact_removed_interval_list_name: varchar(128)
    ---
    -> LFPArtifactDetectionSelection
    artifact_removed_valid_times: longblob
    artifact_times: longblob # np.array of artifact intervals
    """

    # See #630, #664. Excessive key length.
    # 200 enties in database over 128. If string removed as above, all fit.

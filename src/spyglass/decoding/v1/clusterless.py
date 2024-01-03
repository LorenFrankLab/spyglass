"""Pipeline for decoding the animal's mental position and some category of interest
from unclustered spikes and spike waveform features. See [1] for details.

References
----------
[1] Denovellis, E. L. et al. Hippocampal replay of experience at real-world
speeds. eLife 10, e64505 (2021).

"""

import copy
import uuid
from pathlib import Path

import datajoint as dj
import numpy as np
import pandas as pd
import xarray as xr
from non_local_detector.models.base import ClusterlessDetector

from spyglass.common.common_interval import IntervalList  # noqa: F401
from spyglass.common.common_position import IntervalPositionInfo
from spyglass.common.common_session import Session  # noqa: F401
from spyglass.decoding.v1.core import (
    DecodingParameters,
    PositionGroup,
)  # noqa: F401
from spyglass.decoding.v1.waveform_features import (
    UnitWaveformFeatures,
)  # noqa: F401
from spyglass.position.position_merge import PositionOutput  # noqa: F401
from spyglass.settings import config
from spyglass.utils import SpyglassMixin, logger

schema = dj.schema("decoding_clusterless_v1")


@schema
class UnitWaveformFeaturesGroup(SpyglassMixin, dj.Manual):
    definition = """
    -> Session
    waveform_features_group_name: varchar(80)
    """

    class UnitFeatures(SpyglassMixin, dj.Part):
        definition = """
        -> UnitWaveformFeaturesGroup
        -> UnitWaveformFeatures
        """

    def create_group(
        self, nwb_file_name: str, group_name: str, keys: list[dict]
    ):
        group_key = {
            "nwb_file_name": nwb_file_name,
            "waveform_features_group_name": group_name,
        }
        self.insert1(
            group_key,
            skip_duplicates=True,
        )
        for key in keys:
            self.UnitFeatures.insert1(
                {**key, **group_key},
                skip_duplicates=True,
            )


@schema
class ClusterlessDecodingSelection(SpyglassMixin, dj.Manual):
    definition = """
    -> UnitWaveformFeaturesGroup
    -> PositionGroup
    -> DecodingParameters
    -> IntervalList.proj(encoding_interval='interval_list_name')
    -> IntervalList.proj(decoding_interval='interval_list_name')
    estimate_decoding_params = 1 : bool # whether to estimate the decoding parameters
    """


@schema
class ClusterlessDecodingV1(SpyglassMixin, dj.Computed):
    definition = """
    -> ClusterlessDecodingSelection
    ---
    results_path: filepath@analysis # path to the results file
    """

    def make(self, key):
        orig_key = copy.deepcopy(key)

        # Get model parameters
        model_params = (
            DecodingParameters
            & {"decoding_param_name": key["decoding_param_name"]}
        ).fetch1()
        decoding_params, decoding_kwargs = (
            model_params["decoding_params"],
            model_params["decoding_kwargs"],
        )

        # Get position data
        position_group_key = {"position_group_name": key["position_group_name"]}
        position_variable_names = (PositionGroup & position_group_key).fetch1(
            "position_variables"
        )

        position_info = []
        for pos_merge_id in (PositionGroup.Position & position_group_key).fetch(
            "pos_merge_id"
        ):
            position_info.append(
                IntervalPositionInfo._data_to_df(
                    PositionOutput.fetch_nwb({"merge_id": pos_merge_id})[0],
                    prefix="",
                    add_frame_ind=True,
                )
            )
        position_info = pd.concat(position_info, axis=0).dropna()

        # Get the waveform features for the selected units
        waveform_keys = (
            (
                UnitWaveformFeaturesGroup.UnitFeatures
                & {
                    "waveform_features_group_name": key[
                        "waveform_features_group_name"
                    ]
                }
            )
        ).fetch("KEY")

        spike_times, spike_waveform_features = (
            UnitWaveformFeatures & waveform_keys
        ).fetch_data()

        # Get the encoding and decoding intervals
        encoding_interval = (
            IntervalList
            & {
                "nwb_file_name": key["nwb_file_name"],
                "interval_list_name": key["encoding_interval"],
            }
        ).fetch1("valid_times")
        is_training = np.zeros(len(position_info), dtype=bool)
        for interval_start, interval_end in encoding_interval:
            is_training[
                np.logical_and(
                    position_info.index >= interval_start,
                    position_info.index <= interval_end,
                )
            ] = True
        if "is_training" not in decoding_kwargs:
            decoding_kwargs["is_training"] = is_training

        decoding_interval = (
            IntervalList
            & {
                "nwb_file_name": key["nwb_file_name"],
                "interval_list_name": key["decoding_interval"],
            }
        ).fetch1("valid_times")

        # Decode
        classifier = ClusterlessDetector(**decoding_params)

        if key["estimate_decoding_params"]:
            # if estimating parameters, then we need to treat times outside decoding interval as missing
            # this means that times outside the decoding interval will not use the spiking data
            # a better approach would be to treat the intervals as multiple sequences
            # (see https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm#Multiple_sequences)
            is_missing = np.ones(len(position_info), dtype=bool)
            for interval_start, interval_end in decoding_interval:
                is_missing[
                    np.logical_and(
                        position_info.index >= interval_start,
                        position_info.index <= interval_end,
                    )
                ] = False
            if "is_missing" not in decoding_kwargs:
                decoding_kwargs["is_missing"] = is_missing
            results = classifier.estimate_parameters(
                position_time=position_info.index.to_numpy(),
                position=position_info[position_variable_names].to_numpy(),
                spike_times=spike_times,
                spike_waveform_features=spike_waveform_features,
                time=position_info.index.to_numpy(),
                **decoding_kwargs,
            )
        else:
            VALID_FIT_KWARGS = [
                "is_training",
                "encoding_group_labels",
                "environment_labels",
                "discrete_transition_covariate_data",
            ]

            fit_kwargs = {
                key: value
                for key, value in decoding_kwargs.items()
                if key in VALID_FIT_KWARGS
            }
            classifier.fit(
                position_time=position_info.index.to_numpy(),
                position=position_info[position_variable_names].to_numpy(),
                spike_times=spike_times,
                spike_waveform_features=spike_waveform_features,
                **fit_kwargs,
            )
            VALID_PREDICT_KWARGS = [
                "is_missing",
                "discrete_transition_covariate_data",
                "return_causal_posterior",
            ]
            predict_kwargs = {
                key: value
                for key, value in decoding_kwargs.items()
                if key in VALID_PREDICT_KWARGS
            }

            # We treat each decoding interval as a separate sequence
            results = []
            for interval_start, interval_end in decoding_interval:
                interval_time = position_info.loc[
                    interval_start:interval_end
                ].index.to_numpy()

                if interval_time.size == 0:
                    logger.warning(
                        f"Interval {interval_start}:{interval_end} is empty"
                    )
                    continue
                results.append(
                    classifier.predict(
                        position_time=interval_time,
                        position=position_info.loc[interval_start:interval_end][
                            position_variable_names
                        ].to_numpy(),
                        spike_times=spike_times,
                        spike_waveform_features=spike_waveform_features,
                        time=interval_time,
                        **predict_kwargs,
                    )
                )
            results = xr.concat(results, dim="intervals")

        # Save discrete transition and initial conditions
        results["initial_conditions"] = xr.DataArray(
            classifier.initial_conditions_,
            name="initial_conditions",
        )
        results["discrete_state_transitions"] = xr.DataArray(
            classifier.discrete_state_transitions_,
            dims=("states", "states"),
            name="discrete_state_transitions",
        )
        if (
            vars(classifier).get("discrete_transition_coefficients_")
            is not None
        ):
            results[
                "discrete_transition_coefficients"
            ] = classifier.discrete_transition_coefficients_

        # Insert results
        # in future use https://github.com/rly/ndx-xarray and analysis nwb file?

        nwb_file_name = key["nwb_file_name"].strip("_.nwb")

        # Generate a unique path for the results file
        path_exists = True
        while path_exists:
            results_path = (
                Path(config["SPYGLASS_ANALYSIS_DIR"])
                / nwb_file_name
                / f"{nwb_file_name}_{str(uuid.uuid4())}.nc"
            )
            path_exists = results_path.exists()
        classifier.save_results(
            results,
            results_path,
        )
        key["results_path"] = results_path

        # save environment?

        self.insert1(key)

        from spyglass.decoding.decoding_merge import DecodingOutput

        DecodingOutput.insert1(orig_key, skip_duplicates=True)

    def load_results(self):
        return ClusterlessDetector.load_results(self.fetch1("results_path"))

    @staticmethod
    def load_environments(key):
        model_params = (
            DecodingParameters
            & {"decoding_param_name": key["decoding_param_name"]}
        ).fetch1()
        decoding_params, decoding_kwargs = (
            model_params["decoding_params"],
            model_params["decoding_kwargs"],
        )

        (
            position_info,
            position_variable_names,
        ) = ClusterlessDecodingV1.load_position_info(key)
        classifier = ClusterlessDetector(**decoding_params)

        classifier.initialize_environments(
            position=position_info[position_variable_names].to_numpy(),
            environment_labels=decoding_kwargs.get("environment_labels", None),
        )

        return classifier.environments

    @staticmethod
    def load_position_info(key):
        position_group_key = {"position_group_name": key["position_group_name"]}
        position_variable_names = (PositionGroup & position_group_key).fetch1(
            "position_variables"
        )

        position_info = []
        for pos_merge_id in (PositionGroup.Position & position_group_key).fetch(
            "pos_merge_id"
        ):
            position_info.append(
                IntervalPositionInfo._data_to_df(
                    PositionOutput.fetch_nwb({"merge_id": pos_merge_id})[0],
                    prefix="",
                    add_frame_ind=True,
                )
            )
        position_info = pd.concat(position_info, axis=0).dropna()

        return position_info, position_variable_names

    @staticmethod
    def _get_interval_range(key):
        encoding_interval = (
            IntervalList
            & {
                "nwb_file_name": key["nwb_file_name"],
                "interval_list_name": key["encoding_interval"],
            }
        ).fetch1("valid_times")

        decoding_interval = (
            IntervalList
            & {
                "nwb_file_name": key["nwb_file_name"],
                "interval_list_name": key["decoding_interval"],
            }
        ).fetch1("valid_times")

        return (
            min(
                np.asarray(encoding_interval).min(),
                np.asarray(decoding_interval).min(),
            ),
            max(
                np.asarray(encoding_interval).max(),
                np.asarray(decoding_interval).max(),
            ),
        )

    @staticmethod
    def load_spike_data(key):
        waveform_keys = (
            (
                UnitWaveformFeaturesGroup.UnitFeatures
                & {
                    "waveform_features_group_name": key[
                        "waveform_features_group_name"
                    ]
                }
            )
        ).fetch("KEY")
        spike_times, spike_waveform_features = (
            UnitWaveformFeatures & waveform_keys
        ).fetch_data()

        min_time, max_time = ClusterlessDecodingV1._get_interval_range(key)

        new_spike_times = []
        new_waveform_features = []
        for elec_spike_times, elec_waveform_features in zip(
            spike_times, spike_waveform_features
        ):
            is_in_interval = np.logical_and(
                elec_spike_times >= min_time, elec_spike_times <= max_time
            )
            new_spike_times.append(elec_spike_times[is_in_interval])
            new_waveform_features.append(elec_waveform_features[is_in_interval])

        return new_spike_times, new_waveform_features
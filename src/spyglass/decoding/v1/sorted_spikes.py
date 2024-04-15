"""Pipeline for decoding the animal's mental position and some category of interest
from clustered spikes times. See [1] for details.

References
----------
[1] Denovellis, E. L. et al. Hippocampal replay of experience at real-world
speeds. eLife 10, e64505 (2021).

"""

import copy
import uuid
from pathlib import Path

import datajoint as dj
import non_local_detector.analysis as analysis
import numpy as np
import pandas as pd
import xarray as xr
from non_local_detector.models.base import SortedSpikesDetector
from track_linearization import get_linearized_position

from spyglass.common.common_interval import IntervalList  # noqa: F401
from spyglass.common.common_session import Session  # noqa: F401
from spyglass.decoding.v1.core import (
    DecodingParameters,
    PositionGroup,
)  # noqa: F401
from spyglass.position.position_merge import PositionOutput  # noqa: F401
from spyglass.settings import config
from spyglass.spikesorting.analysis.v1.group import SortedSpikesGroup
from spyglass.spikesorting.spikesorting_merge import (
    SpikeSortingOutput,
)  # noqa: F401
from spyglass.utils import SpyglassMixin, logger

schema = dj.schema("decoding_sorted_spikes_v1")


@schema
class SortedSpikesDecodingSelection(SpyglassMixin, dj.Manual):
    definition = """
    -> SortedSpikesGroup
    -> PositionGroup
    -> DecodingParameters
    -> IntervalList.proj(encoding_interval='interval_list_name')
    -> IntervalList.proj(decoding_interval='interval_list_name')
    estimate_decoding_params = 1 : bool # whether to estimate the decoding parameters
    """
    # NOTE: Excessive key length fixed by reducing UnitSelectionParams.unit_filter_params_name


@schema
class SortedSpikesDecodingV1(SpyglassMixin, dj.Computed):
    definition = """
    -> SortedSpikesDecodingSelection
    ---
    results_path: filepath@analysis # path to the results file
    classifier_path: filepath@analysis # path to the classifier file
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
        decoding_kwargs = decoding_kwargs or {}

        # Get position data
        (
            position_info,
            position_variable_names,
        ) = self.fetch_position_info(key)

        # Get the spike times for the selected units
        # Don't need to filter by interval since the non_local_detector code will do that
        spike_times = self.fetch_spike_data(key, filter_by_interval=False)

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
        classifier = SortedSpikesDetector(**decoding_params)

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
            results["discrete_transition_coefficients"] = (
                classifier.discrete_transition_coefficients_
            )

        # Insert results
        # in future use https://github.com/rly/ndx-xarray and analysis nwb file?

        nwb_file_name = key["nwb_file_name"].replace("_.nwb", "")

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

        classifier_path = results_path.with_suffix(".pkl")
        classifier.save_model(classifier_path)
        key["classifier_path"] = classifier_path

        self.insert1(key)

        from spyglass.decoding.decoding_merge import DecodingOutput

        DecodingOutput.insert1(orig_key, skip_duplicates=True)

    def fetch_results(self):
        return SortedSpikesDetector.load_results(self.fetch1("results_path"))

    def fetch_model(self):
        return SortedSpikesDetector.load_model(self.fetch1("classifier_path"))

    @staticmethod
    def fetch_environments(key):
        model_params = (
            DecodingParameters
            & {"decoding_param_name": key["decoding_param_name"]}
        ).fetch1()
        decoding_params, decoding_kwargs = (
            model_params["decoding_params"],
            model_params["decoding_kwargs"],
        )

        if decoding_kwargs is None:
            decoding_kwargs = {}

        (
            position_info,
            position_variable_names,
        ) = SortedSpikesDecodingV1.fetch_position_info(key)
        classifier = SortedSpikesDetector(**decoding_params)

        classifier.initialize_environments(
            position=position_info[position_variable_names].to_numpy(),
            environment_labels=decoding_kwargs.get("environment_labels", None),
        )

        return classifier.environments

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
    def fetch_position_info(key):
        position_group_key = {
            "position_group_name": key["position_group_name"],
            "nwb_file_name": key["nwb_file_name"],
        }
        position_variable_names = (PositionGroup & position_group_key).fetch1(
            "position_variables"
        )

        position_info = []
        for pos_merge_id in (PositionGroup.Position & position_group_key).fetch(
            "pos_merge_id"
        ):
            position_info.append(
                (PositionOutput & {"merge_id": pos_merge_id}).fetch1_dataframe()
            )
        min_time, max_time = SortedSpikesDecodingV1._get_interval_range(key)
        position_info = (
            pd.concat(position_info, axis=0).loc[min_time:max_time].dropna()
        )

        return position_info, position_variable_names

    @staticmethod
    def fetch_linear_position_info(key):
        environment = SortedSpikesDecodingV1.fetch_environments(key)[0]

        position_df = SortedSpikesDecodingV1.fetch_position_info(key)[0]
        position_variable_names = (PositionGroup & key).fetch1(
            "position_variables"
        )
        position = np.asarray(position_df[position_variable_names])

        linear_position_df = get_linearized_position(
            position=position,
            track_graph=environment.track_graph,
            edge_order=environment.edge_order,
            edge_spacing=environment.edge_spacing,
        )
        min_time, max_time = SortedSpikesDecodingV1._get_interval_range(key)
        return (
            pd.concat(
                [linear_position_df.set_index(position_df.index), position_df],
                axis=1,
            )
            .loc[min_time:max_time]
            .dropna()
        )

    @staticmethod
    def fetch_spike_data(key, filter_by_interval=True, time_slice=None):
        spike_times = SortedSpikesGroup.fetch_spike_data(key)
        if not filter_by_interval:
            return spike_times

        if time_slice is None:
            min_time, max_time = SortedSpikesDecodingV1._get_interval_range(key)
        else:
            min_time, max_time = time_slice.start, time_slice.stop

        new_spike_times = []
        for elec_spike_times in spike_times:
            is_in_interval = np.logical_and(
                elec_spike_times >= min_time, elec_spike_times <= max_time
            )
            new_spike_times.append(elec_spike_times[is_in_interval])

        return new_spike_times

    def spike_times_sorted_by_place_field_peak(self, time_slice=None):
        if time_slice is None:
            time_slice = slice(-np.inf, np.inf)

        spike_times = self.fetch_spike_data(self.fetch1())
        classifier = self.fetch_model()

        new_spike_times = {}

        for encoding_model in classifier.encoding_model_:
            place_fields = np.asarray(
                classifier.encoding_model_[encoding_model]["place_fields"]
            )
            neuron_sort_ind = np.argsort(
                np.nanargmax(place_fields, axis=1).squeeze()
            )
            new_spike_times[encoding_model] = [
                spike_times[neuron_ind][
                    np.logical_and(
                        spike_times[neuron_ind] >= time_slice.start,
                        spike_times[neuron_ind] <= time_slice.stop,
                    )
                ]
                for neuron_ind in neuron_sort_ind
            ]

    def get_ahead_behind_distance(self, track_graph=None, time_slice=None):
        # TODO: store in table

        if time_slice is None:
            time_slice = slice(-np.inf, np.inf)

        classifier = self.fetch_model()
        posterior = (
            self.fetch_results()
            .acausal_posterior.sel(time=time_slice)
            .squeeze()
            .unstack("state_bins")
            .sum("state")
        )

        if track_graph is None:
            track_graph = classifier.environments[0].track_graph

        if track_graph is not None:
            linear_position_info = self.fetch_linear_position_info(
                self.fetch1("KEY")
            ).loc[time_slice]

            orientation_name = (
                "orientation"
                if "orientation" in linear_position_info.columns
                else "head_orientation"
            )

            traj_data = analysis.get_trajectory_data(
                posterior=posterior,
                track_graph=track_graph,
                decoder=classifier,
                actual_projected_position=linear_position_info[
                    ["projected_x_position", "projected_y_position"]
                ],
                track_segment_id=linear_position_info["track_segment_id"],
                actual_orientation=linear_position_info[orientation_name],
            )

            return analysis.get_ahead_behind_distance(track_graph, *traj_data)
        else:
            position_info = self.fetch_position_info(self.fetch1("KEY")).loc[
                time_slice
            ]
            map_position = analysis.maximum_a_posteriori_estimate(posterior)

            orientation_name = (
                "orientation"
                if "orientation" in position_info.columns
                else "head_orientation"
            )
            position_variable_names = (
                PositionGroup & self.fetch1("KEY")
            ).fetch1("position_variables")

            return analysis.get_ahead_behind_distance2D(
                position_info[position_variable_names].to_numpy(),
                position_info[orientation_name].to_numpy(),
                map_position,
                classifier.environments[0].track_graphDD,
            )

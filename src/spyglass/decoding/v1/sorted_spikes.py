"""Pipeline for decoding the animal's mental position and some category of
interest from clustered spikes times. See [1] for details.

References
----------
[1] Denovellis, E. L. et al. Hippocampal replay of experience at real-world
speeds. eLife 10, e64505 (2021).

"""

import copy
import uuid
from pathlib import Path
from typing import Optional, Union

import datajoint as dj
import non_local_detector.analysis as analysis
import numpy as np
import pandas as pd
import xarray as xr
from non_local_detector.models.base import SortedSpikesDetector
from track_linearization import get_linearized_position

from spyglass.common.common_interval import IntervalList  # noqa: F401
from spyglass.common.common_session import Session  # noqa: F401
from spyglass.decoding.v1.core import DecodingParameters  # noqa: F401
from spyglass.decoding.v1.core import PositionGroup
from spyglass.decoding.v1.utils import _get_interval_range
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
    estimate_decoding_params = 1 : bool # 1 to estimate the decoding parameters
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
        """Populate the decoding model.

        1. Fetches parameters and position data from DecodingParameters and
            PositionGroup tables.
        2. Decomposes instervals into encoding and decoding.
        3. Optionally estimates decoding parameters, otherwise uses the provided
            parameters.
        4. Uses SortedSpikesDetector from non_local_detector package to decode
            the animal's mental position, including initial and discrete state
            transition information.
        5. Optionally includes the discrete transition coefficients.
        6. Saves the results and model to disk in the analysis directory, under
            the nwb file name's folder.
        7. Inserts the results and model paths into SortedSpikesDecodingV1 and
            DecodingOutput tables.
        """
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

        # Get the spike times for the selected units. Don't need to filter by
        # interval since the non_local_detector code will do that

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
        is_training[
            position_info[position_variable_names].isna().values.max(axis=1)
        ] = False

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
            # if estimating parameters, then we need to treat times outside
            # decoding interval as missing this means that times outside the
            # decoding interval will not use the spiking data a better approach
            # would be to treat the intervals as multiple sequences (see
            # https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm#Multiple_sequences)

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

        # Make sure the results directory exists
        results_dir = Path(config["SPYGLASS_ANALYSIS_DIR"]) / nwb_file_name
        results_dir.mkdir(parents=True, exist_ok=True)

        # Generate a unique path for the results file
        path_exists = True
        while path_exists:
            results_path = (
                results_dir / f"{nwb_file_name}_{str(uuid.uuid4())}.nc"
            )
            # if the results_path already exists, try a different uuid
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

    def fetch_results(self) -> xr.Dataset:
        """Retrieve the decoding results

        Returns
        -------
        xr.Dataset
            The decoding results (posteriors, etc.)
        """
        return SortedSpikesDetector.load_results(self.fetch1("results_path"))

    def fetch_model(self):
        """Retrieve the decoding model"""
        return SortedSpikesDetector.load_model(self.fetch1("classifier_path"))

    @classmethod
    def fetch_environments(cls, key):
        """Fetch the environments for the decoding model

        Parameters
        ----------
        key : dict
            The decoding selection key

        Returns
        -------
        List[TrackGraph]
            list of track graphs in the trained model
        """
        key = cls.get_fully_defined_key(
            key, required_fields=["decoding_param_name"]
        )

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

    @classmethod
    def fetch_position_info(cls, key):
        """Fetch the position information for the decoding model

        Parameters
        ----------
        key : dict
            The decoding selection key

        Returns
        -------
        Tuple[pd.DataFrame, List[str]]
            The position information and the names of the position variables
        """
        key = cls.get_fully_defined_key(
            key,
            required_fields=[
                "position_group_name",
                "nwb_file_name",
                "encoding_interval",
                "decoding_interval",
            ],
        )

        position_group_key = {
            "position_group_name": key["position_group_name"],
            "nwb_file_name": key["nwb_file_name"],
        }
        min_time, max_time = _get_interval_range(key)
        position_info, position_variable_names = (
            PositionGroup & position_group_key
        ).fetch_position_info(min_time=min_time, max_time=max_time)

        return position_info, position_variable_names

    @classmethod
    def fetch_linear_position_info(cls, key):
        """Fetch the position information and project it onto the track graph

        Parameters
        ----------
        key : dict
            The decoding selection key

        Returns
        -------
        pd.DataFrame
            The linearized position information
        """
        key = cls.get_fully_defined_key(
            key,
            required_fields=[
                "position_group_name",
                "nwb_file_name",
                "encoding_interval",
                "decoding_interval",
            ],
        )

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
        min_time, max_time = _get_interval_range(key)

        return pd.concat(
            [linear_position_df.set_index(position_df.index), position_df],
            axis=1,
        ).loc[min_time:max_time]

    @classmethod
    def fetch_spike_data(
        cls,
        key,
        filter_by_interval=True,
        time_slice=None,
        return_unit_ids=False,
    ) -> Union[list[np.ndarray], Optional[list[dict]]]:
        """Fetch the spike times for the decoding model

        Parameters
        ----------
        key : dict
            The decoding selection key
        filter_by_interval : bool, optional
            Whether to filter for spike times in the model interval,
            by default True
        time_slice : Slice, optional
            User provided slice of time to restrict spikes to, by default None
        return_unit_ids : bool, optional
            if True, return the unit_ids along with the spike times, by default
            False Unit ids defined as a list of dictionaries with keys
            'spikesorting_merge_id' and 'unit_number'

        Returns
        -------
        list[np.ndarray]
            List of spike times for each unit in the model's spike group
        """
        key = cls.get_fully_defined_key(
            key,
            required_fields=[
                "encoding_interval",
                "decoding_interval",
            ],
        )

        spike_times, unit_ids = SortedSpikesGroup.fetch_spike_data(
            key, return_unit_ids=True
        )
        if not filter_by_interval:
            return spike_times

        if time_slice is None:
            min_time, max_time = _get_interval_range(key)
        else:
            min_time, max_time = time_slice.start, time_slice.stop

        new_spike_times = []
        for elec_spike_times in spike_times:
            is_in_interval = np.logical_and(
                elec_spike_times >= min_time, elec_spike_times <= max_time
            )
            new_spike_times.append(elec_spike_times[is_in_interval])

        if return_unit_ids:
            return new_spike_times, unit_ids
        return new_spike_times

    def spike_times_sorted_by_place_field_peak(self, time_slice=None):
        """Spike times of units sorted by place field peak location

        Parameters
        ----------
        time_slice : Slice, optional
            time range to limit returned spikes to, by default None
        """
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
        return new_spike_times

    def get_orientation_col(self, df):
        """Examine columns of a input df and return orientation col name"""
        cols = df.columns
        return "orientation" if "orientation" in cols else "head_orientation"

    def get_ahead_behind_distance(self, track_graph=None, time_slice=None):
        """Get relative decoded position from the animal's actual position

        Parameters
        ----------
        track_graph : TrackGraph, optional
            environment track graph to project position on, by default None
        time_slice : Slice, optional
            time intrerval to restrict to, by default None

        Returns
        -------
        distance_metrics : np.ndarray
            Information about the distance of the animal to the mental position.
        """
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

            orientation_name = self.get_orientation_col(linear_position_info)

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

            orientation_name = self.get_orientation_col(position_info)

            position_variable_names = (
                PositionGroup & self.fetch1("KEY")
            ).fetch1("position_variables")

            return analysis.get_ahead_behind_distance2D(
                position_info[position_variable_names].to_numpy(),
                position_info[orientation_name].to_numpy(),
                map_position,
                classifier.environments[0].track_graphDD,
            )

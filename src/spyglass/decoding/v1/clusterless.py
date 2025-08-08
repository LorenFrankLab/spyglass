"""Pipeline for decoding the animal's mental position and some category of
interest from unclustered spikes and spike waveform features. See [1] for
details.

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
from non_local_detector.models.base import ClusterlessDetector
from track_linearization import get_linearized_position

from spyglass.common.common_interval import IntervalList  # noqa: F401
from spyglass.common.common_session import Session  # noqa: F401
from spyglass.decoding.v1.core import DecodingParameters  # noqa: F401
from spyglass.decoding.v1.core import PositionGroup
from spyglass.decoding.v1.utils import _get_interval_range
from spyglass.decoding.v1.waveform_features import (
    UnitWaveformFeatures,
)  # noqa: F401
from spyglass.position.position_merge import PositionOutput  # noqa: F401
from spyglass.settings import config
from spyglass.utils import SpyglassMixin, SpyglassMixinPart, logger
from spyglass.utils.spikesorting import firing_rate_from_spike_indicator

schema = dj.schema("decoding_clusterless_v1")


@schema
class UnitWaveformFeaturesGroup(SpyglassMixin, dj.Manual):
    definition = """
    -> Session
    waveform_features_group_name: varchar(80)
    """

    class UnitFeatures(SpyglassMixinPart):
        definition = """
        -> UnitWaveformFeaturesGroup
        -> UnitWaveformFeatures
        """

    def create_group(
        self, nwb_file_name: str, group_name: str, keys: list[dict]
    ):
        """Create a group of waveform features for a given session"""
        group_key = {
            "nwb_file_name": nwb_file_name,
            "waveform_features_group_name": group_name,
        }
        if self & group_key:
            logger.error(  # No error on duplicate helps with pytests
                f"Group {nwb_file_name}: {group_name} already exists"
                + "please delete the group before creating a new one",
            )
            return
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
    estimate_decoding_params = 1 : bool # 1 to estimate the decoding parameters
    """


@schema
class ClusterlessDecodingV1(SpyglassMixin, dj.Computed):
    definition = """
    -> ClusterlessDecodingSelection
    ---
    results_path: filepath@analysis # path to the results file
    classifier_path: filepath@analysis # path to the classifier file
    """

    def make(self, key):
        """Populate the ClusterlessDecoding table.

        1. Fetches...
            position data from PositionGroup table
            waveform features and spike times from UnitWaveformFeatures table
            decoding parameters from DecodingParameters table
            encoding/decoding intervals from IntervalList table
        2. Decodes via ClusterlessDetector from non_local_detector package
        3. Optionally estimates decoding parameters
        4. Saves the decoding results (initial conditions, discrete state
            transitions) and classifier to disk. May include discrete transition
            coefficients if available.
        5. Inserts into ClusterlessDecodingV1 table and DecodingOutput merge
            table.
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

        # Get the waveform features for the selected units. Don't need to filter
        # by interval since the non_local_detector code will do that

        (
            spike_times,
            spike_waveform_features,
        ) = self.fetch_spike_data(key, filter_by_interval=False)

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
        classifier = ClusterlessDetector(**decoding_params)

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
        return ClusterlessDetector.load_results(self.fetch1("results_path"))

    def fetch_model(self):
        """Retrieve the decoding model"""
        return ClusterlessDetector.load_model(self.fetch1("classifier_path"))

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
        ) = ClusterlessDecodingV1.fetch_position_info(key)
        classifier = ClusterlessDetector(**decoding_params)

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
                "nwb_file_name",
                "position_group_name",
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
                "nwb_file_name",
                "position_group_name",
                "encoding_interval",
                "decoding_interval",
            ],
        )

        environment = ClusterlessDecodingV1.fetch_environments(key)[0]

        position_df = ClusterlessDecodingV1.fetch_position_info(key)[0]
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
    def fetch_spike_data(cls, key, filter_by_interval=True):
        """Fetch the spike times for the decoding model

        Parameters
        ----------
        key : dict
            The decoding selection key
        filter_by_interval : bool, optional
            Whether to filter for spike times in the model interval.
            Default True

        Returns
        -------
        list[np.ndarray]
            List of spike times for each unit in the model's spike group
        """
        key = cls.get_fully_defined_key(
            key,
            required_fields=[
                "nwb_file_name",
                "waveform_features_group_name",
            ],
        )

        waveform_keys = (
            (
                UnitWaveformFeaturesGroup.UnitFeatures
                & {
                    "nwb_file_name": key["nwb_file_name"],
                    "waveform_features_group_name": key[
                        "waveform_features_group_name"
                    ],
                }
            )
        ).fetch("KEY")
        spike_times, spike_waveform_features = (
            UnitWaveformFeatures & waveform_keys
        ).fetch_data()

        if not filter_by_interval:
            return spike_times, spike_waveform_features

        min_time, max_time = _get_interval_range(key)

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

    @classmethod
    def get_spike_indicator(cls, key, time):
        """get spike indicator matrix for the group

        Parameters
        ----------
        key : dict
            key to identify the group
        time : np.ndarray
            time vector for which to calculate the spike indicator matrix

        Returns
        -------
        np.ndarray
            spike indicator matrix with shape (len(time), n_units)
        """
        time = np.asarray(time)
        min_time, max_time = time[[0, -1]]
        spike_times = cls.fetch_spike_data(key)[0]
        spike_indicator = np.zeros((len(time), len(spike_times)))

        for ind, times in enumerate(spike_times):
            times = times[np.logical_and(times >= min_time, times <= max_time)]
            spike_indicator[:, ind] = np.bincount(
                np.digitize(times, time[1:-1]),
                minlength=time.shape[0],
            )

        return spike_indicator

    @classmethod
    def get_firing_rate(
        cls,
        key: dict,
        time: np.ndarray,
        multiunit: bool = False,
        smoothing_sigma: float = 0.015,
    ) -> np.ndarray:
        """Get time-dependent firing rate for units in the group


        Parameters
        ----------
        key : dict
            key to identify the group
        time : np.ndarray
            time vector for which to calculate the firing rate
        multiunit : bool, optional
            if True, return the multiunit firing rate for units in the group.
            Default False
        smoothing_sigma : float, optional
            standard deviation of gaussian filter to smooth firing rates in
            seconds. Default 0.015

        Returns
        -------
        np.ndarray
            time-dependent firing rate with shape (len(time), n_units)
        """
        return firing_rate_from_spike_indicator(
            spike_indicator=cls.get_spike_indicator(key, time),
            time=time,
            multiunit=multiunit,
            smoothing_sigma=smoothing_sigma,
        )

    def get_orientation_col(self, df):
        """Examine columns of a input df and return orientation col name"""
        cols = df.columns
        return "orientation" if "orientation" in cols else "head_orientation"

    def get_ahead_behind_distance(self, track_graph=None, time_slice=None):
        """get the ahead-behind distance for the decoding model

        Returns
        -------
        distance_metrics : np.ndarray
            Information about the distance of the animal to the mental position.
        """
        # TODO: allow specification of specific time interval
        # TODO: allow specification of track graph
        # TODO: Handle decode intervals, store in table

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

            return analysis.get_ahead_behind_distance(
                classifier.environments[0].track_graph, *traj_data
            )
        else:
            # `fetch_position_info` returns a tuple
            position_info = self.fetch_position_info(self.fetch1("KEY"))[0].loc[
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

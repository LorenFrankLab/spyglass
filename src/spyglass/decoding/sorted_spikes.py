"""Pipeline for decoding the animal's mental position and some category of interest
from clustered spikes times. See [1] for details.

References
----------
[1] Denovellis, E. L. et al. Hippocampal replay of experience at real-world
speeds. eLife 10, e64505 (2021).

"""

import pprint

import datajoint as dj
import numpy as np
import pandas as pd
from replay_trajectory_classification.classifier import (
    _DEFAULT_CONTINUOUS_TRANSITIONS,
    _DEFAULT_ENVIRONMENT,
    _DEFAULT_SORTED_SPIKES_MODEL_KWARGS,
)
from replay_trajectory_classification.discrete_state_transitions import DiagonalDiscrete
from replay_trajectory_classification.initial_conditions import UniformInitialConditions
from spyglass.common.common_interval import IntervalList
from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.common.common_position import IntervalPositionInfo
from spyglass.utils.dj_helper_fn import fetch_nwb
from spyglass.decoding.core import (
    convert_epoch_interval_name_to_position_interval_name,
    convert_valid_times_to_slice,
    get_valid_ephys_position_times_by_epoch,
)
from spyglass.decoding.dj_decoder_conversion import (
    convert_classes_to_dict,
    restore_classes,
)
from spyglass.spikesorting.spikesorting_curation import CuratedSpikeSorting

schema = dj.schema("decoding_sortedspikes")


@schema
class SortedSpikesIndicatorSelection(dj.Lookup):
    """Bins spike times into regular intervals given by the sampling rate.
    Start and stop time of the interval are defined by the interval list.
    """

    definition = """
    -> CuratedSpikeSorting
    -> IntervalList
    sampling_rate=500 : float
    ---
    """


@schema
class SortedSpikesIndicator(dj.Computed):
    """Bins spike times into regular intervals given by the sampling rate.
    Useful for GLMs and for decoding.

    """

    definition = """
    -> SortedSpikesIndicatorSelection
    ---
    -> AnalysisNwbfile
    spike_indicator_object_id: varchar(40)
    """

    def make(self, key):
        pprint.pprint(key)
        # TODO: intersection of sort interval and interval list
        interval_times = (IntervalList & key).fetch1("valid_times")

        sampling_rate = (SortedSpikesIndicatorSelection & key).fetch("sampling_rate")

        time = self.get_time_bins_from_interval(interval_times, sampling_rate)

        spikes_nwb = (CuratedSpikeSorting & key).fetch_nwb()
        # restrict to cases with units
        spikes_nwb = [entry for entry in spikes_nwb if "units" in entry]
        spike_times_list = [
            np.asarray(n_trode["units"]["spike_times"]) for n_trode in spikes_nwb
        ]
        if len(spike_times_list) > 0:  # if units
            spikes = np.concatenate(spike_times_list)

            # Bin spikes into time bins
            spike_indicator = []
            for spike_times in spikes:
                spike_times = spike_times[
                    (spike_times > time[0]) & (spike_times <= time[-1])
                ]
                spike_indicator.append(
                    np.bincount(
                        np.digitize(spike_times, time[1:-1]), minlength=time.shape[0]
                    )
                )

            column_names = np.concatenate(
                [
                    [
                        f'{n_trode["sort_group_id"]:04d}_{unit_number:04d}'
                        for unit_number in n_trode["units"].index
                    ]
                    for n_trode in spikes_nwb
                ]
            )
            spike_indicator = pd.DataFrame(
                np.stack(spike_indicator, axis=1),
                index=pd.Index(time, name="time"),
                columns=column_names,
            )

            # Insert into analysis nwb file
            nwb_analysis_file = AnalysisNwbfile()
            key["analysis_file_name"] = nwb_analysis_file.create(key["nwb_file_name"])

            key["spike_indicator_object_id"] = nwb_analysis_file.add_nwb_object(
                analysis_file_name=key["analysis_file_name"],
                nwb_object=spike_indicator.reset_index(),
            )

            nwb_analysis_file.add(
                nwb_file_name=key["nwb_file_name"],
                analysis_file_name=key["analysis_file_name"],
            )

            self.insert1(key)

    @staticmethod
    def get_time_bins_from_interval(interval_times, sampling_rate):
        """Gets the superset of the interval."""
        start_time, end_time = interval_times[0][0], interval_times[-1][-1]
        n_samples = int(np.ceil((end_time - start_time) * sampling_rate)) + 1

        return np.linspace(start_time, end_time, n_samples)

    def fetch_nwb(self, *attrs, **kwargs):
        return fetch_nwb(
            self, (AnalysisNwbfile, "analysis_file_abs_path"), *attrs, **kwargs
        )

    def fetch1_dataframe(self):
        return self.fetch_dataframe()[0]

    def fetch_dataframe(self):
        return pd.concat(
            [data["spike_indicator"].set_index("time") for data in self.fetch_nwb()],
            axis=1,
        )


def make_default_decoding_parameters_cpu():
    classifier_parameters = dict(
        environments=[_DEFAULT_ENVIRONMENT],
        observation_models=None,
        continuous_transition_types=_DEFAULT_CONTINUOUS_TRANSITIONS,
        discrete_transition_type=DiagonalDiscrete(0.98),
        initial_conditions_type=UniformInitialConditions(),
        infer_track_interior=True,
        knot_spacing=10,
        spike_model_penalty=1e1,
    )

    predict_parameters = {
        "is_compute_acausal": True,
        "use_gpu": False,
        "state_names": ["Continuous", "Fragmented"],
    }
    fit_parameters = dict()

    return classifier_parameters, fit_parameters, predict_parameters


def make_default_decoding_parameters_gpu():
    classifier_parameters = dict(
        environments=[_DEFAULT_ENVIRONMENT],
        observation_models=None,
        continuous_transition_types=_DEFAULT_CONTINUOUS_TRANSITIONS,
        discrete_transition_type=DiagonalDiscrete(0.98),
        initial_conditions_type=UniformInitialConditions(),
        infer_track_interior=True,
        sorted_spikes_algorithm="spiking_likelihood_kde",
        sorted_spikes_algorithm_params=_DEFAULT_SORTED_SPIKES_MODEL_KWARGS,
    )

    predict_parameters = {
        "is_compute_acausal": True,
        "use_gpu": True,
        "state_names": ["Continuous", "Fragmented"],
    }

    fit_parameters = dict()

    return classifier_parameters, fit_parameters, predict_parameters


@schema
class SortedSpikesClassifierParameters(dj.Manual):
    """Stores parameters for decoding with sorted spikes"""

    definition = """
    classifier_param_name : varchar(80) # a name for this set of parameters
    ---
    classifier_params :   BLOB    # initialization parameters
    fit_params :          BLOB    # fit parameters
    predict_params :      BLOB    # prediction parameters
    """

    def insert_default(self):
        (
            classifier_parameters,
            fit_parameters,
            predict_parameters,
        ) = make_default_decoding_parameters_cpu()
        self.insert1(
            {
                "classifier_param_name": "default_decoding_cpu",
                "classifier_params": classifier_parameters,
                "fit_params": fit_parameters,
                "predict_params": predict_parameters,
            },
            skip_duplicates=True,
        )

        (
            classifier_parameters,
            fit_parameters,
            predict_parameters,
        ) = make_default_decoding_parameters_gpu()
        self.insert1(
            {
                "classifier_param_name": "default_decoding_gpu",
                "classifier_params": classifier_parameters,
                "fit_params": fit_parameters,
                "predict_params": predict_parameters,
            },
            skip_duplicates=True,
        )

    def insert1(self, key, **kwargs):
        super().insert1(convert_classes_to_dict(key), **kwargs)

    def fetch1(self, *args, **kwargs):
        return restore_classes(super().fetch1(*args, **kwargs))


def get_decoding_data_for_epoch(
    nwb_file_name: str,
    interval_list_name: str,
    position_info_param_name="default_decoding",
    additional_spike_keys={},
):
    valid_ephys_position_times_by_epoch = get_valid_ephys_position_times_by_epoch(
        nwb_file_name
    )
    valid_ephys_position_times = valid_ephys_position_times_by_epoch[interval_list_name]
    valid_slices = convert_valid_times_to_slice(valid_ephys_position_times)
    position_interval_name = convert_epoch_interval_name_to_position_interval_name(
        interval_list_name
    )

    position_info = (
        IntervalPositionInfo()
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": position_interval_name,
            "position_info_param_name": position_info_param_name,
        }
    ).fetch1_dataframe()

    position_info = pd.concat([position_info.loc[times] for times in valid_slices])

    spikes = (
        SortedSpikesIndicator
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": position_interval_name,
            **additional_spike_keys,
        }
    ).fetch_dataframe()
    spikes = pd.concat([spikes.loc[times] for times in valid_slices])

    return position_info, spikes, valid_slices

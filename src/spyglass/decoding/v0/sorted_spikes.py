"""Pipeline for decoding the animal's mental position and some category of
interest from clustered spikes times. See [1] for details.

References
----------
[1] Denovellis, E. L. et al. Hippocampal replay of experience at real-world
speeds. eLife 10, e64505 (2021).

"""

import datajoint as dj
import numpy as np
import pandas as pd

from spyglass.utils import logger

try:
    from replay_trajectory_classification.classifier import (
        _DEFAULT_CONTINUOUS_TRANSITIONS,
        _DEFAULT_ENVIRONMENT,
        _DEFAULT_SORTED_SPIKES_MODEL_KWARGS,
    )
    from replay_trajectory_classification.discrete_state_transitions import (
        DiagonalDiscrete,
    )
    from replay_trajectory_classification.initial_conditions import (
        UniformInitialConditions,
    )
except (ImportError, ModuleNotFoundError) as e:
    (
        _DEFAULT_CONTINUOUS_TRANSITIONS,
        _DEFAULT_ENVIRONMENT,
        _DEFAULT_SORTED_SPIKES_MODEL_KWARGS,
        DiagonalDiscrete,
        UniformInitialConditions,
    ) = [None] * 5
    logger.warning(e)

from spyglass.common.common_behav import (
    convert_epoch_interval_name_to_position_interval_name,
)
from spyglass.common.common_interval import IntervalList
from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.common.common_position import IntervalPositionInfo
from spyglass.decoding.v0.core import (
    convert_valid_times_to_slice,
    get_valid_ephys_position_times_by_epoch,
)
from spyglass.decoding.v0.dj_decoder_conversion import (
    convert_classes_to_dict,
    restore_classes,
)
from spyglass.decoding.v0.utils import (
    get_time_bins_from_interval,
    make_default_decoding_params,
)
from spyglass.spikesorting.v0.spikesorting_curation import CuratedSpikeSorting
from spyglass.utils import SpyglassMixin, logger

schema = dj.schema("decoding_sortedspikes")


@schema
class SortedSpikesIndicatorSelection(SpyglassMixin, dj.Lookup):
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
class SortedSpikesIndicator(SpyglassMixin, dj.Computed):
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
        """Populate the SortedSpikesIndicator table.

        Fetches the spike times from the CuratedSpikeSorting table and bins
        them into regular intervals given by the sampling rate. The spike
        indicator is stored in an AnalysisNwbfile.
        """
        logger.info(key)
        # TODO: intersection of sort interval and interval list
        interval_times = (IntervalList & key).fetch1("valid_times")

        sampling_rate = (SortedSpikesIndicatorSelection & key).fetch(
            "sampling_rate"
        )

        time = get_time_bins_from_interval(interval_times, sampling_rate)

        spikes_nwb = (CuratedSpikeSorting & key).fetch_nwb()
        # restrict to cases with units
        spikes_nwb = [entry for entry in spikes_nwb if "units" in entry]
        spike_times_list = [
            np.asarray(n_trode["units"]["spike_times"])
            for n_trode in spikes_nwb
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
                        np.digitize(spike_times, time[1:-1]),
                        minlength=time.shape[0],
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
            key["analysis_file_name"] = nwb_analysis_file.create(
                key["nwb_file_name"]
            )

            key["spike_indicator_object_id"] = nwb_analysis_file.add_nwb_object(
                analysis_file_name=key["analysis_file_name"],
                nwb_object=spike_indicator.reset_index(),
            )

            nwb_analysis_file.add(
                nwb_file_name=key["nwb_file_name"],
                analysis_file_name=key["analysis_file_name"],
            )

            self.insert1(key)

    def fetch1_dataframe(self) -> pd.DataFrame:
        """Return the first spike indicator as a dataframe."""
        _ = self.ensure_single_entry()
        return self.fetch_dataframe()[0]

    def fetch_dataframe(self) -> list[pd.DataFrame]:
        """Return all spike indicators as a list of dataframes."""
        return pd.concat(
            [
                data["spike_indicator"].set_index("time")
                for data in self.fetch_nwb()
            ],
            axis=1,
        )


@schema
class SortedSpikesClassifierParameters(SpyglassMixin, dj.Manual):
    """Stores parameters for decoding with sorted spikes

    Attributes
    ----------
    classifier_param_name: str
        A name for this set of parameters
    classifier_params: dict
        Initialization parameters, including ...
            environments: list
            observation_models
            continuous_transition_types
            discrete_transition_type: DiagonalDiscrete
            initial_conditions_type: UniformInitialConditions
            infer_track_interior: bool
            clusterless_algorithm: str, optional
            clusterless_algorithm_params: dict, optional
            sorted_spikes_algorithm: str, optional
            sorted_spikes_algorithm_params: dict, optional
        For more information, see replay_trajectory_classification documentation
    fit_params: dict, optional
    predict_params: dict, optional
        Prediction parameters, including ...
            is_compute_acausal: bool
            use_gpu: bool
            state_names: List[str]
    """

    definition = """
    classifier_param_name : varchar(80) # a name for this set of parameters
    ---
    classifier_params :   BLOB    # initialization parameters
    fit_params :          BLOB    # fit parameters
    predict_params :      BLOB    # prediction parameters
    """

    def insert_default(self):
        """Insert default parameters for decoding with sorted spikes"""
        self.insert(
            [
                make_default_decoding_params(),
                make_default_decoding_params(use_gpu=True),
            ],
            skip_duplicates=True,
        )

    def insert1(self, key, **kwargs):
        """Override insert1 to convert classes to dict"""
        super().insert1(convert_classes_to_dict(key), **kwargs)

    def fetch1(self, *args, **kwargs):
        """Override fetch1 to restore classes"""
        return restore_classes(super().fetch1(*args, **kwargs))


def get_spike_indicator(
    key: dict, time_range: tuple[float, float], sampling_rate: float = 500.0
) -> pd.DataFrame:
    """Returns a dataframe with the spike indicator for each unit

    Parameters
    ----------
    key : dict
    time_range : tuple[float, float]
        Start and end time of the spike indicator
    sampling_rate : float, optional

    Returns
    -------
    spike_indicator : pd.DataFrame, shape (n_time, n_units)
        A dataframe with the spike indicator for each unit
    """
    start_time, end_time = time_range
    n_samples = int(np.ceil((end_time - start_time) * sampling_rate)) + 1
    time = np.linspace(start_time, end_time, n_samples)

    spike_indicator = dict()
    spikes_nwb_table = CuratedSpikeSorting() & key

    for n_trode in spikes_nwb_table.fetch_nwb():
        try:
            for unit_id, unit_spike_times in n_trode["units"][
                "spike_times"
            ].items():
                unit_spike_times = unit_spike_times[
                    (unit_spike_times > time[0])
                    & (unit_spike_times <= time[-1])
                ]
                unit_name = f'{n_trode["sort_group_id"]:04d}_{unit_id:04d}'
                spike_indicator[unit_name] = np.bincount(
                    np.digitize(unit_spike_times, time[1:-1]),
                    minlength=time.shape[0],
                )
        except KeyError:
            pass

    return pd.DataFrame(
        spike_indicator,
        index=pd.Index(time, name="time"),
    )


def get_decoding_data_for_epoch(
    nwb_file_name: str,
    interval_list_name: str,
    position_info_param_name: str = "default",
    additional_spike_keys: dict = {},
) -> tuple[pd.DataFrame, pd.DataFrame, list[slice]]:
    """Collects the data needed for decoding

    Parameters
    ----------
    nwb_file_name : str
    interval_list_name : str
    position_info_param_name : str, optional
    additional_spike_keys : dict, optional

    Returns
    -------
    position_info : pd.DataFrame, shape (n_time, n_position_features)
    spikes : pd.DataFrame, shape (n_time, n_units)
    valid_slices : list[slice]

    """

    valid_slices = convert_valid_times_to_slice(
        get_valid_ephys_position_times_by_epoch(nwb_file_name)[
            interval_list_name
        ]
    )

    # position interval
    nwb_dict = dict(nwb_file_name=nwb_file_name)
    pos_interval_dict = dict(
        nwb_dict,
        interval_list_name=convert_epoch_interval_name_to_position_interval_name(
            {
                **nwb_dict,
                "interval_list_name": interval_list_name,
            }
        ),
    )

    position_info = (
        IntervalPositionInfo()
        & {
            **pos_interval_dict,
            "position_info_param_name": position_info_param_name,
        }
    ).fetch1_dataframe()

    # spikes
    valid_times = np.asarray(
        [(times.start, times.stop) for times in valid_slices]
    )

    curated_spikes_key = {
        "nwb_file_name": nwb_file_name,
        **additional_spike_keys,
    }
    spikes = get_spike_indicator(
        curated_spikes_key,
        (valid_times.min(), valid_times.max()),
        sampling_rate=500,
    )
    spikes = pd.concat([spikes.loc[times] for times in valid_slices])

    new_time = spikes.index.to_numpy()
    new_index = pd.Index(
        np.unique(np.concatenate((position_info.index, new_time))),
        name="time",
    )
    position_info = (
        position_info.reindex(index=new_index)
        .interpolate(method="linear")
        .reindex(index=new_time)
    )

    return position_info, spikes, valid_slices


def get_data_for_multiple_epochs(
    nwb_file_name: str,
    epoch_names: list,
    position_info_param_name: str = "decoding",
    additional_spike_keys: dict = {},
) -> tuple[pd.DataFrame, pd.DataFrame, list[slice], np.ndarray, np.ndarray]:
    """Collects the data needed for decoding for multiple epochs

    Parameters
    ----------
    nwb_file_name : str
    epoch_names : list
    position_info_param_name : str, optional
    additional_spike_keys : dict, optional

    Returns
    -------
    position_info : pd.DataFrame, shape (n_time, n_position_features)
    spikes : pd.DataFrame, shape (n_time, n_units)
    valid_slices : list[slice]
    environment_labels : np.ndarray, shape (n_time,)
        The environment label for each time point
    sort_group_ids : np.ndarray, shape (n_units,)
        The sort group of each unit
    """
    data = []
    environment_labels = []

    for epoch in epoch_names:
        logger.info(epoch)
        data.append(
            get_decoding_data_for_epoch(
                nwb_file_name,
                epoch,
                position_info_param_name=position_info_param_name,
                additional_spike_keys=additional_spike_keys,
            )
        )
        n_time = data[-1][0].shape[0]
        environment_labels.append([epoch] * n_time)

    environment_labels = np.concatenate(environment_labels, axis=0)
    position_info, spikes, valid_slices = list(zip(*data))
    position_info = pd.concat(position_info, axis=0)
    spikes = pd.concat(spikes, axis=0)
    valid_slices = {
        epoch: valid_slice
        for epoch, valid_slice in zip(epoch_names, valid_slices)
    }

    assert position_info.shape[0] == spikes.shape[0]

    sort_group_ids = np.asarray(
        [int(col.split("_")[0]) for col in spikes.columns]
    )

    return (
        position_info,
        spikes,
        valid_slices,
        environment_labels,
        sort_group_ids,
    )

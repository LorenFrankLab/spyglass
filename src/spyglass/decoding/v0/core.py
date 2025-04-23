import numpy as np
import pandas as pd

from spyglass.utils import logger

try:
    from replay_trajectory_classification.continuous_state_transitions import (
        RandomWalk,
        Uniform,
    )
    from replay_trajectory_classification.environments import Environment
    from replay_trajectory_classification.observation_model import (
        ObservationModel,
    )
except (ImportError, ModuleNotFoundError) as e:
    RandomWalk, Uniform, Environment, ObservationModel = None, None, None, None
    logger.warning(e)

from spyglass.common.common_behav import PositionIntervalMap, RawPosition
from spyglass.common.common_interval import IntervalList


def get_valid_ephys_position_times_from_interval(
    interval_list_name: str, nwb_file_name: str
) -> np.ndarray:
    """Returns the intersection of valid times across ephys and position data.

    Finds the intersection of the valid times for the interval list, the
    valid times for the ephys data, and the valid times for the position data.

    Parameters
    ----------
    interval_list_name : str
    nwb_file_name : str

    Returns
    -------
    valid_ephys_position_times : np.ndarray, shape (n_valid_times, 2)
    """
    nwb_dict = dict(nwb_file_name=nwb_file_name)
    i_pk = "interval_list_name"

    interval_valid_times = (
        IntervalList & {**nwb_dict, i_pk: interval_list_name}
    ).fetch_interval()

    position_interval_names = (RawPosition & nwb_dict).fetch(i_pk)
    position_interval_names = position_interval_names[
        np.argsort(
            [
                int(name.strip("pos valid time"))
                for name in position_interval_names
            ]
        )
    ]
    valid_pos_times = [  # TODO: replace fetch1 with single query
        (IntervalList & {**nwb_dict, i_pk: pos_interval_name}).fetch1(
            "valid_times"
        )
        for pos_interval_name in position_interval_names
    ]

    valid_ephys_times = (
        IntervalList & {**nwb_dict, i_pk: "raw data valid times"}
    ).fetch_interval()

    return (
        interval_valid_times.intersect(valid_ephys_times)
        .intersect(np.concatenate(valid_pos_times))
        .times
    )


def get_epoch_interval_names(nwb_file_name: str) -> list[str]:
    """Find the interval names that are epochs.

    Parameters
    ----------
    nwb_file_name : str

    Returns
    -------
    epoch_names : list[str]
        List of interval names that are epochs.
    """
    interval_list = pd.DataFrame(
        IntervalList() & {"nwb_file_name": nwb_file_name}
    )

    interval_list = interval_list.loc[
        interval_list.interval_list_name.str.contains(
            r"^(?:\d+)_(?:\w+)$", regex=True, na=False
        )
    ]

    return interval_list.interval_list_name.tolist()


def get_valid_ephys_position_times_by_epoch(
    nwb_file_name: str,
) -> dict[str, np.ndarray]:
    """Get the valid ephys position times for each epoch.

    Parameters
    ----------
    nwb_file_name : str

    Returns
    -------
    valid_ephys_position_times_by_epoch : dict[str, np.ndarray]
        Dictionary of epoch names and valid ephys position times.

    """
    return {
        epoch: get_valid_ephys_position_times_from_interval(
            epoch, nwb_file_name
        )
        for epoch in get_epoch_interval_names(nwb_file_name)
    }


def convert_valid_times_to_slice(valid_times: np.ndarray) -> list[slice]:
    """Converts valid times to a list of slices for easy indexing.

    Parameters
    ----------
    valid_times : np.ndarray, shape (n_valid_times, 2)
        Start and end times for each valid time.

    Returns
    -------
    valid_time_slices : list[slice]

    """
    return [slice(times[0], times[1]) for times in valid_times]


def create_model_for_multiple_epochs(
    epoch_names: list[str], env_kwargs: dict
) -> tuple[list[ObservationModel], list[Environment], list[list[object]]]:
    """Creates the observation model, environment, and continuous transition
    types for multiple epochs for decoding

    Parameters
    ----------
    epoch_names : list[str], length (n_epochs)
    env_kwargs : dict
        Environment keyword arguments.

    Returns
    -------
    observation_models: tuple[list[ObservationModel]
        Observation model for each epoch.
    environments : list[Environment]
        Environment for each epoch.
    continuous_transition_types : list[list[object]]]
        Continuous transition types for each epoch.

    """
    observation_models = []
    environments = []
    continuous_transition_types = []

    for epoch in epoch_names:
        observation_models.append(ObservationModel(epoch))
        environments.append(Environment(epoch, **env_kwargs))

    for epoch1 in epoch_names:
        continuous_transition_types.append([])
        for epoch2 in epoch_names:
            if epoch1 == epoch2:
                continuous_transition_types[-1].append(
                    RandomWalk(epoch1, use_diffusion=False)
                )
            else:
                continuous_transition_types[-1].append(Uniform(epoch1, epoch2))

    return observation_models, environments, continuous_transition_types

"""Converts decoder classes into dictionaries and dictionaries into classes
so that datajoint can store them in tables."""

from spyglass.utils import logger

try:
    from replay_trajectory_classification.continuous_state_transitions import (
        Identity,
        RandomWalk,
        RandomWalkDirection1,
        RandomWalkDirection2,
        Uniform,
    )
    from replay_trajectory_classification.discrete_state_transitions import (
        DiagonalDiscrete,
        RandomDiscrete,
        UniformDiscrete,
        UserDefinedDiscrete,
    )
    from replay_trajectory_classification.environments import Environment
    from replay_trajectory_classification.initial_conditions import (
        UniformInitialConditions,
        UniformOneEnvironmentInitialConditions,
    )
    from replay_trajectory_classification.observation_model import (
        ObservationModel,
    )
except ImportError as e:
    logger.warning(e)
from track_linearization import make_track_graph


def _convert_dict_to_class(d: dict, class_conversion: dict) -> object:
    """Converts a dictionary into a class object

    Parameters
    ----------
    d : dict
    class_conversion : dict

    Returns
    -------
    class_based_on_dict : object

    """
    class_name = d.pop("class_name")
    return class_conversion[class_name](**d)


def _convert_env_dict(env_params: dict) -> Environment:
    """If the track graph is in the environment parameters, convert it to a networkx graph

    Parameters
    ----------
    env_params : dict

    Returns
    -------
    environment : Environment
    """
    if env_params["track_graph"] is not None:
        env_params["track_graph"] = make_track_graph(
            **env_params["track_graph"]
        )

    return Environment(**env_params)


def _to_dict(transition: object) -> dict:
    """Helper function to convert a transition class into a dictionary"""
    parameters = vars(transition)
    parameters["class_name"] = type(transition).__name__

    return parameters


def _convert_transitions_to_dict(
    transitions: list[list[object]],
) -> list[list[dict]]:
    """Converts a list of lists of transition classes into a list of lists of dictionaries"""
    return [
        [_to_dict(transition) for transition in transition_rows]
        for transition_rows in transitions
    ]


def restore_classes(params: dict) -> dict:
    """Converts a dictionary of parameters into a dictionary of classes since datajoint cannot handle classes"""
    continuous_state_transition_types = {
        "RandomWalk": RandomWalk,
        "RandomWalkDirection1": RandomWalkDirection1,
        "RandomWalkDirection2": RandomWalkDirection2,
        "Uniform": Uniform,
        "Identity": Identity,
    }

    discrete_state_transition_types = {
        "DiagonalDiscrete": DiagonalDiscrete,
        "UniformDiscrete": UniformDiscrete,
        "RandomDiscrete": RandomDiscrete,
        "UserDefinedDiscrete": UserDefinedDiscrete,
    }

    initial_conditions_types = {
        "UniformInitialConditions": UniformInitialConditions,
        "UniformOneEnvironmentInitialConditions": UniformOneEnvironmentInitialConditions,
    }

    params["classifier_params"]["continuous_transition_types"] = [
        [
            _convert_dict_to_class(st, continuous_state_transition_types)
            for st in sts
        ]
        for sts in params["classifier_params"]["continuous_transition_types"]
    ]
    params["classifier_params"]["environments"] = [
        _convert_env_dict(env_params)
        for env_params in params["classifier_params"]["environments"]
    ]
    params["classifier_params"]["discrete_transition_type"] = (
        _convert_dict_to_class(
            params["classifier_params"]["discrete_transition_type"],
            discrete_state_transition_types,
        )
    )
    params["classifier_params"]["initial_conditions_type"] = (
        _convert_dict_to_class(
            params["classifier_params"]["initial_conditions_type"],
            initial_conditions_types,
        )
    )

    if params["classifier_params"].get("observation_models"):
        params["classifier_params"]["observation_models"] = [
            ObservationModel(obs)
            for obs in params["classifier_params"]["observation_models"]
        ]

    return params


def _convert_algorithm_params(algo_params):
    """Helper function that adds in the algorithm name to the algorithm parameters dictionary"""
    try:
        algo_params = algo_params.copy()
        algo_params["model"] = algo_params["model"].__name__
    except KeyError:
        pass

    return algo_params


def _convert_environment_to_dict(env: Environment) -> dict:
    """Converts an Environment instance into a dictionary so that datajoint can store it"""
    if env.track_graph is not None:
        track_graph = env.track_graph
        env.track_graph = {
            "node_positions": [
                v["pos"] for v in dict(track_graph.nodes).values()
            ],
            "edges": list(track_graph.edges),
        }

    return vars(env)


def convert_classes_to_dict(key: dict) -> dict:
    """Converts the classifier parameters into a dictionary so that datajoint can store it."""
    try:
        key["classifier_params"]["environments"] = [
            _convert_environment_to_dict(env)
            for env in key["classifier_params"]["environments"]
        ]
    except TypeError:
        key["classifier_params"]["environments"] = [
            _convert_environment_to_dict(
                key["classifier_params"]["environments"]
            )
        ]
    key["classifier_params"]["continuous_transition_types"] = (
        _convert_transitions_to_dict(
            key["classifier_params"]["continuous_transition_types"]
        )
    )
    key["classifier_params"]["discrete_transition_type"] = _to_dict(
        key["classifier_params"]["discrete_transition_type"]
    )
    key["classifier_params"]["initial_conditions_type"] = _to_dict(
        key["classifier_params"]["initial_conditions_type"]
    )

    if key["classifier_params"]["observation_models"] is not None:
        key["classifier_params"]["observation_models"] = [
            vars(obs) for obs in key["classifier_params"]["observation_models"]
        ]

    try:
        key["classifier_params"]["clusterless_algorithm_params"] = (
            _convert_algorithm_params(
                key["classifier_params"]["clusterless_algorithm_params"]
            )
        )
    except KeyError:
        pass

    return key

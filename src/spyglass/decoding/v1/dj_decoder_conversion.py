"""Converts decoder classes into dictionaries and dictionaries into classes
so that datajoint can store them in tables."""

import copy

import datajoint as dj
from non_local_detector import continuous_state_transitions as cst
from non_local_detector import discrete_state_transitions as dst
from non_local_detector import initial_conditions as ic
from non_local_detector.environment import Environment
from non_local_detector.observation_models import ObservationModel
from track_linearization import make_track_graph

schema = dj.schema("decoding_clusterless_v1")


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
    if class_name not in class_conversion:
        raise ValueError(f"Invalid class name: {class_name}")
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


def _map_class_name_to_class(module: object) -> dict:
    """Helper function to map name of class to class

    Parameters
    ----------
    module : object
        The module to get the classes from

    Returns
    -------
    dict
        A dictionary of the classes in the module mapping the class name to the class
    """
    module_attributes = dir(module)
    return {
        attr_name: attr
        for attr_name, attr in [
            (name, getattr(module, name)) for name in module_attributes
        ]
        if hasattr(attr, "__class__") and attr.__class__.__name__ == "type"
    }


def restore_classes(params: dict) -> dict:
    """Converts a dictionary of parameters into a dictionary of classes
    since datajoint cannot handle classes

    Parameters
    ----------
    params : dict
        The parameters to convert

    Returns
    -------
    converted_params : dict
        The converted parameters
    """

    params = copy.deepcopy(params)

    continuous_state_transition_types = _map_class_name_to_class(cst)
    discrete_state_transition_types = _map_class_name_to_class(dst)
    continuous_initial_conditions_types = _map_class_name_to_class(ic)

    params["environments"] = [
        _convert_env_dict(env_params) for env_params in params["environments"]
    ]

    params["continuous_transition_types"] = [
        [
            _convert_dict_to_class(st, continuous_state_transition_types)
            for st in sts
        ]
        for sts in params["continuous_transition_types"]
    ]
    params["discrete_transition_type"] = _convert_dict_to_class(
        params["discrete_transition_type"],
        discrete_state_transition_types,
    )
    params["continuous_initial_conditions_types"] = [
        _convert_dict_to_class(cont_ic, continuous_initial_conditions_types)
        for cont_ic in params["continuous_initial_conditions_types"]
    ]

    if params["observation_models"] is not None:
        params["observation_models"] = [
            ObservationModel(**obs) for obs in params["observation_models"]
        ]

    return params


def _convert_algorithm_params(algo_params: dict) -> dict:
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
    try:
        if env.track_graphDD is not None:
            track_graphDD = env.track_graphDD
            env.track_graphDD = {
                "node_positions": [
                    v["pos"] for v in dict(track_graphDD.nodes).values()
                ],
                "edges": list(track_graphDD.edges),
            }
    except AttributeError:
        pass

    return vars(env)


def convert_classes_to_dict(params: dict) -> dict:
    """Converts the classifier parameters into a dictionary so that datajoint can store it."""
    params = copy.deepcopy(params)
    try:
        params["environments"] = [
            _convert_environment_to_dict(env) for env in params["environments"]
        ]
    except TypeError:
        params["environments"] = [
            _convert_environment_to_dict(params["environments"])
        ]
    params["continuous_transition_types"] = _convert_transitions_to_dict(
        params["continuous_transition_types"]
    )
    params["discrete_transition_type"] = _to_dict(
        params["discrete_transition_type"]
    )
    params["continuous_initial_conditions_types"] = [
        _to_dict(cont_ic)
        for cont_ic in params["continuous_initial_conditions_types"]
    ]

    if params["observation_models"] is not None:
        params["observation_models"] = [
            vars(obs) for obs in params["observation_models"]
        ]

    try:
        params["clusterless_algorithm_params"] = _convert_algorithm_params(
            params["clusterless_algorithm_params"]
        )
    except KeyError:
        pass

    return params

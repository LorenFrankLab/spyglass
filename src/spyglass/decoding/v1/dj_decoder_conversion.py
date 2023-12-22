"""Converts decoder classes into dictionaries and dictionaries into classes
so that datajoint can store them in tables."""


import datajoint as dj
from non_local_detector import ContFragClusterlessClassifier
from non_local_detector.continuous_state_transitions import (
    Discrete,
    EmpiricalMovement,
    RandomWalk,
    RandomWalkDirection1,
    RandomWalkDirection2,
    Uniform,
)
from non_local_detector.discrete_state_transitions import (
    DiscreteNonStationaryCustom,
    DiscreteNonStationaryDiagonal,
    DiscreteStationaryCustom,
    DiscreteStationaryDiagonal,
)
from non_local_detector.environment import Environment
from non_local_detector.initial_conditions import UniformInitialConditions
from non_local_detector.observation_models import ObservationModel
from track_linearization import make_track_graph

schema = dj.schema("decoding_clusterless_v1")


import copy

from non_local_detector import ContFragClusterlessClassifier
from non_local_detector.continuous_state_transitions import (
    Discrete,
    EmpiricalMovement,
    RandomWalk,
    RandomWalkDirection1,
    RandomWalkDirection2,
    Uniform,
)
from non_local_detector.discrete_state_transitions import (
    DiscreteNonStationaryCustom,
    DiscreteNonStationaryDiagonal,
    DiscreteStationaryCustom,
    DiscreteStationaryDiagonal,
)
from non_local_detector.environment import Environment
from non_local_detector.initial_conditions import UniformInitialConditions
from non_local_detector.observation_models import ObservationModel
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


def restore_classes(params: dict) -> dict:
    """Converts a dictionary of parameters into a dictionary of classes since datajoint cannot handle classes"""

    params = copy.deepcopy(params)
    continuous_state_transition_types = {
        "Discrete": Discrete,
        "EmpiricalMovement": EmpiricalMovement,
        "RandomWalk": RandomWalk,
        "RandomWalkDirection1": RandomWalkDirection1,
        "RandomWalkDirection2": RandomWalkDirection2,
        "Uniform": Uniform,
    }

    discrete_state_transition_types = {
        "DiscreteNonStationaryCustom": DiscreteNonStationaryCustom,
        "DiscreteNonStationaryDiagonal": DiscreteNonStationaryDiagonal,
        "DiscreteStationaryCustom": DiscreteStationaryCustom,
        "DiscreteStationaryDiagonal": DiscreteStationaryDiagonal,
    }

    continuous_initial_conditions_types = {
        "UniformInitialConditions": UniformInitialConditions,
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
    params["classifier_params"][
        "discrete_transition_type"
    ] = _convert_dict_to_class(
        params["classifier_params"]["discrete_transition_type"],
        discrete_state_transition_types,
    )
    params["classifier_params"]["continuous_initial_conditions_types"] = [
        _convert_dict_to_class(cont_ic, continuous_initial_conditions_types)
        for cont_ic in params["classifier_params"][
            "continuous_initial_conditions_types"
        ]
    ]

    if params["classifier_params"]["observation_models"] is not None:
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


def convert_classes_to_dict(key: dict) -> dict:
    key = copy.deepcopy(key)
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
    key["classifier_params"][
        "continuous_transition_types"
    ] = _convert_transitions_to_dict(
        key["classifier_params"]["continuous_transition_types"]
    )
    key["classifier_params"]["discrete_transition_type"] = _to_dict(
        key["classifier_params"]["discrete_transition_type"]
    )
    key["classifier_params"]["continuous_initial_conditions_types"] = [
        _to_dict(cont_ic)
        for cont_ic in key["classifier_params"][
            "continuous_initial_conditions_types"
        ]
    ]

    if key["classifier_params"]["observation_models"] is not None:
        key["classifier_params"]["observation_models"] = [
            vars(obs) for obs in key["classifier_params"]["observation_models"]
        ]

    try:
        key["classifier_params"][
            "clusterless_algorithm_params"
        ] = _convert_algorithm_params(
            key["classifier_params"]["clusterless_algorithm_params"]
        )
    except KeyError:
        pass

    return key


@schema
class DecodingParameters(dj.Lookup):
    """Parameters for decoding the animal's mental position and some category of interest"""

    definition = """
    classifier_param_name : varchar(80) # a name for this set of parameters
    ---
    classifier_params :   BLOB        # initialization parameters for model
    estimate_parameters_kwargs : BLOB # keyword arguments for estimate_parameters
    """

    def insert_default(self):
        self.insert1(
            {
                "classifier_param_name": "contfrag_clusterless",
                "classifier_params": vars(ContFragClusterlessClassifier()),
                "estimate_parameters_kwargs": dict(),
            },
            skip_duplicates=True,
        )

    def insert1(self, key, **kwargs):
        super().insert1(convert_classes_to_dict(key), **kwargs)

    def fetch1(self, *args, **kwargs):
        return restore_classes(super().fetch1(*args, **kwargs))

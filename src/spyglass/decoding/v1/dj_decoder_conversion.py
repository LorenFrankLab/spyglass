"""Converts decoder classes into dictionaries and dictionaries into classes
so that datajoint can store them in tables."""

import copy
import inspect

import datajoint as dj
import non_local_detector
from non_local_detector import continuous_state_transitions as cst
from non_local_detector import discrete_state_transitions as dst
from non_local_detector import initial_conditions as ic
from non_local_detector.environment import Environment
from non_local_detector.models import base as nld_base
from non_local_detector.observation_models import ObservationModel
from track_linearization import make_track_graph

schema = dj.schema("decoding_clusterless_v1")


def _model_class_registry() -> dict:
    """Map model class name -> class for decoder reconstruction.

    Covers the public detector/classifier classes (e.g.
    ``NonLocalClusterlessDetector``, ``ContFragSortedSpikesClassifier``) plus
    the base ``ClusterlessDetector`` / ``SortedSpikesDetector``.
    """
    return {
        **_map_class_name_to_class(non_local_detector),
        **_map_class_name_to_class(nld_base),
    }


def _init_param_names(cls: type) -> set:
    """Constructor parameter names of a detector class (excluding ``self``)."""
    return set(inspect.signature(cls.__init__).parameters) - {"self"}


# Modality-exclusive constructor parameters that identify a legacy row's
# detector family, paired with that family's base and NonLocal class names.
_LEGACY_MODALITY_MARKERS = (
    (
        "clusterless_algorithm",
        "ClusterlessDetector",
        "NonLocalClusterlessDetector",
    ),
    (
        "sorted_spikes_algorithm",
        "SortedSpikesDetector",
        "NonLocalSortedSpikesDetector",
    ),
)


def _restore_legacy_detector(params: dict):
    """Reconstruct a legacy (``class_name``-less) parameter dict.

    Legacy ``DecodingParameters`` rows were serialized via ``vars(model)`` and
    record no class name, so two things must be handled:

    1. ``vars(model)`` includes derived internal attributes that are not
       constructor parameters (e.g. ``_frozen_discrete_transition_rows_mask_``,
       a mask computed in ``__init__``); these ``_``-prefixed keys are stripped.
    2. ``NonLocal*`` detectors accept extra constructor parameters
       (``non_local_position_penalty`` / ``non_local_penalty_std``) that the base
       detector rejects. Such rows are rebuilt with the concrete NonLocal class
       so the parameters round-trip instead of raising ``TypeError`` when the
       make() site constructs the base detector.

    Base / ContFrag rows (whose keys the base detector accepts) are returned as a
    dict, so the make() sites construct them with the base detector exactly as
    before. Rows whose keys match neither class -- e.g. serialized by a newer
    non_local_detector than is installed -- are also returned as a dict, so the
    base detector raises loudly rather than this helper silently dropping
    parameters that would change the model.

    Returns
    -------
    object or dict
        A ``NonLocal*`` detector instance for NonLocal legacy rows, otherwise the
        stripped parameter dict.
    """
    params = {
        key: value for key, value in params.items() if not key.startswith("_")
    }
    keys = set(params)
    registry = _model_class_registry()
    for marker, base_name, nonlocal_name in _LEGACY_MODALITY_MARKERS:
        if marker not in keys:
            continue
        base_cls = registry.get(base_name)
        nonlocal_cls = registry.get(nonlocal_name)
        base_accepts = base_cls is not None and keys <= _init_param_names(
            base_cls
        )
        nonlocal_accepts = (
            nonlocal_cls is not None and keys <= _init_param_names(nonlocal_cls)
        )
        if not base_accepts and nonlocal_accepts:
            return nonlocal_cls(**params)
        break
    return params


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
        # ``isinstance(attr, type)`` -- not ``attr.__class__.__name__ ==
        # "type"`` -- so that classes with a non-``type`` metaclass are
        # included. The detector/classifier classes are ``BaseEstimator``
        # subclasses whose metaclass is ``ABCMeta``; the stricter check would
        # silently drop every one of them from the registry.
        if isinstance(attr, type)
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
    model : object or dict
        A reconstructed detector/classifier instance when the stored params
        carry a ``"class_name"`` (the current format), or for legacy NonLocal
        rows whose extra parameters only the concrete NonLocal class accepts.
        For other legacy rows stored without a class name, the converted
        parameter ``dict`` is returned (with derived, non-constructor
        ``_``-prefixed attributes stripped) for backward compatibility.
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

    # Reconstruct the detector instance via its stored class. Reconstructing
    # with the concrete class (rather than the base detector) is what lets
    # subclass-only parameters -- e.g. NonLocal*'s ``non_local_position_penalty``
    # / ``non_local_penalty_std`` -- round-trip. Legacy rows serialized without
    # a class name fall back to the param dict for backward compatibility.
    model_class_name = params.pop("class_name", None)
    if model_class_name is None:
        # Legacy rows serialized via ``vars(model)`` carry no class name: strip
        # derived internal attributes and rebuild NonLocal rows via their
        # concrete class (see ``_restore_legacy_detector``).
        return _restore_legacy_detector(params)

    model_classes = _model_class_registry()
    if model_class_name not in model_classes:
        raise ValueError(
            f"Unknown decoder model class '{model_class_name}'. "
            f"Known classes: {sorted(model_classes)}"
        )
    return model_classes[model_class_name](**params)


def _convert_algorithm_params(algo_params: dict | None) -> dict | None:
    """Helper function that adds in the algorithm name to the algorithm parameters dictionary"""
    # Some detectors default the algorithm params to None (e.g. current
    # non_local_detector clusterless detectors); there is nothing to convert.
    if algo_params is None:
        return None
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

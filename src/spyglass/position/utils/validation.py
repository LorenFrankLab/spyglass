"""Shared parameter validation utilities for Position V1/V2 pipelines.

Consolidates duplicate validation logic with consistent error messages.

Typed parameter dataclasses
---------------------------
Each param family exposes a set of frozen dataclasses whose ``method`` field
is a ``Literal`` discriminator.  Constructing one of these objects gives you
IDE auto-complete, type-checking, and a structural guarantee that all
required fields for the chosen method are present — without touching the
database schema (they are serialised to plain dicts via ``to_dict()``).

Orientation::

    from spyglass.position.utils.validation import TwoPtOrientParams
    orient = TwoPtOrientParams(bodypart1="greenLED", bodypart2="redLED_C")
    orient.to_dict()  # {"method": "two_pt", "bodypart1": ..., ...}

Centroid::

    from spyglass.position.utils.validation import TwoPtCentroidParams
    centroid = TwoPtCentroidParams(
        points={"point1": "greenLED", "point2": "redLED_C"},
        max_LED_separation=15.0,
    )

Smoothing method sub-params::

    from spyglass.position.utils.validation import MovingAvgParams, SmoothingParams
    smoothing = SmoothingParams(
        likelihood_thresh=0.95,
        smooth=True,
        smoothing_params=MovingAvgParams(smoothing_duration=0.05),
    )

All dataclasses are accepted wherever plain dicts were accepted before.
"""

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Literal, Optional, Tuple, Union

# ---------------------------------------------------------------------------
# Typed parameter dataclasses
# ---------------------------------------------------------------------------
# Orientation ---------------------------------------------------------------


@dataclass(frozen=True)
class TwoPtOrientParams:
    """Two-point head-orientation parameters.

    ``method`` is a ``Literal["two_pt"]`` discriminator so static tools and
    runtime code both know which bodypart keys are required.
    """

    bodypart1: str
    bodypart2: str
    method: Literal["two_pt"] = "two_pt"
    interpolate: bool = True
    smooth: bool = True
    smoothing_params: Optional[Dict] = None

    def to_dict(self) -> dict:
        """Return a plain dict suitable for DB insertion."""
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class BisectorOrientParams:
    """Bisector (three-LED) orientation parameters."""

    led1: str
    led2: str
    led3: str
    method: Literal["bisector"] = "bisector"
    interpolate: bool = True
    smooth: bool = True
    smoothing_params: Optional[Dict] = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class NoOrientParams:
    """Sentinel for 'no orientation calculation'."""

    method: Literal["none"] = "none"

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


#: Discriminated union for orientation params.
OrientParams = Union[TwoPtOrientParams, BisectorOrientParams, NoOrientParams]

# Centroid ------------------------------------------------------------------


@dataclass(frozen=True)
class OnePtCentroidParams:
    """Single-point (passthrough) centroid parameters."""

    points: Dict[str, str]  # {"point1": "<bodypart>"}
    method: Literal["1pt"] = "1pt"

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class TwoPtCentroidParams:
    """Two-point centroid parameters."""

    points: Dict[str, str]  # {"point1": "<bp1>", "point2": "<bp2>"}
    max_LED_separation: float
    method: Literal["2pt"] = "2pt"

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class FourPtCentroidParams:
    """Four-LED centroid parameters (Frank Lab standard)."""

    points: Dict[str, str]  # keys: greenLED, redLED_C, redLED_L, redLED_R
    max_LED_separation: float
    method: Literal["4pt"] = "4pt"

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


#: Discriminated union for centroid params.
CentroidParams = Union[
    OnePtCentroidParams, TwoPtCentroidParams, FourPtCentroidParams
]

# Smoothing method sub-params -----------------------------------------------


@dataclass(frozen=True)
class MovingAvgParams:
    """Moving-average smoothing sub-parameters."""

    smoothing_duration: float = 0.05
    method: Literal["moving_avg"] = "moving_avg"

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class SavGolParams:
    """Savitzky-Golay smoothing sub-parameters."""

    window_length: int = 11
    polyorder: int = 3
    method: Literal["savgol"] = "savgol"

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class GaussianParams:
    """Gaussian smoothing sub-parameters."""

    smoothing_duration: float = 0.05
    method: Literal["gaussian"] = "gaussian"

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


#: Discriminated union for smoothing method sub-params.
SmoothingMethodParams = Union[MovingAvgParams, SavGolParams, GaussianParams]


@dataclass
class SmoothingParams:
    """Top-level smoothing / interpolation parameters.

    ``smoothing_params`` accepts either a typed sub-dataclass or a plain dict.
    """

    likelihood_thresh: float
    interpolate: bool = False
    interp_params: Optional[Dict] = None
    smooth: bool = False
    smoothing_params: Optional[Union[SmoothingMethodParams, Dict]] = None

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        # If smoothing_params was a dataclass, asdict already recursed into it.
        return d


def _to_dict_if_dataclass(obj):
    """Return ``obj.to_dict()`` if it is a dataclass, else return ``obj``."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return obj.to_dict()
    return obj


def validate_option(
    option: Any,
    options: Optional[Iterable] = None,
    name: str = "option",
    types: Optional[Union[type, Tuple[type, ...]]] = None,
    val_range: Optional[Tuple[float, float]] = None,
    permit_none: bool = False,
) -> None:
    """Validate that option meets specified criteria.

    Parameters
    ----------
    option : Any
        Option value to validate
    options : Optional[Iterable], optional
        If provided, option must be in this collection
    name : str, optional
        Name of option for error messages, by default "option"
    types : Optional[Union[type, Tuple[type, ...]]], optional
        If provided, option must be instance of these types
    val_range : Optional[Tuple[float, float]], optional
        If provided, option must be in range (min, max) inclusive
    permit_none : bool, optional
        If True, permit option to be None, by default False

    Raises
    ------
    ValueError
        If option fails validation
    TypeError
        If option is wrong type
    KeyError
        If option not in allowed options
    """
    if option is None and not permit_none:
        raise ValueError(f"{name} cannot be None")

    if option is None and permit_none:
        return  # Allow None, skip other checks

    if options is not None and option not in options:
        raise KeyError(
            f"Unknown {name}: {option}. Available options: {list(options)}"
        )

    if types is not None:
        if not isinstance(types, (tuple, list)):
            types = (types,)
        if not isinstance(option, types):
            type_names = [t.__name__ for t in types]
            raise TypeError(
                f"{name} is {type(option).__name__}. "
                f"Expected types: {type_names}"
            )

    if val_range is not None:
        min_val, max_val = val_range
        if min_val is not None and option < min_val:
            raise ValueError(f"{name} must be >= {min_val}, got {option}")
        if max_val is not None and option > max_val:
            raise ValueError(f"{name} must be <= {max_val}, got {option}")


def validate_required_keys(
    params: dict, required_keys: Iterable[str], param_name: str = "parameters"
) -> None:
    """Validate that all required keys exist in parameters dict.

    Parameters
    ----------
    params : dict
        Parameters dictionary to validate
    required_keys : Iterable[str]
        Keys that must exist in params
    param_name : str, optional
        Name for error messages, by default "parameters"

    Raises
    ------
    ValueError
        If any required keys are missing
    """
    if not isinstance(params, dict):
        raise TypeError(
            f"{param_name} must be a dictionary, got {type(params)}"
        )

    missing_keys = [k for k in required_keys if k not in params]
    if missing_keys:
        raise ValueError(
            f"{param_name} missing required keys: {missing_keys}. "
            f"Required: {list(required_keys)}"
        )


def validate_list(
    required_items: list,
    option_list: list = None,
    name: str = "List",
    condition: str = "",
    permit_none: bool = False,
) -> None:
    """Validate that option_list contains all items in required_items.

    Parameters
    ----------
    required_items : list
        Items that must be present in option_list
    option_list : list, optional
        List to validate. If provided, must contain all items in required_items.
    name : str, optional
        Name for error messages, by default "List"
    condition : str, optional
        Additional context for error messages, by default ""
    permit_none : bool, optional
        If True, allow option_list to be None, by default False

    Raises
    ------
    ValueError
        If option_list is None when permit_none is False
    KeyError
        If option_list is missing required items
    """
    if option_list is None:
        if permit_none:
            return
        else:
            raise ValueError(f"{name} cannot be None")

    if condition:
        condition = f" when using {condition}"

    missing_items = [x for x in required_items if x not in option_list]
    if missing_items:
        raise KeyError(
            f"{name} must contain all items in {required_items}{condition}. "
            f"Missing: {missing_items}"
        )


def validate_smoothing_params(
    params: Union["SmoothingParams", dict],
) -> None:
    """Validate smoothing parameters for pose processing.

    Accepts either a :class:`SmoothingParams` dataclass or a plain ``dict``.
    Unified validation for both V1 and V2 smoothing parameter formats.
    Supports both V1-style (smooth boolean + smoothing_params) and
    V2-style (nested structure with method validation).

    Parameters
    ----------
    params : SmoothingParams or dict
        Smoothing parameters to validate

    Raises
    ------
    ValueError
        If smoothing parameters are invalid
    """
    if params is None:
        return  # No smoothing params provided (None)

    params = _to_dict_if_dataclass(params)

    # Normalise nested smoothing_params if it is a typed dataclass
    if (
        isinstance(params.get("smoothing_params"), dict) is False
        and params.get("smoothing_params") is not None
    ):
        params = dict(
            params,
            smoothing_params=_to_dict_if_dataclass(params["smoothing_params"]),
        )

    # Only require likelihood_thresh if smoothing or interpolation is happening
    has_smoothing = params.get("smooth", False) or "smoothing_params" in params
    has_interpolation = (
        params.get("interpolate", False) or "interp_params" in params
    )

    if (
        has_smoothing or has_interpolation
    ) and "likelihood_thresh" not in params:
        raise ValueError("Smoothing params must include 'likelihood_thresh'")

    # Check interpolation requirements
    if params.get("interpolate", False):
        if "interp_params" not in params:
            raise ValueError("interpolate=True requires 'interp_params'")

    # Check smoothing requirements
    if params.get("smooth", False):
        if "smoothing_params" not in params:
            raise ValueError("smooth=True requires 'smoothing_params' key")

        smooth_params = params["smoothing_params"]
        if not isinstance(smooth_params, dict):
            raise TypeError(
                f"smoothing_params must be dict, got {type(smooth_params)}"
            )

        # Validate method exists
        if "method" not in smooth_params:
            raise ValueError("smoothing_params must include 'method'")

        method = smooth_params["method"]

        # Import at runtime to avoid circular imports
        try:
            from spyglass.position.utils.interpolation import SMOOTHING_METHODS

            valid_methods = SMOOTHING_METHODS.keys()
        except ImportError:
            # Fallback if utils not available
            valid_methods = ["moving_avg", "savgol", "gaussian"]

        validate_option(method, options=valid_methods, name="smoothing method")

        # Validate duration/window parameter for method
        if "smoothing_duration" in smooth_params:
            validate_option(
                smooth_params["smoothing_duration"],
                types=(int, float),
                name="smoothing_duration",
                val_range=(0.001, None),  # Minimum 1ms
            )
        elif "window_length" in smooth_params:
            validate_option(
                smooth_params["window_length"],
                types=int,
                name="window_length",
                val_range=(1, None),  # Minimum 1 sample
            )


def validate_orientation_params(
    params: Union["OrientParams", dict],
    method: str = None,
) -> None:
    """Validate orientation calculation parameters.

    Accepts either a typed orientation dataclass (:class:`TwoPtOrientParams`,
    :class:`BisectorOrientParams`, :class:`NoOrientParams`) or a plain ``dict``.

    Parameters
    ----------
    params : OrientParams or dict
        Orientation parameters to validate
    method : str, optional
        Orientation method to validate for. If None, method will be
        extracted from params['method']

    Raises
    ------
    ValueError
        If orientation parameters are invalid
    """
    if params is None:
        return  # No orientation params provided (None)

    params = _to_dict_if_dataclass(params)

    # If params is an empty dict or missing method, that's an error
    method = method or params.get("method")
    if not method:
        raise ValueError("Orientation params must include 'method'")

    # Define method requirements
    method_requirements = {
        "two_pt": ["bodypart1", "bodypart2"],
        "bisector": ["led1", "led2", "led3"],
        "none": [],
    }

    validate_option(
        method, options=method_requirements.keys(), name="orientation method"
    )

    required_keys = method_requirements[method]
    if required_keys:
        validate_required_keys(params, required_keys, "orientation params")


def validate_centroid_params(
    params: Union["CentroidParams", dict],
    method: str = None,
) -> None:
    """Validate centroid calculation parameters.

    Accepts either a typed centroid dataclass (:class:`OnePtCentroidParams`,
    :class:`TwoPtCentroidParams`, :class:`FourPtCentroidParams`) or a plain
    ``dict``.

    Parameters
    ----------
    params : CentroidParams or dict
        Centroid parameters to validate
    method : str, optional
        Centroid method to validate for. If None, method will be
        extracted from params['method']

    Raises
    ------
    ValueError
        If centroid parameters are invalid
    """
    if params is None:
        return  # No centroid params provided (None)

    params = _to_dict_if_dataclass(params)

    # If params is an empty dict or missing method, that's an error
    method = method or params.get("method")
    if not method:
        raise ValueError("Centroid params must specify 'method'")

    # Define method specifications
    method_specs = {
        "1pt": {
            "n_points": 1,
            "required_point_keys": None,
            "extra_required": [],
        },
        "2pt": {
            "n_points": 2,
            "required_point_keys": None,
            "extra_required": ["max_LED_separation"],
        },
        "4pt": {
            "n_points": 4,
            "required_point_keys": {
                "greenLED",
                "redLED_C",
                "redLED_L",
                "redLED_R",
            },
            "extra_required": ["max_LED_separation"],
        },
    }

    validate_option(method, options=method_specs.keys(), name="centroid method")

    spec = method_specs[method]

    # Validate point count - handle both flat and nested "points" dict formats
    if "points" in params and isinstance(params["points"], dict):
        # Nested format: {"method": "2pt", "points": {"point1": "led1", "point2": "led2"}}
        points = params["points"]
    else:
        # Flat format: {"method": "2pt", "point1": "led1", "point2": "led2"}
        points = {
            k: v
            for k, v in params.items()
            if k not in ["method", "points"] + spec["extra_required"]
        }

    if len(points) != spec["n_points"]:
        raise ValueError(
            f"{method} centroid requires exactly {spec['n_points']} point(s), "
            f"got {len(points)}"
        )

    # Validate specific point keys if required
    if spec["required_point_keys"] is not None:
        # For nested format, check the values in the points dict
        # For flat format, check the keys
        point_identifiers = (
            set(points.values()) if "points" in params else set(points.keys())
        )
        if point_identifiers != spec["required_point_keys"]:
            raise ValueError(
                f"{method} centroid requires points: {spec['required_point_keys']}. "
                f"Got: {point_identifiers}"
            )

    # Validate extra required parameters
    validate_required_keys(
        params, spec["extra_required"], f"{method} centroid params"
    )


def validate_interpolation_params(params: dict) -> None:
    """Validate interpolation parameters.

    Parameters
    ----------
    params : dict
        Interpolation parameters to validate

    Raises
    ------
    ValueError
        If interpolation parameters are invalid
    """
    if not params.get("interpolate", False):
        return

    if "interp_params" not in params:
        raise ValueError("interpolate=True requires 'interp_params'")

    interp_params = params["interp_params"]
    if not isinstance(interp_params, dict):
        raise TypeError(
            f"interp_params must be dict, got {type(interp_params)}"
        )

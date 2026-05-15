"""Parameter dataclasses for pose estimation configuration."""

from dataclasses import asdict, dataclass

from spyglass.position.utils.validation import (
    validate_centroid_params,
    validate_orientation_params,
    validate_smoothing_params,
)


@dataclass
class OrientationParams:
    """Dataclass for orientation parameters."""

    method: str
    bodypart1: str = ""
    bodypart2: str = ""
    led1: str = ""
    led2: str = ""
    led3: str = ""
    interpolate: bool = True
    smooth: bool = True
    smoothing_params: dict = None

    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        result = asdict(self)
        if self.method == "two_pt":
            result = {
                k: v
                for k, v in result.items()
                if k not in ["led1", "led2", "led3"] and v != ""
            }
        elif self.method == "bisector":
            result = {
                k: v
                for k, v in result.items()
                if k not in ["bodypart1", "bodypart2"] and v != ""
            }
        elif self.method == "none":
            result = {
                k: v
                for k, v in result.items()
                if k not in ["bodypart1", "bodypart2", "led1", "led2", "led3"]
                and v != ""
            }
        return {k: v for k, v in result.items() if v is not None and v != ""}


@dataclass
class CentroidParams:
    """Dataclass for centroid parameters."""

    method: str
    points: dict
    max_LED_separation: float = None  # noqa: N815

    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        result = asdict(self)
        return {k: v for k, v in result.items() if v is not None}


@dataclass
class SmoothingParams:
    """Dataclass for smoothing parameters."""

    interpolate: bool = True
    interp_params: dict = None
    smooth: bool = True
    smoothing_params: dict = None
    likelihood_thresh: float = 0.95

    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        result = asdict(self)
        return {k: v for k, v in result.items() if v is not None}


@dataclass
class PoseParameterSet:
    """Complete parameter set for pose estimation."""

    params_name: str
    orient: OrientationParams
    centroid: CentroidParams
    smoothing: SmoothingParams

    def validate(self):
        """Validate all parameter components."""
        validate_orientation_params(self.orient.to_dict())
        validate_centroid_params(self.centroid.to_dict())
        validate_smoothing_params(self.smoothing.to_dict())

    def to_params_dict(self) -> dict:
        """Convert to dictionary format for database insertion."""
        return {
            "pose_params_id": self.params_name,
            "orient": self.orient.to_dict(),
            "centroid": self.centroid.to_dict(),
            "smoothing": self.smoothing.to_dict(),
        }

# Import submodules to make them accessible
from spyglass.position.v2 import video
from spyglass.position.v2.estim import (
    PoseEstim,
    PoseEstimParams,
    PoseEstimSelection,
    PoseParams,
    PoseSelection,
    PoseV2,
)
from spyglass.position.v2.train import (
    BodyPart,
    Model,
    ModelParams,
    ModelSelection,
    Skeleton,
)
from spyglass.position.v2.video import VideoGroupParams, VidFileGroup

__all__ = [
    "BodyPart",
    "Model",
    "ModelParams",
    "ModelSelection",
    "PoseEstim",
    "PoseEstimParams",
    "PoseEstimSelection",
    "PoseParams",
    "PoseSelection",
    "PoseV2",
    "Skeleton",
    "VidFileGroup",
    "VideoGroupParams",
    "video",
]

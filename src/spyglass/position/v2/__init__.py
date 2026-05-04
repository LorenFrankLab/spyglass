"""Position V2 pipeline — model training, inference, and pose processing.

Typical workflow
----------------
1. **Train** (or import) a model::

       Model().load("path/to/ndx_pose.nwb")          # import pre-trained
       ModelParams().insert1({...})                    # or configure training
       ModelSelection().insert1({...})
       Model().populate()

2. **Run inference**::

       PoseEstimSelection().insert1({...})
       PoseEstim().populate()

3. **Process pose** (centroid, orientation, smoothing)::

       PoseSelection().insert1({...})
       PoseV2().populate()

Results are registered in ``PositionOutput`` automatically and accessible
via ``PositionOutput.fetch1_dataframe()``.
"""

# Import submodules to make them accessible
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
from spyglass.position.v2.video import (
    Calibration,
    CameraRig,
    VidFileGroup,
    VideoGroupParams,
)

__all__ = [
    "BodyPart",
    "Calibration",
    "CameraRig",
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
]

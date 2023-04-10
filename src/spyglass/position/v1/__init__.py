from .position_dlc_project import DLCProject, BodyPart
from .position_dlc_training import (
    DLCModelTrainingParams,
    DLCModelTrainingSelection,
    DLCModelTraining,
)
from .position_dlc_model import (
    DLCModelSource,
    DLCModelInput,
    DLCModelParams,
    DLCModelSelection,
    DLCModel,
    DLCModelEvaluation,
)
from .position_dlc_pose_estimation import DLCPoseEstimationSelection, DLCPoseEstimation
from .position_dlc_position import (
    DLCSmoothInterpParams,
    DLCSmoothInterpSelection,
    DLCSmoothInterp,
)
from .position_dlc_cohort import DLCSmoothInterpCohortSelection, DLCSmoothInterpCohort
from .position_dlc_centroid import DLCCentroidParams, DLCCentroidSelection, DLCCentroid
from .position_dlc_orient import (
    DLCOrientationParams,
    DLCOrientationSelection,
    DLCOrientation,
)
from .position_dlc_selection import (
    DLCPosSelection,
    DLCPosV1,
    DLCPosVideo,
    DLCPosVideoParams,
    DLCPosVideoSelection,
)
from .position_trodes_position import (
    TrodesPosParams,
    TrodesPosSelection,
    TrodesPosV1,
    TrodesPosVideo,
)
from .dlc_utils import (
    get_dlc_root_data_dir,
    get_dlc_processed_data_dir,
    find_full_path,
    find_root_directory,
    _convert_mp4,
    check_videofile,
    get_video_path,
)
from .dlc_reader import (
    read_yaml,
    save_yaml,
    do_pose_estimation,
)


def schemas():
    return _schemas


_schemas = [
    "position_dlc_project",
    "position_dlc_training",
    "position_dlc_model",
    "position_dlc_pose_estimation",
    "position_dlc_position",
    "position_dlc_cohort",
    "position_dlc_centroid",
    "position_dlc_orient",
    "position_dlc_selection",
    "position_trodes_position",
]

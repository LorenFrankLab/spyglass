from spyglass.utils.dj_mixin import SpyglassMixin  # isort:skip

from spyglass.common.common_behav import (
    PositionIntervalMap,
    PositionSource,
    RawPosition,
    StateScriptFile,
    VideoFile,
    convert_epoch_interval_name_to_position_interval_name,
    get_position_interval_epoch,
)
from spyglass.common.common_device import (
    CameraDevice,
    DataAcquisitionDevice,
    DataAcquisitionDeviceAmplifier,
    DataAcquisitionDeviceSystem,
    Probe,
    ProbeType,
)
from spyglass.common.common_dio import DIOEvents
from spyglass.common.common_ephys import (
    LFP,
    Electrode,
    ElectrodeGroup,
    LFPBand,
    LFPBandSelection,
    LFPSelection,
    Raw,
    SampleCount,
)
from spyglass.common.common_filter import FirFilterParameters
from spyglass.common.common_interval import (
    IntervalList,
    interval_list_censor,
    interval_list_contains,
    interval_list_contains_ind,
    interval_list_excludes,
    interval_list_excludes_ind,
    interval_list_intersect,
    interval_list_union,
    intervals_by_length,
)
from spyglass.common.common_lab import Institution, Lab, LabMember, LabTeam
from spyglass.common.common_nwbfile import AnalysisNwbfile, Nwbfile
from spyglass.common.common_optogenetics import (
    OpticalFiberDevice,
    OpticalFiberImplant,
    OptogeneticProtocol,
    Virus,
    VirusInjection,
)
from spyglass.common.common_position import (
    IntervalLinearizationSelection,
    IntervalLinearizedPosition,
    IntervalPositionInfo,
    IntervalPositionInfoSelection,
    LinearizationParameters,
    PositionInfoParameters,
    PositionVideo,
    TrackGraph,
)
from spyglass.common.common_region import BrainRegion
from spyglass.common.common_sensors import SensorData
from spyglass.common.common_session import Session
from spyglass.common.common_subject import Subject
from spyglass.common.common_task import Task, TaskEpoch
from spyglass.common.common_user import UserEnvironment
from spyglass.common.populate_all_common import populate_all_common
from spyglass.common.prepopulate import populate_from_yaml, prepopulate_default
from spyglass.settings import prepopulate
from spyglass.utils.nwb_helper_fn import (
    close_nwb_files,
    estimate_sampling_rate,
    get_data_interface,
    get_electrode_indices,
    get_nwb_file,
    get_raw_eseries,
    get_valid_intervals,
)

__all__ = [
    "LabTeam",
    "LinearizationParameters",
    "Nwbfile",
    "OpticalFiberDevice",
    "OpticalFiberImplant",
    "OptogeneticProtocol",
    "Virus",
    "VirusInjection",
    "PositionInfoParameters",
    "PositionIntervalMap",
    "PositionSource",
    "PositionVideo",
    "Probe",
    "ProbeType",
    "Raw",
    "RawPosition",
    "SampleCount",
    "SensorData",
    "Session",
    "StateScriptFile",
    "Subject",
    "Task",
    "TaskEpoch",
    "TrackGraph",
    "UserEnvironment",
    "VideoFile",
    "close_nwb_files",
    "convert_epoch_interval_name_to_position_interval_name",
    "estimate_sampling_rate",
    "get_data_interface",
    "get_electrode_indices",
    "get_nwb_file",
    "get_position_interval_epoch",
    "get_raw_eseries",
    "get_valid_intervals",
    "interval_list_censor",
    "interval_list_contains",
    "interval_list_contains_ind",
    "interval_list_excludes",
    "interval_list_excludes_ind",
    "interval_list_intersect",
    "interval_list_union",
    "intervals_by_length",
    "populate_all_common",
    "populate_from_yaml",
    "prepopulate",
    "prepopulate_default",
]


if prepopulate:
    prepopulate_default()

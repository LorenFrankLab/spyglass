import os

import spyglass as sg

from ..utils.nwb_helper_fn import (
    close_nwb_files,
    estimate_sampling_rate,
    get_data_interface,
    get_electrode_indices,
    get_nwb_file,
    get_raw_eseries,
    get_valid_intervals,
)
from .common_behav import (
    PositionSource,
    RawPosition,
    StateScriptFile,
    VideoFile,
    PositionIntervalMap,
    convert_epoch_interval_name_to_position_interval_name,
)
from .common_device import (
    CameraDevice,
    DataAcquisitionDevice,
    DataAcquisitionDeviceAmplifier,
    DataAcquisitionDeviceSystem,
    Probe,
    ProbeType,
)
from .common_dio import DIOEvents
from .common_ephys import (
    LFP,
    Electrode,
    ElectrodeGroup,
    LFPBand,
    LFPBandSelection,
    LFPSelection,
    Raw,
    SampleCount,
)
from .common_filter import FirFilterParameters
from .common_interval import (
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
from .common_lab import Institution, Lab, LabMember, LabTeam
from .common_nwbfile import (
    AnalysisNwbfile,
    AnalysisNwbfileKachery,
    Nwbfile,
    NwbfileKachery,
)
from .common_position import (
    IntervalLinearizationSelection,
    IntervalLinearizedPosition,
    IntervalPositionInfo,
    IntervalPositionInfoSelection,
    LinearizationParameters,
    PositionInfoParameters,
    PositionVideo,
    TrackGraph,
)
from .common_region import BrainRegion
from .common_sensors import SensorData
from .common_session import Session, SessionGroup
from .common_subject import Subject
from .common_task import Task, TaskEpoch
from .populate_all_common import populate_all_common
from .prepopulate import populate_from_yaml, prepopulate_default

if sg.config["prepopulate"]:
    prepopulate_default()

import os
import spyglass as sg

from .common_backup import CuratedSpikeSortingBackUp, SpikeSortingBackUp
from .common_behav import PositionSource, RawPosition, StateScriptFile, VideoFile
from .common_device import (
    CameraDevice,
    DataAcquisitionDevice,
    DataAcquisitionDeviceSystem,
    DataAcquisitionDeviceAmplifier,
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
from .common_filter import FirFilter
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
from ..utils.nwb_helper_fn import (
    close_nwb_files,
    estimate_sampling_rate,
    get_data_interface,
    get_electrode_indices,
    get_nwb_file,
    get_raw_eseries,
    get_valid_intervals,
)
from .populate_all_common import populate_all_common

from .prepopulate import prepopulate_default, populate_from_yaml

if sg.config["prepopulate"] == True:
    prepopulate_default()

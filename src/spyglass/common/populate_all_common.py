from pathlib import Path
from typing import Dict, List, Union

import datajoint as dj
import yaml

from spyglass.common.common_behav import (
    PositionSource,
    RawPosition,
    RawCompassDirection,
    StateScriptFile,
    VideoFile,
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
    Electrode,
    ElectrodeGroup,
    Raw,
    SampleCount,
)
from spyglass.common.common_interval import IntervalList
from spyglass.common.common_lab import Institution, Lab, LabMember, LabTeam
from spyglass.common.common_nwbfile import Nwbfile
from spyglass.common.common_optogenetics import (
    OpticalFiberDevice,
    OpticalFiberImplant,
    OptogeneticProtocol,
    Virus,
    VirusInjection,
)
from spyglass.common.common_sensors import SensorData
from spyglass.common.common_session import Session
from spyglass.common.common_subject import Subject
from spyglass.common.common_task import Task, TaskEpoch
from spyglass.settings import base_dir
from spyglass.utils.dj_helper_fn import declare_all_merge_tables
from spyglass.utils.nwb_helper_fn import get_config


def _plan_only(nwb_file_name: str):
    """Plan a file, stage the plan, report, and write no data table.

    Shared by `dry_run`, which stops here, and the real run, which hands the
    result to `insert_plan`, so what a dry run reports is what a real run
    will do.

    Parameters
    ----------
    nwb_file_name : str
        The copy file registered in Nwbfile.

    Returns
    -------
    IngestionPlan
        Falsy when nothing blocks it; printable as the report.
    """
    from spyglass.common.common_usage import IngestionPlanLog
    from spyglass.data_import.planner import plan_nwbfile

    # The sidecar config sits beside the file, and locating it goes through
    # `Nwbfile.get_abs_path`, which raises for a file with no row. An
    # unregistered file has no plan to make, so skip the lookup and let the
    # planner report `file_not_registered`. Reporting is the whole contract
    # here, and raising from a dry run breaks it.
    registered = bool(Nwbfile & {"nwb_file_name": nwb_file_name})
    config = (
        merged_config(nwb_file_name, lab_config()) if registered else dict()
    )

    plan = plan_nwbfile(nwb_file_name, config=config)
    IngestionPlanLog().stage(plan)
    plan.log_report()

    return plan


def _insert_from_plan(
    nwb_file_name: str,
    allow_partial: bool = False,
):
    """Plan the whole file, then insert what the plan worked out.

    The first pass writes nothing, so every problem is reported before any
    table is touched. The second re-derives nothing: it writes the rows the
    first already checked, skipping those already stored.

    Parameters
    ----------
    nwb_file_name : str
        The copy file registered in Nwbfile.
    allow_partial : bool, optional
        Insert the tables that planned cleanly even though others did not.

    Returns
    -------
    IngestionPlan
        Falsy when everything asked for was inserted.
    """
    from spyglass.data_import.planner import insert_plan

    return insert_plan(_plan_only(nwb_file_name), allow_partial=allow_partial)


def ingestion_tables() -> Dict[str, List[dj.Table]]:
    """Return the tables an ingestion touches, in two categories.

    - `"ingest"`: tables that **parse the file**, parents before children,
      so a cross-reference resolves against the plan. The order is written out
      rather than derived; `tests/common/test_ingestion_table_order.py` checks
      it against DataJoint's graph, so a table added in the wrong place fails
      a test rather than an ingestion.
    - `"targets"`: tables that **receive rows without parsing anything**.
      Order is irrelevant: nothing parses them.

    Returns
    -------
    dict
        `{"ingest": [...], "targets": [...]}`.
    """
    from spyglass.lfp.lfp_electrode import LFPElectrodeGroup
    from spyglass.lfp.lfp_imported import ImportedLFP
    from spyglass.position.v1.imported_pose import ImportedPose
    from spyglass.spikesorting.imported import ImportedSpikeSorting

    ingest = [
        # no parents among these
        CameraDevice,
        DataAcquisitionDeviceAmplifier,
        DataAcquisitionDeviceSystem,
        Institution,
        Lab,
        LabMember,
        LabTeam,
        OpticalFiberDevice,
        ProbeType,
        Subject,
        Virus,
        # devices and probes
        DataAcquisitionDevice,  # -> DataAcq*Amplifier, DataAcq*System
        Probe,  # -> ProbeType
        Probe.Shank,  # -> Probe
        Probe.Electrode,  # -> Probe.Shank
        # the session, and what hangs from it
        Session,  # -> Subject, Institution, Lab
        Session.Experimenter,  # -> Session, LabMember
        Session.DataAcquisitionDevice,  # -> Session, DataAcq*Device
        VirusInjection,  # -> Session, Virus
        ElectrodeGroup,  # -> Session
        ImportedSpikeSorting,  # -> Session
        IntervalList,  # -> Session
        OpticalFiberImplant,  # -> Session, OpticalFiberDevice
        PositionSource,  # -> Session, IntervalList
        Raw,  # -> Session, IntervalList
        RawCompassDirection,  # -> Session, IntervalList
        SampleCount,  # -> Session
        SensorData,  # -> Session, IntervalList
        TaskEpoch,  # -> Session, Task, CameraDevice, IntervalList
        VideoFile,  # -> TaskEpoch
        # last: depend on the above
        DIOEvents,  # -> Session, IntervalList
        Electrode,  # -> ElectrodeGroup, Probe.Electrode
        ImportedLFP,  # -> LFPElectrodeGroup, IntervalList
        ImportedPose,  # -> IntervalList
        OptogeneticProtocol,  # -> TaskEpoch
        StateScriptFile,  # -> TaskEpoch
        # NwbfileKachery, # Not used by default
    ]

    # Emitted by a table above, never parsed themselves. Every one is a
    # parent, child or part of something in `ingest`, since a table can only
    # emit rows for something it is related to. They are listed rather than
    # derived so the set is readable and reviewable.
    targets = [
        Task,  # from TaskEpoch
        RawPosition,  # from PositionSource
        RawPosition.PosObject,  # from PositionSource
        LFPElectrodeGroup,  # from ImportedLFP
        LFPElectrodeGroup.LFPElectrode,  # from ImportedLFP
    ]

    return {"ingest": ingest, "targets": targets}


def lab_config() -> dict:
    """Return the lab-wide config entries, from `entries.yaml` in base_dir.

    Returns
    -------
    dict
        `{TableName: [rows]}`, empty when there is no such file.
    """
    entries_path = Path(base_dir) / "entries.yaml"
    if not entries_path.exists():
        return dict()

    with open(entries_path, "r") as stream:
        # yaml.safe_load returns None for an empty file
        return yaml.safe_load(stream) or dict()


def merged_config(nwb_file_name: str, config: dict = None) -> dict:
    """Return the config a file is ingested with, sidecar over defaults.

    Both sources share the `{TableName: [rows]}` shape, and each table is
    handed the whole mapping. Slicing it per table hands `Electrode` a row
    list where it expects a dict.

    Parameters
    ----------
    nwb_file_name : str
        The file whose sidecar config to read.
    config : dict, optional
        Defaults the sidecar overrides. Default None, no defaults.

    Returns
    -------
    dict
    """
    file_config = (
        get_config(
            Nwbfile.get_abs_path(nwb_file_name),
            calling_table="populate_all_common",
        )
        or dict()
    )
    return {**(config or dict()), **file_config}


def populate_all_common(
    nwb_file_name,
    dry_run=False,
    allow_partial=False,
) -> Union[List, None]:
    """Insert all common tables for a given NWB file.

    Checks the whole file, reports every problem, then writes what was
    checked in one transaction. Nothing is written unless the plan is clean,
    so a file does not half-ingest before failing, and a re-attempt skips
    what is already stored.

    Parameters
    ----------
    nwb_file_name : str
        The name of the NWB file to populate.
    dry_run : bool, optional
        Plan the file and return the report without inserting anything.
        Default False.
    allow_partial : bool, optional
        Insert the tables that planned cleanly even though others did not.
        Default False: a blocking problem inserts nothing, so a half-ingested
        file is a choice rather than an accident.

    Returns
    -------
    IngestionPlan
        Falsy when nothing blocks it, iterable over its blocking problems,
        and printable as the report.
    """
    _ = declare_all_merge_tables()

    if dry_run:
        return _plan_only(nwb_file_name)

    return _insert_from_plan(nwb_file_name, allow_partial=allow_partial)

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

    The plan pass reads the database but never writes to it, so every problem
    in the file is reported at once rather than one per table that happened to
    fail, and nothing is half-ingested afterwards. The plan is staged so a
    later attempt can see what this one worked out.

    Shared by `dry_run`, which stops here, and the real run, which hands the
    result to `insert_plan`. One plan pass serves both, so what a dry run
    reports is what a real run will do.

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
    # planner report `file_not_registered` -- reporting is the whole contract
    # here, and raising from a dry run breaks it.
    registered = bool(Nwbfile & {"nwb_file_name": nwb_file_name})
    config = (
        merged_config(nwb_file_name, lab_config()) if registered else dict()
    )

    plan = plan_nwbfile(nwb_file_name, config=config)
    IngestionPlanLog().stage(plan)
    plan.report()

    return plan


def _insert_from_plan(
    nwb_file_name: str,
    raise_err: bool = False,
    on_divergence: str = "report",
    allow_partial: bool = False,
    rollback_on_miss: bool = False,
):
    """Plan the whole file, then insert what the plan worked out.

    The point of the two passes: the first writes nothing, so every problem in
    the file is reported before any table is touched, and a file that cannot be
    ingested cleanly is not half-ingested first. The second re-derives nothing
    -- it writes the rows the first pass already checked, skipping those already
    stored, and closes the staging area on success.

    Parameters
    ----------
    nwb_file_name : str
        The copy file registered in Nwbfile.
    raise_err : bool, optional
        Raise at the end if anything blocked. Default False, returning the plan
        for the caller to test. Nothing raises mid-pass either way.
    on_divergence : str, optional
        `report` to warn and keep the stored rows, `raise` to decline.
        Default `report`.
    allow_partial : bool, optional
        Insert the tables that planned cleanly even though others did not.
    rollback_on_miss : bool, optional
        Delete the session if a validated plan fails halfway.

    Returns
    -------
    IngestionPlan
        Falsy when everything asked for was inserted.

    Raises
    ------
    ValueError
        When `raise_err` and the plan blocked, with the report attached.
    """
    from spyglass.data_import.planner import insert_plan

    plan = _plan_only(nwb_file_name)

    result = insert_plan(
        plan,
        allow_partial=allow_partial,
        on_divergence=on_divergence,
        rollback_on_miss=rollback_on_miss,
    )

    if raise_err and result:
        raise ValueError(
            f"Ingestion of {nwb_file_name} did not complete:\n"
            + result.report(log=False)
        )

    return result


def ingestion_tables() -> Dict[str, List[dj.Table]]:
    """Return the tables an ingestion touches, in two categories.

    One declared set, shared by the inserter and the planner:

    - `"ingest"` -- tables that **parse the file**, parents before children.
      The planner walks these in order, so each table's parents are planned
      before it and a cross-reference resolves against the plan.
    - `"targets"` -- tables that **receive rows without parsing anything**.
      `TaskEpoch` emits `Task`, `PositionSource` emits `RawPosition`,
      `ImportedLFP` emits `LFPElectrodeGroup`. They are declared here rather
      than discovered by walking the foreign-key graph, so the set a plan can
      name is written down in one place. Order is irrelevant: nothing parses
      them.

    Both lists hold real table classes, so a target carries its own heading,
    parents and mixin -- no name-to-class registry, and no `FreeTable` stand-in.

    The `"ingest"` order is written out rather than derived: the schema is
    fixed at import time, so sorting on every call costs ~0.17s to rediscover
    an answer that cannot change. `tests/common/test_ingestion_table_order.py`
    checks it against DataJoint's own graph, so a table added in the wrong
    place fails a test rather than an ingestion.

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
    # parent, child or part of something in `ingest` -- a table can only
    # emit rows for something it is related to -- but they are listed rather
    # than derived so the set is readable and reviewable.
    targets = [
        Task,  # from TaskEpoch
        RawPosition,  # from PositionSource
        RawPosition.PosObject,  # from PositionSource
        LFPElectrodeGroup,  # from ImportedLFP
        LFPElectrodeGroup.LFPElectrode,  # from ImportedLFP
    ]

    return {"ingest": ingest, "targets": targets}


def ingestion_table_list() -> List[dj.Table]:
    """Return the tables that parse a file, in dependency order.

    Deprecated shim for `ingestion_tables()["ingest"]`; kept because the
    ordered parse list is what most callers want and the name says so.

    Returns
    -------
    list
        SpyglassIngestion table classes, in dependency order.
    """
    return ingestion_tables()["ingest"]


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

    Entries may be declared in a `_spyglass_config.yaml` beside the NWB file
    as well as in `entries.yaml`. Both share the `{TableName: [rows]}` shape
    that `generate_entries_from_config` indexes by name, so each table is
    handed the whole merged mapping -- a per-table lookup would yield a row
    list. The file's own config wins: `entries.yaml` holds lab-wide defaults,
    while the sidecar describes this session.

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
    rollback_on_fail=False,
    raise_err=False,
    dry_run=False,
    on_divergence="report",
    allow_partial=False,
) -> Union[List, None]:
    """Insert all common tables for a given NWB file.

    Checks the whole file, reports every problem, then writes what was
    checked. Nothing is written unless the plan is clean, so a file does not
    half-ingest before failing, and a re-attempt skips what is already stored.

    Parameters
    ----------
    nwb_file_name : str
        The name of the NWB file to populate.
    rollback_on_fail : bool, optional
        Delete the session if a validated plan fails halfway -- a
        `planner_miss`, the one case a rollback is still for. Default False.
    raise_err : bool, optional
        Raise at the end if anything blocked. Default False, returning the
        plan for the caller to test. Nothing raises mid-pass either way.
    dry_run : bool, optional
        Plan the file and return the report without inserting anything.
        Default False.
    on_divergence : str, optional
        What to do when the file disagrees with a stored row: `report`
        keeps the stored rows, warns, and inserts the rest; `raise` declines.
        Default `report`. Nothing prompts (D7).
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

    return _insert_from_plan(
        nwb_file_name,
        raise_err=raise_err,
        on_divergence=on_divergence,
        allow_partial=allow_partial,
        rollback_on_miss=rollback_on_fail,
    )

from pathlib import Path
from typing import List, Union

import datajoint as dj
import yaml

from spyglass.common.common_behav import (
    PositionSource,
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
from spyglass.common.common_task import TaskEpoch
from spyglass.common.common_usage import InsertError
from spyglass.settings import base_dir
from spyglass.utils import logger
from spyglass.utils.dj_helper_fn import declare_all_merge_tables
from spyglass.utils.nwb_helper_fn import get_config


def log_insert_error(
    table: str, err: Exception, error_constants: dict = None
) -> None:
    """Log a given error to the InsertError table.

    Deprecated in favour of `IngestionPlanLog`, which records a whole file's
    problems together with the entries they blocked, rather than one row per
    exception with no memory of what was already staged. `InsertError`
    remains declared and written to so existing queries keep working; it is
    no longer where new work should look.

    Parameters
    ----------
    table : str
        The table name where the error occurred.
    err : Exception
        The exception that was raised.
    error_constants : dict, optional
        Dictionary with keys for dj_user, connection_id, and nwb_file_name.
        Defaults to checking dj.conn and using "Unknown" for nwb_file_name.
    """
    from spyglass.common.common_usage import ActivityLog

    ActivityLog().deprecate_log(
        name="InsertError, written by populate_all_common",
        alt="IngestionPlanLog, which stages entries alongside their problems",
    )

    if error_constants is None:
        error_constants = dict(
            dj_user=dj.config["database.user"],
            connection_id=dj.conn().connection_id,
            nwb_file_name="Unknown",
        )
    InsertError.insert1(
        dict(
            **error_constants,
            table=table.__name__,
            error_type=type(err).__name__,
            error_message=str(err)[:255],  # limit to 255 chars
            error_raw=str(err),
        )
    )


def _plan_only(nwb_file_name: str):
    """Plan a file, stage the plan, report, and write no data table.

    The plan pass reads the database but never writes to it, so every problem
    in the file is reported at once rather than one per table that happened to
    fail, and nothing is half-ingested afterwards. The plan is staged so a
    later attempt can see what this one worked out.

    Shared by `dry_run`, which stops here, and `use_plan`, which hands the
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
    on_divergence: str = "interactive",
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
        `interactive`, `accept` or `raise`. Default `interactive`.
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


def ingestion_table_list() -> List[dj.Table]:
    """Return every table ingested from an NWB file, parents before children.

    One declared set, shared by the inserter and the planner. The order is
    written out rather than derived: the schema is fixed at import time, so
    sorting it on every call costs ~0.17s to rediscover an answer that cannot
    change. `test_ingestion_table_list_is_dependency_ordered` checks the
    order against DataJoint's foreign-key graph instead, so a table added in
    the wrong place fails a test rather than an ingestion.

    Returns
    -------
    list
        SpyglassIngestion table classes, in dependency order.
    """
    from spyglass.lfp.lfp_imported import ImportedLFP
    from spyglass.position.v1.imported_pose import ImportedPose
    from spyglass.spikesorting.imported import ImportedSpikeSorting

    return [
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


def single_transaction_make(
    tables: List[dj.Table],
    nwb_file_name: str,
    raise_err: bool = False,
    error_constants: dict = None,
    config: dict = None,
):
    """Ingest each table from the NWB file, inside one transaction.

    Every table here is a SpyglassIngestion table, so each parses the file
    once via `insert_from_nwbfile` rather than running `make` per key_source
    key. Failures are logged per table unless `raise_err` is set.
    """
    merged = merged_config(nwb_file_name, config)

    with Nwbfile._safe_context():
        for table in tables:
            try:
                table().insert_from_nwbfile(nwb_file_name, config=merged)
            except Exception as err:
                if raise_err:
                    raise err
                log_insert_error(
                    table=table, err=err, error_constants=error_constants
                )


def populate_all_common(
    nwb_file_name,
    rollback_on_fail=False,
    raise_err=False,
    dry_run=False,
    use_plan=False,
    on_divergence="interactive",
    allow_partial=False,
) -> Union[List, None]:
    """Insert all common tables for a given NWB file.

    Parameters
    ----------
    nwb_file_name : str
        The name of the NWB file to populate.
    rollback_on_fail : bool, optional
        If True, will delete the Session entry if any errors occur.
        Defaults to False. Deprecated: planning a file reports every problem
        before anything is written, so there is nothing to undo. A rollback
        now belongs only to a `planner_miss`, where a validated plan failed
        halfway — see `insert_plan(rollback_on_miss=True)`, which is what this
        maps to when `use_plan` is set.
    raise_err : bool, optional
        If True, will raise any errors that occur during population.
        Defaults to False. This will prevent any rollback from occurring.
        With `use_plan`, nothing raises during the pass — every failure becomes
        a problem on the plan — so this raises at the end if anything blocked.
    dry_run : bool, optional
        If True, plan the file and return the report without inserting
        anything. Every problem is reported at once rather than one per
        failed table, and no data table is written — see `IngestionPlan`.
        Default False.
    use_plan : bool, optional
        If True, insert from the plan a dry run would have reported: check the
        whole file first, then write what was checked, skipping what is already
        stored. Nothing is written unless the plan is clean, so a file no longer
        half-ingests before failing. Default False, taking the per-table path
        that stops at each failure and records it in `InsertError`.
    on_divergence : str, optional
        With `use_plan`, what to do when the file disagrees with a stored row:
        `interactive` asks once, `accept` keeps the stored value and inserts the
        rest, `raise` declines. Default `interactive`. Ignored otherwise, where
        divergence is still resolved mid-transaction per table.
    allow_partial : bool, optional
        With `use_plan`, insert the tables that planned cleanly even though
        others did not. Default False: a blocking problem inserts nothing, so a
        half-ingested file is a choice rather than an accident.

    Returns
    -------
    IngestionPlan or List or None
        With `dry_run` or `use_plan`, the plan: falsy when nothing blocks it,
        iterable over its blocking problems, and printable as the report.
        Otherwise a list of keys for InsertError entries if any errors occurred.

    Notes
    -----
    InsertError rows logged by an earlier attempt at the same file, under the
    same user and connection, are cleared before population starts, so the
    returned list only ever describes the current attempt. Neither a dry run
    nor a planned run reads or writes them.
    """
    from spyglass.lfp.lfp_imported import ImportedLFP
    from spyglass.position.v1.imported_pose import ImportedPose
    from spyglass.spikesorting.imported import ImportedSpikeSorting

    _ = declare_all_merge_tables()

    if dry_run:
        return _plan_only(nwb_file_name)

    if use_plan:
        return _insert_from_plan(
            nwb_file_name,
            raise_err=raise_err,
            on_divergence=on_divergence,
            allow_partial=allow_partial,
            rollback_on_miss=rollback_on_fail,
        )

    error_constants = dict(
        dj_user=dj.config["database.user"],
        connection_id=dj.conn().connection_id,
        nwb_file_name=nwb_file_name,
    )

    # Drop errors logged by an earlier attempt at this same file, user, and
    # connection. Without this, the check below reports stale failures and can
    # roll back an otherwise clean ingestion. See issue #1497. InsertError has
    # no dependent tables, so delete_quick is safe here.
    (InsertError & error_constants).delete_quick()

    table_lists: List[List[dj.Table]] = [
        # Tables that can be inserted in a single transaction
        [
            Institution,  # Parent node
            Lab,  # Parent node
            LabMember,  # Parent node
            LabTeam,  # Parent node
            Subject,  # Parent node
            CameraDevice,  # Parent node
            ProbeType,  # Parent node
            DataAcquisitionDeviceAmplifier,  # Parent node
            DataAcquisitionDeviceSystem,  # Parent node
            DataAcquisitionDevice,  # Depends on DataAcq*Amp, DataAcq*Sys
            OpticalFiberDevice,  # Parent node
            Virus,  # Parent node
        ],
        [
            Probe,  # Depends on ProbeType, DataAcquisitionDevice
            Probe.Shank,  # Depends on Probe
            Probe.Electrode,  # Depends on Probe
            Session,  # Depends on Subject, Institution, Lab
            Session.Experimenter,  # Depends on Session
            Session.DataAcquisitionDevice,  # Depends on Sess, DataAcq*Device
            ElectrodeGroup,  # Depends on Session
            Raw,  # Depends on Session
            SampleCount,  # Depends on Session
            DIOEvents,  # Depends on Session
            ImportedSpikeSorting,  # Depends on Session
            SensorData,  # Depends on Session
            IntervalList,  # Depends on Session
            TaskEpoch,  # Depends on Session, Task, CamearaDevice, IntervalList
            # NwbfileKachery, # Not used by default
        ],
        [  # Tables that depend on above transaction
            Electrode,  # Depends on ElectrodeGroup
            PositionSource,  # Depends on Session. Also fills RawPosition
            RawCompassDirection,  # Depends on Session
            VideoFile,  # Depends on TaskEpoch
            StateScriptFile,  # Depends on TaskEpoch
            ImportedPose,  # Depends on Session
            ImportedLFP,  # Depends on ElectrodeGroup
            VirusInjection,  # Depends on Session
            OpticalFiberImplant,  # Depends on Session and OpticalFiberDevice
            OptogeneticProtocol,  # Depends on Session and TaskEpoch
        ],
    ]

    config = lab_config()

    for tables in table_lists:
        single_transaction_make(
            tables=tables,
            nwb_file_name=nwb_file_name,
            raise_err=raise_err,
            error_constants=error_constants,
            config=config,
        )

    err_query = InsertError & error_constants
    nwbfile_query = Nwbfile & {"nwb_file_name": nwb_file_name}

    if err_query and nwbfile_query and rollback_on_fail:
        from spyglass.common.common_usage import ActivityLog

        ActivityLog().deprecate_log(
            name="rollback_on_fail, the blanket undo after a failed ingest",
            alt="plan the file first; insert_plan(rollback_on_miss=True) "
            + "covers the one case a rollback is still for",
        )
        logger.error(f"Rolling back population for {nwb_file_name}...")
        # Should this be safemode=False to prevent confirmation prompt?
        nwbfile_query.super_delete(warn=False)

    if err_query:
        err_tables = err_query.fetch("table")
        logger.error(
            f"Errors occurred during population for {nwb_file_name}:\n\t"
            + f"Failed tables {err_tables}\n\t"
            + "See common_usage.InsertError for more details"
        )
        return err_query.fetch("KEY")

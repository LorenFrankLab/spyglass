from pathlib import Path
from typing import List, Union

import datajoint as dj
import yaml
from datajoint.utils import to_camel_case

from spyglass.common.common_behav import (
    PositionSource,
    RawPosition,
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
    OpticalFiberImplant,
    OptogeneticProtocol,
    VirusInjection,
)
from spyglass.common.common_sensors import SensorData
from spyglass.common.common_session import Session
from spyglass.common.common_subject import Subject
from spyglass.common.common_task import TaskEpoch
from spyglass.common.common_usage import InsertError
from spyglass.settings import base_dir
from spyglass.utils import SpyglassIngestion, logger
from spyglass.utils.dj_helper_fn import declare_all_merge_tables


def log_insert_error(
    table: str, err: Exception, error_constants: dict = None
) -> None:
    """Log a given error to the InsertError table.

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


def _get_config_name(table_obj):
    """Given a table object, return its config name for entries.yaml

    Returns Master.Part for part tables, otherwise just the table name.
    """
    if hasattr(table_obj, "_master"):
        return f"{table_obj._master.__name__}.{table_obj.__class__.__name__}"
    return table_obj.__class__.__name__


def single_transaction_make(
    tables: List[dj.Table],
    nwb_file_name: str,
    raise_err: bool = False,
    error_constants: dict = None,
    config: dict = None,
):
    """For each table, run the `make` method directly instead of `populate`.

    Requires `allow_direct_insert` set to True within each method. Uses
    nwb_file_name search table key_source for relevant key. Currently assumes
    all tables will have exactly one key_source entry per nwb file.
    """

    file_restr = {"nwb_file_name": nwb_file_name}
    with Nwbfile._safe_context():
        for table in tables:
            config_name = _get_config_name(table())
            table_config = config.get(config_name, dict())

            if isinstance(table(), SpyglassIngestion):
                try:
                    table().insert_from_nwbfile(
                        nwb_file_name, config=table_config
                    )
                except Exception as err:
                    if raise_err:
                        raise err
                    log_insert_error(
                        table=table, err=err, error_constants=error_constants
                    )
                continue

            # If imported/computed table, get key from key_source
            logger.info(f"Populating {table.__name__}...")
            key_source = getattr(table, "key_source", None)
            if key_source is None:  # Generate key from parents
                parents = table.parents(as_objects=True)
                key_source = parents[0].proj()
                for parent in parents[1:]:
                    key_source *= parent.proj()

            table_name = to_camel_case(table.table_name)
            if table_name == "PositionSource":
                # PositionSource only uses nwb_file_name - full calls redundant
                key_source = dj.U("nwb_file_name") & key_source
            if table_name in [
                "ImportedPose",
                "ImportedLFP",
                "VirusInjection",
                "OpticalFiberImplant",
            ]:
                key_source = Nwbfile()

            query = key_source & file_restr
            for pop_key in query.fetch("KEY"):
                try:
                    table().make(pop_key)
                except Exception as err:
                    if raise_err:
                        raise err
                    log_insert_error(
                        table=table, err=err, error_constants=error_constants
                    )
            # if table_name in ["TaskEpoch"]:
            #     __import__("pdb").set_trace()


def populate_all_common(
    nwb_file_name, rollback_on_fail=False, raise_err=False
) -> Union[List, None]:
    """Insert all common tables for a given NWB file.

    Parameters
    ----------
    nwb_file_name : str
        The name of the NWB file to populate.
    rollback_on_fail : bool, optional
        If True, will delete the Session entry if any errors occur.
        Defaults to False.
    raise_err : bool, optional
        If True, will raise any errors that occur during population.
        Defaults to False. This will prevent any rollback from occurring.

    Returns
    -------
    List
        A list of keys for InsertError entries if any errors occurred.
    """
    from spyglass.lfp.lfp_imported import ImportedLFP
    from spyglass.position.v1.imported_pose import ImportedPose
    from spyglass.spikesorting.imported import ImportedSpikeSorting

    declare_all_merge_tables()

    error_constants = dict(
        dj_user=dj.config["database.user"],
        connection_id=dj.conn().connection_id,
        nwb_file_name=nwb_file_name,
    )

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
            PositionSource,  # Depends on Session
            VideoFile,  # Depends on TaskEpoch
            StateScriptFile,  # Depends on TaskEpoch
            ImportedPose,  # Depends on Session
            ImportedLFP,  # Depends on ElectrodeGroup
            ImportedSpikeSorting,  # Depends on Session
            VirusInjection,  # Depends on Session
            OpticalFiberImplant,  # Depends on Session
            OptogeneticProtocol,  # Depends on Session and TaskEpoch
        ],
        [
            RawPosition,  # Depends on PositionSource
        ],
    ]

    config = dict()
    entries_path = Path(base_dir) / "entries.yaml"
    if entries_path.exists():
        with open(f"{base_dir}/entries.yaml", "r") as stream:
            config = yaml.safe_load(stream)

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

from typing import List, Union

import datajoint as dj

from spyglass.common.common_behav import (
    PositionSource,
    RawPosition,
    StateScriptFile,
    VideoFile,
)
from spyglass.common.common_dio import DIOEvents
from spyglass.common.common_ephys import (
    Electrode,
    ElectrodeGroup,
    Raw,
    SampleCount,
)
from spyglass.common.common_nwbfile import Nwbfile
from spyglass.common.common_session import Session
from spyglass.common.common_task import TaskEpoch
from spyglass.common.common_usage import InsertError
from spyglass.utils import logger


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
            error_message=str(err),
            error_raw=str(err),
        )
    )


def single_transaction_make(
    tables: List[dj.Table],
    nwb_file_name: str,
    raise_err: bool = False,
    error_constants: dict = None,
):
    """For each table, run the `make` method directly instead of `populate`.

    Requires `allow_direct_insert` set to True within each method. Uses
    nwb_file_name search table key_source for relevant key. Currently assumes
    all tables will have exactly one key_source entry per nwb file.
    """
    file_restr = {"nwb_file_name": nwb_file_name}
    with Nwbfile.connection.transaction:
        for table in tables:
            logger.info(f"Populating {table.__name__}...")

            # If imported/computed table, get key from key_source
            key_source = getattr(table, "key_source", None)
            if key_source is None:  # Generate key from parents
                parents = table.parents(as_objects=True)
                key_source = parents[0].proj()
                for parent in parents[1:]:
                    key_source *= parent.proj()

            if table.__name__ == "PositionSource":
                # PositionSource only uses nwb_file_name - full calls redundant
                key_source = dj.U("nwb_file_name") & key_source

            for pop_key in (key_source & file_restr).fetch("KEY"):
                try:
                    table().make(pop_key)
                except Exception as err:
                    if raise_err:
                        raise err
                    log_insert_error(
                        table=table, err=err, error_constants=error_constants
                    )


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
    from spyglass.spikesorting.imported import ImportedSpikeSorting

    error_constants = dict(
        dj_user=dj.config["database.user"],
        connection_id=dj.conn().connection_id,
        nwb_file_name=nwb_file_name,
    )

    table_lists = [
        [  # Tables that can be inserted in a single transaction
            Session,
            ElectrodeGroup,  # Depends on Session
            Raw,  # Depends on Session
            SampleCount,  # Depends on Session
            DIOEvents,  # Depends on Session
            TaskEpoch,  # Depends on Session
            ImportedSpikeSorting,  # Depends on Session
            # NwbfileKachery, # Not used by default
            # SensorData, # Not used by default. Generates large files
        ],
        [  # Tables that depend on above transaction
            Electrode,  # Depends on ElectrodeGroup
            PositionSource,  # Depends on Session
            VideoFile,  # Depends on TaskEpoch
            StateScriptFile,  # Depends on TaskEpoch
        ],
        [
            RawPosition,  # Depends on PositionSource
        ],
    ]

    for tables in table_lists:
        single_transaction_make(
            tables=tables,
            nwb_file_name=nwb_file_name,
            raise_err=raise_err,
            error_constants=error_constants,
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

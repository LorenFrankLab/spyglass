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
    permissive_insert: bool = True,
    error_constants: dict = None,
):
    """For each table, run the `_no_transaction_make` method.

    Requires `allow_direct_insert` set to True within each method. Uses
    nwb_file_name search table key_source for relevant key. Currently assumes
    all tables will have exactly one key_source entry per nwb file.
    """
    file_restr = {"nwb_file_name": nwb_file_name}
    with Nwbfile.connection.transaction:
        for table in tables:
            logger.info(f"Populating {table.__name__}...")
            pop_key = (table.key_source & file_restr).fetch1("KEY")
            try:
                table()._no_transaction_make(pop_key)
            except Exception as err:
                if not permissive_insert:
                    raise err
                log_insert_error(
                    table=table, err=err, error_constants=error_constants
                )


def populate_all_common(
    nwb_file_name, permissive_insert=True
) -> Union[List, None]:
    """Insert all common tables for a given NWB file.

    Parameters
    ----------
    nwb_file_name : str
        The name of the NWB file to populate.
    permissive_insert : bool
        If True, will insert an error into InsertError and continue if an error
        is encountered. If False, will raise the error.

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

    tables_1 = [
        Session,
        # NwbfileKachery, # Not used by default
        ElectrodeGroup,
        Electrode,
        Raw,
        SampleCount,
        DIOEvents,
        # SensorData, # Not used by default. Generates large files
        TaskEpoch,
        StateScriptFile,
        VideoFile,
        PositionSource,
    ]

    tables_2 = [  # Run separately so that transaction above concludes first
        RawPosition,  # Depends on PositionSource
        ImportedSpikeSorting,  # Depends on Session
    ]

    # Two transactions. If fail in tables_2, tables_1 will still be inserted
    single_transaction_make(
        tables_1, nwb_file_name, permissive_insert, error_constants
    )
    single_transaction_make(
        tables_2, nwb_file_name, permissive_insert, error_constants
    )

    query = InsertError & error_constants
    if query:
        err_tables = query.fetch("table")
        logger.error(
            f"Errors occurred during population for {nwb_file_name}:\n\t"
            + f"Failed tables {err_tables}\n\t"
            + "See common_usage.InsertError for more details"
        )
        return query.fetch("KEY")

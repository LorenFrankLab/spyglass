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


def populate_all_common(nwb_file_name):
    """Insert all common tables for a given NWB file."""
    from spyglass.spikesorting.imported import ImportedSpikeSorting

    key = [(Nwbfile & f"nwb_file_name LIKE '{nwb_file_name}'").proj()]
    tables = [
        Session,
        # NwbfileKachery, # Not used by default
        ElectrodeGroup,
        Electrode,
        Raw,
        SampleCount,
        DIOEvents,
        # SensorData, # Not used by default. Generates large files
        RawPosition,
        TaskEpoch,
        StateScriptFile,
        VideoFile,
        PositionSource,
        RawPosition,
        ImportedSpikeSorting,
    ]
    error_constants = dict(
        dj_user=dj.config["database.user"],
        connection_id=dj.conn().connection_id,
        nwb_file_name=nwb_file_name,
    )

    for table in tables:
        logger.info(f"Populating {table.__name__}...")
        try:
            table.populate(key)
        except Exception as e:
            InsertError.insert1(
                dict(
                    **error_constants,
                    table=table.__name__,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    error_raw=str(e),
                )
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

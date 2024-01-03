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
from spyglass.spikesorting.imported import ImportedSpikeSorting
from spyglass.utils import logger


def populate_all_common(nwb_file_name):
    # Insert session one by one
    fp = [(Nwbfile & {"nwb_file_name": nwb_file_name}).proj()]
    logger.info("Populate Session...")
    Session.populate(fp)

    # If we use Kachery for data sharing we can uncomment the following two lines. TBD
    # logger.info('Populate NwbfileKachery...')
    # NwbfileKachery.populate()

    logger.info("Populate ElectrodeGroup...")
    ElectrodeGroup.populate(fp)

    logger.info("Populate Electrode...")
    Electrode.populate(fp)

    logger.info("Populate Raw...")
    Raw.populate(fp)

    logger.info("Populate SampleCount...")
    SampleCount.populate(fp)

    logger.info("Populate DIOEvents...")
    DIOEvents.populate(fp)

    # sensor data (from analog ProcessingModule) is temporarily removed from NWBFile
    # to reduce file size while it is not being used. add it back in by commenting out
    # the removal code in spyglass/data_import/insert_sessions.py when ready
    # logger.info('Populate SensorData')
    # SensorData.populate(fp)

    logger.info("Populate TaskEpochs")
    TaskEpoch.populate(fp)
    logger.info("Populate StateScriptFile")
    StateScriptFile.populate(fp)
    logger.info("Populate VideoFile")
    VideoFile.populate(fp)
    logger.info("RawPosition...")
    PositionSource.insert_from_nwbfile(nwb_file_name)
    RawPosition.populate(fp)

    logger.info("Populate ImportedSpikeSorting...")
    ImportedSpikeSorting.populate(fp)

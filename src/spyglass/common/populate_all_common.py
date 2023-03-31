from .common_behav import PositionSource, RawPosition, StateScriptFile, VideoFile
from .common_dio import DIOEvents
from .common_ephys import Electrode, ElectrodeGroup, Raw, SampleCount
from .common_nwbfile import Nwbfile
from .common_session import Session
from .common_task import TaskEpoch


def populate_all_common(nwb_file_name):
    # Insert session one by one
    fp = [(Nwbfile & {"nwb_file_name": nwb_file_name}).proj()]
    print("Populate Session...")
    Session.populate(fp)

    # If we use Kachery for data sharing we can uncomment the following two lines. TBD
    # print('Populate NwbfileKachery...')
    # NwbfileKachery.populate()

    print("Populate ElectrodeGroup...")
    ElectrodeGroup.populate(fp)

    print("Populate Electrode...")
    Electrode.populate(fp)

    print("Populate Raw...")
    Raw.populate(fp)

    print("Populate SampleCount...")
    SampleCount.populate(fp)

    print("Populate DIOEvents...")
    DIOEvents.populate(fp)
    # sensor data (from analog ProcessingModule) is temporarily removed from NWBFile
    # to reduce file size while it is not being used. add it back in by commenting out
    # the removal code in spyglass/data_import/insert_sessions.py when ready
    # print('Populate SensorData')
    # SensorData.populate(fp)
    print("Populate TaskEpochs")
    TaskEpoch.populate(fp)
    print("Populate StateScriptFile")
    StateScriptFile.populate(fp)
    print("Populate VideoFile")
    VideoFile.populate(fp)
    print("RawPosition...")
    PositionSource.insert_from_nwbfile(nwb_file_name)
    RawPosition.populate(fp)

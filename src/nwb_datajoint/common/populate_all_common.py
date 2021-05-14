from .common_session import Session, ExperimenterList
# from .common_nwbfile import NwbfileKachery
from .common_ephys import ElectrodeGroup, Electrode, Raw, SampleCount
from .common_sensors import SensorData
from .common_task import TaskEpoch
from .common_behav import RawPosition, StateScriptFile, VideoFile  # HeadDir, Speed, LinPos,
from .common_dio import DIOEvents
from .common_nwbfile import Nwbfile


def populate_all_common(nwb_file_name):
    # Insert session one by one
    fp = [(Nwbfile & {'nwb_file_name': nwb_file_name}).proj()]
    print('Populate Session...')
    # Session().populate()
    Session.populate(fp)
    # If we use Kachery for data sharing we can uncomment the following two lines. TBD
    # print('Populate NwbfileKachery...')
    # NwbfileKachery.populate()
    print('Populate ExperimenterList...')
    ExperimenterList.populate(fp)
    print('Populate ElectrodeGroup...')
    ElectrodeGroup.populate(fp)
    print('Populate Electrode...')
    Electrode.populate(fp)
    print('Populate Raw...')
    Raw.populate(fp)
    print('Populate SampleCount...')
    SampleCount.populate(fp)
    print('Populate DIOEvents...')
    DIOEvents.populate(fp)
    print('Populate SensorData')
    SensorData.populate(fp)
    print('Populate TaskEpochs')
    TaskEpoch.populate(fp)
    print('Populate StateScriptFile')
    StateScriptFile.populate(fp)
    print('Populate VideoFile')
    VideoFile.populate(fp)
    print('RawPosition...')
    RawPosition.populate(fp)
    # print('HeadDir...')
    # HeadDir().populate()
    # print('Speed...')
    # Speed().populate()
    # print('LinPos...')
    # LinPos().populate()

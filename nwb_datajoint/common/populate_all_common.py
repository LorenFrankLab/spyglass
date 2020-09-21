from .common_session import Session, ExperimenterList
from .common_nwbfile import NwbfileKachery
from .common_ephys import ElectrodeGroup, Electrode, Raw, SampleCount
from .common_sensors import SensorData
from .common_task import TaskEpoch
from .common_behav import RawPosition, HeadDir, Speed, LinPos, StateScriptFile, VideoFile
from .common_dio import DIOEvents

def populate_all_common():
    print('Populate Session...')
    Session().populate()
    print('Populate NwbfileKachery...')
    NwbfileKachery.populate()
    print('Populate ExperimenterList...')
    ExperimenterList().populate()
    print('Populate ElectrodeGroup...')
    ElectrodeGroup().populate()
    print('Populate Electrode...')
    Electrode().populate()
    print('Populate Raw...')
    Raw().populate()
    print('Populate SampleCount...')
    SampleCount().populate()
    print('Populate DIOEvants...')
    DIOEvents().populate()   
    print('Populate SensorData')
    SensorData().populate()
    print('Populate TaskEpochs')
    TaskEpoch().populate()
    print('Populate StateScriptFile')
    StateScriptFile().populate()
    print('Populate VideoFile')
    VideoFile().populate()      
    print('RawPosition...')
    RawPosition().populate()
    print('HeadDir...')
    HeadDir().populate()
    print('Speed...')
    Speed().populate()
    print('LinPos...')
    LinPos().populate()
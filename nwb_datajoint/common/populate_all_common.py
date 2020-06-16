from .common_session import Session, ExperimenterList
from .common_ephys import ElectrodeConfig, Raw
from .common_behav import RawPosition, HeadDir, Speed, LinPos

def populate_all_common():
    print('Populate Session...')
    Session().populate()
    print('Populate ExperimenterList...')
    ExperimenterList().populate()
    print('Populate ElectrodeConfig...')
    ElectrodeConfig().populate()
    print('Populate Raw...')
    Raw().populate()
    print('RawPosition...')
    RawPosition().populate()
    print('HeadDir...')
    HeadDir().populate()
    print('Speed...')
    Speed().populate()
    print('LinPos...')
    LinPos().populate()
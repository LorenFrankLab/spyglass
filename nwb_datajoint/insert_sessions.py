from .common_nwbfile import Nwbfile
from .common_session import ExperimenterList, Session
from .common_ephys import ElectrodeConfig, Raw
from .storage_dirs import check_env
from .common_behav import HeadDir, LinPos, RawPosition, Speed
import datajoint as dj

conn = dj.conn()

def insert_sessions(nwb_file_names):
    """Populate the dj database with new sessions

    nwb_file_names is a list of relative file paths, relative
    to $NWB_DATAJOINT_BASE_DIR, pointing to existing .nwb files.
    Each file represents a session.

    Args:
        nwb_file_names (List[str]): List of relative file paths
    """    
    check_env()

    for nwb_file_name in nwb_file_names:
        Nwbfile().insert_from_relative_file_name(nwb_file_name)
    populate_all()
    
def populate_all():
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
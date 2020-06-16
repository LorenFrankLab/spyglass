from ..common.common_nwbfile import Nwbfile
from ..common import populate_all_common
from .storage_dirs import check_env
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
        assert not nwb_file_name.startswith('/'), f'You must use relative paths. nwb_file_name: {nwb_file_name}'
        Nwbfile().insert_from_relative_file_name(nwb_file_name)
    populate_all_common()
# Configure datajoint
# You can use:
# export DJ_HOST=...
# export DJ_USER=...
# export DJ_PASS=...

# Important to do this so that we add the franklab namespace for pynwb
# Note: This is franklab-specific
import ndx_franklab_novela

from .data_import.insert_sessions import insert_sessions
from .data_import.storage_dirs import base_dir, check_env, kachery_storage_dir

# from .lock import file_lock as lock

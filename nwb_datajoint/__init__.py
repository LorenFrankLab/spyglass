import os

# Configure datajoint
# You can use:
# export DJ_HOST=...
# export DJ_USER=...
# export DJ_PASS=...
import datajoint as dj
dj.config["enable_python_native_blobs"] = True

# Important to do this so that we add the franklab namespace for pynwb
# Note: This is franklab-specific -- TODO: move it outside this repo
# Note: we need to sync this with the source repo

from ndx_franklab_novela import probe

from .data_import.storage_dirs import check_env, kachery_storage_dir, base_dir
from .data_import.insert_sessions import insert_sessions


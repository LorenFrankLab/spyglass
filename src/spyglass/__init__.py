# Configure datajoint
# You can use:
# export DJ_HOST=...
# export DJ_USER=...
# export DJ_PASS=...

# Important to do this so that we add the franklab namespace for pynwb
# Note: This is franklab-specific
import ndx_franklab_novela

import importlib.metadata
from .settings import config

__version__ = importlib.metadata.version("spyglass-neuro")

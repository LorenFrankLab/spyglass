# Configure datajoint
# You can use:
# export DJ_HOST=...
# export DJ_USER=...
# export DJ_PASS=...

# Important to do this so that we add the franklab namespace for pynwb
# Note: This is franklab-specific
try:
    import ndx_franklab_novela
except ImportError:
    pass


from .settings import config

try:
    from ._version import __version__
except ImportError:
    pass

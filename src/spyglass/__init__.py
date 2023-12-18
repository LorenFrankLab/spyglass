from .settings import config  # ensure loaded config dirs

try:
    import ndx_franklab_novela
except ImportError:
    pass

try:
    from ._version import __version__
except ImportError:
    pass

__all__ = ["ndx_franklab_novela", "__version__", "config"]

from spyglass.settings import config  # ensure loaded config dirs

try:
    from ._version import __version__
except (ImportError, ModuleNotFoundError):
    pass

__all__ = ["__version__", "config"]

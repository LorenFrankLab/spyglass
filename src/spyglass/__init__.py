from spyglass.settings import config  # ensure loaded config dirs

try:
    from ._version import __version__
except (ImportError, ModuleNotFoundError):
    pass

__all__ = ["__version__", "config"]

# Warn once if this install is dirty or stale. Opens no connection.
try:
    from spyglass.settings import test_mode as _test_mode

    if not _test_mode:
        from spyglass.utils.git_utils import spyglass_install

        spyglass_install.warn_once()
except Exception:
    pass

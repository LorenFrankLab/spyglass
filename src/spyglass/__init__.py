from spyglass.settings import config  # ensure loaded config dirs

try:
    from ._version import __version__
except (ImportError, ModuleNotFoundError):
    pass

__all__ = ["__version__", "config"]

# Warn once if this is a dirty or stale install. Reads git and local state
# only: no connection, no subprocess. The database record is written on the
# session's first insert or populate. Skipped under test_mode so the test
# harness can configure the DB before common schemas are declared, and fully
# guarded so a failure cannot break the import.
try:
    from spyglass.settings import test_mode as _test_mode

    if not _test_mode:
        from spyglass.utils.git_utils import spyglass_install

        spyglass_install.warn_once()
except Exception:
    pass

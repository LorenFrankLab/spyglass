from spyglass.settings import config  # ensure loaded config dirs

try:
    from ._version import __version__
except (ImportError, ModuleNotFoundError):
    pass

__all__ = ["__version__", "config"]

# Warn once, synchronously, if this is a dirty/stale install. For an official
# (clean + current) install this returns before importing anything from
# common/ or touching the DB, so the common import path is paid only by the
# flagged minority. Skipped under test_mode so the test harness can configure
# the DB before common schemas are declared. Fully guarded — never break
# import on failure.
try:
    from spyglass.settings import test_mode as _test_mode

    if not _test_mode:
        from spyglass.utils.git_utils import _warn_once

        _warn_once()
except Exception:
    pass

from spyglass.settings import config  # ensure loaded config dirs

try:
    from ._version import __version__
except (ImportError, ModuleNotFoundError):
    pass

__all__ = ["__version__", "config"]

# Warn once, synchronously, if this is a dirty/stale install. The check is git
# and local state only — no DB connection, no subprocess — so an import never
# stalls behind a VPN or prompts for a password. The database record is written
# on the session's first insert or populate instead. Skipped under test_mode so
# the test harness can configure the DB before common schemas are declared.
# Fully guarded — never break import on failure.
try:
    from spyglass.settings import test_mode as _test_mode

    if not _test_mode:
        from spyglass.utils.git_utils import _warn_once

        _warn_once()
except Exception:
    pass

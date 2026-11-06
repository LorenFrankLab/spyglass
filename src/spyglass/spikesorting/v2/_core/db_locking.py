"""MySQL advisory (named) locks keyed on a hash of table + key.

- :func:`advisory_key_lock`: best-effort. Yields whether it acquired; the
  caller proceeds either way. Use only where a duplicate-tolerant insert makes
  an unserialized run wasteful, not wrong.
- :func:`required_advisory_lock`: fail-closed. Raises
  :class:`AdvisoryLockError` unless ``GET_LOCK`` returns ``1``.

A named lock is bound to the DB session (a crashed run leaves no stale lock)
and lives on the shared server (it coordinates across hosts).
"""

from __future__ import annotations

from contextlib import contextmanager

#: Longer than any single stage; finite so a stuck holder cannot wedge waiters.
POPULATE_LOCK_TIMEOUT_S = 8 * 3600
#: The lifecycle lock spans only a short check + cascade. Both timeouts are
#: read at call time so tests can shorten them.
LIFECYCLE_LOCK_TIMEOUT_S = 120


class AdvisoryLockError(RuntimeError):
    """A required advisory lock could not be acquired (timeout or server error)."""


def _lock_name(table, key) -> str:
    """Derive the 64-char-safe MySQL lock name for a ``(table, key)`` resource.

    Both lock flavors share it, so they contend on the same table + key.
    """
    from datajoint.hash import key_hash

    # MySQL caps lock names at 64 chars. Keep the prefix stable so workers on
    # different Spyglass versions share lock names.
    return "sgv2pop:" + key_hash({"__table__": table.full_table_name, **key})


@contextmanager
def advisory_key_lock(table, key):
    """Best-effort serialize one ``(table, key)`` via a MySQL named lock.

    Yields ``True`` if acquired, ``False`` on timeout or error; never raises.
    Released on exit.
    """
    lock_name = _lock_name(table, key)
    connection = table.connection
    acquired = False
    try:
        try:
            result = connection.query(
                "SELECT GET_LOCK(%s, %s)",
                args=(lock_name, int(POPULATE_LOCK_TIMEOUT_S)),
            ).fetchone()
            acquired = bool(result and result[0] == 1)
        except Exception:  # noqa: BLE001 - best-effort; fall through unacquired
            # A real DB problem resurfaces on the guarded operation.
            acquired = False
        yield acquired
    finally:
        if acquired:
            try:
                connection.query("SELECT RELEASE_LOCK(%s)", args=(lock_name,))
            except Exception:  # noqa: BLE001 - session close frees it anyway
                pass


@contextmanager
def required_advisory_lock(table, key):
    """Fail-closed serialize one ``(table, key)`` via a MySQL named lock.

    Released on exit.

    Raises
    ------
    AdvisoryLockError
        Unless ``GET_LOCK`` returns ``1`` (``0`` = timed out, ``NULL`` = server
        error).
    """
    timeout_s = int(LIFECYCLE_LOCK_TIMEOUT_S)
    lock_name = _lock_name(table, key)
    connection = table.connection
    acquired = False
    try:
        try:
            result = connection.query(
                "SELECT GET_LOCK(%s, %s)", args=(lock_name, timeout_s)
            ).fetchone()
        except Exception as exc:  # GET_LOCK itself failed -> fail closed
            raise AdvisoryLockError(
                f"advisory lock {lock_name!r} query failed: {exc}"
            ) from exc
        outcome = result[0] if result else None
        if outcome != 1:
            raise AdvisoryLockError(
                f"could not acquire advisory lock {lock_name!r} within "
                f"{timeout_s}s (GET_LOCK returned {outcome!r}: "
                f"{'timed out' if outcome == 0 else 'server error'}); refusing "
                "to proceed unserialized."
            )
        acquired = True
        yield
    finally:
        if acquired:
            try:
                connection.query("SELECT RELEASE_LOCK(%s)", args=(lock_name,))
            except Exception:  # noqa: BLE001 - session close frees it anyway
                pass

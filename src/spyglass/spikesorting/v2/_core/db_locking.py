"""MySQL advisory (named) locks keyed on a hash of table + key.

:func:`advisory_key_lock` is best-effort: it yields whether it acquired, and
the caller proceeds either way. Use it only where a duplicate-tolerant insert
makes an unserialized run wasteful, not wrong.

A named lock is bound to the DB session (a crashed run leaves no stale lock)
and lives on the shared server (it coordinates across hosts).
"""

from __future__ import annotations

from contextlib import contextmanager

#: Longer than any single stage; finite so a stuck holder cannot wedge waiters.
POPULATE_LOCK_TIMEOUT_S = 8 * 3600


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

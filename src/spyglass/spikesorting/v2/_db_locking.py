"""Shared MySQL advisory (named) lock primitives for spike-sorting v2.

Two flavors of the same ``GET_LOCK`` / ``RELEASE_LOCK`` named lock, both keyed on
a hash of a table + key so that only operations on the SAME resource serialize
(different keys proceed in parallel):

* :func:`advisory_key_lock` -- BEST-EFFORT. Yields ``True`` if acquired, ``False``
  on timeout/error, and the caller proceeds either way. Correct ONLY where a
  downstream net makes an un-serialized run merely wasteful, never wrong (the
  pipeline's compute-once populate, whose duplicate-tolerant insert is that net).

* :func:`required_advisory_lock` -- FAIL-CLOSED. Raises :class:`AdvisoryLockError`
  unless ``GET_LOCK`` returns exactly ``1``, so an operation whose CORRECTNESS
  depends on the lock (the artifact delete-vs-select serialization) aborts loudly
  instead of proceeding unserialized. Uses a short, caller-tunable timeout.

A named lock is bound to the DB *session*, so a crashed run leaves no stale
reservation, and it lives on the shared server, so it coordinates across
processes and hosts (a local ``FileLock`` would not). This module depends on
neither the schema nor the pipeline, so domain tables can serialize against each
other without importing the large orchestration module.
"""

from __future__ import annotations

from contextlib import contextmanager

#: Best-effort populate lock: a compute stage may run for hours, so a racing run
#: should wait rather than duplicate the work. Longer than any realistic single
#: stage; finite so a stuck (not crashed) holder cannot wedge a waiter forever.
POPULATE_LOCK_TIMEOUT_S = 8 * 3600
#: Fail-closed lifecycle lock: the delete/select serialization holds the lock
#: only across a short check + cascade, so a wait past this means real contention
#: or a stuck holder -- abort loudly instead of blocking for hours. Read at call
#: time (callers pass ``timeout_s=None``), so a test can shorten it.
LIFECYCLE_LOCK_TIMEOUT_S = 120


class AdvisoryLockError(RuntimeError):
    """A required advisory lock could not be acquired (timeout or server error).

    Raised only by :func:`required_advisory_lock`; the best-effort
    :func:`advisory_key_lock` never raises. A distinct type so a caller (or a
    test) can tell a lock-acquisition failure from the guarded operation's own
    errors.
    """


def _lock_name(table, key) -> str:
    """Derive the 64-char-safe MySQL lock name for a ``(table, key)`` resource.

    Both flavors derive the name identically, so a best-effort and a required
    caller on the SAME table + key contend on the same lock -- in particular the
    artifact detection delete and ``SortingSelection.insert_selection`` (both
    keyed on ``ArtifactDetectionOutput`` + ``artifact_detection_id``) serialize
    against each other.
    """
    from datajoint.hash import key_hash

    # MySQL lock names cap at 64 chars; "sgv2pop:" (8) + 32-char hash = 40.
    # The prefix is UNCHANGED from when the best-effort lock lived in
    # _pipeline_run, so a mixed-version populate worker still coordinates on the
    # same name (and the lock-observability test that inspects it stays valid).
    # It is pure namespace: the populate lock and the lifecycle lock never share
    # a name -- they key on different tables -- so one prefix serves both.
    return "sgv2pop:" + key_hash({"__table__": table.full_table_name, **key})


@contextmanager
def advisory_key_lock(table, key, *, timeout_s: int | None = None):
    """Best-effort serialize one ``(table, key)`` via a MySQL named lock.

    Yields ``True`` if the lock was acquired, ``False`` on timeout/error -- the
    caller then proceeds unserialized, which is correct ONLY where a downstream
    net (e.g. a duplicate-tolerant insert) keeps that from being wrong. Never
    raises. Self-releasing: ``RELEASE_LOCK`` on exit, and the session ending
    frees it anyway.
    """
    if timeout_s is None:
        timeout_s = POPULATE_LOCK_TIMEOUT_S
    lock_name = _lock_name(table, key)
    connection = table.connection
    acquired = False
    try:
        try:
            result = connection.query(
                "SELECT GET_LOCK(%s, %s)", args=(lock_name, int(timeout_s))
            ).fetchone()
            acquired = bool(result and result[0] == 1)
        except Exception:  # noqa: BLE001 - best-effort; fall through unacquired
            # A GET_LOCK failure must never fail the caller; any real DB problem
            # resurfaces on the guarded operation below.
            acquired = False
        yield acquired
    finally:
        if acquired:
            try:
                connection.query("SELECT RELEASE_LOCK(%s)", args=(lock_name,))
            except Exception:  # noqa: BLE001 - session close frees it anyway
                pass


@contextmanager
def required_advisory_lock(table, key, *, timeout_s: int | None = None):
    """Fail-closed serialize one ``(table, key)`` via a MySQL named lock.

    Raises :class:`AdvisoryLockError` unless ``GET_LOCK`` returns exactly ``1``
    (``0`` = timed out, ``NULL`` = server error), so an operation whose
    correctness depends on the lock ABORTS instead of proceeding unserialized.
    Yields nothing (entering the block IS the acquisition proof). Self-releasing
    on exit.
    """
    if timeout_s is None:
        timeout_s = LIFECYCLE_LOCK_TIMEOUT_S
    lock_name = _lock_name(table, key)
    connection = table.connection
    acquired = False
    try:
        try:
            result = connection.query(
                "SELECT GET_LOCK(%s, %s)", args=(lock_name, int(timeout_s))
            ).fetchone()
        except Exception as exc:  # GET_LOCK itself failed -> fail closed
            raise AdvisoryLockError(
                f"advisory lock {lock_name!r} query failed: {exc}"
            ) from exc
        outcome = result[0] if result else None
        if outcome != 1:
            raise AdvisoryLockError(
                f"could not acquire advisory lock {lock_name!r} within "
                f"{int(timeout_s)}s (GET_LOCK returned {outcome!r}: "
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

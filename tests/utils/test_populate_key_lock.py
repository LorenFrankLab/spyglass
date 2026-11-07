"""Opt-in per-key populate lock around DataJoint's ``_populate1``.

The DB-free tests run DataJoint's real ``_populate1`` on a stand-in table
whose connection emulates MySQL named locks; the database tests check the
lock SQL on the test server.
"""

import threading

import pytest


class _LockServer:
    """MySQL named locks shared by the stand-in connections."""

    def __init__(self):
        self.locks = {}
        self.guard = threading.Lock()

    def lock(self, name):
        with self.guard:
            return self.locks.setdefault(name, threading.Lock())


class _Result:
    def __init__(self, value):
        self.value = value

    def fetchone(self):
        return (self.value,)


class _Connection:
    """Transactions plus GET_LOCK / RELEASE_LOCK on a shared lock server."""

    def __init__(self, server):
        self.server = server
        self.in_transaction = False
        self.queries = []

    def query(self, sql, args=()):
        self.queries.append(sql.split("(")[0])
        lock = self.server.lock(args[0])
        if "RELEASE_LOCK" in sql:
            lock.release()
            return _Result(1)
        timeout = args[1]
        if timeout == 0:
            return _Result(int(lock.acquire(blocking=False)))
        return _Result(int(lock.acquire(timeout=timeout)))

    def start_transaction(self):
        self.in_transaction = True

    def cancel_transaction(self):
        self.in_transaction = False

    def commit_transaction(self):
        self.in_transaction = False


def _table_class(*, locked, computed, registered, release=None):
    """A stand-in tri-part table; ``release`` holds compute until set."""
    from datajoint.autopopulate import AutoPopulate

    from spyglass.utils.mixins.populate import PopulateMixin

    class Table(PopulateMixin, AutoPopulate):
        table_name = "stand_in"
        full_table_name = "`test`.`stand_in`"
        _populate_key_lock = locked

        def __init__(self, connection):
            self.connection = connection

        @property
        def target(self):
            return self

        def __contains__(self, key):
            return key["id"] in registered

        def make_fetch(self, key):
            return (key["id"],)

        def make_compute(self, key, *fetched):
            computed.append(key["id"])
            if release is not None:
                release.wait(timeout=5)
            return (key["id"],)

        def make_insert(self, key, *computed_result):
            registered.append(key["id"])

    return Table


def _populate1(table, key):
    return table._populate1(
        key, None, suppress_errors=False, return_exception_objects=False
    )


def test_default_takes_no_lock():
    """Tables that do not opt in issue no lock queries."""
    computed, registered = [], []
    connection = _Connection(_LockServer())
    table = _table_class(locked=False, computed=computed, registered=registered)

    assert _populate1(table(connection), {"id": "a"}) is True
    assert connection.queries == []
    assert registered == ["a"]


def test_locked_populate_takes_and_releases_the_key_lock():
    """An opted-in populate holds the key's lock and frees it after."""
    computed, registered = [], []
    server = _LockServer()
    connection = _Connection(server)
    table = _table_class(locked=True, computed=computed, registered=registered)

    assert _populate1(table(connection), {"id": "a"}) is True
    assert connection.queries == ["SELECT GET_LOCK", "SELECT RELEASE_LOCK"]
    assert not any(lock.locked() for lock in server.locks.values())


def test_waiting_job_skips_a_key_another_job_populated(monkeypatch):
    """A second job waits for the first, then skips the key uncomputed."""
    from spyglass.utils.mixins import populate

    messages = []
    monkeypatch.setattr(populate.logger, "info", messages.append)
    computed, registered = [], []
    server = _LockServer()
    release = threading.Event()
    table = _table_class(
        locked=True, computed=computed, registered=registered, release=release
    )
    results = {}

    first = threading.Thread(
        target=lambda: results.setdefault(
            "first", _populate1(table(_Connection(server)), {"id": "a"})
        )
    )
    first.start()
    while not computed:  # the first job is computing and holds the lock
        threading.Event().wait(0.01)
    second = threading.Thread(
        target=lambda: results.setdefault(
            "second", _populate1(table(_Connection(server)), {"id": "a"})
        )
    )
    second.start()
    while not messages:  # the second job found the lock taken
        threading.Event().wait(0.01)
    release.set()
    first.join(5)
    second.join(5)

    assert results == {"first": True, "second": False}
    assert computed == ["a"]
    assert registered == ["a"]
    assert "waiting for another job" in messages[0]


def test_lock_error_still_populates(monkeypatch):
    """A failing GET_LOCK is logged and the populate goes ahead."""
    from spyglass.utils.mixins import populate

    warnings = []
    monkeypatch.setattr(populate.logger, "warning", warnings.append)
    computed, registered = [], []

    class _BrokenLocks(_Connection):
        def query(self, sql, args=()):
            raise RuntimeError("simulated GET_LOCK failure")

    table = _table_class(locked=True, computed=computed, registered=registered)

    assert _populate1(table(_BrokenLocks(_LockServer())), {"id": "a"}) is True
    assert registered == ["a"]
    assert "unavailable" in warnings[0]


class _LockTable:
    full_table_name = "`test`.`populate_key_lock`"


@pytest.mark.usefixtures("dj_conn")
def test_key_lock_excludes_another_session(monkeypatch):
    """On MySQL, a second session waits on a held key lock, then takes it."""
    import datajoint as dj

    from spyglass.utils.mixins import populate

    other = dj.Connection(
        dj.config["database.host"],
        dj.config["database.user"],
        dj.config["database.password"],
        port=dj.config["database.port"],
    )
    holder, waiter = _LockTable(), _LockTable()
    holder.connection, waiter.connection = dj.conn(), other
    key = {"id": "exclusion"}
    monkeypatch.setattr(populate, "POPULATE_LOCK_TIMEOUT_S", 1)
    try:
        with populate.populate_key_lock(holder, key) as held:
            assert held is True
            with populate.populate_key_lock(waiter, key) as while_held:
                assert while_held is False
        with populate.populate_key_lock(waiter, key) as after_release:
            assert after_release is True
    finally:
        other.close()

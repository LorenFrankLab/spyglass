"""Analyzer orphan cleanup waits for a publisher's database commit."""

import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Event, get_ident
from types import SimpleNamespace

import pytest

from tests.spikesorting.v2.test_staged_outputs import _populate1, _table_class


def _assert_other_thread_cannot_acquire(cache, sorting_id):
    from filelock import Timeout

    def contend():
        try:
            with cache.analyzer_cache_lock(sorting_id).acquire(timeout=0):
                return False
        except Timeout:
            return True

    with ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(contend).result(timeout=5)


@pytest.mark.parametrize("failure", [None, "insert", "commit"])
def test_framework_publication_lock_covers_commit_and_rollback(
    tmp_path, monkeypatch, failure
):
    """Use DataJoint's actual tri-part dispatch and transaction callbacks."""
    from spyglass.spikesorting.v2._storage import analyzer_cache as cache

    monkeypatch.setattr(cache, "analyzer_cache_root", lambda: tmp_path)
    sorting_id = uuid.uuid4()
    error = RuntimeError(f"{failure} failed") if failure else None

    class Table(
        cache.AnalyzerPublicationMixin,
        _table_class(
            tmp_path,
            insert_error=error if failure == "insert" else None,
            commit_error=error if failure == "commit" else None,
        ),
    ):
        pass

    table = Table()
    callback = (
        "cancel_transaction" if failure == "insert" else "commit_transaction"
    )
    transaction_finished = getattr(table.connection, callback)
    seen = []

    def finish():
        assert cache.analyzer_population_active(sorting_id)
        _assert_other_thread_cannot_acquire(cache, sorting_id)
        seen.append(callback)
        return transaction_finished()

    monkeypatch.setattr(table.connection, callback, finish)
    key = {"id": "a", "sorting_id": sorting_id}
    if error:
        with pytest.raises(RuntimeError, match=f"{failure} failed"):
            _populate1(table, key)
    else:
        assert _populate1(table, key) is True
    assert seen == [callback]
    assert not cache.analyzer_population_active(sorting_id)
    with cache.analyzer_cache_lock(sorting_id).acquire(timeout=0):
        pass
    if failure == "insert":
        assert not (tmp_path / "a.nwb").exists()


def test_direct_publication_transaction_keeps_lock_through_own_commit(
    tmp_path, monkeypatch
):
    from spyglass.spikesorting.v2._storage import analyzer_cache as cache

    monkeypatch.setattr(cache, "analyzer_cache_root", lambda: tmp_path)
    sorting_id = uuid.uuid4()
    committed = []

    class Table:
        connection = SimpleNamespace(in_transaction=False)

        @contextmanager
        def _safe_context(self):
            self.connection.in_transaction = True
            try:
                yield
                _assert_other_thread_cannot_acquire(cache, sorting_id)
                committed.append(True)
            finally:
                self.connection.in_transaction = False

    with cache.analyzer_publication_transaction(Table(), sorting_id):
        _assert_other_thread_cannot_acquire(cache, sorting_id)
    assert committed == [True]


def test_direct_publication_refuses_caller_transaction(tmp_path, monkeypatch):
    from datajoint.errors import DataJointError

    from spyglass.spikesorting.v2._storage import analyzer_cache as cache

    monkeypatch.setattr(cache, "analyzer_cache_root", lambda: tmp_path)
    table = SimpleNamespace(connection=SimpleNamespace(in_transaction=True))
    with pytest.raises(DataJointError, match="caller-owned"):
        with cache.analyzer_publication_transaction(table, uuid.uuid4()):
            pytest.fail("unsupported publication must not run")


@pytest.mark.parametrize("commits", [False, True])
def test_orphan_sweep_rechecks_owner_after_publisher_finishes(
    tmp_path, monkeypatch, commits
):
    """The scan sees an uncommitted folder; locked cleanup sees its outcome."""
    import datajoint as dj

    from spyglass.spikesorting.v2._storage import analyzer_cache as cache

    monkeypatch.setattr(cache, "analyzer_cache_root", lambda: tmp_path)
    sorting_id = uuid.uuid4()
    canonical = cache.analyzer_path(sorting_id, "test")
    published, reviewed, finish_transaction = Event(), Event(), Event()
    cleanup_trying, rechecked = Event(), Event()
    committed = Event()
    collections = []
    sweep_thread = []
    real_lock = cache.analyzer_cache_lock

    def observed_lock(sorting_id):
        if sweep_thread and get_ident() == sweep_thread[0]:
            cleanup_trying.set()
        return real_lock(sorting_id)

    monkeypatch.setattr(cache, "analyzer_cache_lock", observed_lock)

    class Table:
        connection = SimpleNamespace(in_transaction=False)

        def __and__(self, restriction):
            return self

    table = Table()

    def collect(relation):
        collections.append(committed.is_set())
        if len(collections) > 1:
            rechecked.set()
        return {
            "units_bearing": [],
            "referenced_paths": (
                {str(canonical)} if committed.is_set() else set()
            ),
            "reclaimed_paths": set(),
        }

    monkeypatch.setattr(cache, "collect_analyzer_cache_references", collect)
    monkeypatch.setattr(cache, "cleanup_analyzer_staging", lambda *a, **k: [])

    def confirm(*args, **kwargs):
        reviewed.set()
        return "yes"

    monkeypatch.setattr(dj.utils, "user_choice", confirm)

    def publish():
        with cache.analyzer_population(sorting_id):
            with cache.StagedAnalyzer(canonical) as staged:
                staged.folder.mkdir()
                (staged.folder / "marker").write_text("completed analyzer")
                staged.publish()
            published.set()
            assert finish_transaction.wait(5)
            if commits:
                committed.set()

    def sweep_orphans():
        sweep_thread.append(get_ident())
        return cache.find_orphaned_analyzer_folders(table, dry_run=False)

    with ThreadPoolExecutor(max_workers=2) as pool:
        publisher = pool.submit(publish)
        try:
            assert published.wait(5)
            sweep = pool.submit(sweep_orphans)
            assert reviewed.wait(5)
            assert cleanup_trying.wait(5)
            assert not rechecked.wait(0.1), "ownership was read before COMMIT"
            assert not sweep.done(), "cleanup did not wait for the transaction"
        finally:
            finish_transaction.set()
        publisher.result(timeout=5)
        report = sweep.result(timeout=5)

    assert report["disk_side"] == [str(canonical)]
    assert collections == [False, commits]
    assert canonical.exists() is commits
    if commits:
        assert (canonical / "marker").read_text() == "completed analyzer"


def test_orphan_cleanup_refuses_repeatable_read_transaction(tmp_path):
    from datajoint.errors import DataJointError

    from spyglass.spikesorting.v2._storage import analyzer_cache as cache

    table = SimpleNamespace(connection=SimpleNamespace(in_transaction=True))
    with pytest.raises(DataJointError, match="outside a database transaction"):
        cache.find_orphaned_analyzer_folders(table, dry_run=False)


def test_failed_publish_discards_only_its_attempt(tmp_path, monkeypatch):
    from spyglass.spikesorting.v2._storage import analyzer_cache as cache

    monkeypatch.setattr(cache, "analyzer_cache_root", lambda: tmp_path)
    canonical = cache.analyzer_path(uuid.uuid4(), "test")
    canonical.mkdir()
    (canonical / "marker").write_text("existing analyzer")
    replace = cache.os.replace
    with pytest.raises(PermissionError, match="publish failed"):
        with cache.StagedAnalyzer(canonical) as staged:
            staged.folder.mkdir()
            (staged.folder / "marker").write_text("failed attempt")

            def fail_install(source, target):
                if Path(source) == staged.folder:
                    raise PermissionError("publish failed")
                return replace(source, target)

            monkeypatch.setattr(cache.os, "replace", fail_install)
            staged.publish()
    assert (canonical / "marker").read_text() == "existing analyzer"
    assert not staged.folder.exists()

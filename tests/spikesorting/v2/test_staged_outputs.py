"""Staged-output cleanup around DataJoint's tri-part ``_populate1``.

DB-free: a stand-in table runs DataJoint's real ``AutoPopulate._populate1``
and generator ``make`` against a fake connection, so these tests exercise the
pinned DataJoint flow (fetch, compute, second fetch inside the transaction,
insert) that the cleanup relies on.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import pytest


def test_staged_tables_refuse_unverified_datajoint_lifecycle(monkeypatch):
    from spyglass.spikesorting.v2._core import dj_compat as _dj_compat
    from spyglass.spikesorting.v2._storage.staged_outputs import (
        StagedOutputCleanupMixin,
    )

    monkeypatch.setattr(_dj_compat, "version", lambda name: "0.15.0")
    with pytest.raises(RuntimeError, match="require DataJoint 0.14.9"):

        class UnsupportedTable(StagedOutputCleanupMixin):
            pass


class _Connection:
    """Transaction bookkeeping DataJoint's ``_populate1`` calls."""

    def __init__(self, commit_error=None):
        self.in_transaction = False
        self.commit_error = commit_error

    def start_transaction(self):
        self.in_transaction = True

    def cancel_transaction(self):
        self.in_transaction = False

    def commit_transaction(self):
        self.in_transaction = False
        if self.commit_error is not None:
            raise self.commit_error


class _Owner:
    """A staging owner like ``StagedAnalyzer``: ``close`` discards a folder."""

    def __init__(self, folder: Path):
        self.folder = folder
        self.closed = 0

    def close(self):
        self.closed += 1
        if self.folder.exists():
            self.folder.rmdir()


def _table_class(tmp_path, *, fetch=None, insert_error=None, commit_error=None):
    """Build a stand-in tri-part table staging one file and one folder."""
    from datajoint.autopopulate import AutoPopulate

    from spyglass.spikesorting.v2._storage.staged_outputs import (
        StagedOutputCleanupMixin,
        StagedOutputs,
    )

    class Computed(NamedTuple):
        analysis_file_name: str
        owner: _Owner
        bundle: str

        def staged_outputs(self):
            return StagedOutputs(
                analysis_file_names=(self.analysis_file_name,),
                owners=(self.owner,),
                folders=(self.bundle,),
            )

    class Table(StagedOutputCleanupMixin, AutoPopulate):
        table_name = "stand_in"
        full_table_name = "`test`.`stand_in`"

        def __init__(self):
            self.connection = _Connection(commit_error)
            self.registered: list[str] = []
            self.fetch_calls = 0
            self.owners: dict[str, _Owner] = {}

        @property
        def target(self):
            return self

        def __contains__(self, key):
            return f"{key['id']}.nwb" in self.registered

        def make_fetch(self, key):
            self.fetch_calls += 1
            if fetch is not None:
                return fetch(self.fetch_calls)
            return ("parent", key["id"])

        def make_compute(self, key, *fetched):
            name = f"{key['id']}.nwb"
            (tmp_path / name).write_text("staged")
            folder = tmp_path / f"{key['id']}.build"
            folder.mkdir()
            self.owners[name] = _Owner(folder)
            bundle = tmp_path / f"{key['id']}.bundle"
            bundle.mkdir()
            (bundle / "index.html").write_text("staged")
            return Computed(name, self.owners[name], str(bundle))

        def make_insert(self, key, analysis_file_name, owner, bundle):
            if insert_error is not None:
                raise insert_error
            self.registered.append(analysis_file_name)

        @staticmethod
        def _unlink_analysis_file(analysis_file_name, *, context):
            (tmp_path / analysis_file_name).unlink(missing_ok=True)

    return Table


def _populate1(table, key, *, jobs=None, suppress_errors=False):
    return table._populate1(
        key,
        jobs,
        suppress_errors=suppress_errors,
        return_exception_objects=False,
    )


def _changed_second_fetch(call):
    return ("parent", "relabelled" if call == 2 else "original")


def test_success_keeps_registered_outputs(tmp_path):
    """A committed insert keeps its file and folder; nothing stays recorded."""
    table = _table_class(tmp_path)()

    assert _populate1(table, {"id": "a"}) is True

    assert table.registered == ["a.nwb"]
    assert (tmp_path / "a.nwb").exists()
    assert (tmp_path / "a.build").exists()
    assert (tmp_path / "a.bundle").exists()
    assert table.owners["a.nwb"].closed == 0
    assert table._staged_output_scopes == []


def test_changed_second_fetch_removes_staged_outputs(tmp_path):
    """DataJoint's refused insert (changed fetch) removes what compute staged."""
    from datajoint.errors import DataJointError

    table = _table_class(tmp_path, fetch=_changed_second_fetch)()

    with pytest.raises(DataJointError, match="Referential integrity"):
        _populate1(table, {"id": "a"})

    assert table.fetch_calls == 2
    assert table.registered == []
    assert not (tmp_path / "a.nwb").exists()
    assert not (tmp_path / "a.build").exists()
    assert not (tmp_path / "a.bundle").exists()
    assert table.owners["a.nwb"].closed == 1
    assert table._staged_output_scopes == []


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_failing_second_fetch_removes_staged_outputs(tmp_path, error):
    """A second fetch that raises removes the outputs and re-raises as is."""

    def fetch(call):
        if call == 2:
            raise error("parent gone")
        return ("parent",)

    table = _table_class(tmp_path, fetch=fetch)()

    with pytest.raises(error, match="parent gone"):
        _populate1(table, {"id": "a"})

    assert not (tmp_path / "a.nwb").exists()
    assert table.owners["a.nwb"].closed == 1


def test_suppressed_error_removes_staged_outputs(tmp_path):
    """Under ``suppress_errors`` the failure is returned, outputs removed."""
    table = _table_class(tmp_path, fetch=_changed_second_fetch)()

    key, message = _populate1(table, {"id": "a"}, suppress_errors=True)

    assert key == {"id": "a"}
    assert "Referential integrity" in message
    assert not (tmp_path / "a.nwb").exists()
    assert not (tmp_path / "a.build").exists()


def test_failed_insert_removes_staged_outputs(tmp_path):
    """A ``make_insert`` failure (rolled back) leaves nothing on disk."""
    table = _table_class(tmp_path, insert_error=RuntimeError("duplicate"))()

    with pytest.raises(RuntimeError, match="duplicate"):
        _populate1(table, {"id": "a"})

    assert not (tmp_path / "a.nwb").exists()
    assert not (tmp_path / "a.bundle").exists()
    assert table.owners["a.nwb"].closed == 1


def test_failure_after_insert_leaves_registered_outputs(tmp_path):
    """Outputs ``make_insert`` registered are never removed.

    DataJoint commits and then marks the job complete; a failure there
    (after the commit) escapes ``_populate1``, but the file now belongs to a
    committed row.
    """

    class Jobs:
        def reserve(self, table_name, key):
            return True

        def complete(self, table_name, key):
            raise ConnectionError("jobs table unreachable")

    table = _table_class(tmp_path)()

    with pytest.raises(ConnectionError):
        _populate1(table, {"id": "a"}, jobs=Jobs())

    assert table.registered == ["a.nwb"]
    assert (tmp_path / "a.nwb").exists()
    assert (tmp_path / "a.bundle").exists()
    assert table.owners["a.nwb"].closed == 0


def test_skipped_key_computes_and_removes_nothing(tmp_path):
    """A key DataJoint skips (already populated) never records outputs."""
    table = _table_class(tmp_path)()
    table.registered.append("a.nwb")
    (tmp_path / "a.nwb").write_text("committed")

    assert _populate1(table, {"id": "a"}) is False

    assert table.fetch_calls == 0
    assert (tmp_path / "a.nwb").read_text() == "committed"


def test_each_key_is_cleaned_independently(tmp_path):
    """A failed key's outputs are removed; the next key's are kept."""

    def fetch(call):
        # Key "a" is fetched on calls 1-2 (changed), key "b" on 3-4.
        return ("parent", "changed" if call == 2 else "same")

    table = _table_class(tmp_path, fetch=fetch)()

    key_a, _ = _populate1(table, {"id": "a"}, suppress_errors=True)
    assert _populate1(table, {"id": "b"}) is True

    assert key_a == {"id": "a"}
    assert not (tmp_path / "a.nwb").exists()
    assert (tmp_path / "b.nwb").exists()
    assert table.registered == ["b.nwb"]
    assert table._staged_output_scopes == []


@pytest.mark.parametrize("outer_fails", [False, True])
def test_nested_populate_removes_only_its_own_outputs(tmp_path, outer_fails):
    """An inner populate that fails on the same instance removes only its
    outputs; the outer key's staged outputs wait for the outer outcome."""
    from datajoint.errors import DataJointError

    seen = {}

    class Nested(_table_class(tmp_path)):
        def make_fetch(self, key):
            self.fetch_calls += 1
            if key["id"] == "inner":
                # The inner key's second fetch differs, so it is refused.
                return ("parent", self.fetch_calls)
            if self.fetch_calls == 2:
                # The outer key's in-transaction fetch runs a nested populate
                # after the outer compute has staged its outputs.
                seen["inner"] = _populate1(
                    self, {"id": "inner"}, suppress_errors=True
                )
                seen["outer_staged"] = (tmp_path / "outer.nwb").exists()
                seen["inner_staged"] = (tmp_path / "inner.nwb").exists()
                if outer_fails:
                    return ("parent", "changed")
            return ("parent", "same")

    table = Nested()

    if outer_fails:
        with pytest.raises(DataJointError, match="Referential integrity"):
            _populate1(table, {"id": "outer"})
    else:
        assert _populate1(table, {"id": "outer"}) is True

    assert seen["inner"][0] == {"id": "inner"}
    assert seen["inner_staged"] is False
    assert seen["outer_staged"] is True
    assert (tmp_path / "outer.nwb").exists() is not outer_fails
    assert (tmp_path / "outer.bundle").exists() is not outer_fails
    assert table.owners["outer.nwb"].closed == int(outer_fails)
    assert table.owners["inner.nwb"].closed == 1
    assert table._staged_output_scopes == []


def test_direct_compute_outside_populate_records_nothing(tmp_path):
    """Only ``_populate1`` records; a direct ``make_compute`` is untouched."""
    table = _table_class(tmp_path)()

    table.make_compute({"id": "a"}, "parent")

    assert "_staged_output_scopes" not in vars(table)
    assert (tmp_path / "a.nwb").exists()


def test_wrapping_keeps_generator_make_and_signatures(tmp_path):
    """The tri-part dispatch and the insert signature survive wrapping."""
    import inspect

    table_cls = _table_class(tmp_path)

    assert inspect.isgeneratorfunction(table_cls.make)
    assert list(inspect.signature(table_cls.make_insert).parameters) == [
        "self",
        "key",
        "analysis_file_name",
        "owner",
        "bundle",
    ]

    class Child(table_cls):
        pass

    # A subclass that does not redefine the methods is not wrapped twice.
    assert Child.make_compute is table_cls.make_compute


@pytest.fixture
def analysis_file_resolver(tmp_path, monkeypatch):
    """Supply file paths without importing or declaring a DataJoint schema."""
    import sys
    from types import ModuleType

    calls = []

    class AnalysisNwbfile:
        @staticmethod
        def get_abs_path(name):
            calls.append(name)
            return str(tmp_path / name)

    module = ModuleType("spyglass.common.common_nwbfile")
    module.AnalysisNwbfile = AnalysisNwbfile
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return AnalysisNwbfile, calls


@pytest.mark.parametrize("present", [False, True])
def test_staged_analysis_cleanup_is_idempotent(
    tmp_path, analysis_file_resolver, present
):
    from spyglass.spikesorting.v2._storage.staged_outputs import (
        unlink_staged_analysis_file,
    )

    staged = tmp_path / "attempt.nwb"
    if present:
        staged.write_bytes(b"unregistered attempt")

    unlink_staged_analysis_file(staged.name, context="writer failed")
    unlink_staged_analysis_file(staged.name, context="writer failed")

    assert not staged.exists()


def test_partial_artifact_cleanup_preserves_canonical_file(
    tmp_path, analysis_file_resolver, caplog
):
    """The recording-writer wrapper refuses canonical files before lookup."""
    from spyglass.spikesorting.v2._storage.nwb import _remove_partial_artifact

    canonical = tmp_path / "canonical.nwb"
    canonical.write_bytes(b"canonical artifact")

    _remove_partial_artifact(canonical.name, canonical.name)

    assert canonical.read_bytes() == b"canonical artifact"
    assert analysis_file_resolver[1] == []
    assert "Recording._write_nwb_artifact" in caplog.text
    assert "refusing to unlink a canonical artifact" in caplog.text

    partial = tmp_path / "partial.nwb"
    partial.write_bytes(b"failed attempt")
    _remove_partial_artifact(partial.name, None)
    assert not partial.exists()


def test_units_writer_failure_cleans_staged_file_without_recording_schema(
    tmp_path, monkeypatch, analysis_file_resolver
):
    """A units writer's failure cleanup needs no recording table import."""
    import builtins

    from spyglass.spikesorting.v2._storage import units_nwb as _units_nwb

    staged = tmp_path / "units.nwb"

    def create(self, **kwargs):
        staged.write_bytes(b"unregistered units")
        return staged.name

    monkeypatch.setattr(
        analysis_file_resolver[0], "create", create, raising=False
    )
    original_error = RuntimeError("units write failed")

    def fail_write(**kwargs):
        assert staged.exists()
        raise original_error

    monkeypatch.setattr(_units_nwb, "_write_sorting_units_nwb_body", fail_write)
    import_module = builtins.__import__

    def refuse_recording(name, *args, **kwargs):
        if name == "spyglass.spikesorting.v2.recording":
            raise AssertionError("cleanup must not import the recording schema")
        return import_module(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse_recording)

    with pytest.raises(RuntimeError, match="units write failed") as error:
        _units_nwb.write_sorting_units_nwb(None, None, "parent.nwb")

    assert error.value is original_error
    assert not staged.exists()


@pytest.mark.parametrize("failure", ["lookup", "unlink"])
def test_analysis_cleanup_failure_preserves_populate_error(
    tmp_path, monkeypatch, analysis_file_resolver, caplog, failure
):
    """Cleanup failure is logged while the populate's original error escapes."""
    from spyglass.spikesorting.v2._storage.staged_outputs import (
        StagedOutputCleanupMixin,
    )

    cleanup_error = PermissionError(f"{failure} denied")
    if failure == "lookup":

        def deny_lookup(name):
            raise cleanup_error

        monkeypatch.setattr(
            analysis_file_resolver[0], "get_abs_path", deny_lookup
        )
    else:
        unlink = Path.unlink

        def deny_unlink(path, *args, **kwargs):
            if path == tmp_path / "a.nwb":
                raise cleanup_error
            return unlink(path, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", deny_unlink)

    original_error = RuntimeError("insert failed")
    table_cls = _table_class(tmp_path, insert_error=original_error)
    # Use the production cleanup path rather than this stand-in's normal
    # direct unlink, so the failure exercises lazy resolution and logging.
    table_cls._unlink_analysis_file = staticmethod(
        StagedOutputCleanupMixin._unlink_analysis_file
    )
    table = table_cls()

    with pytest.raises(RuntimeError, match="insert failed") as error:
        _populate1(table, {"id": "a"})

    assert error.value is original_error
    assert (tmp_path / "a.nwb").exists()
    assert not (tmp_path / "a.build").exists()
    assert not (tmp_path / "a.bundle").exists()
    assert "Table.populate" in caplog.text
    assert "a.nwb" in caplog.text
    assert f"{failure} denied" in caplog.text

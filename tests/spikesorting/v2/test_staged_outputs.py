"""v2's opt-in to the shared staged-output cleanup.

DB-free: a stand-in table runs DataJoint's real ``_populate1`` against a fake
connection. The cleanup itself is tested in
``tests/utils/test_populate_staged_outputs.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import pytest


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
    """Build a stand-in v2 tri-part table staging a file, owner and folder."""
    from datajoint.autopopulate import AutoPopulate

    from spyglass.spikesorting.v2._storage.staged_outputs import (
        StagedOutputCleanupMixin,
        StagedOutputs,
    )
    from spyglass.utils.mixins.populate import PopulateMixin

    class Computed(NamedTuple):
        analysis_file_name: str
        owner: _Owner
        bundle: str

        def staged_outputs(self):
            # The stand-in's file is not an AnalysisNwbfile, so it is a path.
            return StagedOutputs(
                paths=(str(tmp_path / self.analysis_file_name), self.bundle),
                owners=(self.owner,),
            )

    class Table(StagedOutputCleanupMixin, PopulateMixin, AutoPopulate):
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

    return Table


def _populate1(table, key, *, jobs=None, suppress_errors=False):
    return table._populate1(
        key,
        jobs,
        suppress_errors=suppress_errors,
        return_exception_objects=False,
    )


@pytest.mark.parametrize("insert_fails", [False, True])
def test_carrier_staged_outputs_reach_shared_cleanup(tmp_path, insert_fails):
    """``StagedOutputCleanupMixin`` hands a carrier's ``staged_outputs()`` to
    ``PopulateMixin``: kept on success, removed when the insert fails."""
    error = RuntimeError("duplicate") if insert_fails else None
    table = _table_class(tmp_path, insert_error=error)()

    if insert_fails:
        with pytest.raises(RuntimeError, match="duplicate"):
            _populate1(table, {"id": "a"})
    else:
        assert _populate1(table, {"id": "a"}) is True

    for suffix in (".nwb", ".build", ".bundle"):
        assert (tmp_path / f"a{suffix}").exists() is not insert_fails
    assert table.owners["a.nwb"].closed == int(insert_fails)
    assert table._staged_output_scopes == []


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
        from tests.spikesorting.v2._provenance_helpers import sorting_provenance

        _units_nwb.write_sorting_units_nwb(
            None, None, "parent.nwb", source_provenance=sorting_provenance()
        )

    assert error.value is original_error
    assert not staged.exists()

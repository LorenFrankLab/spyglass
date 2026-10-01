"""Fixtures shared by the plan-reuse tests."""

import shutil

import pytest


@pytest.fixture
def parse_counter(monkeypatch):
    """Count `_parse` calls per table, so reuse is observable.

    Timing would be the obvious measure and the wrong one: it varies, and a
    reuse check that silently re-parsed would still look fast on a small file.
    """
    from spyglass.utils.mixins.ingestion import IngestionMixin

    calls = []
    original = IngestionMixin._parse

    def counted(self, ctx):
        calls.append(self.full_table_name)
        return original(self, ctx)

    monkeypatch.setattr(IngestionMixin, "_parse", counted)

    return calls


@pytest.fixture
def editable_copy(common, mini_path, mini_copy_name, raw_dir, mini_insert):
    """A private copy of the stripped mini file, safe to edit in place.

    The shared fixture file cannot be the subject: these tests mutate the
    file, and a stray edit -- or a stray mtime, which the external-link
    digest reads -- would change what every later test hashes.

    Registered in `Nwbfile`, as it would be on a real retry: ingestion
    creates that row before it reads anything, and a registered name is what
    lets the planner resolve the path and re-hash it after each edit without
    the test holding the file open across its own writes.

    Yields
    ------
    tuple of (str, pathlib.Path)
        The copy's `nwb_file_name` and its path on disk.
    """
    from spyglass.common.common_nwbfile import Nwbfile
    from spyglass.common.common_usage import IngestionPlanLog

    source = raw_dir / mini_copy_name
    assert source.exists(), f"Premise: {source} exists after mini_insert"

    name = "ffireuseedit_.nwb"
    path = raw_dir / name
    shutil.copy2(source, path)
    Nwbfile.insert_from_relative_file_name(name)

    yield name, path

    IngestionPlanLog().clear(name)
    (Nwbfile & {"nwb_file_name": name}).delete_quick()
    path.unlink(missing_ok=True)

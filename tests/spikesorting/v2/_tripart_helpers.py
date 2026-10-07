"""Checks shared by the tests of v2 tri-part ``make`` tables.

DataJoint runs ``make_compute`` outside the populate transaction and protects
only what ``make_fetch`` read: it hashes the fetched carrier on both of its
fetches and refuses the insert if the hashes differ
(``datajoint/autopopulate.py``). These helpers reproduce that hash and fail
any query issued where no DB access is allowed.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager


def fetch_hash(fetched):
    """The hash DataJoint compares across a tri-part make's two fetches."""
    from deepdiff import DeepHash

    return DeepHash(fetched, ignore_iterable_order=False)[fetched]


def _inside_analysis_nwbfile() -> bool:
    """Whether a caller frame is running an ``AnalysisNwbfile`` method.

    Covers the table's construction, ``create`` and ``get_abs_path`` (and the
    ``Nwbfile`` lookup ``create`` makes), i.e. staging an output file.
    """
    from spyglass.common.common_nwbfile import AnalysisNwbfile

    frame = sys._getframe(1)
    while frame is not None:
        local = frame.f_locals
        if isinstance(local.get("self"), AnalysisNwbfile) or (
            local.get("cls") is AnalysisNwbfile
        ):
            return True
        frame = frame.f_back
    return False


@contextmanager
def forbid_db_queries(monkeypatch, label: str, *, allow_staging=False):
    """Raise on any DataJoint query issued inside the block.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
    label : str
        Names the code under test in the failure message.
    allow_staging : bool, optional
        Allow staging an output file, the one DB access a ``make_compute``
        may keep. Two rules apply:

        - Queries are let through when ANY frame up the stack is an
          ``AnalysisNwbfile`` method (bound ``self`` or ``cls``). This is
          stack-based: it covers ``AnalysisNwbfile()`` construction,
          ``create`` (and the ``Nwbfile`` lookup it makes) and
          ``get_abs_path``, whatever the caller.
        - ``AnalysisNwbfile.get_abs_path`` may resolve only a file that
          ``AnalysisNwbfile().create`` returned inside the block (or be called
          from within ``create``). Resolving any other analysis file -- an
          input that belongs in ``make_fetch`` -- raises.
    """
    import datajoint as dj

    from spyglass.common.common_nwbfile import AnalysisNwbfile

    real = dj.Connection.query

    def _guard(self, query, *args, **kwargs):
        if allow_staging and _inside_analysis_nwbfile():
            return real(self, query, *args, **kwargs)
        raise AssertionError(f"{label} queried the DB: {query}")

    with monkeypatch.context() as patch:
        patch.setattr(dj.Connection, "query", _guard)
        if allow_staging:
            staged: set = set()
            creating = [0]
            real_create = AnalysisNwbfile.create
            real_get_abs_path = AnalysisNwbfile.get_abs_path.__func__

            def _create(self, *args, **kwargs):
                creating[0] += 1
                try:
                    name = real_create(self, *args, **kwargs)
                finally:
                    creating[0] -= 1
                staged.add(name)
                return name

            def _get_abs_path(cls, name, *args, **kwargs):
                if not creating[0] and name not in staged:
                    raise AssertionError(
                        f"{label} resolved an analysis file it did not "
                        f"stage: {name}"
                    )
                return real_get_abs_path(cls, name, *args, **kwargs)

            patch.setattr(AnalysisNwbfile, "create", _create)
            patch.setattr(
                AnalysisNwbfile, "get_abs_path", classmethod(_get_abs_path)
            )
        yield


def change_second_fetch(monkeypatch, table) -> None:
    """Make ``table``'s in-transaction ``make_fetch`` return changed data.

    DataJoint runs ``make_fetch`` once before ``make_compute`` and again inside
    the insert transaction, and refuses the insert when the two differ. Every
    second call here returns the fetched carrier with one extra element, as if
    a parent row changed while ``make_compute`` ran, so populate raises after
    ``make_compute`` has staged its outputs and before ``make_insert`` runs.
    """
    real = table.make_fetch
    calls = [0]

    def _fetch(self, key, **kwargs):
        fetched = real(self, key, **kwargs)
        calls[0] += 1
        if calls[0] % 2 == 0:
            return (*fetched, "changed while make_compute ran")
        return fetched

    monkeypatch.setattr(table, "make_fetch", _fetch)


def record_created_analysis_files(monkeypatch) -> list[str]:
    """Record the name of every ``AnalysisNwbfile`` created from now on."""
    from spyglass.common.common_nwbfile import AnalysisNwbfile

    created: list[str] = []
    real_create = AnalysisNwbfile.create

    def _create(self, *args, **kwargs):
        name = real_create(self, *args, **kwargs)
        created.append(name)
        return name

    monkeypatch.setattr(AnalysisNwbfile, "create", _create)
    return created


def assert_no_staged_analysis_files(created: list[str]) -> None:
    """Fail if any file in ``created`` is on disk or has an AnalysisNwbfile row."""
    from pathlib import Path

    from spyglass.common.common_nwbfile import AnalysisNwbfile

    assert created, "precondition: make_compute must have staged a file"
    on_disk = [
        name
        for name in created
        if Path(AnalysisNwbfile.get_abs_path(name)).exists()
    ]
    registered = [
        name
        for name in created
        if AnalysisNwbfile & {"analysis_file_name": name}
    ]
    assert not on_disk, f"staged files left on disk: {on_disk}"
    assert not registered, f"staged files registered: {registered}"

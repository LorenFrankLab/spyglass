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
        Let through queries made inside an ``AnalysisNwbfile`` method
        (``AnalysisNwbfile().create`` / ``get_abs_path``): staging an output
        file is the one DB access a ``make_compute`` may keep.
    """
    import datajoint as dj

    real = dj.Connection.query

    def _guard(self, query, *args, **kwargs):
        if allow_staging and _inside_analysis_nwbfile():
            return real(self, query, *args, **kwargs)
        raise AssertionError(f"{label} queried the DB: {query}")

    with monkeypatch.context() as patch:
        patch.setattr(dj.Connection, "query", _guard)
        yield

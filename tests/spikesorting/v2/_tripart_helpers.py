"""Checks shared by the tests of v2 tri-part ``make`` tables.

DataJoint runs ``make_compute`` outside the populate transaction and protects
only what ``make_fetch`` read: it hashes the fetched carrier on both of its
fetches and refuses the insert if the hashes differ
(``datajoint/autopopulate.py``). These helpers reproduce that hash and fail
any query issued where no DB access is allowed.
"""

from __future__ import annotations

from contextlib import contextmanager


def fetch_hash(fetched):
    """The hash DataJoint compares across a tri-part make's two fetches."""
    from deepdiff import DeepHash

    return DeepHash(fetched, ignore_iterable_order=False)[fetched]


@contextmanager
def forbid_db_queries(monkeypatch, label: str):
    """Raise on any DataJoint query issued inside the block.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
    label : str
        Names the code under test in the failure message.
    """
    import datajoint as dj

    def _boom(self, query, *args, **kwargs):
        raise AssertionError(f"{label} queried the DB: {query}")

    with monkeypatch.context() as patch:
        patch.setattr(dj.Connection, "query", _boom)
        yield

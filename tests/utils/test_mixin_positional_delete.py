"""Tests for positional restrictions passed to ``delete``.

``Table().delete(restriction)`` applies ``restriction`` before the permission
check and the delete. Delete options are keyword-only, and a positional bool
raises rather than being read as a flag.
"""

import datajoint as dj
import pytest
from datajoint.condition import AndList

SCHEMA_NAME = "test_positional_delete"


@pytest.fixture(scope="module")
def del_schema(dj_conn, teardown):
    """Declare a plain table and a one-source merge table."""
    from spyglass.utils import SpyglassMixin
    from spyglass.utils.dj_merge_tables import _Merge

    class DelTable(SpyglassMixin, dj.Manual):
        definition = """
        id: int
        """

    class DelParent(SpyglassMixin, dj.Manual):
        definition = """
        parent_id: int
        """

    class DelMerge(_Merge, SpyglassMixin):
        definition = """
        merge_id: uuid
        ---
        source: varchar(32)
        """

        class DelParent(SpyglassMixin, dj.Part):
            definition = """
            -> master
            ---
            -> DelParent
            """

    context = dict(DelTable=DelTable, DelParent=DelParent, DelMerge=DelMerge)
    schema = dj.Schema(context=context)
    for table in context.values():
        schema(table)
    schema.activate(SCHEMA_NAME, connection=dj_conn)

    yield context

    if teardown:
        schema.drop(force=True)


@pytest.fixture
def table(del_schema, monkeypatch):
    """DelTable holding ids 1-3, with permission checks recorded.

    Yields
    ------
    tuple
        ``(table, checked)``: a DelTable instance and a list that receives the
        row count of each table the permission check ran on.
    """
    tbl_cls = del_schema["DelTable"]
    tbl = tbl_cls()
    tbl.super_delete(warn=False, safemode=False)
    tbl.insert([{"id": i} for i in (1, 2, 3)])

    checked = []
    monkeypatch.setattr(
        tbl_cls,
        "_check_delete_permission",
        lambda self: checked.append(len(self)),
    )
    monkeypatch.setitem(dj.config, "safemode", False)

    yield tbl, checked


def _ids(tbl):
    return set(tbl.fetch("id"))


@pytest.mark.parametrize(
    "restrictions",
    [
        ({"id": 1},),
        ("id = 1",),
        ([{"id": 1}],),
        (AndList(["id < 3", "id != 2"]),),
        ("id < 3", "id != 2"),
    ],
    ids=["dict", "str", "list", "andlist", "multiple"],
)
def test_delete_positional_restriction(table, restrictions):
    """A positional restriction limits the delete and its permission check."""
    tbl, checked = table

    tbl.delete(*restrictions, safemode=False)

    assert _ids(tbl) == {2, 3}, "Positional restriction not applied"
    assert checked == [1], "Permission check not run on the restricted rows"


def test_delete_query_expression_restriction(table, del_schema):
    """A query expression is accepted as a positional restriction."""
    tbl, checked = table
    parent = del_schema["DelParent"]()
    parent.insert1({"parent_id": 1}, skip_duplicates=True)

    tbl.delete(parent.proj(id="parent_id") & {"id": 1}, safemode=False)

    assert _ids(tbl) == {2, 3}, "Query restriction not applied"
    assert checked == [1], "Permission check not run on the restricted rows"


def test_delete_positional_with_force_permission(table):
    """``force_permission`` stays usable by keyword after a restriction."""
    tbl, checked = table

    tbl.delete({"id": 1}, force_permission=True, safemode=False)

    assert _ids(tbl) == {2, 3}, "Positional restriction not applied"
    assert checked == [], "force_permission=True still ran the check"


def test_cautious_delete_dry_run_positional(table):
    """A dry run with a positional restriction checks those rows only."""
    tbl, checked = table

    ret = tbl.cautious_delete({"id": 1}, dry_run=True)

    assert ret is not None, "Dry run did not return its summary"
    assert _ids(tbl) == {1, 2, 3}, "Dry run deleted rows"
    assert checked == [1], "Dry run did not check the restricted rows"


def test_super_delete_positional_restriction(table):
    """``super_delete`` applies a positional restriction too."""
    tbl, _ = table

    tbl.super_delete({"id": 1}, warn=False, safemode=False)

    assert _ids(tbl) == {2, 3}, "Positional restriction not applied"


@pytest.mark.parametrize(
    "method", ["delete", "cautious_delete", "super_delete"]
)
@pytest.mark.parametrize("flag", [True, False])
def test_positional_bool_raises(table, method, flag):
    """A positional bool is refused instead of read as a delete flag."""
    tbl, _ = table

    with pytest.raises(TypeError, match="by keyword"):
        getattr(tbl, method)(flag, safemode=False)

    assert _ids(tbl) == {1, 2, 3}, "Rows deleted despite the TypeError"


@pytest.fixture
def merge(del_schema, monkeypatch):
    """DelMerge with one entry per DelParent row (ids 1-3)."""
    parent, merge_cls = del_schema["DelParent"](), del_schema["DelMerge"]
    parent.super_delete(warn=False, safemode=False, force_masters=True)
    parent.insert([{"parent_id": i} for i in (1, 2, 3)])
    merge_cls.insert(parent.fetch("KEY"), skip_duplicates=True)
    monkeypatch.setitem(dj.config, "safemode", False)

    yield merge_cls()


def _merge_parent_ids(merge):
    return set((merge * merge.DelParent()).fetch("parent_id"))


def test_merge_delete_positional_restriction(merge, monkeypatch):
    """``Merge.delete`` applies a positional restriction to its entries.

    The part delete is replaced by a direct ``Table.delete``: a
    ``SpyglassMixin, dj.Part`` part reaches ``dj.Part.delete``, which rejects
    the ``force_parts`` that ``Merge.delete`` passes.
    """
    part_calls = []

    def part_delete(self, force_permission=False, **kwargs):
        part_calls.append((len(self), force_permission))
        dj.Table.delete(self, **kwargs)

    monkeypatch.setattr(type(merge.DelParent()), "delete", part_delete)
    merge_id = (merge.DelParent() & {"parent_id": 1}).fetch1("merge_id")

    merge.delete({"merge_id": merge_id}, force_permission=True, safemode=False)

    assert part_calls == [(1, True)], "Parts not restricted or flag dropped"
    assert len(merge) == 2, "Positional restriction not applied"
    assert _merge_parent_ids(merge) == {2, 3}, "Wrong merge entry deleted"


def test_merge_super_delete_positional_restriction(merge):
    """``Merge.super_delete`` applies a positional restriction."""
    merge_id = (merge.DelParent() & {"parent_id": 1}).fetch1("merge_id")

    merge.super_delete({"merge_id": merge_id}, warn=False, safemode=False)

    assert _merge_parent_ids(merge) == {2, 3}, "Wrong merge entry deleted"


@pytest.mark.parametrize("method", ["delete", "super_delete"])
def test_merge_positional_bool_raises(merge, method):
    """``Merge`` deletes refuse a positional bool."""
    with pytest.raises(TypeError, match="by keyword"):
        getattr(merge, method)(True, safemode=False)

    assert len(merge) == 3, "Merge entries deleted despite the TypeError"


@pytest.mark.parametrize(
    "module",
    [
        "spyglass.spikesorting.v0.spikesorting_recompute",
        "spyglass.spikesorting.v1.recompute",
    ],
)
def test_recompute_delete_restricts_before_attempts(dj_conn, module):
    """Recompute deletes restrict before collecting attempt files.

    Otherwise a positional restriction would remove the attempt files of every
    row while deleting only the restricted rows.
    """
    from importlib import import_module

    calls = []

    class Restricted:
        def delete(self, **kwargs):
            calls.append(kwargs)

    class Unrestricted:
        def _restrict_delete_args(self, restrictions):
            calls.append(restrictions)
            return Restricted()

        def __iter__(self):
            raise AssertionError("Attempt files collected before restricting")

    table_cls = import_module(module).RecordingRecompute
    # The override itself, called on a stand-in for the unrestricted table
    table_cls.__dict__["delete"](Unrestricted(), {"id": 1}, safemode=False)

    assert calls == [({"id": 1},), {"safemode": False}], "Restriction not first"

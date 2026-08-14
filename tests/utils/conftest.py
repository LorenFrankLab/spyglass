import datajoint as dj
import pytest


@pytest.fixture(scope="session")  # Elevated from module scope for performance
def merge_table(pos_merge_tables):
    """Return the merge table as activated."""
    yield pos_merge_tables[0]


@pytest.fixture(scope="session")  # Elevated from module scope for performance
def Nwbfile(pos_merge_tables):
    """Return the Nwbfile table as activated."""
    from spyglass.common import Nwbfile as NwbfileTable

    return NwbfileTable()


@pytest.fixture(scope="module")
def merge_repro_schema(SpyglassMixin, _Merge):
    """Two-part merge table, isolated from graph_schema and every real
    pipeline — no FK ties to ParentNode/IntermediateNode/etc. (which would
    perturb TableChain graph traversal elsewhere, as reusing PkNode/SkNode
    did) and no optional dependencies like deeplabcut (which pos_merge's
    DLCPosV1 part needs, and isn't installed in every dev env).

    ``part_attr`` is part of each source's primary key so it propagates into
    the merge Part table's own heading — a bare ``-> Source`` FK only pulls
    primary-key attributes. Sources are heterogeneous (``a_id`` vs ``b_id``)
    so multi-part fetch has to reconcile differing field names. ``SourceA``/
    ``SourceB`` are ``Computed`` so ``populate()`` is exercisable.
    """

    class SourceALookup(SpyglassMixin, dj.Lookup):
        definition = """
        a_id: int
        part_attr: int
        """
        contents = [(1, 1), (2, 2)]

    class SourceBLookup(SpyglassMixin, dj.Lookup):
        definition = """
        b_id: int
        part_attr: int
        """
        contents = [(1, 1)]

    class SourceA(SpyglassMixin, dj.Computed):
        definition = """
        -> SourceALookup
        ---
        a_computed: int
        """

        def make(self, key):
            self.insert1(dict(key, a_computed=key["a_id"] * 10))

    class SourceB(SpyglassMixin, dj.Computed):
        definition = """
        -> SourceBLookup
        ---
        b_computed: int
        """

        def make(self, key):
            self.insert1(dict(key, b_computed=key["b_id"] * 100))

    class MergeRepro(_Merge, SpyglassMixin):
        definition = """
        merge_id: uuid
        ---
        source: varchar(32)
        """

        class SourceA(SpyglassMixin, dj.Part):
            definition = """
            -> master
            ---
            -> SourceA
            """

        class SourceB(SpyglassMixin, dj.Part):
            definition = """
            -> master
            ---
            -> SourceB
            """

    yield {
        "SourceALookup": SourceALookup,
        "SourceBLookup": SourceBLookup,
        "SourceA": SourceA,
        "SourceB": SourceB,
        "MergeRepro": MergeRepro,
    }


@pytest.fixture(scope="module")
def merge_repro_activated(dj_conn, merge_repro_schema):
    """Declare and activate the merge_repro schema once per test module."""
    schema = dj.Schema(context=merge_repro_schema)
    for table in merge_repro_schema.values():
        schema(table)
    schema.activate("test_merge_repro", connection=dj_conn)

    yield merge_repro_schema

    schema.drop(force=True)


@pytest.fixture(scope="function")
def merge_repro(merge_repro_activated):
    """Reset merge_repro to a deterministic 3-row/2-part populated state.

    2 SourceA rows + 1 SourceB row; ``part_attr=1`` matches one row of each
    part, mirroring the reviewer's original repro counts.
    """
    SourceA = merge_repro_activated["SourceA"]
    SourceB = merge_repro_activated["SourceB"]
    MergeRepro = merge_repro_activated["MergeRepro"]

    MergeRepro().super_delete(warn=False, safemode=False)
    SourceA().delete_quick()
    SourceB().delete_quick()

    SourceA().populate()
    SourceB().populate()

    MergeRepro().insert(
        SourceA().fetch("KEY", as_dict=True), part_name="SourceA"
    )
    MergeRepro().insert(
        SourceB().fetch("KEY", as_dict=True), part_name="SourceB"
    )

    yield merge_repro_activated


@pytest.fixture(scope="module")
def schema_test(teardown, dj_conn):
    """This fixture is used to create a schema

    Adapted from datajoint/conftest.py.
    """
    schema_test = dj.Schema("test_conftest", {}, connection=dj_conn)
    # schema_any(Test) # Declare table using schema_any as func
    yield schema_test
    if teardown:
        schema_test.drop(force=True)


@pytest.fixture(scope="module")
def chain(Nwbfile):
    """Return example TableChain object from chains."""
    from spyglass.linearization.merge import (
        LinearizedPositionOutput,  # noqa: F401
    )
    from spyglass.utils.dj_graph import TableChain

    yield TableChain(Nwbfile, LinearizedPositionOutput)


@pytest.fixture(scope="module")
def no_link_chain(Nwbfile):
    """Return example TableChain object with no link."""
    from spyglass.common.common_usage import InsertError
    from spyglass.utils.dj_graph import TableChain

    yield TableChain(Nwbfile, InsertError())


@pytest.fixture(scope="session")  # Elevated from module scope for performance
def _Merge():
    """Return the _Merge class."""
    from spyglass.utils import _Merge

    yield _Merge


@pytest.fixture(scope="session")  # Elevated from module scope for performance
def SpyglassMixin():
    """Return a mixin class."""
    from spyglass.utils import SpyglassMixin

    yield SpyglassMixin


@pytest.fixture(scope="session")  # Elevated from module scope for performance
def graph_schema(SpyglassMixin, _Merge):
    """
    NOTE: Must declare tables within fixture to avoid loading config defaults.
    """
    parent_id = range(10)
    parent_attr = [i + 10 for i in range(2, 12)]
    other_id = range(9)
    other_attr = [i + 10 for i in range(3, 12)]
    intermediate_id = range(2, 10)
    intermediate_attr = [i + 10 for i in range(4, 12)]
    pk_id = range(3, 10)
    pk_attr = [i + 10 for i in range(5, 12)]
    sk_id = range(6)
    sk_attr = [i + 10 for i in range(6, 12)]
    pk_sk_id = range(5)
    pk_sk_attr = [i + 10 for i in range(7, 12)]
    pk_alias_id = range(4)
    pk_alias_attr = [i + 10 for i in range(8, 12)]
    sk_alias_id = range(3)
    sk_alias_attr = [i + 10 for i in range(9, 12)]

    def offset(gen, offset):
        return list(gen)[offset:]

    class ParentNode(SpyglassMixin, dj.Lookup):
        definition = """
        parent_id: int
        ---
        parent_attr : int
        """
        contents = [(i, j) for i, j in zip(parent_id, parent_attr)]

    class OtherParentNode(SpyglassMixin, dj.Lookup):
        definition = """
        other_id: int
        ---
        other_attr : int
        """
        contents = [(i, j) for i, j in zip(other_id, other_attr)]

    class IntermediateNode(SpyglassMixin, dj.Lookup):
        definition = """
        intermediate_id: int
        ---
        -> ParentNode
        intermediate_attr : int
        """
        contents = [
            (i, j, k)
            for i, j, k in zip(
                intermediate_id, offset(parent_id, 1), intermediate_attr
            )
        ]

    class BranchNode(SpyglassMixin, dj.Lookup):
        definition = """
        -> IntermediateNode
        ---
        -> ParentNode
        """
        contents = [(intermediate_id[0], parent_id[3])]

    class PkNode(SpyglassMixin, dj.Lookup):
        definition = """
        pk_id: int
        -> IntermediateNode
        ---
        pk_attr : int
        """
        contents = [
            (i, j, k)
            for i, j, k in zip(pk_id, offset(intermediate_id, 2), pk_attr)
        ]

    class SkNode(SpyglassMixin, dj.Lookup):
        definition = """
        sk_id: int
        ---
        -> IntermediateNode
        sk_attr : int
        """
        contents = [
            (i, j, k)
            for i, j, k in zip(sk_id, offset(intermediate_id, 3), sk_attr)
        ]

    class PkSkNode(SpyglassMixin, dj.Lookup):
        definition = """
        pk_sk_id: int
        -> IntermediateNode
        ---
        -> OtherParentNode
        pk_sk_attr : int
        """
        contents = [
            (i, j, k, m)
            for i, j, k, m in zip(
                pk_sk_id, offset(intermediate_id, 4), other_id, pk_sk_attr
            )
        ]

    class PkAliasNode(SpyglassMixin, dj.Lookup):
        definition = """
        pk_alias_id: int
        -> PkNode.proj(fk_pk_id='pk_id')
        ---
        pk_alias_attr : int
        """
        contents = [
            (i, j, k, m)
            for i, j, k, m in zip(
                pk_alias_id,
                offset(pk_id, 1),
                offset(intermediate_id, 3),
                pk_alias_attr,
            )
        ]

    class SkAliasNode(SpyglassMixin, dj.Lookup):
        definition = """
        sk_alias_id: int
        ---
        -> SkNode.proj(fk_sk_id='sk_id')
        -> PkSkNode
        sk_alias_attr : int
        """
        contents = [
            (i, j, k, m, n)
            for i, j, k, m, n in zip(
                sk_alias_id,
                offset(sk_id, 2),
                offset(pk_sk_id, 1),
                offset(intermediate_id, 5),
                sk_alias_attr,
            )
        ]

    class MergeOutput(_Merge, SpyglassMixin):
        definition = """
        merge_id: uuid
        ---
        source: varchar(32)
        """

        class PkNode(dj.Part):
            definition = """
            -> MergeOutput
            ---
            -> PkNode
            """

    class MergeChild(SpyglassMixin, dj.Manual):
        definition = """
        -> MergeOutput
        merge_child_id: int
        ---
        merge_child_attr: int
        """

    yield {
        "ParentNode": ParentNode,
        "OtherParentNode": OtherParentNode,
        "IntermediateNode": IntermediateNode,
        "BranchNode": BranchNode,
        "PkNode": PkNode,
        "SkNode": SkNode,
        "PkSkNode": PkSkNode,
        "PkAliasNode": PkAliasNode,
        "SkAliasNode": SkAliasNode,
        "MergeOutput": MergeOutput,
        "MergeChild": MergeChild,
    }


@pytest.fixture(scope="module")
def graph_tables(dj_conn, graph_schema):
    schema = dj.Schema(context=graph_schema)

    for table in graph_schema.values():
        schema(table)

    schema.activate("test_graph", connection=dj_conn)

    # Merge inserts after declaring tables
    merge_keys = graph_schema["PkNode"].fetch("KEY", offset=1, as_dict=True)
    graph_schema["MergeOutput"].insert(merge_keys, skip_duplicates=True)
    merge_child_keys = graph_schema["MergeOutput"]().fetch("merge_id", offset=1)
    merge_child_inserts = [
        (i, j, k + 10)
        for i, j, k in zip(merge_child_keys, range(4), range(10, 15))
    ]
    graph_schema["MergeChild"]().insert(
        merge_child_inserts, skip_duplicates=True
    )

    yield graph_schema

    schema.drop(force=True)


@pytest.fixture(scope="module")
def graph_tables_many_to_one(graph_tables):
    ParentNode = graph_tables["ParentNode"]
    IntermediateNode = graph_tables["IntermediateNode"]
    PkSkNode = graph_tables["PkSkNode"]

    pk_sk_keys = PkSkNode().fetch(as_dict=True)[-2:]
    new_inserts = [
        {
            "pk_sk_id": k["pk_sk_id"] + 3,
            "intermediate_id": k["intermediate_id"] + 3,
            "intermediate_attr": k["intermediate_id"] + 16,
            "parent_id": k["intermediate_id"] - 1,
            "parent_attr": k["intermediate_id"] + 11,
            "other_id": k["other_id"],  # No change
            "pk_sk_attr": k["pk_sk_attr"] + 10,
        }
        for k in pk_sk_keys
    ]

    insert_kwargs = {"ignore_extra_fields": True, "skip_duplicates": True}
    ParentNode.insert(new_inserts, **insert_kwargs)
    IntermediateNode.insert(new_inserts, **insert_kwargs)
    PkSkNode.insert(new_inserts, **insert_kwargs)

    yield graph_tables


@pytest.fixture(scope="module")
def add_graph_tables(SpyglassMixin):
    schema = dj.Schema("test_add_graphs")

    @schema
    class A(SpyglassMixin, dj.Lookup):
        definition = """
        a_id: int
        ---
        a_attr: int
        """
        contents = [(i, i + 10) for i in range(5)]

    @schema
    class B1(SpyglassMixin, dj.Lookup):
        definition = """
        -> A
        ---
        b_attr: int
        """
        contents = [(i, i + 20) for i in range(5)]

    @schema
    class B2(SpyglassMixin, dj.Lookup):
        definition = """
        -> A
        ---
        b_attr: int
        """
        contents = [(i, i + 30) for i in range(5)]

    return {
        "A": A(),
        "B1": B1(),
        "B2": B2(),
    }

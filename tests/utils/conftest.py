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
    merge_child_keys = graph_schema["MergeOutput"]().merge_fetch(
        "merge_id", restriction=True, offset=1
    )
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
def two_parent_tables(SpyglassMixin, dj_conn):
    """Tables where an attr name is shared without being the linking foreign key.

    `TpChild` reaches `TpParent` two ways: directly, and via `TpMid`. Because
    `tp_parent_id` is secondary on `TpMid` but primary on `TpChild`, a
    restriction on `TpMid.tp_parent_id` cascading down to `TpChild` must join on
    `tp_mid_id` (the actual foreign key) rather than reuse `tp_parent_id` by
    name. Contents are chosen so the two answers differ in length.

    Declared in a dedicated schema so the extra edges cannot perturb the
    path-finding of tests built on `graph_tables`.
    """
    schema = dj.Schema("test_two_parent", connection=dj_conn)

    @schema
    class TpParent(SpyglassMixin, dj.Lookup):
        definition = """
        tp_parent_id: int
        ---
        tp_parent_attr: int
        """
        contents = [(i, i + 10) for i in range(10)]

    @schema
    class TpMid(SpyglassMixin, dj.Lookup):
        definition = """
        tp_mid_id: int
        ---
        -> TpParent
        tp_mid_attr: int
        """
        contents = [(i, i - 1, i + 12) for i in range(2, 10)]

    @schema
    class TpChild(SpyglassMixin, dj.Lookup):
        definition = """
        -> TpMid
        -> TpParent
        ---
        tp_child_attr: int
        """
        # tp_mid_id=4 maps to tp_parent_id=3 via TpMid. Rows below intentionally
        # disagree with that mapping so a name-based shortcut is detectable:
        #   join on tp_mid_id=4 -> 3 rows; copy of `tp_parent_id = 3` -> 2 rows
        contents = [
            (4, 3, 30),  # both
            (5, 3, 31),  # only a `tp_parent_id = 3` copy
            (4, 8, 32),  # only the correct join
            (6, 8, 33),  # neither
            (4, 9, 34),  # only the correct join
        ]

    yield {"TpParent": TpParent(), "TpMid": TpMid(), "TpChild": TpChild()}

    schema.drop(force=True)


@pytest.fixture(scope="function")
def mutable_graph_tables(SpyglassMixin, dj_conn):
    """Small parent/child schema that tests may insert into.

    Function-scoped and dropped on teardown so row-count assertions elsewhere
    are unaffected by the inserts.
    """
    schema = dj.Schema("test_mutable_graph", connection=dj_conn)

    @schema
    class MutParent(SpyglassMixin, dj.Lookup):
        definition = """
        mut_id: int
        ---
        mut_attr: int
        """
        contents = [(i, i + 10) for i in range(3)]

    @schema
    class MutChild(SpyglassMixin, dj.Lookup):
        definition = """
        -> MutParent
        ---
        mut_child_attr: int
        """
        contents = [(i, i + 20) for i in range(3)]

    yield {"MutParent": MutParent(), "MutChild": MutChild()}

    schema.drop(force=True)


@pytest.fixture(scope="function")
def redeclare_table(SpyglassMixin, dj_conn):
    """Factory declaring the same table name with a caller-supplied definition.

    Used to prove that dropping and re-creating a table is not served a stale
    heading from any FreeTable cache.
    """
    schema_name = "test_redeclare_graph"
    made = []

    def _make(definition, contents):
        schema = dj.Schema(schema_name, connection=dj_conn)

        class Redeclared(SpyglassMixin, dj.Lookup):
            pass

        Redeclared.definition = definition
        Redeclared.contents = contents
        schema(Redeclared)
        made.append(schema)
        return schema, Redeclared()

    yield _make

    for schema in made:
        schema.drop(force=True)


@pytest.fixture(scope="function")
def query_counter(dj_conn):
    """Count SQL statements issued on the shared connection.

    Turns 'is the cascade cheaper?' into an assertion that does not depend on
    wall-clock timing.
    """

    class Counter:
        def __init__(self):
            self.queries = []

        def __len__(self):
            return len(self.queries)

        def reset(self):
            self.queries = []

        def matching(self, substring):
            """Return queries containing substring, case-insensitive."""
            low = substring.lower()
            return [q for q in self.queries if low in q.lower()]

    counter = Counter()
    original = dj_conn.query

    def counting_query(query, *args, **kwargs):
        counter.queries.append(str(query))
        return original(query, *args, **kwargs)

    dj_conn.query = counting_query  # instance attr shadows the bound method

    yield counter

    del dj_conn.query  # restore the class-level method


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

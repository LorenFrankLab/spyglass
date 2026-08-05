"""Regression suite guarding the `dj_graph` cascade optimizations.

These tests characterize current cascade behavior so that performance work on
`RestrGraph` cannot silently change which rows a restriction reaches. They are
deliberately independent of the NWB data pipeline: every fixture is a small,
self-contained DataJoint schema.
"""

import pytest


@pytest.fixture(scope="session")
def RestrGraph():
    from spyglass.utils.dj_graph import RestrGraph

    return RestrGraph


@pytest.fixture(scope="session")
def AbstractGraph():
    from spyglass.utils.dj_graph import AbstractGraph

    return AbstractGraph


def cascade_signature(graph):
    """Return {table_name: sorted primary keys} for every restricted table.

    A stronger equality check than row counts: two cascades agree only if they
    reach the same rows of the same tables.
    """
    return {
        ft.full_table_name: sorted(
            tuple(sorted(row.items())) for row in ft.fetch("KEY")
        )
        for ft in graph.restr_ft
    }


def assert_cascade_parity(AbstractGraph, build, msg=""):
    """Assert a cascade yields identical rows with and without `_fast_bridge`.

    `build` is a zero-argument callable returning a cascaded graph. Until a
    short-circuit is added to `_bridge_restr`, both runs exercise the same code
    and this asserts determinism; afterwards it asserts optimized == legacy.
    """
    fast = cascade_signature(build())

    original = AbstractGraph._fast_bridge
    AbstractGraph._fast_bridge = False
    try:
        legacy = cascade_signature(build())
    finally:
        AbstractGraph._fast_bridge = original

    assert fast.keys() == legacy.keys(), (
        f"Fast and legacy cascades restricted different tables. {msg}\n"
        + f"\tFast only: {set(fast) - set(legacy)}\n"
        + f"\tLegacy only: {set(legacy) - set(fast)}"
    )
    for table in fast:
        assert (
            fast[table] == legacy[table]
        ), f"Fast and legacy cascades disagree on {table}. {msg}"


# ------------------------------ Bridge rule matrix ------------------------------
# (leaf, restriction, direction, target, expected rows, description)
BRIDGE_CASES = [
    (
        "PkNode",
        "intermediate_id = 5",
        "up",
        "IntermediateNode",
        1,
        "pk attr up",
    ),
    (
        "PkNode",
        "intermediate_id = 5",
        "up",
        "ParentNode",
        1,
        "pk attr up 2 hop",
    ),
    ("PkNode", "pk_attr > 18", "up", "IntermediateNode", 2, "sec attr up"),
    ("PkNode", "pk_attr > 18", "up", "ParentNode", 2, "sec attr up 2 hop"),
    ("PkAliasNode", "pk_alias_id < 2", "up", "PkNode", 2, "aliased up"),
    ("IntermediateNode", "intermediate_id = 5", "down", "PkNode", 1, "pk down"),
    (
        "ParentNode",
        "parent_attr > 19",
        "down",
        "IntermediateNode",
        1,
        "sec down",
    ),
    ("MergeChild", "merge_child_attr > 21", "up", "MergeOutput", 2, "merge up"),
]


@pytest.mark.parametrize(
    "leaf, restr, direction, target, expect_n, msg", BRIDGE_CASES
)
def test_bridge_rule_matrix(
    RestrGraph, graph_tables, leaf, restr, direction, target, expect_n, msg
):
    """Exact row counts for each shape of edge a bridge must handle.

    Unlike `test_restr_from_upstream`/`_downstream`, which reach the graph via
    `>>`/`<<` (the `TableChain` path, using `_get_adjacent_path_item`), these
    drive `RestrGraph` directly so `_get_next_tables` is the code under test.
    """
    graph = RestrGraph(
        seed_table=graph_tables[leaf](),
        leaves=[
            {
                "table_name": graph_tables[leaf].full_table_name,
                "restriction": restr,
            }
        ],
        direction=direction,
        cascade=True,
        verbose=False,
    )
    ft = graph._get_ft(graph_tables[target].full_table_name, with_restr=True)
    assert len(ft) == expect_n, f"Unexpected cascade result for {msg}."


@pytest.mark.parametrize(
    "leaf, restr, direction, target, expect_n, msg", BRIDGE_CASES
)
def test_bridge_rule_matrix_parity(
    AbstractGraph,
    RestrGraph,
    graph_tables,
    leaf,
    restr,
    direction,
    target,
    expect_n,
    msg,
):
    """Every bridge shape must survive the fast/legacy toggle unchanged."""
    _ = target, expect_n

    def build():
        return RestrGraph(
            seed_table=graph_tables[leaf](),
            leaves=[
                {
                    "table_name": graph_tables[leaf].full_table_name,
                    "restriction": restr,
                }
            ],
            direction=direction,
            cascade=True,
            verbose=False,
        )

    assert_cascade_parity(AbstractGraph, build, msg=f"Case: {msg}.")


def test_bridge_query_expression_restr(RestrGraph, graph_tables):
    """A leaf restricted by a QueryExpression, as later hops supply."""
    PkNode = graph_tables["PkNode"]()
    graph = RestrGraph(
        seed_table=PkNode,
        leaves=[
            {
                "table_name": PkNode.full_table_name,
                "restriction": PkNode & "pk_attr > 18",
            }
        ],
        direction="up",
        cascade=True,
        verbose=False,
    )
    ft = graph._get_ft(
        graph_tables["IntermediateNode"].full_table_name, with_restr=True
    )
    assert len(ft) == 2, "QueryExpression restriction did not cascade."


def test_bridge_false_restr(RestrGraph, graph_tables):
    """A False restriction must stop the cascade, not propagate everything."""
    graph = RestrGraph(
        seed_table=graph_tables["PkNode"](),
        leaves=[
            {
                "table_name": graph_tables["PkNode"].full_table_name,
                "restriction": False,
            }
        ],
        direction="up",
        cascade=True,
        verbose=False,
    )
    ft = graph._get_ft(
        graph_tables["IntermediateNode"].full_table_name, with_restr=True
    )
    assert len(ft) == 0, "False restriction leaked rows into the cascade."


def test_bridge_shared_attr_not_foreign_key(RestrGraph, two_parent_tables):
    """A shared attribute name must not be mistaken for the linking key.

    `TpChild.tp_parent_id` is primary but is its own foreign key to `TpParent`,
    not the key linking it to `TpMid`. Cascading a `tp_parent_id` restriction
    from `TpMid` must join on `tp_mid_id`. Copying the condition by name would
    return the two rows where `tp_parent_id = 3` instead of the three rows
    reachable through `tp_mid_id`.
    """
    TpMid, TpChild = two_parent_tables["TpMid"], two_parent_tables["TpChild"]

    graph = RestrGraph(
        seed_table=TpMid,
        leaves=[
            {
                "table_name": TpMid.full_table_name,
                "restriction": "tp_parent_id = 3",
            }
        ],
        direction="down",
        cascade=True,
        verbose=False,
    )
    got = graph._get_ft(TpChild.full_table_name, with_restr=True)

    assert len(TpChild & "tp_parent_id = 3") == 2, "Fixture assumption changed."
    assert len(got) == 3, (
        "Cascade must join on the foreign key, not reuse `tp_parent_id` by "
        + "name. Got the rows a name-based copy would return."
    )


# ------------------------- Convergence / union semantics ------------------------


@pytest.mark.parametrize(
    "restr, expect_ids, msg",
    [
        ("intermediate_id = 2", {1, 3}, "both paths contribute"),
        ("intermediate_id = 2 AND parent_id = 3", {1, 3}, "compound restr"),
    ],
)
def test_convergence_union(RestrGraph, graph_tables, restr, expect_ids, msg):
    """Two paths converging on one ancestor must OR their restrictions.

    `BranchNode` reaches `ParentNode` directly and through `IntermediateNode`,
    so the ancestor restriction is the union of both arrivals. Reordering the
    traversal must not turn this union into an intersection.
    """
    graph = RestrGraph(
        seed_table=graph_tables["BranchNode"](),
        leaves=[
            {
                "table_name": graph_tables["BranchNode"].full_table_name,
                "restriction": restr,
            }
        ],
        direction="up",
        cascade=True,
        verbose=False,
    )
    got = set(
        graph._get_ft(
            graph_tables["ParentNode"].full_table_name, with_restr=True
        ).fetch("parent_id")
    )
    assert got == expect_ids, f"Convergence union failed: {msg}."


def test_convergence_included_tables(RestrGraph, graph_tables):
    """The set of tables reached by a cascade is part of the contract."""
    graph = RestrGraph(
        seed_table=graph_tables["BranchNode"](),
        leaves=[
            {
                "table_name": graph_tables["BranchNode"].full_table_name,
                "restriction": "intermediate_id = 2",
            }
        ],
        direction="up",
        cascade=True,
        verbose=False,
    )
    included = {t for t in graph.included_tables if not t.isnumeric()}
    for name in ("BranchNode", "IntermediateNode", "ParentNode"):
        assert (
            graph_tables[name].full_table_name in included
        ), f"{name} missing from cascade."


# ------------------------------ Hash order invariance ---------------------------


def test_hash_leaf_order_invariant(RestrGraph, add_graph_tables):
    """`RestrGraph.hash` must not depend on the order leaves are supplied.

    `all_ft` is topologically sorted, and `dj_topo_sort` ordering feeds
    `_hash_upstream` (see the NOTE in `dj_graph.py`). Reordering the traversal
    must leave the hash stable.
    """
    tables = add_graph_tables
    leaves = [
        {"table_name": tables["B1"].full_table_name, "restriction": "a_id < 2"},
        {"table_name": tables["B2"].full_table_name, "restriction": "a_id > 2"},
    ]

    def build(ordered):
        return RestrGraph(
            seed_table=tables["B1"],
            leaves=ordered,
            direction="up",
            cascade=True,
            verbose=False,
        )

    assert (
        build(leaves).hash == build(list(reversed(leaves))).hash
    ), "Graph hash changed with leaf order."


# --------------------------------- Verbose parity -------------------------------


def test_verbose_parity(RestrGraph, graph_tables, caplog):
    """Verbose logging must not change what a cascade returns.

    No other fixture builds a graph with `verbose=True`, so the logging branch
    of `_bridge_restr` is otherwise untested.
    """

    def build(verbose):
        return RestrGraph(
            seed_table=graph_tables["PkNode"](),
            leaves=[
                {
                    "table_name": graph_tables["PkNode"].full_table_name,
                    "restriction": "pk_attr > 16",
                }
            ],
            direction="up",
            cascade=True,
            verbose=verbose,
        )

    quiet = cascade_signature(build(False))
    with caplog.at_level("INFO", logger="spyglass"):
        loud = cascade_signature(build(True))

    assert quiet == loud, "Verbose cascade returned different rows."
    assert "Bridge Link" in caplog.text, "Verbose cascade logged no bridges."


# ----------------------------- Cache invalidation -------------------------------


def test_new_graph_sees_new_rows(RestrGraph, mutable_graph_tables):
    """Rows inserted between two graphs must be visible to the second.

    Caching table emptiness or restricted results across graphs must not
    outlive the data it describes.
    """
    parent = mutable_graph_tables["MutParent"]
    child = mutable_graph_tables["MutChild"]

    def build():
        return RestrGraph(
            seed_table=child,
            leaves=[
                {
                    "table_name": child.full_table_name,
                    "restriction": "mut_id > 1",
                }
            ],
            direction="up",
            cascade=True,
            verbose=False,
        )

    before = len(build()._get_ft(parent.full_table_name, with_restr=True))

    parent.insert([(9, 99)], skip_duplicates=True)
    child.insert([(9, 99)], skip_duplicates=True)

    after = len(build()._get_ft(parent.full_table_name, with_restr=True))

    assert after == before + 1, "New graph did not see rows inserted since."


def test_empty_table_then_populated(RestrGraph, mutable_graph_tables):
    """An empty table must not be cached as permanently empty."""
    parent = mutable_graph_tables["MutParent"]
    child = mutable_graph_tables["MutChild"]

    child.delete_quick()

    def build():
        return RestrGraph(
            seed_table=parent,
            leaves=[
                {
                    "table_name": parent.full_table_name,
                    "restriction": "mut_id < 2",
                }
            ],
            direction="down",
            cascade=True,
            verbose=False,
        )

    assert (
        len(build()._get_ft(child.full_table_name, with_restr=True)) == 0
    ), "Fixture assumption changed: child should be empty."

    child.insert([(0, 20), (1, 21)], skip_duplicates=True)

    assert (
        len(build()._get_ft(child.full_table_name, with_restr=True)) == 2
    ), "Cascade treated a repopulated table as still empty."


def test_redeclared_table_heading(RestrGraph, redeclare_table):
    """A dropped and re-created table must not be served a stale heading.

    Guards any process-wide FreeTable cache. Module-scoped fixtures in this
    suite drop and re-create their schemas, so a cache keyed on table name
    alone would hand the second declaration the first one's heading.
    """
    first = """
    redeclared_id: int
    ---
    first_attr: int
    """
    second = """
    redeclared_id: int
    ---
    second_attr: int
    """

    schema, table = redeclare_table(first, [(0, 10)])
    graph = RestrGraph(seed_table=table, leaves=[], verbose=False)
    names = set(graph._get_ft(table.full_table_name).heading.names)
    assert "first_attr" in names, "Fixture assumption changed."
    schema.drop(force=True)

    _, table = redeclare_table(second, [(0, 20)])
    graph = RestrGraph(seed_table=table, leaves=[], verbose=False)
    names = set(graph._get_ft(table.full_table_name).heading.names)

    assert "second_attr" in names, "Stale heading served after redeclaration."
    assert (
        "first_attr" not in names
    ), "Stale heading served after redeclaration."


# ----------------------------- Edge resolution ----------------------------------


def test_get_edge_alias(RestrGraph, graph_tables):
    """`_get_edge` must resolve an aliased edge through its alias node.

    The fallback that handles this walks `all_simple_paths` over the whole
    graph. Bounding that walk must not change the resolution.
    """
    graph = RestrGraph(seed_table=graph_tables["PkNode"](), verbose=False)
    _, edge = graph._get_edge(
        graph_tables["PkAliasNode"].full_table_name,
        graph_tables["PkNode"].full_table_name,
    )
    assert "fk_pk_id" in edge.get("attr_map", {}), "Alias edge attr_map lost."


def test_get_edge_direct(RestrGraph, graph_tables):
    """A direct edge resolves without the path-search fallback.

    Pins the polarity of the returned flag, which `_bridge_restr` maps straight
    onto a cascade direction. Despite the parameter names, the flag is False
    when the arguments are ordered (child, parent) and True when they are
    swapped -- it reports "arguments are reversed", not "child is a child".
    """
    graph = RestrGraph(seed_table=graph_tables["PkNode"](), verbose=False)
    pk_node = graph_tables["PkNode"].full_table_name
    intermediate = graph_tables["IntermediateNode"].full_table_name

    reversed_args, edge = graph._get_edge(pk_node, intermediate)
    assert not reversed_args, "Flag set for correctly ordered (child, parent)."
    assert "intermediate_id" in edge.get("attr_map", {}), "attr_map lost."

    reversed_args, edge = graph._get_edge(intermediate, pk_node)
    assert reversed_args, "Flag unset for swapped (parent, child)."
    assert "intermediate_id" in edge.get("attr_map", {}), "attr_map lost."


# -------------------------------- Query budget ----------------------------------

# Ceilings recorded against pre-optimization code; tighten as cascade
# performance work lands. The point is to catch a regression that adds queries
# per edge, not to pin an exact count. Construction, cascade, and realizing the
# results are measured separately: the first two scale with the traversal, the
# third currently scales with the size of the whole database.
GRAPH_BUILD_QUERY_CEILING = 60
CASCADE_QUERY_CEILING = 40


def _budget_graph(RestrGraph, graph_tables):
    """Build an uncascaded graph over a fixed leaf."""
    return RestrGraph(
        seed_table=graph_tables["PkNode"](),
        leaves=[
            {
                "table_name": graph_tables["PkNode"].full_table_name,
                "restriction": "pk_attr > 16",
            }
        ],
        direction="up",
        cascade=False,
        verbose=False,
    )


def test_graph_build_query_budget(RestrGraph, graph_tables, query_counter):
    """Bound the cost of constructing a graph, before any cascade.

    Every `RestrGraph` reloads and copies the dependency graph, and multi-leaf
    cascades build one graph per leaf, so this cost is paid repeatedly.
    """
    _budget_graph(RestrGraph, graph_tables)
    measured = len(query_counter)

    assert measured < GRAPH_BUILD_QUERY_CEILING, (
        f"Graph construction issued {measured} queries, ceiling is "
        + f"{GRAPH_BUILD_QUERY_CEILING}."
    )


def test_cascade_query_budget(RestrGraph, graph_tables, query_counter):
    """Bound the SQL statements one cascade issues, excluding construction.

    Deterministic stand-in for a timing benchmark: cascade performance is
    largely a function of how many statements it issues.
    """
    graph = _budget_graph(RestrGraph, graph_tables)

    query_counter.reset()  # exclude construction, measured separately
    graph.cascade()
    measured = len(query_counter)

    assert measured < CASCADE_QUERY_CEILING, (
        f"Cascade issued {measured} queries, ceiling is "
        + f"{CASCADE_QUERY_CEILING}. If this is an intended trade, update the "
        + "ceiling; if not, an optimization added per-edge queries."
    )


def test_included_tables_limited_to_cascade(RestrGraph, graph_tables):
    """Only tables the cascade actually reached should be reported as included.

    Node truthiness cannot be used to decide this: DataJoint gives every table
    of every loaded schema a `primary_key` node attribute, so a truthiness
    check reports the whole database.
    """
    graph = _budget_graph(RestrGraph, graph_tables)
    graph.cascade()

    reached = graph.visited | graph.leaves
    extra = {t for t in graph.included_tables if not t.isnumeric()} - reached

    assert not extra, (
        f"{len(extra)} tables reported as included were never visited, "
        + f"e.g. {sorted(extra)[:3]}"
    )


def test_restr_ft_scales_with_cascade(RestrGraph, graph_tables, query_counter):
    """Realizing results costs queries per restricted table, not per DB table.

    `all_ft` builds a FreeTable and runs an existence query for every table
    `included_tables` reports, so the two must stay proportional to the
    cascade rather than to the size of the session.
    """
    graph = _budget_graph(RestrGraph, graph_tables)
    graph.cascade()

    query_counter.reset()
    restricted = graph.restr_ft
    measured = len(query_counter)

    budget = 10 * max(len(restricted), 1)
    assert measured <= budget, (
        f"Realizing {len(restricted)} restricted tables issued {measured} "
        + f"queries, budget is {budget}."
    )


def test_verbose_query_overhead(RestrGraph, graph_tables, query_counter):
    """Verbose cascades must not cost dramatically more queries than quiet ones.

    Debug instrumentation in `_bridge_restr` runs an anti-join over the
    unrestricted target table once per edge. This test records that overhead so
    a regression reintroducing it is visible.
    """

    def run(verbose):
        graph = _budget_graph(RestrGraph, graph_tables)
        graph.verbose = verbose
        query_counter.reset()  # exclude construction from the comparison
        graph.cascade()
        return len(query_counter)

    quiet = run(False)
    loud = run(True)

    assert loud <= quiet * 3, (
        f"Verbose cascade issued {loud} queries vs {quiet} quiet. Debug "
        + "instrumentation should not dominate the cascade."
    )

import pytest

from . import schema_graph as sg


@pytest.fixture(scope="session")
def leaf(lin_merge):
    yield lin_merge.LinearizedPositionV1()


@pytest.fixture(scope="session")
def restr_graph(leaf):
    from spyglass.utils.dj_graph import RestrGraph

    yield RestrGraph(
        seed_table=leaf,
        table_name=leaf.full_table_name,
        restriction=True,
        cascade=True,
        verbose=True,
    )


def test_rg_repr(restr_graph, leaf):
    """Test that the repr of a RestrGraph object is as expected."""
    repr_got = repr(restr_graph)

    assert "cascade" in repr_got.lower(), "Cascade not in repr."
    assert leaf.full_table_name in repr_got, "Table name not in repr."


def test_rg_ft(restr_graph):
    """Test FreeTable attribute of RestrGraph."""
    assert len(restr_graph.leaf_ft) == 1, "Unexpected number of leaf tables."
    assert len(restr_graph.all_ft) == 9, "Unexpected number of cascaded tables."


def test_rg_restr_ft(restr_graph):
    """Test get restricted free tables."""
    ft = restr_graph._get_ft(list(restr_graph.visited)[1], with_restr=True)
    assert len(ft) == 1, "Unexpected restricted table length."


def test_rg_file_paths(restr_graph):
    """Test collection of upstream file paths."""
    paths = [p.get("file_path") for p in restr_graph.file_paths]
    assert len(paths) == 1, "Unexpected number of file paths."


@pytest.fixture(scope="session")
def restr_graph_new_leaf(restr_graph, common):
    restr_graph.add_leaf(
        table_name=common.common_behav.PositionSource.full_table_name,
        restriction=True,
    )

    yield restr_graph


def test_add_leaf_cascade(restr_graph_new_leaf):
    assert (
        not restr_graph_new_leaf.cascaded
    ), "Cascaded flag not set when add leaf."


def test_add_leaf_restr_ft(restr_graph_new_leaf):
    restr_graph_new_leaf.cascade()
    ft = restr_graph_new_leaf._get_ft(
        "`common_interval`.`interval_list`", with_restr=True
    )
    assert len(ft) == 2, "Unexpected restricted table length."


@pytest.fixture(scope="session")
def restr_graph_root(restr_graph, common, lfp_band):
    from spyglass.utils.dj_graph import RestrGraph

    yield RestrGraph(
        seed_table=common.Session(),
        table_name=common.Session.full_table_name,
        restriction="True",
        direction="down",
        cascade=True,
        verbose=False,
    )


def test_rg_root(restr_graph_root):
    assert (
        len(restr_graph_root.all_ft) == 29
    ), "Unexpected number of cascaded tables."


@pytest.mark.parametrize(
    "restr, expect_n, msg",
    [
        ("pk_attr > 16", 4, "pk no alias"),
        ("sk_attr > 17", 3, "sk no alias"),
        ("pk_alias_attr > 18", 3, "pk pk alias"),
        ("sk_alias_attr > 19", 2, "sk sk alias"),
        ("merge_child_attr > 21", 2, "merge child down"),
    ],
)
def test_restr_from_upstream(graph_tables, restr, expect_n, msg):
    msg = "Error in `>>` for " + msg
    assert len(graph_tables["ParentNode"]() >> restr) == expect_n, msg


@pytest.mark.parametrize(
    "table, restr, expect_n, msg",
    [
        ("PkNode", "parent_attr > 15", 5, "pk no alias"),
        ("SkNode", "parent_attr > 16", 4, "sk no alias"),
        ("PkAliasNode", "parent_attr > 17", 2, "pk pk alias"),
        ("SkAliasNode", "parent_attr > 18", 2, "sk sk alias"),
        ("MergeChild", "parent_attr > 18", 2, "merge child"),
    ],
)
def test_restr_from_downstream(graph_tables, table, restr, expect_n, msg):
    msg = "Error in `<<` for " + msg
    assert len(graph_tables[table]() << restr) == expect_n, msg

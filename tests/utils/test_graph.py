import pytest
from datajoint.utils import to_camel_case

from tests.conftest import VERBOSE


@pytest.fixture(scope="session")
def leaf(lin_merge):
    yield lin_merge.LinearizedPositionV1()


@pytest.fixture(scope="session")
def restr_graph(leaf, lin_merge_key):
    from spyglass.utils.dj_graph import RestrGraph

    _ = lin_merge_key  # linearization merge table populated

    yield RestrGraph(
        seed_table=leaf,
        leaves={leaf.full_table_name: True},
        include_files=True,
        cascade=True,
        verbose=False,
    )


@pytest.fixture(scope="function")
def add_graph_rgs(add_graph_tables):
    """Return two RestrGraph objects to test addition."""
    from spyglass.utils.dj_graph import RestrGraph

    tables = add_graph_tables

    restr_1 = "a_id < 2"
    restr_2 = {
        "a_id": 2,
    }
    restr_3 = {
        "a_id": 3,
    }

    rg_1 = RestrGraph(
        seed_table=add_graph_tables["B1"],
        leaves={tables["B1"].full_table_name: restr_1},
        direction="up",
        cascade=True,
        verbose=False,
    )
    rg_1.cascade()

    rg_2 = RestrGraph(
        seed_table=add_graph_tables["B2"],
        direction="up",
        cascade=True,
        verbose=False,
    )
    rg_2.add_leaf(table_name=tables["B2"].full_table_name, restriction=restr_2)
    rg_2.cascade()

    rg_3 = RestrGraph(
        seed_table=add_graph_tables["B2"],
        direction="up",
        cascade=True,
        verbose=False,
    )
    rg_3.add_leaf(table_name=tables["B2"].full_table_name, restriction=restr_3)
    rg_3.cascade()

    yield rg_1, rg_2, rg_3


def test_rg_add(add_graph_rgs, add_graph_tables):
    """Test adding tables to RestrGraph."""
    tables = add_graph_tables
    rg_1, rg_2, _ = add_graph_rgs

    assert (
        len(rg_1._get_ft(tables["A"].full_table_name, with_restr=True)) == 2
    ), "Unexpected restricted table length for rg_1."
    assert (
        len(rg_2._get_ft(tables["A"].full_table_name, with_restr=True)) == 1
    ), "Unexpected restricted table length for rg_2."

    rg_union = rg_1 + rg_2

    assert (
        len(rg_union._get_ft(tables["A"].full_table_name, with_restr=True)) == 3
    ), (
        "Unexpected parent restricted table length for union of rg_1 and rg_2."
        + f" IDs: {rg_union._get_ft(tables['A'].full_table_name, with_restr=True).fetch('a_id')}"
    )
    assert (
        len(rg_union._get_ft(tables["B1"].full_table_name, with_restr=True))
        == 2
    ), "Unexpected child restricted table length for union of rg_1 and rg_2."


def test_rg_add_list(add_graph_rgs, add_graph_tables):
    """Test adding tables to RestrGraph."""
    tables = add_graph_tables
    rg_1, rg_2, rg_3 = add_graph_rgs

    assert (
        len(rg_1._get_ft(tables["A"].full_table_name, with_restr=True)) == 2
    ), "Unexpected restricted table length for rg_1."
    assert (
        len(rg_2._get_ft(tables["A"].full_table_name, with_restr=True)) == 1
    ), "Unexpected restricted table length for rg_2."

    rg_union = rg_1 + [rg_2, rg_3]

    assert (
        len(rg_union._get_ft(tables["A"].full_table_name, with_restr=True)) == 4
    ), (
        "Unexpected parent restricted table length for union of rg_1 and rg_2."
        + f" IDs: {rg_union._get_ft(tables['A'].full_table_name, with_restr=True).fetch('a_id')}"
    )
    assert (
        len(rg_union._get_ft(tables["B1"].full_table_name, with_restr=True))
        == 2
    ), "Unexpected child restricted table length for union of rg_1 and rg_2."


def test_rg_repr(restr_graph, leaf):
    """Test that the repr of a RestrGraph object is as expected."""
    repr_got = repr(restr_graph)

    assert "cascade" in repr_got.lower(), "Cascade not in repr."

    assert to_camel_case(leaf.table_name) in repr_got, "Table name not in repr."


def test_rg_len(restr_graph):
    assert len(restr_graph) == len(
        restr_graph.restr_ft
    ), "Unexpected length of RestrGraph."


def test_rg_ft(restr_graph):
    """Test FreeTable attribute of RestrGraph."""
    assert len(restr_graph.leaf_ft) == 1, "Unexpected # of leaf tables."


def test_rg_restr_ft(restr_graph):
    """Test get restricted free tables."""
    ft = restr_graph["spatial_series"]
    assert len(ft) == 2, "Unexpected restricted table length."


def test_rg_file_paths(restr_graph):
    """Test collection of upstream file paths.

    NOTE: This test previously tested how many files were collected, which may
    differ if only subset of tests are run. Instead, we now check which tables
    store collected files. See #1440, #1534 for context.
    """
    expected_tbls = [
        "`position_linearization_v1`.`__linearized_position_v1`",
        "`position_v1_trodes_position`.`__trodes_pos_v1`",
    ]
    stored_files = restr_graph._stored_files(as_dict=True)
    for tbl in expected_tbls:
        assert tbl in stored_files, f"Expected table {tbl} did not show file."

    assert len(restr_graph.file_paths) > 1, "Unexpected file paths collected."


def test_rg_invalid_table(restr_graph):
    """Test that an invalid table raises an error."""
    with pytest.raises(ValueError):
        restr_graph._get_node("invalid_table")


def test_rg_invalid_edge(restr_graph, Nwbfile, common):
    """Test that an invalid edge raises an error."""
    with pytest.raises(ValueError):
        restr_graph._get_edge(Nwbfile, common.common_behav.PositionSource)


def test_rg_restr_subset(restr_graph, leaf):
    prev_ft = restr_graph._get_ft(leaf.full_table_name, with_restr=True)

    restr_graph._set_restr(leaf, restriction=False)

    new_ft = restr_graph._get_ft(leaf.full_table_name, with_restr=True)
    assert len(prev_ft) == len(new_ft), "Subset sestriction changed length."


def test_rg_no_restr(caplog, restr_graph, common):
    restr_graph._set_restr(common.LabTeam, restriction=False)
    ret = restr_graph._get_ft(common.LabTeam.full_table_name, with_restr=True)
    assert not ret, "Expected empty restricted table when no restriction."


def test_rg_invalid_direction(restr_graph, leaf):
    """Test that an invalid direction raises an error."""
    with pytest.raises(ValueError):
        restr_graph._get_next_tables(leaf.full_table_name, "invalid_direction")


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
def restr_graph_root(restr_graph, common, lfp_band, lin_v1, frequent_imports):
    from spyglass.utils.dj_graph import RestrGraph

    _ = lfp_band, lin_v1, frequent_imports  # tables populated

    yield RestrGraph(
        seed_table=common.Session(),
        leaves={common.Session.full_table_name: "True"},
        direction="down",
        cascade=True,
        verbose=False,
    )


def test_rg_root(restr_graph_root):
    assert (
        len(restr_graph_root["trodes_pos_v1"]) >= 1
    ), "Incomplete cascade from root."


@pytest.mark.parametrize(
    "restr, expect_n, msg",
    [
        ("pk_attr > 16", 4, "pk no alias"),
        ("sk_attr > 17", 3, "sk no alias"),
        ("pk_alias_attr > 18", 3, "pk pk alias"),
        ("sk_alias_attr > 19", 2, "sk sk alias"),
        ("merge_child_attr > 21", 2, "merge child down"),
        ({"merge_child_attr": 21}, 1, "dict restr"),
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
        ("MergeChild", {"parent_attr": 19}, 1, "dict restr"),
    ],
)
def test_restr_from_downstream(graph_tables, table, restr, expect_n, msg):
    msg = "Error in `<<` for " + msg
    assert len(graph_tables[table]() << restr) == expect_n, msg


@pytest.mark.skipif(not VERBOSE, reason="No logging to test when quiet-spy.")
def test_ban_node(caplog, graph_tables):
    search_restr = "sk_attr > 17"
    ParentNode = graph_tables["ParentNode"]()
    SkNode = graph_tables["SkNode"]()

    ParentNode.ban_search_table(SkNode)
    ParentNode >> search_restr
    assert "could not be applied" in caplog.text, "Found banned table."

    ParentNode.see_banned_tables()
    assert "Banned tables" in caplog.text, "Banned tables not logged."

    ParentNode.unban_search_table(SkNode)
    assert len(ParentNode >> search_restr) == 3, "Unban failed."


def test_null_restrict_by(graph_tables):
    PkNode = graph_tables["PkNode"]()
    assert (PkNode >> True) == PkNode, "Null restriction failed."


@pytest.mark.skipif(not VERBOSE, reason="No logging to test when quiet-spy.")
def test_restrict_by_this_table(caplog, graph_tables):
    PkNode = graph_tables["PkNode"]()
    dist = (PkNode >> "pk_id > 4").restriction
    plain = (PkNode & "pk_id > 4").restriction
    assert dist == plain, "Restricting by own table did not use existing restr."


def test_invalid_restr_direction(graph_tables):
    PkNode = graph_tables["PkNode"]()
    with pytest.raises(ValueError):
        PkNode.restrict_by("bad_attr > 0", direction="invalid_direction")


def test_warn_nonrestrict(graph_tables):
    ParentNode = graph_tables["ParentNode"]()
    restr_parent = ParentNode & "parent_id > 4 AND parent_id < 9"

    ret = restr_parent >> "sk_id > 0"
    assert len(ret) == len(restr_parent), "Restriction should have no effect."

    ret = restr_parent >> "sk_id > 99"
    assert len(ret) == 0, "Return should be empty."


def test_restr_many_to_one(graph_tables_many_to_one):
    PK = graph_tables_many_to_one["PkSkNode"]()
    OP = graph_tables_many_to_one["OtherParentNode"]()

    msg_template = "Error in `%s` for many to one."

    assert len(PK << "other_attr > 14") == 4, msg_template % "<<"
    assert len(PK << {"other_attr": 15}) == 2, msg_template % "<<"
    assert len(OP >> "pk_sk_attr > 19") == 2, msg_template % ">>"
    assert (
        len(OP >> [{"pk_sk_attr": 19}, {"pk_sk_attr": 20}]) == 2
    ), "Error accepting list of dicts for `>>` for many to one."


def test_restr_invalid_err(graph_tables):
    PkNode = graph_tables["PkNode"]()
    with pytest.raises(ValueError):
        len(PkNode << set(["parent_attr > 15", "parent_attr < 20"]))


@pytest.mark.skipif(not VERBOSE, reason="No logging to test when quiet-spy.")
def test_restr_invalid(caplog, graph_tables):
    graph_tables["PkNode"]() << "invalid_restr=1"
    assert (
        "could not be applied" in caplog.text
    ), "No warning logged on invalid restr."


@pytest.fixture(scope="session")
def direction():
    from spyglass.utils.dj_graph import Direction

    yield Direction


def test_direction_str(direction):
    assert str(direction.UP) == "up", "Direction str not as expected."


def test_direction_invert(direction):
    assert ~direction.UP == direction("down"), "Direction inversion failed."


def test_direction_bool(direction):
    assert bool(direction.UP), "Direction bool not as expected."
    assert not direction.NONE, "Direction bool not as expected."

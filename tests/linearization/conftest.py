import numpy as np
import pytest


@pytest.fixture(scope="session")
def sgl(common):
    from spyglass import linearization

    yield linearization


@pytest.fixture(scope="session")
def sgpl(sgl):
    from spyglass.linearization import v1

    yield v1


@pytest.fixture(scope="session")
def pos_lin_key(trodes_sel_keys):
    yield trodes_sel_keys[-1]


@pytest.fixture(scope="session")
def position_info(pos_merge, pos_merge_key):
    yield (pos_merge & {"merge_id": pos_merge_key}).fetch1_dataframe()


@pytest.fixture(scope="session")
def track_graph_key():
    yield {"track_graph_name": "6 arm"}


@pytest.fixture(scope="session")
def track_graph(teardown, sgpl, track_graph_key):
    node_positions = np.array(
        [
            (79.910, 216.720),  # top left well 0
            (132.031, 187.806),  # top middle intersection 1
            (183.718, 217.713),  # top right well 2
            (132.544, 132.158),  # middle intersection 3
            (87.202, 101.397),  # bottom left intersection 4
            (31.340, 126.110),  # middle left well 5
            (180.337, 104.799),  # middle right intersection 6
            (92.693, 42.345),  # bottom left well 7
            (183.784, 45.375),  # bottom right well 8
            (231.338, 136.281),  # middle right well 9
        ]
    )

    edges = np.array(
        [
            (0, 1),
            (1, 2),
            (1, 3),
            (3, 4),
            (4, 5),
            (3, 6),
            (6, 9),
            (4, 7),
            (6, 8),
        ]
    )

    linear_edge_order = [
        (3, 6),
        (6, 8),
        (6, 9),
        (3, 1),
        (1, 2),
        (1, 0),
        (3, 4),
        (4, 5),
        (4, 7),
    ]
    linear_edge_spacing = 15

    sgpl.TrackGraph.insert1(
        {
            **track_graph_key,
            "environment": track_graph_key["track_graph_name"],
            "node_positions": node_positions,
            "edges": edges,
            "linear_edge_order": linear_edge_order,
            "linear_edge_spacing": linear_edge_spacing,
        },
        skip_duplicates=True,
    )

    yield sgpl.TrackGraph & {"track_graph_name": "6 arm"}
    if teardown:
        sgpl.TrackGraph().delete(safemode=False)


@pytest.fixture(scope="session")
def lin_param_key():
    yield {"linearization_param_name": "default"}


@pytest.fixture(scope="session")
def lin_params(
    teardown,
    sgpl,
    lin_param_key,
):
    param_table = sgpl.LinearizationParameters()
    param_table.insert1(lin_param_key, skip_duplicates=True)
    yield param_table


@pytest.fixture(scope="session")
def lin_sel_key(
    pos_merge_key, track_graph_key, lin_param_key, lin_params, track_graph
):
    yield {
        "pos_merge_id": pos_merge_key["merge_id"],
        **track_graph_key,
        **lin_param_key,
    }


@pytest.fixture(scope="session")
def lin_sel(teardown, sgpl, lin_sel_key):
    sel_table = sgpl.LinearizationSelection()
    sel_table.insert1(lin_sel_key, skip_duplicates=True)
    yield sel_table
    if teardown:
        sel_table.delete(safemode=False)


@pytest.fixture(scope="session")
def lin_v1(teardown, sgpl, lin_sel):
    v1 = sgpl.LinearizedPositionV1()
    v1.populate()
    yield v1
    if teardown:
        v1.delete(safemode=False)


@pytest.fixture(scope="session")
def lin_merge_key(lin_merge, lin_sel_key):
    yield lin_merge.merge_get_part(lin_sel_key).fetch1("KEY")

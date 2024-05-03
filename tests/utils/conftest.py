import datajoint as dj
import pytest

from . import schema_graph


@pytest.fixture(scope="module")
def merge_table(pos_merge_tables):
    """Return the merge table as activated."""
    yield pos_merge_tables[0]


@pytest.fixture(scope="module")
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
    # schema_any(TTest) # Declare table using schema_any as func
    yield schema_test
    if teardown:
        schema_test.drop(force=True)


@pytest.fixture(scope="module")
def chains(Nwbfile):
    """Return example TableChains object from Nwbfile."""
    from spyglass.lfp.lfp_merge import LFPOutput  # noqa: F401
    from spyglass.linearization.merge import (
        LinearizedPositionOutput,
    )  # noqa: F401
    from spyglass.position.position_merge import PositionOutput  # noqa: F401

    _ = LFPOutput, LinearizedPositionOutput, PositionOutput

    yield Nwbfile._get_chain("linear")


@pytest.fixture(scope="module")
def chain(chains):
    """Return example TableChain object from chains."""
    yield chains[0]


@pytest.fixture(scope="module")
def no_link_chain(Nwbfile):
    """Return example TableChain object with no link."""
    from spyglass.common.common_usage import InsertError
    from spyglass.utils.dj_graph import TableChain

    yield TableChain(Nwbfile, InsertError())


@pytest.fixture(scope="module")
def graph_tables(dj_conn):
    lg = schema_graph.LOCALS_GRAPH

    schema = dj.Schema(context=lg)

    for table in lg.values():
        schema(table)

    schema.activate("test_graph", connection=dj_conn)

    merge_keys = lg["PkNode"].fetch("KEY", offset=1, as_dict=True)
    lg["MergeOutput"].insert(merge_keys, skip_duplicates=True)
    merge_child_keys = lg["MergeOutput"].merge_fetch(True, "merge_id", offset=1)
    merge_child_inserts = [
        (i, j, k + 10)
        for i, j, k in zip(merge_child_keys, range(4), range(10, 15))
    ]
    lg["MergeChild"].insert(merge_child_inserts, skip_duplicates=True)

    yield schema_graph.LOCALS_GRAPH

    schema.drop(force=True)

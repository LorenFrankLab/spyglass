import datajoint as dj
import pytest


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
    yield Nwbfile._get_chain("linear")


@pytest.fixture(scope="module")
def chain(chains):
    """Return example TableChain object from chains."""
    yield chains[0]


@pytest.fixture(scope="module")
def no_link_chain(Nwbfile):
    """Return example TableChain object with no link."""
    from spyglass.common.common_usage import InsertError
    from spyglass.utils.dj_chains import TableChain

    yield TableChain(Nwbfile, InsertError())

import datajoint as dj
import pytest

from tests.conftest import VERBOSE


@pytest.fixture(scope="module")
def Mixin():
    from spyglass.utils import SpyglassMixin

    class Mixin(SpyglassMixin, dj.Manual):
        definition = """
        id : int
        """

    yield Mixin

    Mixin().drop_quick()


@pytest.mark.skipif(not VERBOSE, reason="No logging to test when quiet-spy.")
def test_bad_prefix(caplog, dj_conn, Mixin):
    schema_bad = dj.Schema("badprefix", {}, connection=dj_conn)
    schema_bad(Mixin)
    assert "Schema prefix not in SHARED_MODULES" in caplog.text


def test_nwb_table_missing(schema_test, Mixin):
    schema_test(Mixin)
    with pytest.raises(NotImplementedError):
        Mixin().fetch_nwb()


def test_merge_detect(Nwbfile, pos_merge_tables):
    """Test that the mixin can detect merge children of merge."""
    merges_found = set(Nwbfile._merge_chains.keys())
    merges_expected = set([t.full_table_name for t in pos_merge_tables])
    assert merges_expected.issubset(
        merges_found
    ), "Merges not detected by mixin."


def test_get_chain(Nwbfile, pos_merge_tables):
    """Test that the mixin can get the chain of a merge."""
    lin_parts = Nwbfile._get_chain("linear").part_names
    lin_output = pos_merge_tables[1]
    assert lin_parts == lin_output.parts(), "Chain not found."


@pytest.mark.skipif(not VERBOSE, reason="No logging to test when quiet-spy.")
def test_ddm_warning(Nwbfile, caplog):
    """Test that the mixin warns on empty delete_downstream_merge."""
    (Nwbfile & "nwb_file_name LIKE 'BadName'").delete_downstream_merge(
        reload_cache=True, disable_warnings=False
    )
    assert "No merge deletes found" in caplog.text, "No warning issued."

import datajoint as dj
import pytest

from tests.conftest import TEARDOWN, VERBOSE


@pytest.fixture(scope="module")
def Mixin():
    from spyglass.utils import SpyglassMixin

    class Mixin(SpyglassMixin, dj.Manual):
        definition = """
        id : int
        """

    yield Mixin


@pytest.mark.skipif(
    not VERBOSE or not TEARDOWN,
    reason="Error only on verbose or new declare.",
)
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


def test_merge_chain_join(Nwbfile, pos_merge_tables, lin_v1, lfp_merge_key):
    """Test that the mixin can join merge chains."""
    _ = lin_v1, lfp_merge_key  # merge tables populated

    all_chains = [
        chains.cascade(True, direction="down")
        for chains in Nwbfile._merge_chains.values()
    ]
    end_len = [len(chain[0]) for chain in all_chains if chain]

    assert sum(end_len) == 4, "Merge chains not joined correctly."


def test_get_chain(Nwbfile, pos_merge_tables):
    """Test that the mixin can get the chain of a merge."""
    lin_parts = Nwbfile._get_chain("linear").part_names
    lin_output = pos_merge_tables[1]
    assert lin_parts == lin_output.parts(), "Chain not found."


@pytest.mark.skipif(not VERBOSE, reason="No logging to test when quiet-spy.")
def test_ddm_warning(Nwbfile, caplog):
    """Test that the mixin warns on empty delete_downstream_merge."""
    (Nwbfile.file_like("BadName")).delete_downstream_merge(
        reload_cache=True, disable_warnings=False
    )
    assert "No merge deletes found" in caplog.text, "No warning issued."


def test_ddm_dry_run(Nwbfile, common, sgp, pos_merge_tables, lin_v1):
    """Test that the mixin can dry run delete_downstream_merge."""
    _ = lin_v1  # merge tables populated
    pos_output_name = pos_merge_tables[0].full_table_name

    param_field = "trodes_pos_params_name"
    trodes_params = sgp.v1.TrodesPosParams()

    rft = (trodes_params & f'{param_field} LIKE "%ups%"').ddm(
        reload_cache=True, dry_run=True, return_parts=False
    )[pos_output_name][0]
    assert len(rft) == 1, "ddm did not return restricted table."

    table_name = [p for p in pos_merge_tables[0].parts() if "trode" in p][0]
    assert table_name == rft.full_table_name, "ddm didn't grab right table."

    assert (
        rft.fetch1(param_field) == "single_led_upsampled"
    ), "ddm didn't grab right row."

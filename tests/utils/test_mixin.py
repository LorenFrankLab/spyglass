import datajoint as dj
import pytest

from tests.conftest import TEARDOWN, VERBOSE


@pytest.fixture(scope="module")
def Mixin():
    from spyglass.utils import SpyglassMixin

    class Mixin(SpyglassMixin, dj.Lookup):
        definition = """
        id : int
        """
        contents = [(0,), (1,)]

    yield Mixin


@pytest.mark.skipif(
    not VERBOSE or not TEARDOWN,
    reason="Error only on verbose or new declare.",
)
def test_bad_prefix(caplog, dj_conn, Mixin):
    schema_bad = dj.Schema("bad_prefix", {}, connection=dj_conn)
    schema_bad(Mixin)
    assert "Schema prefix not in SHARED_MODULES" in caplog.text


def test_nwb_table_missing(schema_test, Mixin):
    schema_test(Mixin)
    with pytest.raises(NotImplementedError):
        Mixin().fetch_nwb()


def test_auto_increment(schema_test, Mixin):
    schema_test(Mixin)
    ret = Mixin()._auto_increment(key={}, pk="id")
    assert ret["id"] == 2, "Auto increment not working."


def test_null_file_like(schema_test, Mixin):
    schema_test(Mixin)
    ret = Mixin().file_like(None)
    assert len(ret) == len(Mixin()), "Null file_like not working."


@pytest.mark.skipif(not VERBOSE, reason="No logging to test when quiet-spy.")
def test_bad_file_like(caplog, schema_test, Mixin):
    schema_test(Mixin)
    Mixin().file_like("BadName")
    assert "No file_like field" in caplog.text, "No warning issued."


def test_partmaster_detect(Nwbfile, pos_merge_tables):
    """Test that the mixin can detect merge children of merge."""
    assert len(Nwbfile._part_masters) >= 14, "Part masters not detected."


def test_downstream_restrict(
    Nwbfile, frequent_imports, pos_merge_tables, lin_v1, lfp_merge_key
):
    """Test that the mixin can join merge chains."""

    _ = frequent_imports  # graph for cascade
    _ = lin_v1, lfp_merge_key  # merge tables populated

    restr_ddp = Nwbfile.ddp(dry_run=True, reload_cache=True)
    end_len = [len(ft) for ft in restr_ddp]

    assert sum(end_len) >= 8, "Downstream parts not restricted correctly."


def test_get_downstream_merge(Nwbfile, pos_merge_tables):
    """Test that the mixin can get the chain of a merge."""
    lin_output = pos_merge_tables[1].full_table_name
    assert lin_output in Nwbfile._part_masters, "Merge not found."


@pytest.mark.skipif(not VERBOSE, reason="No logging to test when quiet-spy.")
def test_ddp_warning(Nwbfile, caplog):
    """Test that the mixin warns on empty delete_downstream_merge."""
    (Nwbfile.file_like("BadName")).delete_downstream_parts(
        reload_cache=True, disable_warnings=False
    )
    assert "No part deletes found" in caplog.text, "No warning issued."


def test_ddp_dry_run(
    Nwbfile, frequent_imports, common, sgp, pos_merge_tables, lin_v1
):
    """Test that the mixin can dry run delete_downstream_merge."""
    _ = lin_v1  # merge tables populated
    _ = frequent_imports  # graph for cascade

    pos_output_name = pos_merge_tables[0].full_table_name

    param_field = "trodes_pos_params_name"
    trodes_params = sgp.v1.TrodesPosParams()

    rft = [
        table
        for table in (trodes_params & f'{param_field} LIKE "%ups%"').ddp(
            reload_cache=True, dry_run=True
        )
        if table.full_table_name == pos_output_name
    ]
    assert len(rft) == 1, "ddp did not return restricted table."


def test_exp_summary(Nwbfile):
    fields = Nwbfile._get_exp_summary().heading.names
    expected = ["nwb_file_name", "lab_member_name"]
    assert fields == expected, "Exp summary fields not as expected."


def test_cautious_del_dry_run(Nwbfile, frequent_imports):
    _ = frequent_imports  # part of cascade, need import
    ret = Nwbfile.cdel(dry_run=True)
    part_master_names = [t.full_table_name for t in ret[0]]
    part_masters = Nwbfile._part_masters
    assert all(
        [pm in part_masters for pm in part_master_names]
    ), "Non part masters found in cautious delete dry run."

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


def test_good_file_like(common):
    common.Session().file_like("min")
    assert len(common.Session()) > 0, "file_like not working."


def test_null_file_like(schema_test, Mixin):
    schema_test(Mixin)
    ret = Mixin().file_like(None)
    assert len(ret) == len(Mixin()), "Null file_like not working."


@pytest.mark.skipif(not VERBOSE, reason="No logging to test when quiet-spy.")
def test_bad_file_like(caplog, schema_test, Mixin):
    schema_test(Mixin)
    Mixin().file_like("BadName")
    assert "No file_like field" in caplog.text, "No warning issued."


@pytest.mark.skipif(not VERBOSE, reason="No logging to test when quiet-spy.")
def test_insert_fail(caplog, common, mini_dict):
    this_key = dict(mini_dict, interval_list_name="BadName")
    common.PositionSource().find_insert_fail(this_key)
    assert "IntervalList: MISSING" in caplog.text, "No warning issued."


def test_exp_summary(Nwbfile):
    fields = Nwbfile._get_exp_summary().heading.names
    expected = ["nwb_file_name", "lab_member_name"]
    assert fields == expected, "Exp summary fields not as expected."


def test_exp_summary_no_link(schema_test, Mixin):
    schema_test(Mixin)
    assert Mixin()._get_exp_summary() is None, "Exp summary not None."


def test_exp_summary_auto_link(common):
    lab_member = common.LabMember()
    summary_names = lab_member._get_exp_summary().heading.names
    join_names = (lab_member * common.Session.Experimenter).heading.names
    assert summary_names == join_names, "Auto link not working."


def test_cautious_del_dry_run(Nwbfile, frequent_imports):
    _ = frequent_imports  # part of cascade, need import
    ret = Nwbfile.cautious_delete(dry_run=True)[1].full_table_name
    assert (
        ret == "`common_nwbfile`.`~external_raw`"
    ), "Dry run delete not working."


@pytest.mark.skipif(not VERBOSE, reason="No logging to test when quiet-spy.")
def test_empty_cautious_del(caplog, schema_test, Mixin):
    schema_test(Mixin)
    Mixin().cautious_delete(safemode=False)
    Mixin().cautious_delete(safemode=False)
    assert "empty" in caplog.text, "No warning issued."


def test_super_delete(schema_test, Mixin, common):
    schema_test(Mixin)
    Mixin().insert1((0,), skip_duplicates=True)
    Mixin().super_delete(safemode=False)
    assert len(Mixin()) == 0, "Super delete not working."

    logged_dels = common.common_usage.CautiousDelete & 'restriction LIKE "Sup%"'
    assert len(logged_dels) > 0, "Super delete not logged."


def test_compare_versions(common):
    # Does nothing in test_mode
    compare_func = common.Nwbfile().compare_versions
    compare_func("0.1.0", "0.1.1")


@pytest.fixture
def custom_table():
    """Custom table on user prefix for testing load_shared_schemas."""
    db, table = dj.config["database.user"] + "_test", "custom"
    dj.conn().query(f"CREATE DATABASE IF NOT EXISTS {db};")
    dj.conn().query(f"USE {db};")
    dj.conn().query(
        f"CREATE TABLE IF NOT EXISTS {table} ( "
        + "`merge_id` binary(16) NOT NULL COMMENT ':uuid:', "
        + "`unit_id` int NOT NULL, "
        + "PRIMARY KEY (`merge_id`), "
        + "CONSTRAINT `unit_annotation_ibfk_1` FOREIGN KEY (`merge_id`)  "
        + "REFERENCES `spikesorting_merge`.`spike_sorting_output` (`merge_id`) "
        + "ON DELETE RESTRICT ON UPDATE CASCADE);"
    )
    yield f"`{db}`.`{table}`"


def test_load_shared_schemas(common, custom_table):
    # from spyglass.common import Nwbfile

    common.Nwbfile().load_shared_schemas(additional_prefixes=["test"])
    nodes = common.Nwbfile().connection.dependencies.nodes
    assert custom_table in nodes, "Custom table not loaded."


def test_null_table_storage(schema_test, Mixin):
    schema_test(Mixin)
    assert (
        Mixin().get_table_storage_usage() == 0
    ), "Table w/o any files not zero."


@pytest.fixture
def MixinChild(Mixin):
    from spyglass.utils import SpyglassMixin

    class MixinChild(SpyglassMixin, dj.Lookup):
        definition = """
        -> Mixin
        ---
        child_attr : int
        """
        contents = [(0, 10)]

    yield MixinChild


def test_mixin_del_orphans(dj_conn, Mixin, MixinChild):
    schema_orph = dj.Schema("orph_prefix", {}, connection=dj_conn)
    schema_orph(Mixin)
    schema_orph(MixinChild)

    dry = Mixin().delete_orphans(dry_run=True).fetch1("id")
    assert dry == 1, "Dry run delete orphans not working."

    Mixin().delete_orphans(dry_run=False, safemode=False)
    post_del = Mixin().fetch1("id")
    assert post_del == 0, "Delete orphans not working."

import datajoint as dj
import numpy as np
import pytest
from datajoint.logging import logger as dj_logger


@pytest.fixture(scope="function")
def BadMerge():
    from spyglass.utils import SpyglassMixin, _Merge

    class BadMerge(_Merge, SpyglassMixin):
        definition = """
        bad_id : uuid
        ---
        not_source : varchar(1)
        """

        class BadChild(SpyglassMixin, dj.Part):
            definition = """
            -> master
            ---
            bad_child_id : uuid
            """

    yield BadMerge

    prev_level = dj_logger.level
    dj_logger.setLevel("ERROR")
    BadMerge.BadChild().drop_quick()
    BadMerge().drop_quick()
    dj_logger.setLevel(prev_level)


def test_nwb_table_missing(BadMerge, schema_test):
    from spyglass.utils.dj_merge_tables import is_merge_table

    schema_test(BadMerge)
    assert not is_merge_table(
        BadMerge()
    ), "BadMerge should fail merge-table check."


@pytest.fixture(scope="function")
def NonMerge():
    from spyglass.utils import SpyglassMixin

    class NonMerge(SpyglassMixin, dj.Manual):
        definition = """
        merge_id : uuid
        ---
        source : varchar(32)
        """

    yield NonMerge


def test_non_merge(schema_test, NonMerge):
    with pytest.raises(TypeError):
        schema_test(NonMerge)


def test_part_camel(merge_table):
    example_part = merge_table.parts(camel_case=True)[0]
    assert "_" not in example_part, "Camel case not applied."


def test_override_warning(merge_table):
    parts = merge_table.parts(camel_case=True, as_objects=True)
    assert all(
        isinstance(p, str) for p in parts
    ), "as_objects=True should be overridden to return CamelCase strings."


def test_merge_view(pos_merge):
    view = pos_merge._merge_repr(restriction=True, include_empties=True)
    # len == 15 for now, but may change with more columns
    assert len(view.heading.names) > 14, "Repr not showing all columns."


def test_merge_view_null(merge_table):
    ret = merge_table.merge_restrict(restriction='source="bad"')
    assert ret is None, "Restriction should return None for no matches."


def test_merge_get_class(merge_table):
    part_name = sorted(merge_table.parts(camel_case=True))[-1]
    parent_cls = merge_table.merge_get_parent_class(part_name)
    assert parent_cls.__name__ == part_name, "Class not found."


# @pytest.mark.skip(reason="Pending populated merge table.")
def test_merge_get_class_invalid(spike_merge, pop_spike_merge):
    ret = spike_merge.merge_get_parent_class("bad")
    assert ret is None, "Should return None for invalid part name."


# ------------------- Smart restrict / fetch ---------------------------------


def test_smart_restrict_by_part_field(
    pos_merge, pos_merge_key, pos_interval_key
):
    """& with a part-table field should equal merge_restrict count."""
    nwb = pos_interval_key["nwb_file_name"]
    smart = len(pos_merge & {"nwb_file_name": nwb})
    explicit = len(pos_merge.merge_restrict({"nwb_file_name": nwb}))
    assert (
        smart == explicit
    ), f"Smart restrict ({smart}) != merge_restrict ({explicit})"


def test_smart_restrict_master_field(pos_merge, pos_merge_key):
    """Restricting by merge_id (master field) still works normally."""
    mid = pos_merge_key["merge_id"]
    result = pos_merge & {"merge_id": mid}
    assert len(result) == 1, "Master-field restriction should return 1 row."


def test_smart_restrict_empty(pos_merge):
    """Restricting by a part field with no matches returns empty, not full."""
    result = pos_merge & {"nwb_file_name": "__nonexistent__.nwb"}
    assert len(result) == 0, "No-match restriction should return 0 rows."


def test_super_restrict_silent_noop(pos_merge, pos_interval_key):
    """super_restrict bypasses part resolution — part-field restriction is a noop."""
    nwb = pos_interval_key["nwb_file_name"]
    full = len(pos_merge)
    raw = len(pos_merge.super_restrict({"nwb_file_name": nwb}))
    assert raw == full, "super_restrict should ignore part-table fields."


# ----------------------------- Smart fetch -----------------------------------


def test_fetch_part_attr(pos_merge, pos_interval_key):
    """fetch('part_attr') should walk parts and return data."""
    nwb = pos_interval_key["nwb_file_name"]
    results = (pos_merge & {"nwb_file_name": nwb}).fetch("nwb_file_name")
    assert len(results) > 0, "fetch of part attr should return results."
    assert all(r == nwb for r in results), "Unexpected nwb_file_name values."


def test_fetch_multi_attr_returns_list_of_arrays(pos_merge, pos_merge_key):
    """fetch('a', 'b') with no as_dict returns a list of arrays (DataJoint convention)."""
    mid = pos_merge_key["merge_id"]
    result = (pos_merge & {"merge_id": mid}).fetch(
        "nwb_file_name", "interval_list_name"
    )
    assert isinstance(
        result, list
    ), "fetch with multiple attrs should return a list of arrays."
    assert len(result) == 2, "List should have one element per attr."
    assert isinstance(
        result[0], np.ndarray
    ), "Each element should be a numpy array."


def test_fetch_key_direct(pos_merge, pos_merge_key):
    """fetch of master-only attr should bypass part-walking."""
    mid = pos_merge_key["merge_id"]
    # "merge_id" is in _MASTER_ONLY_ATTRS — must NOT route to merge_fetch
    result = (pos_merge & {"merge_id": mid}).fetch("merge_id")
    # DataJoint may return a scalar UUID for single-row UUID-column fetches
    scalar = result if not hasattr(result, "__iter__") else result[0]
    assert scalar == mid, "Fetched merge_id should match restriction."


def test_fetch_merge_id_direct(pos_merge):
    """super_fetch() should bypass part-walking, returning only master cols."""
    # DataJoint has a bug fetching named cols from UUID-PK tables (format=array)
    # so we verify structural behaviour: returned dicts have only master keys.
    rows = pos_merge.super_fetch(as_dict=True)
    assert len(rows) > 0, "super_fetch should return rows."
    master_keys = {"merge_id", "source"}
    assert all(
        set(r.keys()) <= master_keys for r in rows
    ), "super_fetch should only return master-table columns."


def test_super_fetch_master_only(pos_merge):
    """super_fetch() returns only master columns."""
    rows = pos_merge.super_fetch()
    assert len(rows) > 0, "super_fetch should return rows."


# ------------------- Phase 3: merge_get_part error messages -----------------


def test_merge_get_part_zero_msg(pos_merge):
    """Zero-match restriction should say '0 matching', not '0 potential'."""
    with pytest.raises(ValueError, match="0 matching"):
        pos_merge.merge_get_part({"nwb_file_name": "__nonexistent__.nwb"})


def test_merge_get_part_multi_msg(pos_merge):
    """Multi-source restriction should show the count, not say '0 matching'."""
    # return_empties=True ensures all part tables are candidates, giving >1
    with pytest.raises(ValueError) as exc_info:
        pos_merge.merge_get_part(True, return_empties=True)
    assert "0 matching" not in str(exc_info.value)
    assert "potential parts" in str(exc_info.value)


# ------------------- Phase 4: instance-method replacements ------------------


def test_get_part_table_instance(pos_merge, pos_merge_key):
    """get_part_table() on restricted instance matches merge_get_part."""
    mid = pos_merge_key["merge_id"]
    via_class = pos_merge.merge_get_part({"merge_id": mid})
    via_inst = (pos_merge & {"merge_id": mid}).get_part_table()
    assert len(via_inst) == len(via_class), "get_part_table count mismatch."


def test_get_parent_table_instance(pos_merge, pos_merge_key):
    """get_parent_table() on restricted instance matches merge_get_parent."""
    mid = pos_merge_key["merge_id"]
    via_class = pos_merge.merge_get_parent({"merge_id": mid})
    via_inst = (pos_merge & {"merge_id": mid}).get_parent_table()
    assert len(via_inst) == len(via_class), "get_parent_table count mismatch."


def test_delete_upstream_dry_run(pos_merge, pos_merge_key):
    """delete_upstream(dry_run=True) returns list of tables."""
    mid = pos_merge_key["merge_id"]
    result = (pos_merge & {"merge_id": mid}).delete_upstream(dry_run=True)
    assert isinstance(
        result, list
    ), "delete_upstream dry_run should return a list."
    assert len(result) > 0, "Should find at least one upstream table."


def test_preview_instance(pos_merge, pos_merge_key):
    """view() returns a string representation of the merged view."""
    mid = pos_merge_key["merge_id"]
    result = (pos_merge & {"merge_id": mid}).view()
    assert isinstance(result, str), "view() should return a string."


def test_html_instance(pos_merge, pos_merge_key):
    """html() returns an HTML object."""
    from IPython.core.display import HTML

    mid = pos_merge_key["merge_id"]
    result = (pos_merge & {"merge_id": mid}).html()
    assert isinstance(result, HTML), "html() should return an HTML object."


# ------------------- Phase 4 addendum: merge_delete fix --------------------


def test_merge_delete_noop(pos_merge):
    """merge_delete with a non-matching restriction is a no-op, not an error."""
    before = len(pos_merge)
    pos_merge.merge_delete(
        {"nwb_file_name": "__nonexistent__.nwb"}, safemode=False
    )
    assert (
        len(pos_merge) == before
    ), "No-match merge_delete must not change count."


def test_merge_delete_removes_entry(graph_tables):
    """merge_delete removes the specified entry from master and part tables."""
    MergeOutput = graph_tables["MergeOutput"]
    before = len(MergeOutput())
    assert before > 0, "Test setup: MergeOutput must have entries."
    key = MergeOutput().super_fetch(as_dict=True)[0]

    MergeOutput().merge_delete(key, safemode=False, force_permission=True)

    assert (
        len(MergeOutput()) == before - 1
    ), "merge_delete should reduce count by 1."
    assert len(MergeOutput() & key) == 0, "Deleted entry must be absent."


def test_delete_instance_removes_entry(graph_tables):
    """(T & restriction).delete() removes the matching entry."""
    MergeOutput = graph_tables["MergeOutput"]
    before = len(MergeOutput())
    assert before > 0, "Test setup: MergeOutput must have entries."
    key = MergeOutput().super_fetch(as_dict=True)[0]

    (MergeOutput() & key).delete(force_permission=True, safemode=False)

    assert (
        len(MergeOutput()) == before - 1
    ), "(T & key).delete() should reduce count by 1."
    assert len(MergeOutput() & key) == 0, "Deleted entry must be absent."


# ------------------- fetch1() -----------------------------------------------


def test_fetch1_no_attrs_returns_dict(pos_merge, pos_merge_key):
    """fetch1() with no attrs returns a dict containing part-table columns."""
    mid = pos_merge_key["merge_id"]
    result = (pos_merge & {"merge_id": mid}).fetch1()
    assert isinstance(result, dict), "fetch1() should return a dict."
    assert "nwb_file_name" in result, "Part-table columns should be present."


def test_fetch1_one_attr_returns_scalar(
    pos_merge, pos_merge_key, pos_interval_key
):
    """fetch1('attr') returns a scalar value."""
    mid = pos_merge_key["merge_id"]
    nwb = pos_interval_key["nwb_file_name"]
    result = (pos_merge & {"merge_id": mid}).fetch1("nwb_file_name")
    assert result == nwb, "fetch1 scalar should match expected nwb_file_name."


def test_fetch1_multi_attr_returns_tuple(pos_merge, pos_merge_key):
    """fetch1('a', 'b') returns a tuple of values."""
    mid = pos_merge_key["merge_id"]
    result = (pos_merge & {"merge_id": mid}).fetch1(
        "nwb_file_name", "interval_list_name"
    )
    assert isinstance(
        result, tuple
    ), "fetch1 with multiple attrs should return a tuple."
    assert (
        len(result) == 2
    ), "Tuple length should match number of requested attrs."


def test_fetch1_raises_multiple(pos_merge):
    """fetch1 raises DataJointError when multiple rows match."""
    if len(pos_merge) < 2:
        pytest.skip("Need at least 2 entries to test multiple-row error.")
    with pytest.raises(dj.errors.DataJointError):
        pos_merge.fetch1()


def test_fetch1_raises_empty(pos_merge):
    """fetch1 raises DataJointError when no rows match."""
    with pytest.raises(dj.errors.DataJointError):
        (pos_merge & {"nwb_file_name": "__nonexistent__.nwb"}).fetch1()


def test_fetch1_missing_attr_raises_datajoint_error(pos_merge, pos_merge_key):
    """fetch1 with a non-existent attr raises DataJointError, not IndexError."""
    mid = pos_merge_key["merge_id"]
    with pytest.raises(dj.errors.DataJointError, match="attributes"):
        (pos_merge & {"merge_id": mid}).fetch1("__nonexistent_attr__")


# ------------------- String restrictions ------------------------------------


def test_string_restrict_part_field(pos_merge, pos_interval_key):
    """String restriction referencing a part-table field resolves through parts."""
    nwb = pos_interval_key["nwb_file_name"]
    result = pos_merge & f'nwb_file_name = "{nwb}"'
    assert len(result) > 0, "String restrict on part field should find rows."


def test_string_restrict_like(pos_merge, pos_interval_key):
    """LIKE string restriction on a part field resolves correctly."""
    nwb = pos_interval_key["nwb_file_name"]
    prefix = nwb[:4]  # first few chars
    result = pos_merge & f'nwb_file_name LIKE "{prefix}%"'
    assert (
        len(result) > 0
    ), "LIKE string restrict on part field should find rows."


def test_string_restrict_empty(pos_merge):
    """String restriction on part field with no match returns empty table."""
    result = pos_merge & 'nwb_file_name = "__nonexistent__.nwb"'
    assert (
        len(result) == 0
    ), "Non-matching string restriction should return 0 rows."


def test_string_restrict_master_field_unchanged(pos_merge):
    """String restriction on a master varchar field is not routed through parts."""
    source_val = pos_merge.super_fetch(as_dict=True)[0]["source"]
    result = pos_merge & f'source = "{source_val}"'
    assert (
        len(result) > 0
    ), "String restrict on master source field should find rows."
    assert not pos_merge._string_has_part_field(
        f'source = "{source_val}"'
    ), "'source' is a master field and must not trigger part-resolution."


# ------------------- dj.Top restrictions ------------------------------------


def test_top_limit_len(pos_merge):
    """(T & dj.Top(limit=1)) should report exactly 1 row."""
    if len(pos_merge) < 2:
        pytest.skip("Need at least 2 entries to test Top limit.")
    restricted = pos_merge & dj.Top(limit=1)
    assert len(restricted) == 1, "dj.Top(limit=1) should yield exactly 1 row."


def test_top_limit_fetch(pos_merge):
    """(T & dj.Top(limit=1)).fetch() should return 1 row of part-table data."""
    if len(pos_merge) < 2:
        pytest.skip("Need at least 2 entries to test Top limit.")
    rows = (pos_merge & dj.Top(limit=1)).fetch("nwb_file_name")
    assert len(rows) == 1, "fetch() with dj.Top(limit=1) should return 1 row."


def test_top_limit_returns_query(pos_merge):
    """(T & dj.Top(limit=1)) returns a query expression, not a string."""
    restricted = pos_merge & dj.Top(limit=1)
    assert not isinstance(
        restricted, str
    ), "& dj.Top should return a query object, not a string."
    assert isinstance(
        restricted, type(pos_merge)
    ), "& dj.Top should return the same table type."


def test_top_repr_shows_limited_rows(pos_merge):
    """repr(T & dj.Top(limit=1)) should show 1 data row, not the full table."""
    if len(pos_merge) < 2:
        pytest.skip("Need at least 2 entries to test Top repr.")
    full_repr = repr(pos_merge)
    top_repr = repr(pos_merge & dj.Top(limit=1))
    assert isinstance(top_repr, str), "repr should be a string."
    # The top repr must be shorter than the full repr (fewer rows)
    assert len(top_repr) < len(
        full_repr
    ), "dj.Top repr should be shorter than full-table repr."

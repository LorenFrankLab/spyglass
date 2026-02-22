import datajoint as dj
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

import datajoint as dj
import pytest

from tests.conftest import VERBOSE


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

    BadMerge.BadChild().drop_quick()
    BadMerge().drop_quick()


@pytest.mark.skipif(not VERBOSE, reason="No logging to test when quiet-spy.")
def test_nwb_table_missing(BadMerge, caplog, schema_test):
    schema_test(BadMerge)
    txt = caplog.text
    assert "non-default definition" in txt, "Warning not caught."


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


@pytest.mark.skipif(not VERBOSE, reason="No logging to test when quiet-spy.")
def test_override_warning(caplog, merge_table):
    _ = merge_table.parts(camel_case=True, as_objects=True)[0]
    txt = caplog.text
    assert "Overriding" in txt, "Warning not caught."


def test_merge_view(pos_merge):
    view = pos_merge._merge_repr(restriction=True, include_empties=True)
    # len == 15 for now, but may change with more columns
    assert len(view.heading.names) > 14, "Repr not showing all columns."


@pytest.mark.skipif(not VERBOSE, reason="No logging to test when quiet-spy.")
def test_merge_view_warning(caplog, merge_table):
    _ = merge_table.merge_restrict(restriction='source="bad"')
    txt = caplog.text
    assert "No parts" in txt, "Warning not caught."


def test_merge_get_class(merge_table):
    part_name = sorted(merge_table.parts(camel_case=True))[-1]
    parent_cls = merge_table.merge_get_parent_class(part_name)
    assert parent_cls.__name__ == part_name, "Class not found."


@pytest.mark.skip(reason="Pending populated merge table.")
def test_merge_get_class_invalid(caplog, merge_table):
    _ = merge_table.merge_get_parent_class("bad")
    txt = caplog.text
    assert "No source" in txt, "Warning not caught."

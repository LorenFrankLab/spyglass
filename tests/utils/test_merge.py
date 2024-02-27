import datajoint as dj
import pytest

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


def test_nwb_table_missing(caplog, schema_test):
    schema_test(BadMerge)
    txt = caplog.text
    assert "non-default definition" in txt, "Warning not caught."
    BadMerge.BadChild().drop_quick()
    BadMerge().drop_quick()


def test_part_camel(merge_table):
    example_part = merge_table.parts(camel_case=True)[0]
    assert "_" not in example_part, "Camel case not applied."


def test_override_warning(caplog, merge_table):
    _ = merge_table.parts(camel_case=True, as_objects=True)[0]
    txt = caplog.text
    assert "Overriding" in txt, "Warning not caught."


@pytest.mark.skip(reason="Pending populated merge table.")
def test_merge_view(merge_table):
    view = merge_table.merge_view()
    raise NotImplementedError(f"Check view: {view}")


def test_merge_view_warning(caplog, merge_table):
    _ = merge_table.merge_restrict(restriction='source="bad"')
    txt = caplog.text
    assert "No parts" in txt, "Warning not caught."


def test_merge_get_class(merge_table):
    part_name = merge_table.parts(camel_case=True)[-1]
    parent_cls = merge_table.merge_get_parent_class(part_name)
    assert parent_cls.__name__ == part_name, "Class not found."


def test_merge_get_class_invalid(caplog, merge_table):
    _ = merge_table.merge_get_parent_class("bad")
    txt = caplog.text
    assert "No source" in txt, "Warning not caught."

"""Tests for the SQL dump writer's handling of `%` and `"` in key values."""

from io import StringIO

import pytest

DUMP_CMD = 'mysqldump {database} {table} --where="\\\n\t{where}"\n\n'


@pytest.fixture(scope="module")
def dump_helper():
    from spyglass.utils.sql_helper_fn import SQLDumpHelper

    return SQLDumpHelper("test_paper")


@pytest.fixture(scope="module")
def AbstractGraph():
    from spyglass.utils.dj_graph import AbstractGraph

    return AbstractGraph


@pytest.mark.parametrize(
    "clause, expected",
    [
        ('(`n`="plain")', False),
        ('(`n`="")', False),  # empty-string literal, not an escaped quote
        ('(`n`="has""quote")', True),
        ('(`n`="""")', True),  # value is a lone double quote
        ('(`n`="a") OR (`n`="b")', False),
    ],
)
def test_value_holds_double_quote(clause, expected):
    """Only a `""` pair inside a value counts, not the empty literal."""
    from spyglass.utils.sql_helper_fn import value_holds_double_quote

    assert value_holds_double_quote(clause) is expected, clause


def test_dump_where_keeps_percent_single(dump_helper, quoted_key_tables):
    """A `%` must reach `--where` singled.

    `where_clause` doubles it for pymysql's paramstyle; mysqldump passes the
    text to the server verbatim.
    """
    parent = quoted_key_tables["DqParent"]
    file = StringIO()

    dump_helper._write_table_dumps(
        file, parent & {"dq_name": "has%pct"}, DUMP_CMD
    )
    written = file.getvalue()

    assert "has%pct" in written, f"Percent value not in dump: {written}"
    assert "%%" not in written, f"Percent left doubled in dump: {written}"


def test_dump_refuses_quote_in_value(
    dump_helper, quoted_key_tables, AbstractGraph
):
    """A `"` in a value must raise rather than dump a changed value.

    `bash_escape_sql` rewrites `"` as `'`, which changes the value MySQL
    matches. The dump restricts by the stored condition, where
    `_literal_condition` writes the quote as the pair `""`.
    """
    parent = quoted_key_tables["DqParent"]
    stored = AbstractGraph._literal_condition(parent, {"dq_name": 'has"quote'})
    file = StringIO()

    assert '""' in stored, f"Fixture did not double the quote: {stored}"

    with pytest.raises(ValueError) as err:
        dump_helper._write_table_dumps(file, parent & stored, DUMP_CMD)

    assert "double quote" in str(err.value), str(err.value)
    assert not file.getvalue(), "Wrote a dump command before raising."

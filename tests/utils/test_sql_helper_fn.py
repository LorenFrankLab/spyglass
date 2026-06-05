"""Tests for sql_helper_fn.py utility functions."""

from unittest.mock import patch

import pytest


@pytest.fixture(scope="session")
def bash_escape_sql():
    from spyglass.utils.sql_helper_fn import bash_escape_sql

    return bash_escape_sql


@pytest.fixture(scope="session")
def remove_redundant():
    from spyglass.utils.sql_helper_fn import remove_redundant

    return remove_redundant


@pytest.fixture(scope="session")
def SQLDumpHelper():
    from spyglass.utils.sql_helper_fn import SQLDumpHelper

    return SQLDumpHelper


# ------------------------------------------------------------------ #
# remove_redundant
# ------------------------------------------------------------------ #


def test_remove_redundant_basic_string(remove_redundant):
    """Simple values pass through without error."""
    result = remove_redundant("(a=b)")
    assert "a=b" in result


def test_remove_redundant_double_wrapped(remove_redundant):
    """Double-wrapped parentheses are reduced."""
    result = remove_redundant("((a=b))")
    assert "a=b" in result
    assert result.count("((") == 0


def test_remove_redundant_no_parens(remove_redundant):
    """Strings without parentheses pass through."""
    result = remove_redundant("a=b")
    assert "a=b" in result


def test_remove_redundant_and_spacing(remove_redundant):
    """AND keyword is spaced out for readability."""
    result = remove_redundant("(a=b)AND(c=d)")
    assert "a=b" in result
    assert "c=d" in result
    assert "AND" in result.upper()


def test_remove_redundant_or_spacing(remove_redundant):
    """OR keyword is spaced out for readability."""
    result = remove_redundant("(a=b)OR(c=d)")
    assert "a=b" in result
    assert "c=d" in result
    assert "OR" in result.upper()


def test_remove_redundant_complex_nested(remove_redundant):
    """Complex nested expressions preserve all terms."""
    result = remove_redundant("((a=b)OR((c=d)AND((e=f))))")
    assert "a=b" in result
    assert "c=d" in result
    assert "e=f" in result


def test_remove_redundant_returns_string(remove_redundant):
    """Return type is always a string."""
    assert isinstance(remove_redundant("(x=1)"), str)


# ------------------------------------------------------------------ #
# bash_escape_sql
# ------------------------------------------------------------------ #


def test_bash_escape_strips_where(bash_escape_sql):
    """WHERE prefix is stripped from the SQL string."""
    result = bash_escape_sql("WHERE a=b", add_newline=False)
    assert not result.lstrip().startswith("WHERE")
    assert "a=b" in result


def test_bash_escape_strips_where_with_spaces(bash_escape_sql):
    """WHERE followed by spaces is stripped cleanly."""
    result = bash_escape_sql("WHERE  a=b", add_newline=False)
    assert "WHERE" not in result


def test_bash_escape_no_where_passthrough(bash_escape_sql):
    """Strings without WHERE are processed normally."""
    assert "a=b" in bash_escape_sql("a=b", add_newline=False)


def test_bash_escape_balanced_open_parens(bash_escape_sql):
    """Open paren is added when close parens outnumber opens."""
    result = bash_escape_sql("a=b)", add_newline=False)
    assert result.count("(") == result.count(")")


def test_bash_escape_balanced_close_parens(bash_escape_sql):
    """Close paren is added when open parens outnumber closes."""
    result = bash_escape_sql("(a=b", add_newline=False)
    assert result.count("(") == result.count(")")


def test_bash_escape_already_balanced(bash_escape_sql):
    """Already-balanced parens remain balanced."""
    result = bash_escape_sql("(a=b)", add_newline=False)
    assert result.count("(") == result.count(")")


def test_bash_escape_double_quotes_to_single(bash_escape_sql):
    """Double quotes are replaced with single quotes."""
    result = bash_escape_sql('a="value"', add_newline=False)
    assert '"' not in result
    assert "'" in result


def test_bash_escape_backticks_removed(bash_escape_sql):
    """Backticks are removed from the output."""
    result = bash_escape_sql("`table`.`col`=1", add_newline=False)
    assert "`" not in result


def test_bash_escape_newline_and(bash_escape_sql):
    """AND gets a newline when add_newline=True."""
    result = bash_escape_sql("a=b AND c=d", add_newline=True)
    assert "AND" in result
    assert "\n" in result


def test_bash_escape_newline_or(bash_escape_sql):
    """OR gets a newline when add_newline=True."""
    result = bash_escape_sql("a=b OR c=d", add_newline=True)
    assert "OR" in result
    assert "\n" in result


def test_bash_escape_no_newline_and(bash_escape_sql):
    """add_newline=False does not add escape-newlines for AND."""
    result = bash_escape_sql("a=b AND c=d", add_newline=False)
    assert "\\\n" not in result


def test_bash_escape_hash_with_newline(bash_escape_sql):
    """Hash is escaped when add_newline=True."""
    result = bash_escape_sql("a#b=1", add_newline=True)
    assert "\\#" in result


def test_bash_escape_hash_without_newline(bash_escape_sql):
    """Hash is not escaped when add_newline=False."""
    result = bash_escape_sql("a#b=1", add_newline=False)
    assert "\\#" not in result


def test_bash_escape_whitespace_stripped(bash_escape_sql):
    """Output has no leading or trailing whitespace."""
    result = bash_escape_sql("  a=b  ", add_newline=False)
    assert result == result.strip()


def test_bash_escape_percent_no_newline(bash_escape_sql):
    """add_newline=False replaces %%%% with %%."""
    result = bash_escape_sql("a=b%%%%c", add_newline=False)
    assert "%%%%" not in result
    assert "%%" in result


def test_bash_escape_returns_string(bash_escape_sql):
    """Return type is always a string."""
    assert isinstance(bash_escape_sql("a=b", add_newline=False), str)


def test_bash_escape_paren_joiner(bash_escape_sql):
    """)AND( joined form is expanded."""
    result = bash_escape_sql("(a=b)AND(c=d)", add_newline=False)
    assert "AND" in result


# ------------------------------------------------------------------ #
# SQLDumpHelper
# ------------------------------------------------------------------ #


def test_sqldump_cmd_prefix_no_docker(SQLDumpHelper):
    """Without docker_id, returns simple mysqldump prefix."""
    helper = SQLDumpHelper(paper_id="test_paper")
    assert helper._cmd_prefix() == "mysqldump --hex-blob "


def test_sqldump_cmd_prefix_none_docker(SQLDumpHelper):
    """Explicitly passing None docker_id gives base prefix."""
    helper = SQLDumpHelper(paper_id="test_paper")
    result = helper._cmd_prefix(docker_id=None)
    assert "mysqldump" in result
    assert "docker" not in result


def test_sqldump_cmd_prefix_with_docker(SQLDumpHelper):
    """With docker_id, prefix includes docker exec command."""
    helper = SQLDumpHelper(paper_id="test_paper", docker_id="abc123")
    with patch.object(
        helper,
        "_get_credentials",
        return_value={
            "user": "testuser",
            "password": "testpass",
            "host": "localhost",
        },
    ):
        result = helper._cmd_prefix("abc123")
    assert "docker exec -i abc123" in result
    assert "mysqldump" in result
    assert "testuser" in result


def test_sqldump_init_stores_attributes(SQLDumpHelper):
    """Constructor stores paper_id, docker_id, spyglass_version."""
    helper = SQLDumpHelper(
        paper_id="p1", docker_id="d1", spyglass_version="0.5.0"
    )
    assert helper.paper_id == "p1"
    assert helper.docker_id == "d1"
    assert helper.spyglass_version == "0.5.0"


def test_sqldump_init_defaults(SQLDumpHelper):
    """Constructor defaults docker_id and version to None."""
    helper = SQLDumpHelper(paper_id="p1")
    assert helper.docker_id is None
    assert helper.spyglass_version is None

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
    """Simple values pass through, wrapped in a single paren pair."""
    assert remove_redundant("(a=b)") == "(a=b)"


def test_remove_redundant_double_wrapped(remove_redundant):
    """Double-wrapped parentheses are reduced to one pair."""
    assert remove_redundant("((a=b))") == "(a=b)"


def test_remove_redundant_no_parens(remove_redundant):
    """Strings without parentheses gain a single wrapping pair."""
    assert remove_redundant("a=b") == "(a=b)"


def test_remove_redundant_and_spacing(remove_redundant):
    """AND keyword is spaced out for readability."""
    assert remove_redundant("(a=b)AND(c=d)") == "((a=b) AND (c=d))"


def test_remove_redundant_or_spacing(remove_redundant):
    """OR keyword is spaced out for readability."""
    assert remove_redundant("(a=b)OR(c=d)") == "((a=b) OR (c=d))"


def test_remove_redundant_complex_nested(remove_redundant):
    """Complex nested expressions collapse redundant parens."""
    assert (
        remove_redundant("((a=b)OR((c=d)AND((e=f))))")
        == "((a=b) OR ((c=d) AND (e=f)))"
    )


# ------------------------------------------------------------------ #
# bash_escape_sql
# ------------------------------------------------------------------ #


def test_bash_escape_strips_where(bash_escape_sql):
    """WHERE prefix is stripped from the SQL string."""
    assert bash_escape_sql("WHERE a=b", add_newline=False) == "(a=b)"


def test_bash_escape_strips_where_with_spaces(bash_escape_sql):
    """WHERE followed by spaces is stripped cleanly."""
    assert bash_escape_sql("WHERE  a=b", add_newline=False) == "(a=b)"


def test_bash_escape_no_where_passthrough(bash_escape_sql):
    """Strings without WHERE are wrapped and returned."""
    assert bash_escape_sql("a=b", add_newline=False) == "(a=b)"


def test_bash_escape_balanced_open_parens(bash_escape_sql):
    """Open paren is added when close parens outnumber opens."""
    assert bash_escape_sql("a=b)", add_newline=False) == "(a=b)"


def test_bash_escape_balanced_close_parens(bash_escape_sql):
    """Close paren is added when open parens outnumber closes."""
    assert bash_escape_sql("(a=b", add_newline=False) == "(a=b)"


def test_bash_escape_already_balanced(bash_escape_sql):
    """Already-balanced parens remain balanced."""
    assert bash_escape_sql("(a=b)", add_newline=False) == "(a=b)"


def test_bash_escape_double_quotes_to_single(bash_escape_sql):
    """Double quotes are replaced with single quotes."""
    assert bash_escape_sql('a="value"', add_newline=False) == "(a='value')"


def test_bash_escape_backticks_removed(bash_escape_sql):
    """Backticks are removed from the output."""
    assert (
        bash_escape_sql("`table`.`col`=1", add_newline=False) == "(table.col=1)"
    )


def test_bash_escape_newline_and(bash_escape_sql):
    """AND gets an escaped newline and tab when add_newline=True."""
    assert (
        bash_escape_sql("a=b AND c=d", add_newline=True)
        == "(a=b \\\n\tAND c=d)"
    )


def test_bash_escape_newline_or(bash_escape_sql):
    """OR gets an escaped newline and tab when add_newline=True."""
    assert (
        bash_escape_sql("a=b OR c=d", add_newline=True) == "(a=b \\\n\tOR  c=d)"
    )


def test_bash_escape_no_newline_and(bash_escape_sql):
    """add_newline=False leaves AND inline without escape-newlines."""
    assert bash_escape_sql("a=b AND c=d", add_newline=False) == "(a=b AND c=d)"


def test_bash_escape_hash_with_newline(bash_escape_sql):
    """Hash is escaped when add_newline=True."""
    assert bash_escape_sql("a#b=1", add_newline=True) == "(a\\#b=1)"


def test_bash_escape_hash_without_newline(bash_escape_sql):
    """Hash is not escaped when add_newline=False."""
    assert bash_escape_sql("a#b=1", add_newline=False) == "(a#b=1)"


def test_bash_escape_whitespace_stripped(bash_escape_sql):
    """Leading and trailing whitespace is stripped before wrapping."""
    assert bash_escape_sql("  a=b  ", add_newline=False) == "(a=b)"


def test_bash_escape_percent_no_newline(bash_escape_sql):
    """add_newline=False collapses %%%% to %%."""
    assert bash_escape_sql("a=b%%%%c", add_newline=False) == "(a=b%%c)"


def test_bash_escape_paren_joiner(bash_escape_sql):
    """)AND( joined form is expanded and spaced."""
    assert (
        bash_escape_sql("(a=b)AND(c=d)", add_newline=False)
        == "((a=b) AND (c=d))"
    )


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
    assert helper._cmd_prefix(docker_id=None) == "mysqldump --hex-blob "


def test_sqldump_cmd_prefix_with_docker(SQLDumpHelper):
    """With docker_id, prefix wraps mysqldump in docker exec with creds."""
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
    assert result == (
        "docker exec -i abc123 \\\n\tmysqldump --hex-blob "
        "-u testuser --password=testpass \\\n\t"
    )


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

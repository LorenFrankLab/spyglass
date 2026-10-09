"""Tests for the schema prefix accepted by ``SpyglassAnalysis``.

Issue #1510: ``SpyglassAnalysis.__init__`` read the expected prefix from
``custom.database.prefix`` only. With that config unset the comparison ran
against ``None``, so a correctly named ``{username}_nwbfile`` schema was
rejected. ``spyglass.common.custom_nwbfile`` already falls back to
``database.user``; these tests pin that same fallback on the mixin, while
guarding that the fallback does not turn the check into a no-op.

No database is required: each table class is given a stub connection that
reports the table as undeclared, so ``__init__`` runs its validation and
returns before any real query is issued.
"""

from types import SimpleNamespace

import datajoint as dj
import pytest

from spyglass.utils.dj_mixin import SpyglassAnalysis


class UndeclaredConnection:
    """Stand-in for a DataJoint connection reporting no declared tables."""

    def query(self, *args, **kwargs):
        """Return an empty result, as ``SHOW TABLES ... LIKE`` would.

        Returns
        -------
        SimpleNamespace
            Object exposing the ``rowcount`` attribute read by
            ``datajoint.table.Table.is_declared``.
        """
        return SimpleNamespace(rowcount=0)


@pytest.fixture
def analysis_table(monkeypatch):
    """Factory for undeclared ``SpyglassAnalysis`` tables in a given schema.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to neutralize registry inserts, which would need a database.

    Returns
    -------
    callable
        Takes a schema name and returns a class whose instantiation runs the
        ``SpyglassAnalysis`` validation for that schema.
    """
    monkeypatch.setattr(
        SpyglassAnalysis, "_register_table", lambda self: None, raising=True
    )

    def _factory(database):
        class AnalysisNwbfile(SpyglassAnalysis, dj.Manual):
            definition = """
            analysis_file_name: varchar(64)
            """

        AnalysisNwbfile.database = database
        AnalysisNwbfile.connection = UndeclaredConnection()
        return AnalysisNwbfile

    return _factory


@pytest.fixture
def dj_prefix_config(monkeypatch):
    """Factory setting ``database.user`` and ``custom.database.prefix``.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Restores the prior ``dj.config`` values after each test.

    Returns
    -------
    callable
        Takes ``user`` and optional ``prefix`` and applies them to
        ``dj.config``. A ``None`` prefix leaves ``custom.database.prefix``
        unset.
    """

    def _config(user, prefix=None):
        custom = dict(dj.config.get("custom", dict()))
        custom.pop("database.prefix", None)
        if prefix is not None:
            custom["database.prefix"] = prefix
        monkeypatch.setitem(dj.config, "custom", custom)
        monkeypatch.setitem(dj.config, "database.user", user)

    return _config


def test_unset_prefix_accepts_username_schema(analysis_table, dj_prefix_config):
    """Schema named for the database user passes with no configured prefix."""
    dj_prefix_config(user="alice")
    analysis_table("alice_nwbfile")()  # no raise


def test_unset_prefix_rejects_foreign_prefix(analysis_table, dj_prefix_config):
    """Fallback to the username must not accept an unrelated prefix."""
    dj_prefix_config(user="alice")
    with pytest.raises(ValueError, match="does not match"):
        analysis_table("bob_nwbfile")()


def test_unset_prefix_error_names_source(analysis_table, dj_prefix_config):
    """Rejection message reports the prefix used and where it came from."""
    dj_prefix_config(user="alice")
    with pytest.raises(ValueError, match=r"alice \(from database.user\)"):
        analysis_table("bob_nwbfile")()


def test_no_prefix_or_user_rejects_all(analysis_table, dj_prefix_config):
    """With neither config set, no user schema is accepted."""
    dj_prefix_config(user="")
    with pytest.raises(ValueError, match="does not match"):
        analysis_table("alice_nwbfile")()


def test_configured_prefix_accepted(analysis_table, dj_prefix_config):
    """A configured prefix is honored, as before."""
    dj_prefix_config(user="alice", prefix="team")
    analysis_table("team_nwbfile")()  # no raise


def test_configured_prefix_takes_precedence(analysis_table, dj_prefix_config):
    """The configured prefix wins over the database user, as before."""
    dj_prefix_config(user="alice", prefix="team")
    with pytest.raises(
        ValueError, match=r"team \(from custom.database.prefix\)"
    ):
        analysis_table("alice_nwbfile")()


def test_common_schema_exempt(analysis_table, dj_prefix_config):
    """The common schema skips the prefix check entirely."""
    dj_prefix_config(user="alice")
    analysis_table("common_nwbfile")()  # no raise


def test_suffix_still_enforced(analysis_table, dj_prefix_config):
    """The fallback does not relax the ``_nwbfile`` suffix requirement."""
    dj_prefix_config(user="alice")
    with pytest.raises(ValueError, match="requires .*_nwbfile schema"):
        analysis_table("alice_other")()

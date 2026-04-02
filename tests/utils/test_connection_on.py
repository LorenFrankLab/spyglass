"""Live-connection integration tests: SpyglassSchema with a real DB.

Requires a running MySQL instance (Docker).  Run with::

    pytest tests/utils/test_connection_on.py --no-teardown

These tests verify that SpyglassSchema behaves identically to dj.Schema when
a real DB connection is available — i.e. the offline shim is transparent.
"""

import warnings

import datajoint as dj
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_LIVE_DEFN = """
live_id : int
---
live_val : varchar(16)
"""


@pytest.fixture(scope="class")
def live_schema(dj_conn, teardown):
    """SpyglassSchema activated against the real test database."""
    from spyglass.utils.dj_schema import SpyglassSchema

    schema = SpyglassSchema("test_no_conn_live", connection=dj_conn)
    yield schema
    if teardown:
        schema.drop(force=True)


@pytest.fixture(scope="class")
def live_table(live_schema):
    """Minimal SpyglassMixin table registered in the live test schema."""
    from spyglass.utils import SpyglassMixin

    @live_schema
    class NoConnLiveTable(SpyglassMixin, dj.Manual):
        definition = _LIVE_DEFN

    NoConnLiveTable.insert(
        [{"live_id": 1, "live_val": "a"}], skip_duplicates=True
    )
    yield NoConnLiveTable


# ---------------------------------------------------------------------------
# TestSpyglassSchemaLive
# ---------------------------------------------------------------------------


class TestSpyglassSchemaLive:
    """Online integration tests: ``SpyglassSchema`` with a real DB connection."""

    def test_connected_schema_flag_false(self, live_schema):
        """A successfully connected schema must NOT set _no_connection."""
        assert live_schema._no_connection is False

    def test_online_table_no_flag(self, live_table):
        """Table decorated with a connected schema must have _no_connection=False."""
        assert live_table._no_connection is False

    def test_online_table_is_declared(self, live_table):
        assert live_table().is_declared is True

    def test_online_len_positive(self, live_table):
        assert len(live_table()) > 0

    def test_online_restrict_returns_new_object(self, live_table):
        """Connected table: & must return a new QueryExpression, not self."""
        instance = live_table()
        result = instance & {"live_id": 1}
        assert result is not instance

    def test_online_restrict_filters_correctly(self, live_table):
        """Connected table: restriction actually narrows the result set."""
        full = len(live_table())
        restricted = len(live_table() & "live_id > 999")
        assert restricted < full

    def test_online_fetch_no_warn(self, live_table):
        """Connected table: fetch() must NOT emit a [Spyglass] UserWarning."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            live_table().fetch()  # must not raise

    def test_online_repr_no_offline_notice(self, live_table):
        """Connected table repr must not contain the offline notice."""
        assert "[no database connection]" not in repr(live_table())

    def test_online_fetch_returns_data(self, live_table):
        """fetch() on a populated connected table returns non-empty result."""
        result = live_table().fetch()
        assert len(result) > 0

"""No-connection unit tests: SpyglassSchema, NoConnectionMixin, SpyglassMixin.

All tests are pure unit tests — no database required.  Run with::

    pytest tests/utils/offline/test_connection_off.py

Connection failures are simulated by patching ``datajoint.schemas.conn`` so
that it raises ``LostConnectionError`` instead of prompting for a password.
The local ``conftest.py`` shadows the root ``mini_insert`` autouse fixture so
that Docker/MySQL is not required.
"""

import warnings
from unittest.mock import patch

import datajoint as dj
import pytest
from datajoint.errors import DataJointError, LostConnectionError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CONN_PATH = "datajoint.schemas.conn"


def _make_offline_schema(schema_name="test_offline"):
    """Return a SpyglassSchema whose activation was blocked by LostConnectionError.

    Resets the class-level ``_warned`` flag so that each call can independently
    trigger the UserWarning (the flag is an optimisation for production use; in
    tests we want deterministic warn behaviour per call).
    """
    from spyglass.utils.dj_schema import SpyglassSchema

    SpyglassSchema._warned = False  # reset so this call always warns
    with patch(_CONN_PATH, side_effect=LostConnectionError("no host")):
        with pytest.warns(UserWarning, match=r"\[Spyglass\]"):
            schema = SpyglassSchema(schema_name)
    return schema


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def offline_schema():
    """SpyglassSchema whose activation failed due to a lost connection."""
    return _make_offline_schema("test_offline")


@pytest.fixture(scope="module")
def offline_table(offline_schema):
    """Minimal dj.Manual table decorated by an offline schema."""

    @offline_schema
    class OfflineManual(dj.Manual):
        definition = """
        id : int
        ---
        value : varchar(16)
        """

    return OfflineManual


@pytest.fixture(scope="module")
def offline_table_with_part(offline_schema):
    """Master table with a Part inner-class, decorated by an offline schema."""

    @offline_schema
    class OfflineMaster(dj.Manual):
        definition = """
        master_id : int
        ---
        master_attr : int
        """

        class Part(dj.Part):
            definition = """
            -> master
            ---
            part_attr : int
            """

    return OfflineMaster


# ---------------------------------------------------------------------------
# TestNoConnectionSchema — schema-level behaviour
# ---------------------------------------------------------------------------


class TestNoConnectionSchema:
    @pytest.fixture(autouse=True)
    def reset_warned(self):
        """Reset the class-level _warned flag so each test can assert on warns."""
        from spyglass.utils.dj_schema import SpyglassSchema

        SpyglassSchema._warned = False
        yield
        SpyglassSchema._warned = False

    def test_activation_does_not_raise(self):
        """SpyglassSchema.__init__ must not raise even when conn() fails."""
        from spyglass.utils.dj_schema import SpyglassSchema

        with patch(_CONN_PATH, side_effect=LostConnectionError("no host")):
            with pytest.warns(UserWarning):
                _ = SpyglassSchema("test_no_raise")
        # If we reach here the test passes.

    def test_no_connection_flag_set(self, offline_schema):
        assert offline_schema._no_connection is True

    def test_database_name_preserved(self, offline_schema):
        """Schema must remember its name even without a connection."""
        assert offline_schema.database == "test_offline"

    def test_connection_is_none(self, offline_schema):
        """No live Connection object should be stored."""
        assert offline_schema.connection is None

    def test_warning_contains_schema_name(self):
        """UserWarning text must identify the schema that failed."""
        from spyglass.utils.dj_schema import SpyglassSchema

        with patch(_CONN_PATH, side_effect=LostConnectionError("no host")):
            with pytest.warns(UserWarning, match="test_warn_name"):
                SpyglassSchema("test_warn_name")

    def test_warning_contains_reconnect_hint(self):
        """UserWarning must tell the user how to reconnect."""
        from spyglass.utils.dj_schema import SpyglassSchema

        with patch(_CONN_PATH, side_effect=LostConnectionError("no host")):
            with pytest.warns(UserWarning, match="dj.conn"):
                SpyglassSchema("test_hint")

    def test_warning_fires_only_once(self):
        """Second and subsequent failures must NOT re-emit the UserWarning."""
        from spyglass.utils.dj_schema import SpyglassSchema

        with patch(_CONN_PATH, side_effect=LostConnectionError("no host")):
            with pytest.warns(UserWarning):
                SpyglassSchema("first_schema")
            # Second activation — _warned is now True; no warning expected.
            with warnings.catch_warnings():
                warnings.simplefilter("error")  # any warning → AssertionError
                SpyglassSchema("second_schema")  # must not raise

    def test_keyboard_interrupt_caught(self):
        """Cancelled password prompt (KeyboardInterrupt) is handled gracefully."""
        from spyglass.utils.dj_schema import SpyglassSchema

        with patch(_CONN_PATH, side_effect=KeyboardInterrupt):
            with pytest.warns(UserWarning, match=r"\[Spyglass\]"):
                schema = SpyglassSchema("test_kb_interrupt")
        assert schema._no_connection is True

    def test_datajoint_error_caught(self):
        """Generic DataJointError during activation is handled gracefully."""
        from spyglass.utils.dj_schema import SpyglassSchema

        with patch(_CONN_PATH, side_effect=DataJointError("bad config")):
            with pytest.warns(UserWarning, match=r"\[Spyglass\]"):
                schema = SpyglassSchema("test_dj_error")
        assert schema._no_connection is True


# ---------------------------------------------------------------------------
# TestNoConnectionDecorator — table decoration without a DB
# ---------------------------------------------------------------------------


class TestNoConnectionDecorator:
    def test_decorated_class_gets_flag(self, offline_table):
        """Table classes decorated by an offline schema get _no_connection=True."""
        assert offline_table._no_connection is True

    def test_decorated_class_keeps_database_name(self, offline_table):
        assert offline_table.database == "test_offline"

    def test_part_table_gets_flag(self, offline_table_with_part):
        """Part inner-classes must also receive _no_connection=True."""
        assert offline_table_with_part.Part._no_connection is True

    def test_part_table_master_set(self, offline_table_with_part):
        """Part._master must point to the master class."""
        assert offline_table_with_part.Part._master is offline_table_with_part

    def test_decoration_returns_class(self, offline_schema):
        """__call__ must return the class unchanged (decorator contract)."""

        @offline_schema
        class ReturnCheck(dj.Manual):
            definition = "id: int"

        assert ReturnCheck._no_connection is True


# ---------------------------------------------------------------------------
# TestNoConnectionMixin — method-level behaviour (Phase 2)
# ---------------------------------------------------------------------------

_DEFINITION = """
id : int
name : varchar(32)
---
value : float
label : varchar(64)
"""


@pytest.fixture(scope="module")
def mixin_table(offline_schema):
    """Table that mixes in NoConnectionMixin and is decorated offline."""
    from spyglass.utils.mixins.no_connection import NoConnectionMixin

    @offline_schema
    class MixinTable(NoConnectionMixin, dj.Manual):
        definition = _DEFINITION

    return MixinTable


@pytest.fixture(scope="module")
def mixin_instance(mixin_table):
    return mixin_table()


class TestNoConnectionMixin:
    # --- is_declared ----------------------------------------------------------

    def test_is_declared_false(self, mixin_instance):
        assert mixin_instance.is_declared is False

    # --- counting / existence -------------------------------------------------

    def test_len_zero(self, mixin_instance):
        assert len(mixin_instance) == 0

    def test_bool_false(self, mixin_instance):
        assert bool(mixin_instance) is False

    # --- restrict (& - ^ operators) ------------------------------------------

    def test_restrict_returns_self(self, mixin_instance):
        """Table() & anything must return the same offline table, not raise."""
        result = mixin_instance & {"id": 1}
        assert result is mixin_instance

    def test_restrict_dict_returns_self(self, mixin_instance):
        result = mixin_instance & {"id": 1, "name": "x"}
        assert result is mixin_instance

    def test_restrict_string_returns_self(self, mixin_instance):
        result = mixin_instance & "id > 0"
        assert result is mixin_instance

    def test_restrict_subtraction_returns_self(self, mixin_instance):
        """Table() - restriction must also return the same offline table."""
        result = mixin_instance - {"id": 1}
        assert result is mixin_instance

    def test_restrict_chained_returns_self(self, mixin_instance):
        result = (mixin_instance & {"id": 1}) & {"name": "x"}
        assert result is mixin_instance

    # --- fetch ----------------------------------------------------------------

    def test_fetch_warns(self, mixin_instance):
        """fetch() must emit a UserWarning when offline."""
        with pytest.warns(
            UserWarning, match=r"\[Spyglass\].*No database connection"
        ):
            mixin_instance.fetch()

    def test_fetch_with_attrs_warns(self, mixin_instance):
        with pytest.warns(UserWarning, match=r"\[Spyglass\]"):
            mixin_instance.fetch("id")

    def test_fetch_returns_empty_array(self, mixin_instance):
        """Default fetch() returns an empty numpy array."""
        with pytest.warns(UserWarning):
            result = mixin_instance.fetch()
        assert hasattr(result, "__len__")
        assert len(result) == 0

    def test_fetch_with_attrs_returns_empty_list(self, mixin_instance):
        """fetch('attr') returns an empty list, not an array."""
        with pytest.warns(UserWarning):
            result = mixin_instance.fetch("id")
        assert result == []

    def test_fetch_as_dict_returns_empty_list(self, mixin_instance):
        with pytest.warns(UserWarning):
            result = mixin_instance.fetch(as_dict=True)
        assert result == []

    def test_fetch_frame_returns_empty_dataframe(self, mixin_instance):
        import pandas as pd

        with pytest.warns(UserWarning):
            result = mixin_instance.fetch(format="frame")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_fetch_key_returns_empty_list(self, mixin_instance):
        with pytest.warns(UserWarning):
            result = mixin_instance.fetch("KEY")
        assert result == []

    # --- fetch1 ---------------------------------------------------------------

    def test_fetch1_raises(self, mixin_instance):
        with pytest.raises(DataJointError):
            _ = mixin_instance.fetch1

    # --- preview / repr -------------------------------------------------------

    def test_repr_contains_offline_notice(self, mixin_instance):
        assert "[no database connection]" in repr(mixin_instance)

    def test_repr_contains_class_name(self, mixin_instance):
        assert "MixinTable" in repr(mixin_instance)

    def test_repr_contains_primary_keys(self, mixin_instance):
        text = repr(mixin_instance)
        assert "*id" in text
        assert "*name" in text

    def test_repr_contains_secondary_attrs(self, mixin_instance):
        text = repr(mixin_instance)
        assert "value" in text
        assert "label" in text

    def test_preview_matches_repr(self, mixin_instance):
        assert mixin_instance.preview() == repr(mixin_instance)

    def test_repr_contains_types(self, mixin_instance):
        """Offline repr must show column types, not just names."""
        text = repr(mixin_instance)
        assert ": int" in text
        assert ": varchar(32)" in text
        assert ": float" in text
        assert ": varchar(64)" in text

    # --- _repr_html_ ----------------------------------------------------------

    def test_repr_html_is_string(self, mixin_instance):
        assert isinstance(mixin_instance._repr_html_(), str)

    def test_repr_html_contains_offline_notice(self, mixin_instance):
        assert "no database connection" in mixin_instance._repr_html_()

    def test_repr_html_contains_class_name(self, mixin_instance):
        assert "MixinTable" in mixin_instance._repr_html_()

    def test_repr_html_contains_columns(self, mixin_instance):
        html = mixin_instance._repr_html_()
        assert "*id" in html
        assert "value" in html

    def test_repr_html_contains_no_data_row(self, mixin_instance):
        assert "no data" in mixin_instance._repr_html_()

    def test_repr_html_contains_types(self, mixin_instance):
        html = mixin_instance._repr_html_()
        assert "int" in html
        assert "float" in html

    # --- definition parser ----------------------------------------------------

    def test_parse_definition_primary_keys(self):
        from spyglass.utils.mixins.no_connection import _parse_definition

        pk, sec = _parse_definition(_DEFINITION)
        assert [n for n, _ in pk] == ["id", "name"]

    def test_parse_definition_primary_key_types(self):
        from spyglass.utils.mixins.no_connection import _parse_definition

        pk, sec = _parse_definition(_DEFINITION)
        assert dict(pk)["id"] == "int"
        assert dict(pk)["name"] == "varchar(32)"

    def test_parse_definition_secondary(self):
        from spyglass.utils.mixins.no_connection import _parse_definition

        pk, sec = _parse_definition(_DEFINITION)
        assert [n for n, _ in sec] == ["value", "label"]

    def test_parse_definition_secondary_types(self):
        from spyglass.utils.mixins.no_connection import _parse_definition

        pk, sec = _parse_definition(_DEFINITION)
        assert dict(sec)["value"] == "float"
        assert dict(sec)["label"] == "varchar(64)"

    def test_parse_definition_skips_fk(self):
        from spyglass.utils.mixins.no_connection import _parse_definition

        defn = """
        id : int
        -> SomeOtherTable
        ---
        val : float
        """
        pk, sec = _parse_definition(defn)
        assert [n for n, _ in pk] == ["id"]
        assert [n for n, _ in sec] == ["val"]

    def test_parse_definition_skips_blank_and_comments(self):
        from spyglass.utils.mixins.no_connection import _parse_definition

        defn = """
        # a comment
        id : int   # inline comment
        ---
        # another comment

        val : float
        """
        pk, sec = _parse_definition(defn)
        assert [n for n, _ in pk] == ["id"]
        assert [n for n, _ in sec] == ["val"]


# ---------------------------------------------------------------------------
# TestSpyglassMixinOffline — Phase 3: full SpyglassMixin + NoConnectionMixin
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sg_table(offline_schema):
    """SpyglassMixin table decorated by an offline schema."""
    from spyglass.utils import SpyglassMixin

    @offline_schema
    class SgOffline(SpyglassMixin, dj.Manual):
        definition = """
        session_id : varchar(32)
        ---
        subject : varchar(64)
        weight : float
        """

    return SgOffline


@pytest.fixture(scope="module")
def sg_instance(sg_table):
    return sg_table()


@pytest.fixture(scope="module")
def sg_part_table(offline_schema):
    """SpyglassMixin master+part decorated by an offline schema."""
    from spyglass.utils import SpyglassMixin

    @offline_schema
    class SgMaster(SpyglassMixin, dj.Manual):
        definition = """
        master_id : int
        ---
        master_attr : int
        """

        class Part(dj.Part):
            definition = """
            -> master
            ---
            part_attr : float
            """

    return SgMaster


class TestSpyglassMixinOffline:
    # --- instantiation --------------------------------------------------------

    def test_init_does_not_raise(self, sg_table):
        """SpyglassMixin.__init__ must not raise when offline."""
        _ = sg_table()  # would raise before Phase 3

    def test_flag_propagated(self, sg_instance):
        assert sg_instance._no_connection is True

    def test_part_flag_propagated(self, sg_part_table):
        assert sg_part_table.Part._no_connection is True

    # --- NoConnectionMixin overrides still active via SpyglassMixin MRO ------

    def test_is_declared_false(self, sg_instance):
        assert sg_instance.is_declared is False

    def test_len_zero(self, sg_instance):
        assert len(sg_instance) == 0

    def test_bool_false(self, sg_instance):
        assert bool(sg_instance) is False

    def test_fetch_empty(self, sg_instance):
        with pytest.warns(UserWarning):
            assert len(sg_instance.fetch()) == 0

    def test_fetch1_raises(self, sg_instance):
        with pytest.raises(DataJointError):
            _ = sg_instance.fetch1

    # --- repr / preview through SpyglassMixin MRO ----------------------------

    def test_repr_offline_notice(self, sg_instance):
        assert "[no database connection]" in repr(sg_instance)

    def test_repr_primary_key(self, sg_instance):
        assert "*session_id" in repr(sg_instance)

    def test_repr_secondary_attr(self, sg_instance):
        assert "weight" in repr(sg_instance)

    def test_repr_html_offline(self, sg_instance):
        html = sg_instance._repr_html_()
        assert "no database connection" in html
        assert "SgOffline" in html

    def test_preview_matches_repr(self, sg_instance):
        assert sg_instance.preview() == repr(sg_instance)

    # --- SpyglassAnalysis guard -----------------------------------------------

    def test_analysis_init_does_not_raise(self, offline_schema):
        """SpyglassAnalysis.__init__ must not raise when offline."""
        from spyglass.utils import SpyglassMixin
        from spyglass.utils.mixins.analysis import AnalysisMixin

        @offline_schema
        class SgAnalysisOffline(SpyglassMixin, AnalysisMixin, dj.Manual):
            definition = """
            analysis_id : varchar(32)
            ---
            result : float
            """

        instance = SgAnalysisOffline()
        assert instance._no_connection is True


# ---------------------------------------------------------------------------
# TestNoConnectionImport — end-to-end: import without a live DB
# ---------------------------------------------------------------------------


class TestNoConnectionImport:
    """Verify spyglass modules using SpyglassSchema import cleanly without a DB.

    All tests use subprocess isolation so that importing spyglass.common does
    not contaminate the in-process module cache and corrupt the live-DB session
    for other test modules.
    """

    def test_common_nwbfile_module_importable(self):
        """spyglass.common.common_nwbfile must import without raising.

        Runs in a subprocess so that schema activation (which may fail when no
        DB is configured) does not set ``_no_connection`` flags on classes in
        the shared module cache used by live-DB tests.
        """
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "-W",
                "ignore",
                "-c",
                "from spyglass.common.common_nwbfile import Nwbfile; print('ok')",
            ],
            capture_output=True,
            text=True,
        )
        assert "ok" in result.stdout, (
            "Import raised an exception:\n" + result.stderr
        )

    def test_all_schemas_use_spyglass_schema(self):
        """Every spyglass module must use SpyglassSchema, not dj.schema."""
        import subprocess

        result = subprocess.run(
            [
                "grep",
                "-r",
                "--include=*.py",
                "-l",
                r"schema\s*=\s*dj\.\(Schema\|schema\)\s*(",
                "src/spyglass",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1 or result.stdout.strip() == "", (
            "These files still use dj.schema() instead of SpyglassSchema:\n"
            + result.stdout
        )

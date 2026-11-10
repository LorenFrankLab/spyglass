"""The v2 part-table import probe in ``spikesorting_merge``.

An ``ImportError`` (a missing or incompatible package) is tolerated, so an
environment without v2 still loads the merge table, and logged with
``logger.warning``. Any other error is a v2 bug and raises.
"""

import sys
import types

import pytest


def _force_v2_curation_error(monkeypatch, exc):
    """Make ``from spyglass.spikesorting.v2.curation import CurationV2`` raise
    ``exc`` by injecting a stand-in module whose attribute access raises."""
    name = "spyglass.spikesorting.v2.curation"

    class _Boom(types.ModuleType):
        def __getattr__(self, attr):
            raise exc

    monkeypatch.setitem(sys.modules, name, _Boom(name))


def _force_v2_concat_member_error(monkeypatch, exc):
    """Make importing the concat-member merge target raise ``exc``."""
    name = "spyglass.spikesorting.v2.concat_member_curation"

    class _Boom(types.ModuleType):
        def __getattr__(self, attr):
            raise exc

    monkeypatch.setitem(sys.modules, name, _Boom(name))


def test_v2_import_error_is_tolerated_and_logged(dj_conn, monkeypatch, caplog):
    # Import the probe (declaring spikesorting_merge's schema against the live
    # ``dj_conn`` DB) BEFORE patching, so the patch affects only our direct
    # ``_probe_v2_curation()`` call rather than the module's own load-time one.
    from spyglass.spikesorting.spikesorting_merge import _probe_v2_curation

    exc = ImportError("v2 module missing")
    _force_v2_curation_error(monkeypatch, exc)

    with caplog.at_level("WARNING"):
        curation, captured = _probe_v2_curation()

    assert curation is None
    assert captured is exc

    # Surfaced, not silent.
    assert any(
        "spikesorting v2 is unavailable" in r.message
        and type(exc).__name__ in r.message
        for r in caplog.records
    ), (
        "probe failure not warning-logged: "
        f"{[record.message for record in caplog.records]}"
    )


def test_concat_member_import_error_is_tolerated_and_logged(
    dj_conn, monkeypatch, caplog
):
    """The member-output probe tolerates and surfaces an ImportError."""
    from spyglass.spikesorting.spikesorting_merge import (
        _probe_v2_concat_member_curation,
    )

    exc = ImportError("concat member module missing")
    _force_v2_concat_member_error(monkeypatch, exc)
    with caplog.at_level("WARNING"):
        member_table = _probe_v2_concat_member_curation()

    assert member_table is None
    assert any(
        "concat-member outputs are unavailable" in record.message
        and type(exc).__name__ in record.message
        for record in caplog.records
    )


def test_other_v2_import_errors_raise(dj_conn, monkeypatch):
    """A non-ImportError while importing v2 is a bug, not a missing extra."""
    from spyglass.spikesorting.spikesorting_merge import (
        _probe_v2_concat_member_curation,
        _probe_v2_curation,
    )

    _force_v2_curation_error(monkeypatch, RuntimeError("v2 bug at import"))
    with pytest.raises(RuntimeError, match="v2 bug at import"):
        _probe_v2_curation()
    _force_v2_concat_member_error(monkeypatch, RuntimeError("member bug"))
    with pytest.raises(RuntimeError, match="member bug"):
        _probe_v2_concat_member_curation()

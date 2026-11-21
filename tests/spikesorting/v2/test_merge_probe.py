"""The v2 part-table import probe in ``spikesorting_merge``.

Any import error is tolerated so the merge table, which NWB ingestion
declares for every user, still loads without v2. An ``ImportError`` (a missing
or incompatible package) is logged with ``logger.warning``; any other error is
logged with its traceback at ERROR level as a likely v2 bug.
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
        member_table, captured = _probe_v2_concat_member_curation()

    assert member_table is None
    assert captured is exc
    assert any(
        "concat-member outputs are unavailable" in record.message
        and type(exc).__name__ in record.message
        for record in caplog.records
    )


@pytest.mark.parametrize(
    "probe_name, force, module",
    [
        (
            "_probe_v2_curation",
            _force_v2_curation_error,
            "spyglass.spikesorting.v2.curation",
        ),
        (
            "_probe_v2_concat_member_curation",
            _force_v2_concat_member_error,
            "spyglass.spikesorting.v2.concat_member_curation",
        ),
    ],
)
def test_other_v2_import_errors_are_tolerated_and_logged_as_errors(
    dj_conn, monkeypatch, caplog, probe_name, force, module
):
    """A non-ImportError at v2 import must not break the merge table for
    v0/v1 users; it is logged with its traceback, naming the module."""
    from spyglass.spikesorting import spikesorting_merge as m

    exc = RuntimeError("v2 bug at import")
    force(monkeypatch, exc)
    with caplog.at_level("WARNING"):
        table, captured = getattr(m, probe_name)()

    assert table is None
    assert captured is exc
    errors = [
        record
        for record in caplog.records
        if record.levelname == "ERROR" and module in record.getMessage()
    ]
    assert errors, [record.getMessage() for record in caplog.records]
    assert errors[0].exc_info is not None
    assert "v0/v1" in errors[0].getMessage()


@pytest.mark.parametrize(
    "cause, hint",
    [
        (ImportError("no module named x"), "install"),
        (RuntimeError("v2 bug at import"), "traceback"),
    ],
)
def test_raise_v2_unavailable_reports_the_cause(
    dj_conn, monkeypatch, cause, hint
):
    """The error names the caller, the cause's type and message, a hint for
    that kind of failure, and chains the cause."""
    from spyglass.spikesorting import spikesorting_merge as m

    monkeypatch.setattr(m, "_v2_import_error", cause)
    with pytest.raises(RuntimeError) as info:
        m._raise_v2_unavailable("SomeCaller.method")
    message = str(info.value)
    assert "SomeCaller.method" in message
    assert f"{type(cause).__name__}: {cause}" in message
    assert hint in message
    assert info.value.__cause__ is cause

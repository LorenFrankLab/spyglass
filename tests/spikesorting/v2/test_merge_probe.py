"""The eager v2 part-table probe in ``spikesorting_merge``.

``spikesorting_merge`` imports its v2 part targets inside broad
``except Exception`` boundaries so v0/v1-only environments can still load the
merge table when either optional module fails to import.

The fix for silent failures is visibility, not narrowing: the probe logs the
captured cause via ``logger.warning`` while still tolerating it. These tests
exercise the probe helper directly (no schema reload, no DB) by forcing the v2
``curation`` import to raise.
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


@pytest.mark.parametrize(
    "exc",
    [
        ImportError("v2 module missing"),
        RuntimeError("v2 module raised at import (e.g. a version skew)"),
    ],
    ids=["import_error", "non_import_error"],
)
def test_unexpected_v2_import_error_is_logged(
    dj_conn, monkeypatch, caplog, exc
):
    # Import the probe (declaring spikesorting_merge's schema against the live
    # ``dj_conn`` DB) BEFORE patching, so the patch affects only our direct
    # ``_probe_v2_curation()`` call rather than the module's own load-time one.
    from spyglass.spikesorting.spikesorting_merge import _probe_v2_curation

    _force_v2_curation_error(monkeypatch, exc)

    with caplog.at_level("WARNING"):
        curation, captured = _probe_v2_curation()

    # Tolerated, not propagated -- including a non-ImportError, so a v0/v1 env
    # still loads the merge table even if the v2 layer raises at import.
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


@pytest.mark.parametrize(
    "exc",
    [
        ImportError("concat member module missing"),
        RuntimeError("concat member module raised at import"),
    ],
    ids=["import_error", "non_import_error"],
)
def test_unexpected_concat_member_import_error_is_logged(
    dj_conn, monkeypatch, caplog, exc
):
    """The optional member-output probe tolerates and surfaces any failure."""
    from spyglass.spikesorting.spikesorting_merge import (
        _probe_v2_concat_member_curation,
    )

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

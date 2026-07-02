"""The eager v2 part-table probe in ``spikesorting_merge``.

``spikesorting_merge`` imports the v2 ``CurationV2`` part target inside a broad
``except Exception`` so that v0/v1-only environments -- which may lack the
optional modern-SpikeInterface / Pydantic v2 dependencies -- can still load the
merge table when the v2 ``curation`` import fails. That breadth is deliberate:
it tolerates any import-time failure of the optional v2 layer (an ``ImportError``
from a missing dependency, or any other exception raised while the v2 modules
import) rather than letting it break ``spikesorting_merge`` import for everyone.

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
    ), f"probe failure not warning-logged: {[r.message for r in caplog.records]}"

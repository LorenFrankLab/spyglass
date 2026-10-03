"""Hermetic (no-DB) tests for the v1->v2 migration surface.

Covers:

- A stub module listed in ``STUB_MODULES_WITH_V1_FALLBACK`` /
  ``STUB_MODULES_NO_V1_FALLBACK`` (both lists are currently empty) must raise
  an informative ``ImportError`` on public-name access -- whether the caller
  wrote ``from m import X`` or ``import m; m.X`` -- while a bare ``import m``
  (no attribute access) still succeeds, and dunder probes by the import
  machinery do NOT leak into the error message.
- The v1->v2 migration guide covers each migration category.

These tests need no DataJoint server: importing a v2 submodule triggers
the package ``__init__`` (SpikeInterface import only) but never opens a
database connection, mirroring ``test_params_validation.py``.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

# (module path, a representative public symbol, v1 fallback submodule hint)
# ``metric_curation`` and ``figpack_curation`` are intentionally absent: both are
# real v2 table modules -- ``CurationEvaluation`` replaces v1 MetricCuration +
# BurstPair, and ``FigPackCuration`` / ``FigPackCurationSelection`` replace the v1
# FigURL chain -- so neither is a stub.
STUB_MODULES_WITH_V1_FALLBACK: list[tuple[str, str, str]] = []

# (module path, a representative public symbol) -- no v1 fallback exists.
# ``matcher_protocol`` and ``unit_matching`` are intentionally absent: both are
# real modules (the cross-session matcher protocol + registry, and the
# UnitMatch / TrackedUnit DataJoint tables), not stubs.
STUB_MODULES_NO_V1_FALLBACK: list[tuple[str, str]] = []

ALL_STUB_MODULES = [m[0] for m in STUB_MODULES_WITH_V1_FALLBACK] + [
    m[0] for m in STUB_MODULES_NO_V1_FALLBACK
]

_DUNDERS = ("__path__", "__all__", "__spec__", "__file__", "__loader__")


def _from_import(module_path: str, name: str) -> None:
    """Execute a genuine ``from <module_path> import <name>`` statement.

    Using ``exec`` (rather than ``getattr``) exercises CPython's
    from-import error path -- the one that collapses ``AttributeError``
    into a generic "cannot import name" ``ImportError`` but lets a
    ``__getattr__``-raised ``ImportError`` propagate with its message
    intact.
    """
    exec(f"from {module_path} import {name}")


@pytest.mark.parametrize(
    "module_path, symbol, v1_hint", STUB_MODULES_WITH_V1_FALLBACK
)
def test_stub_module_from_import_raises_with_custom_message(
    module_path, symbol, v1_hint
):
    """``from m import X`` keeps the custom hint (v1-fallback case)."""
    with pytest.raises(ImportError) as excinfo:
        _from_import(module_path, symbol)
    message = str(excinfo.value)
    assert symbol in message
    assert v1_hint in message


@pytest.mark.parametrize("module_path, symbol", STUB_MODULES_NO_V1_FALLBACK)
def test_stub_module_from_import_no_v1_fallback_claim(module_path, symbol):
    """v2-only stubs name the symbol but claim no v1 fallback."""
    with pytest.raises(ImportError) as excinfo:
        _from_import(module_path, symbol)
    message = str(excinfo.value)
    assert symbol in message
    assert "spikesorting.v1" not in message
    # Points to a future v2 release rather than a (non-existent) v1 fallback.
    assert "future v2 release" in message


@pytest.mark.parametrize(
    "module_path, symbol, v1_hint", STUB_MODULES_WITH_V1_FALLBACK
)
def test_stub_module_attribute_access_raises_with_custom_message(
    module_path, symbol, v1_hint
):
    """``import m; m.X`` shares the custom message with from-import."""
    module = importlib.import_module(module_path)
    with pytest.raises(ImportError) as excinfo:
        getattr(module, symbol)
    message = str(excinfo.value)
    assert symbol in message
    assert v1_hint in message


@pytest.mark.parametrize("module_path", ALL_STUB_MODULES)
def test_stub_module_bare_import_succeeds(module_path):
    """A bare import (no attribute access) loads with a docstring."""
    module = importlib.import_module(module_path)
    assert module.__doc__
    assert module.__doc__.strip()


@pytest.mark.parametrize("module_path", ALL_STUB_MODULES)
def test_stub_module_from_import_does_not_leak_dunder_in_message(module_path):
    """A missing public name reports itself, never a dunder probe."""
    with pytest.raises(ImportError) as excinfo:
        _from_import(module_path, "DoesNotExist")
    message = str(excinfo.value)
    assert "DoesNotExist" in message
    for dunder in _DUNDERS:
        assert dunder not in message


def _repo_root() -> Path:
    # tests/spikesorting/v2/test_legacy_stub_imports.py -> repo root
    return Path(__file__).resolve().parents[3]


def test_migration_guide_covers_v1_to_v2_categories():
    """The v1->v2 migration guide names each migration category."""
    guide = (
        _repo_root()
        / "docs"
        / "src"
        / "Features"
        / "SpikeSortingV2_Migration.md"
    ).read_text()
    lower = guide.lower()
    # One loose marker per migration category (substring, not exact text).
    for marker in (
        "sorter_param_name",  # API renames
        "recording_id",  # dropped/relocated data
        "noise_levels",  # schema-default flips
        "off-by-one",  # boundary semantics
        "multi-channel",  # multi-channel clusterless differences
        "seed",  # determinism
        "amplitude_thresh_uv",  # default thresholds
        "metriccuration",  # removed v1 features
        "spikesorting_artifact_detection_v2",  # tags
    ):
        assert marker in lower, f"missing migration-guide marker: {marker!r}"

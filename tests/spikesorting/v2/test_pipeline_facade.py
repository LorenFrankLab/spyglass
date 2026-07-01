"""Pin the public import surface of the ``pipeline`` facade.

``pipeline.py`` was split into ``_pipeline_*`` submodules (presets, geometry,
preflight, reporting, run); it re-exports every public name so notebook/user
imports (``from spyglass.spikesorting.v2.pipeline import ...``) stay stable.
This DB-free test fails if any public name stops resolving from the facade --
the contract the split must preserve.
"""

from __future__ import annotations

from spyglass.spikesorting.v2._pipeline_public import (
    PACKAGE_ROOT_REEXPORTS,
    PIPELINE_FACADE_EXPORTS,
)


def test_pipeline_facade_reexports_public_api():
    """Every public name resolves from ``spyglass.spikesorting.v2.pipeline``."""
    import spyglass.spikesorting.v2.pipeline as pl

    assert tuple(pl.__all__) == PIPELINE_FACADE_EXPORTS
    missing = [
        name for name in PIPELINE_FACADE_EXPORTS if not hasattr(pl, name)
    ]
    assert not missing, f"pipeline facade no longer exports: {missing}"


# The primary orchestration entrypoints a first-time user reaches for. These
# must also import from the PACKAGE ROOT (``spyglass.spikesorting.v2``), where
# ``initialize_v2_defaults`` already lives, so the natural
# ``from spyglass.spikesorting.v2 import run_v2_pipeline`` does not raise.
_ROOT_REEXPORTS = PACKAGE_ROOT_REEXPORTS


def test_package_root_reexports_primary_entrypoints():
    """The main entrypoints resolve from ``spyglass.spikesorting.v2`` itself.

    ``initialize_v2_defaults`` is on the package root, so a newcomer who then
    types ``from spyglass.spikesorting.v2 import run_v2_pipeline`` should not
    hit an ImportError over an import-path split.
    """
    import importlib

    v2 = importlib.import_module("spyglass.spikesorting.v2")

    missing = [name for name in _ROOT_REEXPORTS if not hasattr(v2, name)]
    assert not missing, f"package root no longer re-exports: {missing}"

    # The re-exports are the real functions, not shadow definitions.
    assert (
        v2.run_v2_pipeline
        is __import__(
            "spyglass.spikesorting.v2.pipeline", fromlist=["run_v2_pipeline"]
        ).run_v2_pipeline
    )


def test_package_root_reexports_are_discoverable():
    """The root re-exports appear in ``dir()`` and ``__all__`` for discovery."""
    import spyglass.spikesorting.v2 as v2

    listing = dir(v2)
    for name in _ROOT_REEXPORTS:
        assert (
            name in listing
        ), f"{name} missing from dir(spyglass.spikesorting.v2)"
        assert name in v2.__all__, f"{name} missing from __all__"


def test_pipeline_facade_is_a_thin_reexport():
    """The facade defines no implementation itself -- it only re-exports.

    Guards against code creeping back into the facade: every public name it
    exposes must be defined in a ``_pipeline_*`` submodule, not in
    ``pipeline.py``.
    """
    import spyglass.spikesorting.v2.pipeline as pl

    for name in PIPELINE_FACADE_EXPORTS:
        obj = getattr(pl, name)
        module = getattr(obj, "__module__", "")
        assert module != pl.__name__, (
            f"{name} is defined in the facade; it should live in a "
            f"_pipeline_* submodule and be re-exported (got {module!r})"
        )

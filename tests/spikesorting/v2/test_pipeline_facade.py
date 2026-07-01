"""Pin the public import surface of the ``pipeline`` facade.

``pipeline.py`` was split into ``_pipeline_*`` submodules (presets, geometry,
preflight, reporting, run); it re-exports every public name so notebook/user
imports (``from spyglass.spikesorting.v2.pipeline import ...``) stay stable.
This DB-free test fails if any public name stops resolving from the facade --
the contract the split must preserve.
"""

from __future__ import annotations

# The public surface notebooks/users import from the pipeline facade. Keep in
# sync with the re-exports in ``pipeline.py``; a drop here is a breaking change.
_PUBLIC_API = (
    # presets
    "list_pipeline_presets",
    "describe_pipeline_presets",
    "describe_pipeline_preset",
    # geometry
    "describe_sort_groups",
    "plot_sort_group_geometry",
    # preflight
    "PreflightCheck",
    "PreflightReport",
    "PreflightSessionReport",
    "preflight_v2_pipeline",
    "preflight_v2_pipeline_session",
    # reporting
    "describe_parameter_rows",
    "describe_run",
    "describe_units",
    # run
    "run_v2_pipeline",
    "run_v2_pipeline_session",
)


def test_pipeline_facade_reexports_public_api():
    """Every public name resolves from ``spyglass.spikesorting.v2.pipeline``."""
    import spyglass.spikesorting.v2.pipeline as pl

    missing = [name for name in _PUBLIC_API if not hasattr(pl, name)]
    assert not missing, f"pipeline facade no longer exports: {missing}"


# The primary orchestration entrypoints a first-time user reaches for. These
# must also import from the PACKAGE ROOT (``spyglass.spikesorting.v2``), where
# ``initialize_v2_defaults`` already lives, so the natural
# ``from spyglass.spikesorting.v2 import run_v2_pipeline`` does not raise.
_ROOT_REEXPORTS = (
    "run_v2_pipeline",
    "run_v2_pipeline_session",
    "run_v2_unit_match",
    "plan_v2_unit_match",
    "preflight_v2_pipeline",
    "preflight_v2_pipeline_session",
    "describe_run",
    "describe_pipeline_presets",
    "list_pipeline_presets",
)


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

    for name in _PUBLIC_API:
        obj = getattr(pl, name)
        module = getattr(obj, "__module__", "")
        assert module != pl.__name__, (
            f"{name} is defined in the facade; it should live in a "
            f"_pipeline_* submodule and be re-exported (got {module!r})"
        )

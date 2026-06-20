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
    "describe_preset",
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

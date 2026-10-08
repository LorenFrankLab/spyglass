"""Pin the public import surface of the ``pipeline`` facade.

``pipeline.py`` was split into ``_pipeline_*`` submodules (presets, geometry,
preflight, reporting, run); it re-exports every public name so notebook/user
imports (``from spyglass.spikesorting.v2.pipeline import ...``) stay stable.
This DB-free test fails if any public name stops resolving from the facade --
the contract the split must preserve.
"""

from __future__ import annotations

from spyglass.spikesorting.v2._orchestration.exports import (
    PACKAGE_ROOT_REEXPORTS,
    PIPELINE_FACADE_EXPORTS,
)

# These required names are independent of the production export manifest.
# Additive exports are allowed; deleting a manifest entry must not also erase
# the test's expectation and silently shrink the public API.
_ROOT_REEXPORTS = frozenset(
    [
        "run_v2_pipeline",
        "run_v2_pipeline_session",
        "estimate_motion",
        "run_v2_unit_match",
        "plan_v2_unit_match",
        "plan_v2_unit_match_from_sorts",
        "preflight_v2_pipeline",
        "preflight_v2_pipeline_session",
        "describe_run",
        "describe_units",
        "describe_parameter_rows",
        "describe_sort_groups",
        "describe_pipeline_presets",
        "describe_pipeline_preset",
        "describe_recommendation_status",
        "list_pipeline_presets",
        "register_pipeline_preset",
        "clone_pipeline_preset",
        "describe_unit_match_choices",
        "plot_sort_group_geometry",
        "RunResult",
        "CurationRef",
        "EvaluationSpec",
        "EvaluationResult",
        "MergedCuration",
        "MergeEvaluateReceipt",
        "ReviewProfileRef",
        "FigPackReview",
        "CurationChangeSet",
        "MergeLabelConflict",
        "ReviewImportReceipt",
        "ReviewStageStatus",
        "AnnotationDefinitionRef",
        "AnnotationSetRef",
        "read_unit_properties",
        "open_curation_analyzer",
        "select_units_for_analysis",
        "UnitSelectionReceipt",
        "SelectedGroup",
        "V2_UNIT_SELECTION_POLICIES",
    ]
)
_FACADE_REEXPORTS = _ROOT_REEXPORTS | frozenset(
    [
        "PreflightCheck",
        "PreflightReport",
        "PreflightSessionReport",
        "UnitMatchPlan",
        "UnitMatchInputPlan",
        "PipelineOutcome",
        "PipelineStageSeconds",
        "RunV2ConcatSummary",
        "RunV2PipelineInputs",
        "RunV2PipelineSessionFailed",
        "RunV2PipelineSessionInputs",
        "RunV2PipelineSessionOk",
        "RunV2PipelineSessionRequiredInputs",
        "RunV2PipelineSessionResult",
        "RunV2PipelineSummary",
        "RunV2SingleSessionSummary",
        "RunV2UnitMatchSummary",
        "EstimateMotionReceipt",
        "MotionEstimateDiagnostics",
        "MotionMode",
        "SourceMode",
        "StageStatus",
        "UnitMatchCurationChoice",
        "UnitMatchInputSummary",
        "UnitMatchMemberChoices",
        "UnitMatchStageSeconds",
    ]
)


def test_pipeline_facade_reexports_public_api():
    """Every public name resolves from ``spyglass.spikesorting.v2.pipeline``."""
    import spyglass.spikesorting.v2.pipeline as pl

    assert tuple(pl.__all__) == PIPELINE_FACADE_EXPORTS
    assert _FACADE_REEXPORTS <= set(pl.__all__)
    missing = [
        name for name in PIPELINE_FACADE_EXPORTS if not hasattr(pl, name)
    ]
    assert not missing, f"pipeline facade no longer exports: {missing}"


# The primary orchestration entrypoints a first-time user reaches for. These
# must also import from the PACKAGE ROOT (``spyglass.spikesorting.v2``), where
# ``initialize_v2_defaults`` already lives, so the natural
# ``from spyglass.spikesorting.v2 import run_v2_pipeline`` does not raise.
def test_package_root_reexports_primary_entrypoints():
    """The main entrypoints resolve from ``spyglass.spikesorting.v2`` itself.

    ``initialize_v2_defaults`` is on the package root, so a newcomer who then
    types ``from spyglass.spikesorting.v2 import run_v2_pipeline`` should not
    hit an ImportError over an import-path split.
    """
    import importlib

    v2 = importlib.import_module("spyglass.spikesorting.v2")
    assert _ROOT_REEXPORTS <= set(PACKAGE_ROOT_REEXPORTS)

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

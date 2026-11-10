"""Pin the public import surface of the ``pipeline`` facade.

``pipeline.py`` re-exports public names from ``_orchestration`` so
``from spyglass.spikesorting.v2.pipeline import ...`` stays stable. DB-free.
"""

from __future__ import annotations

# Listed independently of the production ``__all__`` lists, so removing an
# entry there cannot silently shrink the public API.
_ROOT_REEXPORTS = frozenset(
    [
        "run_v2_pipeline",
        "run_v2_concat_pipeline",
        "run_v2_pipeline_session",
        "estimate_motion",
        "estimate_concat_motion",
        "run_v2_unit_match",
        "plan_v2_unit_match",
        "plan_v2_unit_match_from_sorts",
        "plan_v2_unit_match_from_curations",
        "preflight_v2_unit_match",
        "preflight_v2_pipeline",
        "preflight_v2_pipeline_session",
        "describe_run",
        "describe_units",
        "describe_parameter_rows",
        "describe_sort_groups",
        "describe_epochs",
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
        "MergeEvaluateReceipt",
        "ReviewProfileRef",
        "FigPackReview",
        "CurationChangeSet",
        "MergeLabelConflict",
        "ReviewImportReceipt",
        "ReviewStageStatus",
        "open_curation_analyzer",
        "select_units_for_analysis",
        "resolve_unit_selection",
        "UnitSelectionSnapshot",
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
        "UnitMatchPreflightCheck",
        "UnitMatchPreflightReport",
        "UnitMatchPlan",
        "UnitMatchInputPlan",
        "PipelineOutcome",
        "PipelineStageSeconds",
        "RunV2ConcatSummary",
        "RunV2ConcatPipelineInputs",
        "RunV2PipelineInputs",
        "RunV2PipelineOptions",
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

    assert len(pl.__all__) == len(set(pl.__all__))
    assert _FACADE_REEXPORTS <= set(pl.__all__)
    missing = [name for name in pl.__all__ if not hasattr(pl, name)]
    assert not missing, f"pipeline facade no longer exports: {missing}"


def test_package_root_reexports_primary_entrypoints():
    """The main entrypoints resolve from ``spyglass.spikesorting.v2`` itself.

    ``from spyglass.spikesorting.v2 import run_v2_pipeline`` works like
    ``initialize_v2_defaults``.
    """
    import importlib

    import spyglass.spikesorting.v2.pipeline as pl

    v2 = importlib.import_module("spyglass.spikesorting.v2")
    assert _ROOT_REEXPORTS <= set(v2.__all__)
    lazy = set(v2.__all__) - {
        "initialize_v2_defaults",
        "verify_v2_default_catalog",
        "CurationLabel",
    }
    assert lazy <= set(pl.__all__)

    missing = [name for name in _ROOT_REEXPORTS if not hasattr(v2, name)]
    assert not missing, f"package root no longer re-exports: {missing}"

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
    """Every public name in the facade is defined in a submodule, not in
    ``pipeline.py``.
    """
    import spyglass.spikesorting.v2.pipeline as pl

    for name in pl.__all__:
        obj = getattr(pl, name)
        module = getattr(obj, "__module__", "")
        assert module != pl.__name__, (
            f"{name} is defined in the facade; it should live in a "
            f"_pipeline_* submodule and be re-exported (got {module!r})"
        )

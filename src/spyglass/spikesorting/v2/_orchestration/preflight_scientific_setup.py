"""The scientific setup a run states: each stage's rows and sort configuration."""

from __future__ import annotations

from spyglass.spikesorting.v2._orchestration.types import MotionMode
from spyglass.spikesorting.v2._orchestration.preflight_motion import (
    MotionRecipe,
)

#: No recipe has passed probe-specific validation.
MOTION_RECIPE_STATUS = (
    "experimental: no motion recipe is validated for a probe; inspect the "
    "saved estimate before relying on a correction"
)


def _describe_motion(
    motion_mode, motion_recipe: "MotionRecipe | None", *, concat: bool
) -> dict:
    """The motion stage a run executes, for the scientific-setup display.

    Parameters
    ----------
    motion_mode : {"off", "estimate", "apply"}
    motion_recipe : MotionRecipe or None
        The resolved recipe; ``None`` for ``"off"`` or when it could not be
        resolved (preflight then reports why).
    concat : bool
        Whether the source is a concatenation.

    Returns
    -------
    dict
        ``mode``, ``description`` and, with a recipe, ``recipe``,
        ``estimation_recipe``, ``estimation_preset`` (resolved), for
        ``"apply"`` ``interpolation_recipe`` and ``border_mode``, and
        ``status``.
    """
    if motion_mode == "off":
        return {
            "mode": "off",
            "description": (
                "No motion stage: the sort reads the masked, uncorrected "
                "source. DriftEstimate is diagnostic only."
            ),
        }
    source = "the concatenation" if concat else "the recording under its mask"
    description = (
        f"Motion is estimated once from {source} and saved for inspection; "
        "the sort reads the uncorrected source (the same sort as 'off')."
        if motion_mode == "estimate"
        else f"Motion is estimated once from {source}, applied to it, and "
        "the sort reads the motion-corrected recording."
    )
    motion = {"mode": motion_mode, "description": description}
    if motion_recipe is None:
        return motion
    motion.update(
        recipe=motion_recipe.recipe["motion_correction_params_name"],
        estimation_recipe=motion_recipe.recipe["motion_estimation_params_name"],
        estimation_preset=motion_recipe.resolved_estimation["preset"],
    )
    if motion_mode == "apply":
        motion.update(
            interpolation_recipe=motion_recipe.recipe[
                "motion_interpolation_params_name"
            ],
            border_mode=motion_recipe.interpolation_params["border_mode"],
        )
    motion["status"] = MOTION_RECIPE_STATUS
    return motion


def describe_scientific_setup(
    bundle,
    group_keys,
    effective_config=None,
    *,
    manual_excluded_times=None,
    concat: bool = False,
    motion_mode: MotionMode = "off",
    motion_recipe: "MotionRecipe | None" = None,
):
    """Resolve the preprocessing and artifact rows execution uses for display.

    ``concat=True`` adds where the artifact mask is applied. ``motion``
    describes the motion stage; a Kilosort4 sort adds its own ``nblocks``.
    """
    from spyglass.spikesorting.v2.artifact import ArtifactDetectionParameters
    from spyglass.spikesorting.v2.recording import (
        PreprocessingParameters,
        SortGroupV2,
    )

    preprocessing = (
        PreprocessingParameters
        & {"preprocessing_params_name": bundle.preprocessing_params_name}
    ).fetch("params")
    artifacts = (
        (
            ArtifactDetectionParameters
            & {
                "artifact_detection_params_name": bundle.artifact_detection_params_name
            }
        ).fetch("params")
        if bundle.artifact_detection_params_name is not None
        else []
    )
    result = {
        "preprocessing_recipe": bundle.preprocessing_params_name,
        "preprocessing": (
            dict(preprocessing[0]) if len(preprocessing) else None
        ),
        "references": (SortGroupV2 & group_keys).fetch(
            "nwb_file_name",
            "sort_group_id",
            "reference_mode",
            "reference_electrode_id",
            as_dict=True,
        ),
        "artifact_recipe": bundle.artifact_detection_params_name,
        "artifact_detection": dict(artifacts[0]) if len(artifacts) else None,
        "motion": _describe_motion(motion_mode, motion_recipe, concat=concat),
    }
    if manual_excluded_times:
        result["manual_excluded_times"] = manual_excluded_times
    if concat:
        result["artifact_application"] = (
            "No artifact masking selected."
            if not manual_excluded_times
            and (
                bundle.artifact_detection_params_name is None
                or (len(artifacts) and not artifacts[0].get("detect", True))
            )
            else "Per member before concatenation."
        )
    if bundle.sorter == "kilosort4":
        params = (effective_config or {}).get("si_sorter_params", {})
        result["motion"]["sorter_correction"] = {
            "sorter": "kilosort4",
            "nblocks": params.get("nblocks"),
        }
    return result


def resolve_preset_sort_config(bundle) -> "dict | None":
    """Resolve what a preset's sort stage would execute, or ``None`` if unset.

    Runs the dispatcher's ``resolve_sort_config`` so preflight, the run
    receipt and execution agree. Returns ``EffectiveSortConfig.as_dict()``.
    """
    from spyglass.spikesorting.v2._sorting.dispatch import resolve_sort_config
    from spyglass.spikesorting.v2.sorting import SorterParameters
    from spyglass.spikesorting.v2._core.job_config import _resolved_job_kwargs

    rows = (
        SorterParameters
        & {
            "sorter": bundle.sorter,
            "sorter_params_name": bundle.sorter_params_name,
        }
    ).fetch(as_dict=True)
    if len(rows) != 1:
        return None
    row = rows[0]
    return resolve_sort_config(
        row["sorter"],
        row["params"],
        job_kwargs=_resolved_job_kwargs(row["job_kwargs"]),
        execution_params=row.get("execution_params"),
    ).as_dict()

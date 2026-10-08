"""Validated parameter schema for the motion-estimation parameter table.

A ``MotionEstimationParameters`` row names one SpikeInterface motion preset plus
per-step overrides. SpikeInterface merges each override one level deep over the
preset step (``dict(preset_step, **user_step)``); the fully resolved
configuration that is actually passed to SpikeInterface, including every
signature default the preset leaves implicit, is built by
``spyglass.spikesorting.v2._motion.estimation.resolve_estimation_params``.

This schema only checks the shape of the blob and rejects overrides that would
re-bind an argument the estimator passes itself, change where outputs go, or
select a compute device other than the CPU. Whether each remaining key is a
real parameter of the selected SpikeInterface method is checked when the row is
resolved (``MotionEstimationParameters.insert`` resolves every row).

No SpikeInterface import: the default rows are built from this schema at
import time of the recipe catalog.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

MOTION_ESTIMATION_SCHEMA_VERSION = 1

#: Override keys that re-bind arguments the estimator supplies itself, redirect
#: outputs, or ask SpikeInterface for extra return values, per step.
_FORBIDDEN_STEP_KEYS: dict[str, frozenset[str]] = {
    "detect_kwargs": frozenset(
        {"recording", "noise_levels", "return_output", "random_slices_kwargs"}
    ),
    "localize_peaks_kwargs": frozenset(
        {"recording", "parents", "return_output", "prototype"}
    ),
    "estimate_motion_kwargs": frozenset(
        {
            "recording",
            "peaks",
            "peak_locations",
            "extra_outputs",
            "progress_bar",
            "verbose",
            "margin_um",
            "precomputed_D_C_maxdisp",
            "post_transform",
            "amp_scale_fn",
        }
    ),
}

#: Arguments of ``compute_motion`` / ``correct_motion`` that do not belong in a
#: per-step override dict at any level.
_FORBIDDEN_ANY_STEP = frozenset(
    {
        "folder",
        "overwrite",
        "output_motion",
        "output_motion_info",
        "raise_error",
    }
)


class MotionEstimationParamsSchema(BaseModel):
    """Validated ``params`` for a ``MotionEstimationParameters`` row.

    Attributes
    ----------
    preset : {"rigid_fast", "dredge", "dredge_fast"}
        The SpikeInterface motion preset the overrides are merged over.
    detect_kwargs, localize_peaks_kwargs, estimate_motion_kwargs : dict
        Per-step overrides merged one level deep over the preset step, as
        SpikeInterface's ``compute_motion`` merges them. A nested dict (for
        example ``thomas_kw``) replaces the preset's value wholesale.
    select_kwargs : dict
        Must stay empty. The estimator reproduces ``compute_motion``'s
        detect-and-localize pipeline, which SpikeInterface runs only when no
        peak selection is configured.
    noise_levels_seed : int
        Seed of the random chunks the per-channel noise levels are estimated
        from. Pinned so a rerun or a cache rebuild detects the same peaks.
    max_gap_s : float
        Required. Longest stretch of unobserved time (s) the estimation clock
        keeps between two continuity spans (an acquisition gap or a
        concatenation member join); a longer real gap is shortened to it. All
        spans are estimated together on that clock, in one reference frame.
    schema_version : int
        Bumped on breaking field changes; rows insert at the current version.
    """

    model_config = ConfigDict(extra="forbid")

    preset: Literal["rigid_fast", "dredge", "dredge_fast"]
    detect_kwargs: dict = Field(default_factory=dict)
    select_kwargs: dict = Field(default_factory=dict)
    localize_peaks_kwargs: dict = Field(default_factory=dict)
    estimate_motion_kwargs: dict = Field(default_factory=dict)
    noise_levels_seed: int = Field(default=0, ge=0)
    max_gap_s: float = Field(ge=0, allow_inf_nan=False)
    schema_version: int = MOTION_ESTIMATION_SCHEMA_VERSION

    @model_validator(mode="after")
    def _reject_contract_changing_overrides(self):
        """Reject overrides the estimator owns or cannot honor."""
        if self.select_kwargs:
            raise ValueError(
                "select_kwargs must be empty: the estimator reproduces "
                "compute_motion's detect-and-localize pipeline, which "
                "SpikeInterface runs only without peak selection. Got "
                f"{sorted(self.select_kwargs)}."
            )
        for step, forbidden in _FORBIDDEN_STEP_KEYS.items():
            overrides = getattr(self, step)
            bad = sorted(set(overrides) & (forbidden | _FORBIDDEN_ANY_STEP))
            if bad:
                raise ValueError(
                    f"{step} may not set {bad}: the estimator supplies these "
                    "arguments itself (or they redirect outputs), so an "
                    "override would change what the stored estimate means."
                )
        device = self.estimate_motion_kwargs.get("device", "cpu")
        if device != "cpu":
            raise ValueError(
                "estimate_motion_kwargs['device'] must be 'cpu'; got "
                f"{device!r}. The estimate is pinned to the CPU so its "
                "numerics do not depend on which accelerator a worker has."
            )
        return self

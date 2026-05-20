"""Validated parameter schema for the motion-correction parameter table.

Schema lands in Phase 1 so ``MotionCorrectionParameters.insert_default()`` can
validate the ``params`` blob at insert time even though the consumer
(``ConcatenatedRecording.make()``) is gated behind ``NotImplementedError``
until Phase 3.

The MVP concat path persists only the sorter-ready corrected
``ElectricalSeries`` (plus member sample boundaries and a content hash); it
does NOT persist motion trajectories or motion-info side artifacts. This
schema rejects the SI ``correct_motion`` kwargs that would change that
contract -- ``output_motion``, ``output_motion_info``, ``folder``, and
``overwrite`` -- because allowing them would write untracked artifacts or
mutate the corrected-recording return type that Phase 3's
``ConcatenatedRecording`` materializer depends on.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


_SI_NATIVE_PRESETS = (
    "dredge",
    "medicine",
    "dredge_fast",
    "nonrigid_accurate",
    "nonrigid_fast_and_accurate",
    "rigid_fast",
    "kilosort_like",
)
"""``correct_motion`` presets exposed by SpikeInterface 0.104.

Recorded by Phase 0c at
``tests/spikesorting/v2/resolver/si0104-runtime.md``. Any drift in the
upstream signature should re-trigger that resolver check; this tuple is the
binding list this schema validates against.
"""


MotionPreset = Literal[
    "none",
    "auto",
    "dredge",
    "medicine",
    "dredge_fast",
    "nonrigid_accurate",
    "nonrigid_fast_and_accurate",
    "rigid_fast",
    "kilosort_like",
]

_FORBIDDEN_PRESET_KWARGS: frozenset[str] = frozenset(
    {"output_motion", "output_motion_info", "folder", "overwrite"}
)


class MotionCorrectionParamsSchema(BaseModel):
    """Validated schema for the motion-correction parameter blob.

    Fields
    ------
    preset
        One of: ``"none"`` (skip motion correction), ``"auto"`` (Spyglass
        alias resolved inside ``ConcatenatedRecording.make()`` -- maps to
        ``"rigid_fast"`` for same-day groups, raises for multi-day), or any
        SpikeInterface 0.104 ``correct_motion`` preset.
    preset_kwargs
        Optional dict of kwargs forwarded to ``correct_motion``. Cannot
        contain ``output_motion``, ``output_motion_info``, ``folder``, or
        ``overwrite``: the MVP concat path treats motion estimates as
        non-queryable, and the forbidden kwargs would either write
        untracked side artifacts or change the function's return type.
    """

    model_config = ConfigDict(extra="forbid")
    schema_version: int = 1
    preset: MotionPreset = "none"
    preset_kwargs: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_preset_kwargs(self) -> "MotionCorrectionParamsSchema":
        forbidden_used = _FORBIDDEN_PRESET_KWARGS & self.preset_kwargs.keys()
        if forbidden_used:
            raise ValueError(
                "preset_kwargs may not override "
                f"{sorted(forbidden_used)}; these kwargs change "
                "correct_motion's return type or write untracked side "
                "artifacts that the MVP concat schema does not persist"
            )
        if self.preset == "none" and self.preset_kwargs:
            raise ValueError(
                "preset_kwargs must be empty when preset='none' "
                "(motion correction is skipped)"
            )
        return self

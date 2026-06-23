"""Validated parameter schema for the motion-correction parameter table.

``MotionCorrectionParameters.insert_default()`` validates the ``params``
blob at insert time; ``ConcatenatedRecording.make()`` reads the chosen row,
resolves the Spyglass ``"auto"`` alias (``rigid_fast`` same-day, rejected
multi-day), and passes the preset to ``correct_motion`` on the concatenated
segment.

The MVP concat path persists only the corrected, unwhitened
``ElectricalSeries`` (plus member sample boundaries and a content hash);
it does NOT persist motion trajectories or motion-info side artifacts.
This schema rejects the SI ``correct_motion`` kwargs that would change
that contract -- ``output_motion``, ``output_motion_info``, ``folder``,
and ``overwrite`` -- because allowing them would write untracked
artifacts or mutate the corrected-recording return type that the future
concat materializer depends on.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

MOTION_CORRECTION_SCHEMA_VERSION = 1

# ``correct_motion`` presets exposed by SpikeInterface 0.104.
#
# The pinned signature was recorded at
# ``tests/spikesorting/v2/resolver/si0104-runtime.md``. Any drift in the
# upstream signature should re-trigger that resolver check; this tuple is the
# binding list this schema validates against.
_SI_NATIVE_PRESETS = (
    "dredge",
    "medicine",
    "dredge_fast",
    "nonrigid_accurate",
    "nonrigid_fast_and_accurate",
    "rigid_fast",
    "kilosort_like",
)


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

    Attributes
    ----------
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

        The forbidden-key check is the only insert-time guard on the
        *contents* of ``preset_kwargs`` (the ``preset='none'`` case also
        requires ``preset_kwargs`` to be empty). The remaining keys are
        validated against ``correct_motion``'s signature by SpikeInterface when
        ``ConcatenatedRecording.make`` forwards them, so an otherwise-bogus key
        surfaces there (at populate time) rather than at insert time. Per-key
        Pydantic modeling against the full ``correct_motion`` signature is
        deliberately not duplicated here.
    """

    model_config = ConfigDict(extra="forbid")
    schema_version: int = MOTION_CORRECTION_SCHEMA_VERSION
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

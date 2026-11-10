"""Validated parameter schema for the motion-interpolation parameter table.

A ``MotionInterpolationParameters`` row names every argument of SpikeInterface's
``interpolate_motion`` that shapes the corrected traces. Each field is required
because SI's signature defaults differ from its presets (``p=1`` vs ``p=2``).

No SpikeInterface import: the recipe catalog builds default rows at import.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MOTION_INTERPOLATION_SCHEMA_VERSION = 1


class MotionInterpolationParamsSchema(BaseModel):
    """Validated ``params`` for a ``MotionInterpolationParameters`` row.

    Attributes
    ----------
    border_mode : {"remove_channels", "force_extrapolate"}
        How channels whose corrected position leaves the probe are handled.
        ``remove_channels`` drops every channel that leaves the probe's depth
        range in any temporal bin; ``force_extrapolate`` keeps all channels
        and extrapolates the kernel. ``force_zeros`` is not allowed: the
        artifact mask and statistics spans cannot describe its zeroed bins.
    spatial_interpolation_method : {"kriging", "idw", "nearest"}
        The spatial interpolation kernel (SI
        ``get_spatial_interpolation_kernel``).
    sigma_um : float
        Kriging length scale (um).
    p : int
        Kriging exponent: the kernel is ``exp(-(distance / sigma_um) ** p)``.
    num_closest : int
        Contacts ``idw`` interpolates from.
    schema_version : int
        Bumped on breaking field changes; rows insert at the current version.
    """

    model_config = ConfigDict(extra="forbid")

    border_mode: Literal["remove_channels", "force_extrapolate"]
    spatial_interpolation_method: Literal["kriging", "idw", "nearest"]
    sigma_um: float = Field(gt=0, allow_inf_nan=False)
    p: int = Field(ge=1)
    num_closest: int = Field(ge=1)
    schema_version: Literal[MOTION_INTERPOLATION_SCHEMA_VERSION] = (
        MOTION_INTERPOLATION_SCHEMA_VERSION
    )

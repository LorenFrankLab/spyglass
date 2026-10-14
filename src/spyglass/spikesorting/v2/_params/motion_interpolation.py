"""Validated parameter schema for the motion-interpolation parameter table.

A ``MotionInterpolationParameters`` row names every argument of SpikeInterface's
``interpolate_motion`` that shapes the corrected traces. Each field is required:
the SpikeInterface signature defaults differ from what the motion presets use
(``p=1`` in ``interpolate_motion`` against ``p=2`` in every preset,
``sortingcomponents/motion/motion_interpolation.py:339-353``), so a row that
left one implicit would silently correct differently from its preset.

No SpikeInterface import: the default rows are built from this schema at import
time of the recipe catalog.
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
        and extrapolates the kernel. SpikeInterface's ``force_zeros`` is not
        allowed: it zeroes whole channels for some time bins, samples that
        the artifact mask and the statistics spans cannot describe.
    spatial_interpolation_method : {"kriging", "idw", "nearest"}
        The spatial interpolation kernel
        (``preprocessing/preprocessing_tools.py``,
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
    schema_version: int = MOTION_INTERPOLATION_SCHEMA_VERSION

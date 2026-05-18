"""Validated parameter schema for the preprocessing parameter table."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BandpassFilterParams(BaseModel):
    """Bandpass filter cutoffs applied before referencing."""

    model_config = ConfigDict(extra="forbid")
    freq_min: float = Field(default=300.0, ge=0.0, le=15000.0)
    freq_max: float = Field(default=6000.0, ge=0.0, le=15000.0)

    @model_validator(mode="after")
    def _check_band(self) -> "BandpassFilterParams":
        if self.freq_min >= self.freq_max:
            raise ValueError(
                f"freq_min ({self.freq_min}) must be below "
                f"freq_max ({self.freq_max})"
            )
        return self


class CommonReferenceParams(BaseModel):
    """Common-reference re-referencing options."""

    model_config = ConfigDict(extra="forbid")
    reference: Literal["global", "single", "local"] = "global"
    operator: Literal["median", "average"] = "median"


class WhitenParams(BaseModel):
    """Whitening options applied after motion correction."""

    model_config = ConfigDict(extra="forbid")
    dtype: str = "float32"


class PreprocessingParamsSchema(BaseModel):
    """Validated schema for the preprocessing parameter blob.

    Split into two stages so motion correction never runs on whitened
    data (SpikeInterface docs warn that whitening destroys the spatial
    amplitude structure motion estimators rely on):

    Stage 1 -- pre_motion (filter + reference): materialized to the
        ``Recording`` NWB-resident artifact (the ``ElectricalSeries``
        inside the ``AnalysisNwbfile``). This is what gets cached.
    Stage 2 -- post_motion (whitening): applied lazily AFTER motion
        correction by the single-recording or concatenated-recording
        sorting path.
    """

    model_config = ConfigDict(extra="forbid")
    schema_version: int = 1
    bandpass_filter: BandpassFilterParams = Field(
        default_factory=BandpassFilterParams
    )
    common_reference: CommonReferenceParams = Field(
        default_factory=CommonReferenceParams
    )
    whiten: WhitenParams | None = Field(default_factory=WhitenParams)
    # whiten=None for sorters that do their own whitening (Kilosort 4).

    def to_pre_motion_dict(self) -> dict:
        """Return the stage-1 dict (filter + reference). Cached."""
        return {
            "bandpass_filter": self.bandpass_filter.model_dump(),
            "common_reference": self.common_reference.model_dump(),
        }

    def to_post_motion_dict(self) -> dict:
        """Return the stage-2 dict; empty if whitening is disabled."""
        if self.whiten is None:
            return {}
        return {"whiten": self.whiten.model_dump()}

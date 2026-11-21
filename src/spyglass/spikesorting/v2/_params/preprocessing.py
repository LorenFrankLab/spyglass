"""Validated parameter schema for the preprocessing parameter table."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

PREPROCESSING_SCHEMA_VERSION = 4


class BandpassFilterParams(BaseModel):
    """Bandpass filter cutoffs, applied before referencing.

    Order matters on the global-median branch, where the median is
    non-linear. Defaults are the Frank-lab production bandpass.
    """

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


class PhaseShiftParams(BaseModel):
    """ADC sample-shift (phase-shift) correction options.

    Corrects per-channel offsets of multiplexed ADCs (e.g. Neuropixels).
    Applied first. A logged no-op when the recording has no
    ``inter_sample_shift`` property.
    """

    model_config = ConfigDict(extra="forbid")
    margin_ms: float = Field(default=100.0, ge=0.0)


class CommonReferenceParams(BaseModel):
    """Common-reference re-referencing options.

    The reference mode comes from ``SortGroupV2.reference_mode``, so there is
    no ``reference`` field. ``operator`` applies on the global-median branch;
    ``"average"`` selects common-average referencing.
    """

    model_config = ConfigDict(extra="forbid")
    operator: Literal["median", "average"] = "median"


class PreprocessingParamsSchema(BaseModel):
    """Validated schema for the preprocessing parameter blob.

    Steps run in order: phase shift, bandpass, bad-channel handling, common
    reference. Whitening is not a field: motion correction must not see
    whitened data, so the sorter and metric analyzer each whiten under their
    own tracked row.

    - ``phase_shift``: ``None`` is off.
    - ``bandpass_filter``: ``None`` skips filtering (the ``"no_filter"``
      preset).
    - ``min_segment_length``: drop sort-interval slivers shorter than this (s).
    - ``bad_channel_handling``: ``"remove"`` keeps curated
      ``Electrode.bad_channel`` flags excluded; ``"interpolate"`` re-includes
      interior flagged channels and fills them from good neighbours. Flags
      come from ``suggest_bad_channels`` or manual curation.
    """

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[PREPROCESSING_SCHEMA_VERSION] = (
        PREPROCESSING_SCHEMA_VERSION
    )
    phase_shift: PhaseShiftParams | None = Field(default=None)
    bandpass_filter: BandpassFilterParams | None = Field(
        default_factory=BandpassFilterParams
    )
    common_reference: CommonReferenceParams = Field(
        default_factory=CommonReferenceParams
    )
    min_segment_length: float = Field(default=1.0, ge=0.0)
    bad_channel_handling: Literal["remove", "interpolate"] = "remove"

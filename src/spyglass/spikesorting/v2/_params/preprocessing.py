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
    """Common-reference re-referencing options.

    v2 hardcodes the reference mode based on ``ref_channel_id`` in
    ``Recording._apply_pre_motion_preprocessing`` (single if a
    specific electrode is named, global if -2 is configured for
    global median, none if -1). The ``reference`` field of v1's
    preprocessing params is intentionally not exposed in v2 because
    no production v1 workflow used it -- v1 hardcoded the same
    dispatch (see ``src/spyglass/spikesorting/v1/recording.py:597-619``).
    Promoted from "silent runtime override" to "field removed" so
    the schema does not lie about what the runtime honors.

    ``operator`` IS used on the global-median branch and stays.
    """

    model_config = ConfigDict(extra="forbid")
    operator: Literal["median", "average"] = "median"


class WhitenParams(BaseModel):
    """Whitening options applied after motion correction.

    ``whiten`` is dead in Phase 1 (whitening is deferred to the sorter
    via ``Sorting._run_sorter``'s external float64 whitening path);
    the field exists as forward-compat scaffolding for Phase 3's
    ``ConcatenatedRecording.make`` motion-correction +
    post-motion-whitening flow. See
    ``PreprocessingParamsSchema.to_post_motion_dict``.
    """

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

    ``schema_version`` was bumped to 2 to mark the schema-
    incompatible edits:
    * added ``min_segment_length`` (drops sub-second slivers from
      the intersected sort interval before the sorter sees them).
    * removed ``CommonReferenceParams.reference`` (dead field).
    """

    model_config = ConfigDict(extra="forbid")
    schema_version: int = 2
    bandpass_filter: BandpassFilterParams = Field(
        default_factory=BandpassFilterParams
    )
    common_reference: CommonReferenceParams = Field(
        default_factory=CommonReferenceParams
    )
    whiten: WhitenParams | None = Field(default_factory=WhitenParams)
    # whiten=None for sorters that do their own whitening (Kilosort 4).
    min_segment_length: float = Field(default=1.0, ge=0.0)
    # Drop disjoint-interval slivers shorter than this many seconds
    # before the sorter sees them. Matches v1's default at
    # ``src/spyglass/spikesorting/v1/recording.py:135``; passed through
    # to ``sort_interval.intersect(..., min_length=...)``.

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

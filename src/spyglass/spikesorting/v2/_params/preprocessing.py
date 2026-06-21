"""Validated parameter schema for the preprocessing parameter table."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

PREPROCESSING_SCHEMA_VERSION = 3


class BandpassFilterParams(BaseModel):
    """Bandpass filter cutoffs applied BEFORE referencing.

    The runtime bandpass-filters first, then references -- the
    signal-processing-preferred order. The order is NOT commutative on the
    global-median common-reference branch (the per-sample median is
    non-linear), so reference-then-filter would differ numerically there; on
    the ``specific`` / ``none`` paths the steps are linear and commute, so
    output is identical either way.

    Defaults ``freq_min=300.0``, ``freq_max=6000.0`` are the Frank-lab
    production bandpass, exposed as schema-level defaults so a user
    constructing ``PreprocessingParamsSchema()`` without arguments gets the
    production preset implicitly.
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

    Compensates the per-channel time offsets introduced by multiplexed
    ADCs (e.g. Neuropixels). Applied FIRST, before bandpass filtering. Only
    meaningful when the recording carries an ``inter_sample_shift``
    property; the runtime skips it (with a log) otherwise, so enabling it
    on non-multiplexed acquisition is a safe no-op. Off by default
    (``PreprocessingParamsSchema.phase_shift = None``).
    """

    model_config = ConfigDict(extra="forbid")
    margin_ms: float = Field(default=100.0, ge=0.0)


class CommonReferenceParams(BaseModel):
    """Common-reference re-referencing options.

    The reference mode is selected from the ``SortGroupV2.reference_mode``
    column in ``Recording._apply_pre_motion_preprocessing`` (single for
    ``"specific"`` -- subtract the named ``reference_electrode_id`` --
    global for ``"global_median"``, none for ``"none"``). A free-standing
    ``reference`` field is intentionally not exposed because that dispatch
    fully determines the reference, so a params-blob field would only be a
    silent runtime override; removing it keeps the schema honest about what
    the runtime honors.

    ``operator`` IS used on the global-median branch and stays, exposed as a
    user knob. The default ``"median"`` is the production behavior; passing
    ``"average"`` selects mean common-average referencing instead.
    """

    model_config = ConfigDict(extra="forbid")
    operator: Literal["median", "average"] = "median"


class WhitenParams(BaseModel):
    """Whitening options applied after motion correction.

    ``whiten`` is currently inert (whitening is deferred to the
    sorter via ``Sorting._run_sorter``'s external float64 whitening
    path); the field exists as forward-compat scaffolding for the
    eventual ``ConcatenatedRecording.make`` motion-correction +
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

    ``schema_version`` history:
    * 2 added ``min_segment_length`` (drops sub-second slivers from
      the intersected sort interval before the sorter sees them) and
      removed ``CommonReferenceParams.reference`` (dead field).
    * 3 made ``bandpass_filter`` optional (``None`` = skip filtering,
      so the ``"no_filter"`` preset is a real disable instead of a
      wide-band pass) and flipped the ``whiten`` default to ``None``
      to match the runtime (whitening is deferred to the sorter, so
      the schema must not default to claiming it is configured). The
      runtime preprocessing ORDER also changed at 3, from
      reference->filter to **bandpass filter->reference** (the
      signal-processing-preferred order; see
      ``apply_pre_motion_preprocessing``). The params blob shape is
      unchanged, so ``schema_version`` is NOT bumped -- only the runtime
      interpretation moved; dev rows are regenerated, not migrated.
    * 3 also added the optional ``phase_shift`` sub-model (ADC sample-shift
      correction for multiplexed acquisition; off by default, ``None``).
      The blob shape only grows by an optional field that defaults to
      ``None``, so existing rows validate unchanged and ``schema_version``
      is again NOT bumped; dev rows are regenerated.
    * 3 also added ``bad_channel_handling`` (``"remove"`` | ``"interpolate"``,
      default ``"remove"``) controlling how curated ``Electrode.bad_channel``
      flags are handled at materialization. ``"remove"`` is byte-identical to
      pre-field behavior (the flagged channels were already excluded at sort-
      group creation), so existing rows validate unchanged and output is
      identical -- ``schema_version`` is again NOT bumped; dev rows are
      regenerated. Detection of bad channels is a separate concern handled by
      ``suggest_bad_channels`` (it writes the flags this field consumes), not a
      preprocessing parameter.
    """

    model_config = ConfigDict(extra="forbid")
    schema_version: int = PREPROCESSING_SCHEMA_VERSION
    phase_shift: PhaseShiftParams | None = Field(default=None)
    # phase_shift defaults to None: off in the franklab default and only a
    # no-op-until-present step for multiplexed-ADC (Neuropixels) recordings.
    bandpass_filter: BandpassFilterParams | None = Field(
        default_factory=BandpassFilterParams
    )
    # bandpass_filter=None disables filtering entirely (the "no_filter"
    # preset); the default ships the production bandpass.
    common_reference: CommonReferenceParams = Field(
        default_factory=CommonReferenceParams
    )
    whiten: WhitenParams | None = Field(default=None)
    # whiten defaults to None: whitening is deferred to the sorter
    # (Kilosort 4 does its own; the external float64 whitening path
    # handles the rest), so the schema does not claim it is on.
    min_segment_length: float = Field(default=1.0, ge=0.0)
    # Drop disjoint-interval slivers shorter than this many seconds
    # before the sorter sees them; passed through to
    # ``sort_interval.intersect(..., min_length=...)``.
    bad_channel_handling: Literal["remove", "interpolate"] = "remove"
    # How curated ``Electrode.bad_channel='True'`` flags are handled at
    # materialization. ``"remove"`` (default) is byte-identical to today: the
    # flagged channels were already excluded at sort-group creation and stay
    # out. ``"interpolate"`` re-includes the group's pitch-adjacent interior
    # flagged channels and fills them from good neighbours so geometry-aware
    # sorters see a complete probe. Detection is NOT done here -- the flags come
    # from ``suggest_bad_channels`` or manual curation.

    def to_pre_motion_dict(self) -> dict:
        """Return the stage-1 dict (phase-shift + filter + reference). Cached.

        ``phase_shift`` / ``bandpass_filter`` are ``None`` when the
        corresponding step is disabled; the runtime skips it in that case.
        ``bad_channel_handling`` is included because the handling step runs in
        stage 1 (between filter and reference).
        """
        return {
            "phase_shift": (
                None
                if self.phase_shift is None
                else self.phase_shift.model_dump()
            ),
            "bandpass_filter": (
                None
                if self.bandpass_filter is None
                else self.bandpass_filter.model_dump()
            ),
            "common_reference": self.common_reference.model_dump(),
            "bad_channel_handling": self.bad_channel_handling,
        }

    def to_post_motion_dict(self) -> dict:
        """Return the stage-2 dict; empty if whitening is disabled."""
        if self.whiten is None:
            return {}
        return {"whiten": self.whiten.model_dump()}

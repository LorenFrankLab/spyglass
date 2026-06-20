"""Validated parameter schema for the artifact-detection parameter table."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

ARTIFACT_DETECTION_SCHEMA_VERSION = 2


class ArtifactDetectionParamsSchema(BaseModel):
    """Validated schema for the artifact-detection parameter blob.

    The recording stage materializes the preprocessed ``ElectricalSeries``
    once; this stage scans that recording for stretches where the signal
    exceeds an amplitude or z-score threshold on a sufficient fraction of
    channels, then writes the artifact-removed valid times into
    ``common.IntervalList`` for the sorter to read.

    Fields replace v1's threshold names (``zscore_thresh``,
    ``amplitude_thresh_uV``, ``proportion_above_thresh``) with explicit v2
    names (``zscore_threshold``, ``amplitude_threshold_uv``,
    ``proportion_above_threshold``), while preserving the same default
    semantics. v2 also adds ``detect`` (so the ``"none"`` preset is a typed
    ``detect=False`` rather than an absent-fields blob) and ``join_window_ms``
    (artifact intervals closer than this are merged into one).

    Detector note: ``zscore_threshold`` is a per-frame *cross-channel*
    z-score (across channels within a frame), so it flags single-channel
    outliers but is BLIND to pure common-mode events -- when every
    channel jumps together the shift cancels in the across-channel mean
    and the z-score stays ~0. Use ``amplitude_threshold_uv`` to catch
    common-mode (e.g. EMG) artifacts; the two thresholds OR together
    when both are set (a frame is flagged if it exceeds the amplitude
    threshold OR the z-score threshold). This is an intentional mode,
    not a configuration error -- setting both is supported.

    ``detect=False`` skips detection entirely and ignores both
    thresholds, so leaving stale threshold values on a ``detect=False``
    row is harmless. When ``detect=True`` at least one threshold is
    required (enforced by ``_check_thresholds``).

    Concurrency parameters (``n_jobs``, ``chunk_duration``, ``progress_bar``)
    are not part of this schema. They live in the per-row ``job_kwargs``
    blob and are resolved at populate time by ``_resolved_job_kwargs``.
    """

    model_config = ConfigDict(extra="forbid")
    schema_version: int = ARTIFACT_DETECTION_SCHEMA_VERSION
    # Bumped to 2 by adding ``min_length_s``: the artifact-removed
    # valid_times are filtered to drop slivers shorter than this
    # many seconds before the sorter sees them. Default ``1.0``
    # matches v1's hardcoded value at
    # ``src/spyglass/spikesorting/v1/artifact.py:327-328``.
    #
    # ``amplitude_threshold_uv=500.0`` is v2's bug-fix default (matches
    # v1's effective Intan-probe behavior within ~15%) and
    # ``proportion_above_threshold`` is ``1.0`` (v1's principled "all
    # channels must exceed" default; a 0.5 default had no documented
    # justification). The CHANGELOG entry explains the v1 unit-
    # conversion bug that motivated keeping 500 uV; users with
    # custom v1 thresholds should translate
    # ``v1_value * probe_gain_uV_per_count`` to the v2-equivalent uV
    # value.
    detect: bool = True
    amplitude_threshold_uv: float | None = Field(default=500.0, ge=0.0)
    zscore_threshold: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "Per-frame cross-channel z-score threshold (z is computed "
            "across channels within each frame, not per channel over "
            "time). It flags single-channel outliers within a frame; "
            "pure common-mode events (every channel jumps together) "
            "produce a ~0 z-score and are NOT detected -- use "
            "amplitude_threshold_uv to catch common-mode (e.g. EMG) "
            "artifacts."
        ),
    )
    proportion_above_threshold: float = Field(default=1.0, gt=0.0, le=1.0)
    removal_window_ms: float = Field(default=1.0, gt=0.0)
    join_window_ms: float = Field(default=1.0, ge=0.0)
    min_length_s: float = Field(default=1.0, gt=0.0)

    @model_validator(mode="after")
    def _check_thresholds(self) -> "ArtifactDetectionParamsSchema":
        if (
            self.detect
            and self.amplitude_threshold_uv is None
            and self.zscore_threshold is None
        ):
            raise ValueError(
                "ArtifactDetectionParamsSchema requires at least one of "
                "amplitude_threshold_uv or zscore_threshold when detect=True"
            )
        return self

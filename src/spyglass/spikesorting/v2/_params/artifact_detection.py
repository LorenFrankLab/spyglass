"""Validated parameter schema for the artifact-detection parameter table."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ARTIFACT_DETECTION_SCHEMA_VERSION = 2


class ArtifactDetectionParamsSchema(BaseModel):
    """Validated schema for the artifact-detection parameter blob.

    Flags stretches where the preprocessed signal exceeds an amplitude or
    z-score threshold on enough channels; the artifact-removed valid times go
    to ``common.IntervalList`` for the sorter.

    - ``zscore_threshold`` is computed across channels within a frame, so it
      misses common-mode events. Use ``amplitude_threshold_uv`` for those.
    - When both thresholds are set they OR together.
    - ``detect=False`` skips detection and ignores both thresholds;
      ``detect=True`` requires at least one.
    - ``amplitude_threshold_uv`` is in microvolts, not raw counts.
    - ``min_length_s`` drops valid-time slivers shorter than this.
    - Concurrency settings live in the row's ``job_kwargs`` blob.
    """

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[ARTIFACT_DETECTION_SCHEMA_VERSION] = (
        ARTIFACT_DETECTION_SCHEMA_VERSION
    )
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

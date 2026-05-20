"""Validated parameter schema for the artifact-detection parameter table."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ArtifactDetectionParamsSchema(BaseModel):
    """Validated schema for the artifact-detection parameter blob.

    The recording stage materializes the preprocessed ``ElectricalSeries``
    once; this stage scans that recording for stretches where the signal
    exceeds an amplitude or z-score threshold on a sufficient fraction of
    channels, then writes the artifact-removed valid times into
    ``common.IntervalList`` for the sorter to read.

    Fields mirror v1's defaults (``zscore_thresh``, ``amplitude_thresh_uV``,
    ``proportion_above_thresh``, ``removal_window_ms``) plus two additions:
    ``detect`` (so the ``"none"`` preset is a typed ``detect=False`` rather
    than an absent-fields blob) and ``join_window_ms`` (artifact intervals
    closer than this are merged into one).

    Concurrency parameters (``n_jobs``, ``chunk_duration``, ``progress_bar``)
    are not part of this schema. They live in the per-row ``job_kwargs``
    blob and are resolved at populate time by ``_resolved_job_kwargs``.
    """

    model_config = ConfigDict(extra="forbid")
    schema_version: int = 1
    detect: bool = True
    amplitude_thresh_uV: float | None = Field(default=500.0, ge=0.0)
    zscore_thresh: float | None = Field(default=None, ge=0.0)
    proportion_above_thresh: float = Field(default=0.5, gt=0.0, le=1.0)
    removal_window_ms: float = Field(default=1.0, gt=0.0)
    join_window_ms: float = Field(default=1.0, ge=0.0)

    @model_validator(mode="after")
    def _check_thresholds(self) -> "ArtifactDetectionParamsSchema":
        if self.detect and self.amplitude_thresh_uV is None and self.zscore_thresh is None:
            raise ValueError(
                "ArtifactDetectionParamsSchema requires at least one of "
                "amplitude_thresh_uV or zscore_thresh when detect=True"
            )
        return self

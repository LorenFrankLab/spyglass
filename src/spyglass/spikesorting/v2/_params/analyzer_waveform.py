"""Validated parameter schema for the tracked analyzer-waveform table.

``AnalyzerWaveformParameters`` records the window (``ms_before`` / ``ms_after``),
subsample (``max_spikes_per_unit``), and whitening of the ``SortingAnalyzer``
that produced a sort's templates / waveforms, so those settings are tracked in
the database rather than hardcoded in the analyzer build (mirroring the v1
``WaveformParameters`` table v2 had regressed from).
``AnalyzerWaveformParamsSchema`` validates that blob.

``return_in_uV`` is intentionally NOT a field: it is derived from ``whiten`` at
build time (an unwhitened display recipe returns real microvolts; a whitened
metric recipe does not, because the whitening preserves channel gains and a
microvolt readback would un-normalize the whitened space). It is part of the
recipe, not a user-tunable knob.

Concurrency parameters (``n_jobs``, ``chunk_duration``, ``progress_bar``) do NOT
live on this schema; the analyzer build resolves its job kwargs from the sort's
``SorterParameters`` row, per the shared Job-Kwargs Resolution convention.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ANALYZER_WAVEFORM_SCHEMA_VERSION = 1


class AnalyzerWaveformParamsSchema(BaseModel):
    """Window / subsample / whitening for one analyzer waveform recipe.

    A display recipe (``purpose="display"``) is always unwhitened and a metric
    recipe (``purpose="metric"``) is always whitened; the two are validated as a
    pair so a row cannot claim one purpose with the other's whitening. The
    schema defaults (1.0 / 2.0 ms window, 20000 spikes, unwhitened display) are
    the wide cortex fallback used for any custom row and for unknown /
    multi-region sorts.
    """

    model_config = ConfigDict(extra="forbid")
    schema_version: int = ANALYZER_WAVEFORM_SCHEMA_VERSION
    ms_before: float = Field(default=1.0, gt=0.0)
    ms_after: float = Field(default=2.0, gt=0.0)
    max_spikes_per_unit: int = Field(default=20000, ge=1)
    whiten: bool = False
    purpose: Literal["display", "metric"] = "display"

    @model_validator(mode="after")
    def _purpose_matches_whiten(self):
        """Bind ``purpose`` and ``whiten`` so a recipe cannot lie about either.

        Display rows feed real-microvolt amplitudes / waveform shapes and so
        must be unwhitened; metric rows feed PC / cluster-separation metrics in
        the decorrelated whitened space and so must be whitened.
        """
        if self.purpose == "display" and self.whiten:
            raise ValueError("display waveform rows must be unwhitened")
        if self.purpose == "metric" and not self.whiten:
            raise ValueError("metric waveform rows must be whitened")
        return self

"""Validated parameter schema for the tracked analyzer-waveform table.

``AnalyzerWaveformParameters`` records the window (``ms_before`` / ``ms_after``),
subsample (``max_spikes_per_unit``), whitening, and channel sparsity of the
``SortingAnalyzer`` that produced a sort's templates / waveforms, so those
settings are tracked in the database rather than hardcoded in the analyzer
build (mirroring the v1 ``WaveformParameters`` table v2 had regressed from).
``AnalyzerWaveformParamsSchema`` validates that blob.

Sparsity (``SparsityParams``) selects which channels each unit's waveforms are
extracted on: ``"radius"`` (channels within ``radius_um`` of the unit's peak
channel, SpikeInterface's default and the shipped recipes' setting),
``"best_channels"`` (the ``num_channels`` largest-amplitude channels -- the
bounded choice for dense probes), or ``"dense"`` (every channel; only sensible
for tetrodes / small groups). The SI ``estimate_sparsity`` sampling settings
(``num_spikes_for_sparsity``, its own ``ms_before`` / ``ms_after`` snippet
window, ``peak_sign``) are recorded explicitly so the effective configuration
-- including SI defaults -- is in the row, not implied by the SI version.

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

ANALYZER_WAVEFORM_SCHEMA_VERSION = 2


class SparsityParams(BaseModel):
    """Channel-sparsity configuration for the analyzer (SI ``estimate_sparsity``).

    ``method="dense"`` builds a dense analyzer (``sparse=False``); the other
    fields are then inert and must be left at their defaults. ``"radius"``
    uses ``radius_um`` (``None`` resolves to SI's 100 um and is recorded as
    such) and ignores ``num_channels``; ``"best_channels"`` requires
    ``num_channels`` and ignores ``radius_um``. The inert one must be ``None``
    so a row cannot record a value the build does not use. Defaults
    equal SpikeInterface 0.104.3's ``estimate_sparsity`` defaults
    (``radius`` / 100 um / ``"neg"`` / 100 spikes / 1.0-2.5 ms snippets), so a
    recipe that omits ``sparsity`` builds exactly what it built before this
    field existed.
    """

    model_config = ConfigDict(extra="forbid")
    method: Literal["dense", "radius", "best_channels"] = "radius"
    radius_um: float | None = Field(default=None, gt=0.0)
    num_channels: int | None = Field(default=None, ge=1)
    peak_sign: Literal["neg", "pos", "both"] = "neg"
    num_spikes_for_sparsity: int = Field(default=100, ge=1)
    ms_before: float = Field(default=1.0, gt=0.0)
    ms_after: float = Field(default=2.5, gt=0.0)

    @model_validator(mode="after")
    def _method_fields(self):
        """Require the method's own field; forbid the other method's field."""
        if self.method == "radius":
            if self.radius_um is None:
                self.radius_um = 100.0  # SI estimate_sparsity default, recorded
            if self.num_channels is not None:
                raise ValueError(
                    "sparsity method 'radius' does not use num_channels; "
                    "leave it None"
                )
        elif self.method == "best_channels":
            if self.num_channels is None:
                raise ValueError(
                    "sparsity method 'best_channels' requires num_channels"
                )
            if self.radius_um is not None:
                raise ValueError(
                    "sparsity method 'best_channels' does not use radius_um; "
                    "leave it None"
                )
        else:  # dense
            if self.radius_um is not None or self.num_channels is not None:
                raise ValueError(
                    "sparsity method 'dense' uses every channel; leave "
                    "radius_um and num_channels None"
                )
        return self

    def si_create_kwargs(self) -> dict:
        """Return the ``create_sorting_analyzer`` sparsity kwargs this recipe means."""
        if self.method == "dense":
            return {"sparse": False}
        kwargs = {
            "sparse": True,
            "method": self.method,
            "peak_sign": self.peak_sign,
            "num_spikes_for_sparsity": int(self.num_spikes_for_sparsity),
            "ms_before": float(self.ms_before),
            "ms_after": float(self.ms_after),
        }
        if self.method == "radius":
            kwargs["radius_um"] = float(self.radius_um)
        else:
            kwargs["num_channels"] = int(self.num_channels)
        return kwargs


class AnalyzerWaveformParamsSchema(BaseModel):
    """Window / subsample / whitening / sparsity for one analyzer waveform recipe.

    A display recipe (``purpose="display"``) is always unwhitened and a metric
    recipe (``purpose="metric"``) is always whitened; the two are validated as a
    pair so a row cannot claim one purpose with the other's whitening. The
    schema defaults (1.0 / 2.0 ms window, 20000 spikes, unwhitened display) are
    the wide cortex fallback used for any custom row and for unknown /
    multi-region sorts. ``schema_version`` 2 added ``sparsity`` (default:
    SI's radius / 100 um, i.e. the previous implicit behavior).
    """

    model_config = ConfigDict(extra="forbid")
    schema_version: int = ANALYZER_WAVEFORM_SCHEMA_VERSION
    ms_before: float = Field(default=1.0, gt=0.0)
    ms_after: float = Field(default=2.0, gt=0.0)
    max_spikes_per_unit: int = Field(default=20000, ge=1)
    whiten: bool = False
    purpose: Literal["display", "metric"] = "display"
    sparsity: SparsityParams = Field(default_factory=SparsityParams)

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

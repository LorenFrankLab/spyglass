"""Validated parameter schema for the tracked analyzer-waveform table.

``AnalyzerWaveformParameters`` records the window, spike subsample, whitening,
and channel sparsity of the ``SortingAnalyzer`` that produced a sort's
templates / waveforms. ``AnalyzerWaveformParamsSchema`` validates that blob.

Sparsity methods:

- ``"radius"``: channels within ``radius_um`` of the peak channel (SI default).
- ``"best_channels"``: the ``num_channels`` largest-amplitude channels; bounded
  for dense probes.
- ``"dense"``: every channel; only sensible for tetrodes / small groups.

Not fields:

- ``peak_sign``: derived from the tracked ``SorterParameters.params`` blob.
- ``return_in_uV``: derived from ``whiten``. A microvolt readback would
  un-normalize the whitened space.
- Concurrency (``n_jobs``, ``chunk_duration``, ``progress_bar``): resolved by
  ``_resolved_job_kwargs``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ANALYZER_WAVEFORM_SCHEMA_VERSION = 3


class SparsityParams(BaseModel):
    """Channel-sparsity configuration for the analyzer (SI ``estimate_sparsity``).

    - ``"dense"``: ``sparse=False``; the other fields keep their defaults.
    - ``"radius"``: uses ``radius_um`` (``None`` records SI's 100 um).
    - ``"best_channels"``: requires ``num_channels``.

    The unused method field must be ``None`` so a row cannot record a value
    the build ignores. Defaults equal SpikeInterface 0.104.3's.
    """

    model_config = ConfigDict(extra="forbid")
    method: Literal["dense", "radius", "best_channels"] = "radius"
    radius_um: float | None = Field(default=None, gt=0.0)
    num_channels: int | None = Field(default=None, ge=1)
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
        """Return recipe kwargs; the caller adds the sorter's ``peak_sign``."""
        if self.method == "dense":
            return {"sparse": False}
        kwargs = {
            "sparse": True,
            "method": self.method,
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

    Display recipes are unwhitened; metric recipes are whitened. Defaults are
    the wide cortex fallback for custom rows and unknown / multi-region sorts.
    """

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[ANALYZER_WAVEFORM_SCHEMA_VERSION] = (
        ANALYZER_WAVEFORM_SCHEMA_VERSION
    )
    ms_before: float = Field(default=1.0, gt=0.0)
    ms_after: float = Field(default=2.0, gt=0.0)
    max_spikes_per_unit: int = Field(default=20000, ge=1)
    whiten: bool = False
    purpose: Literal["display", "metric"] = "display"
    sparsity: SparsityParams = Field(default_factory=SparsityParams)

    @model_validator(mode="after")
    def _purpose_matches_whiten(self):
        """Require display rows unwhitened (microvolts) and metric rows whitened."""
        if self.purpose == "display" and self.whiten:
            raise ValueError("display waveform rows must be unwhitened")
        if self.purpose == "metric" and not self.whiten:
            raise ValueError("metric waveform rows must be whitened")
        return self

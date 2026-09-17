"""Dependency-light normalization for persisted curation review profiles.

The DataJoint table lives in :mod:`review_profile`; this module owns the
canonical semantic payload and hash without importing DataJoint or the optional
FigPack runtime.  It deliberately reuses the existing FigPack configuration
helpers so profile and figure configuration cannot drift on list semantics.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from spyglass.spikesorting.v2._enums import CurationLabel
from spyglass.spikesorting.v2._figpack_curation import (
    default_label_options,
    normalize_displayed_unit_properties,
)

# Pinned SpikeInterface 0.104 quality-metric output columns. Spyglass replaces
# the public ``isi_violation`` value with its bounded fraction while retaining
# SI's two diagnostic columns in the computed frame. Keeping the vocabulary
# here lets a profile validate concrete display columns without importing SI or
# constructing an analyzer at profile-insert time.
_QUALITY_METRIC_OUTPUT_COLUMNS = {
    "num_spikes": ("num_spikes",),
    "firing_rate": ("firing_rate",),
    "presence_ratio": ("presence_ratio",),
    "snr": ("snr",),
    "isi_violation": (
        "isi_violation",
        "isi_violations_ratio",
        "isi_violations_count",
    ),
    "rp_violation": ("rp_contamination", "rp_violations"),
    "sliding_rp_violation": ("sliding_rp_violation",),
    "synchrony": ("sync_spike_2", "sync_spike_4", "sync_spike_8"),
    "firing_range": ("firing_range",),
    "amplitude_cv": ("amplitude_cv_median", "amplitude_cv_range"),
    "amplitude_cutoff": ("amplitude_cutoff",),
    "noise_cutoff": ("noise_cutoff", "noise_ratio"),
    "amplitude_median": ("amplitude_median",),
    "drift": ("drift_ptp", "drift_std", "drift_mad"),
    "sd_ratio": ("sd_ratio",),
    "mahalanobis": ("isolation_distance", "l_ratio"),
    "d_prime": ("d_prime",),
    "nearest_neighbor": ("nn_hit_rate", "nn_miss_rate"),
    "silhouette": ("silhouette",),
    "nn_advanced": ("nn_isolation", "nn_noise_overlap"),
}

_PROFILE_FIELDS = frozenset(
    {
        "review_profile_name",
        "metric_params_name",
        "auto_curation_rules_name",
        "displayed_unit_properties",
        "label_options",
        "label_import_mode",
        "profile_hash",
    }
)


def profile_display_property_vocabulary(
    metric_row: Mapping,
) -> tuple[str, ...]:
    """Return the ordered built-in columns one metric recipe can display."""
    columns: list[str] = [
        "observed_duration_s",
        "observed_firing_rate_hz",
        "observed_presence_ratio",
    ]
    for metric_name in metric_row["metric_names"]:
        columns.extend(
            _QUALITY_METRIC_OUTPUT_COLUMNS.get(metric_name, (metric_name,))
        )
    columns.extend(metric_row.get("template_metric_columns") or [])
    # Preserve recipe order while protecting against a future overlap between
    # metric and template output names.
    return tuple(dict.fromkeys(str(column) for column in columns))


def _normalize_label_options(label_options) -> list[str]:
    """Normalize built-in aliases and configured lab labels."""
    if label_options is None:
        label_options = default_label_options()
    if isinstance(label_options, (str, bytes)) or not isinstance(
        label_options, (list, tuple)
    ):
        raise TypeError("label_options must be a list or tuple of strings.")
    normalized: list[str] = []
    for label in label_options:
        value = CurationLabel.normalize(label)
        if not isinstance(label, str) or not value.strip() or len(value) > 32:
            raise ValueError(
                "Each label must be a non-empty string of at most 32 characters."
            )
        if value != value.strip():
            raise ValueError(
                "Labels must not contain leading or trailing whitespace."
            )
        normalized.append(value)
    duplicates = sorted(
        {label for label in normalized if normalized.count(label) > 1}
    )
    if duplicates:
        raise ValueError(
            f"label_options contains duplicate entries: {duplicates}."
        )
    if not normalized:
        raise ValueError("label_options must contain at least one label.")
    return normalized


def normalize_review_profile(
    row: Mapping,
    *,
    available_displayed_properties: Iterable[str],
) -> dict:
    """Validate one profile input and return its canonical stored row.

    Delivery and review-instance values are rejected as unknown fields.  Only
    the immutable scientific/display configuration participates in the hash.
    """
    if not isinstance(row, Mapping):
        raise TypeError(
            "CurationReviewProfile rows must be mappings, not "
            f"{type(row).__name__}."
        )
    unknown = sorted(set(row) - _PROFILE_FIELDS)
    if unknown:
        raise ValueError(
            "CurationReviewProfile does not accept runtime/review fields "
            f"{unknown}. Publishing location, upload/ephemeral mode, "
            "credentials, and annotation-set selections belong to each "
            "review invocation."
        )
    required = (
        "review_profile_name",
        "metric_params_name",
        "auto_curation_rules_name",
        "displayed_unit_properties",
        "label_import_mode",
    )
    missing = [field for field in required if field not in row]
    if missing:
        raise ValueError(
            f"CurationReviewProfile is missing required field(s) {missing}."
        )

    name = str(row["review_profile_name"])
    metric_name = str(row["metric_params_name"])
    rules_name = str(row["auto_curation_rules_name"])
    if not name or len(name) > 64:
        raise ValueError(
            "review_profile_name must be a non-empty string of at most 64 "
            "characters."
        )
    for field, value in (
        ("metric_params_name", metric_name),
        ("auto_curation_rules_name", rules_name),
    ):
        if not value or len(value) > 64:
            raise ValueError(
                f"{field} must be a non-empty string of at most 64 characters."
            )

    displayed = normalize_displayed_unit_properties(
        row["displayed_unit_properties"]
    )
    # Persist an explicit ordered list. ``None`` is useful for a one-off
    # FigPack selection, but a named review profile must state its display
    # contract so its hash cannot inherit changing backend defaults.
    if displayed is None:
        raise ValueError(
            "CurationReviewProfile displayed_unit_properties must be an "
            "explicit list; None would inherit changing backend defaults."
        )
    available = tuple(str(prop) for prop in available_displayed_properties)
    unavailable = [prop for prop in displayed if prop not in available]
    if unavailable:
        raise ValueError(
            "CurationReviewProfile displayed_unit_properties contains "
            f"columns {unavailable} not produced by metric recipe "
            f"{metric_name!r}; available built-in columns are {list(available)}."
        )

    label_options = _normalize_label_options(row.get("label_options"))
    label_import_mode = str(row["label_import_mode"])
    if label_import_mode not in {"replace", "overlay"}:
        raise ValueError(
            "label_import_mode must be 'replace' or 'overlay'; "
            f"got {label_import_mode!r}."
        )

    semantic = {
        "metric_params_name": metric_name,
        "auto_curation_rules_name": rules_name,
        "displayed_unit_properties": displayed,
        "label_options": label_options,
        "label_import_mode": label_import_mode,
    }
    profile_hash = hashlib.sha256(
        json.dumps(
            semantic,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    supplied_hash = row.get("profile_hash")
    if supplied_hash is not None and str(supplied_hash) != profile_hash:
        raise ValueError(
            "CurationReviewProfile profile_hash does not match its normalized "
            f"semantic content (expected {profile_hash})."
        )
    return {
        "review_profile_name": name,
        **semantic,
        "profile_hash": profile_hash,
    }


def review_profile_label_policy(label_import_mode: str) -> str:
    """Map the profile's public import mode to CurationV2's expert policy."""
    if label_import_mode == "replace":
        return "replace"
    if label_import_mode == "overlay":
        return "inherit"
    raise ValueError(
        "label_import_mode must be 'replace' or 'overlay'; "
        f"got {label_import_mode!r}."
    )


REVIEW_DISPLAY_OPTIONS_VERSION = 2


@dataclass(frozen=True)
class ReviewDisplayOptions:
    """Browser payload budget for one FigPack review (display-only).

    These settings bound what the bundle CONTAINS; they never touch the
    scientific evaluation (metrics, waveform subsample, merge suggestions all
    come from the persisted ``CurationEvaluation`` / analyzer recipes).

    Attributes
    ----------
    max_raster_spikes_per_unit
        Per-unit raster cap, sampled evenly across the full spike train.
    max_amplitudes_per_unit
        Per-unit cap on spike-amplitude points in the amplitude view. Points
        are drawn uniformly at random from the unit's whole spike train (so a
        long recording is sampled across its full duration, not truncated to
        its start), seeded by ``amplitude_sampling_seed`` so a rebuilt bundle
        shows the same points. ``None`` removes the fixed cap; the duration
        budget still applies unless its rate is also ``None``.
    max_initial_points
        Total points per time-based view before deferring that view to explicit
        selected-unit inspection. This controls initial payload, not sampling.
    amplitude_sampling_seed
        Seed for the amplitude subsample.
    min_similarity_for_correlograms
        SpikeInterface's cross-correlogram pair filter: only unit pairs whose
        template similarity reaches this value get a cross-correlogram, which
        bounds the pair-heavy part of the bundle.
    version
        Payload version, persisted with the review configuration.
    """

    max_raster_spikes_per_unit: int | None = None
    max_amplitudes_per_unit: int | None = None
    raster_max_firing_rate: float | None = 50.0
    amplitude_max_firing_rate: float | None = 50.0
    max_initial_points: int = 1_000_000
    amplitude_sampling_seed: int = 0
    min_similarity_for_correlograms: float = 0.2
    version: int = REVIEW_DISPLAY_OPTIONS_VERSION

    def __post_init__(self):
        if (
            isinstance(self.max_initial_points, bool)
            or not isinstance(self.max_initial_points, int)
            or self.max_initial_points < 1
        ):
            raise ValueError("max_initial_points must be a positive integer.")
        if self.max_raster_spikes_per_unit is not None and (
            isinstance(self.max_raster_spikes_per_unit, bool)
            or not isinstance(self.max_raster_spikes_per_unit, int)
            or self.max_raster_spikes_per_unit < 1
        ):
            raise ValueError(
                "max_raster_spikes_per_unit must be a positive integer."
            )
        if self.max_amplitudes_per_unit is not None and (
            isinstance(self.max_amplitudes_per_unit, bool)
            or int(self.max_amplitudes_per_unit) < 1
        ):
            raise ValueError(
                "max_amplitudes_per_unit must be a positive int or None; got "
                f"{self.max_amplitudes_per_unit!r}."
            )
        if isinstance(self.amplitude_sampling_seed, bool) or (
            int(self.amplitude_sampling_seed) < 0
        ):
            raise ValueError(
                "amplitude_sampling_seed must be a non-negative int; got "
                f"{self.amplitude_sampling_seed!r}."
            )
        if not 0.0 <= float(self.min_similarity_for_correlograms) <= 1.0:
            raise ValueError(
                "min_similarity_for_correlograms must be within [0, 1]; got "
                f"{self.min_similarity_for_correlograms!r}."
            )
        import math

        for value in (
            self.raster_max_firing_rate,
            self.amplitude_max_firing_rate,
        ):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(
                    "Display firing-rate limits must be finite and positive, or None."
                )
        if int(self.version) not in (1, REVIEW_DISPLAY_OPTIONS_VERSION):
            raise ValueError(
                "ReviewDisplayOptions version "
                f"{self.version!r} is not the supported "
                f"{REVIEW_DISPLAY_OPTIONS_VERSION}."
            )

    @classmethod
    def from_mapping(cls, value) -> "ReviewDisplayOptions":
        """Build from ``None`` (defaults), a mapping, or an instance."""
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError(
                "display_options must be a ReviewDisplayOptions, a mapping, "
                f"or None; got {type(value).__name__}."
            )
        unknown = sorted(set(value) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(
                f"display_options has unknown field(s) {unknown}; accepted: "
                f"{sorted(cls.__dataclass_fields__)}."
            )
        fields = dict(value)
        if fields.get("version") == 1:
            # Old saved bundles keep their explicit fixed budgets.
            fields.setdefault("max_raster_spikes_per_unit", 2000)
            fields.setdefault("max_amplitudes_per_unit", 2000)
            fields.setdefault("raster_max_firing_rate", None)
            fields.setdefault("amplitude_max_firing_rate", None)
        return cls(**fields)

    def point_limit(self, kind, duration_s):
        """Resolve v1's duration-scaled budget and an optional smaller cap."""
        import math

        rate, cap = (
            (self.raster_max_firing_rate, self.max_raster_spikes_per_unit)
            if kind == "raster"
            else (self.amplitude_max_firing_rate, self.max_amplitudes_per_unit)
        )
        limits = [int(cap)] if cap is not None else []
        if rate is not None:
            limits.append(math.floor(duration_s * rate))
        return min(limits) if limits else None

    def as_dict(self) -> dict:
        """JSON-native form persisted in the review configuration."""
        return {
            "version": int(self.version),
            "raster_max_firing_rate": self.raster_max_firing_rate,
            "amplitude_max_firing_rate": self.amplitude_max_firing_rate,
            "max_initial_points": self.max_initial_points,
            "max_raster_spikes_per_unit": self.max_raster_spikes_per_unit,
            "max_amplitudes_per_unit": (
                None
                if self.max_amplitudes_per_unit is None
                else int(self.max_amplitudes_per_unit)
            ),
            "amplitude_sampling_seed": int(self.amplitude_sampling_seed),
            "min_similarity_for_correlograms": float(
                self.min_similarity_for_correlograms
            ),
        }

    def describe(self) -> str:
        """Short human-readable summary for view titles."""

        def budget(kind, rate, cap):
            limits = []
            if rate is not None:
                limits.append(f"floor(duration × {rate:g})")
            if cap is not None:
                limits.append(str(cap))
            formula = ("minimum of " if len(limits) > 1 else "") + " and ".join(
                limits
            )
            return f"{kind}: " + (
                formula + " points/unit" if limits else "all spikes"
            )

        return (
            budget(
                "raster",
                self.raster_max_firing_rate,
                self.max_raster_spikes_per_unit,
            )
            + "; "
            + budget(
                "amplitudes",
                self.amplitude_max_firing_rate,
                self.max_amplitudes_per_unit,
            )
            + "; "
            f"amplitude seed {self.amplitude_sampling_seed}; "
            "budgets span the full recording. Cross-correlograms for pairs with template "
            f"similarity >= {float(self.min_similarity_for_correlograms):g}"
        )

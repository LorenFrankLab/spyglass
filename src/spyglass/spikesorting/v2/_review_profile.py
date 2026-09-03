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


def profile_display_property_vocabulary(metric_row: Mapping) -> tuple[str, ...]:
    """Return the ordered built-in columns one metric recipe can display."""
    columns: list[str] = []
    for metric_name in metric_row["metric_names"]:
        columns.extend(
            _QUALITY_METRIC_OUTPUT_COLUMNS.get(metric_name, (metric_name,))
        )
    columns.extend(metric_row.get("template_metric_columns") or [])
    # Preserve recipe order while protecting against a future overlap between
    # metric and template output names.
    return tuple(dict.fromkeys(str(column) for column in columns))


def _normalize_label_options(label_options) -> list[str]:
    """Normalize and validate the ordered built-in curation-label palette."""
    if label_options is None:
        label_options = default_label_options()
    if isinstance(label_options, (str, bytes)) or not isinstance(
        label_options, (list, tuple)
    ):
        raise TypeError("label_options must be a list or tuple of strings.")
    valid = {label.value for label in CurationLabel}
    normalized: list[str] = []
    for label in label_options:
        value = CurationLabel.normalize(label)
        if value not in valid:
            raise ValueError(
                f"label_options contains unknown built-in label {label!r}; "
                f"valid labels are {sorted(valid)}."
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

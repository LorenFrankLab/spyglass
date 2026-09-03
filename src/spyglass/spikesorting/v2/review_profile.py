"""Immutable, persisted browser-curation review profiles.

``_PipelinePreset`` already binds the metric and auto-curation recipe names in
memory, while ``CurationEvaluationSelection`` binds those names for one
curation.  Neither is a durable user-facing review configuration.  This lookup
therefore persists the recipe pair together with the ordered display columns,
label palette, and label-import policy under one immutable, content-addressed
name.  Runtime delivery settings and curation-specific annotation selections
remain inputs to a review invocation and are intentionally absent.

This is a database-only module: it reuses dependency-light FigPack config
helpers but never imports the optional FigPack runtime.
"""

from __future__ import annotations

from collections.abc import Mapping

import datajoint as dj

from spyglass.spikesorting.v2._review_profile import (
    normalize_review_profile,
    profile_display_property_vocabulary,
    review_profile_label_policy,
)
from spyglass.spikesorting.v2._figpack_curation import default_label_options
from spyglass.spikesorting.v2._enums import CurationLabel
from spyglass.spikesorting.v2.exceptions import DuplicateParameterContentError
from spyglass.spikesorting.v2.metric_curation import (
    AutoCurationRules,
    QualityMetricParameters,
)
from spyglass.spikesorting.v2.utils import ImmutableParamsLookup
from spyglass.utils import SpyglassMixin, logger

schema = dj.schema("spikesorting_v2_metric_curation")

FRANKLAB_REVIEW_PROFILE = "franklab_hippocampus_2026_06"

# Content aliases must be deliberate shipped compatibility names, never an
# arbitrary caller opt-out. There are no aliases today; future shipped aliases
# must be added as an explicit pair/group here and covered by a migration note.
_SHIPPED_ALIAS_GROUPS: tuple[frozenset[str], ...] = ()
_PROFILE_CONTENT_FIELDS = (
    "metric_params_name",
    "auto_curation_rules_name",
    "displayed_unit_properties",
    "label_options",
    "label_import_mode",
    "profile_hash",
)


def _is_shipped_alias(first: str, second: str) -> bool:
    return any({first, second} <= group for group in _SHIPPED_ALIAS_GROUPS)


@schema
class CurationReviewProfile(ImmutableParamsLookup, SpyglassMixin, dj.Lookup):
    """One immutable recipe + display contract for browser curation.

    Insert through :meth:`insert`; a repeated name/content pair is
    idempotent, changed content requires a new name, and duplicate content under
    a second name is refused unless both names are an explicitly shipped alias.
    List order is semantic and is preserved in both storage and ``profile_hash``.
    """

    definition = """
    review_profile_name: varchar(64)
    ---
    -> QualityMetricParameters
    -> AutoCurationRules
    displayed_unit_properties: blob  # ordered built-in evaluation columns
    label_options: blob               # ordered built-in CurationLabel palette
    label_import_mode: enum('replace', 'overlay')
    profile_hash: char(64)             # sha256 over recipe/display/import semantics
    """

    def insert1(self, row, **kwargs):
        """Validate and insert one profile through the whole-row boundary."""
        self.insert([row], **kwargs)

    def insert(self, rows, *, replace=False, **kwargs):
        """Normalize, content-address, and insert immutable profile rows."""
        if replace:
            raise dj.errors.DataJointError(
                "CurationReviewProfile rows are immutable; replace=True is "
                "unsupported. Insert changed content under a new "
                "review_profile_name."
            )
        if isinstance(rows, Mapping):
            rows = [rows]
        normalized: list[dict] = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise TypeError(
                    "CurationReviewProfile rows must be mappings, not "
                    f"{type(row).__name__}."
                )
            metric_name = row.get("metric_params_name")
            if not metric_name or not (
                QualityMetricParameters & {"metric_params_name": metric_name}
            ):
                raise ValueError(
                    "CurationReviewProfile metric_params_name must resolve to "
                    f"one QualityMetricParameters row; got {metric_name!r}."
                )
            rules_name = row.get("auto_curation_rules_name")
            if not rules_name or not (
                AutoCurationRules & {"auto_curation_rules_name": rules_name}
            ):
                raise ValueError(
                    "CurationReviewProfile auto_curation_rules_name must resolve "
                    f"to one AutoCurationRules row; got {rules_name!r}."
                )
            metric_row = (
                QualityMetricParameters & {"metric_params_name": metric_name}
            ).fetch1()
            normalized.append(
                normalize_review_profile(
                    row,
                    available_displayed_properties=(
                        profile_display_property_vocabulary(metric_row)
                    ),
                )
            )

        stored = {
            row["review_profile_name"]: row for row in self.fetch(as_dict=True)
        }
        claimed_hashes = {
            row["profile_hash"]: row["review_profile_name"]
            for row in stored.values()
        }
        pending: list[dict] = []
        for row in normalized:
            name = row["review_profile_name"]
            existing = stored.get(name)
            if existing is not None:
                if any(
                    existing[field] != row[field]
                    for field in _PROFILE_CONTENT_FIELDS
                ):
                    raise ValueError(
                        f"CurationReviewProfile {name!r} already exists with "
                        "different content. Review profiles are immutable; "
                        "insert the changed configuration under a new dated name."
                    )
                if not kwargs.get("skip_duplicates", False):
                    logger.warning(
                        "CurationReviewProfile %r already exists with the same "
                        "content; returning it.",
                        name,
                    )
                continue
            prior = claimed_hashes.get(row["profile_hash"])
            if prior is not None and not _is_shipped_alias(prior, name):
                raise DuplicateParameterContentError(
                    f"CurationReviewProfile {name!r} duplicates the content of "
                    f"{prior!r} (profile_hash {row['profile_hash'][:12]}). "
                    "Reuse the existing profile name; aliases are permitted "
                    "only through the explicit shipped-alias allowlist."
                )
            claimed_hashes[row["profile_hash"]] = name
            stored[name] = row
            pending.append(row)
        if pending:
            super().insert(pending, replace=False, **kwargs)

    @classmethod
    def insert_default(cls) -> dict:
        """Install the dated Frank-lab hippocampus review profile."""
        row = {
            "review_profile_name": FRANKLAB_REVIEW_PROFILE,
            "metric_params_name": "franklab_default",
            "auto_curation_rules_name": (
                "franklab_default_auto_curation_2026_06"
            ),
            "displayed_unit_properties": [
                "snr",
                "isi_violation",
                "firing_rate",
                "num_spikes",
                "presence_ratio",
                "amplitude_cutoff",
                "nn_isolation",
                "nn_noise_overlap",
                "trough_half_width",
            ],
            # Start from the existing FigPack default palette and add the
            # Frank-lab rule verdict that curators must be able to inspect.
            "label_options": [
                *default_label_options(),
                CurationLabel.reject.value,
            ],
            "label_import_mode": "replace",
        }
        cls().insert1(row, skip_duplicates=True)
        return {"review_profile_name": FRANKLAB_REVIEW_PROFILE}

    @classmethod
    def label_policy(cls, restriction) -> str:
        """Resolve a profile's mode to CurationV2 ``label_policy``."""
        mode = (cls & restriction).fetch1("label_import_mode")
        return review_profile_label_policy(str(mode))

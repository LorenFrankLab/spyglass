"""Immutable unit-population snapshots and pure integrity checks.

Importing this module activates no schemas. Resolving a population is owned by
``analysis_selection``; snapshots contain JSON-native values for matching's
selection identity and DataJoint fetch/compute boundary.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping

from spyglass.spikesorting.v2._core.lookup_validation import (
    _jsonable_blob,
    lossless_int,
)
from spyglass.spikesorting.v2._core.selection_identity import sha256_json
from spyglass.spikesorting.v2._curation.annotation import ANNOTATION_VALUE_TYPES

if TYPE_CHECKING:
    from spyglass.spikesorting.v2.curation_api import CurationRef

_SNAPSHOT_FIELDS = frozenset(
    {
        "sorting_id",
        "curation_id",
        "curation_uuid",
        "policy_name",
        "policy",
        "unit_criteria",
        "included_unit_ids",
        "excluded_units",
        "labels_by_unit",
        "selection_provenance",
        "selection_hash",
    }
)
_ANNOTATION_FIELDS = frozenset(
    {
        "sorting_id",
        "curation_id",
        "curation_uuid",
        "annotation_name",
        "annotation_version",
        "value_type",
        "set_hash",
        "column_name",
    }
)
_EVALUATION_RECIPE_FIELDS = frozenset(
    {"metric_params_name", "auto_curation_rules_name"}
)


def unit_selection_hash(data: Mapping[str, Any]) -> str:
    """Hash the complete population snapshot, excluding its own hash."""
    return sha256_json(
        {key: value for key, value in data.items() if key != "selection_hash"},
        separators=(", ", ": "),
    )


def validate_unit_selection_snapshot(data: Mapping[str, Any]) -> dict:
    """Return a defensive JSON-native copy after checking pins, IDs and hash.

    This validates persisted structure without querying tables or consulting a
    policy's current contents. Consumers additionally verify the curation's
    generation and that these IDs belong to its committed unit namespace.
    """
    if not isinstance(data, Mapping) or set(data) != _SNAPSHOT_FIELDS:
        raise ValueError(
            "Unit selection snapshot has missing or unknown fields."
        )
    payload = _jsonable_blob(dict(data))
    for key in ("sorting_id", "curation_uuid"):
        try:
            canonical = str(uuid.UUID(payload[key]))
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError(
                f"Unit selection snapshot has invalid {key}."
            ) from exc
        if payload[key] != canonical:
            raise ValueError(f"Unit selection snapshot {key} is not canonical.")
    if lossless_int(payload["curation_id"], "curation_id") < 0:
        raise ValueError(
            "Unit selection snapshot curation_id must be nonnegative."
        )
    if (
        not isinstance(payload["policy_name"], str)
        or not payload["policy_name"]
    ):
        raise ValueError("Unit selection snapshot requires a policy_name.")
    policy = payload["policy"]
    if not isinstance(policy, dict) or set(policy) != {
        "include_labels",
        "exclude_labels",
    }:
        raise ValueError("Unit selection snapshot has an invalid label policy.")
    if any(
        not isinstance(labels, list)
        or any(not isinstance(label, str) for label in labels)
        for labels in policy.values()
    ):
        raise ValueError(
            "Unit selection policy labels must be lists of strings."
        )
    included = payload["included_unit_ids"]
    if not isinstance(included, list):
        raise ValueError("included_unit_ids must be a list of integers.")
    included = [lossless_int(value, "unit_id") for value in included]
    if included != sorted(set(included)):
        raise ValueError("included_unit_ids must be sorted and unique.")

    def keyed_units(name):
        value = payload[name]
        if not isinstance(value, dict):
            raise ValueError(
                f"Unit selection snapshot {name} must be a mapping."
            )
        ids = set()
        for key in value:
            try:
                unit_id = int(key)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid unit_id key in {name}: {key!r}."
                ) from exc
            if str(unit_id) != key:
                raise ValueError(
                    f"Noncanonical unit_id key in {name}: {key!r}."
                )
            ids.add(unit_id)
        return ids

    excluded = keyed_units("excluded_units")
    labeled = keyed_units("labels_by_unit")
    if set(included) & excluded or set(included) | excluded != labeled:
        raise ValueError(
            "Unit selection verdicts do not partition the labeled units."
        )
    if any(
        not isinstance(reason, str)
        for reason in payload["excluded_units"].values()
    ):
        raise ValueError("Excluded units must carry string reasons.")
    if any(
        not isinstance(labels, list)
        or any(not isinstance(label, str) for label in labels)
        for labels in payload["labels_by_unit"].values()
    ):
        raise ValueError("Unit labels must be lists of strings.")
    required = set(policy["include_labels"])
    denied = set(policy["exclude_labels"])
    for unit in included:
        labels = set(payload["labels_by_unit"][str(unit)])
        if labels & denied or (required and not labels & required):
            raise ValueError(
                "Included unit labels do not satisfy the snapshot policy."
            )
    if not isinstance(payload["unit_criteria"], dict):
        raise ValueError("Unit selection criteria must be a mapping.")
    provenance = payload["selection_provenance"]
    if (
        not isinstance(provenance, dict)
        or provenance.get("curation_uuid") != payload["curation_uuid"]
        or provenance.get("label_policy") != policy
        or provenance.get("unit_criteria") != payload["unit_criteria"]
    ):
        raise ValueError(
            "Unit selection provenance disagrees with the snapshot."
        )
    if not {
        "evaluation_id",
        "evaluation_recipes",
        "annotation_sets",
    }.issubset(provenance):
        raise ValueError("Unit selection provenance is missing evidence pins.")
    evaluation_id = provenance["evaluation_id"]
    recipes = provenance["evaluation_recipes"]
    if evaluation_id is not None:
        try:
            canonical = str(uuid.UUID(evaluation_id))
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError(
                "Unit selection has an invalid evaluation_id."
            ) from exc
        if canonical != evaluation_id:
            raise ValueError("Unit selection evaluation_id is not canonical.")
        if (
            not isinstance(recipes, dict)
            or set(recipes) != _EVALUATION_RECIPE_FIELDS
            or any(
                not isinstance(name, str) or not name or len(name) > 64
                for name in recipes.values()
            )
        ):
            raise ValueError("Unit selection has invalid evaluation recipes.")
    elif recipes is not None:
        raise ValueError("Evaluation recipes require an evaluation_id.")
    annotations = provenance["annotation_sets"]
    if not isinstance(annotations, list):
        raise ValueError("Unit selection annotation_sets must be a list.")
    seen_annotations = set()
    for annotation in annotations:
        if (
            not isinstance(annotation, dict)
            or set(annotation) != _ANNOTATION_FIELDS
        ):
            raise ValueError("Unit selection has an incomplete annotation pin.")
        lossless_int(annotation["curation_id"], "annotation curation_id")
        if any(
            annotation.get(key) != payload[key]
            for key in ("sorting_id", "curation_id", "curation_uuid")
        ):
            raise ValueError(
                "Annotation selection belongs to another curation."
            )
        name = annotation["annotation_name"]
        if not isinstance(name, str) or not name or len(name) > 64:
            raise ValueError("Unit selection has an invalid annotation_name.")
        version = lossless_int(
            annotation["annotation_version"], "annotation_version"
        )
        if version < 1:
            raise ValueError("annotation_version must be at least 1.")
        if (
            not isinstance(annotation["value_type"], str)
            or annotation["value_type"] not in ANNOTATION_VALUE_TYPES
        ):
            raise ValueError(
                "Unit selection has an invalid annotation value_type."
            )
        set_hash = annotation["set_hash"]
        if (
            not isinstance(set_hash, str)
            or len(set_hash) != 64
            or any(char not in "0123456789abcdef" for char in set_hash)
        ):
            raise ValueError(
                "Unit selection has an invalid annotation set_hash."
            )
        if (
            annotation["column_name"]
            != f"annotation:{name}:v{version}:{set_hash}"
        ):
            raise ValueError(
                "Annotation column_name disagrees with its pinned identity."
            )
        identity = (name, version, set_hash)
        if identity in seen_annotations:
            raise ValueError("Unit selection repeats an annotation pin.")
        seen_annotations.add(identity)
    if payload["selection_hash"] != unit_selection_hash(payload):
        raise ValueError(
            "Unit selection snapshot selection_hash does not match."
        )
    return payload


@dataclass(frozen=True)
class UnitSelectionSnapshot:
    """One exact population, independent of analysis groups and live labels.

    Obtain this handle from ``resolve_unit_selection``.
    Mapping-valued properties return immutable mappings or defensive copies.
    ``snapshot()`` returns a fresh primitive dictionary suitable for storage.
    Evaluation and annotation handles retain the original resolver arguments
    for read-only drift checking immediately before a planned match runs.
    """

    curation: CurationRef
    _payload_json: str = field(repr=False)
    evaluation: Any = field(default=None, repr=False, compare=False)
    annotation_sets: tuple = field(default=(), repr=False, compare=False)
    _criteria_from_policy: bool = field(
        default=False, repr=False, compare=False
    )

    def __post_init__(self):
        payload = validate_unit_selection_snapshot(
            json.loads(self._payload_json)
        )
        if (
            payload["sorting_id"] != str(self.curation.sorting_id)
            or payload["curation_id"] != self.curation.curation_id
            or payload["curation_uuid"] != str(self.curation.curation_uuid)
        ):
            raise ValueError(
                "Unit selection snapshot belongs to another curation."
            )
        object.__setattr__(self, "annotation_sets", tuple(self.annotation_sets))

    def snapshot(self) -> dict:
        """Return the pinned population and hash as a defensive primitive copy."""
        return json.loads(self._payload_json)

    @property
    def policy_name(self):
        return self.snapshot()["policy_name"]

    @property
    def policy(self):
        return MappingProxyType(
            {
                key: tuple(value)
                for key, value in self.snapshot()["policy"].items()
            }
        )

    @property
    def unit_criteria(self):
        return self.snapshot()["unit_criteria"]

    @property
    def included_unit_ids(self):
        return tuple(self.snapshot()["included_unit_ids"])

    @property
    def excluded_units(self):
        return MappingProxyType(
            {
                int(key): value
                for key, value in self.snapshot()["excluded_units"].items()
            }
        )

    @property
    def labels_by_unit(self):
        return MappingProxyType(
            {
                int(key): tuple(value)
                for key, value in self.snapshot()["labels_by_unit"].items()
            }
        )

    @property
    def unlabeled_unit_ids(self):
        return tuple(
            sorted(
                unit
                for unit, labels in self.labels_by_unit.items()
                if not labels
            )
        )

    @property
    def selection_provenance(self):
        return self.snapshot()["selection_provenance"]

    @property
    def selection_hash(self):
        return self.snapshot()["selection_hash"]

    def revalidate(self):
        """Re-resolve read-only and reject changed policy/labels/evidence."""
        from spyglass.spikesorting.v2.analysis_selection import (
            resolve_unit_selection,
        )

        current = resolve_unit_selection(
            self.curation,
            policy=self.policy_name,
            evaluation=self.evaluation,
            unit_criteria=(
                None if self._criteria_from_policy else self.unit_criteria
            ),
            annotation_sets=self.annotation_sets,
        )
        if current.selection_hash != self.selection_hash:
            raise ValueError(
                "Unit selection changed since planning; rebuild and review the plan."
            )
        return current

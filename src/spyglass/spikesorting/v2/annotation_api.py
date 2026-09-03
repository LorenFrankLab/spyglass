"""Dependency-light references and reads for typed unit annotations.

Table imports stay inside consuming methods so the public pipeline facade can
export these value objects without activating a DataJoint schema or requiring a
database connection at import time.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from spyglass.spikesorting.v2.curation_api import CurationRef, EvaluationResult

AnnotationValueType = Literal["float", "int", "bool", "text"]


def _uuid(value) -> uuid.UUID:
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


@dataclass(frozen=True)
class AnnotationDefinitionRef:
    """Immutable reference to one versioned annotation definition."""

    annotation_name: str
    annotation_version: int
    value_type: AnnotationValueType
    physical_unit: str
    description: str

    @classmethod
    def from_key(
        cls, key: Mapping[str, Any] | "AnnotationDefinitionRef"
    ) -> "AnnotationDefinitionRef":
        from spyglass.spikesorting.v2.unit_annotation import (
            UnitAnnotationDefinition,
        )

        if isinstance(key, cls):
            row = (
                UnitAnnotationDefinition
                & {
                    "annotation_name": key.annotation_name,
                    "annotation_version": key.annotation_version,
                }
            ).fetch1()
            current = cls._from_row(row)
            if current != key:
                raise LookupError(
                    "Annotation definition no longer matches its immutable "
                    f"reference: {key.annotation_name!r} version "
                    f"{key.annotation_version}."
                )
            return key
        missing = {"annotation_name", "annotation_version"} - set(key)
        if missing:
            raise ValueError(
                "AnnotationDefinitionRef.from_key requires annotation_name "
                f"and annotation_version; missing {sorted(missing)}."
            )
        row = (
            UnitAnnotationDefinition
            & {
                "annotation_name": str(key["annotation_name"]),
                "annotation_version": int(key["annotation_version"]),
            }
        ).fetch1()
        return cls._from_row(row)

    @classmethod
    def _from_row(cls, row: Mapping[str, Any]):
        return cls(
            annotation_name=str(row["annotation_name"]),
            annotation_version=int(row["annotation_version"]),
            value_type=str(row["value_type"]),
            physical_unit=str(row["physical_unit"]),
            description=str(row["description"]),
        )

    def as_key(self) -> dict[str, Any]:
        AnnotationDefinitionRef.from_key(self)
        return {
            "annotation_name": self.annotation_name,
            "annotation_version": self.annotation_version,
        }


@dataclass(frozen=True)
class AnnotationSetRef:
    """Identity-safe reference to one immutable annotation set."""

    curation: CurationRef
    annotation_name: str
    annotation_version: int
    value_type: AnnotationValueType
    set_hash: str

    @classmethod
    def from_key(
        cls, key: Mapping[str, Any] | "AnnotationSetRef"
    ) -> "AnnotationSetRef":
        from spyglass.spikesorting.v2.unit_annotation import (
            CurationUnitAnnotationSet,
            UnitAnnotationDefinition,
        )

        if isinstance(key, cls):
            key._current_row()
            return key
        required = {
            "sorting_id",
            "curation_id",
            "annotation_name",
            "annotation_version",
            "set_hash",
        }
        missing = required - set(key)
        if missing:
            raise ValueError(
                "AnnotationSetRef.from_key is missing required field(s) "
                f"{sorted(missing)}."
            )
        curation = CurationRef.from_key(key)
        if "curation_uuid" in key and curation.curation_uuid != _uuid(
            key["curation_uuid"]
        ):
            raise LookupError(
                "Annotation set key carries a stale curation_uuid for its "
                "numeric curation key."
            )
        dj_key = {
            **curation.as_key(),
            "annotation_name": str(key["annotation_name"]),
            "annotation_version": int(key["annotation_version"]),
            "set_hash": str(key["set_hash"]),
        }
        rows = (CurationUnitAnnotationSet & dj_key).fetch(as_dict=True)
        if len(rows) != 1:
            raise LookupError(
                "No unique CurationUnitAnnotationSet row for the supplied "
                f"identity: {dj_key}."
            )
        value_type = (UnitAnnotationDefinition & dj_key).fetch1("value_type")
        return cls(
            curation=curation,
            annotation_name=dj_key["annotation_name"],
            annotation_version=dj_key["annotation_version"],
            value_type=str(value_type),
            set_hash=dj_key["set_hash"],
        )

    @classmethod
    def from_snapshot(cls, snapshot: Mapping[str, Any]) -> "AnnotationSetRef":
        """Rehydrate an exact set from a persisted review snapshot."""
        ref = cls.from_key(snapshot)
        expected_uuid = _uuid(snapshot["curation_uuid"])
        if ref.curation.curation_uuid != expected_uuid:
            raise LookupError(
                "Annotation set snapshot targets a different curation UUID."
            )
        if str(snapshot.get("value_type")) != ref.value_type:
            raise LookupError(
                "Annotation set snapshot value_type does not match its "
                "definition."
            )
        if str(snapshot.get("column_name")) != ref.column_name:
            raise LookupError(
                "Annotation set snapshot column_name does not match its "
                "content-addressed identity."
            )
        return ref

    def _current_row(self) -> dict[str, Any]:
        from spyglass.spikesorting.v2.unit_annotation import (
            CurationUnitAnnotationSet,
            UnitAnnotationDefinition,
        )

        self.curation.as_key()
        rows = (CurationUnitAnnotationSet & self._unchecked_key()).fetch(
            as_dict=True
        )
        if len(rows) != 1:
            raise LookupError(
                "AnnotationSetRef no longer resolves to its immutable set: "
                f"{self._unchecked_key()}."
            )
        current_type = (
            UnitAnnotationDefinition & self._unchecked_key()
        ).fetch1("value_type")
        if str(current_type) != self.value_type:
            raise LookupError(
                "AnnotationSetRef definition no longer matches value_type "
                f"{self.value_type!r}."
            )
        return rows[0]

    def _unchecked_key(self) -> dict[str, Any]:
        return {
            "sorting_id": self.curation.sorting_id,
            "curation_id": self.curation.curation_id,
            "annotation_name": self.annotation_name,
            "annotation_version": self.annotation_version,
            "set_hash": self.set_hash,
        }

    def as_key(self) -> dict[str, Any]:
        self._current_row()
        return self._unchecked_key()

    @property
    def column_name(self) -> str:
        """Return the deterministic, collision-free common-reader column."""
        return (
            f"annotation:{self.annotation_name}:v{self.annotation_version}:"
            f"{self.set_hash}"
        )

    def snapshot(self) -> dict[str, Any]:
        """Return the immutable identity embedded in a review configuration."""
        self._current_row()
        return {
            **self._unchecked_key(),
            "sorting_id": str(self.curation.sorting_id),
            "curation_uuid": str(self.curation.curation_uuid),
            "value_type": self.value_type,
            "column_name": self.column_name,
        }

    def to_dataframe(self) -> pd.DataFrame:
        from spyglass.spikesorting.v2.unit_annotation import to_dataframe

        return to_dataframe(self)


def read_unit_properties(
    curation: CurationRef,
    *,
    evaluation: EvaluationResult | None,
    annotation_sets: Sequence[AnnotationSetRef | Mapping[str, Any]],
) -> pd.DataFrame:
    """Read explicitly selected built-in and custom properties by unit id.

    Built-in metric names retain their persisted names. Every custom column is
    namespaced as ``annotation:<name>:v<version>:<full set_hash>`` so two sets
    can never silently overwrite each other or a built-in metric.
    """
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.unit_annotation import to_dataframe

    if not isinstance(curation, CurationRef):
        raise TypeError("read_unit_properties requires curation=CurationRef.")
    curation_key = curation.as_key()
    unit_ids = [
        int(value)
        for value in (CurationV2.Unit & curation_key).fetch(
            "unit_id", order_by="unit_id"
        )
    ]
    result = pd.DataFrame(index=pd.Index(unit_ids, name="unit_id"))
    if evaluation is not None:
        if not isinstance(evaluation, EvaluationResult):
            raise TypeError(
                "evaluation must be an EvaluationResult or explicit None."
            )
        if evaluation.curation != curation:
            raise ValueError(
                "Selected evaluation belongs to a different curation."
            )
        metrics = evaluation.metrics
        try:
            metrics.index = [int(value) for value in metrics.index]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Selected evaluation metrics do not use integer unit ids."
            ) from exc
        if metrics.columns.duplicated().any():
            duplicates = list(metrics.columns[metrics.columns.duplicated()])
            raise ValueError(
                f"Selected evaluation has duplicate metric columns {duplicates}."
            )
        metric_units = set(metrics.index)
        if metric_units != set(unit_ids):
            raise ValueError(
                "Selected evaluation metrics do not match the exact curation "
                f"unit namespace: metrics={sorted(metric_units)}, "
                f"curation={unit_ids}."
            )
        result = metrics.reindex(unit_ids).copy(deep=True)
        result.index.name = "unit_id"

    seen_keys = set()
    for supplied in annotation_sets:
        ref = AnnotationSetRef.from_key(supplied)
        if ref.curation != curation:
            raise ValueError(
                f"Annotation set {ref.set_hash} belongs to a different "
                "curation."
            )
        identity = tuple(ref.as_key().values())
        if identity in seen_keys:
            raise ValueError(
                f"annotation_sets repeats set_hash {ref.set_hash}."
            )
        seen_keys.add(identity)
        column = ref.column_name
        if column in result.columns:
            raise ValueError(
                f"Deterministic unit-property column collision at {column!r}."
            )
        values = to_dataframe(ref).iloc[:, 0]
        result[column] = values.reindex(result.index)
    return result


__all__ = [
    "AnnotationDefinitionRef",
    "AnnotationSetRef",
    "read_unit_properties",
]

"""Typed, immutable unit annotations for exact v2 curation namespaces.

These tables annotate ``CurationV2.Unit`` rows, unlike the downstream v1
``UnitAnnotation`` table, which annotates units exposed through the merge
table.  Annotation values are computed properties, not curation labels, and
never participate in curation identity or ``merge_id`` construction.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import datajoint as dj
import pandas as pd

from spyglass.spikesorting.v2._lookup_validation import lossless_int
from spyglass.spikesorting.v2._unit_annotation import (
    ANNOTATION_VALUE_TYPES,
    annotation_set_hash,
    normalize_annotation_value,
    normalize_producer_parameters,
    producer_parameters_hash,
)
from spyglass.spikesorting.v2.annotation_api import (
    AnnotationDefinitionRef,
    AnnotationSetRef,
    AnnotationValueType,
    read_unit_properties,
)
from spyglass.spikesorting.v2.curation import CurationV2
from spyglass.spikesorting.v2.curation_api import CurationRef
from spyglass.spikesorting.v2.utils import (
    FactoryOnlyMaster,
    ImmutableParamsLookup,
)
from spyglass.utils import SpyglassMixin, SpyglassMixinPart, logger

schema = dj.schema("spikesorting_v2_unit_annotation")

_VALUE_COLUMN = {
    "float": "value_float",
    "int": "value_int",
    "bool": "value_bool",
    "text": "value_text",
}
_VALUE_COLUMNS = tuple(_VALUE_COLUMN.values())


def _special_value_state(value_type: str, value) -> str | None:
    """Return the lossless SQL-side tag for missing/non-finite values."""
    if value is None:
        return "none"
    if value_type == "float" and math.isnan(value):
        return "nan"
    if value_type == "float" and math.isinf(value):
        return "pos_inf" if value > 0 else "neg_inf"
    return None


def _decode_stored_value(value, special: str | None):
    if special is None:
        return value
    if special == "none":
        return None
    if special == "nan":
        return math.nan
    if special == "pos_inf":
        return math.inf
    if special == "neg_inf":
        return -math.inf
    raise RuntimeError(f"Unknown stored annotation special state {special!r}.")


def _bounded_text(value, *, field: str, max_length: int) -> str:
    value = str(value)
    if not value or len(value) > max_length:
        raise ValueError(
            f"{field} must be a non-empty string of at most {max_length} "
            "characters."
        )
    return value


@schema
class UnitAnnotationDefinition(ImmutableParamsLookup, SpyglassMixin, dj.Lookup):
    """Versioned definition of one computed, non-label unit property."""

    definition = """
    annotation_name: varchar(64)
    annotation_version: int
    ---
    value_type: enum('float', 'int', 'bool', 'text')
    physical_unit='': varchar(32)
    description='': varchar(255)
    """

    @classmethod
    def insert_definition(
        cls,
        annotation_name: str,
        annotation_version: int,
        value_type: AnnotationValueType,
        *,
        physical_unit: str = "",
        description: str = "",
    ) -> AnnotationDefinitionRef:
        """Insert/reuse one immutable definition and return its reference."""
        row = {
            "annotation_name": annotation_name,
            "annotation_version": annotation_version,
            "value_type": value_type,
            "physical_unit": physical_unit,
            "description": description,
        }
        cls().insert1(row, skip_duplicates=True)
        return AnnotationDefinitionRef.from_key(row)

    def insert1(self, row, **kwargs):
        """Validate a definition through the whole-row insert boundary."""
        self.insert([row], **kwargs)

    def insert(self, rows, *, replace=False, **kwargs):
        """Normalize rows and reject redefinition under an existing version."""
        if replace:
            raise dj.errors.DataJointError(
                "UnitAnnotationDefinition rows are immutable; replace=True "
                "is unsupported. Insert changed semantics under a new version."
            )
        if isinstance(rows, Mapping):
            rows = [rows]
        normalized = []
        for supplied in rows:
            if not isinstance(supplied, Mapping):
                raise TypeError(
                    "UnitAnnotationDefinition rows must be mappings."
                )
            row = {
                "annotation_name": _bounded_text(
                    supplied.get("annotation_name", ""),
                    field="annotation_name",
                    max_length=64,
                ),
                "annotation_version": lossless_int(
                    supplied.get("annotation_version", 0),
                    "annotation_version",
                ),
                "value_type": str(supplied.get("value_type", "")),
                "physical_unit": str(supplied.get("physical_unit", "")),
                "description": str(supplied.get("description", "")),
            }
            if row["annotation_version"] < 1:
                raise ValueError("annotation_version must be at least 1.")
            if row["value_type"] not in ANNOTATION_VALUE_TYPES:
                raise ValueError(
                    "value_type must be one of "
                    f"{sorted(ANNOTATION_VALUE_TYPES)}; got "
                    f"{row['value_type']!r}."
                )
            if len(row["physical_unit"]) > 32:
                raise ValueError("physical_unit is limited to 32 characters.")
            if len(row["description"]) > 255:
                raise ValueError("description is limited to 255 characters.")
            key = {
                "annotation_name": row["annotation_name"],
                "annotation_version": row["annotation_version"],
            }
            existing = (self & key).fetch(as_dict=True)
            if existing:
                if existing[0] != row:
                    raise ValueError(
                        f"Annotation definition {key} already exists with "
                        "different content. Definitions are immutable; use a "
                        "new annotation_version."
                    )
                if not kwargs.get("skip_duplicates", False):
                    logger.warning(
                        "Annotation definition %r version %s already exists "
                        "with the same content.",
                        row["annotation_name"],
                        row["annotation_version"],
                    )
                continue
            normalized.append(row)
        if normalized:
            super().insert(normalized, replace=False, **kwargs)


@schema
class CurationUnitAnnotationSet(FactoryOnlyMaster, SpyglassMixin, dj.Manual):
    """One immutable producer run over an exact curated-unit namespace."""

    _factory_create_call = "CurationUnitAnnotationSet.from_dataframe()"

    definition = """
    -> CurationV2
    -> UnitAnnotationDefinition
    set_hash: char(64)
    ---
    producer: varchar(128)
    producer_version: varchar(64)
    producer_parameters: blob
    parameters_hash: char(64)
    created_at=CURRENT_TIMESTAMP: timestamp
    created_by='': varchar(128)
    """

    class Value(FactoryOnlyMaster, SpyglassMixinPart):
        """One typed value for a DB-enforced ``CurationV2.Unit`` member."""

        _factory_create_call = "CurationUnitAnnotationSet.from_dataframe()"

        definition = """
        -> master
        -> CurationV2.Unit
        ---
        value_float=null: double
        value_int=null: bigint
        value_bool=null: bool
        value_text=null: varchar(255)
        value_special=null: enum('none','nan','pos_inf','neg_inf')
        """

        def insert(
            self,
            rows,
            replace=False,
            skip_duplicates=False,
            ignore_extra_fields=False,
            *,
            allow_direct_insert=False,
            **kwargs,
        ):
            """Validate the definition's one typed column before insertion."""
            if isinstance(rows, Mapping):
                rows = [rows]
            normalized = []
            value_types = {}
            for supplied in rows:
                row = dict(supplied)
                definition_key = {
                    "annotation_name": row["annotation_name"],
                    "annotation_version": row["annotation_version"],
                }
                definition_id = (
                    row["annotation_name"],
                    row["annotation_version"],
                )
                if definition_id not in value_types:
                    value_types[definition_id] = str(
                        (UnitAnnotationDefinition & definition_key).fetch1(
                            "value_type"
                        )
                    )
                value_type = value_types[definition_id]
                expected = _VALUE_COLUMN[value_type]
                unexpected = [
                    name
                    for name in _VALUE_COLUMNS
                    if name != expected and row.get(name) is not None
                ]
                if unexpected:
                    raise TypeError(
                        f"Definition {definition_key} has value_type "
                        f"{value_type!r}; only {expected} may be populated, "
                        f"not {unexpected}."
                    )
                value = normalize_annotation_value(
                    value_type, row.get(expected)
                )
                special = row.get("value_special")
                if special is None:
                    special = _special_value_state(value_type, value)
                valid_specials = {"none"}
                if value_type == "float":
                    valid_specials.update({"nan", "pos_inf", "neg_inf"})
                if special is not None and special not in valid_specials:
                    raise TypeError(
                        f"value_special={special!r} is invalid for "
                        f"value_type={value_type!r}."
                    )
                if (
                    special is not None
                    and value is not None
                    and not (value_type == "float" and not math.isfinite(value))
                ):
                    raise TypeError(
                        "value_special may only accompany a missing or "
                        "non-finite typed value."
                    )
                row["value_special"] = special
                row[expected] = None if special is not None else value
                normalized.append(row)
            super().insert(
                normalized,
                replace=replace,
                skip_duplicates=skip_duplicates,
                ignore_extra_fields=ignore_extra_fields,
                allow_direct_insert=allow_direct_insert,
                **kwargs,
            )

    @classmethod
    def from_dataframe(
        cls,
        curation: CurationRef,
        definition: AnnotationDefinitionRef | Mapping[str, Any],
        df: pd.DataFrame,
        *,
        producer: str = "manual",
        producer_version: str = "",
        producer_parameters: Mapping[str, Any] | None = None,
        created_by: str | None = None,
    ) -> AnnotationSetRef:
        """Validate, content-address, and atomically insert a DataFrame."""
        if not isinstance(curation, CurationRef):
            raise TypeError(
                "from_dataframe requires curation=CurationRef so the exact "
                "curation UUID is pinned."
            )
        if cls.connection.in_transaction:
            raise RuntimeError(
                "CurationUnitAnnotationSet.from_dataframe must be called "
                "outside an open DataJoint transaction; it atomically inserts "
                "the immutable master and value rows."
            )
        curation_key = curation.as_key()
        definition_ref = AnnotationDefinitionRef.from_key(definition)
        values = _values_from_dataframe(df, definition_ref)
        available_units = {
            int(value)
            for value in (CurationV2.Unit & curation_key).fetch("unit_id")
        }
        unknown = sorted({unit_id for unit_id, _ in values} - available_units)
        if unknown:
            raise ValueError(
                "Annotation values reference unit ids outside the exact "
                f"curation namespace: {unknown}. Available units: "
                f"{sorted(available_units)}."
            )

        producer = _bounded_text(producer, field="producer", max_length=128)
        producer_version = str(producer_version)
        if len(producer_version) > 64:
            raise ValueError("producer_version is limited to 64 characters.")
        parameters = normalize_producer_parameters(producer_parameters)
        params_hash = producer_parameters_hash(parameters)
        set_digest = annotation_set_hash(
            annotation_name=definition_ref.annotation_name,
            annotation_version=definition_ref.annotation_version,
            value_type=definition_ref.value_type,
            producer_parameters=parameters,
            values=values,
        )
        key = {
            **curation_key,
            **definition_ref.as_key(),
            "set_hash": set_digest,
        }
        existing = cls._reuse_existing(
            key,
            producer=producer,
            producer_version=producer_version,
            parameters=parameters,
            parameters_hash=params_hash,
            value_type=definition_ref.value_type,
        )
        if existing is not None:
            return existing

        author = (
            str(dj.config["database.user"])
            if created_by is None
            else str(created_by)
        )
        if len(author) > 128:
            raise ValueError("created_by is limited to 128 characters.")
        master_row = {
            **key,
            "producer": producer,
            "producer_version": producer_version,
            "producer_parameters": parameters,
            "parameters_hash": params_hash,
            "created_by": author,
        }
        value_column = _VALUE_COLUMN[definition_ref.value_type]
        value_rows = [
            {
                **key,
                "unit_id": unit_id,
                value_column: (
                    None
                    if _special_value_state(definition_ref.value_type, value)
                    else value
                ),
                "value_special": _special_value_state(
                    definition_ref.value_type, value
                ),
            }
            for unit_id, value in values
        ]
        try:
            with cls.connection.transaction:
                cls.insert1(master_row, allow_direct_insert=True)
                if value_rows:
                    cls.Value.insert(value_rows, allow_direct_insert=True)
        except dj.errors.DuplicateError:
            # A concurrent caller inserted the same content-addressed set
            # between the reuse check above and this insert: adopt the
            # winner (after the same provenance / stored-value checks).
            # Any other failure propagates; the transaction rolled back.
            existing = cls._reuse_existing(
                key,
                producer=producer,
                producer_version=producer_version,
                parameters=parameters,
                parameters_hash=params_hash,
                value_type=definition_ref.value_type,
            )
            if existing is None:
                raise
            return existing
        return AnnotationSetRef.from_key(key)

    @classmethod
    def _reuse_existing(
        cls,
        key,
        *,
        producer,
        producer_version,
        parameters,
        parameters_hash,
        value_type,
    ) -> AnnotationSetRef | None:
        rows = (cls & key).fetch(as_dict=True)
        if not rows:
            return None
        row = rows[0]
        mismatches = []
        for name, expected in (
            ("producer", producer),
            ("producer_version", producer_version),
            ("producer_parameters", parameters),
            ("parameters_hash", parameters_hash),
        ):
            if row[name] != expected:
                mismatches.append(name)
        if mismatches:
            raise ValueError(
                "Existing annotation set has the same content hash but "
                f"different provenance fields {mismatches}; refusing to "
                "silently retarget provenance."
            )
        stored_values = cls._stored_values(key, value_type)
        stored_hash = annotation_set_hash(
            annotation_name=key["annotation_name"],
            annotation_version=key["annotation_version"],
            value_type=value_type,
            producer_parameters=parameters,
            values=stored_values,
        )
        if stored_hash != key["set_hash"]:
            raise RuntimeError(
                "Stored annotation values do not match their set_hash; the "
                "immutable set was modified outside its factory."
            )
        return AnnotationSetRef.from_key(key)

    @classmethod
    def _stored_values(cls, key, value_type) -> list[tuple[int, object]]:
        column = _VALUE_COLUMN[value_type]
        return [
            (
                int(row["unit_id"]),
                _decode_stored_value(row[column], row["value_special"]),
            )
            for row in (cls.Value & key).fetch(
                "unit_id",
                column,
                "value_special",
                as_dict=True,
                order_by="unit_id",
            )
        ]

    @classmethod
    def to_dataframe(
        cls, set_key: AnnotationSetRef | Mapping[str, Any]
    ) -> pd.DataFrame:
        """Return one immutable set indexed by true curated ``unit_id``."""
        ref = AnnotationSetRef.from_key(set_key)
        return cls._dataframe_from_ref(ref)

    @classmethod
    def _dataframe_from_ref(cls, ref: AnnotationSetRef) -> pd.DataFrame:
        """Read values for a reference already validated by the caller."""
        values = cls._stored_values(ref._unchecked_key(), ref.value_type)
        unit_ids = [unit_id for unit_id, _ in values]
        raw_values = [value for _, value in values]
        if ref.value_type == "float":
            dtype = (
                "object" if any(v is None for v in raw_values) else "float64"
            )
            series = pd.Series(raw_values, dtype=dtype)
        elif ref.value_type == "int":
            dtype = "Int64" if any(v is None for v in raw_values) else "int64"
            series = pd.Series(raw_values, dtype=dtype)
        elif ref.value_type == "bool":
            dtype = "boolean" if any(v is None for v in raw_values) else "bool"
            series = pd.Series(raw_values, dtype=dtype)
        else:
            series = pd.Series(raw_values, dtype="object")
        frame = series.to_frame(name=ref.annotation_name)
        frame.index = pd.Index(unit_ids, name="unit_id", dtype="int64")
        return frame


def _values_from_dataframe(
    df: pd.DataFrame, definition: AnnotationDefinitionRef
) -> list[tuple[int, object]]:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("from_dataframe requires a pandas.DataFrame.")
    if "unit_id" in df.columns:
        value_columns = [name for name in df.columns if name != "unit_id"]
        unit_values = list(df["unit_id"])
    else:
        value_columns = list(df.columns)
        unit_values = list(df.index)
    if len(value_columns) != 1:
        raise ValueError(
            "Annotation DataFrame must contain exactly one value column, plus "
            "an optional unit_id column."
        )
    value_column = value_columns[0]
    raw_values = list(df[value_column])
    values = []
    for raw_unit_id, raw_value in zip(unit_values, raw_values):
        if isinstance(raw_unit_id, bool):
            raise TypeError("unit_id values must be integers, not booleans.")
        try:
            unit_id = int(raw_unit_id)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"unit_id {raw_unit_id!r} is not an integer."
            ) from exc
        if isinstance(raw_unit_id, float) and raw_unit_id != unit_id:
            raise TypeError(f"unit_id {raw_unit_id!r} is not an integer.")
        if raw_value is pd.NA:
            raw_value = None
        value = normalize_annotation_value(definition.value_type, raw_value)
        values.append((unit_id, value))
    if len({unit_id for unit_id, _ in values}) != len(values):
        raise ValueError(
            "Annotation DataFrame contains duplicate unit_id rows."
        )
    return sorted(values)


def from_dataframe(
    curation: CurationRef,
    definition: AnnotationDefinitionRef | Mapping[str, Any],
    df: pd.DataFrame,
    **kwargs,
) -> AnnotationSetRef:
    """Public function form of ``CurationUnitAnnotationSet.from_dataframe``."""
    return CurationUnitAnnotationSet.from_dataframe(
        curation, definition, df, **kwargs
    )


def to_dataframe(set_key: AnnotationSetRef | Mapping[str, Any]) -> pd.DataFrame:
    """Public function form of ``CurationUnitAnnotationSet.to_dataframe``."""
    return CurationUnitAnnotationSet.to_dataframe(set_key)


__all__ = [
    "AnnotationDefinitionRef",
    "AnnotationSetRef",
    "CurationUnitAnnotationSet",
    "UnitAnnotationDefinition",
    "from_dataframe",
    "read_unit_properties",
    "to_dataframe",
]

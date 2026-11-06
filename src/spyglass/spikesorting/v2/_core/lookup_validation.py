"""DB-free validation helpers for the v2 parameter Lookup tables.

Helpers that query receive their table at call time, so the module imports
without a DB connection.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Hashable, Mapping
from typing import TYPE_CHECKING

from spyglass.spikesorting._numerical import integer_scalar
from spyglass.spikesorting.v2._curation.metrics import rules_payloads_match
from spyglass.spikesorting.v2._core.parameter_identity import (
    parameter_fingerprint,
)
from spyglass.spikesorting.v2.exceptions import (
    DuplicateParameterContentError,
)

if TYPE_CHECKING:
    from pydantic import BaseModel


def lossless_int(value, what: str) -> int:
    """Convert a caller-supplied identifier to ``int`` without changing it.

    For user-facing ids. Accepts Python and NumPy integers; rejects booleans,
    fractions, strings and values outside int64.
    """
    return integer_scalar(value, name=what)


def _validate_params(model_cls: type[BaseModel], payload: dict) -> dict:
    """Validate a parameter payload against a Pydantic model.

    Parameters
    ----------
    model_cls : type[pydantic.BaseModel]
        The schema to validate against.
    payload : dict
        The raw parameter dict, typically a Lookup row's ``params`` blob.

    Returns
    -------
    dict
        The validated, normalized payload (``model_dump()`` output).

    Raises
    ------
    pydantic.ValidationError
        If ``payload`` does not satisfy ``model_cls``.
    """
    return model_cls.model_validate(payload).model_dump()


def _assert_schema_version_matches(row: dict, *, table_name: str) -> None:
    """Raise if outer and inner Pydantic ``schema_version`` disagree.

    Code that branches on the outer column would otherwise route the row to
    the wrong schema version.

    Parameters
    ----------
    row : dict
        The row being inserted, with a validated ``params`` blob. A row
        without ``params_schema_version`` passes.
    table_name : str
        Human-readable table name for the error message.

    Raises
    ------
    ValueError
        If ``row['params_schema_version']`` is set and does not match
        ``row['params']['schema_version']``.
    """
    inner = int(row["params"]["schema_version"])
    if "params_schema_version" not in row:
        return
    outer = int(row["params_schema_version"])
    if inner != outer:
        raise ValueError(
            f"{table_name}.insert1: params_schema_version={outer} does "
            f"not match the inner Pydantic schema_version={inner} on the "
            "validated params blob. Drop the column or align it with the "
            "blob's schema_version."
        )


def validate_lookup_rows(
    rows, attr_names, *, schema_for, table_name, per_row_hook=None
):
    """Validate + normalize params-Lookup rows before a (bulk) insert.

    Shared body of the validated-Lookup ``insert`` overrides. Per row:
    normalize to a dict, validate ``params``, run ``per_row_hook``, and check
    ``params_schema_version`` against the blob.

    Parameters
    ----------
    rows : iterable
        The rows handed to ``insert`` (mappings, positional sequences, or a
        ``QueryExpression``); normalized per row via ``_insert_row_to_dict``.
    attr_names : Sequence[str]
        ``self.heading.names``, used to label positional rows.
    schema_for : Callable[[dict], type[BaseModel]]
        Picks the schema for a row (per ``sorter`` for ``SorterParameters``).
    table_name : str
        Table name for the drift-check error message.
    per_row_hook : Callable[[dict, type[BaseModel]], None], optional
        Table-specific step run after validation and before the drift check.
        May mutate the row in place.

    Returns
    -------
    list[dict]
        Validated rows to hand to ``super().insert``.
    """
    validated = []
    for row in rows:
        row = _insert_row_to_dict(row, attr_names)
        schema_cls = schema_for(row)
        row["params"] = _validate_params(schema_cls, row["params"])
        if per_row_hook is not None:
            per_row_hook(row, schema_cls)
        # 0 is the column default ("unspecified"); an explicit non-zero value
        # is kept so the drift check can catch a mismatch.
        if row.get("params_schema_version", 0) == 0:
            row["params_schema_version"] = int(row["params"]["schema_version"])
        _assert_schema_version_matches(row, table_name=table_name)
        validated.append(row)
    return validated


def _jsonable_blob(blob):
    """Coerce a fetched DataJoint blob to JSON-native scalars/containers.

    A fetched blob can hold numpy scalars or arrays, which ``json.dumps``
    rejects; this makes it fingerprint like the plain-scalar insert form.
    ``None`` passes through.
    """
    if blob is None:
        return None

    def _coerce(obj):
        if hasattr(obj, "tolist"):  # numpy scalar / ndarray, duck-typed
            return obj.tolist()
        raise TypeError(
            f"non-JSON-serializable blob value of type {type(obj).__name__}"
        )

    return json.loads(json.dumps(blob, default=_coerce))


def parameter_row_fingerprint(
    table_name: str,
    row: dict,
    *,
    sorter_keyed: bool = False,
    matcher_keyed: bool = False,
) -> str:
    """Return the content fingerprint of one blob-valued parameter row.

    Works on a stored or a validated insert row. The row name is excluded.

    Parameters
    ----------
    table_name : str
        The fingerprint ``table`` field (e.g. ``"SorterParameters"``).
    row : dict
        The parameter row.
    sorter_keyed, matcher_keyed : bool, optional
        Fold ``row["sorter"]`` / ``row["matcher"]`` into the identity
        (``SorterParameters`` / ``MatcherParameters``).
    """
    # An insert dict may omit the column; the blob's version is authoritative.
    version = row.get("params_schema_version")
    if version is None:
        version = row["params"]["schema_version"]
    # Only SorterParameters carries execution_params.
    exec_params = _jsonable_blob(row.get("execution_params"))
    exec_version = None
    if exec_params is not None:
        exec_version = row.get("execution_params_schema_version")
        if exec_version is None:
            exec_version = exec_params.get("schema_version")
        if exec_version is not None:
            exec_version = int(exec_version)
    return parameter_fingerprint(
        table_name,
        params=_jsonable_blob(row["params"]),
        params_schema_version=int(version),
        job_kwargs=_jsonable_blob(row.get("job_kwargs")),
        sorter=row.get("sorter") if sorter_keyed else None,
        matcher=row.get("matcher") if matcher_keyed else None,
        execution_params=exec_params,
        execution_params_schema_version=exec_version,
    )


def reject_duplicate_parameter_content(
    table,
    validated_rows,
    *,
    table_name: str,
    name_attr: str,
    sorter_keyed: bool = False,
    matcher_keyed: bool = False,
    allow_duplicate_params: bool = False,
) -> None:
    """Reject a validated row whose content duplicates an existing row name.

    A second name for the same content forks provenance. Compares against
    stored rows and earlier rows in the batch. Re-inserting the same
    ``(name, content)`` is a no-op.

    Parameters
    ----------
    table : dj.Table
        The Lookup table instance, queried for already-stored rows.
    validated_rows : list[dict]
        Output of :func:`validate_lookup_rows` (plain-scalar ``params``).
    table_name : str
        The fingerprint ``table`` field (e.g. ``"SorterParameters"``).
    name_attr : str
        The params-name primary-key column (e.g. ``"sorter_params_name"``).
    sorter_keyed, matcher_keyed : bool, optional
        Scope detection per sorter (``SorterParameters``) or per matcher
        (``MatcherParameters``).
    allow_duplicate_params : bool, optional
        Opt out of the guard.

    Raises
    ------
    DuplicateParameterContentError
        If a row's content is already stored under a different name.
    """
    if allow_duplicate_params:
        return

    def _fingerprint(row: dict) -> str:
        return parameter_row_fingerprint(
            table_name,
            row,
            sorter_keyed=sorter_keyed,
            matcher_keyed=matcher_keyed,
        )

    def _pk(row: dict):
        return (
            (row["sorter"], row[name_attr]) if sorter_keyed else row[name_attr]
        )

    _reject_duplicate_fingerprints(
        table.fetch(as_dict=True),
        validated_rows,
        table_name=table_name,
        name_attr=name_attr,
        fingerprint=_fingerprint,
        pk=_pk,
    )


def _reject_duplicate_fingerprints(
    stored_rows: list[dict],
    incoming_rows: list[dict],
    *,
    table_name: str,
    name_attr: str,
    fingerprint: Callable[[dict], str],
    pk: Callable[[dict], Hashable],
) -> None:
    """Raise if an incoming row reuses another row name's content fingerprint.

    Parameters
    ----------
    stored_rows : list[dict]
        Rows already in the table; they claim their fingerprints first.
    incoming_rows : list[dict]
        Rows about to be inserted, checked in order against the stored rows
        and the earlier incoming rows.
    table_name : str
        Table name used in the error message.
    name_attr : str
        The params-name column.
    fingerprint : Callable[[dict], str]
        Content fingerprint of one row (row name excluded).
    pk : Callable[[dict], Hashable]
        The row's primary key; an incoming row whose key is already stored is
        a re-insert and is skipped.

    Raises
    ------
    DuplicateParameterContentError
        If an incoming row's content was already claimed by a different name.
    """
    existing_pks = {pk(row) for row in stored_rows}
    # fingerprint -> first row name to claim it
    claimed: dict[str, str] = {}
    for stored in stored_rows:
        claimed.setdefault(fingerprint(stored), stored[name_attr])

    for row in incoming_rows:
        # Keeps insert_default re-runnable after an opted-in duplicate.
        if pk(row) in existing_pks:
            continue
        name = row[name_attr]
        row_fingerprint = fingerprint(row)
        prior = claimed.get(row_fingerprint)
        if prior is not None and prior != name:
            raise DuplicateParameterContentError(
                f"{table_name}: row {name!r} duplicates the content of "
                f"existing row {prior!r} (fingerprint "
                f"{row_fingerprint[:12]}). A second name for identical "
                f"parameters forks provenance. Reuse {prior!r}, change the "
                "parameters, or pass allow_duplicate_params=True to insert "
                "it anyway."
            )
        claimed.setdefault(row_fingerprint, name)


def _quality_metric_content(row: dict) -> dict:
    """Extract one ``QualityMetricParameters`` row's ``params``-like content.

    Excludes the name, ``params_schema_version`` and ``job_kwargs``, as the
    sibling Lookups do. Adding a field changes every stored row's
    fingerprint.
    """
    return {
        "metric_names": row["metric_names"],
        "metric_kwargs": row["metric_kwargs"],
        "template_metric_columns": row["template_metric_columns"],
        "observed_presence_bin_duration_s": row[
            "observed_presence_bin_duration_s"
        ],
        "skip_pc_metrics": bool(row["skip_pc_metrics"]),
    }


def reject_duplicate_quality_metric_content(
    stored_rows,
    incoming_rows,
    *,
    allow_duplicate_params: bool = False,
) -> None:
    """Reject a ``QualityMetricParameters`` row whose content duplicates a name.

    The table spreads content across columns instead of one ``params`` blob,
    so it cannot use :func:`reject_duplicate_parameter_content`. Re-inserting
    a stored name is a no-op.

    Parameters
    ----------
    stored_rows : list[dict]
        QMP rows already in the table (``table.fetch(as_dict=True)``).
    incoming_rows : list[dict]
        The validated rows about to be inserted.
    allow_duplicate_params : bool, optional
        Opt out of the guard.

    Raises
    ------
    DuplicateParameterContentError
        If a row's content is already stored under a different name.
    """
    if allow_duplicate_params:
        return

    def _fingerprint(row: dict) -> str:
        return parameter_fingerprint(
            "QualityMetricParameters",
            params=_jsonable_blob(_quality_metric_content(row)),
            params_schema_version=int(row["params_schema_version"]),
            job_kwargs=_jsonable_blob(row.get("job_kwargs")),
        )

    _reject_duplicate_fingerprints(
        stored_rows,
        incoming_rows,
        table_name="QualityMetricParameters",
        name_attr="metric_params_name",
        fingerprint=_fingerprint,
        pk=lambda row: row["metric_params_name"],
    )


def reject_stale_quality_metric_defaults(stored_rows, default_rows) -> None:
    """Raise if a stored row claims a shipped default's name with stale content.

    ``insert_default`` uses ``skip_duplicates=True``, which would silently keep
    a hand-edited ``franklab_default``. Compares content,
    ``params_schema_version`` and ``job_kwargs`` exactly (bools distinct from
    ints).

    Parameters
    ----------
    stored_rows : list[dict]
        Already-stored ``QualityMetricParameters`` rows
        (``table.fetch(as_dict=True)``).
    default_rows : list[dict]
        The shipped default rows in stored shape (validated through
        ``prepare_quality_metric_row``).

    Raises
    ------
    DuplicateParameterContentError
        If a stored row's content differs from the same-name default.
    """
    stored_by_name = {row["metric_params_name"]: row for row in stored_rows}
    for default in default_rows:
        name = default["metric_params_name"]
        stored = stored_by_name.get(name)
        if stored is None:
            continue
        expected = _jsonable_blob(_quality_metric_content(default))
        actual = _jsonable_blob(_quality_metric_content(stored))
        expected["job_kwargs"] = _jsonable_blob(default.get("job_kwargs"))
        actual["job_kwargs"] = _jsonable_blob(stored.get("job_kwargs"))
        expected["params_schema_version"] = int(
            default["params_schema_version"]
        )
        actual["params_schema_version"] = int(stored["params_schema_version"])
        if not rules_payloads_match(expected, actual):
            raise DuplicateParameterContentError(
                f"QualityMetricParameters {name!r} already exists with "
                "content that differs from the shipped default. Insert your "
                "variant under a new metric_params_name, or delete the "
                f"existing {name!r} row first if nothing depends on it."
            )


def _insert_row_to_dict(row, attr_names) -> dict:
    """Normalize a DataJoint insert row to a mutable dict.

    Accepts mappings and positional sequences (zipped against the heading). A
    ``str``/``bytes`` row is what a DataFrame or CSV-path insert yields, and is
    rejected.

    Parameters
    ----------
    row : Mapping | Sequence
        One row from the iterable handed to ``insert``.
    attr_names : Sequence[str]
        The table heading's attribute names in definition order
        (``self.heading.names``), used to label a positional sequence.

    Returns
    -------
    dict
        A shallow-copied dict suitable for in-place ``params`` rewrite.

    Raises
    ------
    TypeError
        If ``row`` is a ``str``/``bytes``.
    ValueError
        If a positional row's length does not match ``attr_names``.
    """
    if isinstance(row, Mapping):
        return dict(row)
    if isinstance(row, (str, bytes)):
        raise TypeError(
            "validated Lookup insert expects each row to be a mapping or "
            f"a positional sequence; got {type(row).__name__}. Pass a list "
            "of dicts -- DataFrame / CSV-path inserts are not supported "
            "on these Pydantic-validated parameter tables."
        )
    values = tuple(row)
    if len(values) != len(attr_names):
        raise ValueError(
            "validated Lookup positional insert row has the wrong length: "
            f"expected {len(attr_names)} value(s) for attributes "
            f"{tuple(attr_names)!r}, got {len(values)} value(s). Pass a mapping "
            "row or align the positional tuple with the table heading."
        )
    return dict(zip(attr_names, values))


def _ensure_lookup_row_exists(
    lookup_table,
    restriction: dict,
    *,
    helper_name: str,
    insert_default_path: str,
) -> None:
    """Pre-check that a Lookup-row FK target exists before insert_selection.

    Replaces DataJoint's opaque FK ``IntegrityError`` with a ``ValueError``
    that names the missing row and how to insert it.

    Parameters
    ----------
    lookup_table
        The Lookup table class whose row is required (e.g.
        ``PreprocessingParameters``).
    restriction
        The dict identifying the required row (e.g.
        ``{"preprocessing_params_name": "default"}``).
    helper_name
        Name of the insert_selection helper calling us, for the error
        message (e.g. ``"RecordingSelection.insert_selection"``).
    insert_default_path
        Importable path that loads the default rows (e.g.
        ``"PreprocessingParameters.insert_default()"``).
    """
    if not (lookup_table & restriction):
        raise ValueError(
            f"{helper_name}: required Lookup row not found in "
            f"{lookup_table.__name__} for {restriction}. "
            f"Run {insert_default_path} first to install the default "
            "rows, or insert your custom row before retrying. The "
            "one-shot `spyglass.spikesorting.v2.initialize_v2_defaults()`"
            " installs every required default in one call."
        )

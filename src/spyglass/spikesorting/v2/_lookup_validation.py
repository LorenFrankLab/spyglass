"""DB-free validation helpers for the v2 parameter Lookup tables.

The Pydantic params-blob validation, the outer/inner schema-version drift
check, the row-normalization + bulk-validate path, the content-fingerprint
duplicate guard, and the FK-row pre-check that backs ``insert_selection``
all live here. Free of DataJoint / SpikeInterface / spyglass.common at
import (the DB-touching helpers -- ``reject_duplicate_parameter_content``,
``_ensure_lookup_row_exists`` -- receive their table at call time), so the
module imports without a DB connection. ``utils`` re-exports these names so
existing ``from .utils import validate_lookup_rows`` (etc.) call sites are
unchanged.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING

from spyglass.spikesorting.v2._parameter_identity import parameter_fingerprint
from spyglass.spikesorting.v2.exceptions import (
    DuplicateParameterContentError,
)

if TYPE_CHECKING:
    from pydantic import BaseModel


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


def _assert_schema_version_matches(
    row: dict, model_cls: type[BaseModel], *, table_name: str
) -> None:
    """Raise if outer and inner Pydantic ``schema_version`` disagree.

    Each Lookup table stores a ``params_schema_version`` column
    alongside the validated ``params`` blob. The blob also carries a
    ``schema_version`` field (Pydantic-validated). If a user inserts
    a row where the two values disagree, downstream code that
    branches on the outer column will silently route the row to the
    wrong schema version's behavior. This helper catches the drift at
    insert time so the row never lands.

    Parameters
    ----------
    row : dict
        The full row dict being inserted. Must have a ``params``
        entry that already contains a ``schema_version`` (i.e. the
        caller has already run ``_validate_params``).
    model_cls : type[pydantic.BaseModel]
        The schema class. Its default ``schema_version`` is used as
        the fallback when the row omits ``params_schema_version``.
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

    Shared body of the validated-Lookup ``insert`` overrides
    (``PreprocessingParameters``, ``ArtifactDetectionParameters``,
    ``SorterParameters``, ``MotionCorrectionParameters``,
    ``AnalyzerWaveformParameters``): for each row,
    normalize to a dict, validate its ``params`` blob against the row's
    schema, run an optional per-row check, and assert the outer
    ``params_schema_version`` agrees with the validated blob. Returns the
    list of validated row dicts to hand to ``super().insert``. Each table's
    ``insert1`` delegates to ``insert`` (so a single code path validates
    both), mirroring ``CurationV2.UnitLabel``.

    Parameters
    ----------
    rows : iterable
        The rows handed to ``insert`` (mappings, positional sequences, or a
        ``QueryExpression``); normalized per row via ``_insert_row_to_dict``.
    attr_names : Sequence[str]
        ``self.heading.names``, used to label positional rows.
    schema_for : Callable[[dict], type[BaseModel]]
        Picks the schema for a row -- a constant for the single-schema
        tables, per-``sorter`` for ``SorterParameters``.
    table_name : str
        Table name for the drift-check error message.
    per_row_hook : Callable[[dict, type[BaseModel]], None], optional
        Extra table-specific step run after validation and before the
        drift assertion. May mutate the row in place: ``SorterParameters``
        uses it to reject unknown sorter names and to backfill
        ``params_schema_version`` from the validated blob.
    """
    validated = []
    for row in rows:
        row = _insert_row_to_dict(row, attr_names)
        schema_cls = schema_for(row)
        row["params"] = _validate_params(schema_cls, row["params"])
        if per_row_hook is not None:
            per_row_hook(row, schema_cls)
        _assert_schema_version_matches(row, schema_cls, table_name=table_name)
        validated.append(row)
    return validated


def _jsonable_blob(blob):
    """Coerce a fetched DataJoint blob to JSON-native scalars/containers.

    Validated insert rows carry plain Python scalars, but a blob fetched back
    from the table can deserialize numbers as numpy scalars (or arrays). The
    content fingerprint canonicalizes via ``json.dumps``, which rejects those,
    so round-trip the blob with a duck-typed ``tolist()`` fallback (numpy
    scalars and arrays both expose it) to get a value that fingerprints
    identically to the plain-scalar insert form. ``None`` passes through.
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

    The duplicate-content guard shared by the validated parameter Lookups.
    After :func:`validate_lookup_rows` normalizes the batch,
    fingerprint each incoming row (row NAME excluded; ``SorterParameters``
    scoped per sorter and ``MatcherParameters`` per matcher via the
    fingerprint's ``sorter`` / ``matcher`` field) and compare
    against the fingerprints already in the table plus earlier rows in the
    same batch. A second NAME for content that already ships under a different
    name forks provenance, so it raises
    :class:`~spyglass.spikesorting.v2.exceptions.DuplicateParameterContentError`.
    Re-inserting the SAME ``(name, content)`` pair is idempotent and never
    trips the guard, so ``insert_default()`` stays re-runnable under
    ``skip_duplicates=True``. ``allow_duplicate_params=True`` is the documented
    escape hatch (the row then shows a ``duplicate_of`` in
    ``describe_parameter_rows``).

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
    sorter_keyed : bool, optional
        ``True`` for ``SorterParameters`` so detection is scoped per sorter.
    allow_duplicate_params : bool, optional
        Opt out of the guard (the documented escape hatch).
    """
    if allow_duplicate_params:
        return

    def _version(row: dict) -> int:
        # A dict insert may omit the ``params_schema_version`` column (the
        # DataJoint column default fills it at write time, so the validated
        # dict has no such key yet). The validated ``params`` blob always
        # carries the authoritative inner ``schema_version``, and
        # ``_assert_schema_version_matches`` keeps the outer column in
        # lockstep with it, so fall back to the blob when the column is
        # absent rather than KeyError-ing on a perfectly valid insert.
        version = row.get("params_schema_version")
        if version is None:
            version = row["params"]["schema_version"]
        return int(version)

    def _exec_version(row: dict, exec_params) -> int | None:
        # Only SorterParameters carries execution_params; for the single-key
        # Lookups it is absent and the fingerprint omits it (None). A dict
        # insert may omit the outer ``execution_params_schema_version`` column,
        # so fall back to the validated blob's inner ``schema_version`` the same
        # way ``_version`` does for ``params_schema_version``.
        if exec_params is None:
            return None
        version = row.get("execution_params_schema_version")
        if version is None:
            version = exec_params.get("schema_version")
        return int(version) if version is not None else None

    def _fingerprint(row: dict) -> str:
        exec_params = _jsonable_blob(row.get("execution_params"))
        return parameter_fingerprint(
            table_name,
            params=_jsonable_blob(row["params"]),
            params_schema_version=_version(row),
            job_kwargs=_jsonable_blob(row.get("job_kwargs")),
            sorter=row.get("sorter") if sorter_keyed else None,
            matcher=row.get("matcher") if matcher_keyed else None,
            execution_params=exec_params,
            execution_params_schema_version=_exec_version(row, exec_params),
        )

    def _pk(row: dict):
        # The identity DataJoint keys on: the params-name plus the sorter for
        # the per-sorter SorterParameters table.
        return (
            (row["sorter"], row[name_attr]) if sorter_keyed else row[name_attr]
        )

    stored_rows = table.fetch(as_dict=True)
    existing_pks = {_pk(row) for row in stored_rows}
    # fingerprint -> the row name that first claimed it (stored rows first,
    # then earlier rows in this same batch).
    claimed: dict[str, str] = {}
    for stored in stored_rows:
        claimed.setdefault(_fingerprint(stored), stored[name_attr])

    for row in validated_rows:
        # A re-insert of an already-stored row (same primary key) is
        # idempotent, not a new provenance fork: skip it so insert_default /
        # initialize_v2_defaults stay re-runnable even after an opted-in
        # duplicate has been added under another name.
        if _pk(row) in existing_pks:
            continue
        name = row[name_attr]
        fingerprint = _fingerprint(row)
        prior = claimed.get(fingerprint)
        if prior is not None and prior != name:
            raise DuplicateParameterContentError(
                f"{table_name}: row {name!r} duplicates the content of "
                f"existing row {prior!r} (fingerprint {fingerprint[:12]}). A "
                "second name for identical parameters forks provenance. Reuse "
                f"{prior!r}, change the parameters, or pass "
                "allow_duplicate_params=True to insert it anyway."
            )
        claimed.setdefault(fingerprint, name)


def _insert_row_to_dict(row, attr_names) -> dict:
    """Normalize a DataJoint insert row to a mutable dict.

    A bulk ``insert`` accepts both mapping rows (``{"sorter": ...}``,
    what user code passes) and positional sequences (``("mountainsort4",
    name, params, 1, None)``, what each Lookup's ``_DEFAULT_CONTENTS``
    ships to ``insert_default``). The ``insert`` validation overrides
    need a dict in both cases so they can read/rewrite ``row["params"]``;
    a positional tuple is zipped against the table heading's attribute
    order to recover the dict form. A ``QueryExpression`` passed to
    ``insert`` is fine -- iterating it yields per-row dicts, which hit the
    mapping branch.

    A bare ``str``/``bytes`` row is rejected loudly: it is the shape that
    leaks through when a caller passes a ``pandas.DataFrame`` or a CSV
    path to ``insert`` (iterating those yields column-name strings /
    characters), neither of which these validated Lookups support.
    ``zip``-ing a string against the heading would otherwise produce a
    silently malformed row.

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
        If ``row`` is a ``str``/``bytes`` (an unsupported insert form
        for these validated Lookups).
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

    Without this guard, a missing Lookup row produces an opaque
    DataJoint ``IntegrityError`` ("foreign key constraint fails")
    that gives the user no hint about which Lookup table is empty or
    how to populate it. Raise a clear ``ValueError`` instead so the
    notebook user can fix the setup in one step.

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

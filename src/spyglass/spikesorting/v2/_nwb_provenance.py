"""DB-free NWB provenance scratch (de)serialization for the v2 writers.

The v2 NWB writers persist *results* (traces, spikes, pairs, metrics); this
helper writes the *provenance* that makes each artifact self-describing into
NWB scratch space, under stable container names retrieved BY NAME (never by
object id, so no new DataJoint columns are needed to find them). Two shapes:

- a scalar bundle (``build_provenance_table``): a two-column ``key`` /
  ``value_json`` ``DynamicTable``, one row per field, every value JSON-encoded.
  JSON sidesteps HDF5's inability to store a dict / list / ``None`` in a typed
  cell, so nested params and nullable fields round-trip uniformly.
- a typed long table (``build_long_provenance_table``): one row per member /
  pair / boundary with concrete string / int / float / bool columns (concrete
  dtypes so even a zero-row table writes).

Builders return a ``DynamicTable``; the caller embeds it with
``nwbfile.add_scratch(...)`` (directly, or via a writer's ``provenance_tables``
loop).

Every container carries ``provenance_schema_version`` so a future reader can
tell which provenance layout produced the file.

This module touches no DataJoint connection: the builders are pure and the read
functions take an absolute file path the ``@schema`` table layer resolves.
"""

from __future__ import annotations

import json
import uuid

import numpy as np
import pynwb
from hdmf.common import DynamicTable, VectorData

#: Bumped when the provenance container layout changes.
PROVENANCE_SCHEMA_VERSION = 1

#: Stable scratch container names (retrieved by name, not by object id).
RECORDING_PROVENANCE = "spyglass_v2_recording_provenance"
SORTING_PROVENANCE = "spyglass_v2_sorting_provenance"
UNITMATCH_PROVENANCE = "spyglass_v2_unitmatch_provenance"
UNITMATCH_MEMBERS = "spyglass_v2_unitmatch_members"
CURATION_PROVENANCE = "spyglass_v2_curation_provenance"
CURATION_MERGE_LINEAGE = "spyglass_v2_curation_merge_lineage"
CURATION_EVALUATION_PROVENANCE = "spyglass_v2_curation_evaluation_provenance"
CONCAT_PROVENANCE = "spyglass_v2_concat_provenance"
CONCAT_MEMBERS = "spyglass_v2_concat_members"

_KEY_COLUMN = "key"
_VALUE_COLUMN = "value_json"
_SCHEMA_VERSION_FIELD = "provenance_schema_version"

#: Concrete numpy dtype for each supported long-table column python type.
_NUMPY_FOR_PYTYPE = {
    str: object,
    int: np.int64,
    float: np.float64,
    bool: np.bool_,
}


def _json_default(value):
    """JSON encoder fallback for DataJoint-deserialized value types.

    Provenance bundles re-emit values fetched from DataJoint (metric kwargs,
    rule thresholds, hash manifests, row ids), which deserialize as numpy
    scalars / arrays and ``uuid.UUID``; ``json.dumps`` cannot encode those
    natively. UUIDs are stored in their canonical string form.
    """
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, uuid.UUID):
        return str(value)
    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable"
    )


def _to_python(value):
    """Coerce a numpy / HDF5 scalar back to a native Python type."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, str):  # also collapses numpy.str_ to exact str
        return str(value)
    return value


def build_provenance_table(
    name: str, values: dict, *, description: str = ""
) -> DynamicTable:
    """Build a ``key`` / ``value_json`` scalar-provenance table.

    Each entry in ``values`` becomes one row whose ``value_json`` is the
    JSON-encoded value (so dicts / lists / ``None`` survive HDF5). A
    ``provenance_schema_version`` row is always appended.
    """
    payload = dict(values)
    payload[_SCHEMA_VERSION_FIELD] = PROVENANCE_SCHEMA_VERSION
    keys = sorted(payload)
    return DynamicTable(
        name=name,
        description=description or f"{name} (key/value_json provenance)",
        columns=[
            VectorData(
                name=_KEY_COLUMN,
                description="provenance field name",
                data=np.asarray(keys, dtype=object),
            ),
            VectorData(
                name=_VALUE_COLUMN,
                description="JSON-encoded provenance value",
                data=np.asarray(
                    [
                        json.dumps(
                            payload[k], sort_keys=True, default=_json_default
                        )
                        for k in keys
                    ],
                    dtype=object,
                ),
            ),
        ],
    )


def build_long_provenance_table(
    name: str,
    rows: list[dict],
    columns: list[tuple[str, type]],
    *,
    description: str = "",
) -> DynamicTable:
    """Build a typed long provenance table (one row per item).

    ``columns`` is an ordered list of ``(column_name, python_type)`` where the
    python type is one of ``str`` / ``int`` / ``float`` / ``bool``. A constant
    ``provenance_schema_version`` int column is appended. Concrete dtypes mean a
    zero-row ``rows`` still writes a valid (empty) table.
    """
    col_specs = list(columns) + [(_SCHEMA_VERSION_FIELD, int)]
    data: dict[str, list] = {col: [] for col, _ in col_specs}
    for row in rows:
        for col, _ in columns:
            data[col].append(row[col])
        data[_SCHEMA_VERSION_FIELD].append(PROVENANCE_SCHEMA_VERSION)
    vector_columns = [
        VectorData(
            name=col,
            description=col,
            data=np.asarray(data[col], dtype=_NUMPY_FOR_PYTYPE[pytype]),
        )
        for col, pytype in col_specs
    ]
    return DynamicTable(
        name=name,
        description=description or f"{name} (provenance)",
        columns=vector_columns,
    )


def read_provenance_values(abs_path: str, name: str) -> dict:
    """Read a scalar-provenance table back to ``{field: value}`` (JSON-decoded)."""
    with pynwb.NWBHDF5IO(path=abs_path, mode="r", load_namespaces=True) as io:
        frame = io.read().scratch[name].to_dataframe()
    return {
        str(row[_KEY_COLUMN]): json.loads(row[_VALUE_COLUMN])
        for _, row in frame.iterrows()
    }


def read_long_provenance(abs_path: str, name: str) -> list[dict]:
    """Read a typed long provenance table back to a list of native-type dicts."""
    with pynwb.NWBHDF5IO(path=abs_path, mode="r", load_namespaces=True) as io:
        frame = io.read().scratch[name].to_dataframe()
    return [
        {col: _to_python(value) for col, value in record.items()}
        for record in frame.to_dict(orient="records")
    ]

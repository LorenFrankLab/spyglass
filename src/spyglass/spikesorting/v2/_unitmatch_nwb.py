"""DB-free NWB (de)serialization for cross-session match output.

``UnitMatch.make`` writes one scratch table into an ``AnalysisNwbfile``:

``unit_match_pairs``
    a long table, one row per cross-session match pair, carrying both sides'
    ``(sorting_id, curation_id, unit_id)`` plus the match probability and the
    (default) drift / FDR columns. ``fdr_estimate`` has no per-pair backend
    source, so a missing value is stored as a native HDF5 NaN (HDF5 cannot store
    ``None`` in a numeric column) and surfaced back as ``None`` on read.

The structured, FK-validated copy of the same pairs lives in the
``UnitMatch.Pair`` DataJoint part table; this NWB table is the exportable
analysis artifact (it travels with DANDI / kachery / recompute like every other
v2 analysis NWB). A degenerate single-session run writes an empty table.

This module touches no DataJoint connection: the builder is pure and the write /
read functions take an absolute file path the ``@schema`` table layer resolves.
"""

from __future__ import annotations

import numpy as np
import pynwb
from hdmf.common import DynamicTable, VectorData

UNIT_MATCH_PAIRS_TABLE = "unit_match_pairs"

#: Column order written to / read from the NWB pairs table.
_STR_COLUMNS = ("session_a_sorting_id", "session_b_sorting_id")
_INT_COLUMNS = (
    "pair_index",
    "session_a_curation_id",
    "unit_a_id",
    "session_b_curation_id",
    "unit_b_id",
)
_FLOAT_COLUMNS = ("match_probability", "drift_estimate_um", "fdr_estimate")


def build_pairs_table(pairs: list[dict]) -> DynamicTable:
    """Build the long cross-session match-pairs table.

    Parameters
    ----------
    pairs : list[dict]
        Oriented pair dicts (output of
        :func:`~spyglass.spikesorting.v2._matcher_graph.canonicalize_match_pairs`).
        ``pair_index`` is assigned here in list order. A ``None`` ``fdr_estimate``
        is stored as NaN.

    Returns
    -------
    hdmf.common.DynamicTable
        Concrete-dtype columns so even an empty (zero-pair) table writes.
    """
    columns = {name: [] for name in _STR_COLUMNS + _INT_COLUMNS + _FLOAT_COLUMNS}
    for index, pair in enumerate(pairs):
        columns["pair_index"].append(index)
        for name in _STR_COLUMNS:
            columns[name].append(str(pair[name]))
        for name in _INT_COLUMNS:
            if name != "pair_index":
                columns[name].append(int(pair[name]))
        columns["match_probability"].append(float(pair["match_probability"]))
        columns["drift_estimate_um"].append(float(pair["drift_estimate_um"]))
        fdr = pair.get("fdr_estimate")
        columns["fdr_estimate"].append(
            float("nan") if fdr is None else float(fdr)
        )

    # Concrete dtypes so even an empty (zero-pair) table writes: object for the
    # uuid strings, int64 / float64 for the numeric columns.
    dtype_for = dict.fromkeys(_STR_COLUMNS, object)
    dtype_for.update(dict.fromkeys(_INT_COLUMNS, np.int64))
    dtype_for.update(dict.fromkeys(_FLOAT_COLUMNS, np.float64))
    vector_columns = [
        VectorData(
            name=name,
            description=name,
            data=np.asarray(columns[name], dtype=dtype_for[name]),
        )
        for name in _STR_COLUMNS + _INT_COLUMNS + _FLOAT_COLUMNS
    ]
    return DynamicTable(
        name=UNIT_MATCH_PAIRS_TABLE,
        description=(
            "Cross-session unit match pairs; one row per matched (unit_a, "
            "unit_b) across two SessionGroup members."
        ),
        columns=vector_columns,
    )


def write_pairs_table(abs_path: str, pairs: list[dict]) -> str:
    """Append the pairs table to an existing analysis NWB; return its object id.

    Registering the file in the DataJoint ``AnalysisNwbfile`` table is the
    caller's responsibility (done inside the insert transaction).
    """
    table = build_pairs_table(pairs)
    with pynwb.NWBHDF5IO(path=abs_path, mode="a", load_namespaces=True) as io:
        nwbf = io.read()
        nwbf.add_scratch(table)
        object_id = table.object_id
        io.write(nwbf)
    return object_id


def read_pairs(abs_path: str, object_id: str) -> list[dict]:
    """Read the pairs table back to a list of dicts (NaN ``fdr_estimate`` -> None)."""
    with pynwb.NWBHDF5IO(path=abs_path, mode="r", load_namespaces=True) as io:
        nwbf = io.read()
        frame = nwbf.objects[object_id].to_dataframe()
    rows: list[dict] = []
    # itertuples(index=False) is faster + cleaner than iterrows (no per-row
    # Series). Pair columns are valid identifiers, so attribute access works.
    for row in frame.itertuples(index=False):
        fdr = float(row.fdr_estimate)
        rows.append(
            {
                "pair_index": int(row.pair_index),
                "session_a_sorting_id": str(row.session_a_sorting_id),
                "session_a_curation_id": int(row.session_a_curation_id),
                "unit_a_id": int(row.unit_a_id),
                "session_b_sorting_id": str(row.session_b_sorting_id),
                "session_b_curation_id": int(row.session_b_curation_id),
                "unit_b_id": int(row.unit_b_id),
                "match_probability": float(row.match_probability),
                "drift_estimate_um": float(row.drift_estimate_um),
                "fdr_estimate": None if np.isnan(fdr) else fdr,
            }
        )
    return rows

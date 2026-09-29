"""DB-free NWB (de)serialization for cross-session match output.

``UnitMatch.make`` writes the pairs table into an ``AnalysisNwbfile``, next to
the run's provenance tables (``_nwb_provenance``: the run header, one row per
matching input, one row per constituent recording):

``unit_match_pairs``
    a long table, one row per match pair between two matching inputs, carrying
    both sides' ``(sorting_id, curation_id, unit_id)`` (``session_a_*`` is the
    input with the lower ``input_index``) plus the match probability and the
    (default) drift / FDR columns. ``fdr_estimate`` has no per-pair backend
    source, so a missing value is stored as a native HDF5 NaN (HDF5 cannot store
    ``None`` in a numeric column) and surfaced back as ``None`` on read.

The structured, FK-validated copy of the same pairs lives in the
``UnitMatch.Pair`` DataJoint part table; this NWB table is the exportable
analysis artifact (it travels with DANDI / kachery / recompute like every other
v2 analysis NWB). A single-input run writes an empty table.

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
    columns = {
        name: [] for name in _STR_COLUMNS + _INT_COLUMNS + _FLOAT_COLUMNS
    }
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
            "unit_b) across two matching inputs."
        ),
        columns=vector_columns,
    )


def write_pairs_table(
    abs_path: str, pairs: list[dict], *, provenance_tables=None
) -> str:
    """Append the pairs table to an existing analysis NWB; return its object id.

    Registering the file in the DataJoint ``AnalysisNwbfile`` table is the
    caller's responsibility (done inside the insert transaction).

    ``provenance_tables`` (optional, from :mod:`._nwb_provenance`) are embedded
    as scratch in the same write so the artifact is self-describing; ``None``
    leaves only the pairs table.
    """
    table = build_pairs_table(pairs)
    with pynwb.NWBHDF5IO(path=abs_path, mode="a", load_namespaces=True) as io:
        nwbf = io.read()
        nwbf.add_scratch(table)
        object_id = table.object_id
        for prov in provenance_tables or ():
            nwbf.add_scratch(prov)
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


def read_input_provenance(abs_path: str) -> tuple[list[dict], list[dict]]:
    """Read a match run's per-input and per-recording provenance tables.

    ``UnitMatch.make_compute`` writes the two tables next to the pairs table
    (:data:`~spyglass.spikesorting.v2._nwb_provenance.UNITMATCH_INPUTS` and
    :data:`~spyglass.spikesorting.v2._nwb_provenance.UNITMATCH_INPUT_RECORDINGS`).

    Parameters
    ----------
    abs_path : str
        Absolute path of the run's analysis NWB file.

    Returns
    -------
    tuple of (list of dict, list of dict)
        The input rows sorted by ``input_index`` and the recording rows sorted
        by ``(input_index, recording_index)``, with the columns of
        ``UNITMATCH_INPUT_COLUMNS`` / ``UNITMATCH_INPUT_RECORDING_COLUMNS`` as
        written (the layout's ``provenance_schema_version`` column dropped).
    """
    from spyglass.spikesorting.v2._nwb_provenance import (
        UNITMATCH_INPUT_COLUMNS,
        UNITMATCH_INPUT_RECORDING_COLUMNS,
        UNITMATCH_INPUT_RECORDINGS,
        UNITMATCH_INPUTS,
        read_long_provenance,
    )

    def _read(name, columns):
        return [
            {column: row[column] for column, _ in columns}
            for row in read_long_provenance(abs_path, name)
        ]

    inputs = _read(UNITMATCH_INPUTS, UNITMATCH_INPUT_COLUMNS)
    recordings = _read(
        UNITMATCH_INPUT_RECORDINGS, UNITMATCH_INPUT_RECORDING_COLUMNS
    )
    inputs.sort(key=lambda row: row["input_index"])
    recordings.sort(
        key=lambda row: (row["input_index"], row["recording_index"])
    )
    return inputs, recordings

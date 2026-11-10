"""NWB (de)serialization for cross-session match output.

``unit_match_pairs`` has one row per match pair, with both sides'
``(sorting_id, curation_id, unit_id)`` (``session_a_*`` has the lower
``input_index``). Missing drift / FDR values are stored as NaN and read back
as ``None``. ``UnitMatch.Pair`` holds the FK-validated copy; this table is the
exportable artifact. A single-input run writes an empty table.
"""

from __future__ import annotations

import numpy as np
import pynwb
from hdmf.common import DynamicTable

from spyglass.spikesorting.v2._storage.provenance import (
    concrete_vectordata,
    read_table_frames,
)
from spyglass.spikesorting.v2._storage.provenance import (
    UNITMATCH_INPUTS,
    UNITMATCH_INPUT_COLUMNS,
    UNITMATCH_INPUT_RECORDINGS,
    UNITMATCH_INPUT_RECORDING_COLUMNS,
    long_provenance_from_frame,
)

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
_OPTIONAL_FLOAT_COLUMNS = ("drift_estimate_um", "fdr_estimate")


def build_pairs_table(pairs: list[dict]) -> DynamicTable:
    """Build the long cross-session match-pairs table.

    Parameters
    ----------
    pairs : list[dict]
        Oriented pair dicts (output of
        :func:`~spyglass.spikesorting.v2._matching.graph.canonicalize_match_pairs`).
        ``pair_index`` is assigned here in list order. A ``None``
        ``drift_estimate_um`` / ``fdr_estimate`` is stored as NaN.

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
        for name in _OPTIONAL_FLOAT_COLUMNS:
            value = pair.get(name)
            columns[name].append(
                float("nan") if value is None else float(value)
            )

    dtype_for = dict.fromkeys(_STR_COLUMNS, object)
    dtype_for.update(dict.fromkeys(_INT_COLUMNS, np.int64))
    dtype_for.update(dict.fromkeys(_FLOAT_COLUMNS, np.float64))
    vector_columns = [
        concrete_vectordata(name, columns[name], dtype_for[name])
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

    The caller registers the file in ``AnalysisNwbfile``. Optional
    ``provenance_tables`` are added as scratch in the same write.
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
    """Read the pairs table back to a list of dicts (NaN drift / FDR -> None)."""
    (frame,) = read_table_frames(abs_path, object_ids=(object_id,))
    rows: list[dict] = []
    for row in frame.itertuples(index=False):
        drift = float(row.drift_estimate_um)
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
                "drift_estimate_um": None if np.isnan(drift) else drift,
                "fdr_estimate": None if np.isnan(fdr) else fdr,
            }
        )
    return rows


def read_input_provenance(abs_path: str) -> tuple[list[dict], list[dict]]:
    """Read a match run's per-input and per-recording provenance tables.

    Parameters
    ----------
    abs_path : str
        Absolute path of the run's analysis NWB file.

    Returns
    -------
    tuple of (list of dict, list of dict)
        The input rows sorted by ``input_index`` and the recording rows sorted
        by ``(input_index, recording_index)``, without the
        ``provenance_schema_version`` column.
    """
    input_frame, recording_frame = read_table_frames(
        abs_path,
        scratch_names=(UNITMATCH_INPUTS, UNITMATCH_INPUT_RECORDINGS),
    )

    def _rows(frame, columns):
        return [
            {column: row[column] for column, _ in columns}
            for row in long_provenance_from_frame(frame)
        ]

    inputs = _rows(input_frame, UNITMATCH_INPUT_COLUMNS)
    recordings = _rows(recording_frame, UNITMATCH_INPUT_RECORDING_COLUMNS)
    inputs.sort(key=lambda row: row["input_index"])
    recordings.sort(
        key=lambda row: (row["input_index"], row["recording_index"])
    )
    return inputs, recordings

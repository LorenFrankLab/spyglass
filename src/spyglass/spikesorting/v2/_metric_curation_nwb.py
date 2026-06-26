"""DB-free NWB (de)serialization for analyzer-curation outputs.

``CurationEvaluation`` writes three scratch tables into one AnalysisNwbfile:

``quality_metrics``
    a wide table, one row per unit, one column per quality metric. NaN is
    preserved on disk as a native HDF5 float (HDF5 cannot store ``None`` in a
    numeric column); ``None`` is surfaced on the read path (and for the
    FigPack JSON export) by ``sanitize_for_json``.

``merge_suggestions``
    a long table ``(merge_group_index, unit_id)`` -- one row per (group,
    member). Empty when no merges are proposed.

``proposed_labels``
    a wide table, one row per unit, plus a ragged ``curation_label`` column
    that is added ONLY when at least one unit is labeled (mirrors the
    curated-units writer, sidestepping the #1625 "cannot infer dtype of empty
    list" crash for the all-clean-units case).

This module touches no DataJoint connection: builders are pure, and the write
/ read functions take an absolute file path that the ``@schema`` table layer
resolves. A zero-unit curation writes three empty tables: the quality-metrics
and proposed-labels tables are column-less, while the merge-suggestions table
keeps its two columns (``merge_group_index``, ``unit_id``) with zero rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pynwb
from hdmf.common import DynamicTable, VectorData

from spyglass.spikesorting.v2._metric_curation import sanitize_for_json

QUALITY_METRICS_TABLE = "quality_metrics"
MERGE_SUGGESTIONS_TABLE = "merge_suggestions"
PROPOSED_LABELS_TABLE = "proposed_labels"


def _vectordata(name: str, values, dtype) -> VectorData:
    """Build a ``VectorData`` with a concrete numpy dtype.

    Passing a concrete-dtype array (rather than a Python list) lets hdmf write
    even an empty column -- an empty Python list raises "Cannot infer dtype".
    """
    return VectorData(
        name=name, description=name, data=np.asarray(values, dtype=dtype)
    )


def _scalar_or_nan(value, *, context: str = "") -> float:
    """Coerce a metric cell to float; non-scalar / non-numeric -> NaN.

    Every quality metric and every surfaced single-channel template column is a
    plain scalar, so this is belt-and-suspenders: it keeps a future SI column
    whose cell is not a plain number (e.g. an array) from crashing the wide
    one-float-per-column write -- the stray value lands as NaN instead.

    The coercion-failure substitution is on the scientific-result write path
    (a downstream auto-curation rule would silently never fire on a NaN metric),
    so it is logged at warning level with ``context`` (unit / column) rather than
    swallowed silently. A legitimately non-finite SI value (already a float NaN)
    takes the ``float(value)`` success path and is not logged.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        from spyglass.utils import logger

        logger.warning(
            "quality-metrics write: a non-scalar/non-numeric cell "
            f"(type {type(value).__name__!r}) was coerced to NaN"
            + (f" [{context}]" if context else "")
            + "; a future SpikeInterface column may have changed shape."
        )
        return float("nan")


def build_quality_metrics_table(metrics_df: pd.DataFrame) -> DynamicTable:
    """Wide quality + waveform-shape metrics table (one row per unit).

    ``metrics_df`` is indexed by ``unit_id`` and carries the quality metrics
    plus any surfaced template (waveform-shape) columns. NaN is preserved. A
    zero-row frame yields an empty, column-less table (the zero-unit case).
    """
    table = DynamicTable(
        name=QUALITY_METRICS_TABLE,
        description=(
            "SpikeInterface quality + waveform-shape (template) metrics, one "
            "row per unit."
        ),
    )
    if len(metrics_df) == 0:
        return table
    table.add_column(name="unit_id", description="SpikeInterface unit id")
    metric_columns = list(metrics_df.columns)
    for column in metric_columns:
        table.add_column(name=column, description=column)
    for row in metrics_df.itertuples(index=True, name=None):
        unit_id = row[0]
        values = row[1:]
        table.add_row(
            unit_id=int(unit_id),
            **{
                column: _scalar_or_nan(
                    value,
                    context=f"unit_id={int(unit_id)}, column={column!r}",
                )
                for column, value in zip(metric_columns, values)
            },
        )
    return table


def build_merge_suggestions_table(
    merge_groups: list[list[int]],
) -> DynamicTable:
    """Long merge-suggestions table ``(merge_group_index, unit_id)``."""
    group_index: list[int] = []
    unit_ids: list[int] = []
    for index, group in enumerate(merge_groups):
        for unit_id in group:
            group_index.append(index)
            unit_ids.append(int(unit_id))
    return DynamicTable(
        name=MERGE_SUGGESTIONS_TABLE,
        description="Proposed merge groups; one row per (group, unit).",
        columns=[
            _vectordata("merge_group_index", group_index, np.int64),
            _vectordata("unit_id", unit_ids, np.int64),
        ],
    )


def build_proposed_labels_table(
    unit_ids: list[int], labels_by_unit: dict[int, list[str]]
) -> DynamicTable:
    """Wide proposed-labels table; ragged label column only when ≥1 label.

    The ``curation_label`` column is omitted entirely when no unit is labeled,
    so a clean sort never writes an all-empty list-of-lists (#1625).
    """
    table = DynamicTable(
        name=PROPOSED_LABELS_TABLE,
        description="Proposed auto-curation labels, one row per unit.",
    )
    if len(unit_ids) == 0:
        return table
    ordered_units = [int(u) for u in unit_ids]
    table.add_column(name="unit_id", description="SpikeInterface unit id")
    for unit_id in ordered_units:
        table.add_row(unit_id=unit_id)
    label_lists = [list(labels_by_unit.get(u, [])) for u in ordered_units]
    if any(label_lists):
        table.add_column(
            name="curation_label",
            description="proposed curation labels (ragged, per unit)",
            data=label_lists,
            index=True,
        )
    return table


def write_analyzer_curation_tables(
    abs_path: str,
    *,
    metrics_df: pd.DataFrame,
    merge_groups: list[list[int]],
    labels_by_unit: dict[int, list[str]],
    unit_ids: list[int],
) -> tuple[str, str, str]:
    """Write the three curation tables into an existing analysis NWB file.

    Opens ``abs_path`` once in append mode, adds the three scratch tables, and
    returns their object IDs ``(quality_metrics, merge_suggestions,
    proposed_labels)``. Registering the file in the DataJoint
    ``AnalysisNwbfile`` table is the caller's responsibility (done inside the
    insert transaction).
    """
    qm_table = build_quality_metrics_table(metrics_df)
    ms_table = build_merge_suggestions_table(merge_groups)
    pl_table = build_proposed_labels_table(unit_ids, labels_by_unit)
    with pynwb.NWBHDF5IO(path=abs_path, mode="a", load_namespaces=True) as io:
        nwbf = io.read()
        nwbf.add_scratch(qm_table)
        nwbf.add_scratch(ms_table)
        nwbf.add_scratch(pl_table)
        object_ids = (
            qm_table.object_id,
            ms_table.object_id,
            pl_table.object_id,
        )
        io.write(nwbf)
    return object_ids


def read_quality_metrics(abs_path: str, object_id: str) -> pd.DataFrame:
    """Read the quality-metrics table; returns a DataFrame indexed by unit_id.

    Non-finite values are coerced to ``None`` (the on-disk representation is
    NaN). An empty table yields an empty DataFrame.
    """
    with pynwb.NWBHDF5IO(path=abs_path, mode="r", load_namespaces=True) as io:
        nwbf = io.read()
        frame = nwbf.objects[object_id].to_dataframe()
    if "unit_id" not in frame.columns:
        return pd.DataFrame()
    frame = frame.set_index("unit_id")
    frame.index = frame.index.astype(int)
    frame.index.name = "unit_id"
    return sanitize_for_json(frame)


def read_merge_suggestions(abs_path: str, object_id: str) -> list[list[int]]:
    """Read the merge-suggestions table back to a list of unit-id groups."""
    with pynwb.NWBHDF5IO(path=abs_path, mode="r", load_namespaces=True) as io:
        nwbf = io.read()
        frame = nwbf.objects[object_id].to_dataframe()
    if len(frame) == 0:
        return []
    groups: dict[int, list[int]] = {}
    for row in frame.itertuples(index=False):
        groups.setdefault(int(row.merge_group_index), []).append(
            int(row.unit_id)
        )
    return [groups[index] for index in sorted(groups)]


def read_proposed_labels(
    abs_path: str, object_id: str
) -> dict[int, list[str]]:
    """Read proposed labels back to ``{unit_id: [label, ...]}`` (labeled only)."""
    with pynwb.NWBHDF5IO(path=abs_path, mode="r", load_namespaces=True) as io:
        nwbf = io.read()
        frame = nwbf.objects[object_id].to_dataframe()
    if "curation_label" not in frame.columns or len(frame) == 0:
        return {}
    labels: dict[int, list[str]] = {}
    for row in frame.itertuples(index=False):
        unit_labels = [str(label) for label in row.curation_label]
        if unit_labels:
            labels[int(row.unit_id)] = unit_labels
    return labels

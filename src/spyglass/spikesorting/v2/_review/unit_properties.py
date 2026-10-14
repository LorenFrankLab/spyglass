"""Shape a review's official unit properties for SpikeInterface's unit table.

A profile-backed browser review shows its **selected evaluation's** metrics,
annotation-set columns and label / merge proposals in the same selectable
``UnitsTable`` that drives unit selection for the curation control (so a
metric-bearing row IS the unit being curated). SpikeInterface's
unit-table builder takes those columns as
one-dimensional NumPy arrays aligned to the analyzer's unit order and accepts
only integer / unsigned / float / bool / string dtypes -- object arrays
(including ordinary pandas string columns) are silently dropped with a
warning. This module performs that alignment and dtype normalization while
keeping an unavailable value unavailable: numeric gaps stay NaN (SI renders
them as an empty cell), string gaps stay ``""``, and a boolean column with a
gap is rendered as text so a missing annotation never becomes ``False``.

DB-free: NumPy + pandas only.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def missing_rule_metrics(
    metrics: pd.DataFrame, metric_names: Sequence[str]
) -> pd.Series:
    """Name unavailable inputs for enabled rules, keeping the evaluation's IDs.

    A metric absent from the recipe is relevant only if a configured rule
    requires it. This is evidence coverage, not a quality verdict.
    """
    names = list(dict.fromkeys(metric_names))
    values = metrics.reindex(columns=names).to_numpy(
        dtype=float, na_value=np.nan
    )
    return pd.Series(
        [
            ", ".join(
                name for name, available in zip(names, row) if not available
            )
            for row in np.isfinite(values)
        ],
        index=metrics.index,
        name="unavailable_qc",
        dtype=str,
    )


def _text(value) -> str:
    return "" if pd.isna(value) else str(value)


def display_column_array(values: pd.Series) -> np.ndarray:
    """One review column as an SI-accepted 1-D array, gaps preserved.

    Parameters
    ----------
    values : pandas.Series
        The column, already aligned to the analyzer's unit order.

    Returns
    -------
    numpy.ndarray
        ``float64`` for numeric columns (missing -> NaN, so ``int`` and
        ``float`` annotations keep their values and a gap stays empty),
        ``bool`` for a complete boolean column, and a unicode string array
        otherwise (missing -> ``""``; an incomplete boolean column becomes
        ``"True"`` / ``"False"`` / ``""``).
    """
    if pd.api.types.is_bool_dtype(values) and not values.isna().any():
        return values.to_numpy(dtype=bool)
    if pd.api.types.is_numeric_dtype(values) and not (
        pd.api.types.is_bool_dtype(values)
    ):
        return values.to_numpy(dtype="float64", na_value=np.nan)
    return np.array([_text(value) for value in values], dtype=str)


def review_unit_properties(
    table: pd.DataFrame, unit_ids: Sequence[int]
) -> dict[str, np.ndarray]:
    """Align a review table to ``unit_ids`` and normalize every column.

    Parameters
    ----------
    table : pandas.DataFrame
        Indexed by unit id; columns in the order they should be displayed.
    unit_ids : sequence of int
        The analyzer's unit ids in analyzer order (``analyzer.unit_ids``).

    Returns
    -------
    dict[str, numpy.ndarray]
        ``{column: values}`` in the table's column order, one entry per unit
        in ``unit_ids`` order; a unit absent from the table has a gap.

    Raises
    ------
    ValueError
        If the table carries a unit id that is not in ``unit_ids`` (the
        review would be describing units the analyzer does not show).
    """
    ordered = [int(unit_id) for unit_id in unit_ids]
    stray = sorted(set(int(v) for v in table.index) - set(ordered))
    if stray:
        raise ValueError(
            "Review properties reference unit ids absent from the analyzer "
            f"being displayed: {stray}. Analyzer unit ids: {ordered}."
        )
    aligned = table.reindex(ordered)
    return {
        str(column): display_column_array(aligned[column])
        for column in table.columns
    }

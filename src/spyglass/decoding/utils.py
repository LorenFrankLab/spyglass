"""Utilities shared across the decoding pipelines."""

import pandas as pd

# Head-direction column names vary by position source: v1 Trodes/DLC position
# uses "orientation" while legacy common_position uses "head_orientation".
ORIENTATION_COLS = ("orientation", "head_orientation")


def resolve_orientation_col(
    df: pd.DataFrame, orientation_name: str | None = None
) -> str | None:
    """Resolve the orientation column name in a position dataframe.

    Position sources name the head-direction column differently: v1
    Trodes/DLC position uses ``"orientation"`` while legacy ``common_position``
    uses ``"head_orientation"``. A requested ``orientation_name`` is honored
    when present; otherwise the first known orientation column found in ``df``
    is returned (preferring ``"orientation"``).

    Parameters
    ----------
    df : pandas.DataFrame
        Position dataframe whose columns are inspected.
    orientation_name : str, optional
        Preferred column name to use if present in ``df``, by default None.

    Returns
    -------
    str or None
        The resolved orientation column name, or ``None`` if neither the
        requested column nor a known orientation column is present.
    """
    cols = df.columns
    candidates = (orientation_name, *ORIENTATION_COLS)
    return next(
        (col for col in candidates if col is not None and col in cols), None
    )

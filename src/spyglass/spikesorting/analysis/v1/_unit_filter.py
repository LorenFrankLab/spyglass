"""Label-based unit filtering shared by ``SortedSpikesGroup`` and the v2 handoff.

DB-free: ``SortedSpikesGroup.filter_units`` delegates here so the v2
``select_units_for_analysis`` receipt can compute its verdicts with the exact
function ``fetch_spike_data`` applies, without importing the schema module.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("spyglass")


def filter_units_by_labels(
    labels: list[list[str]],
    include_labels: list[str],
    exclude_labels: list[str],
) -> np.ndarray:
    """Return a boolean mask of units passing the include/exclude policy.

    Parameters
    ----------
    labels : list of list of str
        Labels for each unit (a bare ``str`` is treated as one label).
    include_labels : list of str
        If non-empty, a unit must carry at least one of these labels.
    exclude_labels : list of str
        A unit carrying any of these labels is excluded.

    Returns
    -------
    np.ndarray
        Boolean mask, shape ``(n_units,)``. All ``True`` when both lists are
        empty.
    """
    include_labels = np.unique(include_labels)
    exclude_labels = np.unique(exclude_labels)

    if include_labels.size == 0 and exclude_labels.size == 0:
        # if no labels are provided, include all units
        return np.ones(len(labels), dtype=bool)

    include_mask = np.zeros(len(labels), dtype=bool)
    for ind, unit_labels in enumerate(labels):
        if isinstance(unit_labels, str):
            unit_labels = [unit_labels]
        if (
            include_labels.size > 0
            and np.all(~np.isin(unit_labels, include_labels))
        ) or np.any(np.isin(unit_labels, exclude_labels)):
            # if the unit does not have any of the include labels
            # or has any of the exclude labels, skip
            continue
        include_mask[ind] = True
    return include_mask


# Operators accepted in UnitSelectionParams.unit_criteria, each applied
# elementwise to one column of the units table
UNIT_CRITERIA_OPERATORS = {
    ">": np.greater,
    ">=": np.greater_equal,
    "<": np.less,
    "<=": np.less_equal,
    "==": np.equal,
    "!=": np.not_equal,
    "between": lambda vals, target: (vals >= target[0]) & (vals <= target[1]),
    "outside": lambda vals, target: (vals < target[0]) | (vals > target[1]),
    "isin": lambda vals, target: _isin(vals, target),
    "notin": lambda vals, target: ~_isin(vals, target),
}
# Only "isin" and "notin" are list-aware, via _isin. The rest compare a
# unit's value as a whole, so on a column holding a list per unit (e.g. the
# curation labels) numpy compares the list object itself to the target: a
# list never equals a string, making "==" all-False and "!=" all-True
# regardless of the labels. filter_units_by_criteria raises rather than
# return such a mask
SCALAR_ONLY_OPERATORS = frozenset(UNIT_CRITERIA_OPERATORS) - {"isin", "notin"}
# Operators taking a [low, high] pair rather than one value
RANGE_OPERATORS = ("between", "outside")


def _isin(values, target):
    """True where a unit's value, or any item of it, is in target

    Columns of the units table may hold a list per unit (e.g. the curation
    labels), which np.isin cannot compare against target directly.
    """
    target = np.atleast_1d(target)

    if _is_list_valued(values):
        return np.array(
            [np.isin(np.atleast_1d(value), target).any() for value in values]
        )

    return np.isin(values, target)


def _is_list_valued(values):
    """True where a column holds a list per unit, rather than one value"""
    return values.dtype == object and any(
        isinstance(value, (list, tuple, np.ndarray)) for value in values
    )


def filter_units_by_criteria(
    units_df,
    unit_criteria: dict | None = None,
    strict: bool = True,
) -> np.ndarray:
    """Filter units on arbitrary columns of the units table

    Parameters
    ----------
    units_df : pd.DataFrame
        units table of one sorting, one row per unit
    unit_criteria : dict, optional
        column name to {operator: value}, or to a bare value or list of
        values as shorthand for {"isin": value}. A unit is included only if
        it satisfies every criterion. Operators are ">", ">=", "<", "<=",
        "==", "!=", "between" (matches the inclusive [low, high] pair) and
        "outside" (its exact complement), "isin" and "notin". By default
        None, which includes every unit.
    strict : bool, optional
        by default True, raise if a criterion names a column this units
        table does not have. If False, skip that criterion with a warning
        and apply the rest, which lets **every** unit of this sorting pass
        it. `fetch_spike_data` passes False so that it can report every
        sorting in the group missing the column, then raises itself.

    Returns
    -------
    np.ndarray
        boolean mask of shape (n_units,), True for each unit satisfying
        all criteria

    Raises
    ------
    ValueError
        if a criterion holds no operator at all, if an operator is not one
        of those listed above, if "between" or "outside" is given anything
        but a [low, high] pair, if any operator but "isin" or "notin" is
        applied to a column holding a list per unit, or, when strict, if a
        criterion names a column not in units_df

    Notes
    -----
    Units missing a value (NaN, pd.NA, or None from an imported
    units table) fail every criterion on that column, negated ones
    ("!=", "notin") included: a metric that was never computed is no
    evidence that the unit is good. "isin" and "notin" also work on columns
    holding a list per unit (e.g. the curation labels), matching if any
    item of the list is in the target. An empty list is a value, not a
    missing one, so a unit carrying no labels passes "notin".

    Each sorting in a group has its own units table, and those tables may
    not share the same columns (e.g. if they were curated differently), so
    a criterion may apply to only some of them. Passing a criteria column
    that is not in the units table raises an error. Pass strict=False to
    skip criteria on missing columns (passing all units for that criterion)
    with a warning instead of erroring.
    """
    include_mask = np.ones(len(units_df), dtype=bool)

    for column, criterion in (unit_criteria or {}).items():
        if column not in units_df:
            if strict:
                raise ValueError(
                    f"Unit criteria column '{column}' not in units table. "
                    + f"Columns are {list(units_df.columns)}. Pass "
                    + "strict=False to skip criteria on missing columns."
                )
            logger.warning(
                f"Unit criteria column '{column}' not in units table. "
                + "Skipping this criterion: every unit of this sorting "
                + "passes it unfiltered."
            )
            continue

        if not isinstance(criterion, dict):  # shorthand for membership
            criterion = {"isin": criterion}
        if not criterion:
            raise ValueError(
                f"Empty criterion for column '{column}': no operator to "
                + "apply. Drop the column from the criteria to include "
                + "every unit."
            )
        values = units_df[column].to_numpy()
        present = units_df[column].notna().to_numpy(dtype=bool)
        include_mask &= present
        # Nullable pandas scalars cannot participate in NumPy comparisons.
        # Evaluate only present values; missing evidence fails every operator.
        values = values[present]

        for operator, value in criterion.items():
            if operator not in UNIT_CRITERIA_OPERATORS:
                raise ValueError(
                    f"Invalid unit criteria operator '{operator}' for "
                    + f"column '{column}'. Expected one of "
                    + f"{list(UNIT_CRITERIA_OPERATORS)}"
                )
            if operator in RANGE_OPERATORS and (
                not isinstance(value, (list, tuple, np.ndarray))
                or len(value) != 2
            ):
                raise ValueError(
                    f"Unit criteria operator '{operator}' for column "
                    + f"'{column}' takes a [low, high] pair, got "
                    + f"{value!r}."
                )
            if operator in SCALAR_ONLY_OPERATORS and _is_list_valued(values):
                raise ValueError(
                    f"Unit criteria operator '{operator}' cannot filter "
                    + f"column '{column}', which holds a list per unit. "
                    + "Comparing a list to the criterion would mask every "
                    + "unit in or out at once. Use 'isin' or 'notin', "
                    + "which match any item of the list."
                )
            include_mask[present] &= UNIT_CRITERIA_OPERATORS[operator](
                values, value
            )

    return include_mask

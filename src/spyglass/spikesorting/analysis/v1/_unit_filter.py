"""Label-based unit filtering shared by ``SortedSpikesGroup`` and the v2 handoff.

DB-free: ``SortedSpikesGroup.filter_units`` delegates here so the v2
``select_units_for_analysis`` receipt can compute its verdicts with the exact
function ``fetch_spike_data`` applies, without importing the schema module.
"""

from __future__ import annotations

import numpy as np


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

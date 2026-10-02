"""Merge-group bookkeeping shared by the v1 and v2 curation pipelines.

Curation stores merges either as a list of merge groups
(``[[1, 2, 3], [4, 5]]``) or as a per-unit association map
(``{1: [2, 3], 4: [5]}``); these helpers convert the map form into
connected-component merge groups.

This module uses only the standard library: it declares no ``dj.schema`` and
imports no schema module, so database-free code can import it.
"""

from typing import List


def _union_intersecting_lists(lists):
    """Union groups that share any member (connected components).

    Each output group is a ``list`` built from a ``set``, so member order
    within a group is not defined. Groups appear in the order of their first
    input list.
    """
    result = []

    while lists:
        first, *rest = lists
        first = set(first)

        merged = True
        while merged:
            merged = False
            for idx, other in enumerate(rest):
                if first.intersection(other):
                    first.update(other)
                    del rest[idx]
                    merged = True
                    break

        result.append(list(first))
        lists = rest

    return result


def _reverse_associations(assoc_dict):
    """Turn a unit-association map into one candidate group per key.

    Parameters
    ----------
    assoc_dict : dict
        Keys are unit IDs; values are lists of the units associated with
        that key (empty or ``None`` for none).

    Returns
    -------
    list of list
        ``[key, *values]`` for each key, in key order.
    """
    return [
        [key] + values if values else [key]
        for key, values in assoc_dict.items()
    ]


def _merge_dict_to_list(merge_groups: dict) -> List:
    """Converts dict of merge groups to list of merge groups.
    Undoes `spyglass.spikesorting.v1.curation._list_to_merge_dict`.

    Parameters
    ----------
    merge_groups : dict
        dict of merge groups;
        keys are unit IDs and values are the units to be merged

    Returns
    -------
    merge_group_list : list of list
        list of merge groups (list of unit IDs to be merged)

    Example
    -------
    {1: [2, 3], 4: [5]} -> [[1, 2, 3], [4, 5]]
    """
    units_to_merge = _union_intersecting_lists(
        _reverse_associations(merge_groups)
    )
    return [lst for lst in units_to_merge if len(lst) >= 2]

"""DB-free helpers for ``UnitAnnotation`` spike selection.

Isolated from :mod:`unit_annotation` (which activates a DataJoint schema on
import) so the per-id validation is unit-testable without a database or the
JAX-dependent v1 integration fixture. Imports only the standard library.
"""

from __future__ import annotations


def spikes_for_requested_units(
    unit_id_to_spike_times: dict, requested_unit_ids, merge_id
) -> list:
    """Select spike-time arrays for ``requested_unit_ids`` by TRUE unit id.

    ``UnitAnnotation.unit_id`` is the NWB units-table id, not a positional index
    into the spike-times list: positional indexing is wrong for the sparse /
    non-contiguous id sets that v2 and merge-applied v0/v1 curations produce (a
    v1 merge assigns ``max(ids)+1`` and drops constituents). This looks each
    requested id up in the true-id -> spike-times map and raises an actionable
    error -- naming the valid ids and the positional->true-id change -- instead
    of a bare ``KeyError`` when a row written under the older positional contract
    misses on such a curation.

    Parameters
    ----------
    unit_id_to_spike_times : dict
        Map from the NWB's true unit id to that unit's spike-times array.
    requested_unit_ids : iterable
        The annotated unit ids to select (``int`` or integer-like).
    merge_id
        The ``spikesorting_merge_id`` the map belongs to, named in the error.

    Returns
    -------
    list
        The spike-times arrays for ``requested_unit_ids``, in order.

    Raises
    ------
    ValueError
        If a requested id is not a true unit id of ``merge_id``.
    """
    selected = []
    for unit_id in requested_unit_ids:
        key = int(unit_id)
        if key not in unit_id_to_spike_times:
            valid = sorted(int(k) for k in unit_id_to_spike_times)
            raise ValueError(
                f"UnitAnnotation.fetch_unit_spikes: unit_id {unit_id} is not a "
                f"unit id of spikesorting_merge_id {merge_id} (valid unit ids: "
                f"{valid}). UnitAnnotation.unit_id is the NWB units-table id, "
                "not a positional index into the spike-times list; a row "
                "written under the older positional contract can miss or "
                "mis-resolve on a sparse or merge-applied curation. Re-derive "
                "the annotation against the current true unit ids."
            )
        selected.append(unit_id_to_spike_times[key])
    return selected

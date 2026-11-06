"""DB-free helpers for ``UnitAnnotation`` spike selection.

Kept apart from :mod:`unit_annotation`, which activates a schema on import,
so it is testable without a database.
"""

from __future__ import annotations


def spikes_for_requested_units(
    unit_id_to_spike_times: dict, requested_unit_ids, merge_id
) -> list:
    """Select spike-time arrays for ``requested_unit_ids`` by NWB unit id.

    ``UnitAnnotation.unit_id`` is the NWB units-table id, not a position.
    Merged curations have sparse ids, so a positional id may be missing.

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

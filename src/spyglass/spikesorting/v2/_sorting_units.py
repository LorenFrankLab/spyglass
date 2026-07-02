"""``Sorting.Unit`` row construction behind ``Sorting``.

``build_sorting_unit_rows`` turns the per-unit peak metadata that
``Sorting._populate_unit_part`` fetched into the ``Sorting.Unit`` rows;
``_to_int_unit_id`` coerces a sorter's unit id to the int PK v2 stores (raising
the typed ``NonIntegerUnitIDError`` when it cannot). Pure (DB-free) row
construction -- the analyzer load, the peak/sort-group/electrode fetches, and
the ``Sorting.Unit.insert`` stay in the table class.

Why this lives in its own module rather than in ``sorting.py``:
``sorting.py`` is a DataJoint *schema* module -- importing it activates
``dj.schema(...)`` and the source-part / merge dependencies. This row
construction needs none of that at import, so ``Sorting`` stays a thin
orchestrator (fetch -> call these -> insert). Same "thin DataJoint shell over
pure/IO services" direction as ``_artifact_compute`` / ``_selection_identity``
/ ``_analyzer_cache`` / ``_curation_transforms`` / ``_units_nwb`` /
``_sorting_dispatch`` / ``_sorting_artifact_mask`` / ``_sorting_analyzer``.

DB-FREE AT IMPORT. This module activates no ``dj.schema`` and opens no DB
connection at import: ``_to_int_unit_id`` lazy-imports only the typed
``NonIntegerUnitIDError``. Neither function touches the DB at call time.
"""

from __future__ import annotations


def _to_int_unit_id(unit_id):
    """Coerce a sorter unit id to int, or raise the typed NonIntegerUnitIDError.

    v2's ``Sorting.Unit`` PK stores int unit ids; a sorter that emits a
    non-convertible id (e.g. a string label like ``"noise_3"``) must be
    remapped before insertion. Raising the typed error -- not a bare
    ``int()`` ValueError -- lets callers and tests discriminate this case.
    """
    from spyglass.spikesorting.v2.exceptions import NonIntegerUnitIDError

    try:
        return int(unit_id)
    except (TypeError, ValueError) as exc:
        raise NonIntegerUnitIDError(
            f"Sorting.make: sorter returned unit_id {unit_id!r} that does "
            "not convert to int. v2's Sorting.Unit stores int unit_ids; "
            "remap before insertion if the sorter emits non-convertible IDs."
        ) from exc


def build_sorting_unit_rows(
    unit_ids,
    peak_channels,
    peak_amplitudes,
    n_spikes_by_unit,
    electrode_by_id,
    key,
    *,
    sort_group_id,
    nwb_file_name,
) -> list[dict]:
    """Build the ``Sorting.Unit`` rows from per-unit peak metadata.

    Pure (DB-free) row construction extracted from
    ``Sorting._populate_unit_part``; the analyzer load, ``peak_sign`` /
    sort-group / electrode fetches, and the ``Sorting.Unit.insert`` stay in the
    table class. Each unit becomes one row carrying the peak channel's Electrode
    FK fields (resolved through ``electrode_by_id``), the peak template
    amplitude in microvolts, and the precomputed spike count, merged onto
    ``key``.

    Parameters
    ----------
    unit_ids : iterable
        The sorter's unit ids (``sorting.unit_ids``).
    peak_channels : mapping
        ``{unit_id: electrode_id}`` of each unit's peak channel.
    peak_amplitudes : mapping
        ``{unit_id: amplitude_uv}`` peak template amplitude per unit.
    n_spikes_by_unit : mapping
        ``{unit_id: n_spikes}`` precomputed spike count per unit.
    electrode_by_id : mapping
        ``{electrode_id: row}`` of the sort group's ``SortGroupElectrode``
        rows; each row must carry ``nwb_file_name``, ``electrode_group_name``,
        and ``electrode_id``.
    key : dict
        Base primary key merged onto every row.
    sort_group_id : int
        Sort group id, used only in the channel-mismatch error message.
    nwb_file_name : str
        Session file name, used only in the channel-mismatch error message.

    Returns
    -------
    list of dict
        One ``Sorting.Unit`` row per unit.

    Raises
    ------
    NonIntegerUnitIDError
        If a ``unit_id`` does not convert to int.
    RuntimeError
        If a unit's peak channel is not in ``electrode_by_id`` (a sort group /
        recording channel-id mismatch).
    """
    rows = []
    for unit_id in unit_ids:
        int_unit_id = _to_int_unit_id(unit_id)
        peak_chan = int(peak_channels[unit_id])
        if peak_chan not in electrode_by_id:
            raise RuntimeError(
                f"Sorting.make: peak channel {peak_chan} for unit "
                f"{int_unit_id} is not in sort group "
                f"{sort_group_id} for {nwb_file_name!r}. Sort group "
                "/ recording channel-id mismatch."
            )
        rows.append(
            {
                **key,
                "unit_id": int_unit_id,
                **{
                    k: electrode_by_id[peak_chan][k]
                    for k in (
                        "nwb_file_name",
                        "electrode_group_name",
                        "electrode_id",
                    )
                },
                "peak_amplitude_uv": float(peak_amplitudes[unit_id]),
                "n_spikes": int(n_spikes_by_unit[unit_id]),
            }
        )
    return rows

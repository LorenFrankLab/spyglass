"""``Sorting.Unit`` row construction behind ``Sorting``."""

from __future__ import annotations


def _to_int_unit_id(unit_id):
    """Coerce a sorter unit id to int, or raise the typed NonIntegerUnitIDError.

    ``Sorting.Unit`` stores int unit ids.
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

    Each row carries the peak channel's Electrode FK fields, the peak
    template amplitude in µV, and the spike count, merged onto ``key``.

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


def build_unit_rows_from_analyzer(
    *,
    sorting,
    analyzer_folder,
    sorter_row,
    electrode_by_id,
    sort_group_id,
    nwb_file_name,
    key,
):
    """Build the ``Sorting.Unit`` rows from the freshly built analyzer.

    Peak channel and amplitude use the sorter's detection polarity, not
    SI's ``"neg"`` default. The rows feed both the NWB unit columns and the
    ``Sorting.Unit`` insert. Empty for a zero-unit sort, which has no
    analyzer folder.
    """
    if sorting.get_num_units() == 0:
        return []

    from spikeinterface.core import template_tools

    from spyglass.spikesorting.v2._storage.analyzer_cache import (
        load_analyzer_folder,
    )
    from spyglass.spikesorting.v2._core.signal_math import resolve_peak_sign

    analyzer = load_analyzer_folder(analyzer_folder)
    peak_sign = resolve_peak_sign(sorter_row["params"])
    peak_channels = template_tools.get_template_extremum_channel(
        analyzer, peak_sign=peak_sign, outputs="id"
    )
    # SI's default mode="at_index" under-reports the peak.
    peak_amplitudes = template_tools.get_template_extremum_amplitude(
        analyzer, peak_sign=peak_sign, mode="extremum"
    )
    return build_sorting_unit_rows(
        unit_ids=sorting.unit_ids,
        peak_channels=peak_channels,
        peak_amplitudes=peak_amplitudes,
        n_spikes_by_unit=sorting.count_num_spikes_per_unit(),
        electrode_by_id=electrode_by_id,
        key=key,
        sort_group_id=sort_group_id,
        nwb_file_name=nwb_file_name,
    )

"""``Sorting.Unit`` row construction behind ``Sorting``.

``build_unit_rows_from_analyzer`` loads the analyzer ``Sorting.make_compute``
just built and resolves each unit's peak channel, peak amplitude and spike
count; ``build_sorting_unit_rows`` turns that per-unit peak metadata into the
``Sorting.Unit`` rows; ``_to_int_unit_id`` coerces a sorter's unit id to the
int PK v2 stores (raising the typed ``NonIntegerUnitIDError`` when it cannot).
Pure (DB-free) row construction -- the sort-group/electrode fetches
(``_sorting_fetch.fetch_unit_electrode_metadata``) and the
``Sorting.Unit.insert`` live outside this module.

DB-FREE AT IMPORT. This module activates no ``dj.schema`` and opens no DB
connection at import: ``build_unit_rows_from_analyzer`` imports
SpikeInterface and the analyzer loader inside the function, and
``_to_int_unit_id`` lazy-imports the typed ``NonIntegerUnitIDError``. No
function touches the DB at call time.
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

    Pure (DB-free) row construction, called by
    :func:`build_unit_rows_from_analyzer` with the peak metadata it resolved;
    the sort-group / electrode fetches and the ``Sorting.Unit.insert`` live
    outside this module. Each unit becomes one row carrying the peak channel's
    Electrode FK fields (resolved through ``electrode_by_id``), the peak
    template amplitude in microvolts, and the precomputed spike count, merged
    onto ``key``.

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

    Run once in ``make_compute`` after upstream inputs are resolved in
    ``make_fetch``; this helper performs no DB writes. The analyzer folder
    :func:`._sorting.analyzer.build_analyzer` just wrote is loaded here, each unit's peak channel
    + amplitude is resolved under the sorter's configured detection
    polarity (clusterless ``peak_sign`` / MountainSort ``detect_sign``, not
    SI's ``"neg"`` default, so a positive-going detection attributes each
    unit to its true peak channel), and the rows are assembled by
    ``build_sorting_unit_rows`` (which raises on a sort-group/recording
    channel-id mismatch). The resulting rows are reused for BOTH the NWB
    unit columns and the ``Sorting.Unit`` insert, so the file and the DB
    cannot drift.

    Empty for a zero-unit sort: ``build_analyzer`` skips the
    ``create_sorting_analyzer`` call when ``sorting.get_num_units() == 0``
    (SI's ``estimate_sparsity`` crashes on empty sortings), so the analyzer
    folder does not exist; there is nothing to load or insert.
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
    # ``mode="extremum"`` measures the amplitude at the template PEAK
    # (matching ``get_template_extremum_channel`` above), not SI's
    # ``mode="at_index"`` default which reads the alignment-sample value
    # -- that under-reports the peak AND can land on a different channel
    # than the attributed electrode. ``abs_value=True`` (SI default)
    # returns the non-negative magnitude ``peak_amplitude_uv`` stores.
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

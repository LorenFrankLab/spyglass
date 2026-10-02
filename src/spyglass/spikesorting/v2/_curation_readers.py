"""Spike-train readers behind ``CurationV2``.

:func:`get_sorting` and :func:`get_merged_sorting` are the bodies of
``CurationV2.get_sorting`` / ``get_merged_sorting``: they read a curation's
curated-units NWB against its upstream recording's sampling rate and
timestamps (:func:`load_curation_recording_meta`, :func:`upstream_recording_row`)
and, for an unapplied proposed merge, rebuild the merged sorting lazily.
``table_cls`` is ``CurationV2``.

Imports without the DB layer: the DataJoint tables are imported inside the
functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from spyglass.spikesorting.v2._signal_math import _MERGE_DEDUP_DELTA_MS
from spyglass.spikesorting.v2._units_nwb import (
    abs_spike_times_dataframe,
    build_lazy_merged_sorting,
    build_lazy_merged_sorting_from_samples,
    empty_spike_times_dataframe,
    read_units_abs_spike_times,
    read_units_abs_times_and_sample_indices,
    recording_timestamps,
    sorting_from_units_nwb,
)

if TYPE_CHECKING:
    import pandas as pd
    import spikeinterface as si


def get_sorting(
    table_cls, key: dict, as_dataframe: bool = False
) -> si.BaseSorting | pd.DataFrame:
    """Return a curation's units as a sorting or a spike-times DataFrame.

    The body of ``CurationV2.get_sorting`` (see its docstring).
    """
    import spikeinterface as si

    from spyglass.utils import logger

    row, recording_row, fs, abs_path = load_curation_recording_meta(
        table_cls, key
    )

    # A curation created with apply_merge=False records PROPOSED merges in
    # MergeGroup but does NOT apply them: get_sorting (what consumers such
    # as SortedSpikesGroup / decoding read via SpikeSortingOutput) returns
    # the UNMERGED preview units. Warn here so ad-hoc inspection is not
    # silently misled; the decoding CONSUMERS additionally RAISE via
    # SpikeSortingOutput.assert_decoding_merge_ids_ok so a preview curation
    # never reaches a decode.
    if table_cls.has_unapplied_proposed_merges(
        key, merges_applied=row["merges_applied"]
    ):
        logger.warning(
            "CurationV2.get_sorting: curation "
            f"(sorting_id={row['sorting_id']}, "
            f"curation_id={row['curation_id']}) has proposed merges that "
            "are NOT applied (apply_merge=False); this returns the "
            "UNMERGED units. Use get_merged_sorting to apply the proposal, "
            "or re-curate with apply_merge=True to commit it."
        )

    if len(table_cls.Unit & key) == 0:
        # Zero-unit curations are valid (a user may curate a
        # zero-unit sort; the Empty/Boundary invariant allows an
        # empty ``CurationV2.Unit``).
        logger.warning(
            "CurationV2.get_sorting: curation "
            f"(sorting_id={row['sorting_id']}, "
            f"curation_id={row['curation_id']}) has zero units; "
            "returning an empty sorting."
        )
        if not as_dataframe:
            return si.NumpySorting.from_unit_dict({}, sampling_frequency=fs)
        df = empty_spike_times_dataframe()
        df["curation_label"] = []
        return df

    if not as_dataframe:
        return sorting_from_units_nwb(
            abs_path, fs, lambda: recording_timestamps(recording_row)
        )

    abs_times = read_units_abs_spike_times(abs_path)
    # Reuse the shared spike-times DataFrame builder (the same one
    # Sorting.get_sorting(as_dataframe=True) uses) so the base spike_times
    # column + unit_id index cannot drift between the two, then join the
    # ``curation_label`` lists from ``UnitLabel`` so external notebook code
    # reading ``df["curation_label"]`` works without poking the part table.
    labels_by_unit = table_cls._labels_by_unit(key)
    df = abs_spike_times_dataframe(abs_times)
    df["curation_label"] = [labels_by_unit.get(u, []) for u in df.index]
    return df


def get_merged_sorting(table_cls, key: dict) -> si.BaseSorting:
    """Return a curation's units with its proposed merges applied.

    The body of ``CurationV2.get_merged_sorting`` (see its docstring).
    """
    # merges_applied OR no multi-contributor group -> the base sorting
    # already IS the result; delegate to get_sorting (its read is
    # unavoidable in those cases).
    if bool((table_cls & key).fetch1("merges_applied")):
        return table_cls.get_sorting(key)
    merge_groups = table_cls.get_unit_contributor_groups(key)
    units_to_merge = [
        contribs
        for kept_uid, contribs in merge_groups.items()
        if len(contribs) > 1
    ]
    if not units_to_merge:
        return table_cls.get_sorting(key)

    # Read the curated units NWB once, then rebuild the merged sorting via
    # the pure compute core. v2-written units NWBs carry stored sample
    # frames and avoid the recording timeline; older/manual files fall
    # back to the full timestamp-vector mapping. Calling get_sorting here
    # would re-open the units NWB and emit a spurious "merges NOT applied"
    # warning -- we ARE applying them. The merge is deduplicated in ABSOLUTE time
    # (gap-correct on disjoint recordings); see the helper docstrings.
    _row, recording_row, fs, abs_path = load_curation_recording_meta(
        table_cls, key
    )
    # Both columns are needed (dedup is in absolute time, frames are reused
    # when present) -- read them from a single NWB open.
    abs_times, sample_indices, _obs = read_units_abs_times_and_sample_indices(
        abs_path
    )
    if sample_indices is not None:
        return build_lazy_merged_sorting_from_samples(
            abs_times,
            sample_indices,
            units_to_merge,
            fs,
            delta_s=_MERGE_DEDUP_DELTA_MS / 1000.0,
        )
    timestamps = recording_timestamps(recording_row)
    return build_lazy_merged_sorting(
        abs_times,
        units_to_merge,
        timestamps,
        fs,
        delta_s=_MERGE_DEDUP_DELTA_MS / 1000.0,
    )


def load_curation_recording_meta(table_cls, key):
    """Fetch the master row + upstream recording metadata for a curation.

    Returns ``(row, recording_row, fs, abs_path)``: the ``CurationV2``
    master row, the upstream ``Recording`` row, its sampling frequency,
    and the curated-units NWB path. Shared by ``get_sorting`` and
    ``get_merged_sorting`` so the master row is fetched ONCE and its
    ``sorting_id`` threaded into ``upstream_recording_row`` (skipping a
    redundant lookup). The curated-units NWB itself is NOT read here --
    callers read it lazily so ``get_sorting`` can short-circuit a
    zero-unit curation without touching the filesystem.
    """
    from spyglass.common.common_nwbfile import AnalysisNwbfile

    row = (table_cls & key).fetch1()
    recording_row = upstream_recording_row(
        table_cls, key, sorting_id=row["sorting_id"]
    )
    fs = float(recording_row["sampling_frequency"])
    abs_path = AnalysisNwbfile.get_abs_path(row["analysis_file_name"])
    return row, recording_row, fs, abs_path


def upstream_recording_row(table_cls, key, *, sorting_id=None) -> dict:
    """Fetch the upstream Recording row for a CurationV2 key.

    Used by ``get_sorting`` to recover the recording's
    sampling-frequency and timestamp metadata (matching the
    ``Sorting.get_sorting`` round-trip convention). For a concat-backed
    sort this is the ``ConcatenatedRecording`` row (the timeline the curated
    spike times were written against), NOT a per-member ``Recording``.

    ``key`` may be a single dict or the list-of-dict form the
    merge dispatcher passes; the restriction-based fetch
    normalizes both. Pass ``sorting_id`` when the caller already
    holds the master row to skip the redundant ``sorting_id`` lookup.
    """
    from spyglass.spikesorting.v2.recording import Recording
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecording,
    )
    from spyglass.spikesorting.v2.sorting import SortingSelection

    if sorting_id is None:
        sorting_id = (table_cls & key).fetch1("sorting_id")
    source = SortingSelection.resolve_source({"sorting_id": sorting_id})
    if source.kind == "recording":
        return (Recording & source.key).fetch1()
    return (ConcatenatedRecording & source.key).fetch1()

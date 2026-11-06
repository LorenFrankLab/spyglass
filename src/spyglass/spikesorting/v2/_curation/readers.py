"""Spike-train readers behind ``CurationV2``.

:func:`get_sorting` and :func:`get_merged_sorting` are the bodies of
``CurationV2.get_sorting`` / ``get_merged_sorting``: they read a curation's
curated-units NWB using stored sample frames and the upstream sampling rate
(:func:`load_curation_units_meta`) and, for an unapplied proposed merge,
rebuild the merged sorting lazily.
``table_cls`` is ``CurationV2``.

Imports without the DB layer: the DataJoint tables are imported inside the
functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from spyglass.spikesorting.v2._core.signal_math import _MERGE_DEDUP_DELTA_MS
from spyglass.spikesorting.v2._storage.units_nwb import (
    abs_spike_times_dataframe,
    build_lazy_merged_sorting_from_samples,
    empty_spike_times_dataframe,
    read_units_abs_spike_times,
    read_units_abs_times_and_sample_indices,
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

    row = (table_cls & key).fetch1()
    fs, abs_path = load_curation_units_meta(row)

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
        return sorting_from_units_nwb(abs_path, fs)

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
    row = (table_cls & key).fetch1()
    if bool(row["merges_applied"]):
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
    # frames and avoid the recording timeline. Calling get_sorting here
    # would re-open the units NWB and emit a spurious "merges NOT applied"
    # warning -- we ARE applying them. The merge is deduplicated in ABSOLUTE time
    # (gap-correct on disjoint recordings); see the helper docstrings.
    fs, abs_path = load_curation_units_meta(row)
    # Both columns are needed (dedup is in absolute time, frames are reused
    # directly) -- read them from a single NWB open.
    abs_times, sample_indices, _obs = read_units_abs_times_and_sample_indices(
        abs_path
    )
    return build_lazy_merged_sorting_from_samples(
        abs_times,
        sample_indices,
        units_to_merge,
        fs,
        delta_s=_MERGE_DEDUP_DELTA_MS / 1000.0,
    )


def load_curation_units_meta(row) -> tuple[float, str]:
    """Resolve a curation's sampling frequency and curated-units NWB path.

    Parameters
    ----------
    row : Mapping
        The ``CurationV2`` master row, already fetched by the caller.

    Returns
    -------
    fs : float
        The upstream recording's sampling frequency (matching the
        ``Sorting.get_sorting`` round-trip convention).
    abs_path : str
        The curated-units NWB path. The file itself is NOT read here --
        callers read it lazily so ``get_sorting`` can short-circuit a
        zero-unit curation without touching the filesystem.
    """
    from spyglass.common.common_nwbfile import AnalysisNwbfile

    fs = _source_sampling_frequency(row["sorting_id"])
    abs_path = AnalysisNwbfile.get_abs_path(row["analysis_file_name"])
    return fs, abs_path


def _source_sampling_frequency(sorting_id) -> float:
    """Return the sampling frequency of a sort's recording source.

    For a concat-backed sort this is the ``ConcatenatedRecording`` (the
    timeline the curated spike times were written against), NOT a
    per-member ``Recording``.
    """
    from spyglass.spikesorting.v2.recording import Recording
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecording,
    )
    from spyglass.spikesorting.v2.sorting import SortingSelection

    source = SortingSelection.resolve_source({"sorting_id": sorting_id})
    table = Recording if source.kind == "recording" else ConcatenatedRecording
    return float((table & source.key).fetch1("sampling_frequency"))

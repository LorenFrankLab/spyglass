"""get_sorting warns when a curation has unapplied (preview) merges.

A curation created with apply_merge=False records PROPOSED merges in MergeGroup
but does not apply them. Consumers (SortedSpikesGroup / decoding) read units
through CurationV2.get_sorting, which returns the UNMERGED units for such a
curation -- silently using oversplit units. get_sorting now warns (and the
docstring documents the preview semantics). get_merged_sorting still applies
the proposal. A plain root curation (no real merge, only 1-element self-entries)
must NOT warn.
"""

from __future__ import annotations

import pytest

from tests.spikesorting.v2._ingest_helpers import clear_curations_for

# ``planted_two_unit_sort`` lives in conftest.py (shared across the merge-aware
# test modules; see that fixture's docstring).


@pytest.mark.slow
@pytest.mark.integration
def test_get_sorting_warns_on_unapplied_preview_merge(
    planted_two_unit_sort, monkeypatch
):
    import spyglass.spikesorting.v2.curation as curation_mod
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting

    unit_ids = sorted(
        int(u) for u in (Sorting.Unit & planted_two_unit_sort).fetch("unit_id")
    )
    clear_curations_for(planted_two_unit_sort)
    cur = CurationV2.insert_curation(
        sorting_key=planted_two_unit_sort,
        merge_groups=[[unit_ids[0], unit_ids[1]]],
        apply_merge=False,
    )

    seen: list[str] = []
    monkeypatch.setattr(
        curation_mod.logger,
        "warning",
        lambda msg, *a, **k: seen.append(str(msg)),
    )
    try:
        sorting = CurationV2.get_sorting(cur)
        # Preview path returns the UNMERGED units (every original unit kept).
        assert sorting.get_num_units() == len(unit_ids)
        assert any(
            "proposed merges that are NOT applied" in s for s in seen
        ), f"expected an unapplied-preview-merge warning; got {seen}"
    finally:
        clear_curations_for(planted_two_unit_sort)


@pytest.mark.slow
@pytest.mark.integration
def test_get_sorting_no_warning_without_proposed_merges(
    planted_two_unit_sort, monkeypatch
):
    """A plain root curation (no real merge) must NOT emit the preview warning."""
    import spyglass.spikesorting.v2.curation as curation_mod
    from spyglass.spikesorting.v2.curation import CurationV2

    clear_curations_for(planted_two_unit_sort)
    cur = CurationV2.insert_curation(sorting_key=planted_two_unit_sort)

    seen: list[str] = []
    monkeypatch.setattr(
        curation_mod.logger,
        "warning",
        lambda msg, *a, **k: seen.append(str(msg)),
    )
    try:
        CurationV2.get_sorting(cur)
        assert not any(
            "proposed merges that are NOT applied" in s for s in seen
        ), f"root curation should not warn about preview merges; got {seen}"
    finally:
        clear_curations_for(planted_two_unit_sort)


@pytest.mark.slow
@pytest.mark.integration
def test_assert_decoding_merge_ids_raises_on_preview_merge(
    planted_two_unit_sort,
):
    """The consumer-boundary validator RAISES on an apply_merge=False curation
    with proposed merges (the preview get_sorting only warns about)."""
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting

    unit_ids = sorted(
        int(u) for u in (Sorting.Unit & planted_two_unit_sort).fetch("unit_id")
    )
    clear_curations_for(planted_two_unit_sort)
    cur = CurationV2.insert_curation(
        sorting_key=planted_two_unit_sort,
        merge_groups=[[unit_ids[0], unit_ids[1]]],
        apply_merge=False,
    )
    merge_id = (SpikeSortingOutput.CurationV2 & cur).fetch1("merge_id")
    try:
        with pytest.raises(ValueError, match="apply_merge=False"):
            SpikeSortingOutput.assert_decoding_merge_ids_ok([merge_id])
    finally:
        clear_curations_for(planted_two_unit_sort)


@pytest.mark.slow
@pytest.mark.integration
def test_get_spike_times_warns_on_preview_merge(
    planted_two_unit_sort, monkeypatch
):
    """R3 (generic accessor): get_spike_times WARNS when a consumed merge_id is
    a preview (apply_merge=False) curation with unapplied proposed merges; an
    applied-merge curation does NOT warn. Warning (not raising) keeps v0/v1
    consumers working; the strict raise stays on the decoding boundary.
    """
    import spyglass.spikesorting.spikesorting_merge as merge_mod
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting

    unit_ids = sorted(
        int(u) for u in (Sorting.Unit & planted_two_unit_sort).fetch("unit_id")
    )
    clear_curations_for(planted_two_unit_sort)
    root = CurationV2.insert_curation(sorting_key=planted_two_unit_sort)
    preview = CurationV2.insert_curation(
        sorting_key=planted_two_unit_sort,
        parent_curation_id=root["curation_id"],
        merge_groups=[[unit_ids[0], unit_ids[1]]],
        apply_merge=False,
    )
    applied = CurationV2.insert_curation(
        sorting_key=planted_two_unit_sort,
        parent_curation_id=root["curation_id"],
        merge_groups=[[unit_ids[0], unit_ids[1]]],
        apply_merge=True,
    )
    preview_id = (SpikeSortingOutput.CurationV2 & preview).fetch1("merge_id")
    applied_id = (SpikeSortingOutput.CurationV2 & applied).fetch1("merge_id")

    seen: list[str] = []
    monkeypatch.setattr(
        merge_mod.logger,
        "warning",
        lambda msg, *a, **k: seen.append(str(msg)),
    )
    try:
        SpikeSortingOutput().get_spike_times({"merge_id": preview_id})
        assert any("apply_merge=False" in s for s in seen), (
            f"expected a preview-merge warning; got {seen}"
        )
        seen.clear()
        SpikeSortingOutput().get_spike_times({"merge_id": applied_id})
        assert not any("apply_merge=False" in s for s in seen), (
            f"applied-merge curation must not warn; got {seen}"
        )
    finally:
        clear_curations_for(planted_two_unit_sort)

"""Downstream-consumer smoke tests on a v2 ``merge_id``.

Covers the ``SpikeSortingOutput`` merge-dispatch and per-bin APIs and
``SortedSpikesGroup``, including sparse unit ids.
"""

from __future__ import annotations

import numpy as np
import pytest


def _make_v2_root_curation(populated_sorting):
    """Clear prior curations + insert a fresh root for the smoke sort.

    Returns ``(pk, merge_id)``. Deletes merge masters before curations,
    as DataJoint requires.
    """
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.curation import CurationV2

    curation_keys = (CurationV2 & populated_sorting).fetch("KEY", as_dict=True)
    if curation_keys:
        merge_ids = (SpikeSortingOutput.CurationV2 & curation_keys).fetch(
            "merge_id"
        )
        for mid in merge_ids:
            (SpikeSortingOutput & {"merge_id": mid}).super_delete(warn=False)
    (CurationV2 & populated_sorting).super_delete(warn=False)

    pk = CurationV2.insert_curation(sorting_key=populated_sorting, labels={})
    merge_id = (SpikeSortingOutput.CurationV2 & pk).fetch1("merge_id")
    return pk, merge_id


@pytest.mark.slow
@pytest.mark.integration
def test_get_recording_returns_filtered_recording(populated_sorting):
    """``SpikeSortingOutput.get_recording`` on a v2 merge_id returns
    an SI BaseRecording with ``is_filtered=True``.

    The annotation stops downstream sorters from re-bandpassing.
    """
    import spikeinterface as si

    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput

    _, merge_id = _make_v2_root_curation(populated_sorting)
    rec = SpikeSortingOutput().get_recording({"merge_id": merge_id})
    assert isinstance(rec, si.BaseRecording)
    assert rec.get_annotation("is_filtered") is True


@pytest.mark.slow
@pytest.mark.integration
def test_get_sort_group_info_returns_multi_electrode_relation(
    populated_sorting,
):
    """``SpikeSortingOutput.get_sort_group_info`` returns a relation
    with rows for every electrode in the sort group.
    """
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.recording import (
        RecordingSelection,
        SortGroupV2,
    )
    from spyglass.spikesorting.v2.sorting import SortingSelection

    _, merge_id = _make_v2_root_curation(populated_sorting)
    info = SpikeSortingOutput.get_sort_group_info({"merge_id": merge_id})
    rows = info.fetch(as_dict=True)
    source = SortingSelection.resolve_source(populated_sorting)
    recording_selection = (RecordingSelection & source.key).fetch1()
    expected = set(
        (SortGroupV2.SortGroupElectrode & recording_selection).fetch(
            "electrode_id"
        )
    )
    assert len(expected) > 1, "fixture must distinguish all electrodes from one"
    assert {row["electrode_id"] for row in rows} == expected
    assert len(rows) == len(expected)
    # Joined to BrainRegion, so every electrode's region is reported.
    assert "electrode_id" in rows[0]
    assert "region_name" in rows[0]


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.smoke
def test_get_spike_times_returns_per_unit_arrays(populated_sorting):
    """``SpikeSortingOutput.get_spike_times`` returns per-unit arrays
    in seconds. The clusterless-decoding pipeline iterates this output.
    """
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.sorting import Sorting

    _, merge_id = _make_v2_root_curation(populated_sorting)
    spike_times = SpikeSortingOutput().get_spike_times({"merge_id": merge_id})
    assert isinstance(spike_times, list)
    n_units = int((Sorting & populated_sorting).fetch1("n_units"))
    assert len(spike_times) == n_units
    for arr in spike_times:
        assert isinstance(arr, np.ndarray)
        assert np.all(np.isfinite(arr))


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize(
    "method_name",
    ["get_spike_indicator", "get_firing_rate"],
)
def test_consumer_api_shape_contract(populated_sorting, method_name):
    """The two time-binned consumer APIs return ``(n_time, n_units)``
    arrays that are non-negative and finite wherever the units were
    observed, and ``np.nan`` in the bins they were not.
    """
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.sorting import Sorting

    _, merge_id = _make_v2_root_curation(populated_sorting)
    n_units = int((Sorting & populated_sorting).fetch1("n_units"))
    time_array = np.arange(0.0, 4.0, 0.1)
    merge_key = {"merge_id": merge_id}
    # get_firing_rate has no return_validity flag, so resolve the observed
    # bins once and hold both methods to the same mask.
    valid = SpikeSortingOutput.get_observation_intervals(merge_key).valid_bins(
        time_array
    )
    result = getattr(SpikeSortingOutput(), method_name)(merge_key, time_array)
    assert result.shape == (len(time_array), n_units)
    assert np.all(result[valid] >= 0)
    assert np.all(np.isfinite(result[valid]))
    assert np.isnan(result[~valid]).all()


@pytest.mark.slow
@pytest.mark.integration
def test_consumer_spike_indicator_count_alignment(populated_sorting):
    """Per-unit ``get_spike_indicator`` column sums equal each unit's
    in-window ``get_spike_times`` count.

    Catches column ``j`` being wired to a different unit than
    ``get_spike_times()[j]``. Counts only spikes in ``[time[0], time[-1]]``
    and in observed time.
    """
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput

    _, merge_id = _make_v2_root_curation(populated_sorting)
    merge_key = {"merge_id": merge_id}
    spike_times = SpikeSortingOutput().get_spike_times(merge_key)
    assert len(spike_times) >= 1, "fixture must yield at least one unit"

    time_array = np.arange(0.0, 4.0, 0.1)
    indicator, valid = SpikeSortingOutput().get_spike_indicator(
        merge_key, time_array, return_validity=True
    )
    assert indicator.shape == (len(time_array), len(spike_times))
    assert np.isnan(indicator[~valid]).all()

    observation = SpikeSortingOutput.get_observation_intervals(merge_key)
    min_t, max_t = time_array[0], time_array[-1]
    # Unobserved bins hold NaN and count nothing.
    column_sums = indicator[valid].sum(axis=0)

    def counted(arr):
        kept = arr[(arr >= min_t) & (arr <= max_t) & observation.contains(arr)]
        return int(np.count_nonzero(valid[np.digitize(kept, time_array[1:-1])]))

    for j, arr in enumerate(spike_times):
        in_window = counted(arr)
        assert int(column_sums[j]) == in_window, (
            f"unit {j}: indicator column sum {int(column_sums[j])} != "
            f"in-window observed spike count {in_window}; the per-unit "
            "indicator is misaligned with get_spike_times "
            "(sparse unit_ids mishandled)."
        )
    # Totals must agree too.
    total_in_window = sum(counted(arr) for arr in spike_times)
    assert int(column_sums.sum()) == total_in_window


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.smoke
def test_sorted_spikes_decoding_selection_accepts_v2_merge_id(
    populated_sorting,
):
    """A ``SortedSpikesDecodingSelection`` row keyed on a v2 ``merge_id``
    can be inserted, so the FK chain resolves end to end.

    Chain::

        SortedSpikesDecodingSelection
          -> SortedSpikesGroup
            -> SortedSpikesGroup.Units (spikesorting_merge_id=v2 merge_id)
              -> SpikeSortingOutput (master)
                -> SpikeSortingOutput.CurationV2 (v2 dispatch part)
                  -> CurationV2 (v2 master)

    Insert only: populate needs position data the smoke fixture lacks.
    """
    pytest.importorskip(
        "non_local_detector",
        reason="decoding extras not installed; the FK check is decoding-side.",
    )
    pytest.importorskip("track_linearization")
    from spyglass.decoding.v1.core import DecodingParameters, PositionGroup
    from spyglass.decoding.v1.sorted_spikes import (
        SortedSpikesDecodingSelection,
    )
    from spyglass.spikesorting.analysis.v1.group import (
        SortedSpikesGroup,
        UnitSelectionParams,
    )
    from spyglass.spikesorting.v2.recording import RecordingSelection
    from spyglass.spikesorting.v2.sorting import SortingSelection

    _, merge_id = _make_v2_root_curation(populated_sorting)
    UnitSelectionParams().insert_default()
    # ``contents`` short-circuits on a populated table; use insert_default.
    DecodingParameters.insert_default()

    recording_id = SortingSelection.resolve_source(populated_sorting).key[
        "recording_id"
    ]
    actual_nwb = (RecordingSelection & {"recording_id": recording_id}).fetch1(
        "nwb_file_name"
    )

    sorted_group_name = "v2_decoding_fk_test_sorted_spikes_group"
    existing_group = SortedSpikesGroup & {
        "sorted_spikes_group_name": sorted_group_name,
        "nwb_file_name": actual_nwb,
    }
    if existing_group:
        existing_group.super_delete(warn=False)
    SortedSpikesGroup().create_group(
        group_name=sorted_group_name,
        nwb_file_name=actual_nwb,
        unit_filter_params_name="all_units",
        keys=[{"spikesorting_merge_id": merge_id}],
    )

    pos_group_name = "v2_decoding_fk_test_position_group"
    existing_pos = PositionGroup & {
        "nwb_file_name": actual_nwb,
        "position_group_name": pos_group_name,
    }
    if existing_pos:
        existing_pos.super_delete(warn=False)
    # The FK targets PositionGroup, so no parts are needed.
    PositionGroup.insert1(
        {
            "nwb_file_name": actual_nwb,
            "position_group_name": pos_group_name,
        },
        skip_duplicates=True,
    )

    sorted_param_names = [
        name
        for name in DecodingParameters.fetch("decoding_param_name")
        if "sorted" in name
    ]
    assert (
        sorted_param_names
    ), "installed decoder defaults must expose sorted-spikes parameters"
    decoding_param_name = sorted_param_names[0]

    selection_key = {
        "nwb_file_name": actual_nwb,
        "sorted_spikes_group_name": sorted_group_name,
        "unit_filter_params_name": "all_units",
        "position_group_name": pos_group_name,
        "decoding_param_name": decoding_param_name,
        "encoding_interval": "raw data valid times",
        "decoding_interval": "raw data valid times",
        "estimate_decoding_params": True,
    }
    existing_sel = SortedSpikesDecodingSelection & selection_key
    if existing_sel:
        existing_sel.super_delete(warn=False)
    SortedSpikesDecodingSelection.insert1(selection_key)

    inserted = SortedSpikesDecodingSelection & selection_key
    assert len(inserted) == 1, (
        "SortedSpikesDecodingSelection insert succeeded but the row is "
        "not retrievable -- FK chain landed in an inconsistent state."
    )

    # A non-empty join across every hop proves the chain is consistent.
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.curation import CurationV2

    full_chain = (
        SortedSpikesDecodingSelection
        * SortedSpikesGroup.Units
        * SpikeSortingOutput.CurationV2.proj(spikesorting_merge_id="merge_id")
        * CurationV2
        & selection_key
        & {"spikesorting_merge_id": merge_id}
    )
    assert len(full_chain) >= 1, (
        f"Full FK join from SortedSpikesDecodingSelection -> "
        f"SortedSpikesGroup.Units -> SpikeSortingOutput.CurationV2 "
        f"-> CurationV2 returned zero rows for merge_id={merge_id!r}; "
        "at least one hop in the chain is broken."
    )


@pytest.mark.slow
@pytest.mark.integration
def test_sorted_spikes_group_per_unit_and_mua_firing_rate(populated_sorting):
    """``SortedSpikesGroup`` built from a v2 merge_id supports both
    per-unit and multiunit (MUA) firing-rate readouts.

    Per-unit shape is ``(n_time, n_units)``; MUA is ``(n_time, 1)``, with
    sparse unit ids.
    """
    from spyglass.spikesorting.analysis.v1.group import (
        SortedSpikesGroup,
        UnitSelectionParams,
    )
    from spyglass.spikesorting.v2.recording import RecordingSelection
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection

    _, merge_id = _make_v2_root_curation(populated_sorting)
    UnitSelectionParams().insert_default()

    recording_id = SortingSelection.resolve_source(populated_sorting).key[
        "recording_id"
    ]
    actual_nwb = (RecordingSelection & {"recording_id": recording_id}).fetch1(
        "nwb_file_name"
    )

    group_name = "v2_downstream_consumers_sorted_spikes_group"
    existing = SortedSpikesGroup & {
        "sorted_spikes_group_name": group_name,
        "nwb_file_name": actual_nwb,
    }
    if existing:
        existing.super_delete(warn=False)

    SortedSpikesGroup().create_group(
        group_name=group_name,
        nwb_file_name=actual_nwb,
        unit_filter_params_name="all_units",
        keys=[{"spikesorting_merge_id": merge_id}],
    )
    group_key = {
        "sorted_spikes_group_name": group_name,
        "nwb_file_name": actual_nwb,
        "unit_filter_params_name": "all_units",
    }
    time_array = np.arange(0.0, 4.0, 0.1)
    n_units = int((Sorting & populated_sorting).fetch1("n_units"))

    fr_per_unit = SortedSpikesGroup.get_firing_rate(
        group_key, time_array, multiunit=False
    )
    assert fr_per_unit.shape == (len(time_array), n_units)

    fr_mua = SortedSpikesGroup.get_firing_rate(
        group_key, time_array, multiunit=True
    )
    assert fr_mua.shape == (len(time_array), 1)
    assert np.all(fr_mua >= 0)
    assert np.all(np.isfinite(fr_mua))


@pytest.mark.usefixtures("dj_conn")
def test_filter_units():
    """``SortedSpikesGroup.filter_units`` include/exclude/empty semantics.

    Tested directly because ``fetch_spike_data`` skips the filter under
    pytest (``not test_mode`` in ``analysis/v1/group.py``), which keeps the
    shared fixtures' ``default_exclusion`` groups non-empty.
    """
    import numpy as np

    from spyglass.spikesorting.analysis.v1.group import SortedSpikesGroup

    f = SortedSpikesGroup.filter_units

    # ``all_units`` preset.
    np.testing.assert_array_equal(
        f([["noise"], ["accept"], []], [], []),
        np.array([True, True, True]),
    )

    # ``default_exclusion`` preset.
    np.testing.assert_array_equal(
        f([["noise"], ["accept"], ["mua"], []], [], ["noise", "mua"]),
        np.array([False, True, False, True]),
    )

    # All noise/mua -> all False.
    np.testing.assert_array_equal(
        f([["noise"], ["mua"]], [], ["noise", "mua"]),
        np.array([False, False]),
    )

    # Include filter: keep only units carrying an include label.
    np.testing.assert_array_equal(
        f([["accept"], ["noise"], ["accept", "mua"]], ["accept"], []),
        np.array([True, False, True]),
    )

    # Exclude wins over include.
    np.testing.assert_array_equal(
        f([["accept"], ["accept", "noise"]], ["accept"], ["noise"]),
        np.array([True, False]),
    )

    # A bare-string label (not wrapped in a list) is treated as one label.
    np.testing.assert_array_equal(
        f(["noise", "accept"], [], ["noise"]),
        np.array([False, True]),
    )


@pytest.mark.slow
@pytest.mark.integration
def test_all_unlabeled_curation_include_label_filters(populated_sorting):
    """An include/exclude filter applies even when the curated NWB
    omits the label column (every unit unlabeled).

    ``include_labels=["accept"]`` returns no units;
    ``exclude_labels=["noise"]`` returns all.
    """
    from spyglass.spikesorting.analysis.v1.group import (
        SortedSpikesGroup,
        UnitSelectionParams,
    )
    from spyglass.spikesorting.v2.recording import RecordingSelection
    from spyglass.spikesorting.v2.sorting import SortingSelection

    _pk, merge_id = _make_v2_root_curation(populated_sorting)
    UnitSelectionParams().insert_default()
    UnitSelectionParams().insert1(
        {
            "unit_filter_params_name": "cnep2_include_accept",
            "include_labels": ["accept"],
            "exclude_labels": [],
        },
        skip_duplicates=True,
    )
    UnitSelectionParams().insert1(
        {
            "unit_filter_params_name": "cnep2_exclude_noise",
            "include_labels": [],
            "exclude_labels": ["noise"],
        },
        skip_duplicates=True,
    )

    recording_id = SortingSelection.resolve_source(populated_sorting).key[
        "recording_id"
    ]
    nwb = (RecordingSelection & {"recording_id": recording_id}).fetch1(
        "nwb_file_name"
    )

    def _n_units(filter_name):
        gname = f"cnep2_{filter_name}"
        existing = SortedSpikesGroup & {
            "sorted_spikes_group_name": gname,
            "nwb_file_name": nwb,
        }
        if existing:
            existing.super_delete(warn=False)
        SortedSpikesGroup().create_group(
            group_name=gname,
            nwb_file_name=nwb,
            unit_filter_params_name=filter_name,
            keys=[{"spikesorting_merge_id": merge_id}],
        )
        spike_times = SortedSpikesGroup.fetch_spike_data(
            {"sorted_spikes_group_name": gname, "nwb_file_name": nwb}
        )
        return len(spike_times)

    try:
        n_total = _n_units("all_units")
        assert n_total > 0, "baseline curation must contribute units"
        assert _n_units("cnep2_include_accept") == 0
        assert _n_units("cnep2_exclude_noise") == n_total
    finally:
        for filter_name in (
            "all_units",
            "cnep2_include_accept",
            "cnep2_exclude_noise",
        ):
            grp = SortedSpikesGroup & {
                "sorted_spikes_group_name": f"cnep2_{filter_name}",
                "nwb_file_name": nwb,
            }
            if grp:
                grp.super_delete(warn=False)


@pytest.mark.slow
@pytest.mark.integration
def test_labeled_curation_fetch_keeps_accept_drops_noise(
    planted_two_unit_sort, monkeypatch
):
    """The label filter on a labeled curation (one ``accept``, one ``noise``).

    ``include_labels=['accept']`` keeps only the accepted unit;
    ``exclude_labels=['noise']`` drops the noise unit. The test owns its
    group, so it turns ``test_mode`` off to run the real filter.
    """
    import spyglass.spikesorting.analysis.v1.group as group_mod
    from spyglass.spikesorting.analysis.v1.group import (
        SortedSpikesGroup,
        UnitSelectionParams,
    )
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.recording import RecordingSelection
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection

    monkeypatch.setattr(group_mod, "test_mode", False)

    sorting_key = dict(planted_two_unit_sort)
    unit_ids = sorted(
        int(u) for u in (Sorting.Unit & sorting_key).fetch("unit_id")
    )
    assert len(unit_ids) >= 2, "planted sort must yield >=2 units"
    accept_id, noise_id = unit_ids[0], unit_ids[1]

    root_pk, _ = _make_v2_root_curation(sorting_key)
    child = CurationV2.save_manual_curation(
        sorting_key,
        parent_curation_id=root_pk["curation_id"],
        labels={accept_id: ["accept"], noise_id: ["noise"]},
        curation_source="figpack",
    )
    merge_id = (SpikeSortingOutput.CurationV2 & child).fetch1("merge_id")

    UnitSelectionParams().insert_default()
    UnitSelectionParams().insert1(
        {
            "unit_filter_params_name": "lbl_include_accept",
            "include_labels": ["accept"],
            "exclude_labels": [],
        },
        skip_duplicates=True,
    )
    UnitSelectionParams().insert1(
        {
            "unit_filter_params_name": "lbl_exclude_noise",
            "include_labels": [],
            "exclude_labels": ["noise"],
        },
        skip_duplicates=True,
    )
    recording_id = SortingSelection.resolve_source(sorting_key).key[
        "recording_id"
    ]
    nwb = (RecordingSelection & {"recording_id": recording_id}).fetch1(
        "nwb_file_name"
    )

    expected_frame = (CurationV2 & child).fetch_nwb()[0]["object_id"]
    expected_times = {
        int(unit_id): np.asarray(row["spike_times"])
        for unit_id, row in expected_frame.iterrows()
    }

    def _fetch_units(filter_name):
        gname = f"lbl_{filter_name}"
        existing = SortedSpikesGroup & {
            "sorted_spikes_group_name": gname,
            "nwb_file_name": nwb,
        }
        if existing:
            existing.super_delete(warn=False)
        SortedSpikesGroup().create_group(
            group_name=gname,
            nwb_file_name=nwb,
            unit_filter_params_name=filter_name,
            keys=[{"spikesorting_merge_id": merge_id}],
        )
        spikes, identities = SortedSpikesGroup.fetch_spike_data(
            {"sorted_spikes_group_name": gname, "nwb_file_name": nwb},
            return_unit_ids=True,
        )
        assert len(spikes) == len(identities)
        result = {}
        for train, identity in zip(spikes, identities, strict=True):
            assert identity["spikesorting_merge_id"] == merge_id
            unit_id = int(identity["unit_id"])
            assert unit_id not in result
            np.testing.assert_array_equal(train, expected_times[unit_id])
            result[unit_id] = train
        return result

    try:
        all_units = _fetch_units("all_units")
        assert set(all_units) == set(unit_ids)
        assert set(_fetch_units("lbl_include_accept")) == {accept_id}
        assert set(_fetch_units("lbl_exclude_noise")) == set(unit_ids) - {
            noise_id
        }
    finally:
        for filter_name in (
            "all_units",
            "lbl_include_accept",
            "lbl_exclude_noise",
        ):
            grp = SortedSpikesGroup & {
                "sorted_spikes_group_name": f"lbl_{filter_name}",
                "nwb_file_name": nwb,
            }
            if grp:
                grp.super_delete(warn=False)


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.smoke
def test_downstream_consumer_reads_applied_merge_unit_set(
    planted_two_unit_sort,
):
    """An APPLIED-MERGE committed child reaches the decoding consumer surfaces
    with its MERGED unit set (not the pre-merge ids), via SpikeSortingOutput.
    """
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting

    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    sorting_key = dict(planted_two_unit_sort)
    unit_ids = sorted(
        int(u) for u in (Sorting.Unit & sorting_key).fetch("unit_id")
    )
    assert len(unit_ids) >= 2, "need >=2 units to form a merge"

    def _drop_output_and_curations():
        # Merge masters before curations.
        keys = (CurationV2 & sorting_key).fetch("KEY", as_dict=True)
        if keys:
            for mid in (SpikeSortingOutput.CurationV2 & keys).fetch("merge_id"):
                (SpikeSortingOutput & {"merge_id": mid}).super_delete(
                    warn=False
                )
        clear_curations_for(planted_two_unit_sort)

    _drop_output_and_curations()
    try:
        merged = CurationV2.insert_curation(
            sorting_key,
            merge_groups=[[unit_ids[0], unit_ids[1]]],
            apply_merge=True,
        )
        merged_unit_ids = sorted(
            int(u) for u in (CurationV2.Unit & merged).fetch("unit_id")
        )
        assert len(merged_unit_ids) == 1, merged_unit_ids  # two merged into one
        merged_uid = merged_unit_ids[0]

        merge_id = (SpikeSortingOutput.CurationV2 & merged).fetch1("merge_id")

        sorting = SpikeSortingOutput().get_sorting({"merge_id": merge_id})
        assert sorted(int(u) for u in sorting.get_unit_ids()) == merged_unit_ids

        # Length equals the stored post-merge n_spikes.
        spike_times = SpikeSortingOutput().get_spike_times(
            {"merge_id": merge_id}
        )
        assert len(spike_times) == 1
        n_spikes = int(
            (CurationV2.Unit & merged & {"unit_id": merged_uid}).fetch1(
                "n_spikes"
            )
        )
        assert len(spike_times[0]) == n_spikes
    finally:
        _drop_output_and_curations()

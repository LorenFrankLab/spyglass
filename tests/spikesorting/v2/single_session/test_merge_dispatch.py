"""SpikeSortingOutput merge-table dispatch and downstream-consumer tests for the v2 single-session pipeline."""

from __future__ import annotations

import pytest

from tests.spikesorting.v2.single_session._helpers import _clear_curations


@pytest.mark.slow
def test_spike_sorting_output_get_spike_times_v2_dispatch(populated_sorting):
    """``SpikeSortingOutput.get_spike_times`` resolves a v2 Units-NWB
    ``object_id`` to absolute spike times through the merge master.
    """
    import numpy as np

    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting

    _clear_curations(populated_sorting)
    pk = CurationV2.insert_curation(sorting_key=populated_sorting, labels={})
    merge_id = (SpikeSortingOutput.CurationV2 & pk).fetch1("merge_id")

    spike_times = SpikeSortingOutput().get_spike_times({"merge_id": merge_id})
    assert isinstance(spike_times, list)
    assert len(spike_times) > 0, (
        "get_spike_times returned no unit arrays for a v2 curation "
        "that ``Sorting`` reports as having n_units > 0; the merge "
        "dispatch likely broke."
    )

    sort_row = (Sorting & populated_sorting).fetch1()
    expected_units = int(sort_row["n_units"])
    assert len(spike_times) == expected_units, (
        f"get_spike_times returned {len(spike_times)} unit arrays, "
        f"expected {expected_units} from Sorting.n_units."
    )

    # Check against the recording's own [t_start, t_end]; ``>= 0`` would
    # not distinguish absolute from relative times.
    from spyglass.spikesorting.v2.recording import (
        Recording,
    )
    from spyglass.spikesorting.v2.sorting import SortingSelection

    recording_id = SortingSelection.resolve_source(populated_sorting).key[
        "recording_id"
    ]
    rec = Recording().get_recording({"recording_id": recording_id})
    rec_t_start = float(rec.get_times()[0])
    rec_t_end = float(rec.get_times()[-1])

    for unit_idx, times in enumerate(spike_times):
        arr = np.asarray(times)
        assert arr.dtype.kind == "f", (
            f"Unit {unit_idx}: spike times dtype kind "
            f"{arr.dtype.kind!r}, expected 'f' (float seconds)."
        )
        if arr.size:
            # Tolerance: one sample.
            fs = rec.get_sampling_frequency()
            assert arr.min() >= rec_t_start - 1.0 / fs, (
                f"Unit {unit_idx}: spike time {arr.min():.9f} is "
                f"earlier than recording t_start {rec_t_start:.9f}. "
                "v2 stores absolute timestamps; a regression to "
                "relative-time storage (or t_start=0 hardcoding) "
                "would surface here on a non-zero-start session."
            )
            assert arr.max() <= rec_t_end + 1.0 / fs, (
                f"Unit {unit_idx}: spike time {arr.max():.9f} is "
                f"later than recording t_end {rec_t_end:.9f}."
            )


@pytest.mark.slow
def test_get_restricted_merge_ids_v2_resolves_through_chain(populated_sorting):
    """``get_restricted_merge_ids(sources=['v2'])`` resolves the v2
    restriction surface (sorting_id, curation_id, ...) down to a
    merge_id from the CurationV2 part."""
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.curation import CurationV2

    _clear_curations(populated_sorting)
    pk = CurationV2.insert_curation(sorting_key=populated_sorting, labels={})
    merge_ids = SpikeSortingOutput().get_restricted_merge_ids(
        {"sorting_id": pk["sorting_id"], "curation_id": pk["curation_id"]},
        sources=["v2"],
    )
    assert len(merge_ids) == 1
    # Full chain fields, since ``sorter`` alone matches every MS5 sort in
    # the DB; expect exactly one merge_id.
    merge_ids2 = SpikeSortingOutput().get_restricted_merge_ids(
        {
            "sorter": "mountainsort5",
            "sorting_id": pk["sorting_id"],
            "curation_id": pk["curation_id"],
        },
        sources=["v2"],
    )
    assert list(merge_ids2) == list(merge_ids), (
        f"Restricting by sorter+sorting_id+curation_id returned "
        f"{list(merge_ids2)}; expected exactly {list(merge_ids)}."
    )


# ---------- fetch_nwb direct v2 dispatch ---------------------------------


@pytest.mark.slow
def test_fetch_nwb_v2_dispatch(populated_sorting):
    """``SpikeSortingOutput.fetch_nwb`` works for a v2 source.

    Decoding calls ``fetch_nwb`` directly, not via ``get_spike_times``.
    """
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.curation import CurationV2

    _clear_curations(populated_sorting)
    pk = CurationV2.insert_curation(sorting_key=populated_sorting, labels={})
    merge_id = (SpikeSortingOutput.CurationV2 & pk).fetch1("merge_id")

    nwb_results = SpikeSortingOutput().fetch_nwb({"merge_id": merge_id})
    assert isinstance(nwb_results, list)
    assert len(nwb_results) == 1, (
        f"fetch_nwb returned {len(nwb_results)} dicts for a single "
        "merge_id; expected exactly one."
    )
    payload = nwb_results[0]
    # ``object_id`` (v1/v2) or ``units`` (v0) must be non-None.
    assert "object_id" in payload or "units" in payload, (
        f"fetch_nwb result missing both 'object_id' and 'units' keys; "
        f"got keys {sorted(payload.keys())}."
    )
    if "object_id" in payload:
        # The DataFrame fetch_nwb builds from the Units NWB.
        loaded = payload["object_id"]
        assert loaded is not None, (
            "fetch_nwb returned object_id=None; the v2 Units NWB "
            "did not round-trip through the merge dispatch."
        )
        # One row per kept unit.
        expected_unit_ids = set(
            int(u) for u in (CurationV2.Unit & pk).fetch("unit_id")
        )
        loaded_unit_ids = set(int(u) for u in loaded.index)
        assert loaded_unit_ids == expected_unit_ids, (
            f"fetch_nwb returned Units rows for unit_ids "
            f"{sorted(loaded_unit_ids)}; expected "
            f"{sorted(expected_unit_ids)}."
        )
        assert "spike_times" in loaded.columns, (
            f"fetch_nwb Units DataFrame missing spike_times; "
            f"got columns {list(loaded.columns)}."
        )


# ---------- v1 / imported dispatch after v2 import ------------------------


@pytest.mark.slow
def test_imported_dispatch_survives_v2_module_loaded(populated_sorting):
    """v0 / v1 / imported sources still dispatch correctly through
    ``SpikeSortingOutput.source_class_dict`` after the v2 module
    adds itself to the registry.

    Uses ImportedSpikeSorting because v0/v1 sorting needs SI 0.99.
    """
    from spyglass.spikesorting.spikesorting_merge import (
        SpikeSortingOutput,
        source_class_dict,
    )

    assert "CurationV2" in source_class_dict
    assert "ImportedSpikeSorting" in source_class_dict
    assert SpikeSortingOutput().merge_get_parent_class("CurationV2") is not None
    assert (
        SpikeSortingOutput().merge_get_parent_class("ImportedSpikeSorting")
        is not None
    )
    assert hasattr(SpikeSortingOutput, "ImportedSpikeSorting")


@pytest.mark.slow
@pytest.mark.integration
def test_merge_dispatch_get_recording_works_for_v2(populated_sorting):
    """``SpikeSortingOutput.get_recording`` returns a v2 recording
    with ``is_filtered=True`` annotated.

    ``is_filtered`` stops SI consumers from re-applying a bandpass.
    """
    import spikeinterface as si

    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.curation import CurationV2

    _clear_curations(populated_sorting)
    pk = CurationV2.insert_curation(sorting_key=populated_sorting, labels={})
    merge_id = (SpikeSortingOutput.CurationV2 & pk).fetch1("merge_id")

    rec = SpikeSortingOutput().get_recording({"merge_id": merge_id})
    assert isinstance(rec, si.BaseRecording), (
        f"SpikeSortingOutput.get_recording returned {type(rec)}; "
        "expected SI BaseRecording."
    )
    assert rec.get_annotation("is_filtered") is True, (
        "Returned recording must carry is_filtered=True so a "
        "downstream sorter does not re-bandpass already-filtered "
        "data; is_filtered annotation regression."
    )


@pytest.mark.slow
@pytest.mark.integration
def test_merge_dispatch_get_sort_group_info_works_for_v2(populated_sorting):
    """``SpikeSortingOutput.get_sort_group_info`` returns the full
    electrode set for a v2 merge_id.

    The dispatcher calls it on the class, so it must be a classmethod.
    """
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.curation import CurationV2

    _clear_curations(populated_sorting)
    pk = CurationV2.insert_curation(sorting_key=populated_sorting, labels={})
    merge_id = (SpikeSortingOutput.CurationV2 & pk).fetch1("merge_id")

    info = SpikeSortingOutput.get_sort_group_info({"merge_id": merge_id})
    rows = info.fetch(as_dict=True)
    assert len(rows) > 0, (
        "get_sort_group_info returned zero rows; expected one row "
        "per electrode (a fetch(limit=1) would under-report "
        "multi-region groups)."
    )
    # Spot-check electrode and BrainRegion columns.
    for required in ("electrode_id", "region_name"):
        assert required in rows[0], (
            f"get_sort_group_info row missing {required!r}; check "
            "the relation join order."
        )


@pytest.mark.slow
@pytest.mark.integration
def test_merge_dispatch_restrict_by_artifact_honored_in_v2(populated_sorting):
    """``SpikeSortingOutput.get_restricted_merge_ids`` with
    ``restrict_by_artifact=True`` honors the v2 ``f"artifact_detection_
    {artifact_detection_id}"`` IntervalList naming convention.

    The dispatcher maps ``interval_list_name="artifact_detection_<uuid>"``
    to ``artifact_detection_id=<uuid>`` for the v2 join chain.
    """
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import SortingSelection
    from spyglass.spikesorting.v2._artifacts.naming import (
        artifact_detection_interval_list_name,
    )

    _clear_curations(populated_sorting)
    pk = CurationV2.insert_curation(sorting_key=populated_sorting, labels={})
    expected = (SpikeSortingOutput.CurationV2 & pk).fetch1("merge_id")

    # The artifact pass is on the ArtifactDetectionSource part, so use the
    # resolver.
    artifact_detection_id = SortingSelection.resolve_artifact_detection(
        populated_sorting
    )
    artifact_name = artifact_detection_interval_list_name(artifact_detection_id)

    merge_ids = SpikeSortingOutput().get_restricted_merge_ids(
        {"interval_list_name": artifact_name},
        sources=["v2"],
        restrict_by_artifact=True,
    )
    assert expected in merge_ids, (
        f"Artifact-named restriction did not surface the populated "
        f"merge_id (expected {expected!r}, got {list(merge_ids)!r})."
    )


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize(
    "method_name",
    ["get_spike_indicator", "get_firing_rate"],
)
def test_merge_dispatch_consumer_api_works_on_v2_merge_id(
    populated_sorting, method_name
):
    """``get_spike_indicator`` and ``get_firing_rate`` on a v2 merge_id
    return ``(n_time, n_units)``: non-negative and finite in observed bins,
    ``np.nan`` elsewhere.
    """
    import numpy as np

    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting

    _clear_curations(populated_sorting)
    pk = CurationV2.insert_curation(sorting_key=populated_sorting, labels={})
    merge_id = (SpikeSortingOutput.CurationV2 & pk).fetch1("merge_id")

    n_units = int((Sorting & populated_sorting).fetch1("n_units"))
    assert n_units >= 1

    time_array = np.arange(0.0, 4.0, 0.1)
    merge_key = {"merge_id": merge_id}
    # get_firing_rate has no return_validity flag, so resolve the observed
    # bins once and hold both methods to the same mask.
    valid = SpikeSortingOutput.get_observation_intervals(merge_key).valid_bins(
        time_array
    )
    method = getattr(SpikeSortingOutput(), method_name)
    result = method(merge_key, time_array)
    assert result.shape == (len(time_array), n_units), (
        f"{method_name} returned shape {result.shape}; expected "
        f"({len(time_array)}, {n_units}). v2 dispatch regression."
    )
    assert np.all(result[valid] >= 0), (
        f"{method_name} returned negative values in an observed bin; "
        "the indicator/rate contract is non-negative where observed."
    )
    assert np.all(np.isfinite(result[valid]))
    assert np.isnan(result[~valid]).all(), (
        f"{method_name} must report np.nan, not a count or rate, in bins "
        "the units were not observed over."
    )


@pytest.mark.slow
@pytest.mark.integration
def test_sorted_spikes_group_works_with_v2_merge_id(populated_sorting):
    """Downstream-consumer: ``SortedSpikesGroup`` builds and
    fetches spike data + firing rate on a v2 merge_id.

    Covers ``fetch_spike_data``, ``get_spike_indicator`` and
    ``get_firing_rate`` (per-unit and multiunit), including sparse unit ids.
    """
    import numpy as np

    from spyglass.spikesorting.analysis.v1.group import (
        SortedSpikesGroup,
        UnitSelectionParams,
    )
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting

    _clear_curations(populated_sorting)
    pk = CurationV2.insert_curation(sorting_key=populated_sorting, labels={})
    merge_id = (SpikeSortingOutput.CurationV2 & pk).fetch1("merge_id")

    UnitSelectionParams().insert_default()
    from spyglass.spikesorting.v2.recording import RecordingSelection
    from spyglass.spikesorting.v2.sorting import SortingSelection

    recording_id = SortingSelection.resolve_source(populated_sorting).key[
        "recording_id"
    ]
    actual_nwb = (RecordingSelection & {"recording_id": recording_id}).fetch1(
        "nwb_file_name"
    )

    group_name = "v2_test_sorted_spikes_group"
    # Drop any group left by an earlier run.
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

    # 1. fetch_spike_data returns per-unit spike-time arrays.
    spike_times, file_unit_ids = SortedSpikesGroup.fetch_spike_data(
        group_key, return_unit_ids=True
    )
    assert isinstance(spike_times, list)
    assert len(spike_times) >= 1, (
        "SortedSpikesGroup.fetch_spike_data returned zero units for a "
        "v2 merge_id with n_units>=1; sparse-unit_id regression."
    )
    # unit_id comes from the NWB index, not a positional range.
    for entry in file_unit_ids:
        assert entry["spikesorting_merge_id"] == merge_id

    # 2. get_spike_indicator: (n_time, n_units) shape, non-negative.
    time_array = np.arange(0.0, 4.0, 0.1)
    indicator = SortedSpikesGroup.get_spike_indicator(group_key, time_array)
    n_units = int((Sorting & populated_sorting).fetch1("n_units"))
    assert indicator.shape == (len(time_array), n_units)
    assert np.all(indicator >= 0)

    # 3. get_firing_rate per-unit AND multiunit (MUA).
    fr_per_unit = SortedSpikesGroup.get_firing_rate(
        group_key, time_array, multiunit=False
    )
    assert fr_per_unit.shape == (len(time_array), n_units)
    assert np.all(fr_per_unit >= 0)

    fr_mua = SortedSpikesGroup.get_firing_rate(
        group_key, time_array, multiunit=True
    )
    # Multiunit firing rate is a single trace per group.
    assert fr_mua.shape == (len(time_array), 1)
    assert np.all(fr_mua >= 0)
    assert np.all(np.isfinite(fr_mua))

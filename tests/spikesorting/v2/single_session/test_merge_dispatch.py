"""SpikeSortingOutput merge-table dispatch and downstream-consumer tests for the v2 single-session pipeline."""

from __future__ import annotations

import pytest

from tests.spikesorting.v2.single_session._helpers import _clear_curations

# ---------- Tri-part dispatch active smoke gate ---------------------------


def test_tripart_dispatch_active(dj_conn):
    """Each refactored Computed table uses DataJoint's tri-part dispatch.

    Plan-required smoke gate. DataJoint's ``AutoPopulate.populate`` only
    fires the ``make_fetch`` / ``make_compute`` / ``make_insert``
    sequence when ``inspect.isgeneratorfunction(self.make) is True``
    -- i.e. the inherited generator-based ``make`` from
    ``AutoPopulate`` is in use. If a subclass overrides ``make`` with
    a regular function, DataJoint falls back to monolithic and the
    tri-part methods become dead code; the long-transaction
    regression silently persists. This test catches that failure
    mode in milliseconds.

    The parametrize list covers Recording / ArtifactDetection /
    Sorting -- the three v2 tables that use tri-part dispatch --
    so a future "consolidate back into ``make``" change fails
    loudly.
    """
    import inspect

    from spyglass.spikesorting.v2.artifact import ArtifactDetection
    from spyglass.spikesorting.v2.recording import Recording
    from spyglass.spikesorting.v2.sorting import Sorting

    for cls in (Recording, ArtifactDetection, Sorting):
        assert inspect.isgeneratorfunction(cls.make), (
            f"{cls.__name__}.make must remain the inherited generator "
            "from AutoPopulate so tri-part dispatch fires; a regular-"
            "function override would silently re-enable the monolithic "
            "long-transaction path."
        )
        for attr in ("make_fetch", "make_compute", "make_insert"):
            assert getattr(cls, attr, None) is not None, (
                f"{cls.__name__}.{attr} missing; tri-part dispatch "
                "needs all three methods defined."
            )
        assert getattr(cls, "_parallel_make", False) is True, (
            f"{cls.__name__}._parallel_make is not True; the "
            "non-daemon process-pool path is the secondary benefit "
            "of the tri-part refactor and should be enabled."
        )


@pytest.mark.slow
def test_spike_sorting_output_get_spike_times_v2_dispatch(populated_sorting):
    """``SpikeSortingOutput.get_spike_times`` dispatches to a v2 source.

    The previous review caught that the consumer-facing
    ``get_spike_times`` had a v0 / v1 / imported test but no v2
    dispatch path. This regression test pins the v2 NWB
    ``object_id`` -> spike-times resolution through the merge master,
    so a change to the v2 Units-NWB layout that breaks
    ``fetch_nwb`` immediately fails the suite.
    """
    import numpy as np

    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting

    _clear_curations(populated_sorting)
    pk = CurationV2.insert_curation(sorting_key=populated_sorting, labels={})
    merge_id = (SpikeSortingOutput.CurationV2 & pk).fetch1("merge_id")

    # The consumer-facing API returns a list of per-unit spike-time
    # arrays. Pin shape (n_units arrays) + dtype (float seconds) +
    # value range (non-negative, within the recording duration).
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

    # Spike times must fall within the recording's wall-clock window,
    # specifically [t_start, t_end] where t_start is the FIRST
    # timestamp of the upstream recording. The smoke fixture starts
    # at t=0 so a naive ``>= 0.0`` check is satisfied by both correct
    # (absolute-time) and broken (relative-time) implementations.
    # Pinning against the actual t_start (and t_end) catches a
    # regression to relative storage on any non-zero-start session.
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
    )
    from spyglass.spikesorting.v2.sorting import SortingSelection

    recording_id = (
        SortingSelection.RecordingSource & populated_sorting
    ).fetch1("recording_id")
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
            # Strict absolute-time check: any spike before t_start
            # would indicate a relative-time / t_start=0 regression.
            # The tolerance is one sample (1/fs).
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
    # Re-resolving by the FULL set of v2 chain fields (more
    # restrictive than ``sorter`` alone, which would match every v2
    # MS5 sorting from any prior test run) returns the same single
    # merge_id. The previous looser assertion (``merge_ids[0] in
    # merge_ids2``) was vacuously satisfied if state pollution
    # produced multiple matches.
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

    ``get_spike_times`` is the consumer-facing API but it dispatches
    through ``fetch_nwb`` (the merge-table primitive). Downstream
    consumers like decoding use ``fetch_nwb`` directly. This test
    pins the v2 ``object_id`` -> Units NWB resolution at the lower
    level so a change to the merge dispatch / column-name convention
    surfaces immediately.
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
    # v2 stores the Units NWB; ``object_id`` is the convention key
    # downstream consumers use to fetch spike times. Either
    # ``object_id`` (the v2/v1 convention) or ``units`` (the v0
    # convention) must be present AND non-None.
    assert "object_id" in payload or "units" in payload, (
        f"fetch_nwb result missing both 'object_id' and 'units' keys; "
        f"got keys {sorted(payload.keys())}."
    )
    if "object_id" in payload:
        # ``payload["object_id"]`` is the DataFrame that fetch_nwb
        # builds from the Units NWB referenced by CurationV2.object_id.
        # The schema-only assertion above just checked the key is
        # present; this check pins that the DataFrame actually
        # round-tripped from the NWB (non-None, has rows, has the
        # expected columns).
        loaded = payload["object_id"]
        assert loaded is not None, (
            "fetch_nwb returned object_id=None; the v2 Units NWB "
            "did not round-trip through the merge dispatch."
        )
        # DataFrame check: rows correspond to CurationV2.Unit rows
        # (one row per kept unit).
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


# ---------- v1 / imported dispatch regression after v2 import ------------


@pytest.mark.slow
def test_imported_dispatch_survives_v2_module_loaded(populated_sorting):
    """v0 / v1 / imported sources still dispatch correctly through
    ``SpikeSortingOutput.source_class_dict`` after the v2 module
    adds itself to the registry.

    Regression guard for the conditional ``source_class_dict["CurationV2"]
    = CurationV2`` block in ``spikesorting_merge.py``: a future
    refactor that broke v1 dispatch as a side effect of v2 changes
    should fail this test.

    Uses the ImportedSpikeSorting source registered by the test
    bootstrap (via ``mini_insert``) rather than v0/v1 sorting paths,
    because the v2 environment has SpikeInterface 0.104 which is
    incompatible with the SI 0.99-based v0/v1 production code.
    """
    from spyglass.spikesorting.imported import ImportedSpikeSorting
    from spyglass.spikesorting.spikesorting_merge import (
        SpikeSortingOutput,
        source_class_dict,
    )

    # Both v2 and the imported source must be registered.
    assert "CurationV2" in source_class_dict
    assert "ImportedSpikeSorting" in source_class_dict
    # The Merge class also resolves both via merge_get_parent_class.
    assert SpikeSortingOutput().merge_get_parent_class("CurationV2") is not None
    assert (
        SpikeSortingOutput().merge_get_parent_class("ImportedSpikeSorting")
        is not None
    )
    # And the Imported part table is reachable through the master.
    assert hasattr(SpikeSortingOutput, "ImportedSpikeSorting")


@pytest.mark.slow
@pytest.mark.integration
def test_merge_dispatch_get_recording_works_for_v2(populated_sorting):
    """``SpikeSortingOutput.get_recording`` returns a v2 recording
    with ``is_filtered=True`` annotated.

    Without the ``CurationV2.get_recording`` classmethod +
    part-table source-class wiring, the merge dispatcher at
    ``spikesorting_merge.py:317`` raises ``AttributeError`` on
    every v2 ``merge_id``. Without the ``recording.annotate(
    is_filtered=True)`` call, downstream SI consumers may re-apply
    a bandpass to the already-filtered preprocessed recording. Pin
    both invariants in one test.
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

    Before ``CurationV2.get_sort_group_info`` was promoted to a
    classmethod, the merge dispatcher at
    ``spikesorting_merge.py:346`` called it as
    ``source_table.get_sort_group_info(merge_key)`` where
    ``source_table`` is the bound class, not an instance --
    raising ``TypeError: missing self``. With the classmethod
    conversion, the call resolves. This test confirms it
    actually returns a non-empty multi-row relation.
    """
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.curation import CurationV2

    _clear_curations(populated_sorting)
    pk = CurationV2.insert_curation(sorting_key=populated_sorting, labels={})
    merge_id = (SpikeSortingOutput.CurationV2 & pk).fetch1("merge_id")

    info = SpikeSortingOutput.get_sort_group_info({"merge_id": merge_id})
    rows = info.fetch(as_dict=True)
    assert len(rows) > 0, (
        "get_sort_group_info returned zero rows; the v1 "
        "fetch(limit=1) multi-region under-reporting bug has "
        "regressed."
    )
    # The result must include the electrode-level columns the
    # plan documents (rows for every electrode in the sort
    # group, joined to BrainRegion). Spot-check a couple of
    # canonical column names.
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

    The dispatcher converts ``interval_list_name="artifact_detection_<uuid>"``
    back to ``artifact_detection_id=<uuid>`` for the v2 join chain. Without
    this conversion an artifact-named restriction returns no v2
    merge_ids.
    """
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import SortingSelection
    from spyglass.spikesorting.v2.utils import (
        artifact_detection_interval_list_name,
    )

    _clear_curations(populated_sorting)
    pk = CurationV2.insert_curation(sorting_key=populated_sorting, labels={})
    expected = (SpikeSortingOutput.CurationV2 & pk).fetch1("merge_id")

    # Resolve this sort's artifact_detection_id (the "none" preset writes
    # one IntervalList with the artifact-naming convention). The artifact
    # pass lives on the ArtifactDetectionSource part, not the master, so read it
    # through the resolver rather than a master column.
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
    """Downstream-consumer dispatch sanity for the two highest-
    leverage time-binned APIs.

    ``get_spike_indicator`` is the consumer-facing API for
    clusterless decoding; ``get_firing_rate`` is the
    decoding/unit-quality counterpart. Both iterate
    ``get_spike_times(merge_key)`` under the hood -- a regression
    in v2's ``get_spike_times`` dispatch silently breaks
    downstream consumers. Both APIs must return a
    ``(n_time, n_units)`` array, non-negative everywhere,
    finite. Parametrized so the same shape contract is verified
    on both with identical setup.
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
    method = getattr(SpikeSortingOutput(), method_name)
    result = method({"merge_id": merge_id}, time_array)
    assert result.shape == (len(time_array), n_units), (
        f"{method_name} returned shape {result.shape}; expected "
        f"({len(time_array)}, {n_units}). v2 dispatch regression."
    )
    assert np.all(result >= 0), (
        f"{method_name} returned negative values somewhere; "
        "the indicator/rate contract is non-negative."
    )
    assert np.all(np.isfinite(result))


@pytest.mark.slow
@pytest.mark.integration
def test_sorted_spikes_group_works_with_v2_merge_id(populated_sorting):
    """Downstream-consumer: ``SortedSpikesGroup`` builds and
    fetches spike data + firing rate on a v2 merge_id.

    Constructs a ``SortedSpikesGroup`` from a v2 curation's
    merge_id (the cross-pipeline path used by decoding and
    ripple-detection workflows), then exercises
    ``fetch_spike_data``, ``get_spike_indicator``,
    ``get_firing_rate`` (per-unit and MUA-multiunit modes).
    Catches sparse-unit_id and v2-merge-dispatch regressions on
    the actual SortedSpikesGroup surface, not just the
    SpikeSortingOutput primitive APIs.
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

    # Use the default ``all_units`` UnitSelectionParams that ships
    # with the analysis module.
    UnitSelectionParams().insert_default()
    from spyglass.spikesorting.v2.recording import RecordingSelection
    from spyglass.spikesorting.v2.sorting import SortingSelection

    recording_id = (
        SortingSelection.RecordingSource & populated_sorting
    ).fetch1("recording_id")
    actual_nwb = (RecordingSelection & {"recording_id": recording_id}).fetch1(
        "nwb_file_name"
    )

    group_name = "v2_test_sorted_spikes_group"
    # Idempotent: drop any prior group from earlier test runs.
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
    # file_unit_ids is a list of dicts; each carries the v2 merge_id
    # + unit_id from the NWB index (not a positional range).
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

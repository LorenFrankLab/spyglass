"""Downstream-consumer smoke tests for v2 spike-sorting outputs.

This module is the canonical target for the suite-runner contract
``pytest tests/spikesorting/v2/test_downstream_consumers.py -q``.
It exercises the downstream consumer surfaces that a Frank-lab
decoding / ripple-detection workflow would hit on a v2 ``merge_id``:

- ``SpikeSortingOutput.get_recording`` / ``get_sorting`` /
  ``get_sort_group_info`` (the merge-dispatch entrypoints).
- ``SpikeSortingOutput.get_spike_times`` / ``fetch_nwb``
  (the NWB round-trip path that decoding consumers depend on).
- ``SpikeSortingOutput.get_spike_indicator`` / ``get_firing_rate``
  (per-bin consumer APIs).
- ``SortedSpikesGroup`` (analysis-side group surface that decoding
  and ripple-detection construct from a v2 merge_id).

A sparse-unit_id regression in any of these surfaces silently
breaks clusterless decoding; this module is the focused gate. All
tests are integration-tier (require ``populated_sorting`` from
conftest.py) and marked ``slow + integration``.

The ``SortedSpikesGroup`` MUA test exercises
``firing_rate_from_spike_indicator`` with ``multiunit=True``, the
v1-MultiUnitActivity-parity invariant (the analysis-side equivalent
of v1's per-bin multi-unit firing rate).
"""

from __future__ import annotations

import numpy as np
import pytest

# The ``populated_sorting`` fixture (a full Recording -> Artifact ->
# Sorting populate on the smoke fixture) lives in conftest.py so the
# integrity tests can resolve it directly instead of importing it from
# this module (a cross-module import passes vacuously when a CI shard
# split collects the two modules separately).


def _make_v2_root_curation(populated_sorting):
    """Clear prior curations + insert a fresh root for the smoke sort.

    Returns ``(pk, merge_id)``. Reused across most tests in this
    module so each test starts from a clean curation row. The
    cleanup walks merge -> curation -> sorting in master-before-part
    order because DataJoint refuses to drop a part-table row before
    its master (the same cascade rule ``_clean_session_v2`` follows
    in the ``single_session/`` suite).
    """
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.curation import CurationV2

    # Step 1: drop the SpikeSortingOutput master rows whose
    # ``CurationV2`` part points at any curation derived from this
    # sorting. Master-first.
    curation_keys = (CurationV2 & populated_sorting).fetch("KEY", as_dict=True)
    if curation_keys:
        merge_ids = (SpikeSortingOutput.CurationV2 & curation_keys).fetch(
            "merge_id"
        )
        for mid in merge_ids:
            (SpikeSortingOutput & {"merge_id": mid}).super_delete(warn=False)
    # Step 2: drop the curation rows.
    (CurationV2 & populated_sorting).super_delete(warn=False)

    pk = CurationV2.insert_curation(sorting_key=populated_sorting, labels={})
    merge_id = (SpikeSortingOutput.CurationV2 & pk).fetch1("merge_id")
    return pk, merge_id


@pytest.mark.slow
@pytest.mark.integration
def test_get_recording_returns_filtered_recording(populated_sorting):
    """``SpikeSortingOutput.get_recording`` on a v2 merge_id returns
    an SI BaseRecording with ``is_filtered=True``.

    The annotation guards downstream sorters from re-bandpassing an
    already-filtered preprocessed recording. Without the
    ``is_filtered`` annotation + the merge-dispatch get_recording
    wiring, the merge dispatcher raises AttributeError on every v2
    merge_id.
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
    with rows for EVERY electrode in the sort group (not the v1
    ``fetch(limit=1)`` single-row bug).
    """
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput

    _, merge_id = _make_v2_root_curation(populated_sorting)
    info = SpikeSortingOutput.get_sort_group_info({"merge_id": merge_id})
    rows = info.fetch(as_dict=True)
    assert len(rows) > 0
    # Spot-check that the relation joined to BrainRegion (the v1
    # multi-region-underreporting fix verification).
    assert "electrode_id" in rows[0]
    assert "region_name" in rows[0]


@pytest.mark.slow
@pytest.mark.integration
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
    arrays that are non-negative and finite.

    Parametrized so the same shape contract is verified on both
    with identical setup; any v2-merge-dispatch regression in
    ``get_spike_times`` (which both call under the hood) trips here.
    """
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.sorting import Sorting

    _, merge_id = _make_v2_root_curation(populated_sorting)
    n_units = int((Sorting & populated_sorting).fetch1("n_units"))
    time_array = np.arange(0.0, 4.0, 0.1)
    result = getattr(SpikeSortingOutput(), method_name)(
        {"merge_id": merge_id}, time_array
    )
    assert result.shape == (len(time_array), n_units)
    assert np.all(result >= 0)
    assert np.all(np.isfinite(result))


@pytest.mark.slow
@pytest.mark.integration
def test_consumer_spike_indicator_count_alignment(populated_sorting):
    """Per-unit ``get_spike_indicator`` column sums equal each unit's
    in-window ``get_spike_times`` count.

    The shape/sign/finite contract in ``test_consumer_api_shape_contract``
    passes even if column ``j`` of the indicator is wired to a DIFFERENT
    unit than ``get_spike_times()[j]`` (a sparse-unit_id misalignment that
    silently corrupts clusterless decoding). This pins the exact per-unit
    correspondence: ``get_spike_indicator`` bins each unit's spikes that
    fall in ``[time[0], time[-1]]``, so the column sum MUST equal the
    count of that same unit's spike times inside the window.
    """
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput

    _, merge_id = _make_v2_root_curation(populated_sorting)
    spike_times = SpikeSortingOutput().get_spike_times({"merge_id": merge_id})
    assert len(spike_times) >= 1, "fixture must yield at least one unit"

    time_array = np.arange(0.0, 4.0, 0.1)
    indicator = SpikeSortingOutput().get_spike_indicator(
        {"merge_id": merge_id}, time_array
    )
    assert indicator.shape == (len(time_array), len(spike_times))

    min_t, max_t = time_array[0], time_array[-1]
    column_sums = indicator.sum(axis=0)
    for j, arr in enumerate(spike_times):
        in_window = int(np.count_nonzero((arr >= min_t) & (arr <= max_t)))
        assert int(column_sums[j]) == in_window, (
            f"unit {j}: indicator column sum {int(column_sums[j])} != "
            f"in-window spike count {in_window}; the per-unit indicator "
            "is misaligned with get_spike_times (sparse-unit_id bug)."
        )
    # Total spikes in-window across all units must also agree (catches a
    # misalignment that happens to permute counts between equal-count
    # units).
    total_in_window = sum(
        int(np.count_nonzero((arr >= min_t) & (arr <= max_t)))
        for arr in spike_times
    )
    assert int(column_sums.sum()) == total_in_window


@pytest.mark.slow
@pytest.mark.integration
def test_sorted_spikes_decoding_selection_accepts_v2_merge_id(
    populated_sorting,
):
    """A ``SortedSpikesDecodingSelection`` row keyed on a v2 ``merge_id``
    can be INSERTED -- the FK chain resolves end-to-end.

    Without this gate, a broken FK (e.g. ``SpikeSortingOutput.
    CurationV2`` missing or ``SortedSpikesGroup.Units.
    spikesorting_merge_id`` not resolving to v2) would surface only
    when a downstream user tried to populate a decoder on a v2
    sort. The chain this exercises is::

        SortedSpikesDecodingSelection
          -> SortedSpikesGroup
            -> SortedSpikesGroup.Units (spikesorting_merge_id=v2 merge_id)
              -> SpikeSortingOutput (master)
                -> SpikeSortingOutput.CurationV2 (v2 dispatch part)
                  -> CurationV2 (v2 master)

    The test is INSERT-only. ``SortedSpikesDecodingV1.populate`` is
    not exercised here -- the compute step requires position data
    + non_local_detector weights that the smoke fixture lacks; the
    populate surface is exercised in tests/decoding/ on real
    position data.
    """
    pytest.importorskip(
        "non_local_detector",
        reason="decoding extras not installed; FK gate is decoding-side.",
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
    # ``DecodingParameters.insert_default`` is broken upstream
    # (calls ``cls.super()`` which doesn't resolve). Insert through
    # the instance method instead -- ``DecodingParameters.insert`` is
    # overridden to convert classes to dicts on the way in.
    DecodingParameters().insert(
        DecodingParameters.contents, skip_duplicates=True
    )

    recording_id = (
        SortingSelection.RecordingSource & populated_sorting
    ).fetch1("recording_id")
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
    # The decoding selection FK lands on PositionGroup, not
    # PositionGroup.Position, so the parts are not required to
    # satisfy the chain we're proving here.
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
    if not sorted_param_names:
        pytest.skip(
            "DecodingParameters.contents has no sorted-spikes entry "
            "(upstream rename or content drop); cannot exercise the "
            "FK chain without a valid decoding_param_name."
        )
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

    # Explicit join across every hop of the FK chain back to the v2
    # ``CurationV2`` master, restricted to the merge_id we just
    # inserted. A non-empty result PROVES every FK in the chain is
    # consistent; a broken hop would either reject the join or drop
    # the row.
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

    The v1-MultiUnitActivity-parity invariant: the MUA path must
    compose with v2 sparse-unit_id sortings without misindexing.
    Per-unit shape is ``(n_time, n_units)``; MUA collapses to
    ``(n_time, 1)``.
    """
    from spyglass.spikesorting.analysis.v1.group import (
        SortedSpikesGroup,
        UnitSelectionParams,
    )
    from spyglass.spikesorting.v2.recording import RecordingSelection
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection

    _, merge_id = _make_v2_root_curation(populated_sorting)
    UnitSelectionParams().insert_default()

    recording_id = (
        SortingSelection.RecordingSource & populated_sorting
    ).fetch1("recording_id")
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

    This is the label-filtering algorithm that ``fetch_spike_data`` applies
    in production. The ``fetch_spike_data`` invocation is intentionally
    bypassed under pytest (the ``not test_mode`` guard in
    ``analysis/v1/group.py``) because the shared base-env fixtures build
    ``default_exclusion`` groups over noise/mua-labeled curations, so
    running the filter there would empty the group and break unrelated
    tests (verified: dropping the guard makes ``test_fetch_data`` /
    sorted-spikes decoding fail on ``np.concatenate([])``). The filtering
    logic itself is therefore pinned here, directly and exactly, rather
    than through that bypassed integration path.

    Not slow: importing ``SortedSpikesGroup`` declares its DataJoint
    schema (hence ``dj_conn``), but the test calls the pure static method
    with no populate.
    """
    import numpy as np

    from spyglass.spikesorting.analysis.v1.group import SortedSpikesGroup

    f = SortedSpikesGroup.filter_units

    # No include/exclude -> every unit kept (the ``all_units`` preset).
    np.testing.assert_array_equal(
        f([["noise"], ["accept"], []], [], []),
        np.array([True, True, True]),
    )

    # Exclude noise/mua (the ``default_exclusion`` preset): units carrying
    # either label are dropped; unlabeled / other-labeled units survive.
    np.testing.assert_array_equal(
        f([["noise"], ["accept"], ["mua"], []], [], ["noise", "mua"]),
        np.array([False, True, False, True]),
    )

    # A group whose every unit is noise/mua collapses to all-False -- this
    # is exactly why filtering the shared base-env fixtures empties the
    # group and the production guard keeps it off under pytest.
    np.testing.assert_array_equal(
        f([["noise"], ["mua"]], [], ["noise", "mua"]),
        np.array([False, False]),
    )

    # Include filter: keep only units carrying an include label.
    np.testing.assert_array_equal(
        f([["accept"], ["noise"], ["accept", "mua"]], ["accept"], []),
        np.array([True, False, True]),
    )

    # Include + exclude combined: exclude wins for a unit that carries both
    # an include and an exclude label.
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

    An all-unlabeled curated export drops the ``curation_label`` column, so the
    column-present filter path is skipped and an include-only selection used to
    return ALL units. The consumer now synthesizes empty per-unit label lists:
    ``include_labels=["accept"]`` returns NO units; ``exclude_labels=["noise"]``
    returns ALL units.
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

    recording_id = (
        SortingSelection.RecordingSource & populated_sorting
    ).fetch1("recording_id")
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
        # All units are unlabeled -> none carry "accept" -> none returned.
        assert _n_units("cnep2_include_accept") == 0
        # None carry "noise" -> all kept.
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
    """The production label-filter path on a GENUINELY labeled curation.

    A curation with one ``accept`` and one ``noise`` unit, exported downstream:
    ``include_labels=['accept']`` keeps ONLY the accepted unit;
    ``exclude_labels=['noise']`` drops the noise unit and keeps the rest. This
    is the labeled complement to
    ``test_all_unlabeled_curation_include_label_filters`` (which covers the
    no-label-column path). Uses ``planted_two_unit_sort`` because the smoke sort
    yields only one unit -- two distinctly-labeled units are needed to tell the
    include/exclude filters apart.

    ``fetch_spike_data``'s column-present filter is guarded off under pytest (the
    load-bearing ``not test_mode`` at ``analysis/v1/group.py`` -- it exists so a
    default-exclusion filter can't empty the SHARED base-env fixtures). This test
    owns its group + curation, so it locally flips ``test_mode`` off to exercise
    the real filter without touching those fixtures.
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

    # Run the column-present label filter this test relies on (isolated to this
    # test's own group, so it cannot empty another test's fixtures).
    monkeypatch.setattr(group_mod, "test_mode", False)

    sorting_key = dict(planted_two_unit_sort)
    unit_ids = sorted(
        int(u) for u in (Sorting.Unit & sorting_key).fetch("unit_id")
    )
    assert len(unit_ids) >= 2, "planted sort must yield >=2 units"
    accept_id, noise_id = unit_ids[0], unit_ids[1]

    # Clear + root, then a labeled child (the production curate-the-root path).
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
    recording_id = (SortingSelection.RecordingSource & sorting_key).fetch1(
        "recording_id"
    )
    nwb = (RecordingSelection & {"recording_id": recording_id}).fetch1(
        "nwb_file_name"
    )

    def _n_units(filter_name):
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
        return len(
            SortedSpikesGroup.fetch_spike_data(
                {"sorted_spikes_group_name": gname, "nwb_file_name": nwb}
            )
        )

    try:
        n_total = _n_units("all_units")
        assert n_total >= 2, "baseline curation must contribute >=2 units"
        # include=accept -> ONLY the accept-labeled unit.
        assert _n_units("lbl_include_accept") == 1
        # exclude=noise -> everything except the noise-labeled unit.
        assert _n_units("lbl_exclude_noise") == n_total - 1
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
def test_downstream_consumer_reads_applied_merge_unit_set(
    planted_two_unit_sort,
):
    """An APPLIED-MERGE committed child reaches the decoding consumer surfaces
    with its MERGED unit set (not the pre-merge ids), via SpikeSortingOutput.

    The rest of this module builds ROOT curations; this pins that the committed
    merged unit set -- the production curation shape -- flows through the
    merge-dispatch ``get_sorting`` / ``get_spike_times`` path a decoding consumer
    keys off ``merge_id``.
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
        # Master-first: drop SpikeSortingOutput rows pointing at this sort's
        # curations before clearing the curations (the merge master FK-pins them).
        keys = (CurationV2 & sorting_key).fetch("KEY", as_dict=True)
        if keys:
            for mid in (SpikeSortingOutput.CurationV2 & keys).fetch("merge_id"):
                (SpikeSortingOutput & {"merge_id": mid}).super_delete(
                    warn=False
                )
        clear_curations_for(planted_two_unit_sort)

    _drop_output_and_curations()
    try:
        merged = CurationV2.create_merged_curation(
            sorting_key, merge_groups=[[unit_ids[0], unit_ids[1]]]
        )
        merged_unit_ids = sorted(
            int(u) for u in (CurationV2.Unit & merged).fetch("unit_id")
        )
        assert len(merged_unit_ids) == 1, merged_unit_ids  # two merged into one
        merged_uid = merged_unit_ids[0]

        merge_id = (SpikeSortingOutput.CurationV2 & merged).fetch1("merge_id")

        # get_sorting dispatches on the merge_id and yields the MERGED unit set.
        sorting = SpikeSortingOutput().get_sorting({"merge_id": merge_id})
        assert sorted(int(u) for u in sorting.get_unit_ids()) == merged_unit_ids

        # get_spike_times: one per-unit array; its length is the merged unit's
        # stored n_spikes (the authoritative post-merge, post-dedup count).
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

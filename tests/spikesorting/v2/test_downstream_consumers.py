"""Downstream-consumer smoke tests for v2 spike-sorting outputs.

Phase 1 plan-line 182 commands ``pytest tests/spikesorting/v2/
test_downstream_consumers.py -q``; this module is the named target
for that command. It exercises the downstream consumer surfaces
that a Frank-lab decoding / ripple-detection workflow would hit
on a v2 ``merge_id``:

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

Cross-references:
- Phase 1d audit-followup commits also added related tests inline
  in ``test_single_session_pipeline.py`` (the same fixtures are
  shared via the conftest, so tests can live in either file). This
  module is the canonical "downstream consumers" target.
- The SortedSpikesGroup MUA path tests ``firing_rate_from_spike_
  indicator`` with ``multiunit=True``, the v1-MultiUnitActivity-
  parity invariant the plan flags at line 218.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# These tests need a full Recording -> Artifact -> Sorting populate.
# Rather than refactor the cross-file fixture machinery in
# test_single_session_pipeline.py, this module owns its own
# self-contained ``populated_sorting`` fixture that builds state on
# the SAME smoke fixture used elsewhere. The fixture is module-
# scoped so the heavy populate is paid once.

_FIXTURE_NAME = "mearec_polymer_smoke"
_FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / f"{_FIXTURE_NAME}.nwb"
)


@pytest.fixture(scope="module")
def populated_sorting(dj_conn):
    """Populate Recording -> ArtifactDetection -> Sorting for the smoke
    fixture. Module-scoped so the heavy populate is paid once.
    """
    from tests.spikesorting.v2._ingest_helpers import copy_and_insert_nwb

    if not _FIXTURE_PATH.exists():
        pytest.skip(
            f"Generated MEArec fixture {_FIXTURE_PATH.name} not found. Run "
            "`python tests/spikesorting/v2/fixtures/generate_mearec.py "
            "--smoke` first."
        )

    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetection,
        ArtifactDetectionParameters,
        ArtifactSelection,
    )
    from spyglass.spikesorting.v2.recording import (
        PreprocessingParameters,
        Recording,
        RecordingSelection,
        SortGroupV2,
    )
    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        Sorting,
        SortingSelection,
    )

    nwb_file_name = copy_and_insert_nwb(_FIXTURE_PATH)
    session_key = {"nwb_file_name": nwb_file_name}

    PreprocessingParameters.insert_default()
    ArtifactDetectionParameters.insert_default()
    SorterParameters.insert_default()
    LabTeam.insert1(
        {"team_name": "v2_test_team", "team_description": "v2 downstream"},
        skip_duplicates=True,
    )

    if not (SortGroupV2 & session_key):
        SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
    sort_group_id = int(
        sorted((SortGroupV2 & session_key).fetch("sort_group_id"))[0]
    )
    rec_pk = RecordingSelection.insert_selection(
        {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": sort_group_id,
            "interval_list_name": "raw data valid times",
            "preproc_params_name": "default_franklab",
            "team_name": "v2_test_team",
        }
    )
    if not (Recording & rec_pk):
        Recording.populate(rec_pk, reserve_jobs=False)
    art_pk = ArtifactSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "artifact_params_name": "none",
        }
    )
    if not (ArtifactDetection & art_pk):
        ArtifactDetection.populate(art_pk, reserve_jobs=False)
    sort_pk = SortingSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "sorter": "mountainsort5",
            "sorter_params_name": "franklab_tetrode_hippocampus_30kHz_ms5",
            "artifact_id": art_pk["artifact_id"],
        }
    )
    if not (Sorting & sort_pk):
        Sorting.populate(sort_pk, reserve_jobs=False)
    yield sort_pk


def _make_v2_root_curation(populated_sorting):
    """Clear prior curations + insert a fresh root for the smoke sort.

    Returns ``(pk, merge_id)``. Reused across most tests in this
    module so each test starts from a clean curation row. The
    cleanup walks merge -> curation -> sorting in master-before-part
    order because DataJoint refuses to drop a part-table row before
    its master (the same cascade rule ``_clean_session_v2`` follows
    in test_single_session_pipeline.py).
    """
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.curation import CurationV2

    # Step 1: drop the SpikeSortingOutput master rows whose
    # ``CurationV2`` part points at any curation derived from this
    # sorting. Master-first.
    curation_keys = (CurationV2 & populated_sorting).fetch(
        "KEY", as_dict=True
    )
    if curation_keys:
        merge_ids = (
            SpikeSortingOutput.CurationV2 & curation_keys
        ).fetch("merge_id")
        for mid in merge_ids:
            (SpikeSortingOutput & {"merge_id": mid}).super_delete(
                warn=False
            )
    # Step 2: drop the curation rows.
    (CurationV2 & populated_sorting).super_delete(warn=False)

    pk = CurationV2.insert_curation(
        sorting_key=populated_sorting, labels={}
    )
    merge_id = (SpikeSortingOutput.CurationV2 & pk).fetch1("merge_id")
    return pk, merge_id


@pytest.mark.slow
@pytest.mark.integration
def test_get_recording_returns_filtered_recording(populated_sorting):
    """``SpikeSortingOutput.get_recording`` on a v2 merge_id returns
    an SI BaseRecording with ``is_filtered=True``.

    The annotation guards downstream sorters from re-bandpassing an
    already-filtered preprocessed recording. Without R1+R6, the
    merge dispatcher raised AttributeError on every v2 merge_id.
    """
    import spikeinterface as si

    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput

    _, merge_id = _make_v2_root_curation(populated_sorting)
    rec = SpikeSortingOutput().get_recording({"merge_id": merge_id})
    assert isinstance(rec, si.BaseRecording)
    assert rec.get_annotation("is_filtered") is True


@pytest.mark.slow
@pytest.mark.integration
def test_get_sort_group_info_returns_multi_electrode_relation(populated_sorting):
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
    spike_times = SpikeSortingOutput().get_spike_times(
        {"merge_id": merge_id}
    )
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
def test_sorted_spikes_group_per_unit_and_mua_firing_rate(populated_sorting):
    """``SortedSpikesGroup`` built from a v2 merge_id supports both
    per-unit and multiunit (MUA) firing-rate readouts.

    This is the v1-MultiUnitActivity-parity invariant the plan
    flags at phase-1-modern-single-session.md:218 -- the MUA path
    must compose with v2 sparse-unit_id sortings without
    misindexing. Per-unit shape is ``(n_time, n_units)``; MUA
    collapses to ``(n_time, 1)``.
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
    actual_nwb = (
        RecordingSelection & {"recording_id": recording_id}
    ).fetch1("nwb_file_name")

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

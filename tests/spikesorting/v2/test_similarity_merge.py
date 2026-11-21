"""The shipped ``similarity_merge`` rules find a real oversplit, unmocked.

One physical neuron is split into units 0 and 1 by a seeded coin flip per
spike (a strictly alternating split would halve each autocorrelogram's short
intervals, so SpikeInterface's correlogram test rightly refuses it). Unit 2
is a second neuron on other channels. Only the sorter step is substituted.
"""

from __future__ import annotations

import uuid

import numpy as np
import pytest

pytestmark = [pytest.mark.slow, pytest.mark.integration]

_FS = 30_000
_DURATION_S = 120.0


def _poisson_frames(seed, rate_hz=30.0, refractory_s=0.003):
    rng = np.random.default_rng(seed)
    n = int(_DURATION_S * rate_hz * 2)
    times = np.cumsum(rng.exponential(1 / rate_hz, size=n) + refractory_s)
    times = times[(times > 0.01) & (times < _DURATION_S - 0.01)]
    return np.round(times * _FS).astype(np.int64)


def test_similarity_merge_suggests_and_accepts_exactly_the_oversplit_pair(
    dj_conn, tmp_path, monkeypatch, curation_evaluation_defaults
):
    import spikeinterface as si

    from spyglass.common.common_lab import LabTeam
    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2._storage.units_nwb import (
        read_units_abs_times_and_sample_indices,
    )
    from spyglass.spikesorting.v2.artifact import (
        RecordingArtifactDetection,
        RecordingArtifactSelection,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.metric_curation import (
        CurationEvaluation,
        CurationEvaluationSelection,
    )
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
        SortGroupV2,
    )
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection
    from tests.spikesorting.v2._biological_recordings import (
        write_biological_nwb,
    )
    from tests.spikesorting.v2._ingest_helpers import (
        _clean_session_v2,
        clear_curations_for,
        copy_and_insert_nwb,
    )
    from tests.spikesorting.v2._scenario_oracles import _raw_acquisition_clock
    from tests.spikesorting.v2._sorter_stub import plant_sorter

    neuron = _poisson_frames(seed=1)
    other = _poisson_frames(seed=2)
    in_a = np.random.default_rng(0).random(len(neuron)) < 0.5
    frames_by_unit = {0: neuron[in_a], 1: neuron[~in_a], 2: other}

    token = uuid.uuid4().hex[:12]
    raw_source = write_biological_nwb(
        tmp_path / f"similarity_merge_{token}.nwb",
        fixture_name=f"similarity_merge_{token}",
        duration_s=_DURATION_S,
        spike_sources=[
            (neuron, [130.0, 65.0, 25.0, 10.0]),
            (other, [15.0, 30.0, 120.0, 50.0]),
        ],
    )
    raw_clock, _ = _raw_acquisition_clock(raw_source)
    nwb = copy_and_insert_nwb(
        raw_source, dest_name=f"similarity_merge_{token}.nwb"
    )
    session = {"nwb_file_name": nwb}
    sort_pk = None
    try:
        initialize_v2_defaults()
        LabTeam.insert1(
            {"team_name": "v2_test_team", "team_description": "v2 merge"},
            skip_duplicates=True,
        )
        SortGroupV2.set_group_by_shank(nwb_file_name=nwb)
        sort_group_id = int(
            sorted((SortGroupV2 & session).fetch("sort_group_id"))[0]
        )
        rec_pk = RecordingSelection.insert_selection(
            {
                "nwb_file_name": nwb,
                "sort_group_id": sort_group_id,
                "interval_list_name": "raw data valid times",
                "preprocessing_params_name": "default",
                "team_name": "v2_test_team",
            }
        )
        Recording.populate(rec_pk, reserve_jobs=False)
        art_pk = RecordingArtifactSelection.insert_selection(
            {
                "recording_id": rec_pk["recording_id"],
                "artifact_detection_params_name": "none",
            }
        )
        RecordingArtifactDetection.populate(art_pk, reserve_jobs=False)
        sort_pk = SortingSelection.insert_selection(
            {
                "recording_id": rec_pk["recording_id"],
                "sorter": "mountainsort5",
                "sorter_params_name": "franklab_30khz_ms5_2026_06",
                "artifact_detection_id": art_pk["artifact_detection_id"],
            }
        )

        def known_sorter(**kwargs):
            return si.NumpySorting.from_unit_dict(frames_by_unit, _FS)

        plant_sorter(monkeypatch, known_sorter)
        Sorting.populate(sort_pk, reserve_jobs=False)

        root = CurationV2.insert_curation(sorting_key=sort_pk)
        sel = CurationEvaluationSelection.insert_selection(
            {
                **root,
                "metric_params_name": "minimal",
                "auto_curation_rules_name": "similarity_merge",
            }
        )
        CurationEvaluation.populate(sel, reserve_jobs=False)

        # Read back from the evaluation's NWB, not from the compute.
        assert CurationEvaluation.get_suggested_merge_groups(sel) == [[0, 1]]

        child = CurationEvaluation().accept_evaluation_outputs(
            sel, use_all_suggested_merges=True
        )
        abs_path = AnalysisNwbfile.get_abs_path(
            (CurationV2 & child).fetch1("analysis_file_name")
        )
        times, frames, _obs = read_units_abs_times_and_sample_indices(abs_path)
        merged_id = 3  # a merge mints max(unit_id) + 1
        assert sorted(frames) == [2, merged_id]
        np.testing.assert_array_equal(frames[merged_id], neuron)
        np.testing.assert_array_equal(times[merged_id], raw_clock[neuron])
        np.testing.assert_array_equal(frames[2], other)
    finally:
        if sort_pk is not None:
            clear_curations_for(sort_pk)
        _clean_session_v2(session)

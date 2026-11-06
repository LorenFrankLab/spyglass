"""Disjoint and multi-gap sort-interval behavior tests for the v2 single-session pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from tests.spikesorting.v2._ingest_helpers import _clean_session_v2
from tests.spikesorting.v2._sorter_stub import plant_sorter

# ---------- Boundary-spike round-trip (clip decision gate) -----------------


@pytest.mark.slow
def test_boundary_spike_round_trip_does_not_raise(
    polymer_smoke_session, monkeypatch
):
    """A spike at the recording's final sample survives the v2 NWB round-trip.

    ``get_sorting`` maps absolute spike times to frames with
    ``np.searchsorted``, clipping a spike that rounds to ``n_samples``. A
    planted spike at ``n_samples - 1`` must survive both
    ``Sorting.get_sorting`` and ``CurationV2.get_sorting`` without raising.
    """
    import numpy as np

    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.artifact import (
        RecordingArtifactDetection,
        RecordingArtifactSelection,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
        SortGroupV2,
    )
    from spyglass.spikesorting.v2.sorting import (
        Sorting,
        SortingSelection,
    )

    _clean_session_v2(polymer_smoke_session)
    initialize_v2_defaults()
    LabTeam.insert1(
        {
            "team_name": "v2_test_team",
            "team_description": "v2 pipeline tests",
        },
        skip_duplicates=True,
    )
    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
    sort_group_id = int(
        sorted((SortGroupV2 & polymer_smoke_session).fetch("sort_group_id"))[0]
    )
    rec_pk = RecordingSelection.insert_selection(
        {
            "nwb_file_name": nwb_file_name,
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
            "sorter": "clusterless_thresholder",
            "sorter_params_name": "default",
            "artifact_detection_id": art_pk["artifact_detection_id"],
        }
    )
    (Sorting & sort_pk).super_delete(warn=False)

    def _boundary_run_sorter(
        sorter,
        sorter_params,
        recording,
        sorting_id,
        *,
        job_kwargs=None,
        execution_params=None,
        statistics_spans=None,
    ):
        import spikeinterface as si

        n_samples = int(recording.get_num_samples())
        # The early spike keeps analyzer math from degenerating.
        samples = np.array([100, n_samples - 1], dtype=np.int64)
        labels = np.zeros(samples.size, dtype=np.int32)
        return si.NumpySorting.from_samples_and_labels(
            samples_list=[samples],
            labels_list=[labels],
            sampling_frequency=recording.get_sampling_frequency(),
        )

    plant_sorter(monkeypatch, _boundary_run_sorter)

    Sorting.populate(sort_pk, reserve_jobs=False)
    assert Sorting & sort_pk, (
        "Sorting.populate failed when the synthetic sorter produced "
        "a boundary spike; this is the boundary failure mode, inside "
        "write_sorting_units_nwb rather than at read time."
    )

    # Read path 1: Sorting.get_sorting.
    sorting_obj = Sorting().get_sorting(sort_pk)
    assert 0 in sorting_obj.get_unit_ids(), (
        "Boundary-spike sorting did not survive Sorting.get_sorting "
        f"round-trip; unit_ids={list(sorting_obj.get_unit_ids())}."
    )
    n_samples = Recording().get_recording(rec_pk).get_num_samples()
    expected_frames = [100, n_samples - 1]
    np.testing.assert_array_equal(
        sorting_obj.get_unit_spike_train(unit_id=0), expected_frames
    )

    # Read path 2: the curated NWB write, read back by CurationV2.get_sorting.
    curation_pk = CurationV2.insert_curation(
        sorting_key=sort_pk,
        labels={},
        parent_curation_id=-1,
        description="boundary-spike round-trip test",
    )
    curated = CurationV2().get_sorting(curation_pk)
    assert 0 in curated.get_unit_ids(), (
        "Boundary-spike unit lost on the CurationV2 round-trip; "
        f"curated unit_ids={list(curated.get_unit_ids())}."
    )
    np.testing.assert_array_equal(
        curated.get_unit_spike_train(unit_id=0), expected_frames
    )


@pytest.mark.slow
@pytest.mark.integration
def test_get_sorting_recovers_frames_across_disjoint_gap(
    polymer_smoke_session, monkeypatch
):
    """``get_sorting`` recovers original frames across a wall-clock gap.

    Spikes planted before and after the gap of a two-interval Recording must
    read back at the same frames through ``Sorting.get_sorting`` and
    ``CurationV2.get_sorting``. An affine readback
    (``round((t - t_start) * fs)``) would shift post-gap frames by
    ``gap * fs``; spike times round-trip either way, so this asserts on frames.
    """
    import uuid

    import numpy as np

    from spyglass.common.common_interval import IntervalList
    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.artifact import (
        RecordingArtifactDetection,
        RecordingArtifactSelection,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
        SortGroupV2,
    )
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection

    _clean_session_v2(polymer_smoke_session)
    initialize_v2_defaults()
    LabTeam.insert1(
        {"team_name": "v2_test_team", "team_description": "v2 pipeline tests"},
        skip_duplicates=True,
    )
    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)

    raw_times = (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": "raw data valid times",
        }
    ).fetch1("valid_times")
    t0 = float(raw_times[0][0])
    t_end = float(raw_times[-1][-1])
    assert (t_end - t0) >= 2.9, "smoke fixture too short for disjoint test"
    # Two 1.2 s chunks separated by a 0.5 s wall-clock gap.
    chunk1_end, gap_end, chunk2_end = t0 + 1.2, t0 + 1.7, min(t0 + 2.9, t_end)
    disjoint_times = np.array([[t0, chunk1_end], [gap_end, chunk2_end]])

    sort_group_id = int(
        sorted((SortGroupV2 & polymer_smoke_session).fetch("sort_group_id"))[0]
    )
    disjoint_name = f"v2_disjoint_readback_{uuid.uuid4().hex[:8]}"
    IntervalList.insert1(
        {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": disjoint_name,
            "valid_times": disjoint_times,
            "pipeline": "v2_disjoint_test",
        }
    )

    rec_pk = RecordingSelection.insert_selection(
        {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": sort_group_id,
            "interval_list_name": disjoint_name,
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
            "sorter": "clusterless_thresholder",
            "sorter_params_name": "default",
            "artifact_detection_id": art_pk["artifact_detection_id"],
        }
    )
    (Sorting & sort_pk).super_delete(warn=False)

    planted: dict = {}

    def _planted_run_sorter(
        sorter,
        sorter_params,
        recording,
        sorting_id,
        *,
        job_kwargs=None,
        execution_params=None,
        statistics_spans=None,
    ):
        import spikeinterface as si

        n_samples = int(recording.get_num_samples())
        samples = np.array([100, n_samples - 100], dtype=np.int64)
        rec_times = np.asarray(recording.get_times())
        planted["samples"] = samples
        # Must differ from the true frame, or the assertion below is vacuous.
        planted["affine_post_gap"] = int(
            round(
                (rec_times[samples[1]] - rec_times[0])
                * recording.get_sampling_frequency()
            )
        )
        labels = np.zeros(samples.size, dtype=np.int32)
        return si.NumpySorting.from_samples_and_labels(
            samples_list=[samples],
            labels_list=[labels],
            sampling_frequency=recording.get_sampling_frequency(),
        )

    plant_sorter(monkeypatch, _planted_run_sorter)
    Sorting.populate(sort_pk, reserve_jobs=False)
    assert Sorting & sort_pk, "disjoint Sorting.populate failed"
    assert planted["affine_post_gap"] != int(
        planted["samples"][1]
    ), "test setup: gap too small to expose the affine shift"

    si_sorting = Sorting().get_sorting(sort_pk)
    frames = np.asarray(si_sorting.get_unit_spike_train(unit_id=0))
    np.testing.assert_array_equal(
        np.sort(frames),
        np.sort(planted["samples"]),
        err_msg=(
            "Sorting.get_sorting did not recover the original frames "
            "across the wall-clock gap (affine readback shifts post-gap "
            f"frames). got={frames.tolist()}, "
            f"planted={planted['samples'].tolist()}"
        ),
    )

    curation_pk = CurationV2.insert_curation(
        sorting_key=sort_pk,
        labels={},
        parent_curation_id=-1,
        description="disjoint frame-readback test",
    )
    cur_sorting = CurationV2().get_sorting(curation_pk)
    cur_frames = np.asarray(cur_sorting.get_unit_spike_train(unit_id=0))
    np.testing.assert_array_equal(
        np.sort(cur_frames),
        np.sort(planted["samples"]),
        err_msg="CurationV2.get_sorting lost frames across the gap",
    )


@pytest.mark.slow
@pytest.mark.integration
def test_obs_intervals_no_artifact_respects_disjoint_gap(
    polymer_smoke_session, monkeypatch
):
    """No-artifact obs_intervals split at the gap on a disjoint recording.

    Without an artifact pass, each unit's ``obs_intervals`` is one interval
    per recorded chunk, not one envelope spanning the wall-clock gap.
    """
    import uuid

    import numpy as np
    import pynwb

    from spyglass.common.common_interval import IntervalList
    from spyglass.common.common_lab import LabTeam
    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
        SortGroupV2,
    )
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection

    _clean_session_v2(polymer_smoke_session)
    initialize_v2_defaults()
    LabTeam.insert1(
        {"team_name": "v2_test_team", "team_description": "v2 pipeline tests"},
        skip_duplicates=True,
    )
    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)

    raw_times = (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": "raw data valid times",
        }
    ).fetch1("valid_times")
    t0 = float(raw_times[0][0])
    t_end = float(raw_times[-1][-1])
    assert (t_end - t0) >= 2.9, "smoke fixture too short for disjoint test"
    chunk1_end, gap_end, chunk2_end = t0 + 1.2, t0 + 1.7, min(t0 + 2.9, t_end)
    gap_mid = 0.5 * (chunk1_end + gap_end)
    disjoint_times = np.array([[t0, chunk1_end], [gap_end, chunk2_end]])

    sort_group_id = int(
        sorted((SortGroupV2 & polymer_smoke_session).fetch("sort_group_id"))[0]
    )
    disjoint_name = f"v2_disjoint_obs_{uuid.uuid4().hex[:8]}"
    IntervalList.insert1(
        {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": disjoint_name,
            "valid_times": disjoint_times,
            "pipeline": "v2_disjoint_test",
        }
    )
    rec_pk = RecordingSelection.insert_selection(
        {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": sort_group_id,
            "interval_list_name": disjoint_name,
            "preprocessing_params_name": "default",
            "team_name": "v2_test_team",
        }
    )
    Recording.populate(rec_pk, reserve_jobs=False)

    # SortingSelection with NO artifact_detection_id -> obs_intervals=None fallback.
    sort_pk = SortingSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "sorter": "clusterless_thresholder",
            "sorter_params_name": "default",
        }
    )
    (Sorting & sort_pk).super_delete(warn=False)

    def _planted_run_sorter(
        sorter,
        sorter_params,
        recording,
        sorting_id,
        *,
        job_kwargs=None,
        execution_params=None,
        statistics_spans=None,
    ):
        import spikeinterface as si

        n_samples = int(recording.get_num_samples())
        samples = np.array([100, n_samples - 100], dtype=np.int64)
        labels = np.zeros(samples.size, dtype=np.int32)
        return si.NumpySorting.from_samples_and_labels(
            samples_list=[samples],
            labels_list=[labels],
            sampling_frequency=recording.get_sampling_frequency(),
        )

    plant_sorter(monkeypatch, _planted_run_sorter)
    Sorting.populate(sort_pk, reserve_jobs=False)
    assert Sorting & sort_pk, "disjoint no-artifact Sorting.populate failed"

    analysis_file_name = (Sorting & sort_pk).fetch1("analysis_file_name")
    abs_path = AnalysisNwbfile.get_abs_path(analysis_file_name)
    with pynwb.NWBHDF5IO(path=abs_path, mode="r", load_namespaces=True) as io:
        obs = np.asarray(io.read().units["obs_intervals"][0])

    assert obs.shape == (2, 2), (
        "no-artifact obs_intervals over a disjoint recording must be one "
        f"interval per recorded chunk; got {obs.tolist()}"
    )
    np.testing.assert_allclose(obs, disjoint_times, rtol=0, atol=1.0 / 30_000.0)
    for start, end in obs:
        assert not (start < gap_mid < end), (
            f"obs_interval [{start}, {end}] spans the inter-chunk gap "
            f"(gap_mid={gap_mid}); the envelope fallback was used."
        )


@pytest.mark.slow
@pytest.mark.integration
def test_reloaded_two_interval_artifact_exposes_gap(
    polymer_smoke_session, monkeypatch
):
    """The reloaded two-interval recording exposes its selection join, and the
    sort persists statistics spans that never cross it.

    ``boundary_spans_from_timestamps`` splits exactly at the join. With a
    manual exclusion at the start of the first interval, the persisted spans
    (also passed to the sorter) are the rest of the first interval, then the
    whole second interval.
    """
    import uuid

    import numpy as np

    from spyglass.common.common_interval import IntervalList
    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2._sorting.artifact_mask import (
        boundary_spans_from_timestamps,
    )
    from spyglass.spikesorting.v2.artifact import (
        RecordingArtifactDetection,
        RecordingArtifactSelection,
    )
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
        SortGroupV2,
    )
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection

    _clean_session_v2(polymer_smoke_session)
    initialize_v2_defaults()
    LabTeam.insert1(
        {"team_name": "v2_test_team", "team_description": "v2 pipeline tests"},
        skip_duplicates=True,
    )
    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)

    raw_times = (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": "raw data valid times",
        }
    ).fetch1("valid_times")
    t0 = float(raw_times[0][0])
    t_end = float(raw_times[-1][-1])
    assert (t_end - t0) >= 2.9, "smoke fixture too short for disjoint test"
    chunk1_end, gap_end, chunk2_end = t0 + 1.2, t0 + 1.7, min(t0 + 2.9, t_end)
    disjoint_name = f"v2_disjoint_spans_{uuid.uuid4().hex[:8]}"
    IntervalList.insert1(
        {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": disjoint_name,
            "valid_times": np.array([[t0, chunk1_end], [gap_end, chunk2_end]]),
            "pipeline": "v2_disjoint_test",
        }
    )
    sort_group_id = int(
        sorted((SortGroupV2 & polymer_smoke_session).fetch("sort_group_id"))[0]
    )
    rec_pk = RecordingSelection.insert_selection(
        {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": sort_group_id,
            "interval_list_name": disjoint_name,
            "preprocessing_params_name": "default",
            "team_name": "v2_test_team",
        }
    )
    Recording.populate(rec_pk, reserve_jobs=False)

    recording = Recording().get_recording(rec_pk)
    fs = float(recording.get_sampling_frequency())
    n_samples = int(recording.get_num_samples())
    # The first interval [t0, chunk1_end] is closed, so it holds every sample
    # t0 + k / fs with k <= 1.2 * fs.
    join = int(np.floor((chunk1_end - t0) * fs + 1e-6)) + 1
    assert boundary_spans_from_timestamps(recording) == [
        (0, join),
        (join, n_samples),
    ]
    times = np.asarray(recording.get_times())
    assert times[join - 1] <= chunk1_end < gap_end <= times[join]

    # Manually exclude the first 0.15 s. The "none" recipe drops kept
    # intervals shorter than its 1 s ``min_length_s``, so the exclusion sits
    # at an interval edge and leaves 1.05 s of the first interval.
    excl_start, excl_stop = 0, int(0.15 * fs)
    art_pk = RecordingArtifactSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "artifact_detection_params_name": "none",
            "manual_excluded_times": [
                [float(times[excl_start]), float(times[excl_stop])]
            ],
        }
    )
    RecordingArtifactDetection.populate(art_pk, reserve_jobs=False)
    sort_pk = SortingSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "sorter": "clusterless_thresholder",
            "sorter_params_name": "default",
            "artifact_detection_id": art_pk["artifact_detection_id"],
        }
    )
    (Sorting & sort_pk).super_delete(warn=False)

    received = {}

    def _capturing_run_sorter(
        sorter,
        sorter_params,
        recording,
        sorting_id,
        *,
        job_kwargs=None,
        execution_params=None,
        statistics_spans=None,
    ):
        import spikeinterface as si

        received["spans"] = statistics_spans
        received["traces"] = recording.get_traces()
        samples = np.array([100, n_samples - 100], dtype=np.int64)
        return si.NumpySorting.from_samples_and_labels(
            samples_list=[samples],
            labels_list=[np.zeros(samples.size, dtype=np.int32)],
            sampling_frequency=recording.get_sampling_frequency(),
        )

    plant_sorter(monkeypatch, _capturing_run_sorter)
    Sorting.populate(sort_pk, reserve_jobs=False)
    assert Sorting & sort_pk, "disjoint Sorting.populate failed"

    spans = Sorting().get_statistics_spans(sort_pk)
    assert received["spans"] == spans
    # The masked frames lead the first span; the join splits the rest.
    masked_stop = spans[0][0]
    assert abs(masked_stop - excl_stop) <= 1, spans
    assert spans == [(masked_stop, join), (join, n_samples)]
    np.testing.assert_array_equal(received["traces"][:masked_stop], 0)
    assert np.any(received["traces"][masked_stop:join] != 0)
    assert np.any(received["traces"][join:] != 0)


@pytest.mark.slow
@pytest.mark.integration
def test_get_merged_sorting_keeps_cross_gap_pair(
    polymer_smoke_session, monkeypatch
):
    """Lazy get_merged_sorting must not drop a cross-gap boundary pair.

    Spikes at chunk 1's last frame and chunk 2's first frame are
    frame-adjacent but ~0.5 s apart. Dedup is in absolute time, so both
    survive (a frame-space dedup like SI's MergeUnitsSorting drops one).
    """
    import uuid

    import numpy as np

    from spyglass.common.common_interval import IntervalList
    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
        SortGroupV2,
    )
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection

    _clean_session_v2(polymer_smoke_session)
    initialize_v2_defaults()
    LabTeam.insert1(
        {"team_name": "v2_test_team", "team_description": "v2 pipeline tests"},
        skip_duplicates=True,
    )
    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
    raw_times = (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": "raw data valid times",
        }
    ).fetch1("valid_times")
    t0 = float(raw_times[0][0])
    t_end = float(raw_times[-1][-1])
    assert (t_end - t0) >= 2.9, "smoke fixture too short for disjoint test"
    chunk1_end, gap_end, chunk2_end = t0 + 1.2, t0 + 1.7, min(t0 + 2.9, t_end)
    disjoint_times = np.array([[t0, chunk1_end], [gap_end, chunk2_end]])
    sort_group_id = int(
        sorted((SortGroupV2 & polymer_smoke_session).fetch("sort_group_id"))[0]
    )
    disjoint_name = f"v2_disjoint_merge_{uuid.uuid4().hex[:8]}"
    IntervalList.insert1(
        {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": disjoint_name,
            "valid_times": disjoint_times,
            "pipeline": "v2_disjoint_test",
        }
    )
    rec_pk = RecordingSelection.insert_selection(
        {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": sort_group_id,
            "interval_list_name": disjoint_name,
            "preprocessing_params_name": "default",
            "team_name": "v2_test_team",
        }
    )
    Recording.populate(rec_pk, reserve_jobs=False)
    sort_pk = SortingSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "sorter": "clusterless_thresholder",
            "sorter_params_name": "default",
        }
    )
    (Sorting & sort_pk).super_delete(warn=False)

    planted = {}

    def _two_unit_gap_boundary_sorter(
        sorter,
        sorter_params,
        recording,
        sorting_id,
        *,
        job_kwargs=None,
        execution_params=None,
        statistics_spans=None,
    ):
        import spikeinterface as si

        ts = np.asarray(recording.get_times())
        fs_local = recording.get_sampling_frequency()
        k = int(np.flatnonzero(np.diff(ts) > 1.5 / fs_local)[0])  # chunk1 last
        # unit 0: a chunk-1 spike + chunk-1's LAST frame; unit 1: chunk-2's
        # FIRST frame + a later chunk-2 spike. Frames k and k+1 are adjacent
        # but separated by the wall-clock gap.
        planted["k"] = k
        u0 = np.array([100, k], dtype=np.int64)
        u1 = np.array([k + 1, k + 1 + 100], dtype=np.int64)
        return si.NumpySorting.from_unit_dict(
            [{0: u0, 1: u1}], sampling_frequency=fs_local
        )

    plant_sorter(monkeypatch, _two_unit_gap_boundary_sorter)
    Sorting.populate(sort_pk, reserve_jobs=False)
    assert Sorting & sort_pk

    # apply_merge=False preview curation merging the two units, then lazy.
    pk = CurationV2.insert_curation(
        sorting_key=sort_pk,
        labels={},
        merge_groups=[[0, 1]],
        apply_merge=False,
        parent_curation_id=-1,
        description="disjoint cross-gap merge test",
    )
    merged = CurationV2().get_merged_sorting(pk)
    assert merged.get_num_units() == 1, "the two units should merge into one"
    merged_frames = set(
        int(f)
        for uid in merged.get_unit_ids()
        for f in merged.get_unit_spike_train(unit_id=uid)
    )
    k = planted["k"]
    # Both gap-boundary frames survive (frame-space dedup would drop one).
    assert k in merged_frames and (k + 1) in merged_frames, (
        f"cross-gap boundary pair (frames {k}, {k + 1}) was not preserved; "
        f"merged frames={sorted(merged_frames)}"
    )
    assert len(merged_frames) == 4, (
        f"all 4 planted spikes should survive (no cross-gap dedup); "
        f"got {sorted(merged_frames)}"
    )


@pytest.mark.slow
@pytest.mark.integration
def test_disjoint_sort_intervals_concatenated(polymer_smoke_session):
    """``Recording.make`` excludes the gap between disjoint sort intervals.

    The written timestamps are shorter than the gap-inclusive envelope and
    none fall inside the gap.
    """
    import uuid

    import numpy as np
    import pynwb

    from spyglass.common.common_interval import IntervalList
    from spyglass.common.common_lab import LabTeam
    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2.recording import (
        PreprocessingParameters,
        Recording,
        RecordingSelection,
        SortGroupV2,
    )

    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    PreprocessingParameters.insert_default()
    LabTeam.insert1(
        {
            "team_name": "v2_test_team",
            "team_description": "v2 pipeline tests",
        },
        skip_duplicates=True,
    )
    if not (SortGroupV2 & polymer_smoke_session):
        SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)

    raw_times = (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": "raw data valid times",
        }
    ).fetch1("valid_times")
    t0 = float(raw_times[0][0])
    t_end = float(raw_times[-1][-1])
    total = t_end - t0
    # Chunks must exceed the 1.0 s min_segment_length (an exact 1.0 s chunk
    # can drop on float boundaries): 1.2 s chunks, 0.5 s gap, 2.9 s total.
    assert total >= 2.9, (
        f"Smoke fixture is too short ({total}s) for the disjoint "
        "test; need at least 2.9s."
    )
    chunk1_end = t0 + 1.2
    gap_end = t0 + 1.7
    chunk2_end = min(t0 + 2.9, t_end)
    disjoint_times = np.array([[t0, chunk1_end], [gap_end, chunk2_end]])

    sort_group_id = int(
        sorted((SortGroupV2 & polymer_smoke_session).fetch("sort_group_id"))[0]
    )
    disjoint_interval_name = f"v2_disjoint_test_{uuid.uuid4().hex[:8]}"
    IntervalList.insert1(
        {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": disjoint_interval_name,
            "valid_times": disjoint_times,
            "pipeline": "v2_disjoint_test",
        }
    )

    rec_pk = RecordingSelection.insert_selection(
        {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": sort_group_id,
            "interval_list_name": disjoint_interval_name,
            "preprocessing_params_name": "default",
            "team_name": "v2_test_team",
        }
    )
    Recording.populate(rec_pk, reserve_jobs=False)
    row = (Recording & rec_pk).fetch1()
    abs_path = AnalysisNwbfile.get_abs_path(row["analysis_file_name"])

    with pynwb.NWBHDF5IO(path=abs_path, mode="r", load_namespaces=True) as io:
        nwbf = io.read()
        series_path = row["electrical_series_path"]
        series_name = series_path.rsplit("/", 1)[-1]
        series = nwbf.acquisition[series_name]
        written_times = np.asarray(series.timestamps[:])

    # Expect (chunk1 + chunk2 duration) * fs, within 5% for boundary rounding.
    fs = float(row["sampling_frequency"])
    expected_n = int(((chunk1_end - t0) + (chunk2_end - gap_end)) * fs)
    assert 0.95 * expected_n <= len(written_times) <= 1.05 * expected_n, (
        f"Written timestamps length {len(written_times)} differs "
        f"from expected ~{expected_n} (sum of disjoint chunks). "
        "Disjoint-interval concat regression."
    )
    # No written timestamp falls inside the gap, with 1.5 / fs slack.
    tol = 1.5 / fs
    in_gap = (written_times > chunk1_end + tol) & (
        written_times < gap_end - tol
    )
    assert not np.any(in_gap), (
        f"Found {int(in_gap.sum())} written timestamps inside the "
        f"disjoint gap ({chunk1_end}, {gap_end}); the "
        "restrict_recording_times frame_slice + concatenate_recordings "
        "split was bypassed."
    )


@pytest.mark.slow
@pytest.mark.integration
def test_artifact_valid_times_respect_disjoint_gap(polymer_smoke_session):
    """Artifact-removed valid_times never span a gap.

    With the ``none`` preset on a two-chunk Recording, the persisted artifact
    ``IntervalList`` splits at the gap. ``test_disjoint_artifact.py`` covers
    the detect=True path.
    """
    import uuid

    import numpy as np

    from spyglass.common.common_interval import IntervalList
    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2.artifact import (
        RecordingArtifactDetection,
        ArtifactDetectionParameters,
        RecordingArtifactSelection,
    )
    from spyglass.spikesorting.v2.recording import (
        PreprocessingParameters,
        Recording,
        RecordingSelection,
        SortGroupV2,
    )

    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    PreprocessingParameters.insert_default()
    ArtifactDetectionParameters.insert_default()
    LabTeam.insert1(
        {"team_name": "v2_test_team", "team_description": "v2 pipeline tests"},
        skip_duplicates=True,
    )
    if not (SortGroupV2 & polymer_smoke_session):
        SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)

    raw_times = (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": "raw data valid times",
        }
    ).fetch1("valid_times")
    t0 = float(raw_times[0][0])
    t_end = float(raw_times[-1][-1])
    assert (t_end - t0) >= 2.9, "smoke fixture too short for disjoint test"
    chunk1_end, gap_end, chunk2_end = t0 + 1.2, t0 + 1.7, min(t0 + 2.9, t_end)
    gap_mid = 0.5 * (chunk1_end + gap_end)
    disjoint_times = np.array([[t0, chunk1_end], [gap_end, chunk2_end]])

    sort_group_id = int(
        sorted((SortGroupV2 & polymer_smoke_session).fetch("sort_group_id"))[0]
    )
    disjoint_name = f"v2_disjoint_artifact_{uuid.uuid4().hex[:8]}"
    IntervalList.insert1(
        {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": disjoint_name,
            "valid_times": disjoint_times,
            "pipeline": "v2_disjoint_test",
        }
    )
    rec_pk = RecordingSelection.insert_selection(
        {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": sort_group_id,
            "interval_list_name": disjoint_name,
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

    valid_times = RecordingArtifactDetection().get_artifact_removed_intervals(
        art_pk
    )

    assert len(valid_times) >= 2, (
        "disjoint recording must yield >= 2 artifact valid intervals "
        f"(one per chunk); got {valid_times.tolist()}"
    )
    for start, end in valid_times:
        assert not (start < gap_mid < end), (
            f"artifact valid interval [{start}, {end}] spans the "
            f"inter-chunk gap (gap_mid={gap_mid})."
        )


@pytest.mark.slow
@pytest.mark.integration
def test_disjoint_multi_gap_readback_and_artifact(
    polymer_smoke_session, monkeypatch
):
    """Spike readback and artifact valid_times are gap-correct across two gaps.

    On a three-chunk recording, artifact valid_times hold one interval per
    chunk and ``get_sorting`` recovers a spike planted in each chunk exactly.
    """
    import uuid

    from spyglass.common.common_interval import IntervalList
    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.artifact import (
        RecordingArtifactDetection,
        RecordingArtifactSelection,
    )
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
        SortGroupV2,
    )
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection

    _clean_session_v2(polymer_smoke_session)
    initialize_v2_defaults()
    LabTeam.insert1(
        {"team_name": "v2_test_team", "team_description": "v2 pipeline tests"},
        skip_duplicates=True,
    )
    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
    sort_group_id = int(
        sorted((SortGroupV2 & polymer_smoke_session).fetch("sort_group_id"))[0]
    )

    raw_times = (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": "raw data valid times",
        }
    ).fetch1("valid_times")
    t0 = float(raw_times[0][0])
    t_end = float(raw_times[-1][-1])
    # Three 1.05 s chunks separated by two 0.25 s gaps (3.65 s total).
    assert (
        t_end - t0
    ) >= 3.7, "smoke fixture too short for a 3-chunk (2-gap) disjoint test"
    c1 = (t0, t0 + 1.05)
    c2 = (t0 + 1.30, t0 + 2.35)
    c3 = (t0 + 2.60, t0 + 3.65)
    disjoint_times = np.array([list(c1), list(c2), list(c3)])

    disjoint_name = f"v2_multigap_{uuid.uuid4().hex[:8]}"
    IntervalList.insert1(
        {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": disjoint_name,
            "valid_times": disjoint_times,
            "pipeline": "v2_multi_gap_test",
        }
    )
    rec_pk = RecordingSelection.insert_selection(
        {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": sort_group_id,
            "interval_list_name": disjoint_name,
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

    # Artifact valid_times: detect=False ("none") returns the recorded
    # window(s) split per chunk -> exactly 3 intervals, none spanning a gap.
    interval_name = f"artifact_detection_{art_pk['artifact_detection_id']}"
    saved = (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": interval_name,
        }
    ).fetch1("valid_times")
    assert saved.shape == (3, 2), (
        f"expected one valid interval per chunk (3, 2); got {saved.shape}. "
        "base_intervals_and_gaps mis-split a multi-gap timeline."
    )
    gap1_mid = 0.5 * (c1[1] + c2[0])
    gap2_mid = 0.5 * (c2[1] + c3[0])
    for start, end in saved:
        assert not (start < gap1_mid < end), "valid interval spans gap 1"
        assert not (start < gap2_mid < end), "valid interval spans gap 2"

    # Plant one spike per chunk and read all three back exactly.
    sort_pk = SortingSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "sorter": "clusterless_thresholder",
            "sorter_params_name": "default",
            "artifact_detection_id": art_pk["artifact_detection_id"],
        }
    )
    rec = Recording().get_recording(rec_pk)
    times = np.asarray(rec.get_times())
    fs = rec.get_sampling_frequency()
    gaps = np.flatnonzero(np.diff(times) > 1.5 / fs)
    assert len(gaps) == 2, f"expected two gaps, found {len(gaps)}"
    k1, k2 = int(gaps[0]), int(gaps[1])  # last frame of chunks 1, 2
    planted_frames = np.sort(np.array([50, k1 + 10, k2 + 10], dtype=np.int64))
    # Sanity: one frame in each chunk.
    assert 50 < k1 and k1 + 10 < k2 and k2 + 10 < times.size

    def _planted_run_sorter(
        sorter,
        sorter_params,
        recording,
        sorting_id,
        *,
        job_kwargs=None,
        execution_params=None,
        statistics_spans=None,
    ):
        import spikeinterface as si

        labels = np.zeros(planted_frames.size, dtype=np.int32)
        return si.NumpySorting.from_samples_and_labels(
            samples_list=[planted_frames],
            labels_list=[labels],
            sampling_frequency=recording.get_sampling_frequency(),
        )

    plant_sorter(monkeypatch, _planted_run_sorter)
    Sorting.populate(sort_pk, reserve_jobs=False)
    si_sorting = Sorting().get_sorting(sort_pk)
    frames = np.sort(np.asarray(si_sorting.get_unit_spike_train(unit_id=0)))
    np.testing.assert_array_equal(
        frames,
        planted_frames,
        err_msg=(
            "get_sorting did not recover all planted frames across two "
            f"gaps. got={frames.tolist()}, planted={planted_frames.tolist()}"
        ),
    )

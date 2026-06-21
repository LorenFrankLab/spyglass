"""Disjoint and multi-gap sort-interval behavior tests for the v2 single-session pipeline."""

from __future__ import annotations

import numpy as _np
import pytest

from tests.spikesorting.v2._ingest_helpers import _clean_session_v2


# ---------- Boundary-spike round-trip (clip decision gate) -----------------


@pytest.mark.slow
def test_boundary_spike_round_trip_does_not_raise(
    polymer_smoke_session, monkeypatch
):
    """A spike at the recording's final sample survives the v2 NWB round-trip.

    Regression guard for the final-sample boundary. ``_write_units_nwb``
    stores ``timestamps[sample_index]`` as the absolute spike time;
    ``get_sorting`` reads it back by mapping absolute time -> frame with
    ``np.searchsorted`` (``_spike_times_to_frames``), which clips any
    spike that floating-point-rounds to ``n_samples`` (one past the end)
    -- the v1 ``spike_times_to_valid_samples`` clip, now folded in. This
    test pins that a spike at ``n_samples - 1`` survives both read paths
    without raising.

    Construction strategy
    ---------------------
    The shipped sorters do not deterministically produce a boundary
    spike, so we monkey-patch ``Sorting._run_sorter`` to return a
    hand-built ``NumpySorting`` with one unit whose spike train
    includes ``n_samples - 1`` and one earlier in-bounds spike. The
    rest of ``Sorting.make`` runs normally:
    ``_remove_excess_spikes`` should keep both samples
    (``n_samples - 1 < n_samples``), ``_write_units_nwb`` writes
    ``timestamps[sample_index]`` as the absolute time, and
    ``Sorting().get_sorting`` reads back via the searchsorted map.
    The round-trip is also exercised through
    ``CurationV2.insert_curation`` + ``CurationV2.get_sorting`` so
    both production read paths are covered.
    """
    import numpy as _np

    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetection,
        ArtifactDetectionSelection,
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

    art_pk = ArtifactDetectionSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "artifact_detection_params_name": "none",
        }
    )
    ArtifactDetection.populate(art_pk, reserve_jobs=False)

    sort_pk = SortingSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "sorter": "clusterless_thresholder",
            "sorter_params_name": "default",
            "artifact_detection_id": art_pk["artifact_detection_id"],
        }
    )
    (Sorting & sort_pk).super_delete(warn=False)

    # Monkey-patch ``_run_sorter`` to deterministically return a
    # ``NumpySorting`` with a spike at exactly ``n_samples - 1`` on
    # the artifact-masked recording. ``_run_sorter`` is a
    # ``@staticmethod`` so we patch the class attribute directly.
    def _boundary_run_sorter(
        sorter, sorter_params, recording, sorting_id, *, job_kwargs=None
    ):
        import spikeinterface as si

        n_samples = int(recording.get_num_samples())
        # Place one spike near the start and one at the very last
        # sample. The boundary one is what tests SI's read-side
        # bounds check; the earlier one keeps the unit "real" so
        # downstream analyzer math doesn't degenerate.
        samples = _np.array([100, n_samples - 1], dtype=_np.int64)
        labels = _np.zeros(samples.size, dtype=_np.int32)
        return si.NumpySorting.from_samples_and_labels(
            samples_list=[samples],
            labels_list=[labels],
            sampling_frequency=recording.get_sampling_frequency(),
        )

    monkeypatch.setattr(
        Sorting, "_run_sorter", staticmethod(_boundary_run_sorter)
    )

    Sorting.populate(sort_pk, reserve_jobs=False)
    assert Sorting & sort_pk, (
        "Sorting.populate failed when the synthetic sorter produced "
        "a boundary spike; this is the boundary failure mode, inside "
        "_write_units_nwb rather than at read time."
    )

    # First read path: Sorting.get_sorting -> searchsorted frame map
    # must build without raising. Unit set must include the one we
    # planted.
    sorting_obj = Sorting().get_sorting(sort_pk)
    assert 0 in sorting_obj.get_unit_ids(), (
        "Boundary-spike sorting did not survive Sorting.get_sorting "
        f"round-trip; unit_ids={list(sorting_obj.get_unit_ids())}."
    )

    # Second read path: CurationV2 + CurationV2.get_sorting. The
    # curated NWB write goes through a different code path than
    # Sorting._write_units_nwb but reads back via the same
    # searchsorted frame map. Pass labels={} so insert_curation
    # accepts the call on the strict signature.
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


@pytest.mark.slow
@pytest.mark.integration
def test_get_sorting_recovers_frames_across_disjoint_gap(
    polymer_smoke_session, monkeypatch
):
    """``get_sorting`` recovers original frames across a wall-clock gap.

    Builds a Recording over two DISJOINT sort intervals -- so the
    persisted timeline is gap-preserving (non-uniform) -- and
    monkeypatches ``_run_sorter`` to plant a spike near the start
    (chunk 1) and one near the end (chunk 2, after the gap). Both
    ``Sorting.get_sorting`` and ``CurationV2.get_sorting`` must read the
    planted FRAME indices back exactly.

    Regression for the v1-parity readback bug: the old
    ``NwbSortingExtractor`` readback inverts the stored absolute spike
    times affinely (``round((t - t_start) * fs)``), which shifts every
    frame after the gap by ``gap * fs`` (here landing past the
    gap-excluded sample count). v1 -- and now v2 -- map back with
    ``np.searchsorted`` against the actual recording timestamps, which
    recovers the original frame. The spike-TIME surfaces round-trip
    correctly either way; only the FRAME indices expose the bug, so this
    test asserts on frames.
    """
    import uuid as _uuid

    import numpy as _np

    from spyglass.common.common_interval import IntervalList
    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetection,
        ArtifactDetectionSelection,
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
    disjoint_times = _np.array([[t0, chunk1_end], [gap_end, chunk2_end]])

    sort_group_id = int(
        sorted((SortGroupV2 & polymer_smoke_session).fetch("sort_group_id"))[0]
    )
    disjoint_name = f"v2_disjoint_readback_{_uuid.uuid4().hex[:8]}"
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

    art_pk = ArtifactDetectionSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "artifact_detection_params_name": "none",
        }
    )
    ArtifactDetection.populate(art_pk, reserve_jobs=False)

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
        sorter, sorter_params, recording, sorting_id, *, job_kwargs=None
    ):
        import spikeinterface as si

        n_samples = int(recording.get_num_samples())
        samples = _np.array([100, n_samples - 100], dtype=_np.int64)
        rec_times = _np.asarray(recording.get_times())
        planted["samples"] = samples
        # The affine inverse of the post-gap frame's absolute time; if
        # this still equals the original frame the gap is too small to
        # expose the bug, so the assertion below would be vacuous.
        planted["affine_post_gap"] = int(
            round(
                (rec_times[samples[1]] - rec_times[0])
                * recording.get_sampling_frequency()
            )
        )
        labels = _np.zeros(samples.size, dtype=_np.int32)
        return si.NumpySorting.from_samples_and_labels(
            samples_list=[samples],
            labels_list=[labels],
            sampling_frequency=recording.get_sampling_frequency(),
        )

    monkeypatch.setattr(
        Sorting, "_run_sorter", staticmethod(_planted_run_sorter)
    )
    Sorting.populate(sort_pk, reserve_jobs=False)
    assert Sorting & sort_pk, "disjoint Sorting.populate failed"
    assert planted["affine_post_gap"] != int(
        planted["samples"][1]
    ), "test setup: gap too small to expose the affine shift"

    si_sorting = Sorting().get_sorting(sort_pk)
    frames = _np.asarray(si_sorting.get_unit_spike_train(unit_id=0))
    _np.testing.assert_array_equal(
        _np.sort(frames),
        _np.sort(planted["samples"]),
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
    cur_frames = _np.asarray(cur_sorting.get_unit_spike_train(unit_id=0))
    _np.testing.assert_array_equal(
        _np.sort(cur_frames),
        _np.sort(planted["samples"]),
        err_msg="CurationV2.get_sorting lost frames across the gap",
    )


@pytest.mark.slow
@pytest.mark.integration
def test_obs_intervals_no_artifact_respects_disjoint_gap(
    polymer_smoke_session, monkeypatch
):
    """No-artifact obs_intervals split at the gap on a disjoint recording.

    When a sort has NO artifact pass (``artifact_detection_id`` is None),
    ``_write_units_nwb`` falls back to the recording's recorded window(s)
    for each unit's ``obs_intervals``. On a DISJOINT recording that must
    be one interval per recorded chunk, NOT a single envelope spanning the
    wall-clock gap (which would inflate the observation duration /
    firing-rate window). The artifact-backed path was already gap-split;
    this pins the no-ArtifactDetectionSource fallback.
    """
    import uuid as _uuid

    import numpy as _np
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
    disjoint_times = _np.array([[t0, chunk1_end], [gap_end, chunk2_end]])

    sort_group_id = int(
        sorted((SortGroupV2 & polymer_smoke_session).fetch("sort_group_id"))[0]
    )
    disjoint_name = f"v2_disjoint_obs_{_uuid.uuid4().hex[:8]}"
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
        sorter, sorter_params, recording, sorting_id, *, job_kwargs=None
    ):
        import spikeinterface as si

        n_samples = int(recording.get_num_samples())
        samples = _np.array([100, n_samples - 100], dtype=_np.int64)
        labels = _np.zeros(samples.size, dtype=_np.int32)
        return si.NumpySorting.from_samples_and_labels(
            samples_list=[samples],
            labels_list=[labels],
            sampling_frequency=recording.get_sampling_frequency(),
        )

    monkeypatch.setattr(
        Sorting, "_run_sorter", staticmethod(_planted_run_sorter)
    )
    Sorting.populate(sort_pk, reserve_jobs=False)
    assert Sorting & sort_pk, "disjoint no-artifact Sorting.populate failed"

    analysis_file_name = (Sorting & sort_pk).fetch1("analysis_file_name")
    abs_path = AnalysisNwbfile.get_abs_path(analysis_file_name)
    with pynwb.NWBHDF5IO(path=abs_path, mode="r", load_namespaces=True) as io:
        obs = _np.asarray(io.read().units["obs_intervals"][0])

    assert obs.shape == (2, 2), (
        "no-artifact obs_intervals over a disjoint recording must be one "
        f"interval per recorded chunk; got {obs.tolist()}"
    )
    for start, end in obs:
        assert not (start < gap_mid < end), (
            f"obs_interval [{start}, {end}] spans the inter-chunk gap "
            f"(gap_mid={gap_mid}); the envelope fallback was used."
        )


@pytest.mark.slow
@pytest.mark.integration
def test_get_merged_sorting_keeps_cross_gap_pair(
    polymer_smoke_session, monkeypatch
):
    """Lazy get_merged_sorting must not drop a cross-gap boundary pair.

    On a DISJOINT recording the units NWB frames are contiguous across the
    excluded wall-clock gap, so two merged contributors firing at chunk 1's
    last frame and chunk 2's first frame are frame-ADJACENT but ~0.5 s apart
    in real time. A frame-space 0.4 ms dedup (SI's MergeUnitsSorting) would
    wrongly drop one; the abs-time dedup keeps both (and matches the
    apply_merge=True stored train). Regression for the lazy-merge fix.
    """
    import uuid as _uuid

    import numpy as _np

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
    disjoint_times = _np.array([[t0, chunk1_end], [gap_end, chunk2_end]])
    sort_group_id = int(
        sorted((SortGroupV2 & polymer_smoke_session).fetch("sort_group_id"))[0]
    )
    disjoint_name = f"v2_disjoint_merge_{_uuid.uuid4().hex[:8]}"
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
        sorter, sorter_params, recording, sorting_id, *, job_kwargs=None
    ):
        import spikeinterface as si

        ts = _np.asarray(recording.get_times())
        fs_local = recording.get_sampling_frequency()
        k = int(
            _np.flatnonzero(_np.diff(ts) > 1.5 / fs_local)[0]
        )  # chunk1 last
        # unit 0: a chunk-1 spike + chunk-1's LAST frame; unit 1: chunk-2's
        # FIRST frame + a later chunk-2 spike. Frames k and k+1 are adjacent
        # but separated by the wall-clock gap.
        planted["k"] = k
        u0 = _np.array([100, k], dtype=_np.int64)
        u1 = _np.array([k + 1, k + 1 + 100], dtype=_np.int64)
        return si.NumpySorting.from_unit_dict(
            [{0: u0, 1: u1}], sampling_frequency=fs_local
        )

    monkeypatch.setattr(
        Sorting, "_run_sorter", staticmethod(_two_unit_gap_boundary_sorter)
    )
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
    """``Recording.make`` honors disjoint sort intervals.

    Without the ``_consolidate_intervals`` + ``concatenate_recordings``
    pattern, ``Recording.make`` took ``(times[0][0], times[-1][-1])``
    -- the outer envelope, silently including inter-interval gaps.
    This test writes a synthetic IntervalList row whose
    ``valid_times`` has two disjoint chunks with a deliberate gap,
    populates Recording, then re-reads the written
    ``ElectricalSeries.timestamps`` and asserts:

    1. The written timestamps span is SHORTER than the gap-inclusive
       envelope (the gap was actually excluded).
    2. No written timestamp falls inside the gap.

    Without the consolidate-and-concatenate pattern, both
    assertions fail (the gap is included).
    """
    import uuid as _uuid

    import numpy as _np
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
    # Build two disjoint sub-intervals inside the raw window with
    # a deliberate gap. min_segment_length default is 1.0 s, so
    # each chunk must be >= 1s for the helper not to drop them.
    total = t_end - t0
    # min_segment_length default is 1.0 s; intersect's per-chunk
    # length must be strictly > 1.0 s to survive (floating-point
    # boundaries on a 1.0 s-exact chunk are sometimes dropped).
    # Use 1.2 s chunks separated by 0.5 s gap, total 2.9 s --
    # fits in the 4 s smoke fixture.
    assert total >= 2.9, (
        f"Smoke fixture is too short ({total}s) for the disjoint "
        "test; need at least 2.9s."
    )
    chunk1_end = t0 + 1.2
    gap_end = t0 + 1.7
    chunk2_end = min(t0 + 2.9, t_end)
    disjoint_times = _np.array([[t0, chunk1_end], [gap_end, chunk2_end]])

    sort_group_id = int(
        sorted((SortGroupV2 & polymer_smoke_session).fetch("sort_group_id"))[0]
    )
    disjoint_interval_name = f"v2_disjoint_test_{_uuid.uuid4().hex[:8]}"
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
        written_times = _np.asarray(series.timestamps[:])

    # The written length should be approximately
    # (chunk1 duration + chunk2 duration) * fs, not the full
    # envelope. Allow 5% slack for fs jitter and boundary
    # rounding.
    fs = float(row["sampling_frequency"])
    expected_n = int(((chunk1_end - t0) + (chunk2_end - gap_end)) * fs)
    assert 0.95 * expected_n <= len(written_times) <= 1.05 * expected_n, (
        f"Written timestamps length {len(written_times)} differs "
        f"from expected ~{expected_n} (sum of disjoint chunks). "
        "Disjoint-interval concat regression."
    )
    # No written timestamp falls inside the gap (chunk1_end,
    # gap_end). Allow tiny boundary slack via 1.5 / fs.
    tol = 1.5 / fs
    in_gap = (written_times > chunk1_end + tol) & (
        written_times < gap_end - tol
    )
    assert not _np.any(in_gap), (
        f"Found {int(in_gap.sum())} written timestamps inside the "
        f"disjoint gap ({chunk1_end}, {gap_end}); the "
        "_consolidate_intervals + concatenate_recordings split was "
        "bypassed."
    )


@pytest.mark.slow
@pytest.mark.integration
def test_artifact_valid_times_respect_disjoint_gap(polymer_smoke_session):
    """End-to-end: artifact-removed valid_times never span a gap.

    Builds a disjoint Recording (two chunks separated by a wall-clock
    gap), runs ``ArtifactDetection`` with the ``none`` preset
    (detect=False), and asserts the persisted artifact ``IntervalList``
    valid_times split at the gap rather than returning one envelope
    spanning it. Subtracting/returning over a single
    ``[timestamps[0], timestamps[-1]]`` envelope (the old behavior) would
    reintroduce the gap -- inflating obs_intervals duration and letting
    sub-min_length slivers survive. The artifact-detected (detect=True)
    per-chunk subtraction is pinned by the synthetic ``_detect_artifacts``
    test in ``test_disjoint_artifact.py``.
    """
    import uuid as _uuid

    import numpy as _np

    from spyglass.common.common_interval import IntervalList
    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetection,
        ArtifactDetectionParameters,
        ArtifactDetectionSelection,
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
    disjoint_times = _np.array([[t0, chunk1_end], [gap_end, chunk2_end]])

    sort_group_id = int(
        sorted((SortGroupV2 & polymer_smoke_session).fetch("sort_group_id"))[0]
    )
    disjoint_name = f"v2_disjoint_artifact_{_uuid.uuid4().hex[:8]}"
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

    art_pk = ArtifactDetectionSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "artifact_detection_params_name": "none",
        }
    )
    ArtifactDetection.populate(art_pk, reserve_jobs=False)

    valid_times = ArtifactDetection().get_artifact_removed_intervals(art_pk)

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
    """Spike readback + artifact valid_times are gap-correct across MULTIPLE
    gaps (3 chunks, 2 gaps), not just the first.

    Every other disjoint DB test is single-gap (two chunks). This builds a
    three-chunk recording and asserts: ``_base_intervals_from_timestamps``
    (exercised via the populated timeline) yields one valid interval per
    chunk -- none spanning either gap -- and ``get_sorting`` recovers a
    spike planted in EACH chunk exactly (the per-chunk ``searchsorted``
    readback must stay correct past the second gap, not only the first).
    Closes review "not-checked" item #6.
    """
    import uuid as _uuid

    from spyglass.common.common_interval import IntervalList
    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetection,
        ArtifactDetectionSelection,
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
    if (t_end - t0) < 3.7:
        pytest.skip(
            f"smoke fixture window {t_end - t0:.2f}s too short for a "
            "3-chunk (2-gap) disjoint test (need >= 3.7s)."
        )
    c1 = (t0, t0 + 1.05)
    c2 = (t0 + 1.30, t0 + 2.35)
    c3 = (t0 + 2.60, t0 + 3.65)
    disjoint_times = _np.array([list(c1), list(c2), list(c3)])

    disjoint_name = f"v2_multigap_{_uuid.uuid4().hex[:8]}"
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

    art_pk = ArtifactDetectionSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "artifact_detection_params_name": "none",
        }
    )
    ArtifactDetection.populate(art_pk, reserve_jobs=False)

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
        "_base_intervals_from_timestamps mis-split a multi-gap timeline."
    )
    gap1_mid = 0.5 * (c1[1] + c2[0])
    gap2_mid = 0.5 * (c2[1] + c3[0])
    for start, end in saved:
        assert not (start < gap1_mid < end), "valid interval spans gap 1"
        assert not (start < gap2_mid < end), "valid interval spans gap 2"

    # Plant one spike in EACH chunk, recover the boundary frames from the
    # populated timeline, then assert get_sorting reads all three back
    # exactly (past BOTH gaps).
    sort_pk = SortingSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "sorter": "clusterless_thresholder",
            "sorter_params_name": "default",
            "artifact_detection_id": art_pk["artifact_detection_id"],
        }
    )
    rec = Recording().get_recording(rec_pk)
    times = _np.asarray(rec.get_times())
    fs = rec.get_sampling_frequency()
    gaps = _np.flatnonzero(_np.diff(times) > 1.5 / fs)
    assert len(gaps) == 2, f"expected two gaps, found {len(gaps)}"
    k1, k2 = int(gaps[0]), int(gaps[1])  # last frame of chunks 1, 2
    planted_frames = _np.sort(
        _np.array([50, k1 + 10, k2 + 10], dtype=_np.int64)
    )
    # Sanity: one frame in each chunk.
    assert 50 < k1 and k1 + 10 < k2 and k2 + 10 < times.size

    def _planted_run_sorter(
        sorter, sorter_params, recording, sorting_id, *, job_kwargs=None
    ):
        import spikeinterface as si

        labels = _np.zeros(planted_frames.size, dtype=_np.int32)
        return si.NumpySorting.from_samples_and_labels(
            samples_list=[planted_frames],
            labels_list=[labels],
            sampling_frequency=recording.get_sampling_frequency(),
        )

    monkeypatch.setattr(
        Sorting, "_run_sorter", staticmethod(_planted_run_sorter)
    )
    Sorting.populate(sort_pk, reserve_jobs=False)
    si_sorting = Sorting().get_sorting(sort_pk)
    frames = _np.sort(_np.asarray(si_sorting.get_unit_spike_train(unit_id=0)))
    _np.testing.assert_array_equal(
        frames,
        planted_frames,
        err_msg=(
            "get_sorting did not recover all planted frames across two "
            f"gaps. got={frames.tolist()}, planted={planted_frames.tolist()}"
        ),
    )

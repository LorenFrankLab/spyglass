"""Every consumer of a motion-corrected sort reads the corrected traces.

A sort that selected a ``MotionCorrectedRecording`` must be analyzed,
rebuilt, evaluated, audited, exported and matched on exactly the traces its
sorter read, with the same channels, while metadata consumers keep the
original lineage. The fixtures here make that discriminating: the corrected
recording of the planted-drift session uses ``remove_channels`` (so it has
fewer channels than its source) and its traces differ from the source's by
more than 10 uV in the compared window, so a consumer that silently read the
source would fail both the channel and the trace comparison.

Spies only record what a consumer received; the planted sorter stands in for
a sorting algorithm (it records its input and returns two fixed units, so a
merged curation exists).
"""

from __future__ import annotations

import uuid

import numpy as np
import pytest

from tests.spikesorting.v2._motion_db_helpers import (
    MOTION_TEAM,
    drop_motion_selections,
    drop_pipeline_sorts,
    drop_sorts,
    masked_artifact,
    populated_corrected,
    populated_estimate,
    session_start_s,
    sorter_key,
)

#: The manual artifact exclusion, in seconds after the session start.
EXCLUDED_S = (20.0, 21.0)
#: The corrected recording must differ from its source by more than this in
#: the compared window (uV); the planted drift is +/-25 um on a 26 um pitch.
MIN_CORRECTION_UV = 10.0


def _planted_sorter(captured):
    """A ``Sorting._run_sorter`` stand-in: records its input and returns two
    units at fixed frames."""
    import spikeinterface as si

    def _plant(
        sorter,
        sorter_params,
        recording,
        sorting_id,
        *,
        job_kwargs=None,
        execution_params=None,
        statistics_spans=None,
    ):
        captured[str(sorting_id)] = {
            "recording": recording,
            "spans": list(statistics_spans),
        }
        n = int(recording.get_num_samples())
        unit0 = np.arange(3000, n - 3000, 9000, dtype=np.int64)
        samples = np.concatenate([unit0, unit0 + 1500])
        labels = np.repeat(np.array([0, 1], dtype=np.int32), unit0.size)
        order = np.argsort(samples, kind="stable")
        return si.NumpySorting.from_samples_and_labels(
            samples_list=[samples[order]],
            labels_list=[labels[order]],
            sampling_frequency=recording.get_sampling_frequency(),
        )

    return _plant


@pytest.fixture(scope="module")
def corrected_sorts(drift_recording):
    """An uncorrected and a corrected sort of one masked recording.

    The corrected recording drops the end contacts (``remove_channels``).
    Both sorts are populated with the planted sorter; the sorter inputs and
    the recordings the initial analyzers were built from are recorded.
    """
    from spyglass.spikesorting.v2.motion import MotionCorrectedRecording
    from spyglass.spikesorting.v2.recording import Recording
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection

    recording_key = drift_recording["recording_key"]
    t0 = session_start_s(drift_recording["nwb_file_name"])
    artifact_key = masked_artifact(
        recording_key, [t0 + EXCLUDED_S[0], t0 + EXCLUDED_S[1]]
    )
    estimate = populated_estimate(
        recording_id=recording_key["recording_id"], **artifact_key
    )
    corrected_key = populated_corrected(estimate, "kriging_remove_channels_v1")
    corrected_row = (MotionCorrectedRecording & corrected_key).fetch1()
    corrected = MotionCorrectedRecording().get_recording(corrected_key)
    source = Recording().get_recording(recording_key)
    source_ids = source.channel_ids.tolist()
    kept = corrected.channel_ids.tolist()
    assert list(corrected_row["removed_channel_ids"])
    assert 0 < len(kept) < len(source_ids)

    # The compared window: the 1 s inside the first statistics span where the
    # correction changes the kept channels most.
    fs = int(round(corrected.get_sampling_frequency()))
    first_span = [int(f) for f in corrected_row["statistics_spans"][0]]
    columns = [source_ids.index(c) for c in kept]
    best = None
    for start in range(first_span[0], first_span[1] - fs + 1, fs):
        change = np.max(
            np.abs(
                corrected.get_traces(start_frame=start, end_frame=start + fs)
                - source.get_traces(start_frame=start, end_frame=start + fs)[
                    :, columns
                ]
            )
        )
        if best is None or change > best[1]:
            best = (start, change)
    window = (best[0], best[0] + fs)
    assert best[1] > MIN_CORRECTION_UV

    base = {
        "recording_id": recording_key["recording_id"],
        **artifact_key,
        **sorter_key(),
    }
    uncorrected_sort = SortingSelection.insert_selection(base)
    corrected_sort = SortingSelection.insert_selection(
        {**base, **corrected_key}
    )
    sorter_inputs, analyzer_inputs = {}, {}
    build_analyzer = Sorting._build_analyzer

    def _observe_build(*args, **kwargs):
        analyzer_inputs[str(kwargs["key"]["sorting_id"])] = kwargs["recording"]
        return build_analyzer(*args, **kwargs)

    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(
            Sorting, "_run_sorter", staticmethod(_planted_sorter(sorter_inputs))
        )
        mp.setattr(Sorting, "_build_analyzer", staticmethod(_observe_build))
        Sorting.populate([uncorrected_sort, corrected_sort], reserve_jobs=False)
    finally:
        mp.undo()

    yield {
        "recording_key": recording_key,
        "artifact_key": artifact_key,
        "corrected_key": corrected_key,
        "corrected_row": corrected_row,
        "source_ids": source_ids,
        "window": window,
        "expected": corrected.get_traces(
            start_frame=window[0], end_frame=window[1]
        ),
        "source_expected": source.get_traces(
            start_frame=window[0], end_frame=window[1]
        ),
        "uncorrected_sort": uncorrected_sort,
        "corrected_sort": corrected_sort,
        "sorter_inputs": sorter_inputs,
        "analyzer_inputs": analyzer_inputs,
    }

    drop_pipeline_sorts(
        [uncorrected_sort["sorting_id"], corrected_sort["sorting_id"]]
    )


@pytest.fixture
def fresh_curations(corrected_sorts):
    """Drop every curation of both sorts after the test."""
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    yield
    for sort in (
        corrected_sorts["uncorrected_sort"],
        corrected_sorts["corrected_sort"],
    ):
        clear_curations_for(sort)


def _assert_reads_corrected(recording, sorts, what):
    """``recording`` carries the corrected channels and traces."""
    assert recording.channel_ids.tolist() == list(
        sorts["corrected_row"]["channel_ids"]
    ), what
    start, end = sorts["window"]
    np.testing.assert_array_equal(
        recording.get_traces(start_frame=start, end_frame=end),
        sorts["expected"],
        err_msg=what,
    )


def _assert_reads_source(recording, sorts, what):
    """``recording`` carries the uncorrected source's channels and traces."""
    assert recording.channel_ids.tolist() == sorts["source_ids"], what
    start, end = sorts["window"]
    np.testing.assert_array_equal(
        recording.get_traces(start_frame=start, end_frame=end),
        sorts["source_expected"],
        err_msg=what,
    )


def _spy_on(monkeypatch, module, name, captured):
    """Record every return value of ``module.name`` in ``captured``."""
    real = getattr(module, name)

    def _observe(*args, **kwargs):
        result = real(*args, **kwargs)
        captured.append(result)
        return result

    monkeypatch.setattr(module, name, _observe)


def test_all_consumers_resolve_selected_correction(
    corrected_sorts, fresh_curations, curation_evaluation_defaults, monkeypatch
):
    """The sorter input, the initial and rebuilt analyzers, the merged and
    derivative curation analyzers, the metric-curation recording, the
    recompute audit's recording, the curation recording accessor, the
    observed-duration report and the UnitMatch bundle input and geometry of
    a corrected sort all read the corrected recording, never its source."""
    import shutil

    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2 import (
        _pipeline_reporting,
        _sorting_analyzer,
        _source_resolution,
        _unitmatch_backend,
        matcher_protocol,
    )
    from spyglass.spikesorting.v2._analyzer_cache import analyzer_path
    from spyglass.spikesorting.v2._curation_analyzer import (
        _resolve_curation_analyzer,
    )
    from spyglass.spikesorting.v2._nwb_provenance import (
        CURATION_EVALUATION_PROVENANCE,
        read_provenance_values,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.metric_curation import (
        CurationEvaluation,
        CurationEvaluationSelection,
    )
    from spyglass.spikesorting.v2.recompute import (
        _recompute_analyzer_hashes,
        _resolve_analyzer_regen_inputs,
    )
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection
    from spyglass.spikesorting.v2.unit_matching import (
        UnitMatch,
        UnitMatchSelection,
        _member_match_files,
    )

    sorts = corrected_sorts
    sort = sorts["corrected_sort"]
    sorting_id = str(sort["sorting_id"])
    uncorrected_id = str(sorts["uncorrected_sort"]["sorting_id"])
    row = sorts["corrected_row"]

    # The sorter input and the initial analyzer. The uncorrected sort of the
    # same source shows the observation discriminates.
    _assert_reads_corrected(
        sorts["sorter_inputs"][sorting_id]["recording"], sorts, "sorter input"
    )
    assert sorts["sorter_inputs"][sorting_id]["spans"] == [
        (int(a), int(b)) for a, b in row["statistics_spans"]
    ]
    _assert_reads_source(
        sorts["sorter_inputs"][uncorrected_id]["recording"],
        sorts,
        "uncorrected sorter input",
    )
    _assert_reads_corrected(
        sorts["analyzer_inputs"][sorting_id], sorts, "initial analyzer build"
    )
    display = (Sorting & sort).fetch1("display_waveform_params_name")
    _assert_reads_corrected(
        Sorting().get_analyzer(sort).recording, sorts, "initial analyzer"
    )

    # Every unit's peak electrode is one the correction kept.
    assert set((Sorting.Unit & sort).fetch("electrode_id")) <= set(
        int(c) for c in row["channel_ids"]
    )

    # A rebuilt analyzer, after the cache folder is removed.
    folder = analyzer_path(sort["sorting_id"], display)
    shutil.rmtree(folder)
    rebuilt = Sorting().get_analyzer(sort)
    assert folder.exists()
    _assert_reads_corrected(rebuilt.recording, sorts, "rebuilt analyzer")

    # Merged and derivative curation analyzers.
    root = CurationV2.insert_curation(sorting_key=sort)
    merged = CurationV2.create_merged_curation(
        sort, merge_groups=[[0, 1]], parent_curation_id=root["curation_id"]
    )
    merged_analyzer = _resolve_curation_analyzer(merged, display, "display")
    assert len(merged_analyzer.unit_ids) == 1
    _assert_reads_corrected(
        merged_analyzer.recording, sorts, "merged curation analyzer"
    )
    derivative = _resolve_curation_analyzer(
        merged,
        display,
        "display",
        extra_extensions={"isi_histograms": {}},
    )
    assert derivative.has_extension("isi_histograms")
    _assert_reads_corrected(
        derivative.recording, sorts, "derivative curation analyzer"
    )

    # Metric curation, on the cached (root) and merged paths.
    for curation in (root, merged):
        loaded = []
        with monkeypatch.context() as patch:
            _spy_on(
                patch, _source_resolution, "read_effective_recording", loaded
            )
            selection = CurationEvaluationSelection.insert_selection(
                {
                    **curation,
                    "metric_params_name": "minimal",
                    "auto_curation_rules_name": "none",
                }
            )
            CurationEvaluation.populate(selection, reserve_jobs=False)
        assert len(loaded) == 1
        _assert_reads_corrected(loaded[0], sorts, "metric curation")
        provenance = read_provenance_values(
            AnalysisNwbfile.get_abs_path(
                (CurationEvaluation & selection).fetch1("analysis_file_name")
            ),
            CURATION_EVALUATION_PROVENANCE,
        )
        assert provenance["recording_content_hash"] == row["content_hash"]
        assert provenance["motion_corrected_recording_id"] == str(
            sorts["corrected_key"]["motion_corrected_recording_id"]
        )
        assert provenance["source_kind"] == "recording"
        assert provenance["recording_id"] == str(
            sorts["recording_key"]["recording_id"]
        )
        assert provenance["concat_recording_id"] is None

    # The recompute audit rebuilds from the corrected traces and reproduces
    # the stored analyzer.
    regen_inputs = _resolve_analyzer_regen_inputs(
        {"sorting_id": sort["sorting_id"]}, display
    )
    reconstructed = []
    with monkeypatch.context() as patch:
        _spy_on(
            patch,
            _sorting_analyzer,
            "read_canonical_recording",
            reconstructed,
        )
        stored, fresh = _recompute_analyzer_hashes(regen_inputs, 4)
    assert len(reconstructed) == 1
    _assert_reads_corrected(reconstructed[0], sorts, "recompute audit")
    assert stored and fresh == stored

    # The curation recording accessor (the merge table's get_recording).
    _assert_reads_corrected(
        CurationV2.get_recording(root), sorts, "CurationV2.get_recording"
    )

    # The observed duration reads the corrected traces' clock, which is the
    # source's.
    loaded = []
    with monkeypatch.context() as patch:
        _spy_on(patch, _source_resolution, "load_effective_recording", loaded)
        corrected_duration = _pipeline_reporting._observed_duration_s(
            sort["sorting_id"]
        )
    _assert_reads_corrected(loaded[0], sorts, "observed duration")
    assert corrected_duration == _pipeline_reporting._observed_duration_s(
        uncorrected_id
    )

    # UnitMatch: the geometry preflight compares the corrected (reduced)
    # geometry as is, and bundle extraction reads the corrected traces.
    uncorrected_root = CurationV2.insert_curation(
        sorting_key=sorts["uncorrected_sort"]
    )
    np.testing.assert_array_equal(
        UnitMatchSelection._member_channel_positions(root),
        row["channel_locations"],
    )
    corrected_choice = (sort["sorting_id"], root["curation_id"])
    UnitMatchSelection._assert_members_share_geometry(
        {0: corrected_choice, 1: corrected_choice}
    )
    with pytest.raises(ValueError, match="probe geometry"):
        UnitMatchSelection._assert_members_share_geometry(
            {
                0: corrected_choice,
                1: (
                    uncorrected_root["sorting_id"],
                    uncorrected_root["curation_id"],
                ),
            }
        )

    bundle_inputs = {}

    def _record_bundle(session_dir, recording, sorting, **kwargs):
        bundle_inputs[session_dir.name] = recording
        session_dir.mkdir(parents=True, exist_ok=True)

    class _NoPairs:
        def match(self, session_inputs, params):
            return []

    monkeypatch.setattr(
        _unitmatch_backend, "extract_unitmatch_bundle", _record_bundle
    )
    monkeypatch.setattr(
        matcher_protocol, "get_matcher", lambda name: _NoPairs()
    )
    member_plan = [
        {
            "member_index": index,
            "sorting_id": str(curation["sorting_id"]),
            "curation_id": int(curation["curation_id"]),
            "recording_date": f"2023-06-2{index + 2}T12:00:00+00:00",
            "matchable_unit_ids": [
                int(u) for u in CurationV2().get_matchable_unit_ids(curation)
            ],
            **_member_match_files(
                curation,
                SortingSelection.resolve_effective_source(
                    {"sorting_id": curation["sorting_id"]}
                ),
            ),
        }
        for index, curation in enumerate([root, uncorrected_root])
    ]
    pairs, _runtime = UnitMatch._extract_and_match(
        member_plan, "unitmatch", {}, {}
    )
    assert pairs == []
    _assert_reads_corrected(
        bundle_inputs["member_0"], sorts, "UnitMatch bundle input"
    )
    _assert_reads_source(
        bundle_inputs["member_1"], sorts, "uncorrected UnitMatch bundle input"
    )


def test_curation_manifest_identity_changes_with_the_correction(
    corrected_sorts,
):
    """A curation analyzer manifest names the corrected recording's content
    as well as the source's, so it differs from the uncorrected sort's."""
    from spyglass.spikesorting.v2._curation_analyzer import (
        _source_artifact_hashes,
    )
    from spyglass.spikesorting.v2.recording import Recording

    sorts = corrected_sorts
    source_hash = (Recording & sorts["recording_key"]).fetch1("content_hash")
    uncorrected = _source_artifact_hashes(
        sorts["uncorrected_sort"]["sorting_id"]
    )
    corrected = _source_artifact_hashes(sorts["corrected_sort"]["sorting_id"])
    assert uncorrected == {"recording": str(source_hash)}
    assert corrected == {
        "recording": str(source_hash),
        "motion_corrected_recording": str(
            sorts["corrected_row"]["content_hash"]
        ),
    }


def test_curation_restriction_tells_corrected_from_uncorrected(
    corrected_sorts, fresh_curations
):
    """A (recording, sorter) restriction matches the corrected and the
    uncorrected sort of one source; ``motion_corrected_recording_id``
    selects one of them."""
    from spyglass.spikesorting.v2.curation import CurationV2

    sorts = corrected_sorts
    for sort in (sorts["uncorrected_sort"], sorts["corrected_sort"]):
        CurationV2.insert_curation(sorting_key=sort, reuse_existing=True)
    restriction = {
        "recording_id": sorts["recording_key"]["recording_id"],
        **sorter_key(),
    }

    def matched(**extra):
        return {
            str(s)
            for s in CurationV2.resolve_restriction(
                {**restriction, **extra}
            ).fetch("sorting_id")
        }

    uncorrected = str(sorts["uncorrected_sort"]["sorting_id"])
    corrected = str(sorts["corrected_sort"]["sorting_id"])
    assert matched() == {uncorrected, corrected}
    assert matched(motion_corrected_recording_id=None) == {uncorrected}
    assert matched(
        motion_corrected_recording_id=sorts["corrected_key"][
            "motion_corrected_recording_id"
        ]
    ) == {corrected}
    assert matched(motion_corrected_recording_id=uuid.uuid4()) == set()


def test_unitmatch_records_the_waveform_traces(
    corrected_sorts, fresh_curations
):
    """A match run records, per member, whether its waveforms came from the
    source or from a corrected recording (and which)."""
    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2._nwb_provenance import (
        UNITMATCH_MEMBERS,
        read_long_provenance,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.recording import RecordingSelection
    from spyglass.spikesorting.v2.session_group import SessionGroup
    from spyglass.spikesorting.v2.sorting import SortingSelection
    from spyglass.spikesorting.v2.unit_matching import (
        MatcherParameters,
        UnitMatch,
        UnitMatchSelection,
        _member_waveform_traces,
    )
    from tests.spikesorting.v2._ingest_helpers import configure_v2_run_inputs

    def _traces(sort):
        return SortingSelection.resolve_effective_source(
            {"sorting_id": sort["sorting_id"]}
        ).traces

    sorts = corrected_sorts
    corrected_id = str(sorts["corrected_key"]["motion_corrected_recording_id"])
    assert _member_waveform_traces(_traces(sorts["corrected_sort"])) == {
        "waveform_traces": "motion_corrected_recording",
        "motion_corrected_recording_id": corrected_id,
    }
    assert _member_waveform_traces(_traces(sorts["uncorrected_sort"])) == {
        "waveform_traces": "recording",
        "motion_corrected_recording_id": None,
    }

    # One member: UnitMatch writes its provenance without running a matcher.
    group = {
        "session_group_owner": MOTION_TEAM,
        "session_group_name": "motion_consumer_solo",
    }
    nwb_file_name = (RecordingSelection & sorts["recording_key"]).fetch1(
        "nwb_file_name"
    )
    MatcherParameters.insert_default()
    root = CurationV2.insert_curation(sorting_key=sorts["corrected_sort"])
    try:
        SessionGroup.create_group(
            MOTION_TEAM,
            group["session_group_name"],
            [configure_v2_run_inputs(nwb_file_name, MOTION_TEAM)],
        )
        selection = UnitMatchSelection.insert_selection(
            MOTION_TEAM,
            group["session_group_name"],
            "unitmatch_default",
            {0: root},
        )
        UnitMatch.populate(selection, reserve_jobs=False)
        members = read_long_provenance(
            AnalysisNwbfile.get_abs_path(
                (UnitMatch & selection).fetch1("analysis_file_name")
            ),
            UNITMATCH_MEMBERS,
        )
        assert [
            (m["waveform_traces"], m["motion_corrected_recording_id"])
            for m in members
        ] == [("motion_corrected_recording", corrected_id)]
    finally:
        (UnitMatchSelection & group).super_delete(warn=False, safemode=False)
        (SessionGroup & group).super_delete(warn=False, safemode=False)


def test_make_fetch_rechecks_the_sorters_own_correction(
    corrected_sorts, monkeypatch
):
    """A corrected sort whose sorter corrects motion itself is refused at the
    compute boundary, before anything is sorted, even when the insert-time
    check passed (as it would if SpikeInterface's default changed between
    selection and populate)."""
    from spyglass.spikesorting.v2._params import sorter as sorter_params
    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        Sorting,
        SortingSelection,
    )

    sorts = corrected_sorts
    SorterParameters.insert_default()
    with monkeypatch.context() as insert_time:
        insert_time.setattr(
            sorter_params,
            "reject_internal_motion_correction",
            lambda *args, **kwargs: None,
        )
        sort_key = SortingSelection.insert_selection(
            {
                "recording_id": sorts["recording_key"]["recording_id"],
                **sorts["artifact_key"],
                **sorts["corrected_key"],
                "sorter": "spykingcircus2",
                "sorter_params_name": "default",
            }
        )
    try:
        with pytest.raises(ValueError, match="apply_motion_correction=False"):
            Sorting().make_fetch(sort_key)
        assert not (Sorting & sort_key)
    finally:
        drop_sorts([sort_key])


def test_part_drift_on_a_sorted_selection_is_refused(corrected_sorts):
    """A correction part deleted from a corrected sort, or inserted (with
    matching source and mask) onto an uncorrected one, after both were
    sorted, is refused where consumers resolve their traces, instead of
    silently switching them to other traces than the sorter read."""
    from spyglass.spikesorting.v2.exceptions import SchemaBypassError
    from spyglass.spikesorting.v2.sorting import SortingSelection

    sorts = corrected_sorts
    uncorrected, corrected = sorts["uncorrected_sort"], sorts["corrected_sort"]
    correction_part = SortingSelection.MotionCorrectionSource
    drift = "not the ones it was selected with"

    for sort in (uncorrected, corrected):
        SortingSelection.resolve_effective_source(sort)

    correction_part.insert1({**uncorrected, **sorts["corrected_key"]})
    try:
        with pytest.raises(SchemaBypassError, match=drift):
            SortingSelection.resolve_effective_source(uncorrected)
    finally:
        (correction_part & uncorrected).delete_quick()

    (correction_part & corrected).delete_quick()
    try:
        with pytest.raises(SchemaBypassError, match=drift):
            SortingSelection.resolve_effective_source(corrected)
    finally:
        correction_part.insert1({**corrected, **sorts["corrected_key"]})
    assert (
        SortingSelection.resolve_effective_source(corrected).traces.kind
        == "motion_corrected_recording"
    )


@pytest.mark.parametrize("kind", ["recording", "concatenated_recording"])
def test_resolver_rebuilds_a_missing_base_artifact(discontinuous_sources, kind):
    """A sort's missing ``Recording`` or ``ConcatenatedRecording`` file is
    rebuilt by ``ensure_effective_traces`` and then loads the same traces and
    timestamps as before it was deleted."""
    from pathlib import Path

    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2._source_resolution import (
        load_effective_recording,
    )
    from spyglass.spikesorting.v2.sorting import SortingSelection

    source = (
        {"recording_id": discontinuous_sources["member_b"]["recording_id"]}
        if kind == "recording"
        else dict(discontinuous_sources["concat_key"])
    )
    sort_key = SortingSelection.insert_selection({**source, **sorter_key()})
    try:
        _lineage, traces = SortingSelection.resolve_effective_source(sort_key)
        assert traces.kind == kind
        assert traces.apply_artifact_mask is False
        before = load_effective_recording(traces)
        expected_traces = before.get_traces()
        expected_times = before.get_times()
        abs_path = AnalysisNwbfile.get_abs_path(
            traces.row["analysis_file_name"]
        )
        del before
        Path(abs_path).unlink()

        _lineage, traces = SortingSelection.resolve_effective_source(sort_key)
        SortingSelection.ensure_effective_traces(traces)
        assert Path(abs_path).exists()
        after = load_effective_recording(traces)
        np.testing.assert_array_equal(after.get_traces(), expected_traces)
        np.testing.assert_array_equal(after.get_times(), expected_times)
    finally:
        drop_sorts([sort_key])


def test_corrected_concat_sort_end_to_end(
    discontinuous_sources, curation_evaluation_defaults, monkeypatch
):
    """A sort of a corrected concatenation keeps the concatenation's frame
    count and member-join spans, hands its sorter the corrected traces, and
    its evaluation names the concatenation from the lineage."""
    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2._nwb_provenance import (
        CURATION_EVALUATION_PROVENANCE,
        read_provenance_values,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.metric_curation import (
        CurationEvaluation,
        CurationEvaluationSelection,
    )
    from spyglass.spikesorting.v2.motion import MotionCorrectedRecording
    from spyglass.spikesorting.v2.session_group import ConcatenatedRecording
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection

    concat_key = discontinuous_sources["concat_key"]
    corrected_key = populated_corrected(
        populated_estimate(
            concat_recording_id=concat_key["concat_recording_id"]
        )
    )
    row = (MotionCorrectedRecording & corrected_key).fetch1()
    concat_row = (ConcatenatedRecording & concat_key).fetch1()
    corrected = MotionCorrectedRecording().get_recording(corrected_key)
    corrected_traces = corrected.get_traces()
    source_traces = (
        ConcatenatedRecording().get_recording(concat_key).get_traces()
    )
    assert corrected_traces.shape == source_traces.shape
    assert np.max(np.abs(corrected_traces - source_traces)) > MIN_CORRECTION_UV
    concat_spans = [(int(a), int(b)) for a, b in concat_row["statistics_spans"]]
    assert len(concat_spans) > 1

    sort_key = SortingSelection.insert_selection(
        {**concat_key, **sorter_key(), **corrected_key}
    )
    captured = {}
    run_sorter = Sorting._run_sorter

    def _observe(*args, **kwargs):
        captured["traces"] = kwargs["recording"].get_traces()
        captured["spans"] = list(kwargs["statistics_spans"])
        return run_sorter(*args, **kwargs)

    try:
        fetched = Sorting().make_fetch(sort_key)
        assert fetched.source_n_samples == int(concat_row["n_samples"])
        assert fetched.traces.key == corrected_key
        with monkeypatch.context() as patch:
            patch.setattr(Sorting, "_run_sorter", staticmethod(_observe))
            Sorting.populate(sort_key, reserve_jobs=False)

        np.testing.assert_array_equal(captured["traces"], corrected_traces)
        assert captured["spans"] == concat_spans
        assert [
            (int(a), int(b)) for a, b in row["statistics_spans"]
        ] == concat_spans
        assert Sorting().get_statistics_spans(sort_key) == concat_spans

        root = CurationV2.insert_curation(sorting_key=sort_key)
        selection = CurationEvaluationSelection.insert_selection(
            {
                **root,
                "metric_params_name": "minimal",
                "auto_curation_rules_name": "none",
            }
        )
        CurationEvaluation.populate(selection, reserve_jobs=False)
        provenance = read_provenance_values(
            AnalysisNwbfile.get_abs_path(
                (CurationEvaluation & selection).fetch1("analysis_file_name")
            ),
            CURATION_EVALUATION_PROVENANCE,
        )
        assert provenance["source_kind"] == "concatenated_recording"
        assert provenance["concat_recording_id"] == str(
            concat_key["concat_recording_id"]
        )
        assert provenance["recording_id"] is None
        assert provenance["motion_corrected_recording_id"] == str(
            corrected_key["motion_corrected_recording_id"]
        )
        assert provenance["recording_content_hash"] == row["content_hash"]
    finally:
        drop_pipeline_sorts([sort_key["sorting_id"]])


def test_unitmatch_bundle_of_a_corrected_sort(
    corrected_sorts, fresh_curations, tmp_path
):
    """Real bundle extraction on a corrected sort writes the corrected
    channel set and positions, not the source's (needs UnitMatchPy; the
    matching CI lane runs it)."""
    pytest.importorskip("UnitMatchPy")
    from spyglass.spikesorting.v2._unitmatch_backend import (
        extract_unitmatch_bundle,
    )
    from spyglass.spikesorting.v2.curation import CurationV2

    sorts = corrected_sorts
    row = sorts["corrected_row"]
    root = CurationV2.insert_curation(sorting_key=sorts["corrected_sort"])
    sorting = CurationV2.get_sorting(root)
    session_dir = tmp_path / "member_0"
    extract_unitmatch_bundle(
        session_dir,
        CurationV2.get_recording(root),
        sorting,
        max_spikes_per_unit=20,
    )
    np.testing.assert_array_equal(
        np.load(session_dir / "channel_positions.npy"),
        row["channel_locations"],
    )
    unit_id = int(sorting.get_unit_ids()[0])
    waveform = np.load(
        session_dir / "RawWaveforms" / f"Unit{unit_id}_RawSpikes.npy"
    )
    # (spike_width, n_channels, 2): one column per channel the correction kept.
    assert waveform.shape[1:] == (len(row["channel_ids"]), 2)


#: An estimation recipe whose zero gap cap removes every real gap from the
#: estimation clock, so estimation-clock times after a gap differ from the
#: source's real times by the whole gap.
GAP_CAPPED_RECIPE = "dredge_fast_no_gap_test"


def _gap_capped_corrected(source) -> tuple[dict, dict]:
    """The estimate and corrected recording of ``source`` under the
    zero-gap-cap recipe (populated)."""
    from spyglass.spikesorting.v2._params.motion_estimation import (
        MotionEstimationParamsSchema,
    )
    from spyglass.spikesorting.v2.motion import (
        MotionEstimate,
        MotionEstimateSelection,
        MotionEstimationParameters,
    )

    MotionEstimationParameters.insert1(
        {
            "motion_estimation_params_name": GAP_CAPPED_RECIPE,
            "params": MotionEstimationParamsSchema(
                preset="dredge_fast", max_gap_s=0.0
            ).model_dump(),
        },
        skip_duplicates=True,
    )
    estimate_key = MotionEstimateSelection.insert_selection(
        {"motion_estimation_params_name": GAP_CAPPED_RECIPE, **source}
    )
    MotionEstimate.populate(estimate_key, reserve_jobs=False)
    return estimate_key, populated_corrected(estimate_key)


def _planted_frames(frames):
    """A ``Sorting._run_sorter`` stand-in returning one unit at ``frames``."""
    import spikeinterface as si

    def _plant(sorter, sorter_params, recording, sorting_id, **kwargs):
        return si.NumpySorting.from_samples_and_labels(
            samples_list=[np.asarray(frames, dtype=np.int64)],
            labels_list=[np.zeros(len(frames), dtype=np.int32)],
            sampling_frequency=recording.get_sampling_frequency(),
        )

    return _plant


def _drop_estimate(estimate_key) -> None:
    from spyglass.spikesorting.v2.motion import MotionEstimateSelection

    (MotionEstimateSelection & estimate_key).super_delete(
        warn=False, safemode=False
    )


def test_corrected_sort_across_a_capped_gap_keeps_source_frames_and_times(
    discontinuous_sources, monkeypatch
):
    """A corrected sort of a recording whose 1 s acquisition gap is capped
    to zero on the estimation clock reads back exactly the frames its sorter
    returned, and its stored spike times are the source's real timestamps at
    those frames: after the gap they fall in the recording's second valid
    interval, where the estimation clock would have put them inside the
    gap."""
    from spyglass.common import IntervalList
    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2._units_nwb import (
        read_units_abs_times_and_sample_indices,
        recording_timestamps,
    )
    from spyglass.spikesorting.v2.motion import MotionEstimate
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
    )
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection
    from tests.spikesorting.v2._motion_db_helpers import MEMBER_B_INTERVAL

    recording_key = {
        "recording_id": discontinuous_sources["member_b"]["recording_id"]
    }
    estimate_key, corrected_key = _gap_capped_corrected(recording_key)
    sort_key = None
    try:
        clock = MotionEstimate().get_estimation_clock(estimate_key)
        fs = clock.sampling_frequency
        spans = clock.spans
        assert spans.shape == (2, 2)
        first_len_s = (spans[0, 1] - spans[0, 0]) / fs
        assert clock.source_start_s[1] - clock.source_end_s[0] > 0.9
        assert clock.estimation_start_s[1] == pytest.approx(
            clock.estimation_start_s[0] + first_len_s, abs=1e-9
        )

        second = int(spans[1, 0])
        planted = np.array(
            [
                3000,
                60000,
                second + 3000,
                second + 45000,
                int(spans[1, 1]) - 3000,
            ]
        )
        sort_key = SortingSelection.insert_selection(
            {**recording_key, **sorter_key(), **corrected_key}
        )
        with monkeypatch.context() as patch:
            patch.setattr(
                Sorting, "_run_sorter", staticmethod(_planted_frames(planted))
            )
            Sorting.populate(sort_key, reserve_jobs=False)

        sorting = Sorting().get_sorting(sort_key)
        (unit_id,) = sorting.unit_ids
        np.testing.assert_array_equal(
            sorting.get_unit_spike_train(unit_id=unit_id), planted
        )
        abs_times, sample_indices, _ = read_units_abs_times_and_sample_indices(
            AnalysisNwbfile.get_abs_path(
                (Sorting & sort_key).fetch1("analysis_file_name")
            )
        )
        (unit,) = abs_times
        np.testing.assert_array_equal(sample_indices[unit], planted)
        timestamps = recording_timestamps((Recording & recording_key).fetch1())
        np.testing.assert_array_equal(abs_times[unit], timestamps[planted])

        valid = (
            IntervalList
            & {
                "nwb_file_name": (RecordingSelection & recording_key).fetch1(
                    "nwb_file_name"
                ),
                "interval_list_name": MEMBER_B_INTERVAL,
            }
        ).fetch1("valid_times")
        after_gap = abs_times[unit][2:]
        assert ((after_gap >= valid[1][0]) & (after_gap <= valid[1][1])).all()
        estimation_times = (
            clock.estimation_start_s[1] + (planted[2:] - second) / fs
        )
        assert valid[0][1] < estimation_times[0] < valid[1][0]
    finally:
        if sort_key is not None:
            drop_pipeline_sorts([sort_key["sorting_id"]])
        _drop_estimate(estimate_key)


def test_corrected_concat_split_into_members_conserves_spikes(
    discontinuous_sources, monkeypatch
):
    """A corrected concatenation sort (its gaps capped to zero on the
    estimation clock) split into member outputs by ``ConcatMemberCuration``
    keeps every spike: each member holds the sorter's frames that fall in
    it, shifted to member frames, with the member recording's own
    timestamps at those frames."""
    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2._units_nwb import (
        read_units_abs_times_and_sample_indices,
        recording_timestamps,
    )
    from spyglass.spikesorting.v2.concat_member_curation import (
        ConcatMemberCuration,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.recording import Recording
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecordingSelection,
    )
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection

    concat_key = discontinuous_sources["concat_key"]
    estimate_key, corrected_key = _gap_capped_corrected(dict(concat_key))
    snapshots = (
        ConcatenatedRecordingSelection.MemberSnapshot & concat_key
    ).fetch(as_dict=True, order_by="member_index")
    member_timestamps = [
        recording_timestamps(
            (Recording & {"recording_id": m["recording_id"]}).fetch1()
        )
        for m in snapshots
    ]
    lengths = [len(t) for t in member_timestamps]
    assert len(lengths) == 2
    offsets = np.cumsum([0, *lengths])
    planted = np.array(
        [
            3000,
            lengths[0] - 3000,
            offsets[1] + 3000,
            offsets[1] + lengths[1] // 2 + 3000,
            offsets[2] - 3000,
        ]
    )
    sort_key = None
    try:
        sort_key = SortingSelection.insert_selection(
            {**concat_key, **sorter_key(), **corrected_key}
        )
        with monkeypatch.context() as patch:
            patch.setattr(
                Sorting, "_run_sorter", staticmethod(_planted_frames(planted))
            )
            Sorting.populate(sort_key, reserve_jobs=False)
        curation_key = CurationV2.insert_curation(sorting_key=sort_key)
        ConcatMemberCuration.populate(curation_key, reserve_jobs=False)

        n_member_spikes = 0
        for index in range(len(snapshots)):
            member_key = {**curation_key, "member_index": index}
            expected = (
                planted[
                    (planted >= offsets[index]) & (planted < offsets[index + 1])
                ]
                - offsets[index]
            )
            member_sorting = ConcatMemberCuration.get_sorting(member_key)
            (unit_id,) = member_sorting.unit_ids
            frames = member_sorting.get_unit_spike_train(unit_id=unit_id)
            np.testing.assert_array_equal(frames, expected)
            abs_times, sample_indices, _ = (
                read_units_abs_times_and_sample_indices(
                    AnalysisNwbfile.get_abs_path(
                        (ConcatMemberCuration & member_key).fetch1(
                            "analysis_file_name"
                        )
                    )
                )
            )
            (unit,) = abs_times
            np.testing.assert_array_equal(sample_indices[unit], expected)
            np.testing.assert_array_equal(
                abs_times[unit], member_timestamps[index][expected]
            )
            n_member_spikes += len(frames)
        assert n_member_spikes == len(planted)
    finally:
        if sort_key is not None:
            drop_pipeline_sorts([sort_key["sorting_id"]])
        _drop_estimate(estimate_key)


def test_motion_cleanup_drops_the_sorts_of_corrected_recordings(
    corrected_sorts,
):
    """The module teardown's motion cleanup also removes populated sorts of
    the recording's corrected recordings (a test that failed before its own
    cleanup leaves them), instead of failing on their correction parts.

    Runs last: it removes the module's corrected sorts.
    """
    from spyglass.spikesorting.v2.motion import (
        MotionCorrectedRecording,
        MotionEstimateSelection,
    )
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection

    sorts = corrected_sorts
    sort = sorts["corrected_sort"]
    assert Sorting & sort
    drop_motion_selections(sorts["recording_key"])
    assert not (SortingSelection & sort)
    assert not (MotionCorrectedRecording & sorts["corrected_key"])
    assert not (
        MotionEstimateSelection.RecordingSource & sorts["recording_key"]
    )
    assert SortingSelection & sorts["uncorrected_sort"]

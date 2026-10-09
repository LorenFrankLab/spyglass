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
    CONCAT_GROUP,
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
from tests.spikesorting.v2._sorter_stub import active_sorter, plant_sorter

#: The manual artifact exclusion, in seconds after the session start.
EXCLUDED_S = (20.0, 21.0)
#: The corrected recording must differ from its source by more than this in
#: the compared window (uV); the planted drift is +/-25 um on a 26 um pitch.
MIN_CORRECTION_UV = 10.0


def _planted_sorter(captured):
    """A ``plant_sorter`` stand-in: records its input and returns two
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
    from spyglass.spikesorting.v2._sorting import analyzer as _sorting_analyzer
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
    build_analyzer = _sorting_analyzer.build_analyzer

    def _observe_build(*args, **kwargs):
        analyzer_inputs[str(kwargs["key"]["sorting_id"])] = kwargs["recording"]
        return build_analyzer(*args, **kwargs)

    mp = pytest.MonkeyPatch()
    try:
        plant_sorter(mp, _planted_sorter(sorter_inputs))
        mp.setattr(_sorting_analyzer, "build_analyzer", _observe_build)
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


def _trains(sorting):
    """``{unit_id: frames}`` of a SpikeInterface sorting."""
    return {
        int(u): sorting.get_unit_spike_train(unit_id=u).tolist()
        for u in sorting.get_unit_ids()
    }


def _units_readbacks(sort_key, curation, selection):
    """Spike trains from every units readback of a sort and its curation.

    ``Sorting.get_sorting``, ``CurationV2.get_sorting``, and the raw and
    curated units ``CurationEvaluation.make_fetch`` resolves for its compute.
    Also returns those resolved units.
    """
    from spyglass.spikesorting.v2._storage.units_nwb import read_stored_units
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.metric_curation import CurationEvaluation
    from spyglass.spikesorting.v2.sorting import Sorting

    fetched = CurationEvaluation().make_fetch(selection).sorting_inputs
    trains = {
        "Sorting.get_sorting": _trains(Sorting().get_sorting(sort_key)),
        "CurationV2.get_sorting": _trains(CurationV2.get_sorting(curation)),
        "evaluation raw units": _trains(read_stored_units(fetched.raw_units)),
        "evaluation curated units": _trains(
            read_stored_units(fetched.curated_units)
        ),
    }
    return trains, (fetched.raw_units, fetched.curated_units)


def _hand_input_plan(curations):
    """A UnitMatch input plan for ``curations``, built as ``make_fetch`` does.

    For driving ``extract_and_match`` / ``make_compute`` directly
    on sorts of one recording, which a selection would reject as sharing a
    session.
    """
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import SortingSelection
    from spyglass.spikesorting.v2.unit_matching import (
        _add_single_recording_frames,
        _member_match_files,
        _member_waveform_traces,
        _resolve_match_input,
    )

    plan = []
    for index, curation in enumerate(curations):
        resolved = _resolve_match_input(
            curation["sorting_id"], curation["curation_id"], ValueError
        )
        _add_single_recording_frames(resolved)
        source = SortingSelection.resolve_effective_source(
            {"sorting_id": curation["sorting_id"]}
        )
        start = f"2023-06-2{index + 2}T12:00:00+00:00"
        plan.append(
            {
                "input_index": index,
                "sorting_id": str(curation["sorting_id"]),
                "curation_id": int(curation["curation_id"]),
                "curation_uuid": str(resolved["curation_uuid"]),
                "source_kind": resolved["source_kind"],
                "source_id": str(resolved["source_id"]),
                "input_start_time": start,
                "recordings": [
                    {
                        **recording,
                        "recording_id": str(recording["recording_id"]),
                        "session_start_time": start,
                    }
                    for recording in resolved["recordings"]
                ],
                "matchable_unit_ids": [
                    int(u)
                    for u in CurationV2().get_matchable_unit_ids(curation)
                ],
                **_member_waveform_traces(source.traces),
                **_member_match_files(curation),
            }
        )
    return plan


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
    from spyglass.spikesorting.v2._orchestration import (
        reporting as _pipeline_reporting,
    )
    from spyglass.spikesorting.v2._sorting import analyzer as _sorting_analyzer
    from spyglass.spikesorting.v2._recording import source as _source_resolution
    from spyglass.spikesorting.v2._matching import (
        unitmatch_backend as _unitmatch_backend,
    )
    from spyglass.spikesorting.v2 import matcher_protocol
    from spyglass.spikesorting.v2._storage.analyzer_cache import analyzer_path
    from spyglass.spikesorting.v2._curation.analyzer import (
        _resolve_curation_analyzer,
    )
    from spyglass.spikesorting.v2._storage.provenance import (
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
    from spyglass.spikesorting.v2.unit_matching import UnitMatchSelection
    from spyglass.spikesorting.v2._matching.compute import extract_and_match

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
        _spy_on(patch, _source_resolution, "read_persisted_traces", loaded)
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
    UnitMatchSelection._validate_matcher_geometry(
        {0: corrected_choice, 1: corrected_choice}, "unitmatch", {}
    )
    with pytest.raises(ValueError, match="probe geometry"):
        UnitMatchSelection._validate_matcher_geometry(
            {
                0: corrected_choice,
                1: (
                    uncorrected_root["sorting_id"],
                    uncorrected_root["curation_id"],
                ),
            },
            "unitmatch",
            {},
        )

    bundle_inputs = {}

    def _record_bundle(session_dir, recording, sorting, **kwargs):
        bundle_inputs[session_dir.name] = recording
        session_dir.mkdir(parents=True, exist_ok=True)
        return []

    class _NoPairs:
        def match(self, session_inputs, params):
            return []

    monkeypatch.setattr(
        _unitmatch_backend, "extract_unitmatch_bundle", _record_bundle
    )
    monkeypatch.setattr(
        matcher_protocol, "get_matcher", lambda name: _NoPairs()
    )
    input_plan = _hand_input_plan([root, uncorrected_root])
    pairs, _runtime = extract_and_match(input_plan, "unitmatch", {}, {})
    assert pairs == []
    _assert_reads_corrected(
        bundle_inputs["input_0"], sorts, "UnitMatch bundle input"
    )
    _assert_reads_source(
        bundle_inputs["input_1"], sorts, "uncorrected UnitMatch bundle input"
    )


def test_units_readback_of_corrected_and_uncorrected_sorts(
    corrected_sorts,
    fresh_curations,
    curation_evaluation_defaults,
):
    """Every units reader returns planted frames for corrected and raw traces."""
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.metric_curation import (
        CurationEvaluationSelection,
    )

    from spyglass.spikesorting.v2.recording import Recording

    source_rate = float(
        (Recording & corrected_sorts["recording_key"]).fetch1(
            "sampling_frequency"
        )
    )
    for which in ("uncorrected_sort", "corrected_sort"):
        sort = corrected_sorts[which]
        sorter_input = corrected_sorts["sorter_inputs"][str(sort["sorting_id"])]
        n = int(sorter_input["recording"].get_num_samples())
        unit0 = np.arange(3000, n - 3000, 9000, dtype=np.int64)
        expected = {0: unit0.tolist(), 1: (unit0 + 1500).tolist()}

        root = CurationV2.insert_curation(sorting_key=sort)
        selection = CurationEvaluationSelection.insert_selection(
            {
                **root,
                "metric_params_name": "minimal",
                "auto_curation_rules_name": "none",
            }
        )
        trains, units = _units_readbacks(sort, root, selection)
        for reader, train in trains.items():
            assert train == expected, f"{which}: {reader}"
        for stored in units:
            assert stored.sampling_frequency == source_rate
            assert stored.abs_path.endswith(".nwb")


def test_curation_manifest_identity_changes_with_the_correction(
    corrected_sorts,
):
    """A curation analyzer manifest names the corrected recording's content
    as well as the source's, so it differs from the uncorrected sort's."""
    from spyglass.spikesorting.v2._curation.analyzer import (
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
    corrected_sorts, fresh_curations, monkeypatch
):
    """A match run records, per input, whether its waveforms came from the
    source or from a corrected recording (and which)."""
    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2._matching import (
        unitmatch_backend as _unitmatch_backend,
    )
    from spyglass.spikesorting.v2 import matcher_protocol
    from spyglass.spikesorting.v2._storage.provenance import (
        UNITMATCH_INPUTS,
        read_long_provenance,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.recording import (
        _unlink_staged_analysis_file,
    )
    from spyglass.spikesorting.v2.sorting import SortingSelection
    from spyglass.spikesorting.v2.unit_matching import (
        UnitMatch,
        _member_waveform_traces,
    )

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

    class _NoPairs:
        def match(self, session_inputs, params):
            return []

    def _no_bundle(session_dir, recording, sorting, **kwargs):
        session_dir.mkdir(parents=True, exist_ok=True)
        return []

    monkeypatch.setattr(
        _unitmatch_backend, "extract_unitmatch_bundle", _no_bundle
    )
    monkeypatch.setattr(
        matcher_protocol, "get_matcher", lambda name: _NoPairs()
    )
    # Both sorts read one recording, which a selection rejects as one
    # session, so drive make_compute on a hand-built input plan.
    corrected_root = CurationV2.insert_curation(
        sorting_key=sorts["corrected_sort"]
    )
    uncorrected_root = CurationV2.insert_curation(
        sorting_key=sorts["uncorrected_sort"]
    )
    computed = UnitMatch().make_compute(
        {"unitmatch_id": uuid.uuid4()},
        "unitmatch",
        {},
        {},
        _hand_input_plan([corrected_root, uncorrected_root]),
        None,
        None,
        "unitmatch_default",
    )
    try:
        inputs = read_long_provenance(
            AnalysisNwbfile.get_abs_path(computed.analysis_file_name),
            UNITMATCH_INPUTS,
        )
    finally:
        _unlink_staged_analysis_file(
            computed.analysis_file_name, context="waveform traces test"
        )
    assert [
        (
            m["input_index"],
            m["waveform_traces"],
            m["motion_corrected_recording_id"],
        )
        for m in inputs
    ] == [
        (0, "motion_corrected_recording", corrected_id),
        (1, "recording", ""),
    ]


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
    from spyglass.spikesorting.v2._recording.source import (
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
    from spyglass.spikesorting.v2._storage.provenance import (
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
    run_sorter = active_sorter()

    def _observe(*args, **kwargs):
        captured["traces"] = kwargs["recording"].get_traces()
        captured["spans"] = list(kwargs["statistics_spans"])
        return run_sorter(*args, **kwargs)

    try:
        fetched = Sorting().make_fetch(sort_key)
        assert fetched.source_n_samples == int(concat_row["n_samples"])
        assert fetched.traces.key == corrected_key
        with monkeypatch.context() as patch:
            plant_sorter(patch, _observe)
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

        # Stored frames agree across every reader of a corrected concatenation.
        trains, units = _units_readbacks(sort_key, root, selection)
        expected = trains["Sorting.get_sorting"]
        assert expected and any(expected.values())
        assert all(train == expected for train in trains.values()), trains
        assert all(
            stored.sampling_frequency == concat_row["sampling_frequency"]
            for stored in units
        )
    finally:
        drop_pipeline_sorts([sort_key["sorting_id"]])


def test_unitmatch_bundle_of_a_corrected_sort(
    corrected_sorts, fresh_curations, tmp_path
):
    """Real bundle extraction on a corrected sort writes the corrected
    channel set and positions, not the source's (needs UnitMatchPy; the
    matching CI lane runs it)."""
    pytest.importorskip("UnitMatchPy")
    from spyglass.spikesorting.v2._matching.unitmatch_backend import (
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


def _plant_at_span_edges(
    sorter,
    sorter_params,
    recording,
    sorting_id,
    *,
    job_kwargs=None,
    execution_params=None,
    statistics_spans=None,
):
    """Three planted units placed by the sort's statistics spans.

    Unit 0 fires only in the first span; unit 1 fires every 20 frames over
    the 600 frames on each side of every edge between two spans, so several
    of its windows would run across a join or gap; unit 2 fires every 5000
    frames inside the spans.
    """
    import spikeinterface as si

    del sorter, sorter_params, sorting_id, job_kwargs, execution_params
    spans = [(int(a), int(b)) for a, b in statistics_spans]
    inside = np.zeros(recording.get_num_samples(), dtype=bool)
    for start, end in spans:
        inside[start:end] = True
    edges = [end for (_, end), (start, _) in zip(spans, spans[1:])]
    dense = np.concatenate([np.arange(e - 600, e + 600, 20) for e in edges])
    units = [
        np.arange(spans[0][0] + 1000, spans[0][1] - 1000, 2000),
        dense[inside[dense]],
        np.arange(1000, recording.get_num_samples() - 1000, 5000),
    ]
    units[2] = units[2][inside[units[2]]]
    samples = np.concatenate(units).astype(np.int64)
    labels = np.concatenate(
        [
            np.full(len(u), label, dtype=np.int32)
            for label, u in enumerate(units)
        ]
    )
    order = np.argsort(samples, kind="stable")
    return si.NumpySorting.from_samples_and_labels(
        samples_list=[samples[order]],
        labels_list=[labels[order]],
        sampling_frequency=recording.get_sampling_frequency(),
    )


def test_corrected_concat_bundle_keeps_windows_in_spans(
    discontinuous_sources, monkeypatch, tmp_path
):
    """A UnitMatch bundle of a corrected concatenation is cut from the
    corrected traces, only at spikes whose window lies in one statistics span.

    The concatenation has a member join and a gap inside its second member.
    For the corrected sort and the uncorrected sort of the same
    concatenation, real bundle extraction draws exactly the planted frames
    whose window fits one span, each half equals the mean of windows cut at
    those frames from the sort's own effective traces, and the unit that
    fires only in the first member has two nonzero halves. The corrected
    bundle's traces differ from the uncorrected one's, so it is not the
    concatenation's traces (needs UnitMatchPy; the matching CI lane runs
    it).
    """
    pytest.importorskip("UnitMatchPy")
    import shutil
    from pathlib import Path

    from spikeinterface.core import analyzer_extension_core

    from spyglass.spikesorting.v2._matching import (
        unitmatch_backend as _unitmatch_backend,
    )
    from spyglass.spikesorting.v2 import matcher_protocol
    from spyglass.spikesorting.v2._recording.source import (
        load_effective_recording,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.session_group import ConcatenatedRecording
    from spyglass.spikesorting.v2.sorting import (
        Sorting,
        SortingSelection,
    )
    from spyglass.spikesorting.v2._matching.compute import extract_and_match

    concat_key = discontinuous_sources["concat_key"]
    corrected_key = populated_corrected(
        populated_estimate(
            concat_recording_id=concat_key["concat_recording_id"]
        )
    )
    concat_spans = [
        (int(a), int(b))
        for a, b in (ConcatenatedRecording & concat_key).fetch1(
            "statistics_spans"
        )
    ]
    assert len(concat_spans) >= 3  # a member join and a gap in member B
    sort_keys = {
        "input_0": SortingSelection.insert_selection(
            {**concat_key, **sorter_key(), **corrected_key}
        ),
        "input_1": SortingSelection.insert_selection(
            {**concat_key, **sorter_key()}
        ),
    }

    real_extract = _unitmatch_backend.extract_unitmatch_bundle
    extracted = {}

    def _keep_bundle(session_dir, recording, sorting, **kwargs):
        excluded = real_extract(session_dir, recording, sorting, **kwargs)
        name = Path(session_dir).name
        extracted[name] = {
            "recording": recording,
            "spans": kwargs["statistics_spans"],
            "excluded": excluded,
            "dir": shutil.copytree(session_dir, tmp_path / name),
        }
        return excluded

    real_draw = analyzer_extension_core.random_spikes_selection
    drawn = []

    def _record_draw(sorting, *args, **kwargs):
        indices = real_draw(sorting, *args, **kwargs)
        drawn.append(sorting.to_spike_vector()[indices])
        return indices

    class _NoPairs:
        def match(self, session_inputs, params):
            return []

    try:
        with monkeypatch.context() as patch:
            plant_sorter(patch, _plant_at_span_edges)
            Sorting.populate(list(sort_keys.values()), reserve_jobs=False)
        curations = {
            name: CurationV2.insert_curation(sorting_key=key)
            for name, key in sort_keys.items()
        }
        monkeypatch.setattr(
            _unitmatch_backend, "extract_unitmatch_bundle", _keep_bundle
        )
        monkeypatch.setattr(
            analyzer_extension_core, "random_spikes_selection", _record_draw
        )
        monkeypatch.setattr(
            matcher_protocol, "get_matcher", lambda name: _NoPairs()
        )
        input_plan = _hand_input_plan(
            [curations["input_0"], curations["input_1"]]
        )
        assert [plan["waveform_traces"] for plan in input_plan] == [
            "motion_corrected_recording",
            "concatenated_recording",
        ]
        extract_and_match(input_plan, "unitmatch", {}, {})

        assert sorted(extracted) == ["input_0", "input_1"]
        traces_by_input = {}
        for index, name in enumerate(("input_0", "input_1")):
            bundle, draw = extracted[name], drawn[index]
            sort_key = sort_keys[name]
            spans = [
                tuple(span) for span in Sorting().get_statistics_spans(sort_key)
            ]
            assert spans == concat_spans
            assert [tuple(span) for span in bundle["spans"]] == spans
            source = SortingSelection.resolve_effective_source(sort_key)
            assert not source.traces.apply_artifact_mask
            expected_input = load_effective_recording(source.traces)
            traces = expected_input.get_traces(return_in_uV=True)
            np.testing.assert_array_equal(
                bundle["recording"].get_traces(return_in_uV=True), traces
            )
            traces_by_input[name] = traces
            half_window = int(
                1.5 * expected_input.get_sampling_frequency() / 1000.0
            )
            planted = CurationV2.get_sorting(curations[name])
            assert bundle["excluded"] == []
            for unit_index, unit_id in enumerate(planted.get_unit_ids()):
                frames = np.sort(
                    draw["sample_index"][draw["unit_index"] == unit_index]
                )
                train = planted.get_unit_spike_train(unit_id)
                assert frames.tolist() == [
                    int(s)
                    for s in train
                    if any(
                        a <= s - half_window and s + half_window <= b
                        for a, b in spans
                    )
                ], (name, unit_id)
                windows = np.stack(
                    [traces[s - half_window : s + half_window] for s in frames]
                )
                n_half = len(frames) // 2
                waveform = np.load(
                    bundle["dir"]
                    / "RawWaveforms"
                    / f"Unit{unit_id}_RawSpikes.npy"
                )
                for half, expected in enumerate(
                    (
                        windows[:n_half].mean(axis=0),
                        windows[n_half:].mean(axis=0),
                    )
                ):
                    np.testing.assert_allclose(
                        waveform[..., half], expected, rtol=1e-5, atol=1e-6
                    )
                    assert np.any(waveform[..., half] != 0)
            # The dense unit lost the spikes whose window crosses an edge.
            assert np.sum(draw["unit_index"] == 1) < len(
                planted.get_unit_spike_train(planted.get_unit_ids()[1])
            )
            assert np.all(
                draw["sample_index"][draw["unit_index"] == 0] < spans[0][1]
            )
        assert (
            np.max(
                np.abs(traces_by_input["input_0"] - traces_by_input["input_1"])
            )
            > MIN_CORRECTION_UV
        )
    finally:
        drop_pipeline_sorts([key["sorting_id"] for key in sort_keys.values()])


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
    """A ``plant_sorter`` stand-in returning one unit at ``frames``."""
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
    from spyglass.spikesorting.v2._storage.units_nwb import (
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
            plant_sorter(patch, _planted_frames(planted))
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
    from spyglass.spikesorting.v2._storage.units_nwb import (
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
            plant_sorter(patch, _planted_frames(planted))
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


def _window_traces(recording, windows):
    """``recording``'s traces over each ``(start, end)`` frame window."""
    return [
        recording.get_traces(start_frame=start, end_frame=end)
        for start, end in windows
    ]


def _assert_same_windows(recording, expected, windows, what):
    """``recording`` equals ``expected`` (traces and times) on ``windows``."""
    assert recording.channel_ids.tolist() == expected.channel_ids.tolist(), what
    for got, want in zip(
        _window_traces(recording, windows), _window_traces(expected, windows)
    ):
        np.testing.assert_array_equal(got, want, err_msg=what)
    times, expected_times = recording.get_times(), expected.get_times()
    for start, end in windows:
        np.testing.assert_array_equal(
            times[start:end], expected_times[start:end], err_msg=what
        )


def _inside(times, excluded):
    """The frame window strictly inside an excluded ``[start, end)`` time
    range (0.1 s clear of each edge)."""
    start, end = np.searchsorted(times, [excluded[0] + 0.1, excluded[1] - 0.1])
    assert end - start > 1000
    return int(start), int(end)


def test_trace_accessors_have_one_meaning_each(
    corrected_sorts, fresh_curations, drift_recording
):
    """For a single-recording sort pinning an artifact detection, uncorrected
    and corrected: ``get_source_recording`` is the unmasked, uncorrected
    ``Recording`` cache and ``get_sorting_input_recording`` is the traces the
    sorter read (zero over the exclusion; the corrected artifact's channels
    and traces when corrected), both on the acquisition clock.
    ``get_recording`` keeps returning its documented alias: the source for
    the uncorrected sort and the sorting input for the corrected one. The
    UnitMatch bundle input is the sorting input."""
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2._sorting.analyzer import (
        read_canonical_recording,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.motion import MotionCorrectedRecording
    from spyglass.spikesorting.v2.recording import Recording
    from spyglass.spikesorting.v2.unit_matching import _member_match_files

    sorts = corrected_sorts
    source = Recording().get_recording(sorts["recording_key"])
    corrected = MotionCorrectedRecording().get_recording(sorts["corrected_key"])
    t0 = session_start_s(drift_recording["nwb_file_name"])
    excluded = _inside(
        source.get_times(), (t0 + EXCLUDED_S[0], t0 + EXCLUDED_S[1])
    )
    windows = [excluded, sorts["window"]]
    source_excluded = source.get_traces(
        start_frame=excluded[0], end_frame=excluded[1]
    )
    assert np.count_nonzero(source_excluded) == source_excluded.size

    uncorrected = CurationV2.insert_curation(
        sorting_key=sorts["uncorrected_sort"]
    )
    corrected_curation = CurationV2.insert_curation(
        sorting_key=sorts["corrected_sort"]
    )
    for curation, what in (
        (uncorrected, "uncorrected"),
        (corrected_curation, "corrected"),
    ):
        _assert_same_windows(
            CurationV2.get_source_recording(curation),
            source,
            windows,
            f"{what} source recording",
        )
        merge_key = {
            "merge_id": (SpikeSortingOutput.CurationV2 & curation).fetch1(
                "merge_id"
            )
        }
        _assert_same_windows(
            SpikeSortingOutput.get_source_recording(merge_key),
            source,
            windows,
            f"{what} merge source recording",
        )
        _assert_same_windows(
            SpikeSortingOutput.get_sorting_input_recording(merge_key),
            CurationV2.get_sorting_input_recording(curation),
            windows,
            f"{what} merge sorting input",
        )

    # Uncorrected: the sorting input is the source silenced over the
    # exclusion, and unchanged elsewhere.
    sorting_input = CurationV2.get_sorting_input_recording(uncorrected)
    assert sorting_input.channel_ids.tolist() == sorts["source_ids"]
    got_excluded, got_window = _window_traces(sorting_input, windows)
    assert not np.any(got_excluded)
    np.testing.assert_array_equal(got_window, sorts["source_expected"])
    np.testing.assert_array_equal(sorting_input.get_times(), source.get_times())
    _assert_same_windows(
        read_canonical_recording(
            _member_match_files(uncorrected)["sorting_input"]
        ),
        sorting_input,
        windows,
        "UnitMatch bundle input",
    )
    _assert_same_windows(
        CurationV2.get_recording(uncorrected),
        source,
        windows,
        "uncorrected get_recording (the source recording)",
    )

    # Corrected: the sorting input is the corrected artifact, masked.
    sorting_input = CurationV2.get_sorting_input_recording(corrected_curation)
    _assert_reads_corrected(sorting_input, sorts, "corrected sorting input")
    _assert_same_windows(
        sorting_input, corrected, windows, "corrected sorting input"
    )
    assert not np.any(_window_traces(sorting_input, [excluded])[0])
    np.testing.assert_array_equal(sorting_input.get_times(), source.get_times())
    _assert_same_windows(
        CurationV2.get_recording(corrected_curation),
        corrected,
        windows,
        "corrected get_recording (the sorting input)",
    )


def test_concat_member_trace_accessors_have_one_meaning_each(
    discontinuous_sources, monkeypatch
):
    """For a concatenation masking one member, sorted uncorrected and
    corrected: a member's ``get_source_recording`` (and ``get_recording``) is
    that member's unmasked, uncorrected ``Recording``; its
    ``get_sorting_input_recording`` is the parent's sorting input over the
    member's frames ``[start, end)`` (zero over the member's exclusion; the
    corrected concatenation's traces when corrected) on the member's own
    timestamps. The parent curation's ``get_recording`` is its sorting input,
    and its ``get_source_recording`` refuses, naming the per-member
    accessor."""
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.concat_member_curation import (
        ConcatMemberCuration,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.motion import MotionCorrectedRecording
    from spyglass.spikesorting.v2.recording import Recording
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecording,
        ConcatenatedRecordingSelection,
    )
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection

    fx = discontinuous_sources
    t0 = fx["t0"]
    member_excluded_s = (t0 + 17.0, t0 + 18.0)
    member_keys = [fx["member_a"], fx["member_b"]]
    members = [Recording().get_recording(key) for key in member_keys]
    lengths = [int(m.get_num_samples()) for m in members]
    offsets = np.cumsum([0, *lengths])
    artifact_key = masked_artifact(member_keys[0], list(member_excluded_s))
    concat_key, sort_keys = None, []
    try:
        concat_key = ConcatenatedRecordingSelection.insert_selection(
            {
                "session_group_owner": MOTION_TEAM,
                "session_group_name": CONCAT_GROUP,
                "preprocessing_params_name": "default",
            },
            artifact_detection_ids={
                0: artifact_key["artifact_detection_id"],
                1: None,
            },
        )
        ConcatenatedRecording.populate(concat_key, reserve_jobs=False)
        snapshot_ids = (
            ConcatenatedRecordingSelection.MemberSnapshot & concat_key
        ).fetch("recording_id", order_by="member_index")
        assert [str(r) for r in snapshot_ids] == [
            str(key["recording_id"]) for key in member_keys
        ]
        corrected_key = populated_corrected(
            populated_estimate(
                concat_recording_id=concat_key["concat_recording_id"]
            )
        )
        concat = ConcatenatedRecording().get_recording(concat_key)
        corrected = MotionCorrectedRecording().get_recording(corrected_key)
        assert int(concat.get_num_samples()) == offsets[-1]
        assert (
            np.max(np.abs(corrected.get_traces() - concat.get_traces()))
            > MIN_CORRECTION_UV
        )
        excluded = _inside(members[0].get_times(), member_excluded_s)
        source_excluded = members[0].get_traces(
            start_frame=excluded[0], end_frame=excluded[1]
        )
        assert np.count_nonzero(source_excluded) == source_excluded.size

        planted = np.array(
            [3000, lengths[0] - 3000, offsets[1] + 3000, offsets[2] - 3000]
        )
        for parent in (concat_key, corrected_key):
            sort_keys.append(
                SortingSelection.insert_selection(
                    {**concat_key, **sorter_key(), **parent}
                )
            )
        with monkeypatch.context() as patch:
            plant_sorter(patch, _planted_frames(planted))
            Sorting.populate(sort_keys, reserve_jobs=False)

        for sort_key, parent, what in (
            (sort_keys[0], concat, "uncorrected concat"),
            (sort_keys[1], corrected, "corrected concat"),
        ):
            curation = CurationV2.insert_curation(sorting_key=sort_key)
            ConcatMemberCuration.populate(curation, reserve_jobs=False)
            with pytest.raises(
                ValueError, match="ConcatMemberCuration.get_source_recording"
            ):
                CurationV2.get_source_recording(curation)
            windows = [(0, int(offsets[-1]))]
            _assert_same_windows(
                CurationV2.get_sorting_input_recording(curation),
                parent,
                windows,
                f"{what} sorting input",
            )
            _assert_same_windows(
                CurationV2.get_recording(curation),
                parent,
                windows,
                f"{what} get_recording (the sorting input)",
            )
            for index, member in enumerate(members):
                member_key = {**curation, "member_index": index}
                member_windows = [(0, lengths[index])]
                label = f"{what} member {index}"
                _assert_same_windows(
                    ConcatMemberCuration.get_source_recording(member_key),
                    member,
                    member_windows,
                    f"{label} source recording",
                )
                _assert_same_windows(
                    ConcatMemberCuration.get_recording(member_key),
                    member,
                    member_windows,
                    f"{label} get_recording (the source recording)",
                )
                member_input = ConcatMemberCuration.get_sorting_input_recording(
                    member_key
                )
                assert (
                    member_input.channel_ids.tolist()
                    == parent.channel_ids.tolist()
                ), label
                np.testing.assert_array_equal(
                    member_input.get_traces(),
                    parent.get_traces(
                        start_frame=int(offsets[index]),
                        end_frame=int(offsets[index + 1]),
                    ),
                    err_msg=f"{label} sorting input traces",
                )
                np.testing.assert_array_equal(
                    member_input.get_times(),
                    member.get_times(),
                    err_msg=f"{label} sorting input timestamps",
                )
                merge_key = {
                    "merge_id": (
                        SpikeSortingOutput.ConcatMemberCuration & member_key
                    ).fetch1("merge_id")
                }
                _assert_same_windows(
                    SpikeSortingOutput.get_source_recording(merge_key),
                    member,
                    member_windows,
                    f"{label} merge source recording",
                )
                _assert_same_windows(
                    SpikeSortingOutput.get_sorting_input_recording(merge_key),
                    member_input,
                    member_windows,
                    f"{label} merge sorting input",
                )
            member_input = ConcatMemberCuration.get_sorting_input_recording(
                {**curation, "member_index": 0}
            )
            assert not np.any(_window_traces(member_input, [excluded])[0])
    finally:
        drop_pipeline_sorts([key["sorting_id"] for key in sort_keys])
        if concat_key is not None:
            drop_motion_selections(concat_key)
            (ConcatenatedRecording & concat_key).super_delete(
                warn=False, safemode=False
            )
            (ConcatenatedRecordingSelection & concat_key).super_delete(
                warn=False, safemode=False
            )
        from spyglass.spikesorting.v2.artifact import (
            RecordingArtifactDetection,
            RecordingArtifactSelection,
        )

        (RecordingArtifactDetection & artifact_key).delete(safemode=False)
        (RecordingArtifactSelection & artifact_key).super_delete(warn=False)


def test_motion_cleanup_drops_the_sorts_of_corrected_recordings(
    drift_recording, monkeypatch
):
    """Cleanup removes corrected sorts on an independently owned recording."""
    from spyglass.common import IntervalList
    from spyglass.spikesorting.v2.motion import (
        MotionCorrectedRecording,
        MotionEstimateSelection,
    )
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
    )
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection

    nwb = drift_recording["nwb_file_name"]
    t0 = session_start_s(nwb)
    interval = "motion cleanup isolated interval"
    IntervalList.insert1(
        {
            "nwb_file_name": nwb,
            "interval_list_name": interval,
            "valid_times": np.asarray([[t0 + 16.0, t0 + 20.0]]),
            "pipeline": "motion_cleanup_test",
        },
        skip_duplicates=True,
    )
    parent = (RecordingSelection & drift_recording["recording_key"]).fetch1()
    recording_key = RecordingSelection.insert_selection(
        {
            name: parent[name]
            for name in (
                "nwb_file_name",
                "sort_group_id",
                "team_name",
                "preprocessing_params_name",
            )
        }
        | {"interval_list_name": interval}
    )
    sort_keys = []
    try:
        Recording.populate(recording_key, reserve_jobs=False)
        corrected_key = populated_corrected(populated_estimate(**recording_key))
        for correction in ({}, corrected_key):
            sort_keys.append(
                SortingSelection.insert_selection(
                    {**recording_key, **sorter_key(), **correction}
                )
            )
        plant_sorter(monkeypatch, _planted_sorter({}))
        Sorting.populate(sort_keys, reserve_jobs=False)
        uncorrected_sort, corrected_sort = sort_keys
        assert Sorting & corrected_sort

        drop_motion_selections(recording_key)
        assert not (SortingSelection & corrected_sort)
        assert not (MotionCorrectedRecording & corrected_key)
        assert not (MotionEstimateSelection.RecordingSource & recording_key)
        assert SortingSelection & uncorrected_sort
        assert Sorting & uncorrected_sort
    finally:
        drop_pipeline_sorts([key["sorting_id"] for key in sort_keys])
        drop_motion_selections(recording_key)
        (RecordingSelection & recording_key).super_delete(warn=False)
        (
            IntervalList
            & {"nwb_file_name": nwb, "interval_list_name": interval}
        ).delete_quick()

"""Every analyzer rebuild of a masked sort reuses the sort's statistics spans.

A sort's ``noise_levels`` are estimated only from its persisted statistics
spans (artifact-free frames). Every later builder -- the ``get_analyzer``
self-heal, ``CurationEvaluation`` (cached fast path and merged temp
analyzers), the merged-curation analyzer, and the recompute audit --
reconstructs the masked recording and must pass those same spans, or it
reproduces SpikeInterface's estimate over the zero-filled mask instead. The
display ``noise_levels`` depend only on the recording, the spans, and the
seed, not on the units, so every rebuild must match the sort-time value
exactly.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

_SMOKE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "mearec_polymer_smoke.nwb"
)
#: Fraction of the smoke recording masked by the manual exclusion.
_MASK_START_S, _MASK_STOP_S = 1.8, 2.2


def _noise(analyzer) -> np.ndarray:
    return np.asarray(analyzer.get_extension("noise_levels").get_data())


@pytest.fixture(scope="module")
def masked_planted_sort(dj_conn):
    """A two-unit planted sort on the smoke recording with ~10% masked.

    A manual artifact exclusion over ``[1.8, 2.2)`` s (of 4 s) is masked at
    sort time. Yields the sort key, its display recipe name, the sort-time
    display ``noise_levels``, and SpikeInterface's own seeded estimate on
    the same masked recording (what a rebuild without spans would give).
    """
    import spikeinterface as si
    from spikeinterface.core import get_noise_levels

    from spyglass.common.common_interval import IntervalList
    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2._sorting_analyzer import (
        reconstruct_recording_and_sorting,
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
    from tests.spikesorting.v2._ingest_helpers import (
        _clean_session_v2,
        clear_curations_for,
        copy_and_insert_nwb,
    )

    if not _SMOKE_PATH.exists():
        pytest.skip(f"Fixture {_SMOKE_PATH.name} not found.")

    nwb = copy_and_insert_nwb(_SMOKE_PATH, dest_name="mearec_masked_spans.nwb")
    session = {"nwb_file_name": nwb}
    _clean_session_v2(session)
    initialize_v2_defaults()
    LabTeam.insert1(
        {"team_name": "v2_test_team", "team_description": "v2 spans"},
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
    t0 = float(
        (
            IntervalList
            & {**session, "interval_list_name": "raw data valid times"}
        ).fetch1("valid_times")[0][0]
    )
    art_pk = RecordingArtifactSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "artifact_detection_params_name": "none",
            "manual_excluded_times": [[t0 + _MASK_START_S, t0 + _MASK_STOP_S]],
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
        n = int(recording.get_num_samples())
        unit0 = np.arange(1000, n - 1000, 3000, dtype=np.int64)
        samples = np.concatenate([unit0, unit0 + 1500])
        labels = np.repeat(np.array([0, 1], dtype=np.int32), unit0.size)
        order = np.argsort(samples, kind="stable")
        return si.NumpySorting.from_samples_and_labels(
            samples_list=[samples[order]],
            labels_list=[labels[order]],
            sampling_frequency=recording.get_sampling_frequency(),
        )

    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(Sorting, "_run_sorter", staticmethod(_plant))
        Sorting.populate(sort_pk, reserve_jobs=False)
    finally:
        mp.undo()

    spans = Sorting().get_statistics_spans(sort_pk)
    recording, _ = reconstruct_recording_and_sorting(Sorting(), sort_pk)
    n = int(recording.get_num_samples())
    assert len(spans) == 2 and spans[0][0] == 0 and spans[-1][1] == n, spans
    masked = n - sum(b - a for a, b in spans)
    assert 0.08 < masked / n < 0.12, (masked, n)

    display_name = (Sorting & sort_pk).fetch1("display_waveform_params_name")
    sort_time_noise = _noise(Sorting().get_analyzer(sort_pk))
    seed = int((Sorting & sort_pk).fetch1("effective_random_seed"))
    unspanned_noise = get_noise_levels(
        recording,
        return_in_uV=True,
        random_slices_kwargs={"seed": seed},
    )
    # The fixture only discriminates if dropping the spans visibly changes
    # the estimate (the zero-filled mask pulls SI's MAD down).
    assert np.max(np.abs(unspanned_noise / sort_time_noise - 1)) > 0.05

    yield {
        "sort_key": dict(sort_pk),
        "display_name": display_name,
        "noise": sort_time_noise,
        "unspanned_noise": unspanned_noise,
    }
    clear_curations_for(sort_pk)
    _clean_session_v2(session)


def _drop_display_folder(sort):
    from spyglass.spikesorting.v2._analyzer_cache import analyzer_path

    folder = analyzer_path(sort["sort_key"]["sorting_id"], sort["display_name"])
    shutil.rmtree(folder)
    assert not folder.exists()
    return folder


@pytest.mark.slow
def test_self_heal_rebuild_reuses_persisted_spans(masked_planted_sort):
    """``Sorting.get_analyzer`` rebuilds a missing folder with the sort's spans."""
    from spyglass.spikesorting.v2.sorting import Sorting

    sort = masked_planted_sort
    folder = _drop_display_folder(sort)
    rebuilt = Sorting().get_analyzer(sort["sort_key"])
    assert folder.exists()
    np.testing.assert_array_equal(_noise(rebuilt), sort["noise"])


@pytest.mark.slow
def test_curation_evaluation_builds_reuse_persisted_spans(
    masked_planted_sort, curation_evaluation_defaults, monkeypatch
):
    """``CurationEvaluation`` rebuilds the cached analyzer (fast path) and
    builds merged temp analyzers from the sort's persisted spans."""
    from spyglass.spikesorting.v2 import _sorting_analyzer as sa_mod
    from spyglass.spikesorting.v2._analyzer_cache import load_analyzer_folder
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.metric_curation import (
        CurationEvaluation,
        CurationEvaluationSelection,
    )
    from spyglass.spikesorting.v2.sorting import Sorting
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    sort = masked_planted_sort
    sorting_key = sort["sort_key"]
    spans = Sorting().get_statistics_spans(sorting_key)

    built = []
    real_build = sa_mod.build_analyzer

    def _spy_build(sorting, recording, key, **kwargs):
        folder = real_build(sorting, recording, key, **kwargs)
        built.append(
            {
                "spans": kwargs.get("statistics_spans"),
                "whiten": bool(kwargs["waveform_params"].get("whiten")),
                "noise": _noise(load_analyzer_folder(folder)),
            }
        )
        return folder

    monkeypatch.setattr(sa_mod, "build_analyzer", _spy_build)
    clear_curations_for(sorting_key)
    try:
        # Fast path over a missing canonical folder: the resolved loader
        # rebuilds it inside make_compute.
        folder = _drop_display_folder(sort)
        root = CurationV2.insert_curation(sorting_key=sorting_key)
        root_sel = CurationEvaluationSelection.insert_selection(
            {
                **root,
                "metric_params_name": "minimal",
                "auto_curation_rules_name": "none",
            }
        )
        CurationEvaluation.populate(root_sel, reserve_jobs=False)
        assert CurationEvaluation & root_sel
        assert [b["spans"] for b in built] == [spans]
        np.testing.assert_array_equal(built[0]["noise"], sort["noise"])
        np.testing.assert_array_equal(
            _noise(load_analyzer_folder(folder)), sort["noise"]
        )

        # Applied merge: curation-scoped temp analyzers.
        built.clear()
        merged = CurationV2.create_merged_curation(
            sorting_key,
            merge_groups=[[0, 1]],
            parent_curation_id=root["curation_id"],
        )
        merged_sel = CurationEvaluationSelection.insert_selection(
            {
                **merged,
                "metric_params_name": "minimal",
                "auto_curation_rules_name": "none",
            }
        )
        CurationEvaluation.populate(merged_sel, reserve_jobs=False)
        assert CurationEvaluation & merged_sel
        display = [b for b in built if not b["whiten"]]
        assert len(display) == 1
        assert all(b["spans"] == spans for b in built)
        np.testing.assert_array_equal(display[0]["noise"], sort["noise"])
    finally:
        clear_curations_for(sorting_key)


@pytest.mark.slow
def test_merged_curation_analyzer_reuses_persisted_spans(
    masked_planted_sort, tmp_path
):
    """``build_merged_analyzer`` estimates noise from the sort's spans."""
    from spyglass.spikesorting.v2._curation_analyzer import (
        build_merged_analyzer,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    sort = masked_planted_sort
    clear_curations_for(sort["sort_key"])
    try:
        merged = CurationV2.create_merged_curation(
            sort["sort_key"], merge_groups=[[0, 1]], parent_curation_id=-1
        )
        analyzer = build_merged_analyzer(
            merged,
            sort["display_name"],
            "display",
            analyzer_folder=tmp_path / "merged.analyzer",
        )
        assert len(analyzer.unit_ids) == 1
        np.testing.assert_array_equal(_noise(analyzer), sort["noise"])
    finally:
        clear_curations_for(sort["sort_key"])


@pytest.mark.slow
def test_recompute_audit_rebuilds_noise_from_persisted_spans(
    masked_planted_sort,
):
    """The recompute audit's fresh build hashes to the stored noise levels."""
    from spyglass.spikesorting.v2.recompute import _recompute_analyzer_hashes
    from spyglass.spikesorting.v2.sorting import Sorting

    sort = masked_planted_sort
    # The stored build must still carry the sort-time (span) noise, so a
    # matching fresh hash means the audit rebuilt from the spans too.
    np.testing.assert_array_equal(
        _noise(Sorting().get_analyzer(sort["sort_key"], rebuild=False)),
        sort["noise"],
    )
    stored, fresh = _recompute_analyzer_hashes(
        sort["sort_key"], 4, sort["display_name"]
    )
    assert "noise_levels" in stored
    assert fresh == stored

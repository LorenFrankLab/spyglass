"""Tests for committed-curation evaluation and final merged-unit metrics.

``CurationEvaluation`` scores an existing committed ``CurationV2`` row in that
curation's own unit namespace: merged units get metrics recomputed over their
merged spike trains/templates, not inherited from a contributor. Preview
(draft) curations -- ``apply_merge=False`` with a real proposed merge group --
are rejected at the evaluation boundary.

Tiers:

- ``slow`` / ``integration`` end-to-end tests populate ``CurationEvaluation``
  on the planted two-unit sort and the shared MEArec smoke sort.
- hermetic compute-level tests pin the merged-template metric contract with
  controlled planted templates (no DB populate).
"""

from __future__ import annotations

import pandas as pd
import pytest


# ---------- committed-curation predicate ------------------------------------


@pytest.mark.slow
@pytest.mark.integration
def test_is_committed_curation_distinguishes_preview(planted_two_unit_sort):
    """Root, label-only, and applied-merge rows are committed; a preview is not.

    A preview (``apply_merge=False`` with a real >=2-unit proposed merge group)
    is a draft, not a final downstream curation; ``assert_committed_curation``
    raises for it and is a no-op for the committed states.
    """
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting

    sorting_key = dict(planted_two_unit_sort)
    unit_ids = sorted(
        int(u) for u in (Sorting.Unit & sorting_key).fetch("unit_id")
    )
    assert len(unit_ids) >= 2, "need >=2 units to plant a merge"

    clear_curations_for(planted_two_unit_sort)
    try:
        root = CurationV2.insert_curation(sorting_key=sorting_key)
        assert CurationV2.is_committed_curation(root) is True
        CurationV2.assert_committed_curation(root)  # no raise

        label_only = CurationV2.insert_curation(
            sorting_key,
            parent_curation_id=root["curation_id"],
            labels={unit_ids[0]: ["noise"]},
        )
        assert CurationV2.is_committed_curation(label_only) is True
        CurationV2.assert_committed_curation(label_only)

        merged = CurationV2.create_merged_curation(
            sorting_key,
            merge_groups=[[unit_ids[0], unit_ids[1]]],
            parent_curation_id=root["curation_id"],
        )
        assert CurationV2.is_committed_curation(merged) is True
        CurationV2.assert_committed_curation(merged)

        preview = CurationV2.propose_merge_curation(
            sorting_key,
            merge_groups=[[unit_ids[0], unit_ids[1]]],
            parent_curation_id=root["curation_id"],
        )
        assert CurationV2.is_committed_curation(preview) is False
        with pytest.raises(ValueError, match="preview"):
            CurationV2.assert_committed_curation(
                preview, context="CurationEvaluation"
            )
    finally:
        clear_curations_for(planted_two_unit_sort)


# ---------- DB-free resolved reconstruction helpers -------------------------
#
# CurationEvaluation.make_compute runs in a parallel worker with no DB access,
# so it reconstructs the recording / analyzer from inputs make_fetch resolved.
# These parity tests pin the DB-free helpers against the existing DB-coupled
# paths: a divergence here would silently feed metrics the wrong recording.


def _resolved_recording_inputs(sorting_key):
    """Resolve (the way make_fetch will) the DB-free recording inputs."""
    from spyglass.spikesorting.v2._artifact_intervals import (
        read_artifact_removed_intervals,
    )
    from spyglass.spikesorting.v2.recording import Recording, RecordingSelection
    from spyglass.spikesorting.v2.session_group import ConcatenatedRecording
    from spyglass.spikesorting.v2.sorting import SortingSelection

    source = SortingSelection.resolve_source(sorting_key)
    artifact_detection_id = SortingSelection.resolve_artifact_detection(
        sorting_key
    )
    if source.kind == "recording":
        rec_row = (
            Recording & {"recording_id": source.key["recording_id"]}
        ).fetch1()
        recording_id = source.key["recording_id"]
    else:
        rec_row = (ConcatenatedRecording & source.key).fetch1()
        recording_id = None
    valid_times = None
    if source.kind == "recording" and artifact_detection_id is not None:
        nwb = (
            RecordingSelection & {"recording_id": recording_id}
        ).fetch1("nwb_file_name")
        valid_times = read_artifact_removed_intervals(
            {"artifact_detection_id": artifact_detection_id}, as_dict=True
        )[nwb]
    return source, rec_row, recording_id, artifact_detection_id, valid_times


@pytest.mark.slow
@pytest.mark.integration
def test_resolved_recording_reconstruction_matches_db_path(populated_sorting):
    """The DB-free recording reconstruction equals the DB-coupled path.

    ``reconstruct_recording_for_sorting_from_resolved`` reads the cached
    recording NWB + applies the artifact mask from make_fetch-resolved inputs,
    with no DB access; it must reproduce the same channels/samples/traces as the
    DB-coupled ``reconstruct_recording_and_sorting``.
    """
    import numpy as np

    from spyglass.spikesorting.v2._sorting_analyzer import (
        reconstruct_recording_and_sorting,
        reconstruct_recording_for_sorting_from_resolved,
    )
    from spyglass.spikesorting.v2.sorting import Sorting

    sorting_key = dict(populated_sorting)
    ref_rec, _ = reconstruct_recording_and_sorting(Sorting(), sorting_key)

    source, rec_row, recording_id, artifact_id, valid_times = (
        _resolved_recording_inputs(sorting_key)
    )
    out_rec = reconstruct_recording_for_sorting_from_resolved(
        recording_row=rec_row,
        source_kind=source.kind,
        artifact_valid_times=valid_times,
        artifact_detection_id=artifact_id,
        recording_id=recording_id,
    )

    assert out_rec.get_num_channels() == ref_rec.get_num_channels()
    assert out_rec.get_num_samples() == ref_rec.get_num_samples()
    assert bool(out_rec.get_annotation("is_filtered")) is True
    np.testing.assert_allclose(
        out_rec.get_traces(start_frame=0, end_frame=200),
        ref_rec.get_traces(start_frame=0, end_frame=200),
    )


@pytest.mark.slow
@pytest.mark.integration
def test_resolved_analyzer_loader_matches_get_analyzer(populated_sorting):
    """The DB-free analyzer loader reproduces the cached display analyzer.

    ``load_or_rebuild_analyzer_from_resolved`` loads (or rebuilds) the canonical
    display analyzer from make_fetch-resolved inputs without any DB read; it
    must return the same unit ids and templates as ``Sorting().get_analyzer``.
    """
    import numpy as np

    from spyglass.spikesorting.v2._analyzer_cache import analyzer_path
    from spyglass.spikesorting.v2._sorting_analyzer import (
        fetch_waveform_params,
        load_or_rebuild_analyzer_from_resolved,
        reconstruct_recording_and_sorting,
        resolve_display_waveform_params_name,
    )
    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        Sorting,
        SortingSelection,
    )
    from spyglass.spikesorting.v2.utils import _resolved_job_kwargs

    sorting_key = dict(populated_sorting)
    ref = Sorting().get_analyzer(sorting_key)
    ref_unit_ids = sorted(int(u) for u in ref.unit_ids)

    sorting_id, n_units = (Sorting & sorting_key).fetch1("sorting_id", "n_units")
    display_name = resolve_display_waveform_params_name(Sorting(), sorting_id)
    display_params = fetch_waveform_params(display_name)
    recording, raw_sorting = reconstruct_recording_and_sorting(
        Sorting(), sorting_key
    )
    sorter_row = (
        SorterParameters
        & (
            (SortingSelection & sorting_key).proj(
                "sorter", "sorter_params_name"
            )
        )
    ).fetch1()
    out = load_or_rebuild_analyzer_from_resolved(
        sorting_id=sorting_id,
        n_units=int(n_units),
        analyzer_folder=analyzer_path(sorting_id, display_name),
        waveform_params=display_params,
        recording=recording,
        sorting=raw_sorting,
        sorter_row=sorter_row,
        job_kwargs=_resolved_job_kwargs(sorter_row["job_kwargs"]),
    )
    assert sorted(int(u) for u in out.unit_ids) == ref_unit_ids
    np.testing.assert_allclose(
        out.get_extension("templates").get_data(),
        ref.get_extension("templates").get_data(),
    )

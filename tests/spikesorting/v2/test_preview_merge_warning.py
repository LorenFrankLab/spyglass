"""get_sorting warns when a curation has unapplied (preview) merges.

A curation created with apply_merge=False records PROPOSED merges in MergeGroup
but does not apply them. Consumers (SortedSpikesGroup / decoding) read units
through CurationV2.get_sorting, which returns the UNMERGED units for such a
curation -- silently using oversplit units. get_sorting now warns (and the
docstring documents the preview semantics). get_merged_sorting still applies
the proposal. A plain root curation (no real merge, only 1-element self-entries)
must NOT warn.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.spikesorting.v2._ingest_helpers import (
    clear_curations_for,
    copy_and_insert_nwb,
)

_FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "mearec_polymer_smoke.nwb"
)


@pytest.fixture(scope="module")
def planted_two_unit_sort(dj_conn):
    """A populated Sorting with two planted units (so a merge group exists)."""
    from tests.spikesorting.v2._ingest_helpers import (
        _clean_session_v2,
    )

    if not _FIXTURE_PATH.exists():
        pytest.skip(f"Fixture {_FIXTURE_PATH.name} not found.")

    import spikeinterface as si

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

    nwb = copy_and_insert_nwb(_FIXTURE_PATH, dest_name="mearec_preview.nwb")
    session = {"nwb_file_name": nwb}
    _clean_session_v2(session)
    initialize_v2_defaults()
    LabTeam.insert1(
        {"team_name": "v2_test_team", "team_description": "v2 preview"},
        skip_duplicates=True,
    )
    if not (SortGroupV2 & session):
        SortGroupV2.set_group_by_shank(nwb_file_name=nwb)
    sg = int(sorted((SortGroupV2 & session).fetch("sort_group_id"))[0])
    rec_pk = RecordingSelection.insert_selection(
        {
            "nwb_file_name": nwb,
            "sort_group_id": sg,
            "interval_list_name": "raw data valid times",
            "preprocessing_params_name": "default",
            "team_name": "v2_test_team",
        }
    )
    if not (Recording & rec_pk):
        Recording.populate(rec_pk, reserve_jobs=False)
    art_pk = ArtifactDetectionSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "artifact_detection_params_name": "none",
        }
    )
    if not (ArtifactDetection & art_pk):
        ArtifactDetection.populate(art_pk, reserve_jobs=False)
    sort_pk = SortingSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "sorter": "mountainsort5",
            "sorter_params_name": "franklab_30khz_ms5_2026_06",
            "artifact_detection_id": art_pk["artifact_detection_id"],
        }
    )
    (Sorting & sort_pk).super_delete(warn=False)

    def _plant(
        sorter, sorter_params, recording, sorting_id, *, job_kwargs=None
    ):
        samples = np.array(
            [500, 1500, 2500, 3500, 4500, 600, 1600, 2600, 3600, 4600],
            dtype=np.int64,
        )
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int32)
        return si.NumpySorting.from_samples_and_labels(
            samples_list=[samples],
            labels_list=[labels],
            sampling_frequency=recording.get_sampling_frequency(),
        )

    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(Sorting, "_run_sorter", staticmethod(_plant))
        Sorting.populate(sort_pk, reserve_jobs=False)
    finally:
        mp.undo()
    if len(Sorting.Unit & sort_pk) < 2:
        pytest.skip("planted sort did not yield >=2 units")
    yield sort_pk
    clear_curations_for(sort_pk)
    _clean_session_v2(session)


@pytest.mark.slow
@pytest.mark.integration
def test_get_sorting_warns_on_unapplied_preview_merge(
    planted_two_unit_sort, monkeypatch
):
    import spyglass.spikesorting.v2.curation as curation_mod
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting

    unit_ids = sorted(
        int(u) for u in (Sorting.Unit & planted_two_unit_sort).fetch("unit_id")
    )
    clear_curations_for(planted_two_unit_sort)
    cur = CurationV2.insert_curation(
        sorting_key=planted_two_unit_sort,
        merge_groups=[[unit_ids[0], unit_ids[1]]],
        apply_merge=False,
    )

    seen: list[str] = []
    monkeypatch.setattr(
        curation_mod.logger,
        "warning",
        lambda msg, *a, **k: seen.append(str(msg)),
    )
    try:
        sorting = CurationV2.get_sorting(cur)
        # Preview path returns the UNMERGED units (every original unit kept).
        assert sorting.get_num_units() == len(unit_ids)
        assert any(
            "proposed merges that are NOT applied" in s for s in seen
        ), f"expected an unapplied-preview-merge warning; got {seen}"
    finally:
        clear_curations_for(planted_two_unit_sort)


@pytest.mark.slow
@pytest.mark.integration
def test_get_sorting_no_warning_without_proposed_merges(
    planted_two_unit_sort, monkeypatch
):
    """A plain root curation (no real merge) must NOT emit the preview warning."""
    import spyglass.spikesorting.v2.curation as curation_mod
    from spyglass.spikesorting.v2.curation import CurationV2

    clear_curations_for(planted_two_unit_sort)
    cur = CurationV2.insert_curation(sorting_key=planted_two_unit_sort)

    seen: list[str] = []
    monkeypatch.setattr(
        curation_mod.logger,
        "warning",
        lambda msg, *a, **k: seen.append(str(msg)),
    )
    try:
        CurationV2.get_sorting(cur)
        assert not any(
            "proposed merges that are NOT applied" in s for s in seen
        ), f"root curation should not warn about preview merges; got {seen}"
    finally:
        clear_curations_for(planted_two_unit_sort)


@pytest.mark.slow
@pytest.mark.integration
def test_assert_decoding_merge_ids_raises_on_preview_merge(
    planted_two_unit_sort,
):
    """The consumer-boundary validator RAISES on an apply_merge=False curation
    with proposed merges (the preview get_sorting only warns about)."""
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting

    unit_ids = sorted(
        int(u) for u in (Sorting.Unit & planted_two_unit_sort).fetch("unit_id")
    )
    clear_curations_for(planted_two_unit_sort)
    cur = CurationV2.insert_curation(
        sorting_key=planted_two_unit_sort,
        merge_groups=[[unit_ids[0], unit_ids[1]]],
        apply_merge=False,
    )
    merge_id = (SpikeSortingOutput.CurationV2 & cur).fetch1("merge_id")
    try:
        with pytest.raises(ValueError, match="apply_merge=False"):
            SpikeSortingOutput.assert_decoding_merge_ids_ok([merge_id])
    finally:
        clear_curations_for(planted_two_unit_sort)

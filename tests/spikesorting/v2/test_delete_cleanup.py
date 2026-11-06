"""Deletes clean up files only for rows actually deleted.

DataJoint's delete returns normally when the safemode prompt is declined,
so cleanup must not assume the rows are gone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spyglass.spikesorting.v2._core.recipe_catalog import (
    CORTEX_DISPLAY_WAVEFORMS,
)
from tests.spikesorting.v2._ingest_helpers import copy_and_insert_nwb
from tests.spikesorting.v2._sorter_stub import plant_sorter

_FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "mearec_polymer_smoke.nwb"
)
# 'default' preprocessing maps to the cortex display recipe.
_DISPLAY = CORTEX_DISPLAY_WAVEFORMS


@pytest.fixture(scope="module")
def planted_sort(dj_conn, smoke_nwb):
    """A populated Sorting (planted units) with an on-disk analyzer folder."""
    from tests.spikesorting.v2._ingest_helpers import (
        _clean_session_v2,
    )

    if not _FIXTURE_PATH.exists():
        pytest.skip(f"Fixture {_FIXTURE_PATH.name} not found.")

    import spikeinterface as si

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
    from spyglass.spikesorting.v2.sorting import (
        Sorting,
        SortingSelection,
    )

    nwb = copy_and_insert_nwb(_FIXTURE_PATH, dest_name="mearec_delete.nwb")
    session = {"nwb_file_name": nwb}
    _clean_session_v2(session)
    initialize_v2_defaults()
    LabTeam.insert1(
        {"team_name": "v2_test_team", "team_description": "v2 delete"},
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
    art_pk = RecordingArtifactSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "artifact_detection_params_name": "none",
        }
    )
    if not (RecordingArtifactDetection & art_pk):
        RecordingArtifactDetection.populate(art_pk, reserve_jobs=False)
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
        sorter,
        sorter_params,
        recording,
        sorting_id,
        *,
        job_kwargs=None,
        execution_params=None,
        statistics_spans=None,
    ):
        samples = np.array([500, 1500, 2500, 3500, 4500], dtype=np.int64)
        labels = np.zeros(samples.size, dtype=np.int32)
        return si.NumpySorting.from_samples_and_labels(
            samples_list=[samples],
            labels_list=[labels],
            sampling_frequency=recording.get_sampling_frequency(),
        )

    mp = pytest.MonkeyPatch()
    try:
        plant_sorter(mp, _plant)
        Sorting.populate(sort_pk, reserve_jobs=False)
    finally:
        mp.undo()
    assert Sorting & sort_pk, "planted Sorting.populate produced no row"
    yield sort_pk
    _clean_session_v2(session)


@pytest.mark.slow
@pytest.mark.integration
def test_cancelled_delete_preserves_analyzer_folder_and_row(
    planted_sort, monkeypatch
):
    from spyglass.spikesorting.v2.sorting import Sorting
    from spyglass.spikesorting.v2._storage.analyzer_cache import analyzer_path

    folder = analyzer_path(planted_sort["sorting_id"], _DISPLAY)
    assert folder.exists(), "fixture should have created the analyzer folder"

    # Answer "no". Patched at the call site, datajoint.table, not utils.
    monkeypatch.setattr("datajoint.table.user_choice", lambda *a, **k: "no")

    (Sorting & planted_sort).delete(safemode=True)

    assert Sorting & planted_sort, "cancelled delete removed the DB row"
    assert folder.exists(), (
        "cancelled delete destroyed the analyzer folder for a row the user "
        "chose to keep"
    )


@pytest.mark.slow
@pytest.mark.integration
def test_cancelled_artifact_delete_preserves_interval_list(
    planted_sort, monkeypatch
):
    """A cancelled RecordingArtifactDetection.delete keeps the artifact
    IntervalList rows.

    Uses an unreferenced detection; a referenced one is refused outright.
    """
    from spyglass.common import IntervalList
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetectionParameters,
        RecordingArtifactDetection,
        RecordingArtifactSelection,
    )
    from spyglass.spikesorting.v2.artifact_output import (
        ArtifactDetectionOutput,
    )
    from spyglass.spikesorting.v2.recording import RecordingSelection
    from spyglass.spikesorting.v2.sorting import SortingSelection
    from spyglass.spikesorting.v2._artifacts.naming import (
        artifact_detection_interval_list_name,
    )

    rec_id = SortingSelection.resolve_source(planted_sort).key["recording_id"]
    nwb = (RecordingSelection & {"recording_id": rec_id}).fetch1(
        "nwb_file_name"
    )
    # 'default' params, unlike the planted sort's 'none', so unreferenced.
    ArtifactDetectionParameters.insert_default()
    art_pk = RecordingArtifactSelection.insert_selection(
        {"recording_id": rec_id, "artifact_detection_params_name": "default"}
    )
    art_id = art_pk["artifact_detection_id"]
    if not (RecordingArtifactDetection & art_pk):
        RecordingArtifactDetection.populate(art_pk, reserve_jobs=False)
    il_restr = {
        "nwb_file_name": nwb,
        "interval_list_name": artifact_detection_interval_list_name(art_id),
    }
    assert (
        IntervalList & il_restr
    ), "populated detection should have an artifact IntervalList"

    try:
        monkeypatch.setattr("datajoint.table.user_choice", lambda *a, **k: "no")
        (RecordingArtifactDetection & {"artifact_detection_id": art_id}).delete(
            safemode=True
        )

        assert RecordingArtifactDetection & {
            "artifact_detection_id": art_id
        }, "cancelled delete removed the RecordingArtifactDetection master row"
        assert IntervalList & il_restr, (
            "cancelled artifact delete removed the IntervalList row for a "
            "master the user chose to keep"
        )
        # The merge registration survives too.
        assert ArtifactDetectionOutput.get_merge_id(
            {"artifact_detection_id": art_id}
        ), "cancelled delete removed the ArtifactDetectionOutput registration"
    finally:
        (
            RecordingArtifactDetection & {"artifact_detection_id": art_id}
        ).super_delete(warn=False, force_masters=True)
        (RecordingArtifactSelection & art_pk).super_delete(warn=False)


@pytest.mark.slow
@pytest.mark.integration
def test_unrestricted_artifact_delete_with_arg_cleans_interval_list(
    planted_sort,
):
    """``RecordingArtifactDetection().delete(restr)`` cleans only
    actually-deleted rows.

    On the unrestricted instance ``len(self)`` counts unrelated rows, so it
    cannot decide what was deleted.
    """
    from spyglass.common import IntervalList
    from spyglass.spikesorting.v2.artifact import (
        RecordingArtifactDetection,
        RecordingArtifactSelection,
    )
    from spyglass.spikesorting.v2.recording import RecordingSelection
    from spyglass.spikesorting.v2.sorting import SortingSelection
    from spyglass.spikesorting.v2._artifacts.naming import (
        artifact_detection_interval_list_name,
    )

    original_art_id = SortingSelection.resolve_artifact_detection(planted_sort)
    assert original_art_id is not None, "planted sort must be artifact-backed"
    rec_id = SortingSelection.resolve_source(planted_sort).key["recording_id"]
    nwb = (RecordingSelection & {"recording_id": rec_id}).fetch1(
        "nwb_file_name"
    )

    target_pk = RecordingArtifactSelection.insert_selection(
        {
            "recording_id": rec_id,
            "artifact_detection_params_name": "default",
        }
    )
    target_art_id = target_pk["artifact_detection_id"]
    assert target_art_id != original_art_id
    interval_restr = {
        "nwb_file_name": nwb,
        "interval_list_name": artifact_detection_interval_list_name(
            target_art_id
        ),
    }

    # Clean leftovers with super_delete: a master without its part row is
    # not deletable through the public override.
    if RecordingArtifactDetection & target_pk:
        (RecordingArtifactDetection & target_pk).super_delete(warn=False)
    (IntervalList & interval_restr).delete_quick()

    try:
        RecordingArtifactDetection.insert1(target_pk, allow_direct_insert=True)
        IntervalList.insert1(
            {
                **interval_restr,
                "valid_times": np.empty((0, 2)),
            },
            allow_direct_insert=True,
        )
        RecordingArtifactDetection.RemovedInterval.insert1(
            {**target_pk, **interval_restr}
        )
        assert RecordingArtifactDetection & {
            "artifact_detection_id": original_art_id
        }, (
            "fixture should leave an unrelated RecordingArtifactDetection row "
            "in the unrestricted table"
        )

        RecordingArtifactDetection().delete(
            {"artifact_detection_id": target_art_id}, safemode=False
        )

        assert not (RecordingArtifactDetection & target_pk)
        assert not (IntervalList & interval_restr), (
            "unrestricted RecordingArtifactDetection().delete(restr) left the "
            "deleted row's artifact IntervalList behind"
        )
        assert RecordingArtifactDetection & {
            "artifact_detection_id": original_art_id
        }
    finally:
        if RecordingArtifactDetection & target_pk:
            (RecordingArtifactDetection & target_pk).super_delete(warn=False)
        (IntervalList & interval_restr).delete_quick()
        (RecordingArtifactSelection & target_pk).delete_quick()


@pytest.mark.slow
@pytest.mark.integration
def test_unrestricted_sorting_delete_with_positional_restriction_restricts(
    planted_sort,
):
    """``Sorting().delete(restr)`` treats a leading positional dict as a
    restriction, not as ``cautious_delete``'s ``force_permission``.

    A restriction matching nothing deletes nothing.
    """
    import uuid

    from spyglass.spikesorting.v2._storage.analyzer_cache import analyzer_path
    from spyglass.spikesorting.v2.sorting import Sorting

    folder = analyzer_path(planted_sort["sorting_id"], _DISPLAY)
    assert Sorting & planted_sort, "fixture should have a planted Sorting row"
    assert folder.exists(), "fixture should have created the analyzer folder"

    nonexistent = uuid.UUID("00000000-0000-0000-0000-000000000000")
    Sorting().delete({"sorting_id": nonexistent}, safemode=False)

    assert Sorting & planted_sort, (
        "Sorting().delete({nonexistent}) deleted the planted row -- the "
        "leading dict was treated as force_permission, not a restriction"
    )
    assert folder.exists(), (
        "Sorting().delete({nonexistent}) destroyed the planted analyzer "
        "folder for a row that should not have matched the restriction"
    )

"""A cancelled Sorting.delete must NOT destroy the analyzer folder.

Sorting.delete cleaned up the 5-50 GB analyzer scratch folder after
super().delete(). But DataJoint's delete returns normally when the user
answers "no" to the safemode prompt (the cascade is cancelled, rows stay) --
the old code then rmtree'd the folder anyway, destroying data for a row the
user chose to keep. The fix only removes a folder whose DB row was actually
deleted. The commit path (folder removed) is covered by
test_sorting_delete_removes_analyzer_folder.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.spikesorting.v2._ingest_helpers import copy_and_insert_nwb

_FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "mearec_polymer_smoke.nwb"
)


@pytest.fixture(scope="module")
def planted_sort(dj_conn):
    """A populated Sorting (planted units) with an on-disk analyzer folder."""
    from tests.spikesorting.v2.test_single_session_pipeline import (
        _clean_session_v2,
    )

    if not _FIXTURE_PATH.exists():
        pytest.skip(f"Fixture {_FIXTURE_PATH.name} not found.")

    import spikeinterface as si

    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetection,
        ArtifactSelection,
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
            "preproc_params_name": "default_franklab",
            "team_name": "v2_test_team",
        }
    )
    if not (Recording & rec_pk):
        Recording.populate(rec_pk, reserve_jobs=False)
    art_pk = ArtifactSelection.insert_selection(
        {"recording_id": rec_pk["recording_id"], "artifact_params_name": "none"}
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
    (Sorting & sort_pk).super_delete(warn=False)

    def _plant(
        sorter, sorter_params, recording, sorting_id, *, job_kwargs=None
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
        mp.setattr(Sorting, "_run_sorter", staticmethod(_plant))
        Sorting.populate(sort_pk, reserve_jobs=False)
    finally:
        mp.undo()
    if not (Sorting & sort_pk):
        pytest.skip("planted Sorting.populate produced no row")
    yield sort_pk
    _clean_session_v2(session)


@pytest.mark.slow
@pytest.mark.integration
def test_cancelled_delete_preserves_analyzer_folder_and_row(
    planted_sort, monkeypatch
):
    from spyglass.spikesorting.v2.sorting import Sorting
    from spyglass.spikesorting.v2._analyzer_cache import analyzer_path

    folder = analyzer_path(planted_sort["sorting_id"])
    assert folder.exists(), "fixture should have created the analyzer folder"

    # Force a real safemode prompt and answer "no" (cancel the cascade).
    # user_choice is imported into datajoint.table, so patch it there (the
    # call site), not datajoint.utils. A clean "no" makes dj cancel the
    # transaction properly (no EOF, no leaked transaction state).
    monkeypatch.setattr("datajoint.table.user_choice", lambda *a, **k: "no")

    (Sorting & planted_sort).delete(safemode=True)

    # The cascade was cancelled: the row -- and its analyzer folder -- survive.
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
    """A cancelled ArtifactDetection.delete must NOT remove the artifact
    IntervalList rows (the twin of the Sorting.delete cancel guard)."""
    from spyglass.common import IntervalList
    from spyglass.spikesorting.v2.artifact import ArtifactDetection
    from spyglass.spikesorting.v2.recording import RecordingSelection
    from spyglass.spikesorting.v2.sorting import SortingSelection
    from spyglass.spikesorting.v2.utils import artifact_interval_list_name

    art_id = SortingSelection.resolve_artifact(planted_sort)
    assert art_id is not None, "planted sort must be artifact-backed"
    rec_id = SortingSelection.resolve_source(planted_sort).key["recording_id"]
    nwb = (RecordingSelection & {"recording_id": rec_id}).fetch1(
        "nwb_file_name"
    )
    il_restr = {
        "nwb_file_name": nwb,
        "interval_list_name": artifact_interval_list_name(art_id),
    }
    assert (
        IntervalList & il_restr
    ), "fixture should have an artifact IntervalList"

    monkeypatch.setattr("datajoint.table.user_choice", lambda *a, **k: "no")
    (ArtifactDetection & {"artifact_id": art_id}).delete(safemode=True)

    assert ArtifactDetection & {
        "artifact_id": art_id
    }, "cancelled delete removed the ArtifactDetection master row"
    assert IntervalList & il_restr, (
        "cancelled artifact delete removed the IntervalList row for a master "
        "the user chose to keep"
    )

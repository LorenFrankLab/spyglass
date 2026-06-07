"""Parallel (n_jobs>1) artifact detection works and matches serial.

The chunked scan re-hydrates the recording in each worker process via
``recording.to_dict()`` -> ``si.load(dict)`` (artifact.py
``_init_artifact_worker``). That path was untested: it could (a) crash if a
cached NWB-backed recording doesn't round-trip cross-process under SI 0.104,
or (b) silently fall back to serial if the recording isn't serializable
(``ensure_n_jobs`` downgrades ``n_jobs`` to 1). This test runs detection at
``n_jobs=2`` on a REAL cached recording (a NumpyRecording would not exercise
the cross-process path -- it isn't json/pickle serializable) and asserts the
flagged valid_times are identical to the serial run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.spikesorting.v2._ingest_helpers import copy_and_insert_nwb

_FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "mearec_polymer_smoke.nwb"
)


def _insert_artifact_params(name, *, n_jobs):
    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )
    from spyglass.spikesorting.v2.artifact import ArtifactDetectionParameters

    ArtifactDetectionParameters().insert1(
        {
            "artifact_params_name": name,
            "params": ArtifactDetectionParamsSchema(
                detect=True,
                amplitude_thresh_uV=30.0,
                zscore_thresh=None,
                proportion_above_thresh=0.1,
                removal_window_ms=1.0,
                min_length_s=0.001,
            ).model_dump(),
            "params_schema_version": 2,
            "job_kwargs": None if n_jobs == 1 else {"n_jobs": n_jobs},
        },
        skip_duplicates=True,
    )
    return name


@pytest.mark.slow
@pytest.mark.integration
def test_parallel_artifact_detection_matches_serial(dj_conn):
    from tests.spikesorting.v2.test_single_session_pipeline import (
        _clean_session_v2,
    )

    if not _FIXTURE_PATH.exists():
        pytest.skip(f"Fixture {_FIXTURE_PATH.name} not found.")

    from spyglass.common import IntervalList
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

    nwb = copy_and_insert_nwb(_FIXTURE_PATH, dest_name="mearec_njobs.nwb")
    session = {"nwb_file_name": nwb}
    _clean_session_v2(session)
    try:
        initialize_v2_defaults()
        LabTeam.insert1(
            {"team_name": "v2_test_team", "team_description": "v2 njobs"},
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

        # The cached recording must be serializable, else ensure_n_jobs
        # silently downgrades n_jobs>1 to 1 and "parallel" never happens.
        rec_obj = Recording().get_recording(rec_pk)
        serializable = rec_obj.check_serializability(
            "json"
        ) or rec_obj.check_serializability("pickle")
        assert serializable, (
            "cached recording is not serializable; n_jobs>1 artifact "
            "detection would silently fall back to serial (no parallelism)"
        )

        results = {}
        for name, n_jobs in (("v2_njobs_serial", 1), ("v2_njobs_parallel", 2)):
            _insert_artifact_params(name, n_jobs=n_jobs)
            art_pk = ArtifactSelection.insert_selection(
                {
                    "recording_id": rec_pk["recording_id"],
                    "artifact_params_name": name,
                }
            )
            ArtifactDetection.populate(art_pk, reserve_jobs=False)
            results[name] = (
                IntervalList
                & {
                    "nwb_file_name": nwb,
                    "interval_list_name": f"artifact_{art_pk['artifact_id']}",
                }
            ).fetch1("valid_times")

        # n_jobs=2 completed (no cross-process re-hydration crash) AND the
        # flagged valid_times are identical to the serial scan.
        np.testing.assert_array_equal(
            results["v2_njobs_serial"], results["v2_njobs_parallel"]
        )
    finally:
        _clean_session_v2(session)

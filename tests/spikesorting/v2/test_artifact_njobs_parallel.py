"""Parallel (n_jobs>1) artifact detection works and matches serial.

The chunked scan re-hydrates the recording in each worker process via
``recording.to_dict()`` -> ``si.load(dict)`` (artifact.py
``_init_artifact_worker``). That path was untested: it could (a) crash if a
cached NWB-backed recording doesn't round-trip cross-process under SI 0.104,
or (b) fail to re-hydrate in a spawned worker if the cached recording isn't
pickle-serializable (``ensure_n_jobs`` admits a memory-serializable recording,
but the worker pool still needs the ``to_dict()`` blob to load). This test runs
detection at
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
            "artifact_detection_params_name": name,
            "params": ArtifactDetectionParamsSchema(
                detect=True,
                amplitude_threshold_uv=30.0,
                zscore_threshold=None,
                proportion_above_threshold=0.1,
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
    from tests.spikesorting.v2._ingest_helpers import (
        _clean_session_v2,
    )

    if not _FIXTURE_PATH.exists():
        pytest.skip(f"Fixture {_FIXTURE_PATH.name} not found.")

    from spyglass.common import IntervalList
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
                "preprocessing_params_name": "default",
                "team_name": "v2_test_team",
            }
        )
        if not (Recording & rec_pk):
            Recording.populate(rec_pk, reserve_jobs=False)

        # The cached recording must be pickle-serializable, else the spawned
        # worker pool can't re-hydrate it from to_dict() and the n_jobs=2 run
        # errors out instead of running in parallel.
        rec_obj = Recording().get_recording(rec_pk)
        serializable = rec_obj.check_serializability(
            "json"
        ) or rec_obj.check_serializability("pickle")
        assert serializable, (
            "cached recording is not pickle-serializable; the n_jobs>1 worker "
            "pool could not re-hydrate it cross-process"
        )

        results = {}
        for name, n_jobs in (("v2_njobs_serial", 1), ("v2_njobs_parallel", 2)):
            _insert_artifact_params(name, n_jobs=n_jobs)
            art_pk = ArtifactDetectionSelection.insert_selection(
                {
                    "recording_id": rec_pk["recording_id"],
                    "artifact_detection_params_name": name,
                }
            )
            ArtifactDetection.populate(art_pk, reserve_jobs=False)
            results[name] = (
                IntervalList
                & {
                    "nwb_file_name": nwb,
                    "interval_list_name": f"artifact_detection_{art_pk['artifact_detection_id']}",
                }
            ).fetch1("valid_times")

        # n_jobs=2 completed (no cross-process re-hydration crash) AND the
        # flagged valid_times are identical to the serial scan. This run
        # exercises the cross-process re-hydration of a REAL cached NWB-backed
        # recording; the smoke fixture is artifact-free so both sides return a
        # single whole-recording interval. The non-vacuous chunk-seam
        # determinism (an artifact straddling a chunk boundary, detected
        # identically serial vs parallel) is pinned by
        # ``test_chunk_seam_detection_is_deterministic_serial_vs_parallel``.
        np.testing.assert_array_equal(
            results["v2_njobs_serial"], results["v2_njobs_parallel"]
        )
    finally:
        _clean_session_v2(session)


@pytest.mark.slow
@pytest.mark.integration
def test_chunk_seam_detection_is_deterministic_serial_vs_parallel(
    dj_conn, tmp_path
):
    """A real artifact straddling a chunk boundary is detected identically by
    a single-pass serial scan and a small-chunk multi-process scan.

    The production scan is frame-local (each frame's flag depends only on that
    frame's across-channel values), so the chunked output is meant to be
    frame-identical to a single pass regardless of where chunk boundaries
    fall. The fixture-based test above can only confirm this on artifact-free
    data, where both sides trivially return one whole-recording interval. This
    test makes the comparison non-vacuous: a 200 uV transient spans frames
    1000-1199, and a ``chunk_size`` of 1100 frames places a chunk boundary at
    frame 1100 -- INSIDE the transient. A seam bug (a chunk dropping or
    double-counting its boundary frame, or mis-concatenating per-chunk flags)
    would shift the assembled valid-time boundary and break the equality.

    The recording is saved to disk so it is pickle-serializable. An in-memory
    ``NumpyRecording`` is memory-serializable (so ``ensure_n_jobs`` admits
    ``n_jobs>1``) but cannot round-trip into a spawned worker via
    ``to_dict()`` -> ``si.load``, so the multi-process scan needs a
    file-backed recording.
    """
    import numpy as np
    import spikeinterface as si
    from spikeinterface.core.job_tools import ensure_n_jobs

    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )
    from spyglass.spikesorting.v2.artifact import ArtifactDetection

    fs = 30_000.0
    n_samples = 5000
    n_channels = 4
    transient = slice(1000, 1200)  # 200 frames, all channels, 200 uV
    traces = np.zeros((n_samples, n_channels), dtype=np.float32)
    traces[transient, :] = 200.0

    rec_mem = si.NumpyRecording(traces_list=[traces], sampling_frequency=fs)
    rec_mem.set_channel_gains([1.0] * n_channels)  # gain 1 -> traces are uV
    rec_mem.set_channel_offsets([0.0] * n_channels)
    # Persist to disk so the recording round-trips cross-process into spawned
    # workers (an in-memory NumpyRecording is not pickle-serializable).
    rec_disk = rec_mem.save(folder=str(tmp_path / "rec"), overwrite=True)
    rec_disk.set_channel_gains([1.0] * n_channels)
    rec_disk.set_channel_offsets([0.0] * n_channels)
    assert rec_disk.check_serializability(
        "pickle"
    ) or rec_disk.check_serializability("json"), (
        "saved recording is not pickle-serializable; the n_jobs=2 worker pool "
        "could not re-hydrate it and the chunk-seam comparison would be vacuous"
    )
    assert ensure_n_jobs(rec_disk, n_jobs=2) == 2, (
        "ensure_n_jobs did not return 2; the multi-process seam path would "
        "not be exercised"
    )

    params = ArtifactDetectionParamsSchema(
        detect=True,
        amplitude_threshold_uv=50.0,
        zscore_threshold=None,
        proportion_above_threshold=0.5,
        removal_window_ms=0.05,
        join_window_ms=0.0,
        min_length_s=0.001,
    )

    # Reference: single pass (the default 1 s chunk == 30000 frames spans the
    # whole 5000-sample recording), serial.
    reference = ArtifactDetection._detect_artifacts(rec_mem, params)
    # Test: chunk boundary at frame 1100 (inside the transient), multi-process.
    parallel = ArtifactDetection._detect_artifacts(
        rec_disk, params, job_kwargs={"chunk_size": 1100, "n_jobs": 2}
    )

    # Non-vacuity: the transient actually split the timeline into two valid
    # windows -- otherwise "equal" would compare two whole-recording intervals.
    assert reference.shape == (2, 2), (
        "expected the planted transient to split the recording into two valid "
        f"intervals; got {reference.shape} -- comparison would be vacuous"
    )
    np.testing.assert_array_equal(parallel, reference)


def test_artifact_compute_kernels_import_without_db():
    """The parallel-artifact worker kernels import with NO DataJoint-schema /
    ``spyglass.common`` dependency, so a spawned ``n_jobs>1`` worker (macOS
    ``spawn`` re-imports the kernel's DEFINING module) never opens a DB
    connection just to run the DB-free chunk computation.

    Regression guard for the ``BrokenProcessPool`` failure: when the kernels
    lived inline in the DataJoint *schema* module (``artifact.py``), every
    spawned worker activated ``dj.schema`` / ``from spyglass.common import ...``
    at import, so a config/port mismatch -- or a DB-isolated compute node --
    killed the worker before any computation ran. Cold-import the kernel module
    in a fresh subprocess (the spawn worker's view) and assert it pulled in
    neither ``spyglass.common`` nor the ``artifact`` schema module. Hermetic --
    no DB, no container.
    """
    import subprocess
    import sys
    import textwrap

    probe = textwrap.dedent("""
        import sys
        import spyglass.spikesorting.v2._artifact_compute as k

        assert hasattr(k, "_init_artifact_worker")
        assert hasattr(k, "_compute_artifact_chunk")
        leaked = sorted(
            m
            for m in sys.modules
            if m.startswith("spyglass.common")
            or m == "spyglass.spikesorting.v2.artifact"
        )
        assert not leaked, "kernel cold-import pulled in DB-layer modules: " + repr(leaked)
        """)
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True
    )
    assert result.returncode == 0, (
        "artifact compute kernels must import without a DB dependency\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )

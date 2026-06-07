"""Artifact detection works END-TO-END: detect -> persist -> mask -> sort.

The rest of the suite tests artifact detection in three disconnected
pieces: ``ArtifactDetection._detect_artifacts`` on an in-memory recording
(finds a transient), ``Sorting._apply_artifact_mask`` on a *hand-built*
valid_times array (zeros frames), and ``Sorting.make_fetch`` resolving an
``obs_intervals`` (structural). No test runs the full chain -- every
DB-level ``ArtifactDetection.populate`` uses the ``"none"`` preset
(detect=False) or the ``"default"`` preset (500 uV) on clean fixture data,
which detects nothing -- so the seam where a *detected* artifact actually
removes frames from the recording handed to the sorter is unverified.

This test closes that seam with the minimum synthetic surface:

* ``Recording.get_recording`` is monkeypatched to return a synthetic
  recording with a known transient at frames [45000, 45050) on a non-zero
  background (the same substitution pattern as
  ``test_shared_artifact_group_multi_member_union``); the Recording row /
  FK chain is real.
* A real ``ArtifactDetection.populate(detect=True)`` with an amplitude
  threshold *between* the background (100 uV) and the transient (5000 uV)
  runs the real detection + persist path.
* Assertion (a): the persisted ``IntervalList`` valid_times split into two
  intervals straddling the transient time (1.5 s) -- proving detection
  fired and the gap was written. A non-firing run would write a single
  full-window interval and fail this.
* A real ``Sorting.populate`` whose ``_run_sorter`` is monkeypatched only
  to CAPTURE the recording handed to it (masking happens in
  ``make_compute`` *before* ``_run_sorter``, sorting.py:964).
* Assertion (b): the captured recording's transient frames are zeroed and a
  far-away background window is untouched -- proving the *detected* valid
  times propagated into the recording the sorter sees.

The expected transient window is known by construction (we plant it), not
recomputed via the detector's own threshold formula, so the assertions are
not tautological.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.spikesorting.v2._ingest_helpers import copy_and_insert_nwb

_FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "mearec_polymer_smoke.nwb"
)
_SESSION_NWB = "mearec_artifact_e2e.nwb"

_FS = 30000.0
_N_SAMPLES = 90000  # 3 s
_N_CH = 4
_BACKGROUND_UV = 100.0
_TRANSIENT_UV = 5000.0
_ART_LO, _ART_HI = 45000, 45050  # transient at 1.5 s
_AMP_THRESH_UV = 1000.0  # between background and transient
_BG_PROBE_LO, _BG_PROBE_HI = 10000, 10100  # far from the transient


def _synth_recording_with_transient():
    """Synthetic preprocessed recording: constant background + one transient
    on all channels (gain 1.0 so counts == uV).

    A linear probe is attached because ``Sorting.make_compute`` ->
    ``_build_analyzer`` -> ``estimate_sparsity`` requires probe geometry;
    the artifact-detection path does not, but Sorting (which loads the same
    monkeypatched recording) does.
    """
    import probeinterface as pi
    import spikeinterface as si

    traces = np.full((_N_SAMPLES, _N_CH), _BACKGROUND_UV, dtype=np.float32)
    traces[_ART_LO:_ART_HI, :] = _TRANSIENT_UV
    rec = si.NumpyRecording([traces], sampling_frequency=_FS)
    rec.set_channel_gains([1.0] * _N_CH)
    rec.set_channel_offsets([0.0] * _N_CH)
    probe = pi.generate_linear_probe(num_elec=_N_CH, ypitch=20)
    probe.set_device_channel_indices(np.arange(_N_CH))
    return rec.set_probe(probe)


@pytest.fixture(scope="module")
def artifact_e2e_session(dj_conn):
    """Ingest the smoke fixture under a unique session name + a populated
    Recording on sort group 0. Cleans the session on setup and teardown."""
    from tests.spikesorting.v2.test_single_session_pipeline import (
        _clean_session_v2,
    )

    if not _FIXTURE_PATH.exists():
        pytest.skip(
            f"Generated MEArec fixture {_FIXTURE_PATH.name} not found. Run "
            "`python tests/spikesorting/v2/fixtures/generate_mearec.py "
            "--smoke` first."
        )

    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
        SortGroupV2,
    )

    nwb_file_name = copy_and_insert_nwb(_FIXTURE_PATH, dest_name=_SESSION_NWB)
    session_key = {"nwb_file_name": nwb_file_name}

    _clean_session_v2(session_key)
    initialize_v2_defaults()
    LabTeam.insert1(
        {"team_name": "v2_test_team", "team_description": "v2 artifact e2e"},
        skip_duplicates=True,
    )
    if not (SortGroupV2 & session_key):
        SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
    sort_group_id = int(
        sorted((SortGroupV2 & session_key).fetch("sort_group_id"))[0]
    )
    rec_pk = RecordingSelection.insert_selection(
        {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": sort_group_id,
            "interval_list_name": "raw data valid times",
            "preproc_params_name": "default_franklab",
            "team_name": "v2_test_team",
        }
    )
    if not (Recording & rec_pk):
        Recording.populate(rec_pk, reserve_jobs=False)

    yield {
        "nwb_file_name": nwb_file_name,
        "recording_id": rec_pk["recording_id"],
    }

    _clean_session_v2(session_key)


@pytest.mark.slow
@pytest.mark.integration
def test_detected_artifact_is_masked_out_of_the_sorted_recording(
    artifact_e2e_session, monkeypatch
):
    """detect=True -> IntervalList gap -> masked frames in the sorter's input.

    Closes the integration seam between real detection and real masking:
    a planted transient is detected, persisted as a valid-times gap, and
    that gap zeros the corresponding frames in the recording handed to the
    sorter.
    """
    from spyglass.common import IntervalList
    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetection,
        ArtifactDetectionParameters,
        ArtifactSelection,
    )
    from spyglass.spikesorting.v2.recording import Recording
    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        Sorting,
        SortingSelection,
    )

    nwb_file_name = artifact_e2e_session["nwb_file_name"]
    recording_id = artifact_e2e_session["recording_id"]

    # Substitute the loaded preprocessed recording with the synthetic one
    # carrying a known transient. Both ArtifactDetection and Sorting load
    # via Recording().get_recording, so one patch covers both populates.
    monkeypatch.setattr(
        Recording,
        "get_recording",
        lambda self, key: _synth_recording_with_transient(),
    )

    # A detect=True preset whose amplitude threshold sits between the
    # background and the transient; only the transient should fire.
    params_name = "v2_e2e_amp1000"
    ArtifactDetectionParameters().insert1(
        {
            "artifact_params_name": params_name,
            "params": ArtifactDetectionParamsSchema(
                detect=True,
                amplitude_thresh_uV=_AMP_THRESH_UV,
                zscore_thresh=None,
                proportion_above_thresh=1.0,
                removal_window_ms=1.0,
                min_length_s=0.001,
            ).model_dump(),
            "params_schema_version": 2,
            "job_kwargs": None,
        },
        skip_duplicates=True,
    )
    SorterParameters.insert_default()

    art_pk = ArtifactSelection.insert_selection(
        {"recording_id": recording_id, "artifact_params_name": params_name}
    )
    ArtifactDetection.populate(art_pk, reserve_jobs=False)

    # ---- Assertion (a): the persisted valid_times exclude the transient ----
    artifact_id = art_pk["artifact_id"]
    valid_times = (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": f"artifact_{artifact_id}",
        }
    ).fetch1("valid_times")
    assert valid_times.shape == (2, 2), (
        "detect=True should split the window into two valid intervals around "
        f"the transient; got valid_times={valid_times.tolist()!r} (a single "
        "interval means detection did not fire)"
    )
    t_art_start = _ART_LO / _FS
    t_art_end = _ART_HI / _FS
    assert valid_times[0][1] <= t_art_start, valid_times.tolist()
    assert valid_times[1][0] >= t_art_end, valid_times.tolist()

    # ---- Run Sorting, capturing the recording handed to the sorter ----
    sort_pk = SortingSelection.insert_selection(
        {
            "recording_id": recording_id,
            "sorter": "mountainsort5",
            "sorter_params_name": "franklab_tetrode_hippocampus_30kHz_ms5",
            "artifact_id": artifact_id,
        }
    )
    (Sorting & sort_pk).super_delete(warn=False)

    captured: dict = {}

    def _capture_run_sorter(
        sorter, sorter_params, recording, sorting_id, *, job_kwargs=None
    ):
        import spikeinterface as si

        captured["recording"] = recording
        # Plant a couple of spikes in the VALID region so make() completes.
        samples = np.array([1000, 2000, 60000, 70000], dtype=np.int64)
        labels = np.zeros(samples.size, dtype=np.int32)
        return si.NumpySorting.from_samples_and_labels(
            samples_list=[samples],
            labels_list=[labels],
            sampling_frequency=recording.get_sampling_frequency(),
        )

    monkeypatch.setattr(
        Sorting, "_run_sorter", staticmethod(_capture_run_sorter)
    )
    Sorting.populate(sort_pk, reserve_jobs=False)

    # ---- Assertion (b): the detected window is zeroed in the sorter input --
    assert "recording" in captured, "Sorting did not invoke _run_sorter"
    masked = captured["recording"]
    art_traces = masked.get_traces(start_frame=_ART_LO, end_frame=_ART_HI)
    bg_traces = masked.get_traces(
        start_frame=_BG_PROBE_LO, end_frame=_BG_PROBE_HI
    )
    assert np.allclose(art_traces, 0.0), (
        "the detected-artifact frames were NOT zeroed in the recording "
        "handed to the sorter; masking did not propagate from detection. "
        f"max|art|={np.abs(art_traces).max()}"
    )
    assert np.allclose(bg_traces, _BACKGROUND_UV), (
        "a far-from-artifact background window was altered; masking zeroed "
        f"more than the detected window. sample values={bg_traces[0].tolist()}"
    )

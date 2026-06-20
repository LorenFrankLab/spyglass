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
  ``Sorting.make_compute`` *before* the ``_run_sorter`` dispatch).
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
    from tests.spikesorting.v2._ingest_helpers import (
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
            "preprocessing_params_name": "default",
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
        ArtifactDetectionSelection,
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
            "artifact_detection_params_name": params_name,
            "params": ArtifactDetectionParamsSchema(
                detect=True,
                amplitude_threshold_uv=_AMP_THRESH_UV,
                zscore_threshold=None,
                proportion_above_threshold=1.0,
                removal_window_ms=1.0,
                min_length_s=0.001,
            ).model_dump(),
            "params_schema_version": 2,
            "job_kwargs": None,
        },
        skip_duplicates=True,
    )
    SorterParameters.insert_default()

    art_pk = ArtifactDetectionSelection.insert_selection(
        {
            "recording_id": recording_id,
            "artifact_detection_params_name": params_name,
        }
    )
    ArtifactDetection.populate(art_pk, reserve_jobs=False)

    # ---- Assertion (a): the persisted valid_times exclude the transient ----
    artifact_detection_id = art_pk["artifact_detection_id"]
    valid_times = (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": f"artifact_detection_{artifact_detection_id}",
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
            "sorter_params_name": "franklab_30khz_ms5_2026_06",
            "artifact_detection_id": artifact_detection_id,
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


_P60_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "mearec_polymer_128ch_60s.nwb"
)


def _read_gt_spike_frames(nwb_file_name, fs):
    """All ground-truth spike frames (across units) for a session, sorted."""
    import pynwb

    from spyglass.common.common_nwbfile import Nwbfile
    from spyglass.spikesorting.v2._fixtures.mearec_to_nwb import (
        get_ground_truth_units_table,
    )

    raw = Nwbfile().get_abs_path(nwb_file_name)
    times = []
    with pynwb.NWBHDF5IO(raw, "r", load_namespaces=True) as io:
        gt = get_ground_truth_units_table(io.read())
        assert gt is not None, "60s fixture must carry ground-truth units"
        for i in range(len(gt.id[:])):
            times.append(np.asarray(gt["spike_times"][i]))
    all_times = np.concatenate(times)
    return np.sort(np.round(all_times * fs).astype(np.int64))


@pytest.fixture(scope="module")
def gt60_recording(dj_conn):
    """Ingest the 60s polymer GT fixture + populate a real shank-0 Recording.

    Yields (session_key, recording_id). Cleaned on teardown.
    """
    from tests.spikesorting.v2._ingest_helpers import (
        _clean_session_v2,
    )

    if not _P60_PATH.exists():
        pytest.skip(f"60s GT fixture {_P60_PATH.name} not found.")

    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
        SortGroupV2,
    )

    nwb = copy_and_insert_nwb(_P60_PATH, dest_name="mearec_artifact_gt60.nwb")
    session = {"nwb_file_name": nwb}
    _clean_session_v2(session)
    initialize_v2_defaults()
    LabTeam.insert1(
        {"team_name": "v2_test_team", "team_description": "v2 artifact gt"},
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
    yield session, rec_pk["recording_id"]
    _clean_session_v2(session)


@pytest.mark.slow
@pytest.mark.integration
def test_artifact_masking_preserves_clean_gt_spikes(
    gt60_recording, monkeypatch
):
    """Artifacts injected at known GT-spike times are removed from the
    recording, while GT spikes outside the artifact windows are preserved.

    Uses the MEArec ground-truth spike times to probe the detect->mask path:
    inject large transients into the real recording at windows centered on a
    handful of GT spikes, run real detect=True + persist, apply the real
    mask, then assert (a) GT spikes well inside an artifact window are zeroed,
    and (b) GT spikes far from any window keep their original signal. This is
    the deterministic core of "masking removes corrupted spikes without
    destroying clean ones" -- no stochastic sort involved.
    """
    from spyglass.common import IntervalList
    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetection,
        ArtifactDetectionParameters,
        ArtifactDetectionSelection,
    )
    from spyglass.spikesorting.v2.recording import Recording
    from spyglass.spikesorting.v2.sorting import Sorting

    session, recording_id = gt60_recording
    nwb_file_name = session["nwb_file_name"]
    transient_uv = 5000.0
    amp_thresh_uv = 1500.0  # above real spikes, below the injected transient

    # Real preprocessed traces (uV); build an injected copy with transients.
    orig = Recording().get_recording({"recording_id": recording_id})
    fs = float(orig.get_sampling_frequency())
    traces = np.asarray(orig.get_traces(return_in_uV=True), dtype=np.float32)
    n_samples, n_ch = traces.shape

    gt_frames = _read_gt_spike_frames(nwb_file_name, fs)
    gt_frames = gt_frames[(gt_frames >= 0) & (gt_frames < n_samples)]
    assert gt_frames.size >= 20, f"too few GT spikes ({gt_frames.size})"

    # Center a few artifact windows on well-separated GT spikes.
    half = int(round(0.002 * fs))  # +/-2 ms window
    centers = gt_frames[:: max(1, gt_frames.size // 5)][:5]
    injected = traces.copy()
    windows = []
    for c in centers:
        lo, hi = max(0, c - half), min(n_samples, c + half)
        injected[lo:hi, :] = transient_uv
        windows.append((lo, hi))

    # GT spikes well inside a window (zeroed) vs far from every window.
    near = int(round(0.001 * fs))  # within 1 ms of a center -> inside
    far = int(round(0.005 * fs))  # >5 ms from every center -> preserved
    in_art = np.array(
        [f for f in gt_frames if any(abs(f - c) <= near for c in centers)]
    )
    out_art = np.array(
        [f for f in gt_frames if all(abs(f - c) > far for c in centers)]
    )
    assert in_art.size >= 1, "need >=1 GT spike inside an artifact window"
    assert out_art.size >= 5, "need several GT spikes outside artifact windows"

    import spikeinterface as si

    inj_rec = si.NumpyRecording([injected], sampling_frequency=fs)
    inj_rec.set_channel_gains([1.0] * n_ch)
    inj_rec.set_channel_offsets([0.0] * n_ch)
    inj_rec = inj_rec.set_probe(orig.get_probe())

    monkeypatch.setattr(Recording, "get_recording", lambda self, key: inj_rec)

    params_name = "v2_gt_artifact_1500"
    ArtifactDetectionParameters().insert1(
        {
            "artifact_detection_params_name": params_name,
            "params": ArtifactDetectionParamsSchema(
                detect=True,
                amplitude_threshold_uv=amp_thresh_uv,
                zscore_threshold=None,
                proportion_above_threshold=1.0,
                removal_window_ms=1.0,
                min_length_s=0.001,
            ).model_dump(),
            "params_schema_version": 2,
            "job_kwargs": None,
        },
        skip_duplicates=True,
    )
    art_pk = ArtifactDetectionSelection.insert_selection(
        {
            "recording_id": recording_id,
            "artifact_detection_params_name": params_name,
        }
    )
    ArtifactDetection.populate(art_pk, reserve_jobs=False)

    valid_times = (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": f"artifact_detection_{art_pk['artifact_detection_id']}",
        }
    ).fetch1("valid_times")
    # Detection fired on every injected window: each window's center time is
    # excluded from valid_times. (Asserting on the centers rather than the
    # interval count is robust to a center landing near a recording edge,
    # where a window contributes no leading/trailing valid interval.)
    for c in centers:
        ct = c / fs
        covered = any(s <= ct <= e for s, e in valid_times)
        assert not covered, (
            f"injected artifact at {ct:.3f}s was not detected (still inside a "
            "valid interval)"
        )

    masked = Sorting._apply_artifact_mask(
        recording=inj_rec, valid_times=valid_times
    )
    masked_traces = np.asarray(masked.get_traces())

    # (a) GT spikes inside artifact windows are zeroed.
    assert np.allclose(masked_traces[in_art, :], 0.0), (
        "GT spikes inside artifact windows were not masked out: "
        f"max|val|={np.abs(masked_traces[in_art, :]).max()}"
    )
    # (b) GT spikes far from artifacts keep their original (non-zero) signal.
    # Exact equality holds because inj_rec has gain==1.0, so masked raw traces
    # are in the same uV units as ``injected``.
    np.testing.assert_array_equal(
        masked_traces[out_art, :], injected[out_art, :]
    )
    assert (
        np.abs(injected[out_art, :]).max() > 0
    ), "preservation check is vacuous: out-of-artifact signal is all zero"

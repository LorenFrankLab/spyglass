"""ArtifactDetection selection/populate/delete and detect-artifacts tests for the v2 single-session pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from tests.spikesorting.v2.single_session._helpers import (
    _build_synthetic_rec,
)


# ---------- ArtifactDetectionSelection source-part pattern -------------------------


@pytest.mark.slow
def test_artifact_detection_selection_inserts_master_and_source_part(
    populated_recording,
):
    """``insert_selection`` writes exactly one master + one source row."""
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetectionParameters,
        ArtifactDetectionSelection,
    )

    ArtifactDetectionParameters.insert_default()
    key = {
        "recording_id": populated_recording["recording_id"],
        "artifact_detection_params_name": "default",
    }
    # Clean any prior selection so we can assert on the count.
    existing = (
        ArtifactDetectionSelection.RecordingSource
        & {"recording_id": populated_recording["recording_id"]}
    ).fetch("KEY", as_dict=True)
    for row in existing:
        (
            ArtifactDetectionSelection
            & {"artifact_detection_id": row["artifact_detection_id"]}
        ).super_delete(warn=False)

    pk = ArtifactDetectionSelection.insert_selection(key)
    assert isinstance(pk, dict)
    assert set(pk.keys()) == {"artifact_detection_id"}
    assert len(ArtifactDetectionSelection & pk) == 1
    assert len(ArtifactDetectionSelection.RecordingSource & pk) == 1
    assert len(ArtifactDetectionSelection.SharedGroupSource & pk) == 0


@pytest.mark.slow
def test_artifact_detection_selection_is_idempotent(populated_recording):
    """Repeat ``insert_selection`` calls return the same PK; no duplicates."""
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetectionParameters,
        ArtifactDetectionSelection,
    )

    ArtifactDetectionParameters.insert_default()
    key = {
        "recording_id": populated_recording["recording_id"],
        "artifact_detection_params_name": "default",
    }
    pk1 = ArtifactDetectionSelection.insert_selection(key)
    pk2 = ArtifactDetectionSelection.insert_selection(key)
    assert pk1 == pk2
    assert (
        len(
            ArtifactDetectionSelection.RecordingSource
            & {"recording_id": populated_recording["recording_id"]}
            & {"artifact_detection_id": pk1["artifact_detection_id"]}
        )
        == 1
    )


@pytest.mark.slow
def test_artifact_detection_selection_rejects_zero_and_two_sources(
    populated_recording,
):
    """``insert_selection`` requires exactly one source key."""
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetectionParameters,
        ArtifactDetectionSelection,
    )

    ArtifactDetectionParameters.insert_default()

    with pytest.raises(ValueError, match="exactly one source key"):
        ArtifactDetectionSelection.insert_selection(
            {"artifact_detection_params_name": "default"}  # no source
        )

    with pytest.raises(ValueError, match="exactly one source key"):
        ArtifactDetectionSelection.insert_selection(
            {
                "recording_id": populated_recording["recording_id"],
                "shared_artifact_group_name": "fake",
                "artifact_detection_params_name": "default",
            }
        )


@pytest.mark.slow
def test_artifact_detection_selection_resolve_source_returns_recording_kind(
    populated_recording,
):
    """``resolve_source`` returns kind='recording' for a single-rec selection."""
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetectionParameters,
        ArtifactDetectionSelection,
    )
    from spyglass.spikesorting.v2.utils import SourceResolution

    ArtifactDetectionParameters.insert_default()
    pk = ArtifactDetectionSelection.insert_selection(
        {
            "recording_id": populated_recording["recording_id"],
            "artifact_detection_params_name": "default",
        }
    )
    resolution = ArtifactDetectionSelection.resolve_source(pk)
    assert isinstance(resolution, SourceResolution)
    assert resolution.kind == "recording"
    assert resolution.key == {
        "recording_id": populated_recording["recording_id"]
    }


@pytest.mark.slow
def test_shared_artifact_group_insert_validates_inputs(populated_recording):
    """``SharedArtifactGroup.insert_group`` validates session
    consistency, recording existence, sampling-frequency match,
    duration / n_samples match, and non-empty members.

    The time-axis check is intentionally stricter than "same
    nwb_file_name" -- ``RecordingSelection`` identity includes
    ``interval_list_name``, so same NWB does NOT imply same
    n_samples or fs. The invariants asserted here mirror what
    ``si.aggregate_channels`` requires inside ``make_compute``.
    """
    from spyglass.spikesorting.v2.artifact import SharedArtifactGroup

    # Empty members -> ValueError.
    with pytest.raises(ValueError, match="members list is empty"):
        SharedArtifactGroup.insert_group("empty_group", [])

    # Member missing recording_id -> ValueError.
    with pytest.raises(ValueError, match="must include 'recording_id'"):
        SharedArtifactGroup.insert_group(
            "bad_member", [{"not_recording_id": "x"}]
        )

    # Non-existent recording_id -> ValueError.
    import uuid

    bogus_rid = uuid.uuid4()
    with pytest.raises(ValueError, match="not in Recording"):
        SharedArtifactGroup.insert_group(
            "bogus_recording",
            [{"recording_id": bogus_rid}],
        )

    # Happy path: one real recording -> master + Member rows
    # written in one transaction. With a single member every
    # multi-member-consistency check (sessions / fs / duration)
    # trivially passes; the multi-member-divergence checks below
    # exercise the real validation paths.
    SharedArtifactGroup.insert_group(
        "v2_solo_group",
        [{"recording_id": populated_recording["recording_id"]}],
    )
    assert (
        len(
            SharedArtifactGroup
            & {"shared_artifact_group_name": "v2_solo_group"}
        )
        == 1
    )
    assert (
        len(
            SharedArtifactGroup.Member
            & {"shared_artifact_group_name": "v2_solo_group"}
        )
        == 1
    )


@pytest.mark.slow
def test_shared_artifact_group_insert_rejects_mismatched_durations(
    populated_recording, polymer_smoke_session
):
    """``insert_group`` rejects members whose recordings span
    different time windows (different ``interval_list_name`` on the
    upstream selection).

    Same NWB, same fs -- but two ``RecordingSelection`` rows
    pointing at different ``interval_list_name`` values can have
    different sample counts. ``si.aggregate_channels`` would
    crash with an opaque shape mismatch deep inside SI; the
    insert-time guard loads each member's preprocessed recording
    and requires EXACT time-axis parity. For two genuinely
    different-duration members the exact-timestamp check
    (``artifact.py`` ~460-485) trips first -- the timestamp
    vectors have different shapes -- raising before the post-loop
    ``distinct_n_samples`` check is reached; either way the
    mismatch surfaces before the user pays the populate cost. (The
    pure n_samples branch is only reachable when timestamp shapes
    match but counts differ, which the per-member ``len(timestamps)
    == n_samples`` assertion makes impossible for real recordings;
    the dtype branch is covered by the companion stub test below.)
    """
    from spyglass.common.common_interval import IntervalList
    from spyglass.spikesorting.v2.artifact import SharedArtifactGroup
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
        SortGroupV2,
    )

    nwb_file_name = polymer_smoke_session["nwb_file_name"]

    # Build a SECOND RecordingSelection on the same NWB but with a
    # shorter IntervalList, then populate Recording so we have two
    # Recording rows with the same nwb_file_name but different
    # duration_s.
    raw_times = (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": "raw data valid times",
        }
    ).fetch1("valid_times")
    t0 = float(raw_times[0][0])
    truncated = np.asarray([[t0, t0 + 1.5]])  # 1.5s vs full ~4s
    truncated_interval = "v2_shared_group_test_truncated"
    IntervalList.insert1(
        {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": truncated_interval,
            "valid_times": truncated,
            "pipeline": "shared_group_validation_test",
        },
        skip_duplicates=True,
    )
    sort_group_id = int(
        sorted((SortGroupV2 & polymer_smoke_session).fetch("sort_group_id"))[0]
    )
    second_rec_pk = RecordingSelection.insert_selection(
        {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": sort_group_id,
            "interval_list_name": truncated_interval,
            "preprocessing_params_name": "default",
            "team_name": "v2_test_team",
        }
    )
    Recording.populate(second_rec_pk, reserve_jobs=False)

    # Now attempt to group the full-window populated_recording AND
    # the truncated second_rec_pk. The strict time-axis check at
    # insert_group time MUST raise -- different durations give
    # different-shape timestamp vectors, so the exact-timestamp
    # check fires before the post-loop n_samples check.
    with pytest.raises(ValueError, match="differing exact timestamps"):
        SharedArtifactGroup.insert_group(
            "v2_mismatched_durations",
            [
                {"recording_id": populated_recording["recording_id"]},
                {"recording_id": second_rec_pk["recording_id"]},
            ],
        )
    # No master row was created.
    assert (
        len(
            SharedArtifactGroup
            & {"shared_artifact_group_name": "v2_mismatched_durations"}
        )
        == 0
    )


@pytest.mark.slow
def test_shared_artifact_group_insert_rejects_mismatched_dtypes(
    populated_recording, polymer_smoke_session, monkeypatch
):
    """``insert_group`` rejects members whose preprocessed
    recordings have differing ``get_dtype()`` values.

    Constructing two real recordings of the same NWB with
    different dtypes requires preproc-param plumbing the test
    fixtures don't currently exercise; monkeypatch
    ``Recording.get_recording`` to return SI-compatible stubs that
    share ``n_samples`` but differ in ``get_dtype()``. This
    exercises the dtype branch of the strict check; the
    n_samples branch is exercised by
    ``test_shared_artifact_group_insert_rejects_mismatched_durations``
    above.
    """
    from spyglass.common.common_interval import IntervalList
    from spyglass.spikesorting.v2.artifact import SharedArtifactGroup
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
        SortGroupV2,
    )

    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    raw_times = (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": "raw data valid times",
        }
    ).fetch1("valid_times")
    t0 = float(raw_times[0][0])
    second_interval = "v2_shared_group_test_dtype_window"
    # 2.5s window is well above the min_segment_length=1.0s filter
    # the Recording pipeline applies; we only need a SECOND
    # Recording row so the test has two distinct recording_ids to
    # group, the dtype divergence is injected via monkeypatch
    # below.
    IntervalList.insert1(
        {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": second_interval,
            "valid_times": np.asarray([[t0, t0 + 2.5]]),
            "pipeline": "shared_group_dtype_test",
        },
        skip_duplicates=True,
    )
    sort_group_id = int(
        sorted((SortGroupV2 & polymer_smoke_session).fetch("sort_group_id"))[0]
    )
    second_rec_pk = RecordingSelection.insert_selection(
        {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": sort_group_id,
            "interval_list_name": second_interval,
            "preprocessing_params_name": "default",
            "team_name": "v2_test_team",
        }
    )
    Recording.populate(second_rec_pk, reserve_jobs=False)

    rid_a = populated_recording["recording_id"]
    rid_b = second_rec_pk["recording_id"]

    class _FakeRec:
        def __init__(self, n_samples, dtype_str):
            self._n = n_samples
            self._dtype = dtype_str
            self._times = np.arange(n_samples, dtype=np.float64) / 30000.0

        def get_num_samples(self):
            return self._n

        def get_dtype(self):
            return self._dtype

        def get_num_segments(self):
            return 1

        def get_times(self):
            return self._times

    def _fake_get_recording(self, key):
        # Both stubs share ``n_samples`` so the dtype branch -- not
        # the n_samples branch -- is the only thing that can raise.
        if str(key["recording_id"]) == str(rid_a):
            return _FakeRec(n_samples=30000, dtype_str="float32")
        return _FakeRec(n_samples=30000, dtype_str="int16")

    monkeypatch.setattr(Recording, "get_recording", _fake_get_recording)

    with pytest.raises(ValueError, match="differing dtypes"):
        SharedArtifactGroup.insert_group(
            "v2_mismatched_dtypes",
            [{"recording_id": rid_a}, {"recording_id": rid_b}],
        )
    assert (
        len(
            SharedArtifactGroup
            & {"shared_artifact_group_name": "v2_mismatched_dtypes"}
        )
        == 0
    )


@pytest.mark.slow
def test_artifact_detection_populates_and_writes_interval_list(
    populated_recording,
):
    """``ArtifactDetection.make`` writes one ``IntervalList`` row under
    ``f"artifact_detection_{artifact_detection_id}"`` and the row is fetchable via
    ``get_artifact_removed_intervals``."""
    from spyglass.common import IntervalList
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetection,
        ArtifactDetectionParameters,
        ArtifactDetectionSelection,
    )
    from spyglass.spikesorting.v2.recording import RecordingSelection

    ArtifactDetectionParameters.insert_default()
    # Clean any prior selection / detection.
    existing = (
        ArtifactDetectionSelection.RecordingSource
        & {"recording_id": populated_recording["recording_id"]}
    ).fetch("KEY", as_dict=True)
    for row in existing:
        ArtifactDetection & row  # noqa: B018 -- silence linter on unused expr
        (
            ArtifactDetection
            & {"artifact_detection_id": row["artifact_detection_id"]}
        ).delete()
        (
            ArtifactDetectionSelection
            & {"artifact_detection_id": row["artifact_detection_id"]}
        ).super_delete(warn=False)

    pk = ArtifactDetectionSelection.insert_selection(
        {
            "recording_id": populated_recording["recording_id"],
            "artifact_detection_params_name": "none",
        }
    )
    ArtifactDetection.populate(pk, reserve_jobs=False)
    assert len(ArtifactDetection & pk) == 1

    nwb_file_name = (RecordingSelection & populated_recording).fetch1(
        "nwb_file_name"
    )
    interval_list_name = f"artifact_detection_{pk['artifact_detection_id']}"
    saved = (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": interval_list_name,
        }
    ).fetch1("valid_times")
    assert saved.shape[1] == 2  # (n_intervals, 2)
    # The "none" preset means no artifacts removed -> full window.
    assert saved.shape[0] == 1
    assert saved[0][0] < saved[0][-1]

    retrieved = ArtifactDetection().get_artifact_removed_intervals(pk)
    assert (retrieved == saved).all()

    # as_dict=True wraps the single-recording array as a one-key dict
    # keyed by nwb_file_name, so source-agnostic callers can avoid
    # branching on the return type (single ndarray vs shared-group dict).
    retrieved_dict = ArtifactDetection().get_artifact_removed_intervals(
        pk, as_dict=True
    )
    assert isinstance(retrieved_dict, dict)
    assert set(retrieved_dict) == {nwb_file_name}
    assert (retrieved_dict[nwb_file_name] == saved).all()


@pytest.mark.slow
def test_artifact_detection_delete_removes_interval_list_row(
    populated_recording,
):
    """``ArtifactDetection.delete()`` cleans up the matching
    ``IntervalList`` row (DataJoint does not cascade through
    ``interval_list_name``-keyed dependencies)."""
    from spyglass.common import IntervalList
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetection,
        ArtifactDetectionParameters,
        ArtifactDetectionSelection,
    )
    from spyglass.spikesorting.v2.recording import RecordingSelection

    ArtifactDetectionParameters.insert_default()
    existing = (
        ArtifactDetectionSelection.RecordingSource
        & {"recording_id": populated_recording["recording_id"]}
    ).fetch("KEY", as_dict=True)
    for row in existing:
        (
            ArtifactDetection
            & {"artifact_detection_id": row["artifact_detection_id"]}
        ).delete()
        (
            ArtifactDetectionSelection
            & {"artifact_detection_id": row["artifact_detection_id"]}
        ).super_delete(warn=False)

    pk = ArtifactDetectionSelection.insert_selection(
        {
            "recording_id": populated_recording["recording_id"],
            "artifact_detection_params_name": "none",
        }
    )
    ArtifactDetection.populate(pk, reserve_jobs=False)

    nwb_file_name = (RecordingSelection & populated_recording).fetch1(
        "nwb_file_name"
    )
    interval_list_name = f"artifact_detection_{pk['artifact_detection_id']}"
    assert IntervalList & {
        "nwb_file_name": nwb_file_name,
        "interval_list_name": interval_list_name,
    }

    (ArtifactDetection & pk).delete()

    assert not (ArtifactDetection & pk)
    assert not (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": interval_list_name,
        }
    )


@pytest.mark.slow
def test_artifact_detection_selection_resolve_source_detects_bypass(
    populated_recording,
):
    """``resolve_source`` raises ``SchemaBypassError`` when a master has
    zero source part rows (an integrity bug, e.g. from direct dj
    insert1 bypassing ``insert_selection``)."""
    import uuid

    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetectionParameters,
        ArtifactDetectionSelection,
    )
    from spyglass.spikesorting.v2.exceptions import SchemaBypassError

    ArtifactDetectionParameters.insert_default()
    orphan_id = uuid.uuid4()
    # Insert master with NO source part to simulate bypass. The master
    # insert guard requires allow_direct_insert=True for this deliberate
    # raw insert (the bypass the resolve_source check exists to catch).
    dj_table = ArtifactDetectionSelection()
    dj_table.insert1(
        {
            "artifact_detection_id": orphan_id,
            "artifact_detection_params_name": "default",
        },
        allow_direct_insert=True,
    )
    try:
        with pytest.raises(SchemaBypassError, match="0 source part"):
            ArtifactDetectionSelection.resolve_source(
                {"artifact_detection_id": orphan_id}
            )
    finally:
        (
            ArtifactDetectionSelection & {"artifact_detection_id": orphan_id}
        ).super_delete(warn=False)


# ---------- Artifact masking at the signal level -------------------------


@pytest.mark.slow
def test_apply_artifact_mask_zeroes_artifact_frames(populated_recording):
    """``Sorting._apply_artifact_mask`` actually zeros artifact frames.

    Existing artifact tests only verify the ``IntervalList`` row was
    written. The ``sip.remove_artifacts`` branch (the actual signal
    masking) is bypassed by the ``"none"`` artifact preset which
    writes a single all-valid interval. This test:

    1. Writes a synthetic IntervalList row whose ``valid_times``
       exclude a known frame range so the artifact "gap" is the
       known range.
    2. Calls ``_apply_artifact_mask`` against the populated_recording's
       preprocessed SI recording.
    3. Asserts traces are zero inside the gap and unchanged outside.

    Also pins the off-by-one boundary: the LAST artifact frame must
    be zeroed (the original code dropped it because the IntervalList
    stored ``[start, end]`` closed where ``end = timestamps[end_f]``;
    the fix stores ``[start, end+1)`` so the complement subtraction
    captures end_f).
    """
    import numpy as np
    import uuid

    from spyglass.common.common_interval import IntervalList
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
    )
    from spyglass.spikesorting.v2.sorting import Sorting

    recording = Recording().get_recording(
        {"recording_id": populated_recording["recording_id"]}
    )
    timestamps = recording.get_times()
    n_frames = len(timestamps)
    assert n_frames > 400, "fixture too short for boundary test"

    # Carve out a synthetic artifact at frames [100, 200) (half-open).
    # valid_times excludes this range: [t0, t[100]) ∪ [t[200], t[-1]].
    artifact_start_f, artifact_end_f = 100, 200
    valid_times = np.asarray(
        [
            [timestamps[0], timestamps[artifact_start_f]],
            [timestamps[artifact_end_f], timestamps[-1]],
        ]
    )

    nwb_file_name = (
        RecordingSelection
        & {"recording_id": populated_recording["recording_id"]}
    ).fetch1("nwb_file_name")
    synthetic_artifact_id = uuid.uuid4()
    interval_list_name = f"artifact_detection_{synthetic_artifact_id}"
    IntervalList.insert1(
        {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": interval_list_name,
            "valid_times": valid_times,
            "pipeline": "spikesorting_artifact_detection_v2_test",
        }
    )
    try:
        # _apply_artifact_mask now takes valid_times directly (no
        # DB I/O inside Sorting.make_compute per the tri-part
        # contract). Pass the same valid_times the IntervalList row
        # carries.
        masked = Sorting._apply_artifact_mask(
            recording=recording,
            valid_times=valid_times,
        )

        # Inside the artifact window: all frames should be 0 across
        # every channel.
        gap = masked.get_traces(
            start_frame=artifact_start_f, end_frame=artifact_end_f
        )
        assert np.all(gap == 0), (
            "Artifact frames [100, 200) are not all zero; "
            f"max abs value in gap = {np.abs(gap).max()}."
        )

        # Boundary check: frame ``artifact_end_f`` (the half-open end)
        # is the first VALID frame after the gap and should NOT be
        # forced to zero by the masker. We compare to the unmasked
        # value, but first assert the source sample is non-zero so the
        # comparison cannot pass vacuously (if the preprocessed signal
        # were 0 here by coincidence, masked == unmasked would be
        # trivially true and the off-by-one guard would be inert).
        unmasked_boundary = recording.get_traces(
            start_frame=artifact_end_f, end_frame=artifact_end_f + 1
        )
        assert np.abs(unmasked_boundary).max() > 0, (
            f"source sample at boundary frame {artifact_end_f} is all-zero; "
            "the masked-vs-unmasked comparison below would be vacuous -- pick "
            "a boundary frame whose preprocessed signal is non-zero"
        )
        masked_boundary = masked.get_traces(
            start_frame=artifact_end_f, end_frame=artifact_end_f + 1
        )
        np.testing.assert_array_equal(
            masked_boundary,
            unmasked_boundary,
            err_msg=(
                f"Frame {artifact_end_f} (first valid frame after the "
                "artifact gap) differs between masked and unmasked "
                "recordings -- off-by-one boundary regression."
            ),
        )

        # And the last artifact frame (artifact_end_f - 1) MUST be
        # zero. This is the exact frame the original off-by-one bug
        # left unmasked.
        last_artifact = masked.get_traces(
            start_frame=artifact_end_f - 1, end_frame=artifact_end_f
        )
        assert np.all(last_artifact == 0), (
            f"Last artifact frame {artifact_end_f - 1} not zeroed -- "
            "this is the off-by-one boundary the v2 fix corrected."
        )
    finally:
        (
            IntervalList
            & {
                "nwb_file_name": nwb_file_name,
                "interval_list_name": interval_list_name,
            }
        ).super_delete(warn=False)


# ---------- ArtifactDetection: signal-level _detect_artifacts ------------


def test_detect_artifacts_finds_known_transient(dj_conn):
    """``ArtifactDetection._detect_artifacts`` runs its amplitude-
    threshold scan body and produces the expected valid-time
    complement of a known synthetic transient.

    Logically a unit test (the body calls a ``@staticmethod`` on an
    in-memory ``NumpyRecording`` -- no DataJoint queries, no NWB
    I/O). ``dj_conn`` is required only because importing the v2
    ``artifact`` module activates ``dj.schema("spikesorting_v2_artifact")``
    at module load time, which needs a live MySQL connection. The
    test itself runs in well under a second and is not slow-marked
    so it executes on every commit.

    Every existing artifact test uses the ``"none"`` preset which
    short-circuits at the ``if not validated.detect`` guard in
    ``_detect_artifacts`` (artifact.py:846) and writes a single all-
    valid interval. That left the entire amplitude-threshold
    detection body (~50 lines, the actual artifact-finding code) at
    0% coverage.

    This test calls ``_detect_artifacts`` directly with a synthetic
    ``NumpyRecording`` carrying a deterministic 200-sample, 200 uV
    transient at frames 1000-1199. The threshold (50 uV) and
    proportion (50% of channels) are set so the transient
    reliably trips detection and nothing else does. Asserts the
    returned ``valid_times`` has exactly two intervals straddling
    the planted transient and excludes the artifact samples.
    """
    import numpy as np
    import spikeinterface as si

    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )
    from spyglass.spikesorting.v2.artifact import ArtifactDetection

    # Build a synthetic 4-channel, 5000-sample recording with a
    # 200 uV transient at frames 1000-1199 across all channels.
    # Gain = 1.0 so traces are already in uV.
    fs = 30_000.0
    n_samples = 5000
    n_channels = 4
    traces = np.zeros((n_samples, n_channels), dtype=np.float32)
    traces[1000:1200, :] = 200.0
    rec = si.NumpyRecording(traces_list=[traces], sampling_frequency=fs)
    rec.set_channel_gains([1.0] * n_channels)

    # detect=True, 50 uV threshold, 50% of channels (= 2 of 4)
    # must exceed simultaneously. The transient hits all 4
    # channels at 200 uV; nothing else exceeds 50 uV. The
    # minimal positive removal_window_ms (Pydantic enforces > 0)
    # expands the artifact span by ceil(removal_window_ms * fs /
    # 2000) frames on each side. At fs=30 kHz and 0.05 ms that's
    # ``ceil(0.05 * 30000 / 2000) = 1`` frame -- intentionally
    # small so the assertions below can be exact.
    params = ArtifactDetectionParamsSchema(
        detect=True,
        amplitude_threshold_uv=50.0,
        zscore_threshold=None,
        proportion_above_threshold=0.5,
        removal_window_ms=0.05,
        join_window_ms=0.0,
        min_length_s=0.001,  # default 1.0 would wipe synthetic-recording intervals
    )

    valid_times = ArtifactDetection._detect_artifacts(rec, params)
    timestamps = rec.get_times()

    # half_window_frames = ceil(0.05 ms * 30 kHz / 2 / 1000) = 1.
    # Artifact span expands from [1000, 1199] to [999, 1200]. The
    # saved valid intervals straddle this expanded artifact run.
    expected_start_f = 999  # 1000 - half_window
    expected_end_f = 1200  # 1199 + half_window
    expected_end_open_f = expected_end_f + 1  # half-open boundary

    assert valid_times.shape == (2, 2), (
        f"Expected exactly 2 valid intervals around the synthetic "
        f"transient at frames 1000-1199, got shape {valid_times.shape}. "
        "_detect_artifacts either skipped the detection body or "
        "merged the artifact and valid runs incorrectly."
    )
    # First valid interval ends at the start of the expanded
    # artifact span.
    assert valid_times[0][0] == pytest.approx(timestamps[0], abs=1e-9)
    assert valid_times[0][1] == pytest.approx(
        timestamps[expected_start_f], abs=1e-9
    ), (
        f"First valid interval ends at {valid_times[0][1]}, expected "
        f"{timestamps[expected_start_f]} (start of the expanded "
        "artifact span)."
    )
    # Second valid interval starts AT the first sample AFTER the
    # expanded artifact span (half-open boundary: the saved
    # interval starts at ``timestamps[end_f + 1]``, so the LAST
    # artifact frame is NOT silently included in the next valid
    # interval).
    assert valid_times[1][0] == pytest.approx(
        timestamps[expected_end_open_f], abs=1e-9
    ), (
        f"Second valid interval starts at {valid_times[1][0]}, expected "
        f"{timestamps[expected_end_open_f]} (first sample AFTER the "
        "expanded artifact span). If this is "
        f"``timestamps[{expected_end_f}]`` the off-by-one fix at "
        "the END of the artifact interval has regressed."
    )
    assert valid_times[1][1] == pytest.approx(timestamps[-1], abs=1e-9)


def test_detect_artifacts_no_threshold_crossings(dj_conn):
    """``_detect_artifacts`` returns the full recording window as a
    single valid interval when no frame trips the threshold.

    Exercises the early-return branch at ``len(frames_above) == 0``
    (artifact.py:926) without short-circuiting at the
    ``detect=False`` guard. Complements the synthetic-transient
    test by covering the "detection ran but found nothing" path.
    """
    import numpy as np
    import spikeinterface as si

    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )
    from spyglass.spikesorting.v2.artifact import ArtifactDetection

    # Recording with all-zero traces -- nothing can exceed any
    # positive threshold.
    rec = si.NumpyRecording(
        traces_list=[np.zeros((2000, 4), dtype=np.float32)],
        sampling_frequency=30_000.0,
    )
    rec.set_channel_gains([1.0] * 4)

    params = ArtifactDetectionParamsSchema(
        detect=True,
        amplitude_threshold_uv=10.0,
        zscore_threshold=None,
        proportion_above_threshold=0.5,
    )
    valid_times = ArtifactDetection._detect_artifacts(rec, params)
    timestamps = rec.get_times()

    assert valid_times.shape == (1, 2)
    assert valid_times[0][0] == pytest.approx(timestamps[0], abs=1e-9)
    assert valid_times[0][1] == pytest.approx(timestamps[-1], abs=1e-9)


def test_detect_artifacts_zscore_only_detection(dj_conn):
    """``_detect_artifacts`` runs the z-score-only branch when
    ``amplitude_threshold_uv is None``.

    The z-score is **across channels per frame** (matching v1's
    semantics); a frame where one channel deviates substantially
    from the others trips. Common-mode pops where every channel
    jumps together do NOT trip (per-frame mean shifts but
    per-frame std stays ~0, so z ~= 0 everywhere on that row).

    Synthetic: low-amplitude per-channel uncorrelated background +
    a 10-sample artifact on ONE channel (the others stay quiet at
    the artifact frames). Cross-channel z on that channel = high.
    """
    import numpy as np

    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )
    from spyglass.spikesorting.v2.artifact import ArtifactDetection

    rng = np.random.default_rng(42)
    # 32 channels so the cross-channel z-score on a single-channel
    # excursion can plausibly exceed ``zscore_threshold``. With N
    # channels and one channel at X while the rest are near zero,
    # the per-frame z on the spiking channel is bounded by
    # ~sqrt(N-1) (an algebraic upper limit of cross-channel
    # z-score, regardless of X). 4 channels caps at ~1.7; 32
    # channels caps at ~5.6 so a threshold of 4.0 sits inside
    # the achievable range with headroom.
    n_samples, n_channels = 5000, 32
    traces = rng.normal(0.0, 0.5, size=(n_samples, n_channels)).astype(
        np.float32
    )
    # Single-channel artifact: channel 0 spikes to 500 uV at frames
    # 2000-2009; the other 31 channels stay quiet.
    traces[2000:2010, 0] = 500.0
    rec = _build_synthetic_rec(traces)

    params = ArtifactDetectionParamsSchema(
        detect=True,
        amplitude_threshold_uv=None,  # z-score only
        zscore_threshold=4.0,
        proportion_above_threshold=1.0 / n_channels,  # any-1-channel triggers
        removal_window_ms=0.05,
        join_window_ms=0.0,
        min_length_s=0.001,  # default 1.0 would wipe synthetic-recording intervals
    )
    valid_times = ArtifactDetection._detect_artifacts(rec, params)
    timestamps = rec.get_times()

    assert valid_times.shape == (2, 2), (
        f"z-score-only detection expected 2 valid intervals around the "
        f"transient at frames 2000-2009, got shape {valid_times.shape}."
    )
    # Pinpoint the artifact run via the expanded boundaries
    # (half_window_frames=1).
    assert valid_times[0][1] == pytest.approx(timestamps[1999], abs=1e-9)
    assert valid_times[1][0] == pytest.approx(timestamps[2011], abs=1e-9)


def test_detect_artifacts_amplitude_and_zscore_combined(dj_conn):
    """``_detect_artifacts`` OR-combines the amplitude and z-score detectors.

    v2 matches v1's per-channel OR (``above_amp | above_z`` in
    ``_artifact_compute._compute_artifact_chunk``): a channel is flagged when
    EITHER detector trips, then a frame is an artifact when enough channels
    are hit. This test discriminates OR from a hypothetical AND by giving each
    detector its OWN region that the other detector cannot see:

    * frames 1000-1099 -- every channel at 100 uV: trips amplitude (>50 uV)
      but NOT the across-channel z-score (uniform row -> ~0 cross-channel z);
    * frames 3000-3099 -- channel 0 alone at 20 uV: trips the across-channel
      z-score (one channel deviates) but NOT amplitude (20 < 50 uV).

    Under OR both regions are flagged, so the recording splits into three valid
    segments. Under AND neither would be flagged (each region is seen by only
    one detector), and a regression that broke EITHER branch would leave that
    branch's region covered. The previous version used an 80 uV uniform
    baseline that tripped amplitude on every frame, so the z-score branch
    contributed nothing and the test passed on amplitude alone.

    Background, channel count, z-score threshold, and proportion match
    ``test_detect_artifacts_zscore_only_detection`` (a passing test), so the
    uncorrelated background is known not to spuriously trip the z-score.
    """
    import numpy as np

    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )
    from spyglass.spikesorting.v2.artifact import ArtifactDetection

    rng = np.random.default_rng(42)
    n_samples, n_channels = 5000, 32
    traces = rng.normal(0.0, 0.5, size=(n_samples, n_channels)).astype(
        np.float32
    )
    # Amplitude-only region: uniform 100 uV (>50 uV amplitude; uniform across
    # channels -> ~0 cross-channel z, so the z-score detector ignores it).
    traces[1000:1100, :] = 100.0
    # Z-score-only region: channel 0 deviates to 20 uV (<50 uV amplitude, so
    # the amplitude detector ignores it; one deviating channel -> high
    # cross-channel z). The other channels stay at background.
    traces[3000:3100, 0] = 20.0
    rec = _build_synthetic_rec(traces)

    params = ArtifactDetectionParamsSchema(
        detect=True,
        amplitude_threshold_uv=50.0,
        zscore_threshold=4.0,
        proportion_above_threshold=1.0 / n_channels,  # any 1 channel
        removal_window_ms=0.05,
        join_window_ms=0.0,
        min_length_s=0.001,  # default 1.0 would wipe synthetic-recording intervals
    )
    valid_times = ArtifactDetection._detect_artifacts(rec, params)
    timestamps = rec.get_times()

    def _covered(t):
        return any(lo <= t <= hi for lo, hi in valid_times)

    # Under OR, BOTH single-detector regions are excised while a quiet
    # background frame stays valid. An AND combine would leave both regions
    # covered; a broken amplitude or z-score branch would leave its region
    # covered.
    assert not _covered(timestamps[1050]), (
        "amplitude-only region (frames 1000-1099) is still valid -- the "
        "amplitude branch did not contribute, or the combine became AND"
    )
    assert not _covered(timestamps[3050]), (
        "z-score-only region (frames 3000-3099) is still valid -- the "
        "z-score branch did not contribute, or the combine became AND"
    )
    assert _covered(timestamps[500]), (
        "a quiet background frame was flagged; detection is over-triggering"
    )
    assert valid_times.shape == (3, 2), (
        "expected 3 valid intervals around the two single-detector regions, "
        f"got shape {valid_times.shape}"
    )


def test_detect_artifacts_join_window_merges_runs(dj_conn):
    """``_detect_artifacts`` merges two artifact runs separated by
    fewer than ``join_window_frames`` into a single artifact span.

    Exercises the ``cur_end = f`` branch inside
    ``_detect_artifacts``'s join loop which the single-run test
    doesn't hit. Builds two transients separated by 10 frames;
    with join_window_ms set so join_window_frames > 10, the runs
    merge into one. With a smaller join window, the runs stay
    separate.
    """
    import numpy as np

    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )
    from spyglass.spikesorting.v2.artifact import ArtifactDetection

    n_samples, n_channels = 5000, 4
    traces = np.zeros((n_samples, n_channels), dtype=np.float32)
    traces[1000:1100, :] = 100.0  # artifact A
    traces[1110:1210, :] = 100.0  # artifact B; 10 frames after A
    rec = _build_synthetic_rec(traces)

    # join_window_ms = 1.0 -> join_window_frames = ceil(1 * 30 / 1) =
    # 30 frames. Gap between A and B is 10 frames, so they merge.
    merged = ArtifactDetection._detect_artifacts(
        rec,
        ArtifactDetectionParamsSchema(
            detect=True,
            amplitude_threshold_uv=50.0,
            zscore_threshold=None,
            proportion_above_threshold=0.5,
            removal_window_ms=0.05,
            join_window_ms=1.0,
            min_length_s=0.001,  # default 1.0 would wipe synthetic-recording intervals
        ),
    )
    assert merged.shape == (2, 2), (
        f"With join_window_ms=1.0 (= 30 frames > 10-frame gap), the two "
        f"transients should merge into one artifact span, leaving 2 "
        f"valid intervals; got shape {merged.shape}."
    )

    # join_window_ms = 0.1 -> join_window_frames = ceil(0.1 * 30 /
    # 1) = 3 frames. Gap of 10 stays unbridged; two separate
    # artifact runs, three valid intervals.
    #
    # ``min_length_s`` must be < the post-widening gap or the
    # sliver filter eats the middle interval. The middle sliver is
    # ``(10 frames gap) - 2*half_window_frames(=1) = 8 frames``
    # = ~0.27 ms at 30 kHz. Use 0.0001 s (0.1 ms) to keep it.
    unmerged = ArtifactDetection._detect_artifacts(
        rec,
        ArtifactDetectionParamsSchema(
            detect=True,
            amplitude_threshold_uv=50.0,
            zscore_threshold=None,
            proportion_above_threshold=0.5,
            removal_window_ms=0.05,
            join_window_ms=0.1,
            min_length_s=0.0001,
        ),
    )
    assert unmerged.shape == (3, 2), (
        f"With join_window_ms=0.1 (= 3 frames < 10-frame gap), the two "
        f"transients stay separate, leaving 3 valid intervals; got "
        f"shape {unmerged.shape}."
    )


@pytest.mark.slow
def test_artifact_detection_parameters_validates_via_insert1(dj_conn):
    """``ArtifactDetectionParameters.insert1`` validates the
    ``params`` blob through Pydantic before the row is written.

    Exercises lines 92-96 (the custom ``insert1`` override).
    A blob with an unknown key violates ``extra="forbid"`` and the
    insert is rejected before any row reaches the DB.
    """
    from pydantic import ValidationError

    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetectionParameters,
    )

    bad_row = {
        "artifact_detection_params_name": "params_validates_unit_test",
        "params": {
            "detect": True,
            "amplitude_threshold_uv": 100.0,
            "not_a_real_field": "x",  # rejected by extra="forbid"
        },
        "params_schema_version": 1,
        "job_kwargs": None,
    }
    with pytest.raises(ValidationError):
        ArtifactDetectionParameters().insert1(bad_row)

    assert (
        len(
            ArtifactDetectionParameters
            & {"artifact_detection_params_name": "params_validates_unit_test"}
        )
        == 0
    ), "Failed validation should not have written a row."


# ---------- _detect_artifacts cross-channel proportion boundary ----------


def test_detect_artifacts_below_proportion_threshold_ignored(dj_conn):
    """``_detect_artifacts`` does NOT flag artifact frames when
    fewer than ``proportion_above_threshold`` of channels exceed the
    amplitude threshold.

    Existing detection tests put the transient on every channel,
    so the proportion gate is never the binding constraint. This
    test puts a 200 uV transient on a SINGLE channel of a
    4-channel recording with ``proportion_above_threshold=0.5``
    (= ceil(0.5 * 4) = 2 channels required). Detection must not
    fire: the single-channel transient does not meet the cross-
    channel quorum.
    """
    import numpy as np

    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )
    from spyglass.spikesorting.v2.artifact import ArtifactDetection

    n_samples, n_channels = 5000, 4
    traces = np.zeros((n_samples, n_channels), dtype=np.float32)
    # ONE channel has a 200 uV transient at frames 1000-1099; the
    # other three channels stay at zero. 1 of 4 channels is below
    # the 2-of-4 requirement.
    traces[1000:1100, 0] = 200.0
    rec = _build_synthetic_rec(traces)

    params = ArtifactDetectionParamsSchema(
        detect=True,
        amplitude_threshold_uv=50.0,
        zscore_threshold=None,
        proportion_above_threshold=0.5,
        removal_window_ms=0.05,
        join_window_ms=0.0,
        min_length_s=0.001,  # default 1.0 would wipe synthetic-recording intervals
    )
    valid_times = ArtifactDetection._detect_artifacts(rec, params)
    timestamps = rec.get_times()

    assert valid_times.shape == (1, 2), (
        f"Single-channel transient should NOT trip cross-channel "
        f"proportion gate (1/4 < 0.5 required); expected one full "
        f"valid interval but got shape {valid_times.shape}."
    )
    assert valid_times[0][0] == pytest.approx(timestamps[0], abs=1e-9)
    assert valid_times[0][1] == pytest.approx(timestamps[-1], abs=1e-9)


# ---------- ArtifactDetection.delete ownership rows -----------------------


@pytest.mark.slow
def test_artifact_detection_delete_uses_interval_ownership_part_rows(
    populated_recording, monkeypatch
):
    """``ArtifactDetection.delete`` cleans via owned IntervalList part rows.

    Delete cleanup should not reconstruct artifact interval ownership from the
    source selector or interval naming convention. The generated
    ``ArtifactRemovedInterval`` rows are the contract. Patching source
    resolution to fail makes this test a regression pin for that boundary.
    """
    from spyglass.common import IntervalList
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetection,
        ArtifactDetectionParameters,
        ArtifactDetectionSelection,
    )

    ArtifactDetectionParameters.insert_default()
    art_pk = ArtifactDetectionSelection.insert_selection(
        {
            "recording_id": populated_recording["recording_id"],
            "artifact_detection_params_name": "none",
        }
    )
    ArtifactDetection.populate(art_pk, reserve_jobs=False)
    part_rows = (
        ArtifactDetection.ArtifactRemovedInterval & art_pk
    ).fetch("nwb_file_name", "interval_list_name", as_dict=True)
    assert part_rows, "ArtifactDetection.populate must insert ownership rows"

    def _unexpected_resolve(cls, key):
        raise RuntimeError("delete cleanup must not resolve source rows")

    monkeypatch.setattr(
        ArtifactDetectionSelection,
        "resolve_source",
        classmethod(_unexpected_resolve),
    )

    (ArtifactDetection & art_pk).delete(safemode=False)
    assert len(ArtifactDetection & art_pk) == 0
    assert len(IntervalList & part_rows) == 0


@pytest.mark.slow
def test_artifact_detection_delete_requires_interval_ownership_part_rows(
    populated_recording,
):
    """A row without ownership parts is invalid and must fail loudly.

    We intentionally do not reconstruct missing ownership from
    ``artifact_detection_id``-derived interval names. That would reintroduce
    the generic ``IntervalList.cleanup`` orphan problem this part table fixes.
    """
    from spyglass.common import IntervalList
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetection,
        ArtifactDetectionParameters,
        ArtifactDetectionSelection,
    )

    ArtifactDetectionParameters.insert_default()
    art_pk = ArtifactDetectionSelection.insert_selection(
        {
            "recording_id": populated_recording["recording_id"],
            "artifact_detection_params_name": "none",
        }
    )
    ArtifactDetection.populate(art_pk, reserve_jobs=False)
    part_rows = (
        ArtifactDetection.ArtifactRemovedInterval & art_pk
    ).fetch("nwb_file_name", "interval_list_name", as_dict=True)
    assert part_rows, "ArtifactDetection.populate must insert ownership rows"

    (ArtifactDetection.ArtifactRemovedInterval & art_pk).delete_quick()

    try:
        with pytest.raises(ValueError, match="no ArtifactRemovedInterval"):
            (ArtifactDetection & art_pk).delete(safemode=False)
        assert len(ArtifactDetection & art_pk) == 1
    finally:
        if ArtifactDetection & art_pk:
            (ArtifactDetection & art_pk).super_delete(warn=False)
        (IntervalList & part_rows).delete_quick()


# ---------- artifact at recording boundary boundary clamp ----------------


def test_detect_artifacts_clamps_artifact_at_recording_end(dj_conn):
    """``_detect_artifacts`` clamps the half-open end of an artifact
    that runs to the last sample.

    The clamp ``min(end_f + 1, len(timestamps) - 1)`` at
    ``artifact.py:973`` ensures the saved interval doesn't index
    out of bounds when the artifact reaches the recording end. No
    prior test puts a transient near the end of the recording, so
    this clamp branch was unexercised.
    """
    import numpy as np

    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )
    from spyglass.spikesorting.v2.artifact import ArtifactDetection

    n_samples, n_channels = 1000, 4
    traces = np.zeros((n_samples, n_channels), dtype=np.float32)
    # Transient at the very end of the recording.
    traces[990:1000, :] = 200.0
    rec = _build_synthetic_rec(traces)

    params = ArtifactDetectionParamsSchema(
        detect=True,
        amplitude_threshold_uv=50.0,
        zscore_threshold=None,
        proportion_above_threshold=0.5,
        removal_window_ms=0.05,
        join_window_ms=0.0,
        min_length_s=0.001,  # default 1.0 would wipe synthetic-recording intervals
    )
    valid_times = ArtifactDetection._detect_artifacts(rec, params)
    timestamps = rec.get_times()

    # The artifact runs from frame 990 to the end. The clamp
    # produces a single valid interval before the artifact (the
    # tail-valid branch in ``_detect_artifacts`` -- the
    # ``if cursor < valid_end`` guard -- fires only if there is
    # tail after the last artifact, which is false here because
    # the artifact reaches the end).
    assert valid_times.shape == (1, 2), (
        f"Expected one valid interval ending before the boundary "
        f"artifact, got shape {valid_times.shape}."
    )
    assert valid_times[0][0] == pytest.approx(timestamps[0], abs=1e-9)
    # The valid interval ends BEFORE the artifact span (boundary
    # not included).
    assert valid_times[0][1] < timestamps[990] + 1e-9, (
        f"Valid interval ends at {valid_times[0][1]}, after the "
        f"start of the boundary artifact ({timestamps[990]}). "
        "The boundary clamp may have an off-by-one regression."
    )


@pytest.mark.slow
@pytest.mark.integration
def test_shared_artifact_group_populate_end_to_end(
    populated_recording, polymer_smoke_session
):
    """End-to-end shared-group populate writes one IntervalList per
    member ``nwb_file_name``.

    Cross-recording shared-artifact path contract. With a single-
    recording smoke fixture the shared group has one member, so
    ``per_member_nwb_files`` has length 1 and one IntervalList row
    is written. The same code path scales to N members on a
    multi-recording session (out of scope for the smoke fixture).
    """
    from spyglass.common.common_interval import IntervalList
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetection,
        ArtifactDetectionParameters,
        ArtifactDetectionSelection,
        SharedArtifactGroup,
    )
    from spyglass.spikesorting.v2.utils import (
        artifact_detection_interval_list_name,
    )

    ArtifactDetectionParameters.insert_default()
    group_name = "v2_e2e_shared_group"

    # Clean up any prior group from earlier test runs.
    (
        SharedArtifactGroup & {"shared_artifact_group_name": group_name}
    ).super_delete(warn=False)

    SharedArtifactGroup.insert_group(
        group_name,
        [{"recording_id": populated_recording["recording_id"]}],
    )

    art_pk = ArtifactDetectionSelection.insert_selection(
        {
            "shared_artifact_group_name": group_name,
            "artifact_detection_params_name": "none",
        }
    )
    # ``none`` preset writes a full-window valid interval so the test
    # is amplitude-fixture-independent.
    ArtifactDetection.populate(art_pk, reserve_jobs=False)

    # Exactly one IntervalList row (one member), keyed by the
    # member's nwb_file_name with the canonical artifact-named
    # convention.
    interval_name = artifact_detection_interval_list_name(
        art_pk["artifact_detection_id"]
    )
    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    rows = (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": interval_name,
        }
    ).fetch(as_dict=True)
    assert len(rows) == 1
    assert rows[0]["pipeline"] == "spikesorting_artifact_detection_v2"
    assert (
        ArtifactDetection.ArtifactRemovedInterval
        & {
            "artifact_detection_id": art_pk["artifact_detection_id"],
            "nwb_file_name": nwb_file_name,
            "interval_list_name": interval_name,
        }
    ), "ArtifactDetection.populate did not insert the IntervalList ownership part row"

    # ``get_artifact_removed_intervals`` returns a dict keyed by
    # member nwb_file_name for shared-group sources. Today single
    # member -> single entry in the dict; the value array equals
    # the IntervalList row's valid_times.
    import numpy as np

    shared_intervals = ArtifactDetection().get_artifact_removed_intervals(
        art_pk
    )
    assert isinstance(shared_intervals, dict), (
        f"get_artifact_removed_intervals returned {type(shared_intervals)}; "
        "expected dict for shared-group source."
    )
    assert nwb_file_name in shared_intervals
    np.testing.assert_array_equal(
        shared_intervals[nwb_file_name], rows[0]["valid_times"]
    )

    # ``ArtifactDetection.delete()`` cleans up the matching
    # IntervalList rows for ALL member nwb_file_names, not just
    # the recording-source case.
    (ArtifactDetection & art_pk).delete(safemode=False)
    remaining = (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": interval_name,
        }
    ).fetch(as_dict=True)
    assert len(remaining) == 0, (
        "ArtifactDetection.delete() left an orphan IntervalList "
        "row for a shared-group artifact; the delete-cleanup "
        "branch for shared sources is broken."
    )


# ---------- error-handling context in warnings and reads -----------------


def test_artifact_empty_warning_has_context(dj_conn, monkeypatch):
    """The zero-artifact-frames warning names the artifact_detection_id and the
    active thresholds, not just a bare 'zero frames' message.
    """
    import numpy as np

    from spyglass.spikesorting.v2 import artifact as artifact_mod
    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )

    captured = []
    monkeypatch.setattr(
        artifact_mod.logger,
        "warning",
        lambda msg, *a, **k: captured.append(msg),
    )

    rec = _build_synthetic_rec(np.zeros((1000, 4), dtype=np.float32))
    validated = ArtifactDetectionParamsSchema(
        detect=True, amplitude_threshold_uv=500.0
    )
    out = artifact_mod.ArtifactDetection._detect_artifacts(
        rec, validated, context=" for artifact_detection_id=abc-123"
    )
    assert out.shape == (1, 2)  # full window returned on zero artifacts
    assert any("abc-123" in m for m in captured), captured
    assert any("amplitude_threshold_uv" in m for m in captured)


@pytest.mark.slow
def test_artifact_read_aborts_on_unexpected_resolve_error(
    populated_recording, monkeypatch
):
    """An unexpected source-resolution error during read propagates.

    Delete cleanup uses ``ArtifactRemovedInterval`` ownership rows and does
    not resolve the source anymore. Reading still resolves the source to
    choose the return shape, so unexpected resolver failures should remain
    visible to the caller.
    """
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetection,
        ArtifactDetectionParameters,
        ArtifactDetectionSelection,
    )

    ArtifactDetectionParameters.insert_default()
    art_pk = ArtifactDetectionSelection.insert_selection(
        {
            "recording_id": populated_recording["recording_id"],
            "artifact_detection_params_name": "default",
        }
    )
    ArtifactDetection.populate(art_pk, reserve_jobs=False)
    assert len(ArtifactDetection & art_pk) == 1

    def _boom(cls, key):
        raise RuntimeError("boom-unexpected-resolve")

    monkeypatch.setattr(
        ArtifactDetectionSelection, "resolve_source", classmethod(_boom)
    )
    with pytest.raises(RuntimeError, match="boom-unexpected-resolve"):
        ArtifactDetection().get_artifact_removed_intervals(art_pk)
    assert len(ArtifactDetection & art_pk) == 1


@pytest.mark.slow
@pytest.mark.integration
def test_shared_artifact_group_multi_member_union(
    populated_recording, polymer_smoke_session, monkeypatch
):
    """Two-member ``SharedArtifactGroup``: the union scan sees an artifact
    on ONE member's channels, and every member shares the written times.

    Every existing shared-group populate test has a single member, so the
    ``si.aggregate_channels`` union-scan branch of ``make_compute`` is
    untested. Here two time-aligned members are grouped; a supra-threshold
    transient is planted on member A's channels only, member B is clean.
    With ``proportion_above_threshold=0.4`` over the 8-channel union, A's 4
    channels alone exceed the required count, so the union scan removes the
    artifact window -- a per-member scan of the CLEAN member B would not.
    The single session writes one shared ``IntervalList`` row that every
    member resolves to, so all members see identical artifact-removed times.

    The two member recordings are loaded as synthetic SI recordings via a
    monkeypatched ``Recording.get_recording`` (planting a real artifact in
    fixture data is not otherwise possible); the DB rows / FK chain are
    real.
    """
    import spikeinterface as si

    from spyglass.common.common_interval import IntervalList
    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetection,
        ArtifactDetectionParameters,
        ArtifactDetectionSelection,
        SharedArtifactGroup,
    )
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
        SortGroupV2,
    )
    from spyglass.spikesorting.v2.utils import (
        artifact_detection_interval_list_name,
    )

    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    fs = 30000.0
    n_samples = 90000  # 3 s
    n_ch = 4
    art_lo, art_hi = 45000, 45050  # transient at 1.5 s

    def _synth_recording(with_artifact):
        traces = np.zeros((n_samples, n_ch), dtype=np.int16)
        if with_artifact:
            traces[art_lo:art_hi, :] = 5000
        rec = si.NumpyRecording([traces], sampling_frequency=fs)
        rec.set_channel_gains([1.0] * n_ch)  # µV == counts
        rec.set_channel_offsets([0.0] * n_ch)
        return rec

    # Second Recording row in the SAME session (a distinct recording_id to
    # group). Its loaded SI object is replaced below; only the DB row /
    # FK matters here.
    raw_times = (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": "raw data valid times",
        }
    ).fetch1("valid_times")
    t0 = float(raw_times[0][0])
    second_interval = "v2_shared_union_window"
    IntervalList.insert1(
        {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": second_interval,
            "valid_times": np.asarray([[t0, t0 + 2.5]]),
            "pipeline": "shared_group_union_test",
        },
        skip_duplicates=True,
    )
    sort_group_id = int(
        sorted((SortGroupV2 & polymer_smoke_session).fetch("sort_group_id"))[0]
    )
    second_rec_pk = RecordingSelection.insert_selection(
        {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": sort_group_id,
            "interval_list_name": second_interval,
            "preprocessing_params_name": "default",
            "team_name": "v2_test_team",
        }
    )
    Recording.populate(second_rec_pk, reserve_jobs=False)

    rid_a = populated_recording["recording_id"]  # member A: artifact
    rid_b = second_rec_pk["recording_id"]  # member B: clean

    def _fake_get_recording(self, key):
        return _synth_recording(
            with_artifact=str(key["recording_id"]) == str(rid_a)
        )

    monkeypatch.setattr(Recording, "get_recording", _fake_get_recording)

    # Custom param: detect with proportion 0.4 so 4 of 8 union channels
    # (member A's) suffice to flag a frame; min_length 0.1 s so both
    # post-split halves survive.
    params_name = "multi_member_union_test"
    ArtifactDetectionParameters.insert1(
        {
            "artifact_detection_params_name": params_name,
            "params": ArtifactDetectionParamsSchema(
                detect=True,
                amplitude_threshold_uv=1000.0,
                zscore_threshold=None,
                proportion_above_threshold=0.4,
                removal_window_ms=1.0,
                join_window_ms=1.0,
                min_length_s=0.1,
            ).model_dump(),
            "params_schema_version": 2,
        },
        skip_duplicates=True,
    )

    group_name = "v2_multi_member_union_group"

    def _clear_shared_group():
        # Master-first cleanup: drop any ArtifactDetection +
        # ArtifactDetectionSelection whose source part references this group BEFORE
        # the group itself. An interrupted prior run can leave a
        # SharedGroupSource part whose master must go first;
        # deleting the group ahead of it would orphan/block on that part.
        stale_art_ids = (
            ArtifactDetectionSelection.SharedGroupSource
            & {"shared_artifact_group_name": group_name}
        ).fetch("artifact_detection_id")
        for aid in stale_art_ids:
            (ArtifactDetection & {"artifact_detection_id": aid}).delete(
                safemode=False
            )
            (
                ArtifactDetectionSelection & {"artifact_detection_id": aid}
            ).super_delete(warn=False)
        (
            SharedArtifactGroup & {"shared_artifact_group_name": group_name}
        ).super_delete(warn=False)

    _clear_shared_group()
    try:
        SharedArtifactGroup.insert_group(
            group_name,
            [{"recording_id": rid_a}, {"recording_id": rid_b}],
        )
        assert (
            len(
                SharedArtifactGroup.Member
                & {"shared_artifact_group_name": group_name}
            )
            == 2
        )

        art_pk = ArtifactDetectionSelection.insert_selection(
            {
                "shared_artifact_group_name": group_name,
                "artifact_detection_params_name": params_name,
            }
        )
        ArtifactDetection.populate(art_pk, reserve_jobs=False)

        interval_name = artifact_detection_interval_list_name(
            art_pk["artifact_detection_id"]
        )
        rows = (
            IntervalList
            & {
                "nwb_file_name": nwb_file_name,
                "interval_list_name": interval_name,
            }
        ).fetch(as_dict=True)
        # Single session -> one shared row covering every member.
        assert (
            len(rows) == 1
        ), f"expected one shared IntervalList row; got {len(rows)}"
        valid_times = rows[0]["valid_times"]

        # The union scan removed member A's artifact: the single base chunk
        # is split into exactly two valid intervals around the 1.5 s
        # transient, and no valid interval contains it. A clean-only
        # (member B) scan would leave one full-window interval instead.
        art_mid = 0.5 * (art_lo + art_hi) / fs  # ~1.5 s
        assert valid_times.shape == (2, 2), (
            f"union scan should split into 2 intervals at the artifact; got "
            f"valid_times={valid_times.tolist()} (shape {valid_times.shape}). "
            "A per-member scan of the clean member would leave a single "
            "full-window interval -- the union channels were not combined."
        )
        for start, end in valid_times:
            assert not (start < art_mid < end), (
                f"a valid interval [{start}, {end}] spans the artifact at "
                f"{art_mid}s; the union scan did not remove it."
            )
        # Coverage survives on both sides of the artifact.
        assert any(s <= 0.5 <= e for s, e in valid_times), "pre-artifact lost"
        assert any(s <= 2.4 <= e for s, e in valid_times), "post-artifact lost"

        # Every member resolves to the SAME artifact-removed times (single
        # shared row; the per-member dict has one session entry equal to it).
        shared = ArtifactDetection().get_artifact_removed_intervals(art_pk)
        assert isinstance(shared, dict) and nwb_file_name in shared
        np.testing.assert_array_equal(shared[nwb_file_name], valid_times)
    finally:
        _clear_shared_group()

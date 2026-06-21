"""Recording.make and RecordingSelection behavior tests for the v2 single-session pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.spikesorting.v2._ingest_helpers import _clean_session_v2


# ---------- RecordingSelection.insert_selection ---------------------------


@pytest.mark.slow
def test_recording_selection_insert_is_idempotent(polymer_smoke_session):
    """Repeat calls return the same PK and never duplicate rows."""
    from spyglass.spikesorting.v2.recording import (
        PreprocessingParameters,
        RecordingSelection,
        SortGroupV2,
    )
    from spyglass.common.common_lab import LabTeam

    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    _clean_session_v2(polymer_smoke_session)
    SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
    PreprocessingParameters.insert_default()

    # Pick the first sort group + a session interval the fixture provides.
    sort_group_id = (SortGroupV2 & polymer_smoke_session).fetch(
        "sort_group_id"
    )[0]
    interval_list_name = (
        "raw data valid times"  # canonical name written by insert_sessions
    )

    # Ensure a LabTeam exists (the round-trip ingestion does not create one).
    team_name = "v2_test_team"
    LabTeam.insert1(
        {"team_name": team_name, "team_description": "v2 pipeline tests"},
        skip_duplicates=True,
    )

    selection_key = {
        "nwb_file_name": nwb_file_name,
        "sort_group_id": int(sort_group_id),
        "interval_list_name": interval_list_name,
        "preprocessing_params_name": "default",
        "team_name": team_name,
    }
    pk_first = RecordingSelection.insert_selection(selection_key)
    pk_second = RecordingSelection.insert_selection(selection_key)
    assert pk_first == pk_second
    assert isinstance(pk_first, dict)
    assert list(pk_first.keys()) == ["recording_id"]
    # And only one row in the table.
    assert len(RecordingSelection & selection_key) == 1


# ---------- Recording.make + get_recording --------------------------------


@pytest.mark.slow
def test_recording_populates_and_round_trips(
    polymer_smoke_session, recording_selection_key
):
    """``Recording.make`` writes a preprocessed NWB; ``get_recording``
    reads it back with the expected channel count, sampling rate, and
    duration."""
    from spyglass.spikesorting.v2.recording import (
        Recording,
        SortGroupV2,
    )

    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    sort_group_id = int(
        sorted((SortGroupV2 & polymer_smoke_session).fetch("sort_group_id"))[0]
    )
    expected_n_channels = len(
        SortGroupV2.SortGroupElectrode
        & {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": sort_group_id,
        }
    )

    # Clear any prior Recording row for this selection (subsequent module
    # runs would otherwise short-circuit populate()). ``force_masters=True``
    # mirrors what ``cautious_delete`` passes to DataJoint: it deletes the
    # source-polymorphic ``ArtifactDetectionSelection`` / ``SortingSelection``
    # MASTER when the Recording cascade reaches its ``RecordingSource``
    # part (the part FKs Recording but the master does not, so a bare
    # ``super_delete`` trips DataJoint's part-before-master rule whenever
    # an earlier test attached selections to this recording_id).
    (Recording & recording_selection_key).super_delete(
        warn=False, force_masters=True
    )
    Recording.populate(recording_selection_key, reserve_jobs=False)
    row = (Recording & recording_selection_key).fetch1()

    assert row["n_channels"] == expected_n_channels
    assert row["sampling_frequency"] == pytest.approx(30_000, rel=1e-3)
    # The smoke fixture is 4 seconds of polymer data. Compare to the
    # IntervalList's actual valid_times span so a regression that
    # silently truncated to e.g. 0.1 s would surface (rather than
    # passing via ``> 0.0``).
    from spyglass.common.common_interval import IntervalList

    raw_times = (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": "raw data valid times",
        }
    ).fetch1("valid_times")
    expected_duration = float(raw_times[-1][-1] - raw_times[0][0])
    assert row["duration_s"] == pytest.approx(expected_duration, rel=1e-3), (
        f"Recording.duration_s = {row['duration_s']}; expected "
        f"~{expected_duration} from IntervalList valid_times span."
    )
    # ``cache_hash`` is an ``NwbfileHasher`` digest. The exact digest
    # length (currently 32-char MD5 hex) is intentionally not pinned; the
    # ``char(64)`` schema column is headroom for a future hash change.
    # Assert non-empty rather than a fixed length so a hash-algorithm swap
    # does not break this test.
    assert isinstance(row["cache_hash"], str) and row["cache_hash"]

    rec = Recording().get_recording(recording_selection_key)
    assert rec.get_num_channels() == expected_n_channels
    assert rec.get_sampling_frequency() == pytest.approx(
        row["sampling_frequency"], rel=1e-6
    )


@pytest.mark.slow
def test_tetrode_geometry_attached(tetrode_60s_session):
    """A 4-channel ``tetrode_12.5`` sort group gets the explicit tetrode
    probe geometry, and it survives the NWB round-trip.

    Exercises the all-true branch of ``_maybe_apply_tetrode_geometry``
    (single ``tetrode_12.5`` probe, exactly 4 channels, single
    electrode group) end-to-end: a geometry-aware sorter calling
    ``get_recording`` must see the 4 contacts at the
    ``(0,0)-(0,12.5)-(12.5,0)-(12.5,12.5)`` µm square the patch
    installs. (The four negative-condition tests for the same gate are
    covered separately; this test owns the all-true happy path.)
    """
    from spyglass.spikesorting.v2.recording import (
        PreprocessingParameters,
        Recording,
        RecordingSelection,
        SortGroupV2,
    )

    nwb_file_name = tetrode_60s_session["nwb_file_name"]
    from spyglass.common.common_lab import LabTeam

    PreprocessingParameters.insert_default()
    LabTeam.insert1(
        {"team_name": "v2_test_team", "team_description": "v2 pipeline tests"},
        skip_duplicates=True,
    )
    if not (SortGroupV2 & tetrode_60s_session):
        SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
    sort_group_id = int(
        sorted((SortGroupV2 & tetrode_60s_session).fetch("sort_group_id"))[0]
    )
    # Precondition: the gate's all-true branch requires exactly 4
    # channels in this single-tetrode sort group.
    n_channels = len(
        SortGroupV2.SortGroupElectrode
        & {"nwb_file_name": nwb_file_name, "sort_group_id": sort_group_id}
    )
    assert n_channels == 4, (
        "tetrode fixture sort group must have 4 channels for the "
        f"geometry gate; got {n_channels}."
    )

    selection_key = {
        "nwb_file_name": nwb_file_name,
        "sort_group_id": sort_group_id,
        "interval_list_name": "raw data valid times",
        "preprocessing_params_name": "default",
        "team_name": "v2_test_team",
    }
    rec_pk = RecordingSelection.insert_selection(selection_key)
    # Clear any prior Recording row so populate re-materializes.
    (Recording & rec_pk).super_delete(warn=False, force_masters=True)
    Recording.populate(rec_pk, reserve_jobs=False)

    rec = Recording().get_recording(rec_pk)
    assert rec.get_num_channels() == 4

    # The source electrode table for this legacy-style fixture carries no
    # contact positions (all (0,0,0)); without the geometry patch all four
    # contacts would collapse onto a single point and a geometry-aware
    # sorter would be blind to the tetrode layout. Assert the patch
    # installed a 4-corner 12.5 µm square: two distinct x values 12.5 µm
    # apart, two distinct y values 12.5 µm apart, all four corners
    # present. (The patch positions the square at (0,0)-(12.5,12.5), but
    # probeinterface re-centers contacts on their centroid, so the
    # absolute origin shifts to +/-6.25; the 12.5 µm geometry -- the only
    # thing a sorter consumes -- is what we pin, not the origin.)
    locs = [
        (round(float(x), 3), round(float(y), 3))
        for x, y in rec.get_channel_locations()
    ]
    xs = sorted({x for x, _ in locs})
    ys = sorted({y for _, y in locs})
    assert len(xs) == 2 and len(ys) == 2, (
        f"tetrode contacts collapsed -- expected a 2x2 grid, got "
        f"x={xs}, y={ys} (patch did not spread the degenerate "
        "all-zero source positions)."
    )
    assert round(xs[1] - xs[0], 3) == 12.5
    assert round(ys[1] - ys[0], 3) == 12.5
    assert set(locs) == {
        (xs[0], ys[0]),
        (xs[0], ys[1]),
        (xs[1], ys[0]),
        (xs[1], ys[1]),
    }, f"tetrode contacts {sorted(locs)} do not form a 12.5 µm square."


@pytest.mark.slow
def test_write_nwb_artifact_rejects_heterogeneous_gains(
    polymer_smoke_session,
):
    """``Recording._write_nwb_artifact`` raises on heterogeneous channel
    gains instead of silently scaling every channel by ``gains[0]``.

    Exercises the gain gate behaviorally (a recording whose channels
    carry two distinct gains must abort the write), replacing the prior
    source-comment-presence check. The aborted write unlinks its stub
    analysis file, so no orphan outlives the call.
    """
    import numpy as np
    import spikeinterface.core as si_core

    from spyglass.spikesorting.v2.recording import Recording

    rng = np.random.default_rng(0)
    traces = rng.standard_normal((1000, 2)).astype("float32")
    rec = si_core.NumpyRecording([traces], sampling_frequency=30_000.0)
    # Two distinct channel gains -> the single-conversion-factor invariant
    # is violated and the write must raise.
    rec.set_channel_gains([1.0, 2.0])

    with pytest.raises(ValueError, match="heterogeneous channel gains"):
        Recording._write_nwb_artifact(
            rec,
            polymer_smoke_session["nwb_file_name"],
            filtering_description="test",
        )


@pytest.mark.slow
def test_get_recording_currently_raises_checksum_on_missing_cache(
    populated_recording,
):
    """``get_recording`` raises a checksum ``DataJointError`` when the cache
    file is missing and must be rebuilt.

    Pins TODAY's behavior: the on-demand rebuild is NOT byte-deterministic
    versus the external-store checksum (a future byte-deterministic
    ``RecordingArtifactRecompute`` path is not yet implemented). So when the
    canonical cache is absent, ``get_recording`` rebuilds the file, then
    re-resolves it with checksum validation, and DataJoint rejects the
    rebuilt-but-not-identical bytes. This test locks that failure mode: a
    future regression that silently returned the drifted rebuilt bytes
    (skipping the checksum) would flip this assertion and be caught.

    The happy-path "rebuild succeeds, hash matches" case cannot pass against
    current source (it needs the byte-deterministic recompute path) and is
    therefore intentionally not asserted here.
    """
    import shutil

    from datajoint.errors import DataJointError

    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2.recording import Recording

    row = (Recording & populated_recording).fetch1()
    cache_hash = row["cache_hash"]
    assert cache_hash, "populated_recording row must carry a cache_hash"
    analysis_file_name = row["analysis_file_name"]

    # Resolve via from_schema=True (skips checksum) to find the file we are
    # about to remove; this is the same physical file get_recording's
    # existence check (default resolution) will find absent.
    abs_path = AnalysisNwbfile.get_abs_path(
        analysis_file_name, from_schema=True
    )
    assert Path(abs_path).exists()
    # Back up the valid cache: the rebuild overwrites it in place with
    # non-identical bytes, so restore the original afterward or later tests
    # reusing this shared row would themselves fail the checksum read.
    backup = str(abs_path) + ".missing_cache_rebuild.bak"
    shutil.copy2(abs_path, backup)

    try:
        Path(abs_path).unlink()
        with pytest.raises(DataJointError) as excinfo:
            Recording().get_recording(populated_recording)
        msg = str(excinfo.value)
        assert "checksum" in msg.lower(), (
            "expected a checksum DataJointError from the rebuilt-file "
            f"re-read; got: {msg!r}"
        )
        assert analysis_file_name in msg, (
            "checksum error should name the analysis_file_name "
            f"{analysis_file_name!r}; got: {msg!r}"
        )
    finally:
        shutil.move(backup, abs_path)


@pytest.mark.slow
def test_rebuild_nwb_artifact_reaches_hash_then_raises_checksum(
    populated_recording, monkeypatch
):
    """``_rebuild_nwb_artifact`` runs the rebuild pipeline far enough to call
    ``_hash_nwb_recording`` on the rebuilt artifact, then raises a checksum
    ``DataJointError``.

    A naive expectation is that the direct rebuild "runs to completion" with
    only ``get_recording``'s reader rejecting the bytes, but the actual
    behavior is stronger: ``_write_nwb_artifact`` finishes its write and then
    calls ``_hash_nwb_recording`` to record the rebuilt ``cache_hash`` -- and
    ``_hash_nwb_recording`` reads the file through the checksum-validated
    ``get_abs_path``, which rejects the rebuilt-but-not-byte-identical file.
    So the rebuild raises a checksum error from INSIDE its own hashing step,
    not only at ``get_recording``'s re-read. (Confirmed by the call chain
    ``_rebuild_nwb_artifact`` -> ``_compute_recording_artifact`` ->
    ``_write_nwb_artifact`` -> ``_hash_nwb_recording`` -> ``get_abs_path`` ->
    DataJoint external checksum.)

    This pins TWO facts a future regression must not break silently:
    (1) the rebuild machinery is exercised up to and including the
    ``_hash_nwb_recording`` verification call (counter asserts it is reached),
    and (2) the rebuild is non-functional versus the external-store checksum
    today (it raises rather than persisting drifted bytes). When a future
    byte-deterministic ``RecordingArtifactRecompute`` path makes the rebuild
    byte-deterministic (or reconciles the checksum), this test will flip to
    asserting the happy path. The point of this test is that the
    current-failure path must exercise the raising behavior -- this one does.
    """
    import shutil

    from datajoint.errors import DataJointError

    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2 import utils as v2_utils
    from spyglass.spikesorting.v2.recording import Recording

    row = (Recording & populated_recording).fetch1()
    abs_path = AnalysisNwbfile.get_abs_path(
        row["analysis_file_name"], from_schema=True
    )
    # The rebuild overwrites the canonical file in place before it raises;
    # back it up so later tests reusing this shared row read the original
    # (checksum-valid) bytes.
    backup = str(abs_path) + ".rebuild_hash_checksum.bak"
    shutil.copy2(abs_path, backup)

    calls = {"n": 0}
    real_hash = v2_utils._hash_nwb_recording

    def _counting_hash(*args, **kwargs):
        calls["n"] += 1
        return real_hash(*args, **kwargs)

    # _write_nwb_artifact imports _hash_nwb_recording from this module at call
    # time, so patching the source attribute is observed.
    monkeypatch.setattr(v2_utils, "_hash_nwb_recording", _counting_hash)

    try:
        with pytest.raises(DataJointError) as excinfo:
            Recording()._rebuild_nwb_artifact(populated_recording)
        assert "checksum" in str(excinfo.value).lower(), (
            "expected the rebuild's hash step to raise a checksum "
            f"DataJointError; got: {str(excinfo.value)!r}"
        )
        assert calls["n"] >= 1, (
            "_rebuild_nwb_artifact raised before reaching _hash_nwb_recording; "
            "the rebuild's cache_hash verification path was not exercised, so "
            "this does not pin the intended (rebuild-reached-hashing) state."
        )
    finally:
        shutil.move(backup, abs_path)


@pytest.mark.slow
def test_recording_over_request_clips_with_warning(
    polymer_smoke_session, caplog
):
    """A sort interval whose valid_times run PAST the raw recording is clipped
    to raw coverage and a warning is emitted -- it is NOT a hard error (lab
    decision: clip-with-warning, not raise; #1133/#1585). The clip must be
    surfaced (not silent), the Recording row IS inserted, and its saved
    duration reflects the clipped raw coverage, not the over-request.

    Genuine truncation (saved < the raw-clipped expectation, i.e. dropped
    packets) still raises ``RecordingTruncatedError`` in ``make_insert``; that
    path is distinct from this over-request clip.
    """
    import numpy as np

    from spyglass.common import IntervalList
    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2.recording import (
        PreprocessingParameters,
        Recording,
        RecordingSelection,
        SortGroupV2,
    )

    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    PreprocessingParameters.insert_default()
    LabTeam.insert1(
        {
            "team_name": "v2_test_team",
            "team_description": "v2 pipeline tests",
        },
        skip_duplicates=True,
    )
    if not (SortGroupV2 & polymer_smoke_session):
        SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)

    # Write a synthetic IntervalList row whose end-time exceeds the raw
    # recording's last timestamp by 1.0 second -- well past the 1.5 / fs
    # tolerance that absorbs the off-by-one boundary.
    over_request_interval_name = "v2_test_over_request"
    raw_valid_times = (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": "raw data valid times",
        }
    ).fetch1("valid_times")
    raw_end = float(raw_valid_times[-1][-1])
    raw_coverage = float(sum(end - start for start, end in raw_valid_times))
    IntervalList.insert1(
        {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": over_request_interval_name,
            "valid_times": np.asarray([[0.0, raw_end + 1.0]]),
            "pipeline": "v2_test_over_request",
        },
        skip_duplicates=True,
    )

    sort_group_id = int(
        sorted((SortGroupV2 & polymer_smoke_session).fetch("sort_group_id"))[0]
    )
    selection_key = {
        "nwb_file_name": nwb_file_name,
        "sort_group_id": sort_group_id,
        "interval_list_name": over_request_interval_name,
        "preprocessing_params_name": "default",
        "team_name": "v2_test_team",
    }
    pk = RecordingSelection.insert_selection(selection_key)

    # Clear any prior Recording row so populate actually runs make().
    (Recording & pk).super_delete(warn=False, force_masters=True)
    try:
        with caplog.at_level("WARNING"):
            Recording.populate(pk, reserve_jobs=False)
        # Over-request is clipped, not rejected: the row IS inserted.
        assert (
            Recording & pk
        ), "an over-request should be clipped and inserted, not rejected"
        # The clip is surfaced via a warning -- not silent.
        assert any(
            "clipping to raw coverage" in record.getMessage()
            for record in caplog.records
        ), "over-request clip should emit a warning"
        # Saved duration equals the raw coverage (clipped), not the +1 s
        # over-request the IntervalList asked for.
        duration_s = float((Recording & pk).fetch1("duration_s"))
        assert abs(duration_s - raw_coverage) < 0.1, (
            f"saved duration {duration_s:.3f}s should be clipped to raw "
            f"coverage {raw_coverage:.3f}s, not the over-request"
        )
    finally:
        (Recording & pk).super_delete(warn=False, force_masters=True)
        (RecordingSelection & pk).super_delete(warn=False)
        (
            IntervalList
            & {
                "nwb_file_name": nwb_file_name,
                "interval_list_name": over_request_interval_name,
            }
        ).super_delete(warn=False)


@pytest.mark.slow
def test_recording_over_request_multi_interval_clips_with_warning(
    polymer_smoke_session, caplog
):
    """The disjoint multi-interval analogue of
    ``test_recording_over_request_clips_with_warning``: two sort intervals with
    a real inter-segment gap, where the final segment runs 1 s PAST the raw
    recording. The over-request is clipped (not an error) and a warning is
    emitted; the row is inserted with the gap- and over-request-excluded
    duration.

    The concatenated recording is built with ``ignore_times=True``, so its
    ``get_times()`` is 0-based and the saved duration is the sum of kept-frame
    durations -- it EXCLUDES the inter-segment gap. The over-request detection
    likewise excludes gaps (both sides do), so the warning fires on the ~1 s
    past-coverage span, not on the gap.
    """
    import numpy as np

    from spyglass.common import IntervalList
    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2.recording import (
        PreprocessingParameters,
        Recording,
        RecordingSelection,
        SortGroupV2,
    )

    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    PreprocessingParameters.insert_default()
    LabTeam.insert1(
        {"team_name": "v2_test_team", "team_description": "v2 pipeline tests"},
        skip_duplicates=True,
    )
    if not (SortGroupV2 & polymer_smoke_session):
        SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)

    raw_times = (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": "raw data valid times",
        }
    ).fetch1("valid_times")
    raw_start = float(raw_times[0][0])
    raw_end = float(raw_times[-1][-1])
    # First segment [raw_start, +1.5 s]; 1 s gap; second segment from
    # +2.5 s extends 1 s PAST raw_end (the over-request). After clipping
    # to raw_end the second segment is still > 1 s, so it survives the
    # min_segment_length filter and both segments take the concat path.
    assert raw_end - raw_start >= 3.5, (
        f"fixture too short for multi-interval over-request test: raw "
        f"span [{raw_start}, {raw_end}] needs >= 3.5 s"
    )
    seg1_end = raw_start + 1.5
    seg2_start = seg1_end + 1.0
    over_request_s = 1.0
    multi_interval_name = "v2_test_multi_over_request"
    IntervalList.insert1(
        {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": multi_interval_name,
            "valid_times": np.asarray(
                [[raw_start, seg1_end], [seg2_start, raw_end + over_request_s]]
            ),
            "pipeline": "v2_test_over_request",
        },
        skip_duplicates=True,
    )

    sort_group_id = int(
        sorted((SortGroupV2 & polymer_smoke_session).fetch("sort_group_id"))[0]
    )
    selection_key = {
        "nwb_file_name": nwb_file_name,
        "sort_group_id": sort_group_id,
        "interval_list_name": multi_interval_name,
        "preprocessing_params_name": "default",
        "team_name": "v2_test_team",
    }
    pk = RecordingSelection.insert_selection(selection_key)

    (Recording & pk).super_delete(warn=False, force_masters=True)
    try:
        with caplog.at_level("WARNING"):
            Recording.populate(pk, reserve_jobs=False)
        # Over-request is clipped, not rejected: the row IS inserted.
        assert (
            Recording & pk
        ), "a multi-interval over-request should be clipped and inserted"
        # The clip is surfaced via a warning -- not silent.
        assert any(
            "clipping to raw coverage" in record.getMessage()
            for record in caplog.records
        ), "over-request clip should emit a warning"
        # Saved duration = seg1 + (seg2 clipped to raw_end), with the
        # inter-segment gap excluded and the +1 s over-request dropped.
        expected_clipped = (seg1_end - raw_start) + (raw_end - seg2_start)
        duration_s = float((Recording & pk).fetch1("duration_s"))
        assert abs(duration_s - expected_clipped) < 0.15, (
            f"saved duration {duration_s:.3f}s should be the clipped, "
            f"gap-excluded total {expected_clipped:.3f}s"
        )
    finally:
        (Recording & pk).super_delete(warn=False, force_masters=True)
        (RecordingSelection & pk).super_delete(warn=False)
        (
            IntervalList
            & {
                "nwb_file_name": nwb_file_name,
                "interval_list_name": multi_interval_name,
            }
        ).super_delete(warn=False)


@pytest.mark.slow
def test_fetch_sort_group_probe_info_stable_order(
    polymer_smoke_session, monkeypatch
):
    """``_fetch_sort_group_probe_info`` returns a deterministic,
    electrode_id-ordered result.

    The tri-part populate contract hashes the ``RecordingFetched`` tuple
    twice and raises if the two fetches differ; an unordered DB fetch can
    reorder rows between calls. The fetch must therefore pin
    ``order_by="electrode_id"``. This guards that invariant directly by
    spying on DataJoint's fetch call, so dropping the kwarg fails even if
    the database happens to return primary-key/electrode order by default.
    """
    from datajoint.fetch import Fetch

    from spyglass.common.common_device import Probe
    from spyglass.common.common_ephys import Electrode
    from spyglass.spikesorting.v2.recording import Recording, SortGroupV2

    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    if not (SortGroupV2 & polymer_smoke_session):
        SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)

    sort_group_id = int(
        sorted((SortGroupV2 & polymer_smoke_session).fetch("sort_group_id"))[0]
    )
    channel_ids = sorted(
        (
            SortGroupV2.SortGroupElectrode
            & {
                "nwb_file_name": nwb_file_name,
                "sort_group_id": sort_group_id,
            }
        ).fetch("electrode_id"),
        key=int,
    )
    assert len(channel_ids) > 1, "need a multi-channel sort group"

    fetch_order_by = []
    original_fetch_call = Fetch.__call__

    def _spy_fetch(self, *attrs, **kwargs):
        # Record only the explicitly-ordered fetches. A single
        # ``.fetch(named_attrs, order_by=...)`` re-enters ``Fetch.__call__``
        # internally in DataJoint 0.14.x (once more with ``attrs=()``,
        # carrying the same ``order_by``), so the raw call count is an
        # implementation detail. What matters is the INVARIANT: every
        # ordered probe-info fetch is keyed on ``electrode_id`` (never
        # left to unspecified DB order), which is what keeps the tri-part
        # DeepHash stable. Asserting an exact call count would over-fit
        # to DataJoint's internal recursion.
        ob = kwargs.get("order_by")
        if ob is not None:
            fetch_order_by.append(ob)
        return original_fetch_call(self, *attrs, **kwargs)

    monkeypatch.setattr(Fetch, "__call__", _spy_fetch)

    first = Recording._fetch_sort_group_probe_info(nwb_file_name, channel_ids)
    second = Recording._fetch_sort_group_probe_info(nwb_file_name, channel_ids)
    assert fetch_order_by, "probe-info fetch issued no explicitly-ordered fetch"
    assert all(ob == "electrode_id" for ob in fetch_order_by), (
        "probe-info fetch issued a non-electrode_id ordered fetch; the "
        f"order_by values seen were {fetch_order_by}. Every ordered fetch "
        "must be electrode_id-keyed for the tri-part DeepHash to stay stable."
    )
    assert first == second, "probe-info fetch is not deterministic"

    ref_rows = (
        Electrode * Probe
        & {"nwb_file_name": nwb_file_name}
        & [{"electrode_id": int(c)} for c in channel_ids]
    ).fetch(
        "probe_type",
        "electrode_group_name",
        as_dict=True,
        order_by="electrode_id",
    )
    expected = (
        tuple(r["probe_type"] for r in ref_rows),
        tuple(r["electrode_group_name"] for r in ref_rows),
    )
    assert first == expected, "probe-info fetch is not electrode_id-ordered"
    assert len(first[0]) == len(channel_ids)


# ---------- Persisted-timestamp readback regression ----------------------


@pytest.mark.slow
def test_recording_timestamps_reads_persisted_vector(tmp_path, dj_conn):
    """``Sorting._recording_timestamps`` returns the upstream Recording's
    actual persisted timestamp vector (first value non-zero), not a
    hardcoded/affine grid from 0.0.

    ``get_sorting`` maps absolute spike times back to frames via
    ``np.searchsorted`` against this vector, so a wrong t_start (the
    old v1-era 0.0 hardcode) or an affine reconstruction would
    mis-map every frame. The smoke and 60s polymer fixtures both start
    at t=0, which hid the bug; this writes a synthetic NWB with
    ``timestamps[0] = 12345.6`` and pins that the full vector comes
    back via the ``electrical_series_path`` indirection. Stays out of
    the DB by monkeypatching ``AnalysisNwbfile.get_abs_path``.
    """
    import numpy as np
    import pynwb
    from datetime import datetime
    from unittest.mock import patch

    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2.sorting import Sorting

    fs = 30_000.0
    t0 = 12345.6
    n_samples = 100
    timestamps = t0 + np.arange(n_samples) / fs
    data = np.zeros((n_samples, 4), dtype=np.int16)

    nwbf = pynwb.NWBFile(
        session_description="t_start regression",
        identifier="t_start_regression",
        session_start_time=datetime.now().astimezone(),
    )
    device = nwbf.create_device(name="probe")
    eg = nwbf.create_electrode_group(
        name="grp", description="grp", location="hpc", device=device
    )
    for ch in range(4):
        nwbf.add_electrode(location="hpc", group=eg, group_name="grp")
    region = nwbf.create_electrode_table_region(
        region=list(range(4)), description="all four"
    )
    series = pynwb.ecephys.ElectricalSeries(
        name="ProcessedElectricalSeries",
        data=data,
        electrodes=region,
        timestamps=timestamps,
        filtering="bandpass",
        description="synthetic for t_start regression",
        conversion=1e-6,
    )
    nwbf.add_acquisition(series)

    nwb_path = tmp_path / "synthetic_t_start.nwb"
    with pynwb.NWBHDF5IO(path=str(nwb_path), mode="w") as io:
        io.write(nwbf)

    fake_row = {
        "analysis_file_name": "synthetic_t_start.nwb",
        "electrical_series_path": "acquisition/ProcessedElectricalSeries",
    }
    with patch.object(
        AnalysisNwbfile, "get_abs_path", return_value=str(nwb_path)
    ):
        recovered = Sorting._recording_timestamps(fake_row)

    assert len(recovered) == n_samples, (
        f"_recording_timestamps returned {len(recovered)} samples; "
        f"expected the full {n_samples}-sample vector."
    )
    assert recovered[0] == pytest.approx(t0, abs=1e-6), (
        f"_recording_timestamps[0] returned {recovered[0]}; expected "
        f"{t0}. The v1-era t_start=0 hardcode would silently pass on "
        "t=0 fixtures but break on real session recordings that start "
        "mid-day."
    )
    assert np.allclose(recovered, timestamps), (
        "_recording_timestamps must return the persisted timestamp "
        "vector verbatim (searchsorted readback depends on it)."
    )


# ---------- Heterogeneous gain rejection ---------------------------------


@pytest.mark.slow
def test_recording_make_rejects_heterogeneous_gains(populated_recording):
    """``Recording._write_nwb_artifact`` raises ValueError when the
    SI recording reports unequal channel gains across the sort group.

    A divergence here would indicate a probe-metadata bug upstream
    (different conversion factors per channel can't be represented
    by the single-conversion ElectricalSeries). The check is in
    place at recording.py but had no test; this regression test
    builds a synthetic NumpyRecording with mismatched gains and
    pins the raise.
    """
    import numpy as np
    import spikeinterface as si

    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
    )

    nwb_file_name = (
        RecordingSelection
        & {"recording_id": populated_recording["recording_id"]}
    ).fetch1("nwb_file_name")

    n_channels = 4
    n_samples = 1000
    fs = 30_000.0
    traces = np.zeros((n_samples, n_channels), dtype=np.float32)
    synthetic_rec = si.NumpyRecording(
        traces_list=[traces], sampling_frequency=fs
    )
    # Heterogeneous gains across channels -- different conversion
    # factors. The check uses ``np.unique`` so any non-uniformity
    # trips it.
    synthetic_rec.set_channel_gains([0.195, 0.195, 0.5, 0.5])

    with pytest.raises(ValueError, match="heterogeneous channel"):
        Recording._write_nwb_artifact(
            recording=synthetic_rec,
            nwb_file_name=nwb_file_name,
            filtering_description="test",
        )


# ---------- Recording.make rollback file cleanup -------------------------


@pytest.mark.slow
def test_recording_make_rollback_cleans_analysis_nwb(
    polymer_smoke_session, monkeypatch
):
    """``Recording.make`` unlinks its staged AnalysisNwbfile on a
    transaction rollback.

    Mirrors ``test_sorting_make_rollback_cleans_units_nwb`` for the
    Recording stage. Patches ``Recording.insert1`` to raise inside
    the transaction (after ``AnalysisNwbfile().add`` has run), so
    the transaction rolls back. The except block in
    ``Recording.make`` is responsible for unlinking the staged
    preprocessed NWB.
    """
    import pathlib

    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
        SortGroupV2,
    )

    _clean_session_v2(polymer_smoke_session)
    initialize_v2_defaults()
    LabTeam.insert1(
        {
            "team_name": "v2_test_team",
            "team_description": "v2 pipeline tests",
        },
        skip_duplicates=True,
    )
    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
    sort_group_id = int(
        sorted((SortGroupV2 & polymer_smoke_session).fetch("sort_group_id"))[0]
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
    (Recording & rec_pk).super_delete(warn=False, force_masters=True)

    from spyglass.settings import analysis_dir as ad

    analysis_dir = pathlib.Path(ad)
    before = (
        {p.name for p in analysis_dir.rglob("*.nwb")}
        if analysis_dir.exists()
        else set()
    )

    # Force a failure AFTER the NWB is written and registered
    # (Recording.make wraps AnalysisNwbfile().add + self.insert1 in
    # transaction_or_noop; patching self.insert1 makes the
    # transaction roll back with the file already on disk).
    original_insert1 = Recording.insert1

    def _broken_insert1(self, *args, **kwargs):
        raise RuntimeError("simulated Recording.insert1 failure")

    monkeypatch.setattr(Recording, "insert1", _broken_insert1)

    Recording.populate(rec_pk, reserve_jobs=False, suppress_errors=True)

    # Restore (monkeypatch teardown also restores).
    monkeypatch.setattr(Recording, "insert1", original_insert1)

    # No Recording row, no orphan NWB file.
    assert (
        len(Recording & rec_pk) == 0
    ), "Recording row present after rollback; transaction did not roll back."
    after = (
        {p.name for p in analysis_dir.rglob("*.nwb")}
        if analysis_dir.exists()
        else set()
    )
    new_files = after - before
    assert not new_files, (
        f"Recording.make rollback left orphan analysis files: "
        f"{new_files}. The except-block must unlink the staged file."
    )


# ---------- global-median reference (reference_mode) ------------


@pytest.mark.slow
def test_recording_make_global_median_reference(polymer_smoke_session):
    """``Recording._apply_pre_motion_preprocessing`` applies the
    global-median reference when ``reference_mode == "global_median"``.

    Every other recording test uses ``reference_mode = "none"``, so the
    ``elif reference_mode == "global_median"`` branch in recording.py has
    never been exercised. This test populates a Recording with
    ``reference_mode = "global_median"`` and pins:

    1. The populate completes without raising (the cross-schema
       ``validated.common_reference.operator`` field resolution
       works -- a load-bearing untested code path).
    2. The resulting traces differ materially from a no-reference
       (``-1``) recording on the same data (the CMR subtracts the
       per-sample channel median, shifting the signal vs the
       unreferenced version).
    3. The runtime order is filter-then-reference: because the
       global-median CMR runs LAST (after bandpass), the recording's
       per-sample median ACROSS channels is ~0 (the median is
       translation-equivariant). The old reference-then-filter order
       (per-channel bandpass AFTER CMR) does not preserve that, so a
       ~0 per-sample median pins the intentional v1-divergent order.
    """
    import numpy as np

    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
        SortGroupV2,
    )

    _clean_session_v2(polymer_smoke_session)
    initialize_v2_defaults()
    LabTeam.insert1(
        {
            "team_name": "v2_test_team",
            "team_description": "v2 pipeline tests",
        },
        skip_duplicates=True,
    )
    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
    sort_group_id = int(
        sorted((SortGroupV2 & polymer_smoke_session).fetch("sort_group_id"))[0]
    )

    # No-reference baseline (-1).
    rec_pk_unref = RecordingSelection.insert_selection(
        {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": sort_group_id,
            "interval_list_name": "raw data valid times",
            "preprocessing_params_name": "default",
            "team_name": "v2_test_team",
        }
    )
    Recording.populate(rec_pk_unref, reserve_jobs=False)
    traces_unref = Recording().get_recording(rec_pk_unref).get_traces()

    # Switch SortGroupV2 to global-median reference and repopulate. A
    # new RecordingSelection gets minted because the cache_hash depends
    # on input data.
    SortGroupV2.update1(
        {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": sort_group_id,
            "reference_mode": "global_median",
        }
    )
    assert (
        SortGroupV2
        & {"nwb_file_name": nwb_file_name, "sort_group_id": sort_group_id}
    ).fetch1("reference_mode") == "global_median"
    # Drop the unref Recording so the new selection materializes a
    # fresh global-median recording rather than reusing the cached
    # bytes.
    (Recording & rec_pk_unref).super_delete(warn=False, force_masters=True)
    Recording.populate(rec_pk_unref, reserve_jobs=False)
    traces_ref = Recording().get_recording(rec_pk_unref).get_traces()

    # Pin (1): same shape (the channel set is unchanged for -2;
    # -1 also doesn't drop a channel, so shapes match).
    assert traces_ref.shape == traces_unref.shape

    # Pin (2): the recordings are materially different. The
    # global-median reference subtracts the per-sample channel
    # median; even after bandpass that produces a measurable shift
    # in the recording's first moment vs the unreferenced version.
    delta = float(np.abs(traces_ref - traces_unref).mean())
    assert delta > 1e-6, (
        f"Global-median (-2) and no-reference (-1) recordings are "
        f"indistinguishable (mean |diff| = {delta:.2e}). The -2 "
        "branch may not be applying CMR; check the call to "
        "sip.common_reference(reference='global')."
    )

    # Pin (3): filter-then-reference order. The global-median CMR is the
    # LAST step, so subtracting the per-sample cross-channel median leaves
    # that median at ~0 (median is translation-equivariant). The old
    # reference-then-filter order (per-channel bandpass AFTER CMR) does not
    # preserve this, so a ~0 per-sample median is the numeric signature of
    # the intentional v1-divergent order. (Requires the default
    # preset's operator="median"; an "average" operator would zero the per-
    # sample MEAN, not the median.)
    scale = max(1.0, float(np.abs(traces_ref).max()))
    per_sample_median = np.median(traces_ref, axis=1)
    max_abs_median = float(np.abs(per_sample_median).max())
    assert max_abs_median < 1e-3 * scale, (
        "global-median recording's per-sample cross-channel median is "
        f"{max_abs_median:.3e} (scale {scale:.3e}); expected ~0, which "
        "holds only when CMR is the FINAL step (bandpass-then-reference)."
    )


@pytest.mark.slow
def test_recording_no_filter_preset_skips_bandpass(polymer_smoke_session):
    """The ``no_filter`` preset materializes an UNfiltered recording.

    ``bandpass_filter`` is optional: the ``no_filter`` preset ships
    ``bandpass_filter=None`` and ``_apply_pre_motion_preprocessing`` must
    SKIP the bandpass step (not silently pass a wide-band that still
    filters). This populates two recordings on the same data -- one with
    the bandpassed ``default`` preset, one with ``no_filter`` --
    and pins: (1) the no_filter populate completes (the ``None`` guard
    does not crash), and (2) its traces differ materially from the
    bandpassed version, which only holds if the filter was actually
    skipped. If the ``if bandpass_filter is not None`` guard were missing
    or inverted, no_filter would still bandpass and the two recordings
    would be (nearly) identical.
    """
    import numpy as np

    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
        SortGroupV2,
    )

    _clean_session_v2(polymer_smoke_session)
    initialize_v2_defaults()
    LabTeam.insert1(
        {"team_name": "v2_test_team", "team_description": "v2 pipeline tests"},
        skip_duplicates=True,
    )
    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
    sort_group_id = int(
        sorted((SortGroupV2 & polymer_smoke_session).fetch("sort_group_id"))[0]
    )

    common = {
        "nwb_file_name": nwb_file_name,
        "sort_group_id": sort_group_id,
        "interval_list_name": "raw data valid times",
        "team_name": "v2_test_team",
    }
    # Bandpassed reference (default ships a real 300-6000 Hz band).
    bp_pk = RecordingSelection.insert_selection(
        {**common, "preprocessing_params_name": "default"}
    )
    Recording.populate(bp_pk, reserve_jobs=False)
    traces_bp = Recording().get_recording(bp_pk).get_traces()

    # no_filter ships bandpass_filter=None -> the filter step is skipped.
    # Distinct preprocessing_params_name => distinct cache_hash => distinct row,
    # so no delete/repopulate dance is needed.
    nf_pk = RecordingSelection.insert_selection(
        {**common, "preprocessing_params_name": "no_filter"}
    )
    Recording.populate(nf_pk, reserve_jobs=False)
    traces_nf = Recording().get_recording(nf_pk).get_traces()

    # (1) shapes match (no_filter still references the same channels).
    assert traces_nf.shape == traces_bp.shape
    # (2) skipping the bandpass leaves low-frequency / DC content the
    # 300 Hz high-pass would have removed, so the unfiltered traces are
    # materially different from the bandpassed ones.
    delta = float(np.abs(traces_nf - traces_bp).mean())
    assert delta > 1e-6, (
        "no_filter and default recordings are indistinguishable "
        f"(mean |diff| = {delta:.2e}); the bandpass was NOT skipped for "
        "no_filter. Check the `if bandpass_filter is not None` guard in "
        "_apply_pre_motion_preprocessing."
    )


# ---------- ADC phase-shift no-op regressions -------------------


@pytest.mark.slow
def test_phase_shift_field_does_not_change_default_recording(
    polymer_smoke_session,
):
    """A pre-field params blob materializes identically to the default.

    The optional ``phase_shift`` field defaults to ``None``, so a params blob
    written BEFORE the field existed (no ``phase_shift`` key) must materialize
    to the **same traces** as the ``default`` preset (whose blob now
    carries ``phase_shift=None``). Pins the headline behavior-preservation
    metric: adding the optional field changes nothing on the default path, and
    existing rows validate + materialize unchanged.

    Note: the materialized recordings are compared by **trace equality**, not
    ``cache_hash``. ``cache_hash`` is an ``NwbfileHasher`` digest that includes
    per-write NWB metadata (the analysis file's ``object_id`` UUIDs and
    ``file_create_date``), so two separate fresh writes of byte-identical
    preprocessing produce different hashes; the preprocessed traces are the
    behavioral output to compare.
    """
    import numpy as np

    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2._params.preprocessing import (
        PreprocessingParamsSchema,
    )
    from spyglass.spikesorting.v2.recording import (
        PreprocessingParameters,
        Recording,
        RecordingSelection,
        SortGroupV2,
    )

    _clean_session_v2(polymer_smoke_session)
    initialize_v2_defaults()
    LabTeam.insert1(
        {"team_name": "v2_test_team", "team_description": "v2 pipeline tests"},
        skip_duplicates=True,
    )
    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
    sort_group_id = int(
        sorted((SortGroupV2 & polymer_smoke_session).fetch("sort_group_id"))[0]
    )
    common = {
        "nwb_file_name": nwb_file_name,
        "sort_group_id": sort_group_id,
        "interval_list_name": "raw data valid times",
        "team_name": "v2_test_team",
    }

    # default (carries phase_shift=None in its blob).
    fl_pk = RecordingSelection.insert_selection(
        {**common, "preprocessing_params_name": "default"}
    )
    Recording.populate(fl_pk, reserve_jobs=False)
    traces_fl = Recording().get_recording(fl_pk).get_traces()

    # A pre-field blob: the default schema dump with the phase_shift key
    # removed (what a row written before the field looked like). It validates
    # (phase_shift defaults to None) and must materialize identically.
    legacy_blob = PreprocessingParamsSchema().model_dump()
    legacy_blob.pop("phase_shift")
    # After validation this re-fills phase_shift=None, so the blob is content-
    # identical to the shipped ``default`` row -- which is exactly the point
    # (it must materialize identically). Opt out of the duplicate guard.
    PreprocessingParameters.insert1(
        {
            "preprocessing_params_name": "_pytest_legacy_no_phase_shift",
            "params": legacy_blob,
            "params_schema_version": 3,
            "job_kwargs": None,
        },
        skip_duplicates=True,
        allow_duplicate_params=True,
    )
    legacy_pk = RecordingSelection.insert_selection(
        {**common, "preprocessing_params_name": "_pytest_legacy_no_phase_shift"}
    )
    try:
        Recording.populate(legacy_pk, reserve_jobs=False)
        traces_legacy = Recording().get_recording(legacy_pk).get_traces()

        assert np.array_equal(traces_legacy, traces_fl), (
            "A pre-field params blob (no phase_shift key) materialized to "
            "different traces than default; the optional phase_shift "
            "field perturbed the default path."
        )
    finally:
        # Clean up the throwaway params row + its lineage: it has the SAME
        # content as 'default', so leaking it would make a later
        # describe_parameter_rows duplicate-of resolution ambiguous in another
        # test (a cross-test isolation bug).
        (Recording & legacy_pk).delete_quick()
        (RecordingSelection & legacy_pk).delete_quick()
        (
            PreprocessingParameters
            & {"preprocessing_params_name": "_pytest_legacy_no_phase_shift"}
        ).delete_quick()


@pytest.mark.slow
def test_neuropixels_preset_phase_shift_inert_without_property(
    polymer_smoke_session, caplog
):
    """The ``default_neuropixels`` preset is a safe no-op on non-multiplexed data.

    ``default_neuropixels`` enables phase-shift; ``default`` does not,
    and the two presets are otherwise identical. The smoke fixture carries no
    ``inter_sample_shift`` property, so the requested phase-shift is **skipped**
    (with a logged warning) and the two presets must materialize to the **same
    traces** -- proving the blessed NP recipe never alters the output on a
    recording lacking the property.

    Compared by trace equality rather than ``cache_hash`` (see
    ``test_phase_shift_field_does_not_change_default_recording`` for why
    ``cache_hash`` is not stable across two fresh writes).
    """
    import numpy as np

    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
        SortGroupV2,
    )

    _clean_session_v2(polymer_smoke_session)
    initialize_v2_defaults()
    LabTeam.insert1(
        {"team_name": "v2_test_team", "team_description": "v2 pipeline tests"},
        skip_duplicates=True,
    )
    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
    sort_group_id = int(
        sorted((SortGroupV2 & polymer_smoke_session).fetch("sort_group_id"))[0]
    )
    common = {
        "nwb_file_name": nwb_file_name,
        "sort_group_id": sort_group_id,
        "interval_list_name": "raw data valid times",
        "team_name": "v2_test_team",
    }

    fl_pk = RecordingSelection.insert_selection(
        {**common, "preprocessing_params_name": "default"}
    )
    Recording.populate(fl_pk, reserve_jobs=False)
    traces_fl = Recording().get_recording(fl_pk).get_traces()

    np_pk = RecordingSelection.insert_selection(
        {**common, "preprocessing_params_name": "default_neuropixels"}
    )
    with caplog.at_level("WARNING"):
        Recording.populate(np_pk, reserve_jobs=False)
    traces_np = Recording().get_recording(np_pk).get_traces()

    assert np.array_equal(traces_np, traces_fl), (
        "default_neuropixels and default produced different traces "
        "on a fixture without inter_sample_shift; the phase-shift was NOT "
        "skipped (or the presets differ by more than phase-shift)."
    )
    assert any(
        "skipping phase-shift" in record.getMessage()
        for record in caplog.records
    ), "the absent-property phase-shift skip was not logged during populate"


# ---------- DuplicateSelectionError on insert_selection -----------


@pytest.mark.slow
def test_recording_selection_raises_on_duplicate_logical_identity(
    polymer_smoke_session,
):
    """``RecordingSelection.insert_selection`` raises
    ``DuplicateSelectionError`` when the table already contains
    multiple rows with the same logical identity (an integrity bug
    that should never happen via the helper but can happen if a
    caller bypasses it with raw ``insert``).

    Without this test the integrity guard at
    ``recording.py:727-730`` is dead code that, if it silently
    broke, would let downstream consumers index ``[0]`` on
    duplicate rows.
    """
    import uuid

    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.exceptions import DuplicateSelectionError
    from spyglass.spikesorting.v2.recording import (
        RecordingSelection,
        SortGroupV2,
    )

    _clean_session_v2(polymer_smoke_session)
    initialize_v2_defaults()
    LabTeam.insert1(
        {
            "team_name": "v2_test_team",
            "team_description": "v2 pipeline tests",
        },
        skip_duplicates=True,
    )
    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
    sort_group_id = int(
        sorted((SortGroupV2 & polymer_smoke_session).fetch("sort_group_id"))[0]
    )
    logical_identity = {
        "nwb_file_name": nwb_file_name,
        "sort_group_id": sort_group_id,
        "interval_list_name": "raw data valid times",
        "preprocessing_params_name": "default",
        "team_name": "v2_test_team",
    }
    # Bypass the helper to plant the duplicate (this is what the
    # guard exists to catch in case some script uses raw dj.insert).
    planted_ids = []
    RecordingSelection().insert1(
        {**logical_identity, "recording_id": uuid.uuid4()},
        allow_direct_insert=True,
    )
    RecordingSelection().insert1(
        {**logical_identity, "recording_id": uuid.uuid4()},
        allow_direct_insert=True,
    )
    planted_ids = list(
        (RecordingSelection & logical_identity).fetch("recording_id")
    )

    try:
        with pytest.raises(DuplicateSelectionError, match="duplicate"):
            RecordingSelection.insert_selection(logical_identity)
    finally:
        # CRITICAL: clean up the duplicates we planted so the next
        # test's RecordingSelection.insert_selection doesn't trip
        # the same guard. Without this teardown, the function-scoped
        # populated_recording fixture (whose body calls
        # insert_selection) would fail in every subsequent test.
        for rid in planted_ids:
            (RecordingSelection & {"recording_id": rid}).super_delete(
                warn=False
            )


@pytest.mark.slow
@pytest.mark.integration
def test_cache_hash_uses_nwbfile_hasher(populated_recording):
    """``Recording.cache_hash`` matches ``_hash_nwb_recording``.

    An earlier cache_hash was ``hashlib.sha256(data.tobytes())``
    -- content-blind to timestamps, electrodes, and conversion
    metadata. The recording-cache contract binds cache_hash to
    ``NwbfileHasher`` (Spyglass's content hasher used by the v1
    recompute machinery), so the digest covers the full NWB content
    rather than just the raw data bytes. This test independently
    recomputes the hash and asserts the stored value matches -- a
    regression to the data-only sha256 would silently pass the
    existing ``isinstance(cache_hash, str)`` check.
    """
    from spyglass.spikesorting.v2.recording import Recording
    from spyglass.spikesorting.v2.utils import _hash_nwb_recording

    row = (Recording & populated_recording).fetch1()
    expected = _hash_nwb_recording(row["analysis_file_name"])
    assert row["cache_hash"] == expected, (
        f"cache_hash {row['cache_hash']!r} does not match "
        f"independently-computed NwbfileHasher digest {expected!r}. "
        "NwbfileHasher contract violated -- check that "
        "Recording._write_nwb_artifact still routes through "
        "_hash_nwb_recording instead of a data-only sha256."
    )


@pytest.mark.slow
@pytest.mark.integration
def test_recording_get_recording_honors_electrical_series_path(
    populated_recording, monkeypatch
):
    """``Recording.get_recording`` forwards
    ``electrical_series_path`` to SI's ``read_nwb_recording``.

    Without the forwarding, a future AnalysisNwbfile with more
    than one ElectricalSeries (e.g., LFP next to raw) would let
    SI auto-detect the wrong source. Monkey-patches
    ``se.read_nwb_recording`` to capture the kwarg.
    """
    import spikeinterface.extractors as se

    from spyglass.spikesorting.v2.recording import Recording

    captured_kwargs = {}
    real_fn = se.read_nwb_recording

    def _capture(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return real_fn(*args, **kwargs)

    monkeypatch.setattr(se, "read_nwb_recording", _capture)
    Recording().get_recording(populated_recording)

    expected = (Recording & populated_recording).fetch1(
        "electrical_series_path"
    )
    assert captured_kwargs.get("electrical_series_path") == expected, (
        f"se.read_nwb_recording called without electrical_series_path"
        f"={expected!r}; got {captured_kwargs.get('electrical_series_path')!r}. "
        "Regression -- the stored column is being ignored."
    )


@pytest.mark.slow
def test_recording_specific_reference_drops_ref_channel(polymer_smoke_session):
    """specific mode references then drops the reference electrode.

    The reference electrode is included in the channel slice so
    ``common_reference`` can subtract it, then removed -- the cached
    recording must NOT carry it, while the sort group's own electrodes
    remain. Uses a cross-shank reference electrode that is NOT a member of
    any created sort group (the helper now fails early if a ``"specific"``
    reference is a member of the group it references), so only the target
    shank is grouped and the reference electrode's own shank is left out.
    """
    from spyglass.common.common_ephys import Electrode
    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2.recording import (
        PreprocessingParameters,
        Recording,
        RecordingSelection,
        SortGroupV2,
    )

    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    _clean_session_v2(polymer_smoke_session)
    PreprocessingParameters.insert_default()
    LabTeam.insert1(
        {"team_name": "v2_test_team", "team_description": "v2 pipeline tests"},
        skip_duplicates=True,
    )

    # Map each shank to its (good) electrode ids so we can group ONE shank
    # and reference it to an electrode on a DIFFERENT, un-grouped shank.
    elec_rows = (
        Electrode & {"nwb_file_name": nwb_file_name, "bad_channel": "False"}
    ).fetch("electrode_id", "probe_shank", as_dict=True)
    by_shank: dict[int, list[int]] = {}
    for row in elec_rows:
        by_shank.setdefault(int(row["probe_shank"]), []).append(
            int(row["electrode_id"])
        )
    shanks = sorted(by_shank)
    assert len(shanks) >= 2, "need >=2 shanks to isolate a cross-shank ref"
    target_shank, ref_shank = shanks[0], shanks[1]
    target_elecs = sorted(by_shank[target_shank])
    ref_id = sorted(by_shank[ref_shank])[0]
    assert ref_id not in target_elecs

    # One sort group (the target shank), referenced to the cross-shank
    # electrode. The reference electrode's own shank is intentionally NOT
    # grouped, so the specific reference is outside every created group.
    SortGroupV2.set_group_by_electrode_table_column(
        nwb_file_name=nwb_file_name,
        column="probe_shank",
        groups=[[target_shank]],
        reference_mode="specific",
        reference_electrode_id=ref_id,
    )
    target_sg = int(
        (SortGroupV2 & polymer_smoke_session).fetch("sort_group_id")[0]
    )

    rec_pk = RecordingSelection.insert_selection(
        {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": target_sg,
            "interval_list_name": "raw data valid times",
            "preprocessing_params_name": "default",
            "team_name": "v2_test_team",
        }
    )
    Recording.populate(rec_pk, reserve_jobs=False)
    rec = Recording().get_recording(rec_pk)
    cached_ids = {int(c) for c in rec.get_channel_ids()}
    assert ref_id not in cached_ids, (
        "specific-mode reference electrode leaked into the cached "
        "recording; it must be dropped after referencing."
    )
    assert (
        set(target_elecs) <= cached_ids
    ), "sort-group electrodes missing from the cached recording."


@pytest.mark.slow
def test_recording_multi_interval_saved_times_use_concat_path(
    polymer_smoke_session,
):
    """A multi-interval (disjoint) selection derives ``saved_times`` from
    the gap-EXCLUDED concat path, not the gap-spanning override.

    For ``n_selected_intervals > 1`` the recording is built with
    ``concatenate_recordings(ignore_times=True)``, so ``saved_times`` is the
    0-based concat times whose span excludes the inter-segment gap. The row's
    ``duration_s`` must therefore equal the SUM of the kept-interval durations
    (gap-excluded), clearly shorter than the wall-clock envelope
    (last_end - first_start). This pins the persisted side of the
    gap-exclusion guarantee for disjoint multi-interval selections.
    """
    import numpy as np

    from spyglass.common import IntervalList
    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2.recording import (
        PreprocessingParameters,
        Recording,
        RecordingSelection,
        SortGroupV2,
    )

    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    _clean_session_v2(polymer_smoke_session)
    PreprocessingParameters.insert_default()
    LabTeam.insert1(
        {"team_name": "v2_test_team", "team_description": "v2 pipeline tests"},
        skip_duplicates=True,
    )
    SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)

    raw_times = (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": "raw data valid times",
        }
    ).fetch1("valid_times")
    raw_start = float(raw_times[0][0])
    raw_end = float(raw_times[-1][-1])
    total = raw_end - raw_start
    # Each 0.3*total segment must clear min_segment_length (1.0 s), so the
    # fixture needs >= ~3.5 s of raw coverage.
    assert total >= 3.5, f"fixture too short ({total}s) for multi-interval test"

    # Two disjoint segments, both well within raw coverage, separated by a
    # real gap of ~0.3*total.
    seg1 = [raw_start, raw_start + 0.3 * total]
    seg2 = [raw_start + 0.6 * total, raw_start + 0.9 * total]
    gap_excluded = (seg1[1] - seg1[0]) + (seg2[1] - seg2[0])
    envelope = seg2[1] - seg1[0]

    interval_name = "v2_multi_interval_valid"
    IntervalList.insert1(
        {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": interval_name,
            "valid_times": np.asarray([seg1, seg2]),
            "pipeline": "v2_multi_interval",
        },
        skip_duplicates=True,
    )
    sort_group_id = int(
        sorted((SortGroupV2 & polymer_smoke_session).fetch("sort_group_id"))[0]
    )
    pk = RecordingSelection.insert_selection(
        {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": sort_group_id,
            "interval_list_name": interval_name,
            "preprocessing_params_name": "default",
            "team_name": "v2_test_team",
        }
    )
    try:
        Recording.populate(pk, reserve_jobs=False)
        duration_s = float((Recording & pk).fetch1("duration_s"))
        # Concat path: duration ~= sum of kept durations (gap-excluded).
        assert abs(duration_s - gap_excluded) < 0.1 * total, (
            f"duration_s={duration_s} not close to the gap-excluded sum "
            f"{gap_excluded} (concat path); total={total}"
        )
        # And clearly NOT the gap-spanning wall-clock envelope.
        assert duration_s < envelope - 0.1 * total, (
            f"duration_s={duration_s} looks like the gap-SPANNING envelope "
            f"{envelope}; the override path was used instead of concat"
        )
    finally:
        (Recording & pk).super_delete(warn=False, force_masters=True)
        (RecordingSelection & pk).super_delete(warn=False)
        (
            IntervalList
            & {
                "nwb_file_name": nwb_file_name,
                "interval_list_name": interval_name,
            }
        ).super_delete(warn=False)


@pytest.mark.slow
def test_recording_fresh_write_cleanup_unlinks_staged_file(
    polymer_smoke_session, monkeypatch
):
    """A failure AFTER the fresh ``_write_nwb_artifact`` unlinks the
    staged file (no half-written artifact outlives a failed compute).

    On the fresh-write path (``existing_analysis_file_name is None``), if a
    later metadata step raises after the file is on disk, the staged file is
    unlinked before the exception propagates. We wrap ``_write_nwb_artifact``
    to record the staged name, then make the post-write
    ``_get_recording_timestamps`` raise; afterwards the staged file must be
    gone and no Recording row may exist.
    """
    from spyglass.common import IntervalList  # noqa: F401
    from spyglass.common.common_lab import LabTeam
    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2 import recording as rec_mod
    from spyglass.spikesorting.v2 import utils as utils_mod
    from spyglass.spikesorting.v2.recording import (
        PreprocessingParameters,
        Recording,
        RecordingSelection,
        SortGroupV2,
    )

    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    _clean_session_v2(polymer_smoke_session)
    PreprocessingParameters.insert_default()
    LabTeam.insert1(
        {"team_name": "v2_test_team", "team_description": "v2 pipeline tests"},
        skip_duplicates=True,
    )
    SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
    sort_group_id = int(
        sorted((SortGroupV2 & polymer_smoke_session).fetch("sort_group_id"))[0]
    )
    pk = RecordingSelection.insert_selection(
        {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": sort_group_id,
            "interval_list_name": "raw data valid times",
            "preprocessing_params_name": "default",
            "team_name": "v2_test_team",
        }
    )

    captured = {}
    real_write = Recording._write_nwb_artifact

    def _capture_write(*args, **kwargs):
        result = real_write(*args, **kwargs)
        captured["analysis_file_name"] = result[0]
        return result

    real_ts = utils_mod._get_recording_timestamps

    def _raise_after_write(*args, **kwargs):
        # Earlier calls (restrict / repair) succeed; only the post-write
        # saved-times derivation raises so we exercise the fresh-write
        # cleanup branch rather than failing before the file is staged.
        if "analysis_file_name" in captured:
            raise RuntimeError("staged-file cleanup probe: post-write boom")
        return real_ts(*args, **kwargs)

    monkeypatch.setattr(
        Recording, "_write_nwb_artifact", staticmethod(_capture_write)
    )
    monkeypatch.setattr(
        utils_mod, "_get_recording_timestamps", _raise_after_write
    )
    # recording.py imports the symbol at call time from utils, so patching
    # the utils module is sufficient; guard against a module-level rebind.
    monkeypatch.setattr(
        rec_mod, "_get_recording_timestamps", _raise_after_write, raising=False
    )

    try:
        with pytest.raises(Exception, match="post-write boom"):
            Recording.populate(pk, reserve_jobs=False)
        assert (
            "analysis_file_name" in captured
        ), "precondition: _write_nwb_artifact must have run and staged a file"
        staged = captured["analysis_file_name"]
        abs_path = AnalysisNwbfile.get_abs_path(staged)
        assert not Path(
            abs_path
        ).exists(), (
            f"fresh-write cleanup did not unlink the staged file {staged!r}"
        )
        assert not (Recording & pk), "no Recording row may be inserted"
    finally:
        (Recording & pk).super_delete(warn=False, force_masters=True)
        (RecordingSelection & pk).super_delete(warn=False)


@pytest.mark.slow
def test_recording_rebuild_path_keeps_existing_file_on_failure(
    populated_sorting, monkeypatch
):
    """A post-write failure on the REBUILD path (the complement of the
    fresh-write case: ``existing_analysis_file_name`` set) does NOT unlink
    the file.

    The fresh-write path unlinks a staged file on failure; the rebuild path
    must not, because there the file IS the canonical cache (a mid-write
    failure surfaces via the caller's hash-mismatch check, not by destroying
    the only copy). We drive ``_compute_recording_artifact`` directly with an
    ``existing_analysis_file_name`` and the same post-write
    ``_get_recording_timestamps`` failure, then assert the file survives. This
    pins the false branch of the ``existing_analysis_file_name is None`` guard.

    Uses a freshly created AnalysisNwbfile as the stand-in cache so the shared
    fixture's real recording file is never touched.
    """
    from spyglass.common.common_nwbfile import (
        AnalysisNwbfile,
        Nwbfile,
    )
    from spyglass.spikesorting.v2 import utils as utils_mod
    from spyglass.spikesorting.v2.recording import Recording, RecordingSelection
    from spyglass.spikesorting.v2.sorting import SortingSelection

    recording_id = SortingSelection.resolve_source(populated_sorting).key[
        "recording_id"
    ]
    fetched = Recording().make_fetch({"recording_id": recording_id})
    nwb_file_name = fetched.sel["nwb_file_name"]

    # A real on-disk file standing in for the canonical cache.
    existing = AnalysisNwbfile().create(nwb_file_name)
    existing_abs = AnalysisNwbfile.get_abs_path(existing)
    assert Path(existing_abs).exists(), "precondition: cache file created"

    captured = {}
    real_write = Recording._write_nwb_artifact
    real_ts = utils_mod._get_recording_timestamps

    def _capture_write(*args, **kwargs):
        result = real_write(*args, **kwargs)
        captured["written"] = True
        return result

    def _raise_after_write(*args, **kwargs):
        if "written" in captured:
            raise RuntimeError("rebuild post-write boom")
        return real_ts(*args, **kwargs)

    monkeypatch.setattr(
        Recording, "_write_nwb_artifact", staticmethod(_capture_write)
    )
    monkeypatch.setattr(
        utils_mod, "_get_recording_timestamps", _raise_after_write
    )

    raw_path = Nwbfile().get_abs_path(nwb_file_name)
    try:
        with pytest.raises(Exception, match="rebuild post-write boom"):
            Recording()._compute_recording_artifact(
                raw_path=raw_path,
                nwb_file_name=nwb_file_name,
                interval_list_name=fetched.sel["interval_list_name"],
                channel_ids=fetched.channel_ids,
                reference_mode=fetched.reference_mode,
                reference_electrode_id=fetched.reference_electrode_id,
                sort_valid_times=fetched.sort_valid_times,
                raw_valid_times=fetched.raw_valid_times,
                preprocessing_params=fetched.preprocessing_params,
                probe_types=fetched.probe_types,
                electrode_group_names=fetched.electrode_group_names,
                existing_analysis_file_name=existing,  # REBUILD path
            )
        assert (
            "written" in captured
        ), "precondition: _write_nwb_artifact must have run before the failure"
        # The rebuild path must NOT unlink the canonical cache file.
        assert Path(existing_abs).exists(), (
            "rebuild path unlinked the existing cache file -- it must only "
            "unlink on the fresh-write (existing is None) path"
        )
    finally:
        Path(existing_abs).unlink(missing_ok=True)
        # The create() call may have registered an AnalysisNwbfile row.
        if AnalysisNwbfile & {"analysis_file_name": existing}:
            (AnalysisNwbfile & {"analysis_file_name": existing}).delete(
                safemode=False
            )

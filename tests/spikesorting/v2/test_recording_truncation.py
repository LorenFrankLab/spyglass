"""Recording truncation guard fires only on TRUE truncation.

A sort interval may contain a chunk shorter than ``min_segment_length``;
``_restrict_recording`` intentionally drops such slivers (same as v1). The
truncation guard must compare the saved duration against the
``min_segment_length``-filtered intended duration, NOT the raw requested
chunks -- otherwise the intentionally-dropped sliver is mis-flagged as
truncation and ``Recording.populate`` crashes (and deletes the written file)
on a perfectly valid multi-epoch request.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import pytest

from tests.spikesorting.v2._ingest_helpers import copy_and_insert_nwb

_FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "mearec_polymer_smoke.nwb"
)


@pytest.fixture(scope="module")
def truncation_session(dj_conn):
    """Ingest the smoke fixture under a unique session name; clean around."""
    from tests.spikesorting.v2.test_single_session_pipeline import (
        _clean_session_v2,
    )

    if not _FIXTURE_PATH.exists():
        pytest.skip(f"Fixture {_FIXTURE_PATH.name} not found.")

    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.recording import SortGroupV2

    nwb = copy_and_insert_nwb(_FIXTURE_PATH, dest_name="mearec_truncation.nwb")
    session = {"nwb_file_name": nwb}
    _clean_session_v2(session)
    initialize_v2_defaults()
    LabTeam.insert1(
        {"team_name": "v2_test_team", "team_description": "v2 truncation"},
        skip_duplicates=True,
    )
    if not (SortGroupV2 & session):
        SortGroupV2.set_group_by_shank(nwb_file_name=nwb)
    yield session
    _clean_session_v2(session)


@pytest.mark.slow
@pytest.mark.integration
def test_sub_min_segment_length_sliver_is_not_false_truncation(
    truncation_session,
):
    """A 0.5 s sliver (< the 1.0 s default min_segment_length) is dropped,
    not treated as truncation: Recording.populate succeeds and saves only
    the surviving chunk."""
    from spyglass.common.common_interval import IntervalList
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
        SortGroupV2,
    )

    nwb = truncation_session["nwb_file_name"]
    raw_times = (
        IntervalList
        & {"nwb_file_name": nwb, "interval_list_name": "raw data valid times"}
    ).fetch1("valid_times")
    t0 = float(raw_times[0][0])
    t_end = float(raw_times[-1][-1])
    assert (t_end - t0) >= 2.9, "smoke fixture too short for this test"

    # A 0.5 s sliver (dropped by min_segment_length=1.0) + a ~1.9 s chunk
    # (kept). Raw requested total ~2.4 s; intended-after-filtering ~1.9 s.
    sliver_end = t0 + 0.5
    chunk_start, chunk_end = t0 + 1.0, min(t0 + 2.9, t_end)
    valid_times = np.array([[t0, sliver_end], [chunk_start, chunk_end]])
    interval_name = f"v2_truncation_sliver_{uuid.uuid4().hex[:8]}"
    IntervalList.insert1(
        {
            "nwb_file_name": nwb,
            "interval_list_name": interval_name,
            "valid_times": valid_times,
            "pipeline": "v2_truncation_test",
        }
    )

    sg = int(
        sorted((SortGroupV2 & truncation_session).fetch("sort_group_id"))[0]
    )
    rec_pk = RecordingSelection.insert_selection(
        {
            "nwb_file_name": nwb,
            "sort_group_id": sg,
            "interval_list_name": interval_name,
            "preproc_params_name": "default_franklab",
            "team_name": "v2_test_team",
        }
    )

    # Before the fix this raised RecordingTruncatedError (counted the dropped
    # sliver as missing); it must now populate cleanly.
    Recording.populate(rec_pk, reserve_jobs=False)
    assert Recording & rec_pk, "Recording.populate did not create a row"

    # Only the surviving ~1.9 s chunk is saved (sliver dropped), not the raw
    # ~2.4 s -- proving the sliver was filtered, not that the guard was
    # simply removed.
    duration_s = float((Recording & rec_pk).fetch1("duration_s"))
    kept = chunk_end - chunk_start
    assert abs(duration_s - kept) < 0.1, (
        f"saved duration {duration_s:.3f}s should match the kept chunk "
        f"{kept:.3f}s (sliver dropped), not the raw requested ~2.4 s"
    )


@pytest.mark.usefixtures("dj_conn")
def test_truncation_tolerance_scales_with_interval_count():
    """The truncation tolerance grows with the consolidated-interval count.

    Each kept interval's ``[start, end]`` snaps to the raw sample grid
    INDEPENDENTLY, so the per-boundary quantization error (up to ~1 sample
    each) accumulates across disjoint epochs. The pre-fix guard used a
    fixed ``1.5 / fs`` slack, which false-positived on legitimate
    multi-epoch sorts and then deleted the just-written file. The scaled
    tolerance must accept that legitimate slop while still flagging a
    genuine packet drop. Deterministic (no populate) guard for the policy.
    """
    from spyglass.spikesorting.v2.recording import Recording

    fs = 30000.0
    T = 1.0 / fs
    fixed_old_slack = 1.5 * T  # the removed, interval-count-blind tolerance

    for n in (1, 2, 3, 5, 10, 20):
        tol = Recording._truncation_tolerance(n, fs)
        # Pin the exact coefficient so a silent change of 1.5 -> e.g. 1.2 is
        # caught, not just the inequality bounds below. Use the SAME
        # operation order as production ((n+1.5)/fs, not *T) for bit-equality.
        assert tol == (n + 1.5) / fs
        # Worst-case legitimate "missing" from grid snapping is bounded by
        # ~(n + 1) samples: n interval boundaries + the (N-1)/fs concat
        # off-by-one. (Numerically the observed worst is even smaller.)
        worst_legit_missing = (n + 1.0) * T
        assert worst_legit_missing <= tol, (
            f"n={n}: legitimate missing {worst_legit_missing:.3e}s exceeds "
            f"scaled tolerance {tol:.3e}s -- a valid multi-epoch sort would "
            "falsely raise RecordingTruncatedError"
        )
        # The OLD fixed slack would have raised on that same legitimate
        # slop -- exactly the false positive the fix removes.
        assert worst_legit_missing > fixed_old_slack, (
            f"n={n}: expected the legitimate slop to exceed the old fixed "
            "1.5-sample tolerance (otherwise this test proves nothing)"
        )
        # A genuine packet drop (far more than n+1.5 samples) still raises.
        genuine_drop = (n + 100.0) * T
        assert genuine_drop > tol, (
            f"n={n}: a genuine {int(n + 100)}-sample drop must exceed "
            "tolerance and still raise RecordingTruncatedError"
        )


@pytest.mark.slow
@pytest.mark.integration
def test_disjoint_multichunk_not_false_truncation(truncation_session):
    """Two disjoint, off-grid epochs (both > min_segment_length) populate
    without a false ``RecordingTruncatedError``, and the saved duration
    matches the summed kept-epoch durations within sample-level slop.

    Exercises the ``n_intended_intervals >= 2`` wiring end-to-end (the
    sliver / over-request tests only reach ``n == 1``): the consolidated
    interval count must be threaded make_compute -> make_insert so the
    tolerance scales, otherwise the grid-snapping slop accumulates across
    the two epochs and the guard spuriously fires.
    """
    from spyglass.common.common_interval import IntervalList
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
        SortGroupV2,
    )

    nwb = truncation_session["nwb_file_name"]
    raw_times = (
        IntervalList
        & {"nwb_file_name": nwb, "interval_list_name": "raw data valid times"}
    ).fetch1("valid_times")
    t0 = float(raw_times[0][0])
    t_end = float(raw_times[-1][-1])
    if (t_end - t0) < 2.7:
        pytest.skip("smoke fixture too short for a 2-epoch disjoint test")

    # Two ~1.2 s epochs (each > the 1.0 s default min_segment_length) with a
    # 0.2 s gap, with deliberately non-sample-aligned endpoints.
    c1 = (t0 + 0.05, t0 + 1.25)
    c2 = (t0 + 1.45, t0 + 2.65)
    valid_times = np.array([list(c1), list(c2)])
    interval_name = f"v2_truncation_disjoint_{uuid.uuid4().hex[:8]}"
    IntervalList.insert1(
        {
            "nwb_file_name": nwb,
            "interval_list_name": interval_name,
            "valid_times": valid_times,
            "pipeline": "v2_truncation_test",
        }
    )

    sg = int(
        sorted((SortGroupV2 & truncation_session).fetch("sort_group_id"))[0]
    )
    rec_pk = RecordingSelection.insert_selection(
        {
            "nwb_file_name": nwb,
            "sort_group_id": sg,
            "interval_list_name": interval_name,
            "preproc_params_name": "default_franklab",
            "team_name": "v2_test_team",
        }
    )

    # Must NOT raise RecordingTruncatedError on a valid disjoint request.
    Recording.populate(rec_pk, reserve_jobs=False)
    assert Recording & rec_pk, "disjoint multi-epoch Recording.populate failed"

    duration_s = float((Recording & rec_pk).fetch1("duration_s"))
    kept = (c1[1] - c1[0]) + (c2[1] - c2[0])  # ~2.4 s
    assert abs(duration_s - kept) < 0.01, (
        f"saved duration {duration_s:.4f}s should match the summed kept "
        f"epochs {kept:.4f}s within sample-level slop (both epochs kept)"
    )

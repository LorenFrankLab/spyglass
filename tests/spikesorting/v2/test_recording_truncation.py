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

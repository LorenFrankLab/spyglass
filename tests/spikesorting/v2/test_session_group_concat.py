"""Behavioral tests for same-day chronic concatenate-and-sort.

Covers the ``SessionGroup`` grouping table (multi-day gate, derived dates,
atomic create, owner namespacing), the ``ConcatenatedRecordingSelection``
missing-recording precondition, and -- behind the ``slow`` marker -- the
``ConcatenatedRecording`` materializer plus the concat-backed
``Sorting.populate`` + ``split_sorting_by_session`` round trip.

The substrate is the package-scoped ``chronic_2_session_minirec`` fixture
(two same-day single-tetrode sessions with populated ``Recording`` caches,
plus one next-day session for the multi-day gate).
"""

from __future__ import annotations

import datetime as dt

import pytest


@pytest.mark.slow
def test_chronic_fixture_provides_two_same_day_and_one_next_day(
    chronic_2_session_minirec,
):
    """The fixture yields two same-day members (with populated Recordings) and
    one next-day member; dates derive from ``Session.session_start_time``."""
    from spyglass.common import Session
    from spyglass.spikesorting.v2.recording import Recording

    sub = chronic_2_session_minirec
    same_day = sub["same_day_members"]
    next_day = sub["next_day_member"]

    assert len(same_day) == 2
    # Both same-day members have a populated Recording cache keyed by the
    # content-addressed recording_id the fixture stashed.
    for member in same_day:
        assert Recording & {"recording_id": member["recording_id"]}

    def _date(member):
        return (
            Session & {"nwb_file_name": member["nwb_file_name"]}
        ).fetch1("session_start_time").date()

    d0, d1 = _date(same_day[0]), _date(same_day[1])
    assert d0 == d1, "the two same-day members must share a date"
    assert _date(next_day) != d0, "the next-day member must differ in date"
    # Identical channel positions are what makes the members concatenable.
    rec0 = Recording().get_recording(
        {"recording_id": same_day[0]["recording_id"]}
    )
    rec1 = Recording().get_recording(
        {"recording_id": same_day[1]["recording_id"]}
    )
    assert list(rec0.get_channel_ids()) == list(rec1.get_channel_ids())
    assert (
        rec0.get_channel_locations() == rec1.get_channel_locations()
    ).all()

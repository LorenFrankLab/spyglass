"""Tests for ``spyglass.common.common_session.Session``.

These exercise the parts of ``common_session`` not reached by ordinary
ingestion (the populate path only ever sees well-formed NWB metadata): the
``Experimenter`` name-mapping branches and the ``with_date_str`` projection.
"""

from types import SimpleNamespace


def test_experimenter_entries_from_multiple_names(common):
    """``generate_entries_from_nwb_object`` maps each name via ``decompose_name``.

    A synthetic NWB object (plain input stand-in, not an oracle) drives the
    real method; the assertion pins the real ``decompose_name`` behaviour —
    space- and comma-delimited names both normalise to ``"First Last"``.
    """
    exp = common.Session.Experimenter()
    nwb_obj = SimpleNamespace(experimenter=["Alice Wonderland", "Builder, Bob"])
    result = exp.generate_entries_from_nwb_object(nwb_obj)
    entries = next(iter(result.values()))
    assert [e["lab_member_name"] for e in entries] == [
        "Alice Wonderland",
        "Bob Builder",
    ]


def test_experimenter_entries_empty_when_no_metadata(common):
    """Missing experimenter metadata takes the early-return branch."""
    exp = common.Session.Experimenter()
    nwb_obj = SimpleNamespace(experimenter=None)
    assert exp.generate_entries_from_nwb_object(nwb_obj) == {}


def test_session_with_date_str(common, mini_restr):
    """``with_date_str`` projects ``session_date_str`` as YYYYMMDD of start."""
    sess = common.Session() & mini_restr
    start = sess.fetch1("session_start_time")
    date_str = sess.with_date_str.fetch1("session_date_str")
    assert date_str == start.strftime("%Y%m%d")

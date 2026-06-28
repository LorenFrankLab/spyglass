"""Hermetic (no-DB) tests for the cross-session matcher plugin interface.

The matcher protocol + registry is pure Python with no DataJoint dependency,
so these tests import it directly and exercise registration, lookup, the
runtime-checkable protocol, and the bundle/pair dataclasses without a database.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def clean_registry():
    """Snapshot and restore the module-global matcher registry per test."""
    from spyglass.spikesorting.v2 import matcher_protocol as mp

    saved_matchers = dict(mp._MATCHER_REGISTRY)
    saved_schemas = dict(mp._SCHEMA_REGISTRY)
    try:
        yield mp
    finally:
        mp._MATCHER_REGISTRY.clear()
        mp._MATCHER_REGISTRY.update(saved_matchers)
        mp._SCHEMA_REGISTRY.clear()
        mp._SCHEMA_REGISTRY.update(saved_schemas)


def _dummy_matcher(mp, name="dummy"):
    """A minimal external matcher (proves the protocol is implementable)."""

    class DummyMatcher:
        def __init__(self):
            self.name = name

        def match(self, session_inputs, params):
            if len(session_inputs) < 2:
                return []
            return []

    return DummyMatcher()


class DummySchema:
    """Stand-in per-matcher params schema."""


def test_dummy_matcher_satisfies_runtime_protocol(clean_registry):
    mp = clean_registry
    matcher = _dummy_matcher(mp)
    assert isinstance(matcher, mp.MatcherProtocol)


def test_register_and_get_matcher_round_trip(clean_registry):
    mp = clean_registry
    matcher = _dummy_matcher(mp)
    mp.register_matcher(matcher, DummySchema)
    assert mp.get_matcher("dummy") is matcher
    assert "dummy" in mp._registered_matchers()
    assert mp._get_matcher_schema("dummy") is DummySchema


def test_register_matcher_rejects_non_conforming_object(clean_registry):
    mp = clean_registry
    with pytest.raises(TypeError):
        mp.register_matcher(object(), DummySchema)


def test_register_matcher_duplicate_name_raises_unless_replace(clean_registry):
    """A second backend cannot silently claim an existing matcher name."""
    mp = clean_registry
    first = _dummy_matcher(mp, name="collide")
    second = _dummy_matcher(mp, name="collide")
    mp.register_matcher(first, DummySchema)
    with pytest.raises(ValueError, match="already registered"):
        mp.register_matcher(second, DummySchema)
    # The original backend is untouched by the rejected registration.
    assert mp.get_matcher("collide") is first
    # Explicit opt-in replaces it.
    mp.register_matcher(second, DummySchema, replace=True)
    assert mp.get_matcher("collide") is second


def test_registry_no_silent_reroute(clean_registry):
    """The built-in 'unitmatch' name cannot be silently re-pointed at different
    code: a different backend under that name raises without the explicit
    maintenance flag, and the resolved backend stays the built-in one."""
    mp = clean_registry
    mp.register_default_matchers()  # ensure the built-in is present
    builtin_cls = type(mp.get_matcher("unitmatch"))

    impostor = _dummy_matcher(mp, name="unitmatch")  # different class
    with pytest.raises(ValueError, match="already registered"):
        mp.register_matcher(impostor, DummySchema)
    # The name still resolves to the built-in backend, not the impostor.
    resolved = mp.get_matcher("unitmatch")
    assert type(resolved) is builtin_cls
    assert resolved is not impostor


def test_same_backend_reregisters_idempotently(clean_registry):
    """Re-registering the SAME backend class is idempotent (no maintenance flag)
    -- this is how register_default_matchers self-heals, so the built-in path no
    longer relies on a blunt replace=True that would also mask a real collision.
    """
    mp = clean_registry
    mp.register_default_matchers()
    builtin_cls = type(mp.get_matcher("unitmatch"))
    schema = mp._get_matcher_schema("unitmatch")
    # A fresh instance of the same class registers without raising or a flag.
    mp.register_matcher(builtin_cls(), schema)
    assert type(mp.get_matcher("unitmatch")) is builtin_cls


def test_get_matcher_unknown_raises_with_guidance(clean_registry):
    mp = clean_registry
    from spyglass.spikesorting.v2.exceptions import UnknownMatcherError

    with pytest.raises(UnknownMatcherError) as excinfo:
        mp.get_matcher("nope")
    message = str(excinfo.value)
    assert "nope" in message
    assert "register_matcher" in message


def test_get_matcher_schema_unknown_raises(clean_registry):
    mp = clean_registry
    from spyglass.spikesorting.v2.exceptions import UnknownMatcherError

    with pytest.raises(UnknownMatcherError):
        mp._get_matcher_schema("nope")


def test_degenerate_single_session_returns_empty(clean_registry):
    mp = clean_registry
    matcher = _dummy_matcher(mp)
    one = mp.SessionMatcherInput(
        session_key={"sorting_id": "s", "curation_id": 0},
        waveform_dir=Path("/tmp/x"),
        channel_positions_path=Path("/tmp/x/channel_positions.npy"),
        recording_date=None,
    )
    assert matcher.match([one], {}) == []


def test_match_pair_carries_both_side_keys(clean_registry):
    mp = clean_registry
    pair = mp.MatchPair(
        session_a_sorting_id="a",
        session_a_curation_id=0,
        unit_a_id=3,
        session_b_sorting_id="b",
        session_b_curation_id=1,
        unit_b_id=7,
        match_probability=0.9,
    )
    assert pair.unit_a_id == 3 and pair.unit_b_id == 7
    assert pair.drift_estimate_um == 0.0
    assert pair.fdr_estimate is None

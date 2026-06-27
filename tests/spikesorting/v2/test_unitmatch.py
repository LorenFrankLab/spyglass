"""Cross-session unit tracking: schema + matcher validation.

Covers the ten cross-session matching validation goals. The graph-logic goals
(pair canonicalization, strict-clique tracked-unit derivation, the bounded-search
cap, unmatched singletons) are exercised as pure ``_matcher_graph`` functions
with no database; the table-wiring goals (registry-validating insert, the
explicit per-member curation selection, the ``make()`` provenance recheck, and
the ``Pair`` foreign-key integrity) drive the real DataJoint tables over a
two-session synthetic sort. The ground-truth AUC gate runs UnitMatch end to end
on the polymer two-session fixture and is the ship criterion.

The pure-logic and external-matcher sections need neither a database nor
UnitMatchPy. The database sections request ``dj_conn``; the gate additionally
requires the polymer two-session fixture and the optional UnitMatchPy extra and
skips cleanly when either is absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# --------------------------------------------------------------------------- #
# Pure graph logic: pair canonicalization (goal 7, no DB) and strict-clique     #
# tracked-unit derivation (goals 8/9/10, no DB).                                #
# --------------------------------------------------------------------------- #


def test_unit_semantics_for_sorter():
    """unit_semantics is derived from the sorter (single source of truth): the
    clusterless thresholder emits ONE threshold-crossing pseudo-unit, not a
    sorted neuron; every real sorter emits sorted units."""
    from spyglass.spikesorting.v2._sorting_dispatch import (
        unit_semantics_for_sorter,
    )

    assert (
        unit_semantics_for_sorter("clusterless_thresholder")
        == "clusterless_threshold_crossings"
    )
    assert unit_semantics_for_sorter("mountainsort5") == "sorted_units"
    assert unit_semantics_for_sorter("kilosort4") == "sorted_units"


def _pair(a_curation, a_unit, b_curation, b_unit, prob=0.9):
    """Build a MatchPair from ``(sorting_id, curation_id)`` side keys."""
    from spyglass.spikesorting.v2.matcher_protocol import MatchPair

    return MatchPair(
        session_a_sorting_id=a_curation[0],
        session_a_curation_id=a_curation[1],
        unit_a_id=a_unit,
        session_b_sorting_id=b_curation[0],
        session_b_curation_id=b_curation[1],
        unit_b_id=b_unit,
        match_probability=prob,
    )


# Two pinned member curations, one per member_index.
_CUR_A = ("sortA", 0)
_CUR_B = ("sortB", 0)
_MEMBER_INDEX = {_CUR_A: 0, _CUR_B: 1}


def test_canonicalize_orients_by_ascending_member_index():
    """A pair given B->A is stored A->B (ascending member_index)."""
    from spyglass.spikesorting.v2._matcher_graph import (
        canonicalize_match_pairs,
    )

    # Provide the pair "reversed" (member 1 first); canonicalization flips it.
    reversed_pair = _pair(_CUR_B, 7, _CUR_A, 3, prob=0.8)
    out = canonicalize_match_pairs([reversed_pair], _MEMBER_INDEX)
    assert len(out) == 1
    row = out[0]
    assert (row["session_a_sorting_id"], row["session_a_curation_id"]) == _CUR_A
    assert row["unit_a_id"] == 3
    assert (row["session_b_sorting_id"], row["session_b_curation_id"]) == _CUR_B
    assert row["unit_b_id"] == 7
    assert row["match_probability"] == pytest.approx(0.8)


def test_canonicalize_rejects_same_member_pair():
    """A pair whose two sides are the same member (incl. self-pairs) raises."""
    from spyglass.spikesorting.v2._matcher_graph import (
        canonicalize_match_pairs,
    )

    same_member = _pair(_CUR_A, 1, _CUR_A, 2)
    with pytest.raises(ValueError, match="same-member"):
        canonicalize_match_pairs([same_member], _MEMBER_INDEX)


def test_canonicalize_rejects_reversed_duplicate():
    """The same unit pair given in both orientations cannot both survive."""
    from spyglass.spikesorting.v2._matcher_graph import (
        canonicalize_match_pairs,
    )

    forward = _pair(_CUR_A, 3, _CUR_B, 7)
    backward = _pair(_CUR_B, 7, _CUR_A, 3)
    with pytest.raises(ValueError, match="duplicate"):
        canonicalize_match_pairs([forward, backward], _MEMBER_INDEX)


def test_canonicalize_rejects_unpinned_curation():
    """A pair referencing a curation not in the pinned member set raises."""
    from spyglass.spikesorting.v2._matcher_graph import (
        canonicalize_match_pairs,
    )

    stray = _pair(_CUR_A, 1, ("sortC", 0), 2)
    with pytest.raises(ValueError, match="not pinned|not one of"):
        canonicalize_match_pairs([stray], _MEMBER_INDEX)


def _nodes(*specs):
    """``("A", 1)`` -> ``("A", 0, 1)`` curated-unit node tuples."""
    return [(sorting_id, 0, unit_id) for sorting_id, unit_id in specs]


def test_derive_tracked_units_full_triangle_is_one_component():
    """Three sessions, all pairwise edges high -> one clique of all three."""
    from spyglass.spikesorting.v2._matcher_graph import derive_tracked_units

    a, b, c = _nodes(("A", 1), ("B", 1), ("C", 1))
    edges = [(a, b, 0.9), (b, c, 0.9), (a, c, 0.9)]
    tracked = derive_tracked_units(
        [a, b, c], edges, threshold=0.5, max_strict_nodes=100
    )
    assert len(tracked) == 1
    tu = tracked[0]
    assert set(tu["members"]) == {a, b, c}
    assert tu["n_sessions_observed"] == 3
    assert tu["policy_used"] == "strict"
    assert tu["median_match_probability"] == pytest.approx(0.9)


def test_derive_tracked_units_open_path_splits():
    """A<->B and B<->C high but A<->C low -> >=2 components, none with A and C."""
    from spyglass.spikesorting.v2._matcher_graph import derive_tracked_units

    a, b, c = _nodes(("A", 1), ("B", 1), ("C", 1))
    edges = [(a, b, 0.9), (b, c, 0.9), (a, c, 0.2)]
    tracked = derive_tracked_units(
        [a, b, c], edges, threshold=0.5, max_strict_nodes=100
    )
    assert len(tracked) >= 2
    for tu in tracked:
        members = set(tu["members"])
        assert not ({a, c} <= members), "A and C must not share a component"


def test_derive_tracked_units_partition_no_unit_in_two_groups():
    """Overlapping maximal cliques must not assign one unit to two identities.

    Edges A1-B1, A1-B2, A2-B1 form three overlapping size-2 cliques sharing
    A1/B1. Strict tracking is a partition, so every curated unit appears in
    exactly one tracked unit (greedy maximal-clique cover), not duplicated.
    """
    from spyglass.spikesorting.v2._matcher_graph import derive_tracked_units

    a1, a2 = ("A", 0, 1), ("A", 0, 2)
    b1, b2 = ("B", 0, 1), ("B", 0, 2)
    edges = [(a1, b1, 0.9), (a1, b2, 0.8), (a2, b1, 0.7)]
    tracked = derive_tracked_units(
        [a1, a2, b1, b2], edges, threshold=0.5, max_strict_nodes=100
    )
    all_members = [node for tu in tracked for node in tu["members"]]
    assert len(all_members) == len(set(all_members)), (
        "a curated unit was assigned to more than one tracked unit"
    )
    # Every node is covered exactly once (a true partition of the universe).
    assert set(all_members) == {a1, a2, b1, b2}
    # The strongest edge's clique survives whole.
    assert any(set(tu["members"]) == {a1, b1} for tu in tracked)


def test_derive_tracked_units_partition_prefers_stronger_clique():
    """Among equal-size overlapping cliques, the strongest edge wins the unit.

    A1-B1 (0.80) and A1-B2 (0.95) overlap on A1. The lexically-first clique is
    {A1,B1}, but the stronger match is {A1,B2}, so the probability-aware
    tie-break must keep {A1,B2} whole and drop B1 to a singleton.
    """
    from spyglass.spikesorting.v2._matcher_graph import derive_tracked_units

    a1 = ("A", 0, 1)
    b1, b2 = ("B", 0, 1), ("B", 0, 2)
    edges = [(a1, b1, 0.80), (a1, b2, 0.95)]
    tracked = derive_tracked_units(
        [a1, b1, b2], edges, threshold=0.5, max_strict_nodes=100
    )
    assert any(set(tu["members"]) == {a1, b2} for tu in tracked)
    assert any(tu["members"] == [b1] for tu in tracked)
    all_members = [node for tu in tracked for node in tu["members"]]
    assert len(all_members) == len(set(all_members))


def test_derive_tracked_units_unmatched_singleton():
    """A node with only sub-threshold edges is a singleton tracked unit."""
    from spyglass.spikesorting.v2._matcher_graph import derive_tracked_units

    a, b = _nodes(("A", 1), ("B", 1))
    edges = [(a, b, 0.2)]  # below threshold -> no edge in the graph
    tracked = derive_tracked_units(
        [a, b], edges, threshold=0.5, max_strict_nodes=100
    )
    assert len(tracked) == 2
    for tu in tracked:
        assert tu["n_sessions_observed"] == 1
        assert tu["median_match_probability"] is None
        assert tu["policy_used"] == "strict"


def test_derive_tracked_units_budget_cap_raises():
    """A node universe above ``max_strict_nodes`` refuses the clique search."""
    from spyglass.spikesorting.v2._matcher_graph import derive_tracked_units
    from spyglass.spikesorting.v2.exceptions import (
        TrackedUnitBudgetExceededError,
    )

    nodes = [("S", 0, u) for u in range(10)]
    with pytest.raises(TrackedUnitBudgetExceededError, match="10"):
        derive_tracked_units(nodes, [], threshold=0.5, max_strict_nodes=5)


def test_derive_tracked_units_under_cap_succeeds_strict():
    """Under the cap the search runs and every row is policy 'strict'."""
    from spyglass.spikesorting.v2._matcher_graph import derive_tracked_units

    nodes = [("S", 0, u) for u in range(5)]
    tracked = derive_tracked_units(
        nodes, [], threshold=0.5, max_strict_nodes=5
    )
    assert len(tracked) == 5
    assert all(tu["policy_used"] == "strict" for tu in tracked)


def test_derive_tracked_units_rejects_edge_outside_universe():
    """An edge endpoint absent from the node universe raises (networkx
    add_edge would otherwise silently create the node, smuggling a unit past
    the node budget and into the partition)."""
    from spyglass.spikesorting.v2._matcher_graph import derive_tracked_units

    a, b = _nodes(("A", 1), ("B", 1))
    stray = ("C", 0, 1)  # not in node_universe
    with pytest.raises(ValueError, match="absent from node_universe"):
        derive_tracked_units(
            [a, b], [(a, stray, 0.9)], threshold=0.5, max_strict_nodes=100
        )


def test_derive_tracked_units_equal_strength_tie_break_is_member_sorted():
    """Equal-size AND equal-strength overlapping cliques resolve by sorted members.

    {A1,B1} and {A1,B2} are both size 2 with identical 0.90 strength, so the final
    deterministic tie-break (sorted members) must keep the lexically-first clique
    {A1,B1} whole and drop B2 to a singleton -- locking the determinism that
    ``tracked_unit_id`` assignment relies on.
    """
    from spyglass.spikesorting.v2._matcher_graph import derive_tracked_units

    a1 = ("A", 0, 1)
    b1, b2 = ("B", 0, 1), ("B", 0, 2)
    edges = [(a1, b1, 0.90), (a1, b2, 0.90)]  # no b1-b2 edge -> two 2-cliques
    tracked = derive_tracked_units(
        [a1, b1, b2], edges, threshold=0.5, max_strict_nodes=100
    )
    assert any(set(tu["members"]) == {a1, b1} for tu in tracked)
    assert any(tu["members"] == [b2] for tu in tracked)
    # True partition: every node covered exactly once.
    all_members = [node for tu in tracked for node in tu["members"]]
    assert sorted(all_members) == sorted([a1, b1, b2])
    assert len(all_members) == len(set(all_members))


def test_chronological_member_order_sorts_by_date_then_index():
    """recording_date drives matcher feed order; member_index only breaks ties.

    Locks the drift-alignment invariant: UnitMatch aligns each session to the
    previous one, so members must reach the matcher in recording order even when
    their member_index order disagrees with their dates.
    """
    from spyglass.spikesorting.v2._matcher_graph import (
        chronological_member_order,
    )

    # member_index 0 recorded AFTER member_index 1 -> date must reorder them.
    plan = [
        {"member_index": 0, "recording_date": "2023-03-02T00:00:00+00:00"},
        {"member_index": 1, "recording_date": "2023-03-01T00:00:00+00:00"},
    ]
    assert [p["member_index"] for p in chronological_member_order(plan)] == [
        1,
        0,
    ]

    # Same-day members keep member_index order (tie-break).
    same_day = [
        {"member_index": 1, "recording_date": "2023-03-01T00:00:00+00:00"},
        {"member_index": 0, "recording_date": "2023-03-01T00:00:00+00:00"},
    ]
    assert [
        p["member_index"] for p in chronological_member_order(same_day)
    ] == [0, 1]


def test_curation_set_hash_order_independent_and_type_stable():
    """The curation-set hash is the content address of a UnitMatch selection.

    It must (a) ignore the order member choices are presented in (it sorts by
    member_index), (b) treat a ``uuid.UUID`` and its ``str`` form identically
    (so a fetched UUID and a passed str hash the same), and (c) change when any
    ``(member_index, sorting_id, curation_id)`` changes. ``insert_selection``
    (minting) and ``make_fetch`` (verifying) both call it, so it is the single
    source of truth for selection identity.
    """
    import uuid

    from spyglass.spikesorting.v2._matcher_graph import curation_set_hash

    sid0 = uuid.UUID("00000000-0000-0000-0000-000000000001")
    sid1 = uuid.UUID("00000000-0000-0000-0000-000000000002")
    base = curation_set_hash([(0, sid0, 1), (1, sid1, 2)])
    # (a) order-independent
    assert curation_set_hash([(1, sid1, 2), (0, sid0, 1)]) == base
    # (b) uuid and str hash identically
    assert curation_set_hash([(0, str(sid0), 1), (1, str(sid1), 2)]) == base
    # (c) any change flips the digest
    assert curation_set_hash([(0, sid0, 1), (1, sid1, 3)]) != base
    assert curation_set_hash([(0, sid0, 1)]) != base


def test_curation_set_hash_matches_legacy_inline_form():
    """The helper reproduces the byte-for-byte digest the old inline
    insert_selection code produced, so curation_set_hash values already stored
    on existing rows stay stable (idempotency is preserved across the refactor).
    """
    import hashlib
    import json
    import uuid

    from spyglass.spikesorting.v2._matcher_graph import curation_set_hash

    sid0 = uuid.uuid4()
    sid1 = uuid.uuid4()
    # Legacy inline form: [[member_index, str(sorting_id), curation_id], ...] in
    # ascending member_index order, hashed via json.dumps(sort_keys=True).
    legacy = hashlib.sha256(
        json.dumps(
            [[0, str(sid0), 1], [1, str(sid1), 2]], sort_keys=True
        ).encode("utf-8")
    ).hexdigest()
    assert curation_set_hash([(0, sid0, 1), (1, sid1, 2)]) == legacy


# --------------------------------------------------------------------------- #
# NWB pairs (de)serialization round-trip (pure I/O, no DB).                     #
# --------------------------------------------------------------------------- #


def test_pairs_nwb_round_trip_preserves_fdr_none(tmp_path):
    """Non-empty + empty pairs round-trip; a None fdr stores as NaN and reads
    back as None, while a real fdr survives and string ids stay str."""
    from datetime import datetime, timezone

    from pynwb import NWBHDF5IO, NWBFile

    from spyglass.spikesorting.v2._unitmatch_nwb import (
        build_pairs_table,
        read_pairs,
        write_pairs_table,
    )

    def _fresh_nwb(name):
        path = tmp_path / name
        nwbf = NWBFile(
            session_description="t",
            identifier=name,
            session_start_time=datetime(2020, 1, 1, tzinfo=timezone.utc),
        )
        with NWBHDF5IO(str(path), "w") as io:
            io.write(nwbf)
        return str(path)

    pairs = [
        {
            "session_a_sorting_id": "11111111-1111-1111-1111-111111111111",
            "session_a_curation_id": 0,
            "unit_a_id": 3,
            "session_b_sorting_id": "22222222-2222-2222-2222-222222222222",
            "session_b_curation_id": 1,
            "unit_b_id": 7,
            "match_probability": 0.85,
            "drift_estimate_um": 0.0,
            "fdr_estimate": None,
        },
        {
            "session_a_sorting_id": "11111111-1111-1111-1111-111111111111",
            "session_a_curation_id": 0,
            "unit_a_id": 4,
            "session_b_sorting_id": "22222222-2222-2222-2222-222222222222",
            "session_b_curation_id": 1,
            "unit_b_id": 9,
            "match_probability": 0.91,
            "drift_estimate_um": 0.0,
            "fdr_estimate": 0.05,
        },
    ]
    object_id = write_pairs_table(_fresh_nwb("pairs.nwb"), pairs)
    assert isinstance(object_id, str)
    back = read_pairs(
        str(tmp_path / "pairs.nwb"),
        object_id,
    )
    assert [row["pair_index"] for row in back] == [0, 1]
    assert back[0]["fdr_estimate"] is None
    assert back[1]["fdr_estimate"] == pytest.approx(0.05)
    assert isinstance(back[0]["session_a_sorting_id"], str)
    assert back[0]["unit_a_id"] == 3 and back[0]["unit_b_id"] == 7

    # Empty table writes and reads back empty (concrete dtypes, no inference).
    empty_oid = write_pairs_table(_fresh_nwb("empty.nwb"), [])
    assert read_pairs(str(tmp_path / "empty.nwb"), empty_oid) == []
    assert len(build_pairs_table([]).columns) > 0  # columns exist even when empty


# --------------------------------------------------------------------------- #
# MatcherProtocol is implementable by external code (no v2 internals touched).  #
# --------------------------------------------------------------------------- #


class _DummyMatcher:
    """A ten-line external matcher proving the protocol needs no v2 internals.

    It imports only the public ``matcher_protocol`` data types, matches the
    first unit of every session pair, and reports a fixed probability.
    """

    name = "dummy_external"

    def match(self, session_inputs, params):
        from spyglass.spikesorting.v2.matcher_protocol import MatchPair

        pairs = []
        for i, left in enumerate(session_inputs):
            for right in session_inputs[i + 1 :]:
                pairs.append(
                    MatchPair(
                        session_a_sorting_id=str(left.session_key["sorting_id"]),
                        session_a_curation_id=int(left.session_key["curation_id"]),
                        unit_a_id=0,
                        session_b_sorting_id=str(right.session_key["sorting_id"]),
                        session_b_curation_id=int(right.session_key["curation_id"]),
                        unit_b_id=0,
                        match_probability=float(params.get("prob", 0.99)),
                    )
                )
        return pairs


def test_external_matcher_satisfies_protocol_and_runs():
    """A dummy backend implements MatcherProtocol with only public imports."""
    from spyglass.spikesorting.v2.matcher_protocol import (
        MatcherProtocol,
        SessionMatcherInput,
    )

    matcher = _DummyMatcher()
    assert isinstance(matcher, MatcherProtocol)

    inputs = [
        SessionMatcherInput(
            session_key={"sorting_id": s, "curation_id": 0},
            waveform_dir=Path("/unused"),
            channel_positions_path=Path("/unused/cp.npy"),
        )
        for s in ("A", "B")
    ]
    pairs = matcher.match(inputs, {"prob": 0.9})
    assert len(pairs) == 1
    assert pairs[0].match_probability == pytest.approx(0.9)
    # Degenerate single-session case returns no pairs.
    assert matcher.match(inputs[:1], {}) == []


# --------------------------------------------------------------------------- #
# Table-wiring goals (database): registry-validating insert (goal 1), the       #
# explicit per-member selection (goals 4/5), the make() provenance recheck      #
# (goal 6), the degenerate single-session make (goal 2), and Pair FK integrity  #
# (goal 7). Built over two synthetic single-session sorts on the chronic        #
# minirec substrate.                                                            #
# --------------------------------------------------------------------------- #

_GROUP_NAME = "unitmatch_two_session"
_SOLO_NAME = "unitmatch_solo"


def _ensure_minirec_clusterless_params():
    """Insert the smoke clusterless sorter row (used only by the clusterless
    unit-semantics test, NOT the cross-session matching fixtures)."""
    from spyglass.spikesorting.v2.sorting import SorterParameters

    from tests.spikesorting.v2._smoke_constants import (
        SMOKE_CLUSTERLESS_PARAM_NAME,
        SMOKE_CLUSTERLESS_PARAMS,
    )

    SorterParameters.insert1(
        {
            "sorter": "clusterless_thresholder",
            "sorter_params_name": SMOKE_CLUSTERLESS_PARAM_NAME,
            "params": SMOKE_CLUSTERLESS_PARAMS,
        },
        skip_duplicates=True,
        allow_duplicate_params=True,
    )
    return SMOKE_CLUSTERLESS_PARAM_NAME


#: mountainsort5 params name for the planted minirec sorts. The sorts are
#: PLANTED (``_run_sorter`` monkeypatched), so this row only exists for the
#: selection FK + identity; the params are never run. A real sorter name gives
#: the sort ``sorted_units`` semantics -- the right substrate for cross-session
#: matching, which tracks sorted neurons, not clusterless threshold crossings.
_MINIREC_MS5_PARAMS = "franklab_30khz_ms5_2026_06"


def _ensure_minirec_ms5_params():
    """Ensure the shipped mountainsort5 params row exists (idempotent)."""
    from spyglass.spikesorting.v2.sorting import SorterParameters

    SorterParameters.insert_default()
    return _MINIREC_MS5_PARAMS


def _plant_single_unit(
    sorter,
    sorter_params,
    recording,
    sorting_id,
    *,
    job_kwargs=None,
    execution_params=None,
):
    """Plant one deterministic sorted unit spread across the recording.

    Enough spikes (and span) for the split-half UnitMatch bundle extraction and
    the display-analyzer template. Replaces a real sorter run via a
    ``Sorting._run_sorter`` monkeypatch -- fast and deterministic.
    """
    import numpy as np
    import spikeinterface as si

    n = recording.get_num_samples()
    samples = np.arange(1000, n - 1000, 5000, dtype=np.int64)
    labels = np.zeros(len(samples), dtype=np.int32)
    return si.NumpySorting.from_samples_and_labels(
        samples_list=[samples],
        labels_list=[labels],
        sampling_frequency=recording.get_sampling_frequency(),
    )


@pytest.fixture(scope="module")
def two_session_curated_group(chronic_2_session_minirec):
    """Two single-session sorts + curations under one SessionGroup.

    Builds on the package-scoped chronic minirec substrate: creates a two-member
    SessionGroup (and a one-member solo group for the degenerate case), and
    root-curates each same-day member with a PLANTED single-unit mountainsort5
    sort (``_run_sorter`` monkeypatched -- fast, deterministic, and
    ``sorted_units`` semantics, the right substrate for cross-session matching).
    Yields the group identities plus the per-member curation choices; tears down
    its own UnitMatch lineage, groups, sorts, and curations afterward.
    """
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.session_group import SessionGroup
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection
    from spyglass.spikesorting.v2.unit_matching import MatcherParameters

    from tests.spikesorting.v2._ingest_helpers import (
        clean_session_groups_for_owner,
        clear_curations_for,
    )

    sub = chronic_2_session_minirec
    owner = sub["owner"]
    members = sub["same_day_members"]
    recording_pks = sub["recording_pks"]

    sorter_params_name = _ensure_minirec_ms5_params()
    MatcherParameters.insert_default()

    # Start clean: drop any leftover groups (and their UnitMatch lineage) for
    # the owner from a prior interrupted run.
    clean_session_groups_for_owner(owner)
    SessionGroup.create_group(owner, _GROUP_NAME, members)
    SessionGroup.create_group(owner, _SOLO_NAME, members[:1])

    sort_pks = []
    choices = {}
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(Sorting, "_run_sorter", staticmethod(_plant_single_unit))
        for index, rec_pk in enumerate(recording_pks):
            sort_pk = SortingSelection.insert_selection(
                {
                    "recording_id": rec_pk["recording_id"],
                    "sorter": "mountainsort5",
                    "sorter_params_name": sorter_params_name,
                }
            )
            if not (Sorting & sort_pk):
                Sorting.populate(sort_pk, reserve_jobs=False)
            sort_pks.append(sort_pk)
            clear_curations_for(sort_pk)
            curation_key = CurationV2.insert_curation(
                sorting_key={"sorting_id": sort_pk["sorting_id"]}
            )
            choices[index] = {
                "sorting_id": curation_key["sorting_id"],
                "curation_id": curation_key["curation_id"],
            }
    finally:
        mp.undo()

    yield {
        "owner": owner,
        "group_name": _GROUP_NAME,
        "solo_name": _SOLO_NAME,
        "choices": choices,
        "members": members,
        "sort_pks": sort_pks,
    }

    clean_session_groups_for_owner(owner)
    for sort_pk in sort_pks:
        for mid in (SpikeSortingOutput.CurationV2 & sort_pk).fetch("merge_id"):
            (SpikeSortingOutput & {"merge_id": mid}).super_delete(warn=False)
        (CurationV2 & sort_pk).super_delete(warn=False)
        (Sorting & sort_pk).super_delete(warn=False)
        (SortingSelection & sort_pk).super_delete(warn=False)


@pytest.mark.slow
def test_matcher_parameters_rejects_unknown_matcher(dj_conn):
    """Goal 1: an unregistered matcher name raises at insert, before commit."""
    from spyglass.spikesorting.v2.exceptions import UnknownMatcherError
    from spyglass.spikesorting.v2.unit_matching import MatcherParameters

    with pytest.raises(UnknownMatcherError, match="register_matcher"):
        MatcherParameters().insert1(
            {
                "matcher_params_name": "typo_matcher",
                "matcher": "unitmatchh",  # typo
                "params": {},
            }
        )
    assert len(MatcherParameters & {"matcher_params_name": "typo_matcher"}) == 0


@pytest.mark.slow
def test_matcher_parameters_validates_params_per_matcher(dj_conn):
    """Goal 1: per-matcher Pydantic dispatch rejects an out-of-range param."""
    from pydantic import ValidationError

    from spyglass.spikesorting.v2.unit_matching import MatcherParameters

    with pytest.raises(ValidationError):
        MatcherParameters().insert1(
            {
                "matcher_params_name": "bad_params",
                "matcher": "unitmatch",
                "params": {"match_threshold": 5.0},  # > 1.0
            }
        )
    assert len(MatcherParameters & {"matcher_params_name": "bad_params"}) == 0


@pytest.mark.slow
def test_matcher_parameters_bulk_insert_is_validated(dj_conn):
    """Goal 1: the bulk insert() path (not just insert1) rejects a typo'd
    matcher -- it must not be a validation bypass."""
    from spyglass.spikesorting.v2.exceptions import UnknownMatcherError
    from spyglass.spikesorting.v2.unit_matching import MatcherParameters

    with pytest.raises(UnknownMatcherError):
        MatcherParameters().insert(
            [
                {
                    "matcher_params_name": "bulk_typo",
                    "matcher": "unitmatchh",  # typo
                    "params": {},
                }
            ]
        )
    assert len(MatcherParameters & {"matcher_params_name": "bulk_typo"}) == 0


@pytest.mark.slow
def test_matcher_parameters_duplicate_content_is_matcher_scoped(dj_conn):
    """Duplicate-content detection is scoped per matcher: a second matcher with
    byte-identical params is NOT a duplicate (it dispatches different code), but
    a second row for the SAME matcher with identical params is rejected."""
    from pydantic import BaseModel, ConfigDict

    from spyglass.spikesorting.v2 import matcher_protocol as mp
    from spyglass.spikesorting.v2._params.matcher import UnitMatchParamsSchema
    from spyglass.spikesorting.v2.exceptions import (
        DuplicateParameterContentError,
    )
    from spyglass.spikesorting.v2.matcher_protocol import register_matcher
    from spyglass.spikesorting.v2.unit_matching import MatcherParameters

    MatcherParameters.insert_default()
    default_params = (
        MatcherParameters & {"matcher_params_name": "unitmatch_default"}
    ).fetch1("params")

    class _OtherMatcher:
        name = "dup_scope_matcher"

        def match(self, session_inputs, params):
            return []

    class _OtherSchema(BaseModel):
        model_config = ConfigDict(extra="allow")

    saved_m = dict(mp._MATCHER_REGISTRY)
    saved_s = dict(mp._SCHEMA_REGISTRY)
    register_matcher(_OtherMatcher(), _OtherSchema)
    try:
        # Same params, DIFFERENT matcher -> not a duplicate.
        MatcherParameters().insert1(
            {
                "matcher_params_name": "other_same_params",
                "matcher": "dup_scope_matcher",
                "params": dict(default_params),
            }
        )
        assert len(
            MatcherParameters & {"matcher_params_name": "other_same_params"}
        )
        # Same params, SAME matcher -> rejected as a content duplicate.
        with pytest.raises(DuplicateParameterContentError):
            MatcherParameters().insert1(
                {
                    "matcher_params_name": "unitmatch_dup",
                    "matcher": "unitmatch",
                    "params": UnitMatchParamsSchema().model_dump(),
                }
            )
    finally:
        (
            MatcherParameters
            & {"matcher_params_name": "other_same_params"}
        ).super_delete(warn=False)
        mp._MATCHER_REGISTRY.clear()
        mp._MATCHER_REGISTRY.update(saved_m)
        mp._SCHEMA_REGISTRY.clear()
        mp._SCHEMA_REGISTRY.update(saved_s)


@pytest.mark.slow
def test_insert_selection_idempotent_and_hash_sensitive(
    two_session_curated_group,
):
    """Goal 4: identical inputs return the same id; a changed curation differs."""
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.unit_matching import UnitMatchSelection

    grp = two_session_curated_group
    pk1 = UnitMatchSelection.insert_selection(
        grp["owner"], grp["group_name"], "unitmatch_default", grp["choices"]
    )
    pk2 = UnitMatchSelection.insert_selection(
        grp["owner"],
        grp["group_name"],
        "unitmatch_default",
        dict(grp["choices"]),
    )
    assert pk1 == pk2

    # A child curation on member 0 changes that member's choice -> new hash/id.
    member0 = grp["choices"][0]
    child = CurationV2.insert_curation(
        sorting_key={"sorting_id": member0["sorting_id"]},
        parent_curation_id=member0["curation_id"],
    )
    changed = dict(grp["choices"])
    changed[0] = {
        "sorting_id": child["sorting_id"],
        "curation_id": child["curation_id"],
    }
    pk3 = UnitMatchSelection.insert_selection(
        grp["owner"], grp["group_name"], "unitmatch_default", changed
    )
    assert pk3["unitmatch_id"] != pk1["unitmatch_id"]


@pytest.mark.slow
def test_unitmatch_selection_accepts_curation_evaluation_committed_children(
    two_session_curated_group, curation_evaluation_defaults
):
    """Final ``CurationEvaluation`` children are valid UnitMatch inputs.

    This pins the downstream object users should select after curation:
    evaluate each member curation, accept the label verdict with
    ``use_evaluation_labels``, then pin those committed children in
    ``UnitMatchSelection``. ``make_fetch`` must see the accepted child
    curation ids and their frozen matchable unit sets.
    """
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.metric_curation import (
        CurationEvaluation,
        CurationEvaluationSelection,
    )
    from spyglass.spikesorting.v2.unit_matching import (
        UnitMatch,
        UnitMatchSelection,
    )

    grp = two_session_curated_group
    accepted_choices = {}
    for member_index, choice in grp["choices"].items():
        sel = CurationEvaluationSelection.insert_selection(
            {
                **choice,
                "metric_params_name": "minimal",
                "auto_curation_rules_name": "none",
            }
        )
        CurationEvaluation.populate(sel, reserve_jobs=False)
        child = CurationEvaluation().use_evaluation_labels(sel)
        assert CurationV2.is_committed_curation(child)
        assert (CurationV2 & child).fetch1("curation_source") == (
            "curation_evaluation"
        )
        accepted_choices[int(member_index)] = {
            "sorting_id": child["sorting_id"],
            "curation_id": child["curation_id"],
        }

    pk = UnitMatchSelection.insert_selection(
        grp["owner"],
        grp["group_name"],
        "unitmatch_default",
        accepted_choices,
    )
    try:
        fetched = UnitMatch().make_fetch(pk)
        plan_by_member = {
            int(plan["member_index"]): plan for plan in fetched.member_plan
        }
        assert set(plan_by_member) == set(accepted_choices)
        for member_index, choice in accepted_choices.items():
            plan = plan_by_member[member_index]
            assert plan["sorting_id"] == str(choice["sorting_id"])
            assert int(plan["curation_id"]) == int(choice["curation_id"])
            expected_matchable = [
                int(u) for u in CurationV2().get_matchable_unit_ids(choice)
            ]
            assert plan["matchable_unit_ids"] == expected_matchable
    finally:
        (UnitMatchSelection & pk).super_delete(warn=False)


def _plant_two_unit_sort_on_first_member(grp):
    """Plant a deterministic 2-unit mountainsort5 sort on member 0's recording.

    The chronic fixture's sort yields a single unit; a real merge (or a real
    proposed merge) needs >=2. A distinct params name gives a distinct
    content-addressed sort on the SAME recording, so the planted sort shares
    member 0's identity without colliding with the fixture's single-unit sort.
    The sort is PLANTED (``_run_sorter`` monkeypatched), so the params are never
    run; a real sorter name keeps the units ``sorted_units``. Returns
    ``(two_unit_sort_key, sorted_unit_ids, sorter_params_name)``; the caller owns
    teardown (clear_curations_for + dropping the sort/params).
    """
    import numpy as np
    import spikeinterface as si

    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        Sorting,
        SortingSelection,
    )

    rec_id = (
        SortingSelection.RecordingSource & grp["sort_pks"][0]
    ).fetch1("recording_id")
    two_unit_params = "minirec_ms5_two_unit"
    default_ms5 = (
        SorterParameters
        & {"sorter": "mountainsort5", "sorter_params_name": _MINIREC_MS5_PARAMS}
    ).fetch1("params")
    SorterParameters.insert1(
        {
            "sorter": "mountainsort5",
            "sorter_params_name": two_unit_params,
            "params": dict(default_ms5),
        },
        skip_duplicates=True,
        allow_duplicate_params=True,
    )

    def _plant(
        sorter,
        sorter_params,
        recording,
        sorting_id,
        *,
        job_kwargs=None,
        execution_params=None,
    ):
        samples = np.array([500, 1000, 1500, 600, 1100, 1600], dtype=np.int64)
        labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
        return si.NumpySorting.from_samples_and_labels(
            samples_list=[samples],
            labels_list=[labels],
            sampling_frequency=recording.get_sampling_frequency(),
        )

    two_unit_sort = SortingSelection.insert_selection(
        {
            "recording_id": rec_id,
            "sorter": "mountainsort5",
            "sorter_params_name": two_unit_params,
        }
    )
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(Sorting, "_run_sorter", staticmethod(_plant))
        if not (Sorting & two_unit_sort):
            Sorting.populate(two_unit_sort, reserve_jobs=False)
    finally:
        mp.undo()
    unit_ids = sorted(
        int(u) for u in (Sorting.Unit & two_unit_sort).fetch("unit_id")
    )
    assert len(unit_ids) >= 2, "planted sort must yield >=2 units"
    return two_unit_sort, unit_ids, two_unit_params


@pytest.mark.slow
def test_unitmatch_rejects_curation_evaluation_preview_child(
    two_session_curated_group, curation_evaluation_defaults
):
    """A ``CurationEvaluation`` preview child is not a UnitMatch input.

    ``create_preview_curation`` records proposed merges without applying them;
    matching that draft would feed oversplit units into the matcher. Both
    ``UnitMatchSelection.insert_selection`` and ``UnitMatch.make_fetch`` must
    reject it. The fixture's planted sort yields a single unit, so plant a
    2-unit sort on member 0's recording to host a real proposed merge.
    """
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.exceptions import (
        UnitMatchSelectionIntegrityError,
    )
    from spyglass.spikesorting.v2.metric_curation import (
        CurationEvaluation,
        CurationEvaluationSelection,
    )
    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        Sorting,
        SortingSelection,
    )
    from spyglass.spikesorting.v2.unit_matching import (
        UnitMatch,
        UnitMatchSelection,
    )
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    grp = two_session_curated_group
    two_unit_sort, unit_ids, two_unit_params = (
        _plant_two_unit_sort_on_first_member(grp)
    )

    clear_curations_for(two_unit_sort)
    try:
        root0 = CurationV2.insert_curation(
            sorting_key={"sorting_id": two_unit_sort["sorting_id"]}
        )
        eval_sel = CurationEvaluationSelection.insert_selection(
            {
                **root0,
                "metric_params_name": "minimal",
                "auto_curation_rules_name": "none",
            }
        )
        CurationEvaluation.populate(eval_sel, reserve_jobs=False)
        preview0 = CurationEvaluation().create_preview_curation(
            eval_sel,
            merge_groups=[[unit_ids[0], unit_ids[1]]],
            labels={},
        )
        root_choice = {
            "sorting_id": root0["sorting_id"],
            "curation_id": root0["curation_id"],
        }
        preview_choice = {
            "sorting_id": preview0["sorting_id"],
            "curation_id": preview0["curation_id"],
        }

        # Site 1: insert_selection rejects a pinned preview member.
        with pytest.raises(ValueError, match="apply_merge=False"):
            UnitMatchSelection.insert_selection(
                grp["owner"],
                grp["group_name"],
                "unitmatch_default",
                {0: preview_choice, 1: grp["choices"][1]},
            )

        # Site 2: a direct-insert selection (bypassing insert_selection) is
        # rejected when UnitMatch.make_fetch validates. Build a valid selection
        # with the member-0 ROOT, then swap member 0's part to the preview
        # curation. make_fetch's _validate_member_curations runs before the
        # curation_set_hash check, so the preview guard fires on the swap.
        pk = UnitMatchSelection.insert_selection(
            grp["owner"],
            grp["group_name"],
            "unitmatch_default",
            {0: root_choice, 1: grp["choices"][1]},
        )
        master_row = (UnitMatchSelection & pk).fetch1()
        parts = (UnitMatchSelection.MemberCuration & pk).fetch(
            as_dict=True, order_by="member_index"
        )
        (UnitMatchSelection & pk).super_delete(warn=False)
        UnitMatchSelection.insert1(master_row, allow_direct_insert=True)
        for part in parts:
            if int(part["member_index"]) == 0:
                part["curation_id"] = preview0["curation_id"]
        UnitMatchSelection.MemberCuration.insert(parts, allow_direct_insert=True)
        try:
            with pytest.raises(
                UnitMatchSelectionIntegrityError, match="apply_merge=False"
            ):
                UnitMatch().make_fetch(pk)
        finally:
            (UnitMatchSelection & pk).super_delete(warn=False)
    finally:
        clear_curations_for(two_unit_sort)
        (Sorting & two_unit_sort).super_delete(warn=False)
        (SortingSelection & two_unit_sort).super_delete(warn=False)
        (
            SorterParameters & {"sorter_params_name": two_unit_params}
        ).super_delete(warn=False)


@pytest.mark.slow
def test_unitmatch_accepts_committed_merged_child_member(
    two_session_curated_group,
):
    """A committed APPLIED-MERGE child is a valid UnitMatch member.

    Coverage otherwise pins label-only committed children (and rejects
    previews); this pins the production merged shape: plant a 2-unit sort on
    member 0's recording, COMMIT a merge into a child, pin it, and assert
    make_fetch's member plan carries the child's MERGED unit set (one unit), not
    the two pre-merge ids.
    """
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        Sorting,
        SortingSelection,
    )
    from spyglass.spikesorting.v2.unit_matching import (
        UnitMatch,
        UnitMatchSelection,
    )
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    initialize_v2_defaults()
    grp = two_session_curated_group
    two_unit_sort, unit_ids, two_unit_params = (
        _plant_two_unit_sort_on_first_member(grp)
    )

    clear_curations_for(two_unit_sort)
    try:
        merged0 = CurationV2.create_merged_curation(
            {"sorting_id": two_unit_sort["sorting_id"]},
            merge_groups=[[unit_ids[0], unit_ids[1]]],
        )
        assert CurationV2.is_committed_curation(merged0)
        merged_choice = {
            "sorting_id": merged0["sorting_id"],
            "curation_id": merged0["curation_id"],
        }
        expected = [
            int(u) for u in CurationV2().get_matchable_unit_ids(merged_choice)
        ]
        assert len(expected) == 1  # two pre-merge units -> one merged unit

        pk = UnitMatchSelection.insert_selection(
            grp["owner"],
            grp["group_name"],
            "unitmatch_default",
            {0: merged_choice, 1: grp["choices"][1]},
        )
        try:
            fetched = UnitMatch().make_fetch(pk)
            plan0 = next(
                p for p in fetched.member_plan if int(p["member_index"]) == 0
            )
            assert int(plan0["curation_id"]) == int(merged0["curation_id"])
            assert plan0["matchable_unit_ids"] == expected
        finally:
            (UnitMatchSelection & pk).super_delete(warn=False)
    finally:
        clear_curations_for(two_unit_sort)
        (Sorting & two_unit_sort).super_delete(warn=False)
        (SortingSelection & two_unit_sort).super_delete(warn=False)
        (
            SorterParameters & {"sorter_params_name": two_unit_params}
        ).super_delete(warn=False)


@pytest.mark.slow
def test_insert_selection_rejects_forged_master_with_mismatched_parts(
    two_session_curated_group,
):
    """A deterministic master whose MemberCuration parts no longer realize its
    curation_set_hash (here: a part was dropped) is rejected by
    insert_selection's _find_existing_pk -- caught up front, not returned as a
    'valid' PK that only UnitMatch.make_fetch would later reject.
    """
    from spyglass.spikesorting.v2.exceptions import SchemaBypassError
    from spyglass.spikesorting.v2.unit_matching import UnitMatchSelection

    grp = two_session_curated_group
    pk = UnitMatchSelection.insert_selection(
        grp["owner"], grp["group_name"], "unitmatch_default", grp["choices"]
    )
    # Tear down the consistent selection, then re-insert ONLY the master row
    # (same deterministic id + curation_set_hash, no MemberCuration parts) -- a
    # raw-insert orphan that DataJoint parts cannot be deleted into directly.
    master_row = (UnitMatchSelection & pk).fetch1()
    (UnitMatchSelection & pk).super_delete(warn=False)
    UnitMatchSelection.insert1(master_row, allow_direct_insert=True)
    try:
        with pytest.raises(SchemaBypassError, match="curation_set_hash"):
            UnitMatchSelection.insert_selection(
                grp["owner"],
                grp["group_name"],
                "unitmatch_default",
                grp["choices"],
            )
    finally:
        (UnitMatchSelection & pk).super_delete(warn=False)


@pytest.mark.slow
def test_insert_selection_rejects_foreign_group_parts_with_matching_hash(
    two_session_curated_group,
):
    """A forged master whose parts hash to the right choices but belong to a
    DIFFERENT SessionGroup is rejected by insert_selection. curation_set_hash
    folds only (member_index, sorting_id, curation_id), so the foreign-group
    check -- not the hash -- is what catches copied cross-group parts. Distinct
    from the missing-parts case above.
    """
    from spyglass.spikesorting.v2.exceptions import SchemaBypassError
    from spyglass.spikesorting.v2.session_group import SessionGroup
    from spyglass.spikesorting.v2.unit_matching import UnitMatchSelection

    grp = two_session_curated_group
    solo_key = {
        "session_group_owner": grp["owner"],
        "session_group_name": grp["solo_name"],
    }
    solo_member_indexes = {
        int(m["member_index"])
        for m in (SessionGroup.Member & solo_key).fetch(as_dict=True)
    }
    if 0 not in solo_member_indexes:
        pytest.skip("solo group has no member_index 0 to borrow for the forgery")

    pk = UnitMatchSelection.insert_selection(
        grp["owner"], grp["group_name"], "unitmatch_default", grp["choices"]
    )
    master_row = (UnitMatchSelection & pk).fetch1()
    real_parts = (UnitMatchSelection.MemberCuration & pk).fetch(
        as_dict=True, order_by="member_index"
    )
    # Re-forge: same master + same (member_index, sorting_id, curation_id) per
    # part, but point member_index 0 at the SOLO group (its FK to
    # SessionGroup.Member still resolves; its curation FK is unchanged). The
    # recomputed hash equals the real one, so only the foreign-group check fires.
    (UnitMatchSelection & pk).super_delete(warn=False)
    UnitMatchSelection.insert1(master_row, allow_direct_insert=True)
    forged_parts = []
    for part in real_parts:
        forged = dict(part)
        if int(part["member_index"]) == 0:
            forged["session_group_name"] = grp["solo_name"]
        forged_parts.append(forged)
    UnitMatchSelection.MemberCuration.insert(
        forged_parts, allow_direct_insert=True
    )
    try:
        with pytest.raises(SchemaBypassError, match="different SessionGroup"):
            UnitMatchSelection.insert_selection(
                grp["owner"],
                grp["group_name"],
                "unitmatch_default",
                grp["choices"],
            )
    finally:
        (UnitMatchSelection & pk).super_delete(warn=False)


@pytest.mark.slow
def test_insert_selection_rejects_wrong_member_curation(
    two_session_curated_group,
):
    """Goal 5: a curation from another member is rejected, atomically."""
    from spyglass.spikesorting.v2.unit_matching import UnitMatchSelection

    grp = two_session_curated_group
    before = len(UnitMatchSelection())
    # Pin each member to the OTHER member's curation (wrong provenance).
    swapped = {0: grp["choices"][1], 1: grp["choices"][0]}
    with pytest.raises(ValueError, match="belongs to|does not exist"):
        UnitMatchSelection.insert_selection(
            grp["owner"], grp["group_name"], "unitmatch_default", swapped
        )
    # Atomic: no master row was created for the rejected selection.
    assert len(UnitMatchSelection()) == before


@pytest.mark.slow
def test_insert_selection_rejects_incomplete_coverage(
    two_session_curated_group,
):
    """Goal 5: missing a member's choice is rejected before any insert."""
    from spyglass.spikesorting.v2.unit_matching import UnitMatchSelection

    grp = two_session_curated_group
    with pytest.raises(ValueError, match="exactly cover|Missing"):
        UnitMatchSelection.insert_selection(
            grp["owner"],
            grp["group_name"],
            "unitmatch_default",
            {0: grp["choices"][0]},  # member 1 missing
        )


@pytest.mark.slow
def test_make_rechecks_member_provenance(two_session_curated_group):
    """Goal 6: a direct-inserted wrong-member selection fails make() before
    any matcher input is extracted; no UnitMatch / Pair rows are created."""
    import uuid

    from spyglass.spikesorting.v2.exceptions import (
        UnitMatchSelectionIntegrityError,
    )
    from spyglass.spikesorting.v2.session_group import SessionGroup
    from spyglass.spikesorting.v2.unit_matching import (
        UnitMatch,
        UnitMatchSelection,
    )

    grp = two_session_curated_group
    group_key = {
        "session_group_owner": grp["owner"],
        "session_group_name": grp["group_name"],
    }
    member_rows = (SessionGroup.Member & group_key).fetch(
        as_dict=True, order_by="member_index"
    )
    unitmatch_id = uuid.uuid4()
    # Direct-insert (bypassing insert_selection) a provenance-invalid selection:
    # pin each member to the OTHER member's curation.
    UnitMatchSelection.insert1(
        {
            "unitmatch_id": unitmatch_id,
            **group_key,
            "matcher_params_name": "unitmatch_default",
            "curation_set_hash": "0" * 64,
        },
        # UnitMatchSelection now guards direct inserts; this test deliberately
        # bypasses insert_selection to forge a provenance-invalid master.
        allow_direct_insert=True,
    )
    UnitMatchSelection.MemberCuration.insert(
        [
            {
                "unitmatch_id": unitmatch_id,
                **group_key,
                "member_index": int(member_rows[0]["member_index"]),
                "sorting_id": grp["choices"][1]["sorting_id"],
                "curation_id": grp["choices"][1]["curation_id"],
            },
            {
                "unitmatch_id": unitmatch_id,
                **group_key,
                "member_index": int(member_rows[1]["member_index"]),
                "sorting_id": grp["choices"][0]["sorting_id"],
                "curation_id": grp["choices"][0]["curation_id"],
            },
        ],
        allow_direct_insert=True,
    )
    try:
        with pytest.raises(UnitMatchSelectionIntegrityError):
            UnitMatch.populate({"unitmatch_id": unitmatch_id}, reserve_jobs=False)
        assert len(UnitMatch & {"unitmatch_id": unitmatch_id}) == 0
        assert len(UnitMatch.Pair & {"unitmatch_id": unitmatch_id}) == 0
    finally:
        (UnitMatchSelection & {"unitmatch_id": unitmatch_id}).super_delete(
            warn=False
        )


@pytest.mark.slow
def test_make_rejects_curation_set_hash_mismatch(two_session_curated_group):
    """A raw-insert master whose stored curation_set_hash disagrees with its
    (otherwise valid, owned, complete) MemberCuration rows is rejected by
    make_fetch -- the gap the ownership/coverage checks alone do NOT catch.
    Without the recompute, such a master could claim one curation set in its
    hash while matching on the units pinned by its parts.
    """
    import uuid

    from spyglass.spikesorting.v2.exceptions import (
        UnitMatchSelectionIntegrityError,
    )
    from spyglass.spikesorting.v2.session_group import SessionGroup
    from spyglass.spikesorting.v2.unit_matching import (
        UnitMatch,
        UnitMatchSelection,
    )

    grp = two_session_curated_group
    group_key = {
        "session_group_owner": grp["owner"],
        "session_group_name": grp["group_name"],
    }
    member_rows = (SessionGroup.Member & group_key).fetch(
        as_dict=True, order_by="member_index"
    )
    unitmatch_id = uuid.uuid4()
    # Pin each member to its OWN correct curation (ownership + coverage pass),
    # but store a bogus curation_set_hash so ONLY the recompute can catch it.
    UnitMatchSelection.insert1(
        {
            "unitmatch_id": unitmatch_id,
            **group_key,
            "matcher_params_name": "unitmatch_default",
            "curation_set_hash": "0" * 64,
        },
        allow_direct_insert=True,
    )
    UnitMatchSelection.MemberCuration.insert(
        [
            {
                "unitmatch_id": unitmatch_id,
                **group_key,
                "member_index": int(member["member_index"]),
                "sorting_id": grp["choices"][int(member["member_index"])][
                    "sorting_id"
                ],
                "curation_id": grp["choices"][int(member["member_index"])][
                    "curation_id"
                ],
            }
            for member in member_rows
        ],
        allow_direct_insert=True,
    )
    try:
        with pytest.raises(
            UnitMatchSelectionIntegrityError, match="curation_set_hash"
        ):
            UnitMatch.populate(
                {"unitmatch_id": unitmatch_id}, reserve_jobs=False
            )
        assert len(UnitMatch & {"unitmatch_id": unitmatch_id}) == 0
    finally:
        (UnitMatchSelection & {"unitmatch_id": unitmatch_id}).super_delete(
            warn=False
        )


@pytest.mark.slow
def test_make_rejects_foreign_group_member_curation(two_session_curated_group):
    """Goal 6: a MemberCuration row from a DIFFERENT SessionGroup attached under
    the same unitmatch_id (its group fields are not unified with the master) is
    rejected by make() before the member-index collapse can hide it."""
    import uuid

    from spyglass.spikesorting.v2.exceptions import (
        UnitMatchSelectionIntegrityError,
    )
    from spyglass.spikesorting.v2.unit_matching import (
        UnitMatch,
        UnitMatchSelection,
    )

    grp = two_session_curated_group
    group_key = {
        "session_group_owner": grp["owner"],
        "session_group_name": grp["group_name"],
    }
    unitmatch_id = uuid.uuid4()
    UnitMatchSelection.insert1(
        {
            "unitmatch_id": unitmatch_id,
            **group_key,
            "matcher_params_name": "unitmatch_default",
            "curation_set_hash": "0" * 64,
        },
        # UnitMatchSelection now guards direct inserts; this test deliberately
        # bypasses insert_selection to forge a provenance-invalid master.
        allow_direct_insert=True,
    )
    # Attach a member_index=0 row from the SOLO group (a different
    # session_group_name) under this unitmatch_id -- schema-valid (its
    # SessionGroup.Member FK resolves), but foreign to the master's group.
    UnitMatchSelection.MemberCuration.insert1(
        {
            "unitmatch_id": unitmatch_id,
            "session_group_owner": grp["owner"],
            "session_group_name": grp["solo_name"],
            "member_index": 0,
            "sorting_id": grp["choices"][0]["sorting_id"],
            "curation_id": grp["choices"][0]["curation_id"],
        },
        allow_direct_insert=True,
    )
    try:
        with pytest.raises(
            UnitMatchSelectionIntegrityError, match="belongs to SessionGroup"
        ):
            UnitMatch.populate(
                {"unitmatch_id": unitmatch_id}, reserve_jobs=False
            )
        assert len(UnitMatch & {"unitmatch_id": unitmatch_id}) == 0
    finally:
        (UnitMatchSelection & {"unitmatch_id": unitmatch_id}).super_delete(
            warn=False
        )


@pytest.mark.slow
def test_degenerate_single_session_zero_pairs(two_session_curated_group):
    """Goal 2: a one-member group produces zero pairs (no matcher call)."""
    from spyglass.spikesorting.v2.unit_matching import (
        UnitMatch,
        UnitMatchSelection,
    )

    grp = two_session_curated_group
    pk = UnitMatchSelection.insert_selection(
        grp["owner"], grp["solo_name"], "unitmatch_default", {0: grp["choices"][0]}
    )
    UnitMatch.populate(pk, reserve_jobs=False)
    row = (UnitMatch & pk).fetch1()
    assert row["n_pairs"] == 0
    assert len(UnitMatch.Pair & pk) == 0
    # The empty NWB pairs table round-trips to an empty DataFrame.
    assert len(UnitMatch().get_pairs(pk)) == 0


@pytest.mark.usefixtures("dj_conn")
def test_unitmatch_computed_matches_make_insert_signature():
    """The tri-part dispatch splats ``UnitMatchComputed`` POSITIONALLY into
    ``make_insert(key, *computed)``, so the NamedTuple field order is a wire
    contract. The three provenance fields were appended to both -- a misalignment
    would silently mis-bind the str-adjacent slots without a TypeError."""
    import inspect

    from spyglass.spikesorting.v2.unit_matching import (
        UnitMatch,
        UnitMatchComputed,
    )

    params = list(inspect.signature(UnitMatch.make_insert).parameters)
    assert params[:2] == ["self", "key"]
    assert tuple(params[2:]) == UnitMatchComputed._fields


@pytest.mark.slow
def test_unitmatch_records_backend_version(two_session_curated_group):
    """A populated UnitMatch row records the producer provenance: the SI version,
    the resolved backend module path, and the backend package version -- resolved
    from the registry even on the degenerate single-session path."""
    import importlib.metadata

    import spikeinterface as si

    from spyglass.spikesorting.v2.unit_matching import (
        UnitMatch,
        UnitMatchSelection,
    )

    grp = two_session_curated_group
    pk = UnitMatchSelection.insert_selection(
        grp["owner"],
        grp["solo_name"],
        "unitmatch_default",
        {0: grp["choices"][0]},
    )
    UnitMatch.populate(pk, reserve_jobs=False)
    row = (UnitMatch & pk).fetch1()
    assert row["spikeinterface_version"] == si.__version__
    assert row["matcher_backend"] == (
        "spyglass.spikesorting.v2._unitmatch_backend"
    )
    assert row["matcher_backend_version"] == importlib.metadata.version(
        "unitmatchpy"
    )


@pytest.mark.slow
def test_clusterless_unit_semantics_derived_and_warned(
    chronic_2_session_minirec, caplog
):
    """A clusterless sort's units are threshold-crossings, not sorted neurons:
    CurationV2.get_unit_semantics derives that from the sorter, and
    UnitMatchSelection.insert_selection warns that matching them across sessions
    is degenerate -- a consuming surface honoring the semantics, not just a
    column. Built standalone (a real clusterless sort + a solo group) because
    the matching fixtures deliberately use a sorted-units sorter, exactly so
    they do NOT match threshold crossings.
    """
    import logging

    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.session_group import SessionGroup
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection
    from spyglass.spikesorting.v2.unit_matching import (
        MatcherParameters,
        UnitMatchSelection,
        _warn_clusterless_match_once,
    )
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    sub = chronic_2_session_minirec
    owner = sub["owner"]
    member = sub["same_day_members"][0]
    rec_pk = sub["recording_pks"][0]
    group_name = "unitmatch_clusterless_warn"

    MatcherParameters.insert_default()
    params_name = _ensure_minirec_clusterless_params()
    sort_pk = SortingSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "sorter": "clusterless_thresholder",
            "sorter_params_name": params_name,
        }
    )
    group_key = {
        "session_group_owner": owner,
        "session_group_name": group_name,
    }
    try:
        if not (Sorting & sort_pk):
            Sorting.populate(sort_pk, reserve_jobs=False)
        clear_curations_for(sort_pk)
        curation = CurationV2.insert_curation(
            sorting_key={"sorting_id": sort_pk["sorting_id"]}
        )
        if not (SessionGroup & group_key):
            SessionGroup.create_group(owner, group_name, [member])

        assert (
            CurationV2.get_unit_semantics(
                {"sorting_id": sort_pk["sorting_id"]}
            )
            == "clusterless_threshold_crossings"
        )

        _warn_clusterless_match_once.cache_clear()
        with caplog.at_level(logging.WARNING, logger="spyglass"):
            UnitMatchSelection.insert_selection(
                owner,
                group_name,
                "unitmatch_default",
                {
                    0: {
                        "sorting_id": curation["sorting_id"],
                        "curation_id": curation["curation_id"],
                    }
                },
            )
        assert any(
            "threshold" in r.getMessage().lower()
            and "neuron" in r.getMessage().lower()
            for r in caplog.records
        )
    finally:
        if SessionGroup & group_key:
            (SessionGroup & group_key).super_delete(warn=False)
        clear_curations_for(sort_pk)
        (Sorting & sort_pk).super_delete(warn=False)
        (SortingSelection & sort_pk).super_delete(warn=False)


@pytest.mark.slow
def test_raw_pair_insert_rejects_unpinned_endpoint(two_session_curated_group):
    """Goal 7: a raw Pair referencing a unit absent from the pinned curation is
    rejected. The validated ``Pair.insert`` guard now fronts the
    DataJoint foreign key: the bogus pair's second endpoint pins member 1's
    curation, which is NOT in the solo selection's ``MemberCuration``, so the
    guard rejects it (before the FK, which remains as defense-in-depth)."""
    from spyglass.spikesorting.v2.exceptions import (
        UnitMatchPairIntegrityError,
    )
    from spyglass.spikesorting.v2.unit_matching import (
        UnitMatch,
        UnitMatchSelection,
    )

    grp = two_session_curated_group
    pk = UnitMatchSelection.insert_selection(
        grp["owner"], grp["solo_name"], "unitmatch_default", {0: grp["choices"][0]}
    )
    UnitMatch.populate(pk, reserve_jobs=False)
    side_a, side_b = grp["choices"][0], grp["choices"][1]
    bogus = {
        **pk,
        "pair_index": 999,
        "session_a_sorting_id": side_a["sorting_id"],
        "session_a_curation_id": side_a["curation_id"],
        "unit_a_id": 10**9,  # no such curated unit
        "session_b_sorting_id": side_b["sorting_id"],  # member 1: not pinned here
        "session_b_curation_id": side_b["curation_id"],
        "unit_b_id": 10**9,
        "match_probability": 0.9,
    }
    with pytest.raises(UnitMatchPairIntegrityError, match="not a pinned"):
        UnitMatch.Pair.insert1(bogus, allow_direct_insert=True)


@pytest.mark.slow
def test_tracked_unit_make_seeds_singletons(two_session_curated_group):
    """``TrackedUnit.make`` seeds the node universe from the curated units and,
    with no matches (the solo group), emits one strict singleton per matchable
    unit -- exercising the DB wiring of the pure clique derivation."""
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.unit_matching import (
        TrackedUnit,
        UnitMatch,
        UnitMatchSelection,
    )

    grp = two_session_curated_group
    pk = UnitMatchSelection.insert_selection(
        grp["owner"], grp["solo_name"], "unitmatch_default", {0: grp["choices"][0]}
    )
    UnitMatch.populate(pk, reserve_jobs=False)
    TrackedUnit.populate(pk, reserve_jobs=False)

    n_matchable = len(
        CurationV2().get_matchable_unit_ids(grp["choices"][0])
    )
    tracked = (TrackedUnit & pk).fetch(as_dict=True)
    assert len(tracked) == n_matchable
    for row in tracked:
        # No pairs -> every unit is its own singleton tracked unit.
        assert row["n_sessions_observed"] == 1
        assert row["median_match_probability"] is None
        assert row["policy_used"] == "strict"
    # Member rows reference the pinned curated units (FK-validated).
    assert len(TrackedUnit.Member & pk) == n_matchable


def _install_fixture_pairer(
    monkeypatch,
    *,
    matcher_name: str,
    matcher_params_name: str,
    pairs: list[list[int]],
    probability: float = 0.99,
    seen_unit_ids: list[list[int]] | None = None,
):
    """Register a lightweight matcher and stub bundle extraction for DB tests."""
    from pydantic import BaseModel, ConfigDict, Field

    from spyglass.spikesorting.v2 import _unitmatch_backend
    from spyglass.spikesorting.v2 import matcher_protocol as mp
    from spyglass.spikesorting.v2.matcher_protocol import MatchPair
    from spyglass.spikesorting.v2.matcher_protocol import register_matcher
    from spyglass.spikesorting.v2.unit_matching import MatcherParameters

    class _FixtureMatcherParams(BaseModel):
        """Params schema for a test-only matcher."""

        model_config = ConfigDict(extra="forbid")
        tracked_unit_threshold: float = 0.5
        max_strict_nodes: int = 2000
        probability: float = 0.99
        pairs: list = Field(default_factory=list)
        schema_version: int = 1

    class _FixturePairer:
        """Emits the listed (unit_a, unit_b) pairs."""

        name = matcher_name

        def match(self, session_inputs, params):
            left = session_inputs[0].session_key
            right = session_inputs[1].session_key
            return [
                MatchPair(
                    session_a_sorting_id=str(left["sorting_id"]),
                    session_a_curation_id=int(left["curation_id"]),
                    unit_a_id=int(pair_a),
                    session_b_sorting_id=str(right["sorting_id"]),
                    session_b_curation_id=int(right["curation_id"]),
                    unit_b_id=int(pair_b),
                    match_probability=float(params.get("probability", 0.99)),
                )
                for pair_a, pair_b in params.get("pairs", [])
            ]

    def _noop_extract(session_dir, recording, sorting, **kwargs):
        Path(session_dir).mkdir(parents=True, exist_ok=True)
        if seen_unit_ids is not None:
            seen_unit_ids.append([int(u) for u in sorting.get_unit_ids()])

    monkeypatch.setattr(
        _unitmatch_backend, "extract_unitmatch_bundle", _noop_extract
    )

    saved = (dict(mp._MATCHER_REGISTRY), dict(mp._SCHEMA_REGISTRY))
    register_matcher(_FixturePairer(), _FixtureMatcherParams)
    try:
        MatcherParameters().insert1(
            {
                "matcher_params_name": matcher_params_name,
                "matcher": matcher_name,
                "params": {"pairs": pairs, "probability": probability},
            },
            skip_duplicates=True,
        )
    except Exception:
        _restore_matcher_registry(saved)
        raise
    return saved


def _restore_matcher_registry(saved_registry) -> None:
    """Undo a test-only matcher registration."""
    from spyglass.spikesorting.v2 import matcher_protocol as mp

    saved_matchers, saved_schemas = saved_registry
    mp._MATCHER_REGISTRY.clear()
    mp._MATCHER_REGISTRY.update(saved_matchers)
    mp._SCHEMA_REGISTRY.clear()
    mp._SCHEMA_REGISTRY.update(saved_schemas)


@pytest.mark.slow
def test_make_runs_full_matcher_table_path(
    two_session_curated_group, monkeypatch
):
    """A registered lightweight matcher drives a real two-member
    ``UnitMatch.populate()``: the full make() path extracts bundles, runs the
    matcher, writes ``Pair`` rows + the NWB pairs table, and ``TrackedUnit``
    groups the matched pair. Proves ``MatcherProtocol`` plugs into the table
    layer end to end without UnitMatchPy (bundle extraction is stubbed since the
    fixture matcher reads its pairs from ``params``, not the bundles)."""
    from pydantic import BaseModel, ConfigDict, Field

    from spyglass.spikesorting.v2 import _unitmatch_backend
    from spyglass.spikesorting.v2 import matcher_protocol as mp
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.matcher_protocol import (
        MatchPair,
        register_matcher,
    )
    from spyglass.spikesorting.v2.unit_matching import (
        MatcherParameters,
        TrackedUnit,
        UnitMatch,
        UnitMatchSelection,
    )

    class _FixtureMatcherParams(BaseModel):
        """Params schema for the test-only ``fixture_pairer`` matcher."""

        model_config = ConfigDict(extra="forbid")
        tracked_unit_threshold: float = 0.5
        max_strict_nodes: int = 2000
        probability: float = 0.99
        pairs: list = Field(default_factory=list)
        schema_version: int = 1

    grp = two_session_curated_group
    matchable_a = CurationV2().get_matchable_unit_ids(grp["choices"][0])
    matchable_b = CurationV2().get_matchable_unit_ids(grp["choices"][1])
    if len(matchable_a) == 0 or len(matchable_b) == 0:
        pytest.skip("minirec sort produced no matchable units to pair")
    unit_a, unit_b = int(matchable_a[0]), int(matchable_b[0])

    class _FixturePairer:
        """Emits the (unit_a, unit_b) pairs listed in ``params``."""

        name = "fixture_pairer"

        def match(self, session_inputs, params):
            left = session_inputs[0].session_key
            right = session_inputs[1].session_key
            return [
                MatchPair(
                    session_a_sorting_id=str(left["sorting_id"]),
                    session_a_curation_id=int(left["curation_id"]),
                    unit_a_id=int(pair_a),
                    session_b_sorting_id=str(right["sorting_id"]),
                    session_b_curation_id=int(right["curation_id"]),
                    unit_b_id=int(pair_b),
                    match_probability=float(params.get("probability", 0.99)),
                )
                for pair_a, pair_b in params.get("pairs", [])
            ]

    # Stub bundle extraction: the fixture matcher reads pairs from params, so it
    # needs no real waveform bundle (and thus no UnitMatchPy).
    def _noop_extract(session_dir, recording, sorting, **kwargs):
        Path(session_dir).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        _unitmatch_backend, "extract_unitmatch_bundle", _noop_extract
    )

    saved_matchers = dict(mp._MATCHER_REGISTRY)
    saved_schemas = dict(mp._SCHEMA_REGISTRY)
    register_matcher(_FixturePairer(), _FixtureMatcherParams)
    selection_pk = None
    try:
        MatcherParameters().insert1(
            {
                "matcher_params_name": "fixture_pairer_params",
                "matcher": "fixture_pairer",
                "params": {"pairs": [[unit_a, unit_b]], "probability": 0.99},
            },
            skip_duplicates=True,
        )
        selection_pk = UnitMatchSelection.insert_selection(
            grp["owner"],
            grp["group_name"],
            "fixture_pairer_params",
            grp["choices"],
        )

        UnitMatch.populate(selection_pk, reserve_jobs=False)

        assert (UnitMatch & selection_pk).fetch1("n_pairs") == 1
        assert len(UnitMatch.Pair & selection_pk) == 1
        pairs_df = UnitMatch().get_pairs(selection_pk)
        assert len(pairs_df) == 1
        assert int(pairs_df.iloc[0]["unit_a_id"]) == unit_a
        assert int(pairs_df.iloc[0]["unit_b_id"]) == unit_b

        TrackedUnit.populate(selection_pk, reserve_jobs=False)
        matched = [
            row
            for row in (TrackedUnit & selection_pk).fetch(as_dict=True)
            if row["n_sessions_observed"] == 2
        ]
        assert len(matched) == 1
        assert matched[0]["median_match_probability"] == pytest.approx(0.99)
        assert (
            len(
                TrackedUnit.Member
                & {**selection_pk, "tracked_unit_id": matched[0]["tracked_unit_id"]}
            )
            == 2
        )

        # Purity guard: make_compute must NOT re-derive the matchable unit set
        # -- make_fetch resolves it and threads it through the carrier. Run
        # fetch, then break get_matchable_unit_ids, then run compute directly: it
        # must succeed using the threaded set. Catches a regression that moves
        # the curation-label DB read back into compute (where DB state could
        # drift from the fetched carrier).
        fetched = UnitMatch().make_fetch(selection_pk)
        original_matchable = CurationV2.get_matchable_unit_ids

        def _forbidden_matchable(self, *args, **kwargs):
            raise AssertionError(
                "UnitMatch.make_compute called get_matchable_unit_ids; the "
                "matchable set must come from make_fetch's carrier."
            )

        CurationV2.get_matchable_unit_ids = _forbidden_matchable
        try:
            computed = UnitMatch().make_compute(
                selection_pk, **fetched._asdict()
            )
        finally:
            CurationV2.get_matchable_unit_ids = original_matchable
        # make_compute staged a fresh analysis NWB (separate from the populated
        # one); unlink it so the probe leaves no orphan even if the assertion
        # below fails.
        from spyglass.spikesorting.v2.recording import (
            _unlink_staged_analysis_file,
        )

        try:
            assert computed.n_pairs == 1
        finally:
            _unlink_staged_analysis_file(
                computed.analysis_file_name, context="unitmatch purity guard"
            )
    finally:
        if selection_pk is not None:
            (UnitMatch & selection_pk).super_delete(warn=False)
            (UnitMatchSelection & selection_pk).super_delete(warn=False)
        (
            MatcherParameters
            & {"matcher_params_name": "fixture_pairer_params"}
        ).super_delete(warn=False)
        mp._MATCHER_REGISTRY.clear()
        mp._MATCHER_REGISTRY.update(saved_matchers)
        mp._SCHEMA_REGISTRY.clear()
        mp._SCHEMA_REGISTRY.update(saved_schemas)


@pytest.mark.slow
def test_full_unitmatch_workflow_with_accepted_evaluation_children(
    two_session_curated_group, curation_evaluation_defaults, monkeypatch
):
    """The user workflow runs end-to-end from accepted eval children.

    Evaluate each member curation, accept the evaluation labels into committed
    ``CurationV2`` children, pin those exact children in ``UnitMatchSelection``,
    then run ``UnitMatch.populate`` / ``get_pairs`` / ``TrackedUnit.populate``.
    """
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.metric_curation import (
        CurationEvaluation,
        CurationEvaluationSelection,
    )
    from spyglass.spikesorting.v2.unit_matching import (
        MatcherParameters,
        TrackedUnit,
        UnitMatch,
        UnitMatchSelection,
    )

    grp = two_session_curated_group
    accepted_choices = {}
    accepted_children = []
    matcher_params_name = "fixture_eval_child_pairer_params"
    saved_registry = None
    selection_pk = None
    try:
        for member_index, choice in grp["choices"].items():
            sel = CurationEvaluationSelection.insert_selection(
                {
                    **choice,
                    "metric_params_name": "minimal",
                    "auto_curation_rules_name": "none",
                }
            )
            CurationEvaluation.populate(sel, reserve_jobs=False)
            child = CurationEvaluation().use_evaluation_labels(
                sel,
                description="unitmatch workflow accepted labels",
                reuse_existing=False,
            )
            assert CurationV2.is_committed_curation(child)
            assert (CurationV2 & child).fetch1("curation_source") == (
                "curation_evaluation"
            )
            accepted_children.append(child)
            accepted_choices[int(member_index)] = {
                "sorting_id": child["sorting_id"],
                "curation_id": child["curation_id"],
            }

        matchable = {
            member_index: [
                int(u) for u in CurationV2().get_matchable_unit_ids(choice)
            ]
            for member_index, choice in accepted_choices.items()
        }
        if any(len(units) == 0 for units in matchable.values()):
            pytest.skip("minirec sort produced no matchable units to pair")
        unit_a, unit_b = matchable[0][0], matchable[1][0]

        saved_registry = _install_fixture_pairer(
            monkeypatch,
            matcher_name="fixture_eval_child_pairer",
            matcher_params_name=matcher_params_name,
            pairs=[[unit_a, unit_b]],
            probability=0.97,
        )
        selection_pk = UnitMatchSelection.insert_selection(
            grp["owner"],
            grp["group_name"],
            matcher_params_name,
            accepted_choices,
        )
        UnitMatch.populate(selection_pk, reserve_jobs=False)

        assert (UnitMatch & selection_pk).fetch1("n_pairs") == 1
        frozen = {
            (
                int(row["member_index"]),
                str(row["sorting_id"]),
                int(row["curation_id"]),
                int(row["unit_id"]),
            )
            for row in (UnitMatch.MatchableUnit & selection_pk).fetch(as_dict=True)
        }
        for member_index, choice in accepted_choices.items():
            for unit_id in matchable[member_index]:
                assert (
                    member_index,
                    str(choice["sorting_id"]),
                    int(choice["curation_id"]),
                    unit_id,
                ) in frozen

        pairs_df = UnitMatch().get_pairs(selection_pk)
        assert len(pairs_df) == 1
        assert int(pairs_df.iloc[0]["unit_a_id"]) == unit_a
        assert int(pairs_df.iloc[0]["unit_b_id"]) == unit_b
        assert pairs_df.iloc[0]["match_probability"] == pytest.approx(0.97)

        TrackedUnit.populate(selection_pk, reserve_jobs=False)
        matched = [
            row
            for row in (TrackedUnit & selection_pk).fetch(as_dict=True)
            if row["n_sessions_observed"] == 2
        ]
        assert len(matched) == 1
        assert matched[0]["median_match_probability"] == pytest.approx(0.97)
    finally:
        if selection_pk is not None:
            (UnitMatch & selection_pk).super_delete(warn=False)
            (UnitMatchSelection & selection_pk).super_delete(warn=False)
        (
            MatcherParameters & {"matcher_params_name": matcher_params_name}
        ).super_delete(warn=False)
        if saved_registry is not None:
            _restore_matcher_registry(saved_registry)
        for child in accepted_children:
            (CurationV2 & child).super_delete(warn=False)


@pytest.mark.slow
def test_unitmatch_populate_with_committed_merged_child_member(
    two_session_curated_group, monkeypatch
):
    """A committed merged child can run through full ``UnitMatch.populate``.

    This extends the make-fetch coverage: the populate path must load the merged
    child's ``CurationV2.get_sorting`` result, select the merged unit, write a
    pair row, and allow ``TrackedUnit`` to group that merged unit with another
    member's unit.
    """
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        Sorting,
        SortingSelection,
    )
    from spyglass.spikesorting.v2.unit_matching import (
        MatcherParameters,
        TrackedUnit,
        UnitMatch,
        UnitMatchSelection,
    )
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    grp = two_session_curated_group
    two_unit_sort, unit_ids, two_unit_params = (
        _plant_two_unit_sort_on_first_member(grp)
    )
    sorting_key = {"sorting_id": two_unit_sort["sorting_id"]}
    clear_curations_for(two_unit_sort)

    matcher_params_name = "fixture_merged_child_pairer_params"
    selection_pk = None
    seen_unit_ids: list[list[int]] = []
    saved_registry = None
    try:
        merged0 = CurationV2.create_merged_curation(
            sorting_key,
            merge_groups=[[unit_ids[0], unit_ids[1]]],
        )
        assert CurationV2.is_committed_curation(merged0)
        merged_choice = {
            "sorting_id": merged0["sorting_id"],
            "curation_id": merged0["curation_id"],
        }
        merged_matchable = [
            int(u) for u in CurationV2().get_matchable_unit_ids(merged_choice)
        ]
        assert len(merged_matchable) == 1
        merged_uid = merged_matchable[0]

        member1_choice = grp["choices"][1]
        member1_matchable = [
            int(u) for u in CurationV2().get_matchable_unit_ids(member1_choice)
        ]
        if not member1_matchable:
            pytest.skip("member 1 minirec sort produced no matchable units")
        unit_b = member1_matchable[0]

        saved_registry = _install_fixture_pairer(
            monkeypatch,
            matcher_name="fixture_merged_child_pairer",
            matcher_params_name=matcher_params_name,
            pairs=[[merged_uid, unit_b]],
            probability=0.98,
            seen_unit_ids=seen_unit_ids,
        )
        selection_pk = UnitMatchSelection.insert_selection(
            grp["owner"],
            grp["group_name"],
            matcher_params_name,
            {0: merged_choice, 1: member1_choice},
        )
        UnitMatch.populate(selection_pk, reserve_jobs=False)

        assert [merged_uid] in seen_unit_ids
        assert (UnitMatch & selection_pk).fetch1("n_pairs") == 1
        pair = (UnitMatch.Pair & selection_pk).fetch1()
        assert str(pair["session_a_sorting_id"]) == str(merged_choice["sorting_id"])
        assert int(pair["session_a_curation_id"]) == int(
            merged_choice["curation_id"]
        )
        assert int(pair["unit_a_id"]) == merged_uid
        assert int(pair["unit_b_id"]) == unit_b

        TrackedUnit.populate(selection_pk, reserve_jobs=False)
        matched = [
            row
            for row in (TrackedUnit & selection_pk).fetch(as_dict=True)
            if row["n_sessions_observed"] == 2
        ]
        assert len(matched) == 1
    finally:
        if selection_pk is not None:
            (UnitMatch & selection_pk).super_delete(warn=False)
            (UnitMatchSelection & selection_pk).super_delete(warn=False)
        (
            MatcherParameters & {"matcher_params_name": matcher_params_name}
        ).super_delete(warn=False)
        if saved_registry is not None:
            _restore_matcher_registry(saved_registry)
        clear_curations_for(two_unit_sort)
        (Sorting & sorting_key).super_delete(warn=False)
        (SortingSelection & sorting_key).super_delete(warn=False)
        (
            SorterParameters & {"sorter_params_name": two_unit_params}
        ).super_delete(warn=False)


# --------------------------------------------------------------------------- #
# Ground-truth AUC gate (slow / heavy). The ship criterion: UnitMatch          #
# discriminates planted cross-session correspondences on the polymer probe with #
# AUC > 0.85. Skips when the two-session fixture or UnitMatchPy is absent.       #
# --------------------------------------------------------------------------- #

_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
_POLYMER_S1 = _FIXTURE_DIR / "mearec_polymer_128ch_2sessions_s1.nwb"
_POLYMER_S2 = _FIXTURE_DIR / "mearec_polymer_128ch_2sessions_s2.nwb"


def _ground_truth_sorting(nwb_path, sampling_frequency):
    """Read the planted ground-truth spike trains into a NumpySorting."""
    from pynwb import NWBHDF5IO
    from spikeinterface.core import NumpySorting

    with NWBHDF5IO(str(nwb_path), "r") as io:
        gt = io.read().processing["ground_truth"].data_interfaces["units"]
        trains = {
            int(unit): np.asarray(gt["spike_times"][i])
            for i, unit in enumerate(gt.id[:])
        }
    times = np.concatenate([trains[u] for u in trains])
    labels = np.concatenate(
        [np.full(len(trains[u]), u) for u in trains]
    )
    order = np.argsort(times)
    return NumpySorting.from_times_and_labels(
        times[order], labels[order], sampling_frequency=sampling_frequency
    )



def _auc(scores, labels):
    """Mann-Whitney AUC = P(score_pos > score_neg), ties counted as 0.5."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        raise AssertionError(
            f"AUC needs both classes; got {len(pos)} positive / {len(neg)} "
            "negative cross-session pairs."
        )
    wins = 0.0
    for value in pos:
        wins += np.sum(value > neg) + 0.5 * np.sum(value == neg)
    return wins / (len(pos) * len(neg))


def _bandpassed_polymer_recording(fixture_path):
    """Read + bandpass the raw polymer fixture (matcher-bundle input)."""
    import spikeinterface.preprocessing as spre
    from spikeinterface.extractors import read_nwb_recording

    recording = read_nwb_recording(
        str(fixture_path), electrical_series_path="acquisition/e-series"
    )
    recording = spre.bandpass_filter(recording, freq_min=300.0, freq_max=6000.0)
    return recording.set_probe(recording.get_probe().to_2d())


@pytest.mark.slow
@pytest.mark.integration
def test_v2_unitmatch_polymer_mearec_ground_truth(dj_conn, tmp_path):
    """Ship gate: AUC of UnitMatch probability vs ground-truth correspondence
    on the two-session polymer probe is > 0.85.

    The two sessions are built from one MEArec template set sharing a
    ``template_seed`` (so ground-truth unit ``i`` is the same neuron in both
    sessions -- the planted cross-session correspondences), differing only in
    spike-train realization and a small inter-session drift. The v2
    single-session sort + curation pipeline is run on both sessions to prove it
    ingests and processes the polymer fixture end to end; the matcher's
    discrimination is then scored against the ground-truth sortings, because the
    MEArec template amplitudes sit below the production sorter's detection
    threshold (the same reason the smoke fixture uses a fixture-tuned sorter
    row), so scoring on the ground-truth isolates cross-session matching from
    sorter yield. The ROC labels every cross-session unit pair by whether the
    two sides are the same ground-truth neuron and uses the matcher probability
    as the score.

    Coverage note (the three validations meet in the middle): this gate proves
    the matcher SCIENCE (AUC vs ground truth); ``test_make_runs_full_matcher_
    table_path`` proves the DataJoint TABLE path (selection -> make -> Pair ->
    get_pairs -> TrackedUnit) on real v2-curated units; and
    ``test_unitmatch_backend.test_match_recovers_planted_correspondences``
    proves real-UnitMatchPy bundle extraction + matching on a real recording.
    """
    if not (_POLYMER_S1.exists() and _POLYMER_S2.exists()):
        pytest.skip(
            "two-session polymer fixture absent; generate with "
            "generate_mearec.py --only mearec_polymer_128ch_2sessions_s1 "
            "--only mearec_polymer_128ch_2sessions_s2"
        )
    pytest.importorskip("UnitMatchPy")

    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2._lookup_validation import _validate_params
    from spyglass.spikesorting.v2._params.sorter import _get_sorter_schema
    from spyglass.spikesorting.v2._unitmatch_backend import (
        UnitMatchBackend,
        extract_unitmatch_bundle,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.matcher_protocol import SessionMatcherInput
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
        SortGroupV2,
    )
    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        Sorting,
        SortingSelection,
    )

    from tests.spikesorting.v2._ingest_helpers import (
        clear_curations_for,
        copy_and_insert_nwb,
    )

    owner = "unitmatch_polymer_owner"
    initialize_v2_defaults()
    LabTeam.insert1(
        {"team_name": owner, "team_description": "unitmatch polymer gate"},
        skip_duplicates=True,
    )
    # MEArec simulated templates sit below the production 5.5-sigma MS5
    # detection threshold (the fixture analog of the smoke clusterless 5-uV row).
    mearec_ms5_params = "mearec_polymer_ms5"
    SorterParameters.insert1(
        {
            "sorter": "mountainsort5",
            "sorter_params_name": mearec_ms5_params,
            "params": _validate_params(
                _get_sorter_schema("mountainsort5"), {"detect_threshold": 4.0}
            ),
        },
        skip_duplicates=True,
        allow_duplicate_params=True,
    )

    sort_pks = []
    nwb_file_names = []
    try:
        # 1. v2 sort + curation on both sessions (pipeline runs end to end).
        for index, fixture_path in enumerate((_POLYMER_S1, _POLYMER_S2)):
            nwb_file_name = copy_and_insert_nwb(
                fixture_path, dest_name=f"unitmatch_polymer_{index}.nwb"
            )
            nwb_file_names.append(nwb_file_name)
            session_key = {"nwb_file_name": nwb_file_name}
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
                    "team_name": owner,
                    "preprocessing_params_name": "default",
                }
            )
            if not (Recording & rec_pk):
                Recording.populate(rec_pk, reserve_jobs=False)
            sort_pk = SortingSelection.insert_selection(
                {
                    "recording_id": rec_pk["recording_id"],
                    "sorter": "mountainsort5",
                    "sorter_params_name": mearec_ms5_params,
                }
            )
            if not (Sorting & sort_pk):
                Sorting.populate(sort_pk, reserve_jobs=False)
            sort_pks.append(sort_pk)
            clear_curations_for(sort_pk)
            curation_key = CurationV2.insert_curation(
                sorting_key={"sorting_id": sort_pk["sorting_id"]}
            )
            assert len(CurationV2 & curation_key) == 1

        # 2. Score the matcher against ground truth. Build one bundle per
        #    session from the bandpassed recording + the planted ground-truth
        #    sorting, run UnitMatch at threshold 0 (so every cross-session pair
        #    gets a probability), and compute the AUC of probability vs
        #    same-ground-truth-neuron.
        session_inputs = []
        gt_unit_ids = []
        for index, fixture_path in enumerate((_POLYMER_S1, _POLYMER_S2)):
            recording = _bandpassed_polymer_recording(fixture_path)
            gt_sorting = _ground_truth_sorting(
                fixture_path, recording.get_sampling_frequency()
            )
            gt_unit_ids.append([int(u) for u in gt_sorting.get_unit_ids()])
            session_dir = tmp_path / f"gt_session_{index}"
            extract_unitmatch_bundle(session_dir, recording, gt_sorting)
            session_inputs.append(
                SessionMatcherInput(
                    session_key={
                        "sorting_id": f"polymer_gt_{index}",
                        "curation_id": 0,
                    },
                    waveform_dir=session_dir,
                    channel_positions_path=(
                        session_dir / "channel_positions.npy"
                    ),
                )
            )

        pairs = UnitMatchBackend().match(
            session_inputs, {"match_threshold": 0.0}
        )
        score_by_units = {
            (pair.unit_a_id, pair.unit_b_id): pair.match_probability
            for pair in pairs
        }
        scores = []
        labels = []
        for unit_a in gt_unit_ids[0]:
            for unit_b in gt_unit_ids[1]:
                scores.append(score_by_units.get((unit_a, unit_b), 0.0))
                labels.append(unit_a == unit_b)

        auc = _auc(scores, labels)
        assert auc > 0.85, (
            f"UnitMatch polymer ground-truth AUC {auc:.3f} <= 0.85 "
            f"({sum(labels)} true / {len(labels) - sum(labels)} false pairs)"
        )
    finally:
        from tests.spikesorting.v2._ingest_helpers import _clean_session_v2

        for sort_pk in sort_pks:
            clear_curations_for(sort_pk)
        for nwb_file_name in nwb_file_names:
            _clean_session_v2({"nwb_file_name": nwb_file_name})


@pytest.mark.slow
def test_pair_insert_rejects_unpinned_curation(two_session_curated_group):
    """A raw ``UnitMatch.Pair.insert`` outside the selection's pinned
    ``MemberCuration`` -- an unpinned-curation endpoint, a same-member edge, a
    reversed / duplicate edge, or an out-of-range ``match_probability`` -- raises
    ``UnitMatchPairIntegrityError``.

    Validates against the pinned ``MemberCuration`` rows (created by
    ``insert_selection``), so it needs no populated ``UnitMatch`` -- the
    single-unit chronic fixture is degenerate for the matcher backend. The
    canonical ``make_insert`` path routes through this same validated
    ``Pair.insert`` and is covered by the multi-unit end-to-end matcher tests
    (which populate successfully and would fail here if the guard rejected valid,
    oriented, deduped pairs).
    """
    import uuid

    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.exceptions import (
        UnitMatchPairIntegrityError,
    )
    from spyglass.spikesorting.v2.unit_matching import (
        UnitMatch,
        UnitMatchSelection,
    )

    grp = two_session_curated_group
    pk = UnitMatchSelection.insert_selection(
        grp["owner"], grp["group_name"], "unitmatch_default", grp["choices"]
    )
    try:
        members = (UnitMatchSelection.MemberCuration & pk).fetch(
            as_dict=True, order_by="member_index"
        )
        m0, m1 = members[0], members[1]
        node0 = (
            str(m0["sorting_id"]),
            int(m0["curation_id"]),
            int(
                (
                    CurationV2.Unit
                    & {
                        "sorting_id": m0["sorting_id"],
                        "curation_id": m0["curation_id"],
                    }
                ).fetch("unit_id")[0]
            ),
        )
        node1 = (
            str(m1["sorting_id"]),
            int(m1["curation_id"]),
            int(
                (
                    CurationV2.Unit
                    & {
                        "sorting_id": m1["sorting_id"],
                        "curation_id": m1["curation_id"],
                    }
                ).fetch("unit_id")[0]
            ),
        )

        def pair_row(a, b, *, prob=0.9, idx=900):
            return {
                "unitmatch_id": pk["unitmatch_id"],
                "pair_index": idx,
                "session_a_sorting_id": a[0],
                "session_a_curation_id": a[1],
                "unit_a_id": a[2],
                "session_b_sorting_id": b[0],
                "session_b_curation_id": b[1],
                "unit_b_id": b[2],
                "match_probability": prob,
            }

        # Out-of-range match_probability.
        with pytest.raises(UnitMatchPairIntegrityError, match="outside"):
            UnitMatch.Pair.insert1(pair_row(node0, node1, prob=1.5))

        # Endpoint pinned to a curation NOT in this selection's MemberCuration
        # (a fake curation: the guard checks the pinned set before the FK fires).
        unpinned = (str(uuid.uuid4()), 0, 0)
        with pytest.raises(UnitMatchPairIntegrityError, match="not a pinned"):
            UnitMatch.Pair.insert1(pair_row(unpinned, node1))

        # Same-member edge (both endpoints pin member 0): a unit cannot match
        # itself across sessions.
        with pytest.raises(UnitMatchPairIntegrityError, match="same member"):
            UnitMatch.Pair.insert1(pair_row(node0, node0))

        # Reversed / duplicate undirected edge within one batch.
        with pytest.raises(
            UnitMatchPairIntegrityError, match="duplicate / reversed"
        ):
            UnitMatch.Pair.insert(
                [
                    pair_row(node0, node1, idx=901),
                    pair_row(node1, node0, idx=902),
                ]
            )
    finally:
        (UnitMatchSelection & pk).super_delete(warn=False)


@pytest.mark.slow
def test_tracked_unit_uses_frozen_universe_after_relabel(
    two_session_curated_group,
):
    """``TrackedUnit`` reads ``UnitMatch``'s FROZEN ``MatchableUnit``
    snapshot, not current curation labels. Relabeling a singleton (no-``Pair``)
    member unit to ``noise`` AFTER ``UnitMatch`` populated must NOT drop it from
    the tracked-unit graph -- the frozen universe keeps it. Fails on pre-change
    code, where ``TrackedUnit.make`` re-derived the universe from current labels
    and dropped the relabeled unit.
    """
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.unit_matching import (
        TrackedUnit,
        UnitMatch,
        UnitMatchSelection,
    )

    grp = two_session_curated_group
    choice = grp["choices"][0]
    pk = UnitMatchSelection.insert_selection(
        grp["owner"], grp["solo_name"], "unitmatch_default", {0: choice}
    )
    # Solo group -> degenerate make (0 pairs, no matcher backend) but still
    # freezes the MatchableUnit snapshot.
    UnitMatch.populate(pk, reserve_jobs=False)
    frozen = {
        (str(r["sorting_id"]), int(r["curation_id"]), int(r["unit_id"]))
        for r in (UnitMatch.MatchableUnit & pk).fetch(as_dict=True)
    }
    assert frozen, "MatchableUnit snapshot should be non-empty"
    target_unit = sorted(int(unit) for _, _, unit in frozen)[0]

    noise_label = {**choice, "unit_id": target_unit, "curation_label": "noise"}
    try:
        # Relabel the frozen singleton so the CURRENT matchable set excludes it.
        CurationV2.UnitLabel.insert1(noise_label)
        assert target_unit not in [
            int(u) for u in CurationV2().get_matchable_unit_ids(choice)
        ], "relabel should drop the unit from the CURRENT matchable set"

        TrackedUnit.populate(pk, reserve_jobs=False)
        tracked_units = {
            int(member["unit_id"])
            for member in (TrackedUnit.Member & pk).fetch(as_dict=True)
        }
        assert target_unit in tracked_units, (
            "the relabeled singleton must survive in TrackedUnit: UnitMatch's "
            "frozen MatchableUnit -- not current labels -- is the node universe"
        )
    finally:
        (CurationV2.UnitLabel & noise_label).delete(safemode=False)
        (TrackedUnit & pk).delete(safemode=False)
        (UnitMatchSelection & pk).super_delete(warn=False)


@pytest.mark.slow
def test_geometry_preflight_fails_before_extraction(
    two_session_curated_group, monkeypatch
):
    """A cross-probe / cross-day channel-geometry mismatch is rejected at
    ``UnitMatchSelection.insert_selection`` -- BEFORE any dense bundle extraction
    runs (extraction lives in ``UnitMatch.make``, which never runs here)."""
    import numpy as np

    from spyglass.spikesorting.v2 import _unitmatch_backend
    from spyglass.spikesorting.v2.unit_matching import UnitMatchSelection

    grp = two_session_curated_group
    sid0 = str(grp["choices"][0]["sorting_id"])

    def fake_positions(curation_key):
        # member 0: 4-channel probe; the other member: 8-channel -> mismatch.
        n = 4 if str(curation_key["sorting_id"]) == sid0 else 8
        return np.zeros((n, 2))

    monkeypatch.setattr(
        UnitMatchSelection,
        "_member_channel_positions",
        staticmethod(fake_positions),
    )
    # Force the new-insert path so the preflight runs even though this selection
    # may already exist from an earlier test (no shared-state mutation).
    monkeypatch.setattr(
        UnitMatchSelection,
        "_find_existing_pk",
        classmethod(lambda cls, *a, **k: None),
    )

    def _boom(*args, **kwargs):
        raise AssertionError(
            "dense bundle extraction ran before the geometry preflight"
        )

    monkeypatch.setattr(_unitmatch_backend, "extract_unitmatch_bundle", _boom)

    with pytest.raises(ValueError, match="probe geometry"):
        UnitMatchSelection.insert_selection(
            grp["owner"], grp["group_name"], "unitmatch_default", grp["choices"]
        )

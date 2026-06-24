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
    """Insert the smoke clusterless sorter row (cheap minirec sort)."""
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


@pytest.fixture(scope="module")
def two_session_curated_group(chronic_2_session_minirec):
    """Two single-session sorts + curations under one SessionGroup.

    Builds on the package-scoped chronic minirec substrate: creates a two-member
    SessionGroup (and a one-member solo group for the degenerate case), sorts and
    root-curates each same-day member with the cheap clusterless thresholder, and
    yields the group identities plus the per-member curation choices. Tears down
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

    sorter_params_name = _ensure_minirec_clusterless_params()
    MatcherParameters.insert_default()

    # Start clean: drop any leftover groups (and their UnitMatch lineage) for
    # the owner from a prior interrupted run.
    clean_session_groups_for_owner(owner)
    SessionGroup.create_group(owner, _GROUP_NAME, members)
    SessionGroup.create_group(owner, _SOLO_NAME, members[:1])

    sort_pks = []
    choices = {}
    for index, rec_pk in enumerate(recording_pks):
        sort_pk = SortingSelection.insert_selection(
            {
                "recording_id": rec_pk["recording_id"],
                "sorter": "clusterless_thresholder",
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
        }
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


@pytest.mark.slow
def test_pair_fk_rejects_unknown_unit(two_session_curated_group):
    """Goal 7: a Pair referencing a unit absent from the pinned curation is
    rejected by the DataJoint foreign key."""
    import datajoint as dj

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
        "session_b_sorting_id": side_b["sorting_id"],
        "session_b_curation_id": side_b["curation_id"],
        "unit_b_id": 10**9,
        "match_probability": 0.9,
    }
    with pytest.raises(dj.errors.IntegrityError):
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

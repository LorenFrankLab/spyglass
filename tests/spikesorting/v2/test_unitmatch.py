"""Cross-session unit tracking: schema + matcher validation.

Covers the ten cross-session matching validation goals. The graph-logic goals
(pair canonicalization, strict-clique tracked-unit derivation, the bounded-search
cap, unmatched singletons) are exercised as pure ``_matcher_graph`` functions
with no database; the table-wiring goals (registry-validating insert, the
explicit per-member curation selection, the ``make()`` provenance recheck, and
the ``Pair`` foreign-key integrity) drive the real DataJoint tables over a
two-session synthetic sort. The ground-truth AUC gate runs UnitMatch end to end
on the polymer two-session fixture; it is verified locally and is not yet
CI-enforced (the two-session fixtures are unhosted, so it skips in CI).

The pure-logic and external-matcher sections need neither a database nor
UnitMatchPy. The database sections request ``dj_conn``; the gate additionally
requires the polymer two-session fixture and the optional UnitMatchPy extra and
skips cleanly when either is absent.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from tests.spikesorting.v2._unitmatch_helpers import (
    install_fixture_pairer as _install_fixture_pairer,
)
from tests.spikesorting.v2._unitmatch_helpers import (
    restore_matcher_registry as _restore_matcher_registry,
)
from tests.spikesorting.v2._sorter_stub import plant_sorter


def test_unitmatch_params_reject_asymmetric_waveform_window():
    """UnitMatch assumes the trough is centered in its waveform window."""
    from pydantic import ValidationError

    from spyglass.spikesorting.v2._params.matcher import UnitMatchParamsSchema

    with pytest.raises(ValidationError, match="symmetric waveform window"):
        UnitMatchParamsSchema(ms_before=0.5, ms_after=2.0)

    params = UnitMatchParamsSchema(ms_before=0.5, ms_after=0.5)
    assert params.ms_before == params.ms_after == 0.5


def test_unitmatch_bundle_rejects_asymmetric_window_before_io():
    """The public bundle builder enforces the same centered-window contract."""
    from pydantic import ValidationError

    from spyglass.spikesorting.v2._matching.unitmatch_backend import (
        extract_unitmatch_bundle,
    )

    with pytest.raises(ValidationError, match="symmetric waveform window"):
        extract_unitmatch_bundle(
            "unused",
            recording=object(),
            sorting=object(),
            ms_before=0.5,
            ms_after=2.0,
        )


# --------------------------------------------------------------------------- #
# Pure graph logic (no DB): pair canonicalization and strict-clique            #
# tracked-unit derivation.                                                      #
# --------------------------------------------------------------------------- #


def test_unit_semantics_for_sorter():
    """unit_semantics is derived from the sorter (single source of truth): the
    clusterless thresholder emits ONE threshold-crossing pseudo-unit, not a
    sorted neuron; every real sorter emits sorted units."""
    from spyglass.spikesorting.v2._sorting.dispatch import (
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


# Two pinned input curations, one per input_index.
_CUR_A = ("sortA", 0)
_CUR_B = ("sortB", 0)
_INPUT_INDEX = {_CUR_A: 0, _CUR_B: 1}


def test_canonicalize_orients_by_ascending_input_index():
    """A pair given B->A is stored A->B (ascending input_index)."""
    from spyglass.spikesorting.v2._matching.graph import (
        canonicalize_match_pairs,
    )

    # Provide the pair "reversed" (input 1 first); canonicalization flips it.
    reversed_pair = _pair(_CUR_B, 7, _CUR_A, 3, prob=0.8)
    out = canonicalize_match_pairs([reversed_pair], _INPUT_INDEX)
    assert len(out) == 1
    row = out[0]
    assert (row["session_a_sorting_id"], row["session_a_curation_id"]) == _CUR_A
    assert row["unit_a_id"] == 3
    assert (row["session_b_sorting_id"], row["session_b_curation_id"]) == _CUR_B
    assert row["unit_b_id"] == 7
    assert row["match_probability"] == pytest.approx(0.8)


def test_canonicalize_rejects_same_input_pair():
    """A pair whose two sides are the same input (incl. self-pairs) raises."""
    from spyglass.spikesorting.v2._matching.graph import (
        canonicalize_match_pairs,
    )

    same_input = _pair(_CUR_A, 1, _CUR_A, 2)
    with pytest.raises(ValueError, match="same-input"):
        canonicalize_match_pairs([same_input], _INPUT_INDEX)


def test_canonicalize_rejects_reversed_duplicate():
    """The same unit pair given in both orientations cannot both survive."""
    from spyglass.spikesorting.v2._matching.graph import (
        canonicalize_match_pairs,
    )

    forward = _pair(_CUR_A, 3, _CUR_B, 7)
    backward = _pair(_CUR_B, 7, _CUR_A, 3)
    with pytest.raises(ValueError, match="duplicate"):
        canonicalize_match_pairs([forward, backward], _INPUT_INDEX)


def test_canonicalize_rejects_unpinned_curation():
    """A pair referencing a curation not in the pinned input set raises."""
    from spyglass.spikesorting.v2._matching.graph import (
        canonicalize_match_pairs,
    )

    stray = _pair(_CUR_A, 1, ("sortC", 0), 2)
    with pytest.raises(ValueError, match="not pinned|not one of"):
        canonicalize_match_pairs([stray], _INPUT_INDEX)


def _nodes(*specs):
    """``("A", 1)`` -> ``("A", 0, 1)`` curated-unit node tuples."""
    return [(sorting_id, 0, unit_id) for sorting_id, unit_id in specs]


def test_derive_tracked_units_full_triangle_is_one_component():
    """Three sessions, all pairwise edges high -> one clique of all three."""
    from spyglass.spikesorting.v2._matching.graph import derive_tracked_units

    a, b, c = _nodes(("A", 1), ("B", 1), ("C", 1))
    edges = [(a, b, 0.9), (b, c, 0.9), (a, c, 0.9)]
    tracked = derive_tracked_units(
        [a, b, c], edges, threshold=0.5, max_strict_nodes=100
    )
    assert len(tracked) == 1
    tu = tracked[0]
    assert set(tu["members"]) == {a, b, c}
    assert tu["n_sessions_detected"] == 3
    assert tu["n_matching_inputs"] == 3
    assert tu["policy_used"] == "strict"
    assert tu["median_match_probability"] == pytest.approx(0.9)


def test_derive_tracked_units_open_path_splits():
    """A<->B and B<->C high but A<->C low -> >=2 components, none with A and C."""
    from spyglass.spikesorting.v2._matching.graph import derive_tracked_units

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
    from spyglass.spikesorting.v2._matching.graph import derive_tracked_units

    a1, a2 = ("A", 0, 1), ("A", 0, 2)
    b1, b2 = ("B", 0, 1), ("B", 0, 2)
    edges = [(a1, b1, 0.9), (a1, b2, 0.8), (a2, b1, 0.7)]
    tracked = derive_tracked_units(
        [a1, a2, b1, b2], edges, threshold=0.5, max_strict_nodes=100
    )
    all_members = [node for tu in tracked for node in tu["members"]]
    assert len(all_members) == len(
        set(all_members)
    ), "a curated unit was assigned to more than one tracked unit"
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
    from spyglass.spikesorting.v2._matching.graph import derive_tracked_units

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
    from spyglass.spikesorting.v2._matching.graph import derive_tracked_units

    a, b = _nodes(("A", 1), ("B", 1))
    edges = [(a, b, 0.2)]  # below threshold -> no edge in the graph
    tracked = derive_tracked_units(
        [a, b], edges, threshold=0.5, max_strict_nodes=100
    )
    assert len(tracked) == 2
    for tu in tracked:
        assert tu["n_sessions_detected"] == 1
        assert tu["n_matching_inputs"] == 1
        assert tu["median_match_probability"] is None
        assert tu["policy_used"] == "strict"


def test_derive_tracked_units_budget_cap_raises():
    """A node universe above ``max_strict_nodes`` refuses the clique search."""
    from spyglass.spikesorting.v2._matching.graph import derive_tracked_units
    from spyglass.spikesorting.v2.exceptions import (
        TrackedUnitBudgetExceededError,
    )

    nodes = [("S", 0, u) for u in range(10)]
    with pytest.raises(TrackedUnitBudgetExceededError, match="10"):
        derive_tracked_units(nodes, [], threshold=0.5, max_strict_nodes=5)


def test_derive_tracked_units_under_cap_succeeds_strict():
    """Under the cap the search runs and every row is policy 'strict'."""
    from spyglass.spikesorting.v2._matching.graph import derive_tracked_units

    nodes = [("S", 0, u) for u in range(5)]
    tracked = derive_tracked_units(nodes, [], threshold=0.5, max_strict_nodes=5)
    assert len(tracked) == 5
    assert all(tu["policy_used"] == "strict" for tu in tracked)


def test_derive_tracked_units_rejects_edge_outside_universe():
    """An edge endpoint absent from the node universe raises (networkx
    add_edge would otherwise silently create the node, smuggling a unit past
    the node budget and into the partition)."""
    from spyglass.spikesorting.v2._matching.graph import derive_tracked_units

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
    from spyglass.spikesorting.v2._matching.graph import derive_tracked_units

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


def test_derive_tracked_units_counts_sessions_by_nwb_not_curation():
    """A tracked unit spanning two sortings from the SAME nwb counts as ONE
    session. ``n_sessions_detected`` is a cross-session-identity claim, so it
    must key on the recording session (nwb), not on (sorting_id, curation_id) --
    otherwise a within-day match across two sort groups inflates to multi-session.
    """
    from spyglass.spikesorting.v2._matching.graph import derive_tracked_units

    # Sortings "A" and "B" are two sort groups of the SAME day; "C" is another day.
    a, b, c = _nodes(("A", 1), ("B", 1), ("C", 1))
    edges = [(a, b, 0.9), (b, c, 0.9), (a, c, 0.9)]
    tracked = derive_tracked_units(
        [a, b, c],
        edges,
        threshold=0.5,
        max_strict_nodes=100,
        input_by_node={a: 0, b: 1, c: 2},
        detected_sessions_by_node={
            a: {"day1.nwb"},
            b: {"day1.nwb"},
            c: {"day2.nwb"},
        },
    )
    assert len(tracked) == 1
    # Three sortings, but only two distinct recording sessions (day1, day2).
    assert tracked[0]["n_sessions_detected"] == 2
    assert tracked[0]["n_matching_inputs"] == 3


def test_divergent_electrode_space_members_flags_distinct_probe():
    """A member with identical channel GEOMETRY but different electrode identity
    (group / ids / regions) is reported as divergent. The signal is ADVISORY,
    not a rejection: electrode-group names / ids are taken from each NWB and are
    not guaranteed stable across labs' ingestion, so UnitMatch warns rather than
    blocking a legitimate chronic match."""
    from spyglass.spikesorting.v2._matching.graph import (
        divergent_electrode_space_members,
    )

    # Same (electrode_id, region) layout, different electrode_group_name.
    sigs = {
        0: (("probeA", 0, "ca1"), ("probeA", 1, "ca1")),
        1: (("probeB", 0, "ca1"), ("probeB", 1, "ca1")),
    }
    assert divergent_electrode_space_members(sigs) == [1]


def test_divergent_electrode_space_members_empty_when_matching():
    """Members on the same electrode space (identical signature) -> no divergent
    members; a single-member selection trivially has none."""
    from spyglass.spikesorting.v2._matching.graph import (
        divergent_electrode_space_members,
    )

    sig = (("probeA", 0, "ca1"), ("probeA", 1, "ca1"))
    assert divergent_electrode_space_members({0: sig, 1: sig}) == []
    assert divergent_electrode_space_members({0: sig}) == []


# --------------------------------------------------------------------------- #
# NWB pairs (de)serialization round-trip (pure I/O, no DB).                     #
# --------------------------------------------------------------------------- #


def test_pairs_nwb_round_trip_preserves_fdr_none(tmp_path):
    """Non-empty + empty pairs round-trip; a None fdr stores as NaN and reads
    back as None, while a real fdr survives and string ids stay str."""
    from datetime import datetime, timezone

    from pynwb import NWBHDF5IO, NWBFile

    from spyglass.spikesorting.v2._storage.matches_nwb import (
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
            "drift_estimate_um": 2.5,
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
            "drift_estimate_um": -3.125,
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
    assert back == [
        {"pair_index": index, **row} for index, row in enumerate(pairs)
    ]

    # Empty table writes and reads back empty (concrete dtypes, no inference).
    empty_oid = write_pairs_table(_fresh_nwb("empty.nwb"), [])
    assert read_pairs(str(tmp_path / "empty.nwb"), empty_oid) == []
    assert (
        len(build_pairs_table([]).columns) > 0
    )  # columns exist even when empty


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
                        session_a_sorting_id=str(
                            left.curation_key["sorting_id"]
                        ),
                        session_a_curation_id=int(
                            left.curation_key["curation_id"]
                        ),
                        unit_a_id=0,
                        session_b_sorting_id=str(
                            right.curation_key["sorting_id"]
                        ),
                        session_b_curation_id=int(
                            right.curation_key["curation_id"]
                        ),
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
            curation_key={"sorting_id": s, "curation_id": 0},
            bundle_dir=Path("/unused"),
            geometry_path=Path("/unused/cp.npy"),
        )
        for s in ("A", "B")
    ]
    pairs = matcher.match(inputs, {"prob": 0.9})
    assert len(pairs) == 1
    assert pairs[0].match_probability == pytest.approx(0.9)
    # Degenerate single-session case returns no pairs.
    assert matcher.match(inputs[:1], {}) == []


@pytest.mark.slow
def test_driftout_units_recovered_pooled(tmp_path):
    """The production ``per_unit`` bundle construction
    (:func:`extract_unitmatch_bundle`) recovers drift-out units without
    degrading healthy units, on seeds 10..19.

    Builds the synthetic control / driftout_A / driftout_AB scenarios (see
    ``tests/spikesorting/v2/scripts/unitmatch_half_split_experiment.py`` for
    the dataset, scoring and gate definitions -- this test imports them
    rather than duplicating them) for seeds 10..19, condition ``per_unit``
    only, and asserts the pooled acceptance gates for each drift-out
    scenario:

    - Drift-out recall >= 0.80.
    - Healthy template bit-identity: every non-S unit's saved
      cross-validation-half templates are bit-identical to the same seed's
      control run, pooled over seeds. PASS iff identical == total.
    - Healthy true-pair probability drop: for each non-S unit, its true
      cross-session pair's two directed UnitMatch probabilities give ``q_u =
      min(p(A_u -> B_u), p(B_u -> A_u))``; the drop is the pooled mean
      ``q_u`` in the control run minus the pooled mean ``q_u`` in the
      scenario run, over the same paired non-S units, pooled over all seeds.
      PASS iff drop <= 0.04.
    - Healthy false-pair rate increase vs control <= 0.005.

    The S x S false-pair rate among drift-out units is computed and printed
    but not asserted. UnitMatch derives its match threshold, prior and score
    distributions from the units present in each run (UnitMatchPy
    ``metric_functions.get_threshold``); with about 20 units per session that
    per-run fit is unstable, and a near-tie in the threshold search can flip
    on a single redrawn template and admit a burst of false pairs -- observed
    on one of these ten seeds, independent of how the cross-validation
    halves are built. Gating on it would fail the construction under test
    for a limitation of UnitMatch's own calibration, not a defect this test
    can localize, so it is reported as a diagnostic instead (see
    ``evaluate_gates`` for the detailed rationale).

    (The excess probability drop over the recording-half ``time_half``
    construction is not evaluated here because ``time_half`` does not run in
    this test. The script also prints a count-based paired healthy recall
    drop as a diagnostic -- it is not asserted here because a few healthy
    pairs near probability 0.5 can flip between the scenario and control
    runs by chance: UnitMatch refits its match-probability kernels,
    candidate threshold and prior on the whole population on every call, so
    a pair's pass/fail label is not robust to that refit even when its two
    cross-validation-half templates are bit-identical to the no-drift
    control. The template bit-identity gate isolates the template side of
    that comparison directly, and the probability-drop gate replaces the
    pass/fail count with an average of the underlying probabilities, so
    neither is sensitive to threshold flips the same way -- this is why
    healthy units are checked by template identity and mean probability
    rather than a probability-threshold pass/fail count.)

    Seeds, scenarios and thresholds match the script's default run. A
    failure here is a result about the current per-unit construction, not
    a test bug: do not loosen the thresholds, seeds, scenarios or scoring,
    or xfail/skip it, to make it pass. The checked gates are selected by a
    positive allowlist of (gate name, scenario) pairs -- the four gates
    above x driftout_A/driftout_AB (the S x S rate is excluded: it is a
    printed diagnostic, not an acceptance gate) -- and their presence is
    asserted before their pass/fail, so a construction that silently drops
    a scenario (``pooled_counts``, ``pooled_paired_counts`` or
    ``pooled_true_pair_prob_drop`` returning ``None``) fails loudly instead
    of passing vacuously on an empty or incomplete selection. The assertion
    message lists every scenario's gates so a regression elsewhere stays
    visible.

    DB-free: this test requests no ``dj_conn`` fixture and the functions it
    imports (:func:`make_dataset`, :func:`make_scenario_sessions`,
    :func:`run_one`, :func:`evaluate_gates`) reach a database only through
    ``spyglass.spikesorting.v2._matching.unitmatch_backend`` and ``.matcher_protocol``,
    neither of which opens a DataJoint connection at import time or at call
    time here -- ``run_one`` builds bundles on disk and matches them through
    ``UnitMatchBackend.match``, with no table access. pytest fixtures are
    resolved per test at setup, not at collection, so the ``dj_conn``-taking
    tests elsewhere in this module do not force a database for this one.
    """
    pytest.importorskip("UnitMatchPy")
    from tests.spikesorting.v2.scripts.unitmatch_half_split_experiment import (
        DEFAULT_FIRST_SEED,
        DEFAULT_SEEDS,
        DRIFT_OUT_SCENARIOS,
        SCENARIOS,
        choose_drift_out_units,
        evaluate_gates,
        format_gate_line,
        make_dataset,
        make_scenario_sessions,
        run_one,
    )

    condition = "per_unit"
    records = []
    for seed in range(DEFAULT_FIRST_SEED, DEFAULT_FIRST_SEED + DEFAULT_SEEDS):
        recording, sorting = make_dataset(seed)
        drift_out_units = choose_drift_out_units(seed)
        for scenario in SCENARIOS:
            sessions = make_scenario_sessions(
                recording, sorting, scenario, drift_out_units
            )
            records.append(
                run_one(
                    seed,
                    scenario,
                    condition,
                    sessions,
                    drift_out_units,
                    tmp_path,
                )
            )

    gates = evaluate_gates(records, condition)
    table = "\n".join(format_gate_line(g) for g in gates)
    # Positive allowlist: a name-based exclusion would pass vacuously if
    # evaluate_gates silently dropped a scenario (e.g. pooled_counts,
    # pooled_paired_counts or pooled_true_pair_prob_drop returning None),
    # leaving `checked` empty or short. Diagnostics carry a
    # " (diagnostic, not gated)" suffix, so they never match these names.
    expected = {
        (gate_name, scenario)
        for gate_name in (
            "drift-out recall",
            "healthy template bit-identity",
            "healthy true-pair probability drop",
            "healthy false-pair rate increase",
        )
        for scenario in DRIFT_OUT_SCENARIOS
    }
    by_pair = {(g.name, g.scenario): g for g in gates}
    missing = expected - by_pair.keys()
    assert not missing, f"missing gates {sorted(missing)}\n{table}"
    checked = [by_pair[pair] for pair in expected]
    assert all(g.passed for g in checked), table


# --------------------------------------------------------------------------- #
# Table wiring (database): registry-validating insert, the explicit            #
# per-member selection, the make() provenance recheck, the degenerate           #
# single-session make, and Pair FK integrity. Built over two synthetic          #
# single-session sorts on the chronic minirec substrate.                        #
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
#: PLANTED (``plant_sorter``), so this row only exists for the
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
    statistics_spans=None,
):
    """Plant one deterministic sorted unit spread across the recording.

    Enough spikes (and span) for the split-half UnitMatch bundle extraction and
    the display-analyzer template. Replaces a real sorter run via
    ``plant_sorter`` -- fast and deterministic.
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
    sort (``plant_sorter`` -- fast, deterministic, and
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
        drop_unitmatch_selections_for,
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
        plant_sorter(mp, _plant_single_unit)
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
    drop_unitmatch_selections_for(sort_pks)
    for sort_pk in sort_pks:
        for mid in (SpikeSortingOutput.CurationV2 & sort_pk).fetch("merge_id"):
            (SpikeSortingOutput & {"merge_id": mid}).super_delete(warn=False)
        (CurationV2 & sort_pk).super_delete(warn=False)
        (Sorting & sort_pk).super_delete(warn=False)
        (SortingSelection & sort_pk).super_delete(warn=False)


@pytest.mark.slow
def test_matcher_parameters_rejects_unknown_matcher(dj_conn):
    """An unregistered matcher name raises at insert, before commit."""
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
    """Per-matcher Pydantic dispatch rejects an out-of-range param."""
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
    """The bulk insert() path (not just insert1) rejects a typo'd
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
    saved_preparers = dict(mp._PREPARER_REGISTRY)
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
            MatcherParameters & {"matcher_params_name": "other_same_params"}
        ).super_delete(warn=False)
        mp._MATCHER_REGISTRY.clear()
        mp._MATCHER_REGISTRY.update(saved_m)
        mp._SCHEMA_REGISTRY.clear()
        mp._SCHEMA_REGISTRY.update(saved_s)
        mp._PREPARER_REGISTRY.clear()
        mp._PREPARER_REGISTRY.update(saved_preparers)


@pytest.mark.slow
def test_insert_selection_idempotent_and_hash_sensitive(
    two_session_curated_group,
):
    """Identical inputs return the same id; a changed curation differs."""
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
    accepted_children = []
    pk = None
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
            child = CurationEvaluation().use_evaluation_labels(sel)
            accepted_children.append(child)
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
        fetched = UnitMatch().make_fetch(pk)
        # Member a is recorded before member b, so input_index follows
        # member_index for this group.
        plan_by_member = {
            int(plan["input_index"]): plan for plan in fetched.input_plan
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
        from spyglass.spikesorting.spikesorting_merge import (
            SpikeSortingOutput,
        )

        if pk is not None:
            (UnitMatchSelection & pk).super_delete(warn=False)
        # The accepted children are curation_evaluation curations of the
        # shared members; left behind, a later auto_curated plan would find
        # them. Accepting registers each child on SpikeSortingOutput, so drop
        # the merge master (it cascades to its CurationV2 part) before the
        # curation -- DataJoint refuses to delete the part ahead of its master.
        for child in accepted_children:
            for mid in (SpikeSortingOutput.CurationV2 & child).fetch(
                "merge_id"
            ):
                (SpikeSortingOutput & {"merge_id": mid}).super_delete(
                    warn=False
                )
            (CurationV2 & child).super_delete(warn=False)


def _plant_sort_on_first_member(grp, sorter_params_name, samples_by_unit):
    """Plant a deterministic mountainsort5 sort on member 0's recording.

    Unit ``i`` fires at ``samples_by_unit[i]`` (frame indices). A distinct
    params name gives a distinct content-addressed sort on the SAME recording,
    so the planted sort shares member 0's identity without colliding with the
    fixture's single-unit sort. The sort is PLANTED (``plant_sorter``),
    so the params are never run; a real sorter name keeps the
    units ``sorted_units``. Returns ``(sort_key, sorted_unit_ids)``; the caller
    owns teardown (clear_curations_for + dropping the sort/params).
    """
    import numpy as np
    import spikeinterface as si

    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        Sorting,
        SortingSelection,
    )

    rec_id = SortingSelection.resolve_source(grp["sort_pks"][0]).key[
        "recording_id"
    ]
    default_ms5 = (
        SorterParameters
        & {"sorter": "mountainsort5", "sorter_params_name": _MINIREC_MS5_PARAMS}
    ).fetch1("params")
    SorterParameters.insert1(
        {
            "sorter": "mountainsort5",
            "sorter_params_name": sorter_params_name,
            "params": dict(default_ms5),
        },
        skip_duplicates=True,
        allow_duplicate_params=True,
    )
    samples = np.concatenate(
        [np.asarray(unit, dtype=np.int64) for unit in samples_by_unit]
    )
    labels = np.concatenate(
        [
            np.full(len(unit), label, dtype=np.int32)
            for label, unit in enumerate(samples_by_unit)
        ]
    )

    def _plant(
        sorter,
        sorter_params,
        recording,
        sorting_id,
        *,
        job_kwargs=None,
        execution_params=None,
        statistics_spans=None,
    ):
        return si.NumpySorting.from_samples_and_labels(
            samples_list=[samples],
            labels_list=[labels],
            sampling_frequency=recording.get_sampling_frequency(),
        )

    sort_key = SortingSelection.insert_selection(
        {
            "recording_id": rec_id,
            "sorter": "mountainsort5",
            "sorter_params_name": sorter_params_name,
        }
    )
    mp = pytest.MonkeyPatch()
    try:
        plant_sorter(mp, _plant)
        if not (Sorting & sort_key):
            Sorting.populate(sort_key, reserve_jobs=False)
    finally:
        mp.undo()
    unit_ids = sorted(
        int(u) for u in (Sorting.Unit & sort_key).fetch("unit_id")
    )
    return sort_key, unit_ids


def _plant_two_unit_sort_on_first_member(grp):
    """Plant a deterministic 2-unit mountainsort5 sort on member 0's recording.

    The chronic fixture's sort yields a single unit; a real merge (or a real
    proposed merge) needs >=2. Returns
    ``(two_unit_sort_key, sorted_unit_ids, sorter_params_name)``; the caller owns
    teardown (clear_curations_for + dropping the sort/params).
    """
    two_unit_params = "minirec_ms5_two_unit"
    two_unit_sort, unit_ids = _plant_sort_on_first_member(
        grp, two_unit_params, [[500, 1000, 1500], [600, 1100, 1600]]
    )
    assert len(unit_ids) >= 2, "planted sort must yield >=2 units"
    return two_unit_sort, unit_ids, two_unit_params


@pytest.mark.slow
def test_unitmatch_rejects_curation_evaluation_preview_child(
    two_session_curated_group, curation_evaluation_defaults
):
    """A ``CurationEvaluation`` preview child is not a UnitMatch input.

    ``preview_merges`` records proposed merges without applying them;
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
    from tests.spikesorting.v2._unitmatch_helpers import reforge_selection

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
        preview0 = CurationEvaluation().preview_merges(
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
        # with the member-0 ROOT, then point input 0 at the preview curation
        # (its id and generation) and store the matching hash, so the preview
        # guard -- not the hash or the snapshot check -- fires.
        pk = UnitMatchSelection.insert_selection(
            grp["owner"],
            grp["group_name"],
            "unitmatch_default",
            {0: root_choice, 1: grp["choices"][1]},
        )
        reforge_selection(
            pk,
            input_edits={
                0: {
                    "curation_id": preview0["curation_id"],
                    "curation_uuid": (CurationV2 & preview0).fetch1(
                        "curation_uuid"
                    ),
                }
            },
            rehash=True,
        )
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
                p for p in fetched.input_plan if int(p["input_index"]) == 0
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
    """A deterministic master whose Input parts no longer realize its
    input_set_hash (here: every part was dropped) is rejected by
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
    # (same deterministic id + input_set_hash, no parts) -- a raw-insert
    # orphan that DataJoint parts cannot be deleted into directly.
    master_row = (UnitMatchSelection & pk).fetch1()
    (UnitMatchSelection & pk).super_delete(warn=False)
    UnitMatchSelection.insert1(master_row, allow_direct_insert=True)
    try:
        with pytest.raises(SchemaBypassError, match="input_set_hash"):
            UnitMatchSelection.insert_selection(
                grp["owner"],
                grp["group_name"],
                "unitmatch_default",
                grp["choices"],
            )
    finally:
        (UnitMatchSelection & pk).super_delete(warn=False)


@pytest.mark.slow
def test_insert_selection_rejects_orphan_recording_parts_with_matching_hash(
    two_session_curated_group,
):
    """A forged master whose parts still hash to its input_set_hash but carry
    an InputRecording row for an input that does not exist is rejected by
    insert_selection. The hash ignores recording rows without an input, so the
    structure check -- not the hash -- catches the orphan.
    """
    import uuid

    from spyglass.spikesorting.v2.exceptions import SchemaBypassError
    from spyglass.spikesorting.v2.unit_matching import UnitMatchSelection
    from tests.spikesorting.v2._unitmatch_helpers import reforge_selection

    grp = two_session_curated_group
    pk = UnitMatchSelection.insert_selection(
        grp["owner"], grp["group_name"], "unitmatch_default", grp["choices"]
    )
    template = (
        UnitMatchSelection.InputRecording & pk & {"input_index": 0}
    ).fetch1()
    orphan = {
        **{k: v for k, v in template.items() if k != "unitmatch_id"},
        "input_index": 2,
        "recording_id": uuid.uuid4(),
    }
    master = reforge_selection(pk, extra_recordings=[orphan])
    stored = (UnitMatchSelection & pk).fetch1("input_set_hash")
    assert stored == master["input_set_hash"]
    try:
        with pytest.raises(SchemaBypassError, match="missing input_index"):
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
    """A curation from another member is rejected, atomically."""
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
    """Missing a member's choice is rejected before any insert."""
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
def test_make_rechecks_input_provenance(two_session_curated_group):
    """A direct-inserted selection whose frozen recordings belong to the other
    input fails make() before any matcher input is extracted; no UnitMatch /
    Pair rows are created, even though its stored hash agrees with its parts.
    """
    from spyglass.spikesorting.v2.exceptions import (
        UnitMatchSelectionIntegrityError,
    )
    from spyglass.spikesorting.v2.unit_matching import (
        UnitMatch,
        UnitMatchSelection,
    )
    from tests.spikesorting.v2._unitmatch_helpers import reforge_selection

    grp = two_session_curated_group
    pk = UnitMatchSelection.insert_selection(
        grp["owner"], grp["group_name"], "unitmatch_default", grp["choices"]
    )
    fields = (
        "nwb_file_name",
        "interval_list_name",
        "recording_id",
        "recording_content_hash",
    )
    rows = {
        int(row["input_index"]): row
        for row in (UnitMatchSelection.InputRecording & pk).fetch(as_dict=True)
    }
    # Swap the two inputs' frozen recordings: each curation now claims the
    # other member's recording.
    reforge_selection(
        pk,
        recording_edits={
            (0, 0): {field: rows[1][field] for field in fields},
            (1, 0): {field: rows[0][field] for field in fields},
        },
        rehash=True,
    )
    try:
        with pytest.raises(
            UnitMatchSelectionIntegrityError, match="frozen snapshot"
        ):
            UnitMatch.populate(pk, reserve_jobs=False)
        assert len(UnitMatch & pk) == 0
        assert len(UnitMatch.Pair & pk) == 0
    finally:
        (UnitMatchSelection & pk).super_delete(warn=False)


@pytest.mark.slow
def test_make_rejects_input_set_hash_mismatch(two_session_curated_group):
    """A raw-insert master whose stored input_set_hash disagrees with its
    (otherwise valid) Input / InputRecording rows is rejected by make_fetch.
    Without the recompute, such a master could claim one input set in its hash
    while matching on the curations pinned by its parts.
    """
    from spyglass.spikesorting.v2.exceptions import (
        UnitMatchSelectionIntegrityError,
    )
    from spyglass.spikesorting.v2.unit_matching import (
        UnitMatch,
        UnitMatchSelection,
    )
    from tests.spikesorting.v2._unitmatch_helpers import reforge_selection

    grp = two_session_curated_group
    pk = UnitMatchSelection.insert_selection(
        grp["owner"], grp["group_name"], "unitmatch_default", grp["choices"]
    )
    reforge_selection(pk, master_edits={"input_set_hash": "0" * 64})
    try:
        with pytest.raises(
            UnitMatchSelectionIntegrityError, match="input_set_hash"
        ):
            UnitMatch.populate(pk, reserve_jobs=False)
        assert len(UnitMatch & pk) == 0
    finally:
        (UnitMatchSelection & pk).super_delete(warn=False)


@pytest.mark.slow
def test_make_rejects_malformed_input_parts(two_session_curated_group):
    """An input left without its frozen recording rows is rejected by make()
    before any hash or source check."""
    from spyglass.spikesorting.v2.exceptions import (
        UnitMatchSelectionIntegrityError,
    )
    from spyglass.spikesorting.v2.unit_matching import (
        UnitMatch,
        UnitMatchSelection,
    )
    from tests.spikesorting.v2._unitmatch_helpers import reforge_selection

    grp = two_session_curated_group
    pk = UnitMatchSelection.insert_selection(
        grp["owner"], grp["group_name"], "unitmatch_default", grp["choices"]
    )
    reforge_selection(pk, drop_recordings=[(1, 0)], rehash=True)
    try:
        with pytest.raises(UnitMatchSelectionIntegrityError, match="malformed"):
            UnitMatch.populate(pk, reserve_jobs=False)
        assert len(UnitMatch & pk) == 0
    finally:
        (UnitMatchSelection & pk).super_delete(warn=False)


@pytest.mark.slow
def test_degenerate_single_session_zero_pairs(two_session_curated_group):
    """A one-member group produces zero pairs (no matcher call)."""
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
    assert row["n_pairs"] == 0
    assert len(UnitMatch.Pair & pk) == 0
    # The empty NWB pairs table round-trips to an empty DataFrame.
    assert len(UnitMatch().get_pairs(pk)) == 0


@pytest.mark.slow
def test_degenerate_single_session_needs_no_member_traces(
    two_session_curated_group, monkeypatch
):
    """A one-member group writes its zero-pair row without reading the
    member's traces, so a traces file that is missing and cannot be rebuilt
    does not fail the populate."""
    from pathlib import Path

    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2.recording import Recording
    from spyglass.spikesorting.v2.sorting import SortingSelection
    from spyglass.spikesorting.v2.unit_matching import (
        UnitMatch,
        UnitMatchSelection,
    )

    grp = two_session_curated_group
    choice = grp["choices"][0]
    pk = UnitMatchSelection.insert_selection(
        grp["owner"], grp["solo_name"], "unitmatch_default", {0: choice}
    )
    (UnitMatch & pk).super_delete(warn=False)
    traces = SortingSelection.resolve_effective_source(
        {"sorting_id": choice["sorting_id"]}
    ).traces
    assert traces.kind == "recording"
    traces_path = Path(
        AnalysisNwbfile.get_abs_path(traces.row["analysis_file_name"])
    )
    aside = traces_path.with_name(traces_path.name + ".aside")

    def _unrebuildable(self, key):
        raise RuntimeError("simulated unrebuildable traces file")

    monkeypatch.setattr(Recording, "_rebuild_nwb_artifact", _unrebuildable)
    traces_path.rename(aside)
    try:
        UnitMatch.populate(pk, reserve_jobs=False)
        assert (UnitMatch & pk).fetch1("n_pairs") == 0
    finally:
        aside.rename(traces_path)
        (UnitMatch & pk).super_delete(warn=False)


@pytest.mark.slow
def test_selection_without_inputs_is_rejected(two_session_curated_group):
    """A selection needs at least one input: no explicit input is rejected
    before any row is written."""
    from spyglass.spikesorting.v2.unit_matching import UnitMatchSelection

    before = len(UnitMatchSelection())
    with pytest.raises(ValueError, match="at least one"):
        UnitMatchSelection.insert_inputs([], "unitmatch_default")
    assert len(UnitMatchSelection()) == before


@pytest.mark.usefixtures("dj_conn")
def test_unitmatch_computed_matches_make_insert_signature():
    """The tri-part dispatch splats ``UnitMatchComputed`` POSITIONALLY into
    ``make_insert(key, *computed)``, so the NamedTuple field order is a wire
    contract. The per-recording spike counts follow the matchable universe and
    the three provenance fields close both -- a misalignment would silently
    mis-bind the list- and str-adjacent slots without a TypeError."""
    import inspect

    from spyglass.spikesorting.v2.unit_matching import (
        UnitMatch,
        UnitMatchComputed,
    )

    params = list(inspect.signature(UnitMatch.make_insert).parameters)
    assert params[:2] == ["self", "key"]
    assert tuple(params[2:]) == UnitMatchComputed._fields


@pytest.mark.usefixtures("dj_conn")
def test_unitmatch_fetched_matches_make_compute_signature():
    """The tri-part dispatch splats ``UnitMatchFetched`` POSITIONALLY into
    ``make_compute(key, *fetched)``, so the NamedTuple field order is a wire
    contract. The selection-identity fields were appended to both -- a
    misalignment would silently mis-bind the str-adjacent slots without a
    TypeError."""
    import inspect

    from spyglass.spikesorting.v2.unit_matching import (
        UnitMatch,
        UnitMatchFetched,
    )

    params = list(inspect.signature(UnitMatch.make_compute).parameters)
    assert params[:2] == ["self", "key"]
    assert tuple(params[2:]) == UnitMatchFetched._fields


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
        "spyglass.spikesorting.v2._matching.unitmatch_backend"
    )
    # The single-session path never calls UnitMatchPy, so it also runs (and
    # records NULL) where the optional package is not installed.
    try:
        expected_version = importlib.metadata.version("unitmatchpy")
    except importlib.metadata.PackageNotFoundError:
        expected_version = None
    assert row["matcher_backend_version"] == expected_version
    provenance = row["matcher_provenance"]
    assert provenance["backend"]["qualified_name"] == (
        "spyglass.spikesorting.v2._matching.unitmatch_backend.UnitMatchBackend"
    )
    assert provenance["backend"]["version"] == expected_version
    assert provenance["preparer"]["qualified_name"] == (
        "spyglass.spikesorting.v2._matching.unitmatch_backend.UnitMatchInputPreparer"
    )
    assert provenance["preparer"]["version"] == provenance["spyglass_version"]


@pytest.mark.serial
def test_matcher_provenance_upgrade_preserves_existing_runs(
    two_session_curated_group, tmp_path
):
    """Run the documented nullable-column upgrade against a retained match."""
    import re
    from pathlib import Path

    import datajoint as dj

    from spyglass.spikesorting.v2.unit_matching import (
        UnitMatch,
        UnitMatchSelection,
    )
    from tests.spikesorting.v2.test_preproduction_migration import (
        _column_ddl,
        _restore_column_snapshot,
        _run_script,
    )

    grp = two_session_curated_group
    key = UnitMatchSelection.insert_selection(
        grp["owner"],
        grp["solo_name"],
        "unitmatch_default",
        {0: grp["choices"][0]},
    )
    UnitMatch.populate(key, reserve_jobs=False)
    before = UnitMatch.fetch(as_dict=True)
    column = "matcher_provenance"
    definitions = {column: _column_ddl(UnitMatch, column)}
    snapshot = UnitMatch.proj(column).fetch(as_dict=True)
    config_file = tmp_path / "test-db.json"
    dj.config.save(str(config_file))
    config_file.chmod(0o600)
    document = (
        Path(__file__).resolve().parents[3]
        / "docs/src/Features/SpikeSortingV2_Migration.md"
    ).read_text()
    section = document.split("### Adding matcher producer provenance", 1)[1]
    code = re.search(r"```python\n(.*?)\n```", section, re.DOTALL).group(1)
    code = code.replace(".alter(context=", ".alter(prompt=False, context=")
    script = tmp_path / "upgrade.py"
    script.write_text(
        f"import datajoint as dj\ndj.config.load({str(config_file)!r})\n" + code
    )
    try:
        UnitMatch.connection.query(
            f"ALTER TABLE {UnitMatch.full_table_name} DROP COLUMN `{column}`"
        )
        for _ in range(2):
            result = _run_script(script)
            assert result.returncode == 0, result.stdout + result.stderr
        upgraded = dj.FreeTable(dj.conn(), UnitMatch.full_table_name).fetch(
            as_dict=True
        )
        assert all(row[column] is None for row in upgraded)
        assert [
            {k: v for k, v in row.items() if k != column} for row in upgraded
        ] == [{k: v for k, v in row.items() if k != column} for row in before]
    finally:
        config_file.unlink(missing_ok=True)
        _restore_column_snapshot(UnitMatch, definitions, snapshot)


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
            CurationV2.get_unit_semantics({"sorting_id": sort_pk["sorting_id"]})
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
    """A raw Pair referencing a unit absent from the pinned curation is
    rejected. The validated ``Pair.insert`` guard fronts the DataJoint foreign
    key: the bogus pair's second endpoint pins member 1's curation, which is
    NOT a pinned ``UnitMatchSelection.Input`` of the solo selection, so the
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
        grp["owner"],
        grp["solo_name"],
        "unitmatch_default",
        {0: grp["choices"][0]},
    )
    UnitMatch.populate(pk, reserve_jobs=False)
    side_a, side_b = grp["choices"][0], grp["choices"][1]
    bogus = {
        **pk,
        "pair_index": 999,
        "session_a_sorting_id": side_a["sorting_id"],
        "session_a_curation_id": side_a["curation_id"],
        "unit_a_id": 10**9,  # no such curated unit
        "session_b_sorting_id": side_b[
            "sorting_id"
        ],  # member 1: not pinned here
        "session_b_curation_id": side_b["curation_id"],
        "unit_b_id": 10**9,
        "match_probability": 0.9,
    }
    with pytest.raises(UnitMatchPairIntegrityError, match="not a pinned"):
        UnitMatch.Pair.insert1(bogus, allow_direct_insert=True)


@pytest.mark.parametrize(
    "field",
    [
        "session_a_curation_id",
        "session_b_curation_id",
        "unit_a_id",
        "unit_b_id",
    ],
)
@pytest.mark.parametrize("value", [17.9, True, "17"])
def test_raw_pair_insert_rejects_noninteger_identifiers(dj_conn, field, value):
    import uuid

    from spyglass.spikesorting.v2.exceptions import UnitMatchPairIntegrityError
    from spyglass.spikesorting.v2.unit_matching import UnitMatch

    row = {
        "unitmatch_id": uuid.UUID(int=1),
        "pair_index": 0,
        "session_a_sorting_id": uuid.UUID(int=2),
        "session_a_curation_id": 17,
        "unit_a_id": 17,
        "session_b_sorting_id": uuid.UUID(int=3),
        "session_b_curation_id": 17,
        "unit_b_id": 17,
        "match_probability": 0.9,
        field: value,
    }
    with pytest.raises(
        UnitMatchPairIntegrityError, match=field + " must be an integer"
    ):
        UnitMatch.Pair.insert1(row, allow_direct_insert=True)


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
        grp["owner"],
        grp["solo_name"],
        "unitmatch_default",
        {0: grp["choices"][0]},
    )
    UnitMatch.populate(pk, reserve_jobs=False)
    TrackedUnit.populate(pk, reserve_jobs=False)

    n_matchable = len(CurationV2().get_matchable_unit_ids(grp["choices"][0]))
    tracked = (TrackedUnit & pk).fetch(as_dict=True)
    assert len(tracked) == n_matchable
    for row in tracked:
        # No pairs -> every unit is its own singleton tracked unit.
        assert row["n_sessions_detected"] == 1
        assert row["n_matching_inputs"] == 1
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
    fixture matcher reads its pairs from ``params``, not the bundles). The
    backend recipe omits grouping controls, exercising shared tracking defaults.
    """
    from pydantic import BaseModel, ConfigDict, Field

    from spyglass.spikesorting.v2 import matcher_protocol as mp
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.matcher_protocol import (
        MatchPair,
        PreparedMatcherInput,
        SessionMatcherInput,
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
        probability: float = 0.99
        pairs: list = Field(default_factory=list)
        schema_version: int = 1

    grp = two_session_curated_group
    matchable_a = CurationV2().get_matchable_unit_ids(grp["choices"][0])
    matchable_b = CurationV2().get_matchable_unit_ids(grp["choices"][1])
    assert len(matchable_a) and len(
        matchable_b
    ), "planted units must be matchable"
    unit_a, unit_b = int(matchable_a[0]), int(matchable_b[0])

    class _FixturePairer:
        """Emits the (unit_a, unit_b) pairs listed in ``params``."""

        name = "fixture_pairer"

        def match(self, session_inputs, params):
            left = session_inputs[0].curation_key
            right = session_inputs[1].curation_key
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
        return []

    saved_matchers = dict(mp._MATCHER_REGISTRY)
    saved_schemas = dict(mp._SCHEMA_REGISTRY)
    saved_preparers = dict(mp._PREPARER_REGISTRY)

    class FixtureInputPreparer:
        def prepare(self, source, directory, params, job_kwargs):
            _noop_extract(directory, source.recording, source.sorting)
            return PreparedMatcherInput(
                SessionMatcherInput(
                    curation_key=dict(source.curation_key),
                    bundle_dir=directory,
                    geometry_path=directory / "channel_positions.npy",
                    recording_date=source.recording_date,
                )
            )

    register_matcher(
        _FixturePairer(),
        _FixtureMatcherParams,
        input_preparer=FixtureInputPreparer(),
    )
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
            if row["n_sessions_detected"] == 2
        ]
        assert len(matched) == 1
        assert matched[0]["median_match_probability"] == pytest.approx(0.99)
        assert (
            len(
                TrackedUnit.Member
                & {
                    **selection_pk,
                    "tracked_unit_id": matched[0]["tracked_unit_id"],
                }
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
            MatcherParameters & {"matcher_params_name": "fixture_pairer_params"}
        ).super_delete(warn=False)
        mp._MATCHER_REGISTRY.clear()
        mp._MATCHER_REGISTRY.update(saved_matchers)
        mp._SCHEMA_REGISTRY.clear()
        mp._SCHEMA_REGISTRY.update(saved_schemas)
        mp._PREPARER_REGISTRY.clear()
        mp._PREPARER_REGISTRY.update(saved_preparers)


@pytest.mark.slow
def test_make_compute_reads_fetched_members_and_only_stages_output(
    two_session_curated_group, monkeypatch
):
    """``make_fetch`` resolves each member's traces file and curated units
    NWB (its hash is stable across DataJoint's two fetches), and
    ``make_compute`` builds each bundle from those files with no DB access
    beyond staging the pairs NWB. The bundle inputs equal what
    ``CurationV2.get_recording`` / ``get_sorting`` return for the member."""
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.matcher_protocol import get_input_preparer
    from spyglass.spikesorting.v2.recording import (
        _unlink_staged_analysis_file,
    )
    from spyglass.spikesorting.v2.unit_matching import (
        MatcherParameters,
        UnitMatch,
        UnitMatchSelection,
    )
    from tests.spikesorting.v2._tripart_helpers import (
        fetch_hash,
        forbid_db_queries,
    )

    grp = two_session_curated_group
    saved = _install_fixture_pairer(
        monkeypatch,
        matcher_name="fixture_pairer_reads",
        matcher_params_name="fixture_pairer_reads_params",
        pairs=[],
    )
    bundle_inputs = {}

    def _capture(session_dir, recording, sorting, **kwargs):
        Path(session_dir).mkdir(parents=True, exist_ok=True)
        bundle_inputs[Path(session_dir).name] = (recording, sorting)
        return []

    # Observe the registered preparer, which need not use UnitMatch's
    # compatibility extraction entry point.
    monkeypatch.setattr(
        get_input_preparer("fixture_pairer_reads"), "extract", _capture
    )
    selection_pk = None
    try:
        selection_pk = UnitMatchSelection.insert_selection(
            grp["owner"],
            grp["group_name"],
            "fixture_pairer_reads_params",
            grp["choices"],
        )
        table = UnitMatch()
        fetched = table.make_fetch(selection_pk)
        assert fetch_hash(table.make_fetch(selection_pk)) == fetch_hash(fetched)
        with forbid_db_queries(
            monkeypatch, "UnitMatch.make_compute", allow_staging=True
        ):
            computed = table.make_compute(selection_pk, *fetched)
        _unlink_staged_analysis_file(
            computed.analysis_file_name, context="unitmatch read guard"
        )
        assert computed.n_pairs == 0

        # Member a is recorded first, so member_index equals input_index.
        assert sorted(bundle_inputs) == [
            f"input_{index}" for index in sorted(grp["choices"])
        ]
        for index, choice in grp["choices"].items():
            recording, sorting = bundle_inputs[f"input_{index}"]
            expected_recording = CurationV2.get_recording(choice)
            expected_sorting = CurationV2.get_sorting(choice).select_units(
                CurationV2().get_matchable_unit_ids(choice)
            )
            assert list(recording.channel_ids) == list(
                expected_recording.channel_ids
            )
            np.testing.assert_array_equal(
                recording.get_channel_locations(),
                expected_recording.get_channel_locations(),
            )
            assert (
                recording.get_sampling_frequency()
                == expected_recording.get_sampling_frequency()
            )
            assert (
                recording.get_num_samples()
                == expected_recording.get_num_samples()
            )
            np.testing.assert_array_equal(
                recording.get_traces(start_frame=0, end_frame=3000),
                expected_recording.get_traces(start_frame=0, end_frame=3000),
            )
            assert list(sorting.unit_ids) == list(expected_sorting.unit_ids)
            assert len(sorting.unit_ids) > 0
            for unit_id in sorting.unit_ids:
                np.testing.assert_array_equal(
                    sorting.get_unit_spike_train(unit_id),
                    expected_sorting.get_unit_spike_train(unit_id),
                )
    finally:
        if selection_pk is not None:
            (UnitMatch & selection_pk).super_delete(warn=False)
            (UnitMatchSelection & selection_pk).super_delete(warn=False)
        (
            MatcherParameters
            & {"matcher_params_name": "fixture_pairer_reads_params"}
        ).super_delete(warn=False)
        _restore_matcher_registry(saved)


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
        assert all(
            matchable.values()
        ), "planted accepted units must be matchable"
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
        # Member a is recorded first, so member_index equals input_index.
        frozen = {
            (
                int(row["input_index"]),
                str(row["sorting_id"]),
                int(row["curation_id"]),
                int(row["unit_id"]),
            )
            for row in (UnitMatch.MatchableUnit & selection_pk).fetch(
                as_dict=True
            )
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
            if row["n_sessions_detected"] == 2
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
        from spyglass.spikesorting.spikesorting_merge import (
            SpikeSortingOutput,
        )

        for child in accepted_children:
            # Accepting an evaluation registers the child curation on the
            # SpikeSortingOutput merge table, so drop the merge master (it
            # cascades to its CurationV2 part) BEFORE deleting the curation --
            # DataJoint refuses to delete the part ahead of its master.
            for mid in (SpikeSortingOutput.CurationV2 & child).fetch(
                "merge_id"
            ):
                (SpikeSortingOutput & {"merge_id": mid}).super_delete(
                    warn=False
                )
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
        assert member1_matchable, "member 1's planted unit must be matchable"
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
        assert str(pair["session_a_sorting_id"]) == str(
            merged_choice["sorting_id"]
        )
        assert int(pair["session_a_curation_id"]) == int(
            merged_choice["curation_id"]
        )
        assert int(pair["unit_a_id"]) == merged_uid
        assert int(pair["unit_b_id"]) == unit_b

        TrackedUnit.populate(selection_pk, reserve_jobs=False)
        matched = [
            row
            for row in (TrackedUnit & selection_pk).fetch(as_dict=True)
            if row["n_sessions_detected"] == 2
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


#: Member 0's planted sort for the bundle-exclusion tests: unit 0 fires once
#: mid-session (one sampled spike -> no two cross-validation halves, so the
#: bundle leaves it out); unit 1 fires 35 times across the session (kept).
_ONE_SPIKE_BESIDE_KEPT_UNIT = [[75_000], list(range(3_000, 140_000, 4_000))]
#: Member 0's planted sort where every unit fires once (all left out).
_ALL_ONE_SPIKE_UNITS = [[50_000], [100_000]]


@contextmanager
def _real_bundle_selection(
    grp, monkeypatch, *, name, samples_by_unit, pairs, seen_unit_ids=None
):
    """A two-member UnitMatch selection built from real bundles.

    Plants ``samples_by_unit`` as member 0's sort (member 1 keeps the fixture's
    single-unit sort), registers a bundle-reading pairer that emits the listed
    ``pairs`` only between units present in the bundles, and inserts the
    selection. Yields ``selection_pk``, member ``choices``, and member 0's
    ``unit_spike_counts``; tears it all down afterwards.
    """
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        Sorting,
        SortingSelection,
    )
    from spyglass.spikesorting.v2.unit_matching import (
        MatcherParameters,
        UnitMatch,
        UnitMatchSelection,
    )
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    sorter_params_name = f"minirec_ms5_{name}"
    matcher_params_name = f"{name}_pairer_params"
    sort_key, _ = _plant_sort_on_first_member(
        grp, sorter_params_name, samples_by_unit
    )
    sorting_key = {"sorting_id": sort_key["sorting_id"]}
    clear_curations_for(sort_key)
    saved_registry = None
    selection_pk = None
    try:
        root0 = CurationV2.insert_curation(sorting_key=sorting_key)
        choice0 = {
            "sorting_id": root0["sorting_id"],
            "curation_id": root0["curation_id"],
        }
        sorting0 = CurationV2.get_sorting(choice0)
        unit_spike_counts = {
            int(u): len(sorting0.get_unit_spike_train(u))
            for u in sorting0.unit_ids
        }
        assert sorted(unit_spike_counts.values()) == sorted(
            len(unit) for unit in samples_by_unit
        ), "precondition: the curated sort keeps every planted spike"
        saved_registry = _install_fixture_pairer(
            monkeypatch,
            matcher_name=f"{name}_pairer",
            matcher_params_name=matcher_params_name,
            pairs=pairs(unit_spike_counts),
            seen_unit_ids=seen_unit_ids,
            read_bundles=True,
        )
        choices = {0: choice0, 1: grp["choices"][1]}
        selection_pk = UnitMatchSelection.insert_selection(
            grp["owner"], grp["group_name"], matcher_params_name, choices
        )
        yield {
            "selection_pk": selection_pk,
            "choices": choices,
            "unit_spike_counts": unit_spike_counts,
        }
    finally:
        if selection_pk is not None:
            (UnitMatch & selection_pk).super_delete(warn=False)
            (UnitMatchSelection & selection_pk).super_delete(warn=False)
        (
            MatcherParameters & {"matcher_params_name": matcher_params_name}
        ).super_delete(warn=False)
        if saved_registry is not None:
            _restore_matcher_registry(saved_registry)
        clear_curations_for(sort_key)
        (Sorting & sorting_key).super_delete(warn=False)
        (SortingSelection & sorting_key).super_delete(warn=False)
        (
            SorterParameters & {"sorter_params_name": sorter_params_name}
        ).super_delete(warn=False)


def _units_with_spike_count(unit_spike_counts, n_spikes):
    """Unit ids of member 0's planted sort that fire exactly ``n_spikes``."""
    return [u for u, n in unit_spike_counts.items() if n == n_spikes]


def _member_identity(grp, member_index, choice):
    """The input identity fields a UnitMatch message must name.

    Member a is recorded before member b, so a member's input_index equals
    its member_index for this group.
    """
    return [
        f"input_index {member_index}",
        grp["members"][member_index]["nwb_file_name"],
        f"sorting_id={choice['sorting_id']}",
        f"curation_id={choice['curation_id']}",
    ]


@pytest.mark.slow
def test_make_logs_exclusions_per_session(
    two_session_curated_group, monkeypatch, caplog
):
    """A member whose bundle leaves a unit out gets exactly one warning.

    Member 0 plants a one-spike unit beside a 35-spike unit; member 1's single
    unit fires 30 times. The real bundle extraction leaves member 0's
    one-spike unit out, and ``UnitMatch.populate`` warns once, naming member
    0's identity and the excluded unit id, and never for member 1.
    """
    import logging

    pytest.importorskip("UnitMatchPy")
    from spyglass.spikesorting.v2.unit_matching import UnitMatch

    grp = two_session_curated_group
    with _real_bundle_selection(
        grp,
        monkeypatch,
        name="exclusion_log",
        samples_by_unit=_ONE_SPIKE_BESIDE_KEPT_UNIT,
        pairs=lambda counts: [],
    ) as run:
        (excluded,) = _units_with_spike_count(run["unit_spike_counts"], 1)
        with caplog.at_level(logging.WARNING, logger="spyglass"):
            UnitMatch.populate(run["selection_pk"], reserve_jobs=False)
        assert UnitMatch & run["selection_pk"]

    exclusion_warnings = [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING
        and "fewer than two sampled spikes" in r.getMessage()
    ]
    assert len(exclusion_warnings) == 1, exclusion_warnings
    (message,) = exclusion_warnings
    for field in _member_identity(grp, 0, run["choices"][0]):
        assert field in message, (field, message)
    assert f"[{excluded}]" in message, message
    member1 = run["choices"][1]
    assert f"sorting_id={member1['sorting_id']}" not in message
    assert grp["members"][1]["nwb_file_name"] not in message


@pytest.mark.slow
def test_excluded_bundle_units_stay_in_frozen_universe(
    two_session_curated_group, monkeypatch
):
    """A unit the bundle leaves out stays in the frozen matchable universe.

    Planted correspondence: both of member 0's units match member 1's unit.
    The pairer reads the real bundles, so only the kept 35-spike unit can pair;
    the one-spike unit is absent from its bundle, is still a ``MatchableUnit``
    row, is in no ``Pair`` row, and becomes a singleton ``TrackedUnit``.
    """
    pytest.importorskip("UnitMatchPy")
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.unit_matching import (
        TrackedUnit,
        UnitMatch,
    )

    grp = two_session_curated_group
    (unit_b,) = [
        int(u) for u in CurationV2().get_matchable_unit_ids(grp["choices"][1])
    ]
    seen_unit_ids: list[list[int]] = []
    with _real_bundle_selection(
        grp,
        monkeypatch,
        name="exclusion_universe",
        samples_by_unit=_ONE_SPIKE_BESIDE_KEPT_UNIT,
        pairs=lambda counts: [[u, unit_b] for u in sorted(counts)],
        seen_unit_ids=seen_unit_ids,
    ) as run:
        pk = run["selection_pk"]
        choice0 = run["choices"][0]
        (excluded,) = _units_with_spike_count(run["unit_spike_counts"], 1)
        (kept,) = _units_with_spike_count(run["unit_spike_counts"], 35)
        UnitMatch.populate(pk, reserve_jobs=False)

        # The bundles hold exactly the units with two halves.
        assert seen_unit_ids == [[kept], [unit_b]]

        member0_universe = {
            int(r["unit_id"])
            for r in (UnitMatch.MatchableUnit & pk & choice0).fetch(
                as_dict=True
            )
        }
        assert member0_universe == {excluded, kept}

        pairs = (UnitMatch.Pair & pk).fetch(as_dict=True)
        assert [(int(p["unit_a_id"]), int(p["unit_b_id"])) for p in pairs] == [
            (kept, unit_b)
        ]
        assert str(pairs[0]["session_a_sorting_id"]) == str(
            choice0["sorting_id"]
        )

        TrackedUnit.populate(pk, reserve_jobs=False)
        members = (TrackedUnit.Member & pk).fetch(as_dict=True)
        tracked_of = {
            (str(m["sorting_id"]), int(m["unit_id"])): m["tracked_unit_id"]
            for m in members
        }
        sid0 = str(choice0["sorting_id"])
        sid1 = str(run["choices"][1]["sorting_id"])
        excluded_tracked = tracked_of[(sid0, excluded)]
        assert tracked_of[(sid0, kept)] == tracked_of[(sid1, unit_b)]
        assert excluded_tracked != tracked_of[(sid0, kept)]
        excluded_row = (
            TrackedUnit & pk & {"tracked_unit_id": excluded_tracked}
        ).fetch1()
        # The excluded unit's one planted spike is still counted, so its
        # session is detected.
        assert {
            int(r["unit_id"]): int(r["n_spikes"])
            for r in (
                UnitMatch.RecordingSpikeCount & pk & {"input_index": 0}
            ).fetch(as_dict=True)
        } == {excluded: 1, kept: 35}
        assert excluded_row["n_sessions_detected"] == 1
        assert excluded_row["n_matching_inputs"] == 1
        assert excluded_row["median_match_probability"] is None


@pytest.mark.slow
def test_all_excluded_member_raises_with_member_identity(
    two_session_curated_group, monkeypatch
):
    """A member whose every unit is left out of its bundle fails the populate.

    The error names the member (not the temporary bundle directory), keeps
    the reason, and no UnitMatch row or staged pairs NWB is left behind.
    """
    pytest.importorskip("UnitMatchPy")
    from spyglass.spikesorting.v2._matching.unitmatch_backend import (
        NoMatchableUnitsError,
    )
    from spyglass.spikesorting.v2.unit_matching import UnitMatch
    from tests.spikesorting.v2._tripart_helpers import (
        assert_no_staged_analysis_files,
        record_created_analysis_files,
    )

    grp = two_session_curated_group
    with _real_bundle_selection(
        grp,
        monkeypatch,
        name="all_excluded",
        samples_by_unit=_ALL_ONE_SPIKE_UNITS,
        pairs=lambda counts: [],
    ) as run:
        with monkeypatch.context() as patch:
            created = record_created_analysis_files(patch)
            with pytest.raises(NoMatchableUnitsError) as excinfo:
                UnitMatch.populate(run["selection_pk"], reserve_jobs=False)
            assert not (UnitMatch & run["selection_pk"])
            assert_no_staged_analysis_files(created)

    message = str(excinfo.value)
    for field in _member_identity(grp, 0, run["choices"][0]):
        assert field in message, (field, message)
    assert "fewer than two sampled spikes" in message
    assert "unitmatch_" not in message, "names the temp dir, not the member"
    assert isinstance(excinfo.value.__cause__, NoMatchableUnitsError)


# --------------------------------------------------------------------------- #
# Ground-truth AUC gate (slow / heavy). Target: UnitMatch discriminates planted #
# cross-session correspondences on the polymer probe with AUC > 0.85. Verified  #
# LOCALLY only: the two-session polymer fixtures are unhosted (their _fetch.py  #
# URLs are None), so this skips wherever they or UnitMatchPy are absent. The    #
# nightly / manual CI step that runs it requires both fixtures, so that step    #
# fails, naming them, until they are uploaded and their URLs set.               #
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
    labels = np.concatenate([np.full(len(trains[u]), u) for u in trains])
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
    """Ship-readiness target (verified locally; its CI step fails until the
    fixtures are hosted -- see the gate banner above): AUC of UnitMatch
    probability vs ground-truth correspondence on the two-session polymer
    probe is > 0.85.

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
    from spyglass.spikesorting.v2._core.lookup_validation import (
        _validate_params,
    )
    from spyglass.spikesorting.v2._params.sorter import _get_sorter_schema
    from spyglass.spikesorting.v2._matching.unitmatch_backend import (
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
            session_filter = {"nwb_file_name": nwb_file_name}
            if not (SortGroupV2 & session_filter):
                SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
            sort_group_id = int(
                sorted((SortGroupV2 & session_filter).fetch("sort_group_id"))[0]
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
                    curation_key={
                        "sorting_id": f"polymer_gt_{index}",
                        "curation_id": 0,
                    },
                    bundle_dir=session_dir,
                    geometry_path=(session_dir / "channel_positions.npy"),
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
    ``Input`` curations -- an unpinned-curation endpoint, a same-curation edge,
    a reversed / duplicate edge, or an out-of-range ``match_probability`` --
    raises ``UnitMatchPairIntegrityError``.

    Validates against the pinned ``UnitMatchSelection.Input`` rows (created by
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
        members = (UnitMatchSelection.Input & pk).fetch(
            as_dict=True, order_by="input_index"
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

        # Endpoint pinned to a curation NOT among this selection's inputs
        # (a fake curation: the guard checks the pinned set before the FK fires).
        unpinned = (str(uuid.uuid4()), 0, 0)
        with pytest.raises(UnitMatchPairIntegrityError, match="not a pinned"):
            UnitMatch.Pair.insert1(pair_row(unpinned, node1))

        # Same-curation edge (both endpoints pin input 0): a unit cannot match
        # itself across sessions.
        with pytest.raises(UnitMatchPairIntegrityError, match="same input"):
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
        (CurationV2.UnitLabel & noise_label).delete_quick()
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

    from spyglass.spikesorting.v2._matching import (
        unitmatch_backend as _unitmatch_backend,
    )
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


@pytest.mark.parametrize("has_validator", [False, True])
def test_selection_geometry_dispatches_without_global_policy(
    dj_conn, monkeypatch, has_validator
):
    from pydantic import BaseModel

    from spyglass.spikesorting.v2 import matcher_protocol as protocol
    from spyglass.spikesorting.v2.unit_matching import UnitMatchSelection

    calls = []
    positions = {1: [[0, 0]], 2: [[0, 0], [0, 20]]}

    class Backend:
        name = "geometry_dispatch_fixture"

        def match(self, inputs, params):
            raise AssertionError("Preflight must not run inference")

    backend = Backend()
    if has_validator:
        backend.validate_geometry = lambda named, params: calls.append(
            (named, params)
        )

    # Isolate registry state without changing the built-in backend entry.
    monkeypatch.setitem(protocol._MATCHER_REGISTRY, backend.name, backend)
    monkeypatch.setitem(protocol._SCHEMA_REGISTRY, backend.name, BaseModel)

    def read_positions(key):
        if not has_validator:
            raise AssertionError(
                "Backend without geometry validation read geometry"
            )
        return positions[key["sorting_id"]]

    monkeypatch.setattr(
        UnitMatchSelection,
        "_member_channel_positions",
        staticmethod(read_positions),
    )
    params = {"allow_remapping": True}
    UnitMatchSelection._validate_matcher_geometry(
        {1: (2, 0), 0: (1, 0)}, backend.name, params
    )
    assert calls == (
        [([("input_0", positions[1]), ("input_1", positions[2])], params)]
        if has_validator
        else []
    )


@pytest.mark.slow
def test_unitmatch_nwb_self_describes(two_session_curated_group):
    """The UnitMatch NWB is interpretable standalone.

    The pairs table carries only side ids; without the DB a reader cannot tell
    which match run, session group, or matcher produced it. Assert the artifact
    embeds the run/group/matcher header -- re-emitting the same producer
    provenance stored on the ``UnitMatch`` row (matcher backend, backend
    version, SpikeInterface version) -- plus the per-input and per-recording
    maps.

    Uses the single-input (solo) path so the provenance write is exercised
    without the cross-session matcher (whose ground-truth correctness is the
    polymer gate's job). The header and both maps are written on this path
    too: ``write_pairs_table`` runs and the backend is resolved regardless of
    input count.
    """
    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2._storage.provenance import (
        UNITMATCH_INPUT_RECORDINGS,
        UNITMATCH_INPUTS,
        UNITMATCH_PROVENANCE,
        read_long_provenance,
        read_provenance_values,
    )
    from spyglass.spikesorting.v2.unit_matching import (
        UnitMatch,
        UnitMatchSelection,
    )

    grp = two_session_curated_group
    choice = grp["choices"][0]
    pk = UnitMatchSelection.insert_selection(
        grp["owner"], grp["solo_name"], "unitmatch_default", {0: choice}
    )
    UnitMatch.populate(pk, reserve_jobs=False)

    row = (UnitMatch & pk).fetch1()
    abs_path = AnalysisNwbfile.get_abs_path(row["analysis_file_name"])

    header = read_provenance_values(abs_path, UNITMATCH_PROVENANCE)
    assert header["unitmatch_id"] == str(pk["unitmatch_id"])
    assert header["session_group_owner"] == grp["owner"]
    assert header["session_group_name"] == grp["solo_name"]
    assert header["matcher_params_name"] == "unitmatch_default"
    # Re-emits the EXACT producer provenance stored on the row.
    assert header["matcher_backend"] == row["matcher_backend"]
    assert header["matcher_backend_version"] == row["matcher_backend_version"]
    assert header["spikeinterface_version"] == row["spikeinterface_version"]
    assert header["matcher_provenance"] == row["matcher_provenance"]

    inputs = read_long_provenance(abs_path, UNITMATCH_INPUTS)
    by_index = {m["input_index"]: m for m in inputs}
    assert set(by_index) == {0}
    got = by_index[0]
    assert got["sorting_id"] == str(choice["sorting_id"])
    assert got["curation_id"] == int(choice["curation_id"])
    assert got["source_kind"] == "recording"
    # A real session start time (ISO 8601), not a placeholder.
    assert "T" in got["input_start_time"]
    (recording,) = read_long_provenance(abs_path, UNITMATCH_INPUT_RECORDINGS)
    assert recording["input_index"] == 0
    assert recording["nwb_file_name"] == grp["members"][0]["nwb_file_name"]
    assert recording["session_start_time"] == got["input_start_time"]


@pytest.mark.slow
def test_get_unit_brain_regions_keeps_disambiguators(two_session_curated_group):
    """``TrackedUnit.get_unit_brain_regions`` returns the chronic-identity
    disambiguators resolved at this point -- ``unitmatch_id`` /
    ``tracked_unit_id`` / ``input_index`` / ``nwb_file_name`` /
    ``recording_date`` / ``curation_id`` -- not just ``sorting_id`` /
    ``unit_id`` / ``region_name``."""
    from spyglass.spikesorting.v2.unit_matching import (
        TrackedUnit,
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
    TrackedUnit.populate(pk, reserve_jobs=False)

    tracked_keys = (TrackedUnit & pk).fetch("KEY")
    assert tracked_keys, "no tracked units were populated"

    df = TrackedUnit().get_unit_brain_regions(tracked_keys[0])

    expected = {
        "unitmatch_id",
        "tracked_unit_id",
        "input_index",
        "nwb_file_name",
        "recording_date",
        "sorting_id",
        "curation_id",
        "unit_id",
        "region_name",
    }
    assert expected <= set(df.columns), (
        "get_unit_brain_regions dropped chronic-identity disambiguators; "
        f"columns={list(df.columns)}"
    )
    if len(df):
        for col in (
            "unitmatch_id",
            "tracked_unit_id",
            "input_index",
            "nwb_file_name",
            "recording_date",
            "curation_id",
        ):
            assert df[col].notna().all(), f"{col} must be populated, not null"


@pytest.mark.slow
def test_run_v2_unit_match_full_chain(two_session_curated_group, monkeypatch):
    """run_v2_unit_match: insert selection -> UnitMatch -> TrackedUnit in one call.

    ``describe_unit_match_choices`` surfaces each member's pinned root curation,
    and ``run_v2_unit_match`` drives the full cross-session chain to a
    tracked-unit manifest, idempotent on rerun. Uses a registered lightweight
    fixture matcher (emitting one pair, bundle extraction stubbed) so the match
    is deterministic without UnitMatchPy's multi-unit metric path -- the same
    substrate the table-layer matcher tests use.
    """
    from pydantic import BaseModel, ConfigDict, Field

    from spyglass.spikesorting.v2 import matcher_protocol as mp
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.matcher_protocol import (
        MatchPair,
        PreparedMatcherInput,
        SessionMatcherInput,
        register_matcher,
    )
    from spyglass.spikesorting.v2.pipeline import (
        describe_unit_match_choices,
        plan_v2_unit_match,
        run_v2_unit_match,
    )
    from spyglass.spikesorting.v2.unit_matching import (
        MatcherParameters,
        TrackedUnit,
        UnitMatch,
        UnitMatchSelection,
    )

    grp = two_session_curated_group

    # Discovery: each member's available curations include the pinned root.
    described = describe_unit_match_choices(grp["owner"], grp["group_name"])
    assert sorted(described["member_index"].unique().tolist()) == [0, 1]
    for index in (0, 1):
        pinned = grp["choices"][index]
        member_rows = described[described["member_index"] == index]
        assert (
            (member_rows["sorting_id"] == pinned["sorting_id"])
            & (member_rows["curation_id"] == pinned["curation_id"])
        ).any()

    matchable_a = CurationV2().get_matchable_unit_ids(grp["choices"][0])
    matchable_b = CurationV2().get_matchable_unit_ids(grp["choices"][1])
    assert len(matchable_a) and len(
        matchable_b
    ), "planted units must be matchable"
    unit_a, unit_b = int(matchable_a[0]), int(matchable_b[0])

    class _FixtureMatcherParams(BaseModel):
        model_config = ConfigDict(extra="forbid")
        tracked_unit_threshold: float = 0.5
        max_strict_nodes: int = 2000
        probability: float = 0.99
        pairs: list = Field(default_factory=list)
        schema_version: int = 1

    class _FixturePairer:
        name = "fixture_pairer"

        def match(self, session_inputs, params):
            left = session_inputs[0].curation_key
            right = session_inputs[1].curation_key
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
        return []

    saved_matchers = dict(mp._MATCHER_REGISTRY)
    saved_schemas = dict(mp._SCHEMA_REGISTRY)
    saved_preparers = dict(mp._PREPARER_REGISTRY)

    class FixtureInputPreparer:
        def prepare(self, source, directory, params, job_kwargs):
            _noop_extract(directory, source.recording, source.sorting)
            return PreparedMatcherInput(
                SessionMatcherInput(
                    curation_key=dict(source.curation_key),
                    bundle_dir=directory,
                    geometry_path=directory / "channel_positions.npy",
                    recording_date=source.recording_date,
                )
            )

    register_matcher(
        _FixturePairer(),
        _FixtureMatcherParams,
        input_preparer=FixtureInputPreparer(),
    )
    matcher_name = "fixture_pairer_run_params"
    selection_pk = None
    try:
        MatcherParameters().insert1(
            {
                "matcher_params_name": matcher_name,
                "matcher": "fixture_pairer",
                "params": {"pairs": [[unit_a, unit_b]], "probability": 0.99},
            },
            skip_duplicates=True,
        )

        # Plan-then-run: the members' root curations are pinned by a curation
        # curation strategy (each member has exactly one root), reviewed, then run --
        # exercising plan_v2_unit_match + the run_v2_unit_match(plan) overload.
        plan = plan_v2_unit_match(
            grp["owner"],
            grp["group_name"],
            curation_strategy="root",
            matcher_params_name=matcher_name,
        )
        assert plan.ok, plan.errors
        assert set(plan.curation_choices) == set(grp["choices"])
        assert not plan.as_dataframe().empty

        # describe_unit_match_choices surfaces curation_source (the
        # auto_curated curation strategy keys off it). The fixture members are
        # 'manual' roots, so
        # against real fetched data auto_curated resolves nothing -> not ok.
        described = describe_unit_match_choices(grp["owner"], grp["group_name"])
        # every offered curation row carries curation_source (auto_curated keys
        # off it); rows for a sortless member have a null sorting_id.
        offered = described[described["sorting_id"].notna()]
        assert offered["curation_source"].notna().all()
        auto_plan = plan_v2_unit_match(
            grp["owner"],
            grp["group_name"],
            curation_strategy="auto_curated",
            matcher_params_name=matcher_name,
        )
        assert not auto_plan.ok
        assert any("auto_curate" in e for e in auto_plan.errors)

        # Full chain via the orchestrator, driven by the plan.
        summary = run_v2_unit_match(plan)
        selection_pk = {"unitmatch_id": summary["unit_match_id"]}
        assert summary["session_group_name"] == grp["group_name"]
        assert summary["matcher_params_name"] == matcher_name
        assert summary["unit_match_status"] in {"computed", "reused"}
        assert summary["tracked_unit_status"] in {"computed", "reused"}
        assert summary["n_pairs"] == 1
        assert summary["n_tracked_units"] >= 1
        assert set(summary["stage_seconds"]) == {"unit_match", "tracked_unit"}
        # describe_run fills the receipt's per-stage status from
        # ``f"{stage}_status"``, so the summary's status-key stem must match the
        # stage_seconds key. A regression that renamed one but not the other
        # would render a blank status here.
        from spyglass.spikesorting.v2._orchestration.reporting import (
            describe_run,
        )

        receipt = describe_run(summary)
        stage_status = (
            receipt[receipt["row_type"] == "stage"]
            .set_index("stage")["status"]
            .to_dict()
        )
        assert stage_status["unit_match"] in {"computed", "reused"}
        assert stage_status["tracked_unit"] in {"computed", "reused"}
        # The matched pair forms exactly one 2-session tracked unit.
        matched = [
            row
            for row in (TrackedUnit & selection_pk).fetch(as_dict=True)
            if row["n_sessions_detected"] == 2
        ]
        assert len(matched) == 1

        # The same-probe chronic fixture diverges in no electrode space, so the
        # receipt's warnings are empty -- the field is populated, not a dead [].
        assert summary["warnings"] == []

        # Surfacing: force a divergence and confirm it reaches the receipt's
        # warnings on the next run (it is also logged in the selection path).
        from spyglass.spikesorting.v2.unit_matching import UnitMatchSelection

        monkeypatch.setattr(
            UnitMatchSelection,
            "_divergent_electrode_space_members",
            classmethod(lambda cls, choices: [1]),
        )

        # Idempotent: a rerun reuses both stages.
        rerun = run_v2_unit_match(
            session_group_owner=grp["owner"],
            session_group_name=grp["group_name"],
            matcher_params_name=matcher_name,
            curation_choices=grp["choices"],
        )
        assert rerun["unit_match_id"] == summary["unit_match_id"]
        assert rerun["unit_match_status"] == "reused"
        assert rerun["tracked_unit_status"] == "reused"
        assert rerun["n_pairs"] == 1
        assert any("electrode space" in w for w in rerun["warnings"]), rerun[
            "warnings"
        ]
    finally:
        if selection_pk is not None:
            (UnitMatch & selection_pk).super_delete(warn=False)
            (UnitMatchSelection & selection_pk).super_delete(warn=False)
        (
            MatcherParameters & {"matcher_params_name": matcher_name}
        ).super_delete(warn=False)
        mp._MATCHER_REGISTRY.clear()
        mp._MATCHER_REGISTRY.update(saved_matchers)
        mp._SCHEMA_REGISTRY.clear()
        mp._SCHEMA_REGISTRY.update(saved_schemas)
        mp._PREPARER_REGISTRY.clear()
        mp._PREPARER_REGISTRY.update(saved_preparers)


@pytest.mark.slow
def test_run_v2_unit_match_warning_survives_stage_failure(
    two_session_curated_group, monkeypatch
):
    """A UnitMatch-stage failure's partial summary still carries the warning.

    The divergent-electrode advisory is known from the selection, before either
    populate stage runs. ``_run_stage`` snapshots the run summary into the raised
    ``PipelineStageError``, so a stage failure must not drop the warning --
    mirrors the single-session curation-stage invariant where ``warnings`` is
    known pre-stage (see test_pipeline_observability).
    """
    from spyglass.spikesorting.v2.exceptions import PipelineStageError
    from spyglass.spikesorting.v2.pipeline import run_v2_unit_match
    from spyglass.spikesorting.v2.unit_matching import (
        UnitMatch,
        UnitMatchSelection,
    )

    grp = two_session_curated_group
    sentinel = RuntimeError("injected unit_match failure")

    def _boom(*args, **kwargs):
        raise sentinel

    # Force the advisory warning, then break the first populate stage so the
    # summary is only ever observed through the failure snapshot.
    monkeypatch.setattr(
        UnitMatchSelection,
        "_divergent_electrode_space_members",
        classmethod(lambda cls, choices: [1]),
    )
    monkeypatch.setattr(UnitMatch, "populate", _boom)

    try:
        with pytest.raises(PipelineStageError) as exc:
            run_v2_unit_match(
                session_group_owner=grp["owner"],
                session_group_name=grp["group_name"],
                matcher_params_name="unitmatch_default",
                curation_choices=grp["choices"],
            )
        err = exc.value
        assert err.stage == "unit_match"
        assert err.__cause__ is sentinel
        assert "warnings" in err.partial_run_summary
        assert any(
            "electrode space" in w for w in err.partial_run_summary["warnings"]
        ), err.partial_run_summary["warnings"]
    finally:
        (
            UnitMatchSelection
            & {
                "session_group_owner": grp["owner"],
                "session_group_name": grp["group_name"],
            }
        ).super_delete(warn=False)


@pytest.mark.slow
def test_describe_unit_match_choices_excludes_other_team(
    two_session_curated_group, monkeypatch
):
    """A curation under a different team (same nwb/sort-group/interval) is not
    offered as a choice.

    ``describe_unit_match_choices`` filters on the FULL member identity including
    ``team_name``, matching ``UnitMatchSelection``'s ownership validator. A sort
    of the same session/sort-group/interval under a different team tag must NOT
    appear as a pickable curation -- otherwise the helper would offer a choice
    that ``run_v2_unit_match`` then rejects. The member's OWN curation must still
    be offered.
    """
    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.pipeline import describe_unit_match_choices
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
    )
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection

    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    grp = two_session_curated_group
    member = grp["members"][0]
    own_choice = grp["choices"][0]
    other_team = "xteam_describe_filter"

    LabTeam.insert1(
        {
            "team_name": other_team,
            "team_description": "describe filter regression",
        },
        skip_duplicates=True,
    )
    # Same session/sort-group/interval/preproc as member 0, different team tag --
    # a distinct recording_id (team is part of the content-addressed identity).
    other_rec = RecordingSelection.insert_selection(
        {
            **member,
            "preprocessing_params_name": "default",
            "team_name": other_team,
        }
    )
    other_sort = None
    try:
        if not (Recording & other_rec):
            Recording.populate(other_rec, reserve_jobs=False)
        plant_sorter(monkeypatch, _plant_single_unit)
        other_sort = SortingSelection.insert_selection(
            {
                "recording_id": other_rec["recording_id"],
                "sorter": "mountainsort5",
                "sorter_params_name": _ensure_minirec_ms5_params(),
            }
        )
        if not (Sorting & other_sort):
            Sorting.populate(other_sort, reserve_jobs=False)
        clear_curations_for(other_sort)
        other_curation = CurationV2.insert_curation(
            sorting_key={"sorting_id": other_sort["sorting_id"]}
        )

        described = describe_unit_match_choices(grp["owner"], grp["group_name"])
        member0 = described[described["member_index"] == 0]
        offered = set(zip(member0["sorting_id"], member0["curation_id"]))
        # The member's own (member-team) curation IS offered ...
        assert (own_choice["sorting_id"], own_choice["curation_id"]) in offered
        # ... but the same-session curation under another team is NOT.
        assert (
            other_curation["sorting_id"],
            other_curation["curation_id"],
        ) not in offered
    finally:
        if other_sort is not None:
            for mid in (SpikeSortingOutput.CurationV2 & other_sort).fetch(
                "merge_id"
            ):
                (SpikeSortingOutput & {"merge_id": mid}).super_delete(
                    warn=False
                )
            (CurationV2 & other_sort).super_delete(warn=False)
            (Sorting & other_sort).super_delete(warn=False)
            (SortingSelection & other_sort).super_delete(warn=False)
        (Recording & other_rec).super_delete(warn=False)
        (RecordingSelection & other_rec).super_delete(warn=False)
        (LabTeam & {"team_name": other_team}).super_delete(warn=False)


@pytest.mark.slow
def test_describe_unit_match_choices_unsorted_member_and_multiple_recordings(
    two_session_curated_group, chronic_2_session_minirec, monkeypatch
):
    """Discovery offers every curation across a member's recordings and an
    empty choice list for a member with no recording or sort yet.

    A member sorted under two preprocessing recipes (two ``RecordingSelection``
    rows on the same full member identity) has BOTH sorts' curations offered,
    ordered by ``(sorting_id, curation_id)``; a member with no recording at all
    still appears, with no choices (``describe_unit_match_choices`` shows it as
    one null-curation row so it reads "sort me first").
    """
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2._orchestration.matching import (
        _unit_match_member_choices,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.pipeline import describe_unit_match_choices
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
    )
    from spyglass.spikesorting.v2.session_group import SessionGroup
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection

    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    grp = two_session_curated_group
    sub = chronic_2_session_minirec
    owner = grp["owner"]
    group_name = "unitmatch_discovery_shapes"
    sorted_member = grp["members"][0]
    own_choice = grp["choices"][0]
    unsorted_member = sub["next_day_member"]

    (
        SessionGroup
        & {"session_group_owner": owner, "session_group_name": group_name}
    ).super_delete(warn=False)
    SessionGroup.create_group(
        owner,
        group_name,
        [sorted_member, unsorted_member],
        allow_multi_day=True,
    )
    # A second recording of the SAME member identity (same team) under another
    # preprocessing recipe: a distinct recording_id, sorted and root-curated.
    second_rec = RecordingSelection.insert_selection(
        {
            **sorted_member,
            "preprocessing_params_name": "no_filter",
            "team_name": owner,
        }
    )
    second_sort = None
    try:
        if not (Recording & second_rec):
            Recording.populate(second_rec, reserve_jobs=False)
        plant_sorter(monkeypatch, _plant_single_unit)
        second_sort = SortingSelection.insert_selection(
            {
                "recording_id": second_rec["recording_id"],
                "sorter": "mountainsort5",
                "sorter_params_name": _ensure_minirec_ms5_params(),
            }
        )
        if not (Sorting & second_sort):
            Sorting.populate(second_sort, reserve_jobs=False)
        clear_curations_for(second_sort)
        second_curation = CurationV2.insert_curation(
            sorting_key={"sorting_id": second_sort["sorting_id"]}
        )

        members = _unit_match_member_choices(owner, group_name)
        assert [m["member_index"] for m in members] == [0, 1]
        offered = [
            (c["sorting_id"], c["curation_id"]) for c in members[0]["choices"]
        ]
        assert (own_choice["sorting_id"], own_choice["curation_id"]) in offered
        assert (
            second_curation["sorting_id"],
            second_curation["curation_id"],
        ) in offered
        assert offered == sorted(
            offered
        ), "choices ordered by sorting_id, curation_id"
        assert members[1]["choices"] == []
        assert members[1]["nwb_file_name"] == unsorted_member["nwb_file_name"]

        described = describe_unit_match_choices(owner, group_name)
        unsorted_rows = described[described["member_index"] == 1]
        assert len(unsorted_rows) == 1
        assert unsorted_rows["curation_id"].isna().all()
        assert len(described[described["member_index"] == 0]) == len(offered)
    finally:
        (
            SessionGroup
            & {"session_group_owner": owner, "session_group_name": group_name}
        ).super_delete(warn=False)
        if second_sort is not None:
            for mid in (SpikeSortingOutput.CurationV2 & second_sort).fetch(
                "merge_id"
            ):
                (SpikeSortingOutput & {"merge_id": mid}).super_delete(
                    warn=False
                )
            (CurationV2 & second_sort).super_delete(warn=False)
            (Sorting & second_sort).super_delete(warn=False)
            (SortingSelection & second_sort).super_delete(warn=False)
        (Recording & second_rec).super_delete(warn=False)
        (RecordingSelection & second_rec).super_delete(warn=False)

"""Cross-session unit tracking: schema and matcher validation.

Graph logic runs as pure functions; table wiring uses ``dj_conn`` and a
two-session synthetic sort. The ground-truth AUC gate needs the unhosted
polymer two-session fixture and UnitMatchPy, so it skips in CI.
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
from tests.spikesorting.v2._ingest_helpers import matched_unit_ids


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
    """unit_semantics is derived from the sorter: the clusterless thresholder
    emits threshold crossings; every real sorter emits sorted units."""
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

    # Input 1 first; canonicalization flips it.
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
    """Three sessions with all pairwise edges high form one clique."""
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
    """A<->B and B<->C high but A<->C low splits A and C apart."""
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

    Edges A1-B1, A1-B2, A2-B1 form three overlapping size-2 cliques; each
    unit lands in exactly one tracked unit.
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
    # Every node is covered exactly once.
    assert set(all_members) == {a1, a2, b1, b2}
    # The strongest edge's clique survives whole.
    assert any(set(tu["members"]) == {a1, b1} for tu in tracked)


def test_derive_tracked_units_partition_prefers_stronger_clique():
    """Among equal-size overlapping cliques, the strongest edge wins the unit.

    A1-B1 (0.80) and A1-B2 (0.95) overlap on A1: {A1,B2} stays whole and B1
    becomes a singleton, though {A1,B1} sorts first.
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
    edges = [(a, b, 0.2)]  # below threshold
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
    add_edge would silently create the node)."""
    from spyglass.spikesorting.v2._matching.graph import derive_tracked_units

    a, b = _nodes(("A", 1), ("B", 1))
    stray = ("C", 0, 1)  # not in node_universe
    with pytest.raises(ValueError, match="absent from node_universe"):
        derive_tracked_units(
            [a, b], [(a, stray, 0.9)], threshold=0.5, max_strict_nodes=100
        )


def test_derive_tracked_units_equal_strength_tie_break_is_member_sorted():
    """Equal-size AND equal-strength overlapping cliques resolve by sorted members.

    {A1,B1} stays whole and B2 becomes a singleton, keeping
    ``tracked_unit_id`` assignment deterministic.
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
    # Every node covered exactly once.
    all_members = [node for tu in tracked for node in tu["members"]]
    assert sorted(all_members) == sorted([a1, b1, b2])
    assert len(all_members) == len(set(all_members))


def test_derive_tracked_units_counts_sessions_by_nwb_not_curation():
    """``n_sessions_detected`` counts recording sessions (nwb), so two
    sortings from the same nwb count as one session."""
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
    """A member with identical channel geometry but different electrode
    identity is reported as divergent (advisory only: electrode names and ids
    are not stable across labs' ingestion)."""
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
    """Members with identical signatures, or a single member, have no
    divergent members."""
    from spyglass.spikesorting.v2._matching.graph import (
        divergent_electrode_space_members,
    )

    sig = (("probeA", 0, "ca1"), ("probeA", 1, "ca1"))
    assert divergent_electrode_space_members({0: sig, 1: sig}) == []
    assert divergent_electrode_space_members({0: sig}) == []


# --------------------------------------------------------------------------- #
# NWB pairs (de)serialization round-trip (pure I/O, no DB).                     #
# --------------------------------------------------------------------------- #


def test_pairs_nwb_round_trip_preserves_missing_drift_and_fdr(tmp_path):
    """Non-empty and empty pairs round-trip; a None fdr reads back as None, a
    real fdr survives, and string ids stay str."""
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
            "drift_estimate_um": None,
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
    assert back[0]["drift_estimate_um"] is None
    assert back[0]["fdr_estimate"] is None
    assert back[1]["drift_estimate_um"] == pytest.approx(-3.125)
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
    """An external matcher using only public ``matcher_protocol`` types.

    Matches the first unit of every session pair with a fixed probability.
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
    """The ``per_unit`` bundle construction recovers drift-out units without
    degrading healthy units, on seeds 10..19.

    Scenarios, scoring and gates come from
    ``scripts/unitmatch_half_split_experiment.py``. Gates for driftout_A and
    driftout_AB:

    - drift-out recall >= 0.80;
    - healthy units' half templates bit-identical to the control run;
    - healthy true-pair probability drop <= 0.04;
    - healthy false-pair rate increase <= 0.005.

    The S x S false-pair rate is a printed diagnostic, not a gate (see
    ``evaluate_gates``). Do not loosen thresholds, seeds or scoring, or
    skip, to make this pass. DB-free.
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
    # Allowlist, so a silently dropped scenario fails instead of passing.
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
# Table wiring (database), over two planted sorts on the chronic minirec.     #
# --------------------------------------------------------------------------- #

_GROUP_NAME = "unitmatch_two_session"
_SOLO_NAME = "unitmatch_solo"


def _ensure_minirec_clusterless_params():
    """Insert the smoke clusterless sorter row for the unit-semantics test."""
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


#: Params row for the planted minirec sorts (never run). A real sorter name
#: gives ``sorted_units`` semantics, which matching requires.
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

    Enough spikes for split-half bundle extraction and the display template.
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

    Creates a two-member SessionGroup and a one-member solo group, each member
    root-curated with a planted single-unit mountainsort5 sort. Yields the
    group identities and per-member curation choices; tears everything down
    afterward.
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

    # Drop leftovers from an interrupted run.
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
    """The bulk insert() path also rejects a typo'd matcher."""
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
    """Duplicate-content detection is per matcher: identical params under
    another matcher are allowed, under the same matcher rejected."""
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
        grp["owner"],
        grp["group_name"],
        "unitmatch_default",
        grp["choices"],
        unit_policy="v2_unflagged_single_units",
    )
    pk2 = UnitMatchSelection.insert_selection(
        grp["owner"],
        grp["group_name"],
        "unitmatch_default",
        dict(grp["choices"]),
        unit_policy="v2_unflagged_single_units",
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
        grp["owner"],
        grp["group_name"],
        "unitmatch_default",
        changed,
        unit_policy="v2_unflagged_single_units",
    )
    assert pk3["unitmatch_id"] != pk1["unitmatch_id"]


@pytest.mark.slow
def test_unitmatch_selection_accepts_curation_evaluation_committed_children(
    two_session_curated_group, curation_evaluation_defaults
):
    """Final ``CurationEvaluation`` children are valid UnitMatch inputs.

    Children accepted with ``use_evaluation_labels`` can be pinned in
    ``UnitMatchSelection``, and ``make_fetch`` sees their ids and frozen
    matchable unit sets.
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
            unit_policy="v2_unflagged_single_units",
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
            expected_matchable = [int(u) for u in matched_unit_ids(choice)]
            assert plan["matchable_unit_ids"] == expected_matchable
    finally:
        from spyglass.spikesorting.spikesorting_merge import (
            SpikeSortingOutput,
        )

        if pk is not None:
            (UnitMatchSelection & pk).super_delete(warn=False)
        # Leftover children would leak into later auto_curated plans. Drop
        # the merge master first; DataJoint will not delete a part before it.
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

    Unit ``i`` fires at frames ``samples_by_unit[i]``. A distinct params name
    keeps it from colliding with the fixture's sort. Returns
    ``(sort_key, sorted_unit_ids)``; the caller owns teardown.
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

    A merge needs >= 2 units. Returns
    ``(two_unit_sort_key, sorted_unit_ids, sorter_params_name)``; the caller
    owns teardown.
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

    Its proposed merges are unapplied, so units would be oversplit. Both
    ``UnitMatchSelection.insert_selection`` and ``UnitMatch.make_fetch``
    reject it.
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
                unit_policy="v2_unflagged_single_units",
            )

        # Site 2: make_fetch rejects a direct-insert selection pointing at the
        # preview, with a matching hash so only the preview guard fires.
        pk = UnitMatchSelection.insert_selection(
            grp["owner"],
            grp["group_name"],
            "unitmatch_default",
            {0: root_choice, 1: grp["choices"][1]},
            unit_policy="v2_unflagged_single_units",
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

    make_fetch's member plan carries the child's merged unit, not the two
    pre-merge ids.
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
        merged0 = CurationV2.insert_curation(
            {"sorting_id": two_unit_sort["sorting_id"]},
            merge_groups=[[unit_ids[0], unit_ids[1]]],
            apply_merge=True,
        )
        assert CurationV2.is_committed_curation(merged0)
        merged_choice = {
            "sorting_id": merged0["sorting_id"],
            "curation_id": merged0["curation_id"],
        }
        expected = [int(u) for u in matched_unit_ids(merged_choice)]
        assert len(expected) == 1  # two pre-merge units -> one merged unit

        pk = UnitMatchSelection.insert_selection(
            grp["owner"],
            grp["group_name"],
            "unitmatch_default",
            {0: merged_choice, 1: grp["choices"][1]},
            unit_policy="v2_unflagged_single_units",
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
    """insert_selection rejects an existing master whose Input parts do not
    realize its input_set_hash, rather than returning its PK."""
    from spyglass.spikesorting.v2.exceptions import SchemaBypassError
    from spyglass.spikesorting.v2.unit_matching import UnitMatchSelection

    grp = two_session_curated_group
    pk = UnitMatchSelection.insert_selection(
        grp["owner"],
        grp["group_name"],
        "unitmatch_default",
        grp["choices"],
        unit_policy="v2_unflagged_single_units",
    )
    # Re-insert only the master row (same id and hash, no parts).
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
                unit_policy="v2_unflagged_single_units",
            )
    finally:
        (UnitMatchSelection & pk).super_delete(warn=False)


@pytest.mark.slow
def test_insert_selection_rejects_orphan_recording_parts_with_matching_hash(
    two_session_curated_group,
):
    """insert_selection rejects a master whose parts match its hash but carry
    an InputRecording row for a missing input (the hash ignores such rows)."""
    import uuid

    from spyglass.spikesorting.v2.exceptions import SchemaBypassError
    from spyglass.spikesorting.v2.unit_matching import UnitMatchSelection
    from tests.spikesorting.v2._unitmatch_helpers import reforge_selection

    grp = two_session_curated_group
    pk = UnitMatchSelection.insert_selection(
        grp["owner"],
        grp["group_name"],
        "unitmatch_default",
        grp["choices"],
        unit_policy="v2_unflagged_single_units",
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
                unit_policy="v2_unflagged_single_units",
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
            grp["owner"],
            grp["group_name"],
            "unitmatch_default",
            swapped,
            unit_policy="v2_unflagged_single_units",
        )
    # No master row was created.
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
            unit_policy="v2_unflagged_single_units",
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
        grp["owner"],
        grp["group_name"],
        "unitmatch_default",
        grp["choices"],
        unit_policy="v2_unflagged_single_units",
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
    # Swap the two inputs' frozen recordings: each curation claims the
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
    """make_fetch rejects a raw-insert master whose stored input_set_hash
    disagrees with its otherwise valid parts."""
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
        grp["owner"],
        grp["group_name"],
        "unitmatch_default",
        grp["choices"],
        unit_policy="v2_unflagged_single_units",
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
        grp["owner"],
        grp["group_name"],
        "unitmatch_default",
        grp["choices"],
        unit_policy="v2_unflagged_single_units",
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
        unit_policy="v2_unflagged_single_units",
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
        grp["owner"],
        grp["solo_name"],
        "unitmatch_default",
        {0: choice},
        unit_policy="v2_unflagged_single_units",
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
        UnitMatchSelection.insert_inputs(
            [], "unitmatch_default", unit_policy="v2_unflagged_single_units"
        )
    assert len(UnitMatchSelection()) == before


@pytest.mark.usefixtures("dj_conn")
def test_unitmatch_computed_matches_make_insert_signature():
    """``UnitMatchComputed`` field order matches ``make_insert``, which
    receives it positionally."""
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
    """``UnitMatchFetched`` field order matches ``make_compute``, which
    receives it positionally."""
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
    """A populated UnitMatch row records the SI version, backend module path,
    and backend package version, even on the single-session path."""
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
        unit_policy="v2_unflagged_single_units",
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


def test_matcher_provenance_is_required_for_new_runs(two_session_curated_group):
    """A newly populated run has complete metadata and a non-null DB column."""
    from spyglass.spikesorting.v2._matching.provenance import (
        validate_matcher_provenance,
    )
    from spyglass.spikesorting.v2.unit_matching import (
        UnitMatch,
        UnitMatchSelection,
    )

    group = two_session_curated_group
    key = UnitMatchSelection.insert_selection(
        group["owner"],
        group["solo_name"],
        "unitmatch_default",
        {0: group["choices"][0]},
        unit_policy="v2_unflagged_single_units",
    )
    UnitMatch.populate(key, reserve_jobs=False)
    row = (UnitMatch & key).fetch1()
    validate_matcher_provenance(row["matcher_provenance"])
    assert (
        row["matcher_provenance"]["runtime_environment"]["job_kwargs"]["n_jobs"]
        == 1
    )
    assert not UnitMatch.heading.attributes["matcher_provenance"].nullable


@pytest.mark.slow
def test_clusterless_unit_semantics_derived_and_warned(
    chronic_2_session_minirec, caplog
):
    """A clusterless sort's unit semantics are threshold crossings, and
    UnitMatchSelection.insert_selection warns when matching them."""
    import logging

    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.session_group import SessionGroup
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection
    from spyglass.spikesorting.v2._matching.inputs import (
        _warn_clusterless_match_once,
    )
    from spyglass.spikesorting.v2.unit_matching import (
        MatcherParameters,
        UnitMatchSelection,
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
                unit_policy="v2_unflagged_single_units",
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
    """``Pair.insert`` rejects a raw pair whose endpoint is not a pinned
    ``UnitMatchSelection.Input``, before the foreign key check."""
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
        unit_policy="v2_unflagged_single_units",
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
    """With no matches, ``TrackedUnit.make`` emits one strict singleton per
    matchable unit."""
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
        unit_policy="v2_unflagged_single_units",
    )
    UnitMatch.populate(pk, reserve_jobs=False)
    TrackedUnit.populate(pk, reserve_jobs=False)

    n_matchable = len(matched_unit_ids(grp["choices"][0]))
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
    """A registered lightweight matcher drives a two-member
    ``UnitMatch.populate()`` that writes ``Pair`` rows and the NWB pairs
    table, and ``TrackedUnit`` groups the matched pair. Needs no UnitMatchPy.
    """
    from pydantic import BaseModel, ConfigDict, Field

    from spyglass.spikesorting.v2 import matcher_protocol as mp
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
    matchable_a = matched_unit_ids(grp["choices"][0])
    matchable_b = matched_unit_ids(grp["choices"][1])
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

    # The fixture matcher reads pairs from params, not bundles.
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
            unit_policy="v2_unflagged_single_units",
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

        # make_compute runs on make_fetch's carrier alone.
        fetched = UnitMatch().make_fetch(selection_pk)
        computed = UnitMatch().make_compute(selection_pk, **fetched._asdict())
        # Unlink make_compute's staged NWB before asserting.
        from spyglass.spikesorting.v2._storage.staged_outputs import (
            unlink_staged_analysis_file as _unlink_staged_analysis_file,
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
    """``make_compute`` builds each bundle from the files ``make_fetch``
    resolved, with no DB access beyond staging the pairs NWB; the inputs
    equal ``CurationV2.get_recording`` / ``get_sorting``."""
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.matcher_protocol import get_input_preparer
    from spyglass.spikesorting.v2._storage.staged_outputs import (
        unlink_staged_analysis_file as _unlink_staged_analysis_file,
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
            unit_policy="v2_unflagged_single_units",
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
                matched_unit_ids(choice)
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
    """UnitMatch and TrackedUnit populate end to end from accepted evaluation
    children."""
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
            member_index: [int(u) for u in matched_unit_ids(choice)]
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
            unit_policy="v2_unflagged_single_units",
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
            # Drop the merge master first; DataJoint will not delete a part
            # before it.
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

    The merged unit gets a pair row and ``TrackedUnit`` groups it with
    another member's unit.
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
        merged0 = CurationV2.insert_curation(
            sorting_key,
            merge_groups=[[unit_ids[0], unit_ids[1]]],
            apply_merge=True,
        )
        assert CurationV2.is_committed_curation(merged0)
        merged_choice = {
            "sorting_id": merged0["sorting_id"],
            "curation_id": merged0["curation_id"],
        }
        merged_matchable = [int(u) for u in matched_unit_ids(merged_choice)]
        assert len(merged_matchable) == 1
        merged_uid = merged_matchable[0]

        member1_choice = grp["choices"][1]
        member1_matchable = [int(u) for u in matched_unit_ids(member1_choice)]
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
            unit_policy="v2_unflagged_single_units",
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
        # Unknown drift is stored as NULL, not 0 um.
        assert pair["drift_estimate_um"] is None

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

    Plants ``samples_by_unit`` as member 0's sort and registers a pairer that
    emits ``pairs`` only between units present in the bundles. Yields
    ``selection_pk``, member ``choices``, and member 0's
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
            grp["owner"],
            grp["group_name"],
            matcher_params_name,
            choices,
            unit_policy="v2_unflagged_single_units",
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

    For this group, input_index equals member_index.
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

    Member 0's one-spike unit is left out of its bundle; ``UnitMatch.populate``
    warns once, naming member 0 and the unit, and never for member 1.
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

    The one-spike unit cannot pair, but is still a ``MatchableUnit`` row and
    becomes a singleton ``TrackedUnit``.
    """
    pytest.importorskip("UnitMatchPy")
    from spyglass.spikesorting.v2.unit_matching import (
        TrackedUnit,
        UnitMatch,
    )

    grp = two_session_curated_group
    (unit_b,) = [int(u) for u in matched_unit_ids(grp["choices"][1])]
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
    from spyglass.spikesorting.v2._matching.waveforms import (
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
# Ground-truth AUC gate: AUC > 0.85 on the polymer probe. Skips without the    #
# two-session polymer fixtures (unhosted; _fetch.py URLs are None) or         #
# UnitMatchPy.                                                                 #
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
    """UnitMatch probability vs ground-truth correspondence on the
    two-session polymer probe has AUC > 0.85.

    Both sessions share a MEArec ``template_seed``, so unit ``i`` is the same
    neuron in each. The v2 sort and curation run on both sessions, but the
    matcher is scored on the ground-truth sortings because MEArec amplitudes
    sit below the production sorter's threshold.
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

        # 2. Score the matcher on ground-truth bundles; threshold 0 gives every
        #    cross-session pair a probability.
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
    """A raw ``UnitMatch.Pair.insert`` raises ``UnitMatchPairIntegrityError``
    for an unpinned-curation endpoint, a same-curation edge, a reversed or
    duplicate edge, or an out-of-range ``match_probability``.

    Validation reads the pinned ``UnitMatchSelection.Input`` rows, so no
    populated ``UnitMatch`` is needed.
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
        grp["owner"],
        grp["group_name"],
        "unitmatch_default",
        grp["choices"],
        unit_policy="v2_unflagged_single_units",
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

        # A fake curation not among the inputs (checked before the FK).
        unpinned = (str(uuid.uuid4()), 0, 0)
        with pytest.raises(UnitMatchPairIntegrityError, match="not a pinned"):
            UnitMatch.Pair.insert1(pair_row(unpinned, node1))

        # Same-curation edge (both endpoints pin input 0).
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
    """``TrackedUnit`` reads ``UnitMatch``'s frozen ``MatchableUnit`` snapshot,
    so relabeling a unit ``noise`` after populate does not drop it."""
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.unit_matching import (
        TrackedUnit,
        UnitMatch,
        UnitMatchSelection,
    )

    grp = two_session_curated_group
    choice = grp["choices"][0]
    pk = UnitMatchSelection.insert_selection(
        grp["owner"],
        grp["solo_name"],
        "unitmatch_default",
        {0: choice},
        unit_policy="v2_unflagged_single_units",
    )
    # The solo group has 0 pairs but still freezes the snapshot.
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
            int(u) for u in matched_unit_ids(choice)
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
    """A channel-geometry mismatch is rejected at
    ``UnitMatchSelection.insert_selection``, before bundle extraction."""
    import numpy as np

    from spyglass.spikesorting.v2._matching import (
        unitmatch_backend as _unitmatch_backend,
    )
    from spyglass.spikesorting.v2.unit_matching import UnitMatchSelection

    grp = two_session_curated_group
    sid0 = str(grp["choices"][0]["sorting_id"])

    def fake_positions(curation_key, **_):
        # member 0: 4-channel probe; the other member: 8-channel -> mismatch.
        n = 4 if str(curation_key["sorting_id"]) == sid0 else 8
        return np.zeros((n, 2))

    monkeypatch.setattr(
        UnitMatchSelection,
        "_member_channel_positions",
        staticmethod(fake_positions),
    )
    # Force the new-insert path so the preflight runs even if the row exists.
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
            grp["owner"],
            grp["group_name"],
            "unitmatch_default",
            grp["choices"],
            unit_policy="v2_unflagged_single_units",
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

    def read_positions(key, **_):
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

    The file embeds the run, group and matcher header (the same provenance as
    the ``UnitMatch`` row) and the per-input and per-recording maps. Uses the
    solo path, which writes them too.
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
        grp["owner"],
        grp["solo_name"],
        "unitmatch_default",
        {0: choice},
        unit_policy="v2_unflagged_single_units",
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
    """``TrackedUnit.get_unit_brain_regions`` includes ``unitmatch_id``,
    ``tracked_unit_id``, ``input_index``, ``nwb_file_name``,
    ``recording_date`` and ``curation_id``."""
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
        unit_policy="v2_unflagged_single_units",
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
    """``run_v2_unit_match`` runs selection, UnitMatch and TrackedUnit in one
    call, idempotently.

    Uses a one-pair fixture matcher, so UnitMatchPy is not needed.
    """
    from pydantic import BaseModel, ConfigDict, Field

    from spyglass.spikesorting.v2 import matcher_protocol as mp
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

    matchable_a = matched_unit_ids(grp["choices"][0])
    matchable_b = matched_unit_ids(grp["choices"][1])
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

        # Plan, then run the plan (each member has exactly one root).
        plan = plan_v2_unit_match(
            grp["owner"],
            grp["group_name"],
            curation_strategy="root",
            matcher_params_name=matcher_name,
            unit_policy="v2_unflagged_single_units",
        )
        assert plan.ok, plan.errors
        assert set(plan.curation_choices) == set(grp["choices"])
        assert not plan.as_dataframe().empty

        # The fixture roots are 'manual', so auto_curated resolves nothing.
        described = describe_unit_match_choices(grp["owner"], grp["group_name"])
        # Every offered row carries curation_source, which auto_curated reads.
        offered = described[described["sorting_id"].notna()]
        assert offered["curation_source"].notna().all()
        auto_plan = plan_v2_unit_match(
            grp["owner"],
            grp["group_name"],
            curation_strategy="auto_curated",
            matcher_params_name=matcher_name,
            unit_policy="v2_unflagged_single_units",
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
        # describe_run reads f"{stage}_status", so status and stage_seconds
        # keys must share a stem.
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

        # The same-probe inputs agree in electrode space, while the explicit
        # auto-only policy reports that both root populations are unlabeled.
        assert len(summary["warnings"]) == 2
        assert all(
            "included units are unlabeled under policy 'v2_unflagged_single_units'"
            in warning
            for warning in summary["warnings"]
        )

        # A forced divergence reaches the receipt's warnings on the next run.
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
            unit_policy="v2_unflagged_single_units",
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

    The divergent-electrode advisory is known before either populate stage,
    so the ``PipelineStageError`` snapshot keeps it.
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
                unit_policy="v2_unflagged_single_units",
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
    offered as a choice; the member's own curation is.

    The filter matches ``UnitMatchSelection``'s ownership validator, which
    includes ``team_name``.
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
    # Same as member 0 but another team, so a distinct recording_id.
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

    A member sorted under two preprocessing recipes has both sorts'
    curations, ordered by ``(sorting_id, curation_id)``; a member with no
    recording appears as one null-curation row.
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

"""Known-answer benchmark for matching independently sorted daily concatenations.

The dataset, scoring and gate functions live in
``tests/spikesorting/v2/scripts/unitmatch_daily_concat_benchmark.py``. The
tests here check that the simulated days are what the benchmark claims (a
known-answer check of the dataset itself) and that the scoring counts a
hand-built run exactly. Both need SpikeInterface but neither a database nor
UnitMatchPy.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from tests.spikesorting.v2.scripts import (
    unitmatch_daily_concat_benchmark as bench,
)


@pytest.fixture(scope="module", params=bench.SCENARIOS)
def dataset(request):
    """A development-seed dataset of each scenario."""
    return bench.make_dataset(request.param, 0)


def test_days_join_two_members_with_an_exclusion(dataset):
    """Each day is one segment of two unequal members, silenced once."""
    specs = bench.SCENARIO_DAYS[dataset.scenario]
    assert len(dataset.days) == len(specs)
    for day, spec in zip(dataset.days, specs):
        n_member = [int(round(s * bench.FS)) for s in spec.member_durations_s]
        assert n_member[0] != n_member[1]
        assert day.recording.get_num_segments() == 1
        assert day.recording.get_num_samples() == sum(n_member)
        assert day.member_spans == [
            (0, n_member[0]),
            (n_member[0], sum(n_member)),
        ]
        member, start_s, duration_s = spec.exclusion
        start = day.member_spans[member][0] + int(round(start_s * bench.FS))
        stop = start + int(round(duration_s * bench.FS))
        assert day.exclusion == (start, stop)
        # The spans are the members, with the excluded range cut out.
        expected = []
        for index, (lo, hi) in enumerate(day.member_spans):
            if index == member:
                expected += [(lo, start), (stop, hi)]
            else:
                expected.append((lo, hi))
        assert day.statistics_spans == expected
        silenced = day.recording.get_traces(start_frame=start, end_frame=stop)
        assert np.all(silenced == 0)
        around = day.recording.get_traces(
            start_frame=start - 30, end_frame=start
        )
        assert np.all(np.any(around != 0, axis=1))


def test_units_per_day_and_ids_do_not_reveal_identity(dataset):
    """Unit counts per day are as documented and no id repeats a neuron."""
    counts = bench.SCENARIO_COUNTS[dataset.scenario]
    if dataset.scenario == "two_day":
        expected = [24, 24]
    else:
        expected = [25, 21, 25]
    got = [len(day.sorting.get_unit_ids()) for day in dataset.days]
    assert got == expected
    assert all(n >= 20 for n in got)
    for neuron in dataset.neurons:
        ids = list(neuron["unit_ids"].values())
        assert len(set(ids)) == len(ids)
    n_distractor = sum(n["cls"] == "distractor" for n in dataset.neurons)
    assert n_distractor == counts["distractor"] * len(dataset.days)
    for day in dataset.days:
        ids = np.asarray(day.sorting.get_unit_ids())
        assert np.all(np.diff(ids) > 0)
        # Sparse: not the contiguous range 0..n-1.
        assert not np.array_equal(ids, np.arange(ids.size))


def test_spike_trains_follow_the_planted_members(dataset):
    """Each unit fires only in its active members, and the counts agree."""
    for neuron in dataset.neurons:
        for d, uid in neuron["unit_ids"].items():
            day = dataset.days[d]
            train = day.sorting.get_unit_spike_train(uid)
            per_member = [
                int(np.count_nonzero((train >= lo) & (train < hi)))
                for lo, hi in day.member_spans
            ]
            assert per_member == day.planted_counts[uid]
            assert sum(per_member) == train.size
            for member, n_spikes in enumerate(per_member):
                active = member in neuron["active_members"][d]
                # 10 Hz over at least 20 s gives far more than 50 spikes.
                assert (n_spikes > 50) if active else (n_spikes == 0)
    if dataset.scenario == "two_day":
        partial = [n for n in dataset.neurons if n["cls"] == "partial"]
        patterns = {
            tuple(tuple(n["active_members"][d]) for d in (0, 1))
            for n in partial
        }
        assert patterns == {
            ((0,), (0,)),
            ((1,), (0,)),
            ((0,), (1,)),
            ((1,), (1,)),
        }


def test_neurons_are_distinct_and_changes_are_as_stated(dataset):
    """Distinct neurons keep the minimum distance; changes match the constants."""
    min_distance = bench.UNIT_LOCATION_KWARGS["minimum_distance"]
    drawn = [
        n
        for n in dataset.neurons
        if not (n["cls"] == "conflict" and n["neuron_id"] > n["partner"])
    ]
    first = [n["location_um"][n["days"][0]] for n in drawn]
    for a, b in combinations(range(len(first)), 2):
        assert np.linalg.norm(first[a] - first[b]) >= min_distance
    neurons = {n["neuron_id"]: n for n in dataset.neurons}
    for neuron in dataset.neurons:
        days = neuron["days"]
        base = neuron["location_um"][days[0]]
        if neuron["cls"] == "gradual":
            for d in days:
                shift = neuron["location_um"][d] - base
                assert np.allclose(
                    np.abs(shift), [0, bench.GRADUAL_SHIFT_UM[d], 0]
                )
                assert neuron["amplitude_scale"][d] == pytest.approx(
                    bench.GRADUAL_AMPLITUDE_SCALE[d]
                )
        elif (
            neuron["cls"] == "conflict"
            and neuron["neuron_id"] < neuron["partner"]
        ):
            partner = neurons[neuron["partner"]]
            assert partner["params"] == neuron["params"]
            gaps = [
                np.linalg.norm(
                    partner["location_um"][d] - neuron["location_um"][d]
                )
                for d in days
            ]
            assert np.allclose(
                gaps,
                [bench.CONFLICT_OFFSET_UM - s for s in bench.CONFLICT_SHIFT_UM],
            )
        else:
            for d in days:
                assert np.array_equal(neuron["location_um"][d], base)
                assert neuron["amplitude_scale"][d] == 1.0
        if neuron["cls"] == "reappear":
            assert days == [0, 2]


def test_dataset_is_deterministic():
    """The same scenario and seed rebuild identical days."""
    a = bench.make_dataset("two_day", 1)
    b = bench.make_dataset("two_day", 1)
    for day_a, day_b in zip(a.days, b.days):
        assert list(day_a.sorting.get_unit_ids()) == list(
            day_b.sorting.get_unit_ids()
        )
        for uid in day_a.sorting.get_unit_ids():
            assert np.array_equal(
                day_a.sorting.get_unit_spike_train(uid),
                day_b.sorting.get_unit_spike_train(uid),
            )
        assert np.array_equal(
            day_a.recording.get_traces(start_frame=0, end_frame=3000),
            day_b.recording.get_traces(start_frame=0, end_frame=3000),
        )


def _hand_dataset():
    """Three days: a stable, a reappearing, a conflict pair and a distractor."""
    neurons = [
        {"neuron_id": 0, "cls": "stable", "unit_ids": {0: 10, 1: 20, 2: 30}},
        {"neuron_id": 1, "cls": "reappear", "unit_ids": {0: 11, 2: 31}},
        {
            "neuron_id": 2,
            "cls": "conflict",
            "partner": 3,
            "unit_ids": {0: 12, 1: 22, 2: 32},
        },
        {
            "neuron_id": 3,
            "cls": "conflict",
            "partner": 2,
            "unit_ids": {0: 13, 1: 23, 2: 33},
        },
        {"neuron_id": 4, "cls": "distractor", "unit_ids": {1: 24}},
    ]
    planted = {
        "day1": {10: [5, 5], 11: [5, 5], 12: [5, 5], 13: [5, 5]},
        "day2": {20: [0, 7], 22: [5, 5], 23: [5, 5], 24: [5, 5]},
        "day3": {30: [5, 5], 31: [5, 5], 32: [5, 5], 33: [5, 5]},
    }
    days = [
        bench.Day(
            label=label,
            recording=None,
            sorting=None,
            member_spans=[],
            exclusion=(0, 0),
            statistics_spans=[],
            planted_counts=planted[label],
            session_names=[f"{label}_member0", f"{label}_member1"],
        )
        for label in ("day1", "day2", "day3")
    ]
    return bench.Dataset("three_day", 0, neurons, days)


def _pair(a, ua, b, ub, probability=0.9):
    return {
        "session_a_sorting_id": a,
        "unit_a_id": ua,
        "session_b_sorting_id": b,
        "unit_b_id": ub,
        "match_probability": probability,
    }


def _tracked(members, n_sessions):
    return {
        "members": [(label, 0, uid) for label, uid in members],
        "n_sessions_detected": n_sessions,
        "n_matching_inputs": len({label for label, _ in members}),
    }


def test_score_run_counts_a_hand_built_run():
    """Every count of a hand-built run equals the hand-derived value."""
    dataset = _hand_dataset()
    pairs = [
        _pair("day1", 10, "day2", 20),  # stable, adjacent
        _pair("day2", 20, "day3", 30),  # stable, adjacent
        _pair("day1", 11, "day3", 31),  # reappear, day 1 -- day 3
        _pair("day1", 12, "day2", 22),  # conflict mover, adjacent
        _pair("day2", 22, "day3", 33),  # false: mover with its partner
        _pair("day2", 24, "day3", 32),  # false: distractor
    ]
    tracked = [
        # sessions: day1 m0+m1, day2 m1 only (unit 20), day3 m0+m1 -> 5
        _tracked([("day1", 10), ("day2", 20), ("day3", 30)], 5),
        _tracked([("day1", 11), ("day3", 31)], 4),
        # true value 4; a wrong count must be caught
        _tracked([("day1", 12), ("day2", 22)], 3),
        _tracked([("day2", 24), ("day3", 32)], 4),
        _tracked([("day1", 13)], 2),
        _tracked([("day2", 23)], 2),
        _tracked([("day3", 33)], 2),
    ]
    recording_counts = {
        label: {uid: list(c) for uid, c in day.items()}
        for label, day in ((d.label, d.planted_counts) for d in dataset.days)
    }
    recording_counts["day3"][31] = [5, 4]  # one production count disagrees
    counts = bench.score_run(
        dataset,
        pairs,
        tracked,
        {"day1": [], "day2": [], "day3": []},
        recording_counts,
    )
    expected = {
        "pair_emitted:all": [4, 10],
        "pair_tracked:all": [5, 10],
        "pair_emitted:stable": [2, 3],
        "pair_tracked:stable": [3, 3],
        "pair_emitted:stable:adjacent": [2, 2],
        "pair_emitted:stable:skip": [0, 1],
        "pair_tracked:stable:skip": [1, 1],
        "pair_emitted:reappear": [1, 1],
        "pair_tracked:reappear": [1, 1],
        "pair_tracked:reappear:skip": [1, 1],
        "pair_emitted:conflict": [1, 6],
        "pair_tracked:conflict": [1, 6],
        "pair_precision": [4, 6],
        "false_pair:conflict_partner": [1, 6],
        "false_pair:distractor": [1, 6],
        "false_pair:other_neuron": [0, 6],
        "incorrect_identity": [1, 4],
        "same_input_group": [0, 7],
        "distractor_emitted": [1, 1],
        "distractor_grouped": [1, 1],
        "singleton": [3, 11],
        "identity_complete:all": [2, 4],
        "identity_complete:stable": [1, 1],
        "identity_complete:reappear": [1, 1],
        "identity_complete:conflict": [0, 2],
        "recording_count_mismatch": [1, 12],
        "sessions_detected_mismatch": [1, 7],
        "matching_inputs_mismatch": [0, 7],
    }
    for key, value in expected.items():
        assert counts[key] == value, key
    assert "partial_bundled" not in counts


def test_score_run_flags_a_tracked_unit_holding_one_input_twice():
    """A tracked unit with two units of one day is a structural violation."""
    dataset = _hand_dataset()
    tracked = [
        _tracked([("day1", 10), ("day1", 11)], 2),
        *(
            _tracked([(label, uid)], 2)
            for label, uid in (
                ("day1", 12),
                ("day1", 13),
                ("day2", 20),
                ("day2", 22),
                ("day2", 23),
                ("day2", 24),
                ("day3", 30),
                ("day3", 31),
                ("day3", 32),
                ("day3", 33),
            )
        ),
    ]
    recording_counts = {d.label: dict(d.planted_counts) for d in dataset.days}
    counts = bench.score_run(
        dataset,
        [],
        tracked,
        {"day1": [], "day2": [], "day3": []},
        recording_counts,
    )
    assert counts["same_input_group"] == [1, 11]
    assert counts["incorrect_identity"] == [1, 1]
    assert counts["pair_precision"] == [0, 0]


def _records(scenario, per_seed_counts):
    """Hand-built run records: one ``counts`` dict per seed."""
    return [
        {"scenario": scenario, "seed": seed, "counts": counts}
        for seed, counts in enumerate(per_seed_counts)
    ]


def test_derive_gate_applies_the_margin_rule():
    """The floor, the rounding direction and the coin-flip cut-off hold."""
    flat = _records("two_day", [{"m": [9, 10]}] * 40)
    up = bench.derive_gate(flat, "two_day", "m", ">=", 40)
    # No seed-to-seed spread: the margin is the floor, 0.90 - 0.05.
    assert up["se_dev"] == pytest.approx(0.0, abs=1e-12)
    assert up["margin"] == bench.MARGIN_FLOOR
    assert up["bound"] == 0.85 and up["gated"]
    low = _records("two_day", [{"m": [1, 10]}] * 40)
    down = bench.derive_gate(low, "two_day", "m", "<=", 40)
    assert down["bound"] == 0.15 and down["gated"]
    # A spread of seeds widens the margin past the floor.
    spread = _records("two_day", [{"m": [10, 10]}, {"m": [5, 10]}] * 20)
    wide = bench.derive_gate(spread, "two_day", "m", ">=", 40)
    assert wide["pooled_rate"] == 0.75
    assert wide["margin"] > bench.MARGIN_FLOOR
    assert wide["margin"] == pytest.approx(
        bench.MARGIN_SE_MULTIPLIER * wide["se_dev"] * np.sqrt(2.0)
    )
    assert wide["bound"] == np.floor((0.75 - wide["margin"]) * 100) / 100
    assert (wide["min"], wide["median"], wide["max"]) == (0.5, 0.75, 1.0)
    # A bound below a coin flip is reported, not gated.
    weak = _records("two_day", [{"m": [5, 10]}] * 40)
    assert not bench.derive_gate(weak, "two_day", "m", ">=", 40)["gated"]


def test_evaluate_gates_compares_pooled_rates_exactly():
    """Pooled rates on the threshold pass; below, violations or gaps fail."""
    base = {
        "pair_tracked:all": [39, 50],
        "pair_tracked:partial": [37, 50],
        "pair_precision": [37, 50],
        "incorrect_identity": [3, 50],
        "distractor_emitted": [23, 100],
        "partial_bundled": [8, 8],
        **{metric: [0, 10] for metric in bench.INVARIANT_METRICS},
    }
    records = _records("two_day", [base, base])
    gates = {g.gate_id: g for g in bench.evaluate_gates(records, "two_day")}
    expected_ids = {spec.gate_id for spec in bench.GATES["two_day"]} | {
        f"invariant-{metric}" for metric in bench.INVARIANT_METRICS
    }
    assert set(gates) == expected_ids
    # 78/100 == 0.78, 74/100 == 0.74, 6/100 == 0.06, 46/200 == 0.23.
    assert all(g.passed is True for g in gates.values())
    below = dict(base, **{"pair_tracked:all": [38, 50]})
    violated = dict(base, same_input_group=[1, 10])
    records = _records("two_day", [base, below])
    by_id = {g.gate_id: g for g in bench.evaluate_gates(records, "two_day")}
    assert by_id["two-day-recall"].passed is False
    records = _records("two_day", [base, violated])
    by_id = {g.gate_id: g for g in bench.evaluate_gates(records, "two_day")}
    assert by_id["invariant-same_input_group"].passed is False
    missing = {k: v for k, v in base.items() if k != "pair_precision"}
    records = _records("two_day", [missing])
    by_id = {g.gate_id: g for g in bench.evaluate_gates(records, "two_day")}
    assert by_id["two-day-precision"].passed is None


def _held_out_gates(scenario, out_dir):
    """Run the held-out seeds of ``scenario`` and evaluate its gates."""
    seeds = range(
        bench.HELD_OUT_FIRST_SEED,
        bench.HELD_OUT_FIRST_SEED + bench.HELD_OUT_SEEDS,
    )
    development = range(
        bench.DEVELOPMENT_FIRST_SEED,
        bench.DEVELOPMENT_FIRST_SEED + bench.DEVELOPMENT_SEEDS,
    )
    assert not set(seeds) & set(development)
    records = [bench.run_one(scenario, seed, out_dir) for seed in seeds]
    gates = bench.evaluate_gates(records, scenario)
    table = "\n".join(bench.format_gate_line(g) for g in gates)
    expected = {spec.gate_id for spec in bench.GATES[scenario]} | {
        f"invariant-{metric}" for metric in bench.INVARIANT_METRICS
    }
    assert {g.gate_id for g in gates} == expected, table
    return gates, table


@pytest.mark.slow
def test_daily_concat_matches_planted_units(tmp_path):
    """Held-out evaluation of the two-day gates (seeds 100..139).

    Each day is a same-day concatenation of two independently simulated
    members, and every neuron's day units are sorted into sparse, unrelated
    ids. Pooled over the held-out seeds, with real SpikeInterface bundle
    extraction inside each day's statistics spans, real UnitMatchPy and the
    production tracked-unit graph, the run must meet every gate in
    ``GATES["two_day"]``: pair recall through tracked units, recall of
    neurons firing in only one member of each day, pair precision, the
    incorrect-identity rate, the distractor false-match rate, every
    partial-member unit bundled, and zero structural violations (no tracked
    unit holds two units of one day; production per-recording counts and
    tracked-unit session / input counts equal the planted truth).

    The gates were derived from development seeds 0..39 only, with the
    margin rule in the benchmark module's docstring. Do not change the
    thresholds, seeds, dataset or scoring to make this pass, and do not
    xfail or skip it: a failure is a reported result. DB-free.
    """
    pytest.importorskip("UnitMatchPy")
    gates, table = _held_out_gates("two_day", tmp_path)
    assert all(g.passed is True for g in gates), table


@pytest.mark.slow
def test_three_day_concat_matches_planted_units(tmp_path):
    """Held-out evaluation of the three-day gates (seeds 100..139).

    Three concatenated days with stable neurons, neurons absent on day 2,
    neurons whose templates change gradually, conflicting pairs of
    near-identical neurons and per-day distractors. Pooled over the held-out
    seeds, the run must meet every gate in ``GATES["three_day"]``: stable
    pair recall through tracked units, the day 1 -- day 3 linkage of neurons
    absent on day 2, pair precision, the incorrect-identity rate, the
    distractor false-match rate and zero structural violations.

    Gradual-change and conflicting-pair recall are printed, not asserted:
    on the development seeds their derived bounds fell below 0.50
    (``UNGATED_DIAGNOSTICS``). The gates were derived from development seeds
    0..39 only. Do not change the thresholds, seeds, dataset or scoring to
    make this pass, and do not xfail or skip it. DB-free.
    """
    pytest.importorskip("UnitMatchPy")
    gates, table = _held_out_gates("three_day", tmp_path)
    assert all(g.passed is True for g in gates), table

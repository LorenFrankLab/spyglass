"""Known-answer benchmark: matching independently sorted daily concatenations.

Each simulated day is a same-day concatenation of two member recordings of
unequal length, joined into ONE SpikeInterface segment with the production
:func:`spyglass.spikesorting.v2._concat_recording.build_concatenated_recording`.
One member of every day carries a silenced artifact exclusion
(:func:`spyglass.spikesorting.v2._sorting_artifact_mask.silence_frame_ranges`),
and the day's statistics spans come from the production
:func:`spyglass.spikesorting.v2._sorting_artifact_mask.statistics_spans`. The
"sort" of a day is the planted ground truth (no sorter), with sparse unit ids
drawn independently per day, so a neuron's identity cannot be read off its id.

Every day goes through the production path the ``UnitMatch`` table runs:
:func:`spyglass.spikesorting.v2._unitmatch_backend.extract_unitmatch_bundle`
with the day's statistics spans, :meth:`UnitMatchBackend.match` (real
UnitMatchPy) over the days in chronological order,
:func:`spyglass.spikesorting.v2._matcher_graph.canonicalize_match_pairs`, and
:func:`spyglass.spikesorting.v2._matcher_graph.derive_tracked_units` with the
per-recording detection map built by
:func:`spyglass.spikesorting.v2._matcher_graph.count_recording_spikes`. Every
target comes from the planted truth, never from the code under test.

Neurons (per seed; templates from SpikeInterface ``generate_templates`` with
per-neuron waveform parameters drawn once from SpikeInterface's default
ranges, drawn locations from ``generate_unit_locations`` at a 20 um minimum
distance (see "Spacing" below), spikes from ``generate_sorting`` at 10 Hz,
noise 5 uV, 16 channels in two columns at 20 um pitch, 30 kHz -- the
half-split experiment's probe, noise and rates):

``two_day``
    Two days. 16 ``shared`` neurons fire in both members of both days; 4
    ``partial`` neurons are present on both days but fire in only ONE member of
    each day (the member alternates by neuron); 4 ``distractor`` neurons per
    day are present on that day only (each its own location and waveform
    draw). 24 units per day.
``three_day``
    Three days. 10 ``stable`` neurons (same template every day); 4
    ``gradual`` neurons whose amplitude scales by 1.00 / 0.85 / 0.70 and whose
    location moves 0 / 4 / 8 um along the probe axis on days 1 / 2 / 3; 4
    ``reappear`` neurons present on days 1 and 3 and absent on day 2; 2
    ``conflict`` pairs, each two neurons with identical waveform parameters
    16 um apart along the probe axis, one of which moves 0 / 4 / 8 um toward
    the other on days 1 / 2 / 3; 3 ``distractor`` neurons per day. 25 / 21 / 25
    units on days 1 / 2 / 3.

Spacing: in ``two_day`` no neuron moves, so every two neurons of a day are
at least 20 um apart. In ``three_day`` only the drawn neurons (every neuron
but the conflict partners) are held to 20 um, and only at their first-day
location. A conflict partner is placed 16 um from its mover with no check
against the other neurons, and the gradual and mover shifts are not checked
either, so other same-day pairs closer than 20 um exist by construction (on
the development seeds: 379 such pairs outside the designed mover / partner
pairs, in all 40 seeds, the closest 4.3 um).

Day layout (member durations in s; the exclusion is 1 s, silenced, inside the
named member): ``two_day`` day 1 (35, 25), exclusion in member 0 at 14 s;
day 2 (25, 40), exclusion in member 1 at 16 s. ``three_day`` day 1 (35, 25),
member 0 at 14 s; day 2 (30, 20), member 1 at 8 s; day 3 (25, 40), member 1
at 16 s. Each member is its own recording session for the detected-session
count.

Scored per run (:func:`score_run`) and pooled over seeds
(:func:`pooled_counts`), every value an integer ``[numerator,
denominator]``:

- ``pair_emitted:<class>`` / ``pair_tracked:<class>``: planted cross-day pairs
  (every two days a neuron is present on) emitted by the backend / whose two
  units share one tracked unit; ``<class>`` is a neuron class or ``all``.
  For ``three_day`` the pairs are also split into ``adjacent`` days and the
  day 1 -- day 3 ``skip`` pair (``pair_tracked:reappear`` IS the day 1 -- day 3
  linkage of disappear/reappear neurons).
- ``pair_precision``: emitted pairs that are planted pairs, over emitted
  pairs; ``false_pair:<kind>`` counts the false ones by kind
  (``distractor``, ``conflict_partner``, ``other_neuron``) over emitted pairs.
- ``incorrect_identity``: tracked units holding units of more than one
  planted neuron, over tracked units with two or more members.
- ``distractor_emitted`` / ``distractor_grouped``: distractor units in any
  emitted pair / in a tracked unit with two or more members, over distractor
  units.
- ``singleton``: units of neurons present on two or more days that end in a
  singleton tracked unit, over those units.
- ``identity_complete:<class>``: neurons present on two or more days whose
  tracked unit is exactly their own units, over those neurons.
- ``partial_bundled``: partial-member units that entered their day's bundle.
- Structural checks, each ``[violations, checked]``: ``same_input_group``
  (a tracked unit with two units of one day), ``recording_count_mismatch``
  (production per-recording counts differ from the planted per-member counts),
  ``sessions_detected_mismatch`` (a tracked unit's detected-session count
  differs from the value derived from the planted per-member counts) and
  ``matching_inputs_mismatch`` (a tracked unit's input count differs from
  the number of days among its own members -- a consistency check of the
  production count, not a comparison with the planted truth).

Manifest (fixed before any held-out seed was run):

- Development seeds: 0..39 per scenario (``DEVELOPMENT_*``), the only seeds
  run to derive the gates. A 20-seed pilot (seeds 0..19) came first; its
  standard errors for the classes with four pairs per seed were too wide to
  derive gates from, so the development and held-out counts were both set to
  40 seeds, and the gates below come from seeds 0..39.
- Held-out seeds: 100..139 per scenario (``HELD_OUT_*``), disjoint from the
  development seeds. They were evaluated once against these gates
  (2026-09-29, macOS arm64: every gate passed) and are now spent;
  ``test_daily_concat_matches_planted_units`` and
  ``test_three_day_concat_matches_planted_units`` re-run them as regression
  checks, not as new held-out evaluations.
- Dataset: every module constant above (probe, noise, firing rate, template
  window, location draw, day layouts, class counts, change magnitudes, unit
  id pool).
- Bundle: ``ms_before = ms_after = 1.5``, ``max_spikes_per_unit = 100``,
  ``seed = 0`` (the ``UnitMatchParamsSchema`` defaults), statistics spans
  from each day. Matching: ``match_threshold = 0.5``,
  ``tracked_unit_threshold = 0.5``, ``max_strict_nodes = 2000`` (the schema
  defaults).
- Margin rule (:func:`derive_gate`): ``margin = max(3 * se_diff, 0.05)``,
  ``se_diff = se_dev * sqrt(1 + n_dev / n_held_out)``, ``se_dev`` the
  seed-bootstrap standard error of the development pooled rate. A ``>=``
  gate is ``floor_0.01(pooled - margin)``, a ``<=`` gate is
  ``ceil_0.01(pooled + margin)``; a candidate whose ``>=`` bound falls below
  0.50 (or ``<=`` bound above 0.50) is reported, not gated. The 0.50 cut-off
  is a policy threshold for how weak a gate is still worth asserting, not a
  chance level (random pairing would recover about 1 / n_units of the pairs).
- Gates (:data:`GATES`, pooled over the held-out seeds, exact rational
  comparison; development pooled value, seed min / median / max, margin in
  brackets):

  ``two_day``: ``pair_tracked:all >= 0.78`` [668/800 = 0.835; 0.70 / 0.85 /
  0.95; 0.050]; ``pair_tracked:partial >= 0.74`` [137/160 = 0.856; 0.50 /
  1.00 / 1.00; 0.111]; ``pair_precision >= 0.74`` [669/807 = 0.829; 0.50 /
  0.86 / 1.00; 0.086]; ``incorrect_identity <= 0.06`` [6/674 = 0.009; 0.00 /
  0.00 / 0.07; 0.050]; ``distractor_emitted <= 0.23`` [40/320 = 0.125; 0.00 /
  0.13 / 0.63; 0.102]; ``partial_bundled >= 1.0`` (every partial-member unit
  gets two halves; an invariant, 320/320).

  ``three_day``: ``pair_tracked:stable >= 0.73`` [949/1200 = 0.791; 0.60 /
  0.77 / 0.93; 0.061]; ``pair_tracked:reappear >= 0.50`` (day 1 -- day 3
  linkage) [107/160 = 0.669; 0.25 / 0.75 / 1.00; 0.164]; ``pair_precision >=
  0.65`` [1668/2293 = 0.727; 0.51 / 0.76 / 0.97; 0.077]; ``incorrect_identity
  <= 0.09`` [26/699 = 0.037; 0.00 / 0.00 / 0.17; 0.050]; ``distractor_emitted
  <= 0.23`` [44/360 = 0.122; 0.00 / 0.11 / 0.56; 0.102].

  Both: zero violations of ``same_input_group``, ``recording_count_mismatch``,
  ``sessions_detected_mismatch`` and ``matching_inputs_mismatch``
  (invariants; zero in every development run).
- Not gated (:data:`UNGATED_DIAGNOSTICS`): ``three_day``
  ``pair_tracked:gradual`` (166/480 = 0.346, bound 0.22) and
  ``pair_tracked:conflict`` (276/480 = 0.575, bound 0.44), plus every
  ``pair_emitted``, ``identity_complete``, ``singleton``,
  ``distractor_grouped`` and ``false_pair`` count.

The dataset, scoring and gate functions are importable; UnitMatchPy is
imported only when a bundle is built or matched.

Usage (from the repo root, in the spikesorting-v2 environment with the
matching extra installed)::

    python tests/spikesorting/v2/scripts/unitmatch_daily_concat_benchmark.py \\
        --scenario two_day [--first-seed 0] [--seeds 40] [--out-dir DIR]

The defaults run the development seeds.

Per-run JSON (``<scenario>/seed<seed>/run.json``), ``results.json`` and
``summary.md`` are written to ``--out-dir`` (default: a new temporary
directory) and the summary is printed.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import tempfile
import time
import warnings
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np

FS = 30_000.0
N_CHANNELS = 16
PROBE_KWARGS = {
    "num_columns": 2,
    "num_contact_per_column": [N_CHANNELS // 2] * 2,
    "xpitch": 20,
    "ypitch": 20,
    "contact_shapes": "circle",
    "contact_shape_params": {"radius": 6},
}
NOISE_LEVEL_UV = 5.0
FIRING_RATE_HZ = 10.0
REFRACTORY_MS = 4.0
TEMPLATE_MS_BEFORE = 1.0
TEMPLATE_MS_AFTER = 3.0
UNIT_LOCATION_KWARGS = {
    "margin_um": 10.0,
    "minimum_z": 5.0,
    "maximum_z": 50.0,
    "minimum_distance": 20.0,
    "max_iteration": 500,
}
#: Location draws retried (each with a fresh sub-seed from the seed's
#: generator) before giving up; SpikeInterface raises when one draw cannot
#: meet the minimum distance.
LOCATION_ATTEMPTS = 20
#: Day unit ids are drawn without replacement from ``1 .. UNIT_ID_POOL``.
UNIT_ID_POOL = 300

GRADUAL_AMPLITUDE_SCALE = (1.00, 0.85, 0.70)
GRADUAL_SHIFT_UM = (0.0, 4.0, 8.0)
CONFLICT_OFFSET_UM = 16.0
CONFLICT_SHIFT_UM = (0.0, 4.0, 8.0)

MS_BEFORE = MS_AFTER = 1.5
MAX_SPIKES_PER_UNIT = 100
BUNDLE_SEED = 0
MATCH_THRESHOLD = 0.5
TRACKED_UNIT_THRESHOLD = 0.5
MAX_STRICT_NODES = 2000
JOB_KWARGS = {"n_jobs": 1, "progress_bar": False}

SCENARIOS = ("two_day", "three_day")
_SCENARIO_CODE = {"two_day": 2, "three_day": 3}

#: Development seeds (run to derive the gates) and held-out seeds (a
#: disjoint range, evaluated once against the committed gates).
DEVELOPMENT_FIRST_SEED = 0
DEVELOPMENT_SEEDS = 40
HELD_OUT_FIRST_SEED = 100
HELD_OUT_SEEDS = 40

#: Margin rule (:func:`derive_gate`).
MARGIN_SE_MULTIPLIER = 3.0
MARGIN_FLOOR = 0.05
MIN_GATED_RECALL = 0.50
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 0


@dataclass(frozen=True)
class GateSpec:
    """One acceptance gate on a pooled metric rate."""

    gate_id: str
    metric: str
    comparison: str
    threshold: float


#: Acceptance gates for the held-out seeds, pooled over them. Every
#: threshold was derived from the development seeds only
#: (:func:`derive_gate`); see the module docstring for the derivation.
GATES = {
    "two_day": (
        GateSpec("two-day-recall", "pair_tracked:all", ">=", 0.78),
        GateSpec("two-day-partial-recall", "pair_tracked:partial", ">=", 0.74),
        GateSpec("two-day-precision", "pair_precision", ">=", 0.74),
        GateSpec(
            "two-day-incorrect-identity", "incorrect_identity", "<=", 0.06
        ),
        GateSpec("two-day-distractor", "distractor_emitted", "<=", 0.23),
        GateSpec("two-day-partial-bundled", "partial_bundled", ">=", 1.0),
    ),
    "three_day": (
        GateSpec("three-day-stable-recall", "pair_tracked:stable", ">=", 0.73),
        GateSpec(
            "three-day-reappear-linkage", "pair_tracked:reappear", ">=", 0.50
        ),
        GateSpec("three-day-precision", "pair_precision", ">=", 0.65),
        GateSpec(
            "three-day-incorrect-identity", "incorrect_identity", "<=", 0.09
        ),
        GateSpec("three-day-distractor", "distractor_emitted", "<=", 0.23),
    ),
}

#: Correctness invariants gated in every scenario at zero violations.
INVARIANT_METRICS = (
    "same_input_group",
    "recording_count_mismatch",
    "sessions_detected_mismatch",
    "matching_inputs_mismatch",
)

#: Gate candidates the margin rule left ungated, with the reason.
UNGATED_DIAGNOSTICS = {
    "three_day": {
        "pair_tracked:gradual": (
            "per-neuron amplitude loss of 15 % a day plus 4 um a day of "
            "movement is out of reach: development pooled recall 0.346, "
            "derived bound 0.22 (below 0.50)"
        ),
        "pair_tracked:conflict": (
            "the class falls below the 0.50 cut-off (development pooled "
            "recall 0.575, derived bound 0.44); its day 1 -- day 3 mover pair "
            "is tied by construction, the day-3 mover being as close to its "
            "partner as to its own day-1 location"
        ),
    },
}

#: Metrics the margin rule is applied to, with the passing direction.
GATE_CANDIDATES = {
    "two_day": (
        ("pair_tracked:all", ">="),
        ("pair_tracked:partial", ">="),
        ("pair_precision", ">="),
        ("incorrect_identity", "<="),
        ("distractor_emitted", "<="),
    ),
    "three_day": (
        ("pair_tracked:stable", ">="),
        ("pair_tracked:gradual", ">="),
        ("pair_tracked:reappear", ">="),
        ("pair_tracked:conflict", ">="),
        ("pair_precision", ">="),
        ("incorrect_identity", "<="),
        ("distractor_emitted", "<="),
    ),
}


@dataclass(frozen=True)
class DaySpec:
    """Layout of one simulated day.

    Attributes
    ----------
    member_durations_s : tuple of float
        Duration of each member recording, in member order.
    exclusion : tuple of (int, float, float)
        ``(member_index, start_s, duration_s)``: a silenced artifact exclusion
        inside that member, ``start_s`` measured from the member's start.
    """

    member_durations_s: tuple[float, float]
    exclusion: tuple[int, float, float]


SCENARIO_DAYS = {
    "two_day": (
        DaySpec((35.0, 25.0), (0, 14.0, 1.0)),
        DaySpec((25.0, 40.0), (1, 16.0, 1.0)),
    ),
    "three_day": (
        DaySpec((35.0, 25.0), (0, 14.0, 1.0)),
        DaySpec((30.0, 20.0), (1, 8.0, 1.0)),
        DaySpec((25.0, 40.0), (1, 16.0, 1.0)),
    ),
}

#: Neuron counts per class. ``distractor`` is per day; ``conflict`` counts
#: pairs (two neurons each).
SCENARIO_COUNTS = {
    "two_day": {"shared": 16, "partial": 4, "distractor": 4},
    "three_day": {
        "stable": 10,
        "gradual": 4,
        "reappear": 4,
        "conflict": 2,
        "distractor": 3,
    },
}

#: Classes whose neurons are present on two or more days, per scenario.
MULTI_DAY_CLASSES = {
    "two_day": ("shared", "partial"),
    "three_day": ("stable", "gradual", "reappear", "conflict"),
}


# ----------------------------------------------------------------------------
# Synthetic data
# ----------------------------------------------------------------------------
@dataclass
class Day:
    """One simulated day: the sorter's recording and the planted sorting.

    Attributes
    ----------
    label : str
        ``"day1"``, ``"day2"``, ...; the day's ``sorting_id``.
    recording : spikeinterface BaseRecording
        One segment: the two members joined, the exclusion silenced.
    sorting : spikeinterface BaseSorting
        Planted spike trains in the joined frame space, keyed by the day's
        sparse unit ids (sorted ascending).
    member_spans : list of (int, int)
        Half-open frame span of each member in the joined recording.
    exclusion : (int, int)
        Half-open silenced frame range.
    statistics_spans : list of (int, int)
        Artifact-free spans that never cross the member join.
    planted_counts : dict[int, list[int]]
        ``{unit_id: [spikes in member 0, spikes in member 1]}``.
    session_names : list of str
        Recording session of each member.
    """

    label: str
    recording: object
    sorting: object
    member_spans: list
    exclusion: tuple
    statistics_spans: list
    planted_counts: dict
    session_names: list


@dataclass
class Dataset:
    """A scenario's planted neurons and simulated days for one seed."""

    scenario: str
    seed: int
    neurons: list
    days: list

    def neuron_of(self) -> dict:
        """``{(day label, unit id): neuron id}``."""
        return {
            (self.days[d].label, uid): n["neuron_id"]
            for n in self.neurons
            for d, uid in n["unit_ids"].items()
        }


def make_probe():
    """The benchmark probe (16 contacts, two columns, 20 um pitch)."""
    from probeinterface import generate_multi_columns_probe

    probe = generate_multi_columns_probe(**PROBE_KWARGS)
    probe.set_device_channel_indices(np.arange(N_CHANNELS))
    return probe


def _unit_locations(n, channel_locations, rng) -> np.ndarray:
    """``n`` 3D neuron locations at least the minimum distance apart."""
    from spikeinterface.core.generate import generate_unit_locations

    for _ in range(LOCATION_ATTEMPTS):
        try:
            return generate_unit_locations(
                n,
                channel_locations,
                distance_strict=True,
                seed=int(rng.integers(2**31)),
                **UNIT_LOCATION_KWARGS,
            ).astype(float)
        except ValueError:
            continue
    raise RuntimeError(
        f"no {n} neuron locations {UNIT_LOCATION_KWARGS['minimum_distance']} "
        f"um apart in {LOCATION_ATTEMPTS} draws"
    )


def make_neurons(scenario: str, seed: int, channel_locations) -> list[dict]:
    """Draw the planted neuron catalog of one scenario and seed.

    Parameters
    ----------
    scenario : str
        One of :data:`SCENARIOS`.
    seed : int
        Dataset seed.
    channel_locations : np.ndarray, shape (n_channels, 2)
        Probe contact positions (um).

    Returns
    -------
    list of dict
        One dict per neuron, in ``neuron_id`` order: ``neuron_id``, ``cls``,
        ``days`` (day indices it is present on), ``params`` (SpikeInterface
        ``generate_templates`` waveform parameters, one scalar each),
        ``location_um`` and ``amplitude_scale`` (``{day: value}``),
        ``active_members`` (``{day: member indices it fires in}``) and, for a
        conflict neuron, ``partner`` (the other neuron's id). ``unit_ids``
        (``{day: unit id}``) is filled in by :func:`make_dataset`.
    """
    from spikeinterface.core.generate import default_unit_params_range

    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}")
    rng = np.random.default_rng([seed, _SCENARIO_CODE[scenario]])
    counts = SCENARIO_COUNTS[scenario]
    n_days = len(SCENARIO_DAYS[scenario])
    all_days = tuple(range(n_days))

    # (cls, days, role) per neuron; a conflict partner is placed relative to
    # its mover rather than drawn.
    plan = []
    if scenario == "two_day":
        plan += [("shared", all_days, None)] * counts["shared"]
        plan += [("partial", all_days, None)] * counts["partial"]
    else:
        plan += [("stable", all_days, None)] * counts["stable"]
        plan += [("gradual", all_days, None)] * counts["gradual"]
        plan += [("reappear", (0, n_days - 1), None)] * counts["reappear"]
        for _ in range(counts["conflict"]):
            plan += [("conflict", all_days, "mover")]
            plan += [("conflict", all_days, "partner")]
    for day in all_days:
        plan += [("distractor", (day,), None)] * counts["distractor"]

    drawn = [i for i, (_, _, role) in enumerate(plan) if role != "partner"]
    locations = np.zeros((len(plan), 3))
    locations[drawn] = _unit_locations(len(drawn), channel_locations, rng)
    params = {
        key: rng.uniform(lo, hi, len(plan))
        for key, (lo, hi) in default_unit_params_range.items()
    }
    y_mid = float(np.mean(channel_locations[:, 1]))

    neurons = []
    n_partial = 0
    for i, (cls, days, role) in enumerate(plan):
        neuron = {
            "neuron_id": i,
            "cls": cls,
            "days": list(days),
            "params": {key: float(values[i]) for key, values in params.items()},
            "location_um": {d: locations[i].copy() for d in days},
            "amplitude_scale": {d: 1.0 for d in days},
            "active_members": {d: [0, 1] for d in days},
        }
        if cls == "partial":
            # The member alternates by neuron: (0, 0), (1, 0), (0, 1), (1, 1).
            neuron["active_members"] = {
                days[0]: [n_partial % 2],
                days[1]: [(n_partial // 2) % 2],
            }
            n_partial += 1
        elif cls == "gradual":
            sign = float(rng.choice([-1.0, 1.0]))
            for d in days:
                neuron["location_um"][d] = locations[i] + [
                    0.0,
                    sign * GRADUAL_SHIFT_UM[d],
                    0.0,
                ]
                neuron["amplitude_scale"][d] = GRADUAL_AMPLITUDE_SCALE[d]
        elif role == "mover":
            toward = 1.0 if locations[i][1] < y_mid else -1.0
            for d in days:
                neuron["location_um"][d] = locations[i] + [
                    0.0,
                    toward * CONFLICT_SHIFT_UM[d],
                    0.0,
                ]
            neuron["partner"] = i + 1
            neuron["toward"] = toward
        elif role == "partner":
            mover = neurons[i - 1]
            base = locations[i - 1] + [
                0.0,
                mover["toward"] * CONFLICT_OFFSET_UM,
                0.0,
            ]
            neuron["params"] = dict(mover["params"])
            neuron["location_um"] = {d: base.copy() for d in days}
            neuron["partner"] = i - 1
        neurons.append(neuron)
    for neuron in neurons:
        neuron.pop("toward", None)
    return neurons


def _day_unit_ids(neurons, n_days, rng) -> None:
    """Assign each neuron a sparse unit id per day, in place.

    Ids are drawn without replacement from ``1 .. UNIT_ID_POOL`` independently
    per day, and redrawn until no neuron present on two days carries the same
    id on both, so matching by id would never be right.
    """
    for neuron in neurons:
        neuron["unit_ids"] = {}
    for day in range(n_days):
        present = [n for n in neurons if day in n["days"]]
        while True:
            ids = rng.choice(UNIT_ID_POOL, size=len(present), replace=False)
            ids = [int(u) + 1 for u in ids]
            if all(
                uid not in n["unit_ids"].values()
                for n, uid in zip(present, ids)
            ):
                break
        for neuron, uid in zip(present, ids):
            neuron["unit_ids"][day] = uid


def _day_templates(day_neurons, day, channel_locations) -> np.ndarray:
    """Templates ``(n_units, n_samples, n_channels)`` of one day's neurons."""
    from spikeinterface.core.generate import generate_templates

    unit_params = {
        key: np.array([n["params"][key] for n in day_neurons])
        for key in day_neurons[0]["params"]
    }
    unit_params["alpha"] = unit_params["alpha"] * np.array(
        [n["amplitude_scale"][day] for n in day_neurons]
    )
    locations = np.array([n["location_um"][day] for n in day_neurons])
    return generate_templates(
        channel_locations,
        locations,
        FS,
        TEMPLATE_MS_BEFORE,
        TEMPLATE_MS_AFTER,
        seed=0,
        unit_params=unit_params,
    )


def make_day(label, day, spec, neurons, probe, rng) -> Day:
    """Simulate one day's two members, join them and plant the sort.

    Parameters
    ----------
    label : str
        The day's label (``sorting_id``).
    day : int
        Day index.
    spec : DaySpec
        Member durations and exclusion.
    neurons : list of dict
        The catalog from :func:`make_neurons`, with ``unit_ids`` assigned.
    probe : probeinterface.Probe
        The benchmark probe.
    rng : numpy.random.Generator
        Source of the member seeds.
    """
    import spikeinterface as si

    from spyglass.spikesorting.v2._concat_recording import (
        build_concatenated_recording,
        cumulative_member_boundaries,
    )
    from spyglass.spikesorting.v2._sorting_artifact_mask import (
        silence_frame_ranges,
        statistics_spans,
    )

    day_neurons = [n for n in neurons if day in n["days"]]
    templates = _day_templates(day_neurons, day, probe.contact_positions)
    member_recordings, member_trains = [], []
    for member, duration in enumerate(spec.member_durations_s):
        silent = [
            i
            for i, n in enumerate(day_neurons)
            if member not in n["active_members"][day]
        ]
        member_seed = int(rng.integers(2**31))
        member_sorting = si.generate_sorting(
            num_units=len(day_neurons),
            sampling_frequency=FS,
            durations=[duration],
            firing_rates=FIRING_RATE_HZ,
            refractory_period_ms=REFRACTORY_MS,
            empty_units=silent or None,
            seed=member_seed,
        )
        recording, _ = si.generate_ground_truth_recording(
            durations=[duration],
            sampling_frequency=FS,
            sorting=member_sorting,
            probe=probe,
            templates=templates,
            ms_before=TEMPLATE_MS_BEFORE,
            ms_after=TEMPLATE_MS_AFTER,
            noise_kwargs={
                "noise_levels": NOISE_LEVEL_UV,
                "strategy": "on_the_fly",
            },
            seed=member_seed,
        )
        member_recordings.append(recording)
        member_trains.append(
            [
                member_sorting.get_unit_spike_train(uid).astype(np.int64)
                for uid in member_sorting.get_unit_ids()
            ]
        )

    n_samples = [r.get_num_samples() for r in member_recordings]
    ends = cumulative_member_boundaries(n_samples)
    member_spans = [(0, ends[0])] + [
        (ends[k - 1], ends[k]) for k in range(1, len(ends))
    ]
    joined = build_concatenated_recording(member_recordings)
    member, start_s, duration_s = spec.exclusion
    start = member_spans[member][0] + int(round(start_s * FS))
    exclusion = (start, start + int(round(duration_s * FS)))
    if exclusion[1] > member_spans[member][1]:
        raise ValueError(f"{label}: the exclusion leaves member {member}")
    recording = silence_frame_ranges(joined, [exclusion])
    spans = statistics_spans(ends[-1], [exclusion], member_spans)

    trains, planted_counts = {}, {}
    for i, neuron in enumerate(day_neurons):
        uid = neuron["unit_ids"][day]
        parts = [
            member_trains[m][i] + member_spans[m][0]
            for m in range(len(member_spans))
        ]
        trains[uid] = np.concatenate(parts)
        planted_counts[uid] = [int(p.size) for p in parts]
    sorting = si.NumpySorting.from_unit_dict(
        {uid: trains[uid] for uid in sorted(trains)}, FS
    )
    return Day(
        label=label,
        recording=recording,
        sorting=sorting,
        member_spans=[tuple(int(v) for v in s) for s in member_spans],
        exclusion=exclusion,
        statistics_spans=[tuple(int(v) for v in s) for s in spans],
        planted_counts=planted_counts,
        session_names=[f"{label}_member{m}" for m in range(len(member_spans))],
    )


def make_dataset(scenario: str, seed: int) -> Dataset:
    """Build one scenario's neurons and days for ``seed``.

    Deterministic in ``(scenario, seed)``: every random draw comes from one
    generator seeded by both.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}")
    probe = make_probe()
    neurons = make_neurons(scenario, seed, probe.contact_positions)
    rng = np.random.default_rng([seed, _SCENARIO_CODE[scenario], 1])
    specs = SCENARIO_DAYS[scenario]
    _day_unit_ids(neurons, len(specs), rng)
    days = [
        make_day(f"day{d + 1}", d, spec, neurons, probe, rng)
        for d, spec in enumerate(specs)
    ]
    return Dataset(scenario=scenario, seed=seed, neurons=neurons, days=days)


# ----------------------------------------------------------------------------
# Bundles, matching and tracked units (the production path)
# ----------------------------------------------------------------------------
def build_bundle(session_dir, day: Day) -> list[int]:
    """Build one day's bundle with the production extractor and its spans.

    Returns
    -------
    list of int
        Unit ids the extractor left out of the bundle.
    """
    from spyglass.spikesorting.v2._unitmatch_backend import (
        extract_unitmatch_bundle,
    )

    excluded = extract_unitmatch_bundle(
        session_dir,
        day.recording,
        day.sorting,
        ms_before=MS_BEFORE,
        ms_after=MS_AFTER,
        max_spikes_per_unit=MAX_SPIKES_PER_UNIT,
        seed=BUNDLE_SEED,
        job_kwargs=JOB_KWARGS,
        statistics_spans=day.statistics_spans,
    )
    return sorted(int(u) for u in excluded)


def match_days(session_dirs, labels) -> tuple[list, dict | None]:
    """Match the day bundles (chronological order) with the production backend.

    The fitted match-class prior UnitMatch passes to its naive-Bayes step is
    recorded by wrapping ``bayes_functions.apply_naive_bayes`` for the call
    (it only observes the arguments; the result is returned unchanged).

    Returns
    -------
    raw_pairs : list of MatchPair
        :meth:`UnitMatchBackend.match` output.
    fitted : dict or None
        ``match_class_prior``, ``n_expected_matches`` and ``n_units``;
        ``None`` if UnitMatch never reached its naive-Bayes step.
    """
    from spyglass.spikesorting.v2._unitmatch_backend import (
        UnitMatchBackend,
        _require_unitmatch,
    )
    from spyglass.spikesorting.v2.matcher_protocol import SessionMatcherInput

    inputs = [
        SessionMatcherInput(
            curation_key={"sorting_id": label, "curation_id": 0},
            waveform_dir=Path(d),
            channel_positions_path=Path(d) / "channel_positions.npy",
        )
        for label, d in zip(labels, session_dirs)
    ]
    um = _require_unitmatch()
    original_bayes = um.bayes_functions.apply_naive_bayes
    fitted = {}

    def recording_naive_bayes(
        parameter_kernels, priors, predictors, param, cond
    ):
        fitted["match_class_prior"] = float(priors[1])
        fitted["n_expected_matches"] = int(param["n_expected_matches"])
        fitted["n_units"] = int(param["n_units"])
        return original_bayes(
            parameter_kernels, priors, predictors, param, cond
        )

    um.bayes_functions.apply_naive_bayes = recording_naive_bayes
    try:
        raw_pairs = UnitMatchBackend().match(
            inputs, {"match_threshold": MATCH_THRESHOLD}
        )
    finally:
        um.bayes_functions.apply_naive_bayes = original_bayes
    return raw_pairs, (fitted or None)


def track_units(dataset: Dataset, oriented_pairs) -> tuple[list, dict]:
    """Derive tracked units the way ``TrackedUnit.make`` does.

    The node universe is every planted unit of every day (units left out of
    a bundle stay as unmatched nodes, as in ``UnitMatch.MatchableUnit``); the
    detected sessions of a node are the member sessions in which the
    production :func:`count_recording_spikes` finds a spike.

    Returns
    -------
    tracked : list of dict
        :func:`derive_tracked_units` output.
    recording_counts : dict
        ``{day label: {unit id: [count per member]}}`` from
        :func:`count_recording_spikes`.
    """
    from spyglass.spikesorting.v2._matcher_graph import (
        count_recording_spikes,
        derive_tracked_units,
    )

    nodes, input_by_node, detected, recording_counts = [], {}, {}, {}
    for input_index, day in enumerate(dataset.days):
        trains = {
            int(uid): day.sorting.get_unit_spike_train(uid)
            for uid in day.sorting.get_unit_ids()
        }
        counts = count_recording_spikes(trains, day.member_spans)
        recording_counts[day.label] = counts
        for uid, per_member in counts.items():
            node = (day.label, 0, uid)
            nodes.append(node)
            input_by_node[node] = input_index
            detected[node] = {
                day.session_names[m]
                for m, n_spikes in enumerate(per_member)
                if n_spikes > 0
            }
    edges = [
        (
            (p["session_a_sorting_id"], p["session_a_curation_id"]),
            (p["session_b_sorting_id"], p["session_b_curation_id"]),
            p,
        )
        for p in oriented_pairs
    ]
    edges = [
        (
            (a[0], a[1], int(p["unit_a_id"])),
            (b[0], b[1], int(p["unit_b_id"])),
            float(p["match_probability"]),
        )
        for a, b, p in edges
    ]
    tracked = derive_tracked_units(
        nodes,
        edges,
        threshold=TRACKED_UNIT_THRESHOLD,
        max_strict_nodes=MAX_STRICT_NODES,
        input_by_node=input_by_node,
        detected_sessions_by_node=detected,
    )
    return tracked, recording_counts


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------
def truth_pairs(dataset: Dataset) -> dict:
    """Every planted cross-day pair.

    Returns
    -------
    dict
        ``{frozenset({(label_i, uid_i), (label_j, uid_j)}): (cls, span)}`` for
        each neuron and each two days ``i < j`` it is present on; ``span`` is
        ``"adjacent"`` when ``j == i + 1``, else ``"skip"``.
    """
    out = {}
    for neuron in dataset.neurons:
        for i, j in combinations(sorted(neuron["unit_ids"]), 2):
            key = frozenset(
                {
                    (dataset.days[i].label, neuron["unit_ids"][i]),
                    (dataset.days[j].label, neuron["unit_ids"][j]),
                }
            )
            out[key] = (neuron["cls"], "adjacent" if j == i + 1 else "skip")
    return out


def _add(counts, key, hit: bool) -> None:
    pair = counts.setdefault(key, [0, 0])
    pair[0] += int(hit)
    pair[1] += 1


def score_run(
    dataset: Dataset, oriented_pairs, tracked, excluded, recording_counts
) -> dict[str, list[int]]:
    """Score one run against the planted truth.

    Parameters
    ----------
    dataset : Dataset
        The planted neurons and days.
    oriented_pairs : list of dict
        :func:`canonicalize_match_pairs` output.
    tracked : list of dict
        :func:`derive_tracked_units` output (members ``(label, 0, uid)``).
    excluded : dict[str, list[int]]
        Unit ids each day's bundle left out.
    recording_counts : dict
        :func:`count_recording_spikes` output per day label.

    Returns
    -------
    dict[str, list[int]]
        ``{metric: [numerator, denominator]}``; see the module docstring.
    """
    counts: dict[str, list[int]] = {}
    neuron_of = dataset.neuron_of()
    neurons = {n["neuron_id"]: n for n in dataset.neurons}
    multi_day = set(MULTI_DAY_CLASSES[dataset.scenario])
    three_day = dataset.scenario == "three_day"

    emitted = {
        frozenset(
            {
                (p["session_a_sorting_id"], int(p["unit_a_id"])),
                (p["session_b_sorting_id"], int(p["unit_b_id"])),
            }
        )
        for p in oriented_pairs
    }
    group_of = {
        (label, uid): index
        for index, tu in enumerate(tracked)
        for label, _, uid in tu["members"]
    }
    group_size = {index: len(tu["members"]) for index, tu in enumerate(tracked)}

    planted = truth_pairs(dataset)
    for key, (cls, span) in planted.items():
        a, b = tuple(key)
        was_emitted = key in emitted
        same_group = group_of[a] == group_of[b]
        labels = [cls, "all"] + ([f"{cls}:{span}"] if three_day else [])
        for label in labels:
            _add(counts, f"pair_emitted:{label}", was_emitted)
            _add(counts, f"pair_tracked:{label}", same_group)

    counts["pair_precision"] = [0, 0]
    for kind in ("distractor", "conflict_partner", "other_neuron"):
        counts[f"false_pair:{kind}"] = [0, 0]
    for key in emitted:
        a, b = tuple(key)
        na, nb = neurons[neuron_of[a]], neurons[neuron_of[b]]
        true = key in planted
        _add(counts, "pair_precision", true)
        if true:
            kind = None
        elif "distractor" in (na["cls"], nb["cls"]):
            kind = "distractor"
        elif na.get("partner") == nb["neuron_id"]:
            kind = "conflict_partner"
        else:
            kind = "other_neuron"
        for k in ("distractor", "conflict_partner", "other_neuron"):
            _add(counts, f"false_pair:{k}", kind == k)

    counts["incorrect_identity"] = [0, 0]
    counts["same_input_group"] = [0, 0]
    for tu in tracked:
        members = [(label, uid) for label, _, uid in tu["members"]]
        labels = [label for label, _ in members]
        _add(counts, "same_input_group", len(set(labels)) != len(labels))
        if len(members) >= 2:
            ids = {neuron_of[m] for m in members}
            _add(counts, "incorrect_identity", len(ids) > 1)

    emitted_units = {unit for key in emitted for unit in key}
    counts["distractor_emitted"] = [0, 0]
    counts["distractor_grouped"] = [0, 0]
    counts["singleton"] = [0, 0]
    if "partial" in multi_day:
        counts["partial_bundled"] = [0, 0]
    for neuron in dataset.neurons:
        units = [
            (dataset.days[d].label, uid)
            for d, uid in neuron["unit_ids"].items()
        ]
        if neuron["cls"] == "distractor":
            for unit in units:
                _add(counts, "distractor_emitted", unit in emitted_units)
                _add(
                    counts, "distractor_grouped", group_size[group_of[unit]] > 1
                )
            continue
        if neuron["cls"] == "partial":
            for label, uid in units:
                _add(counts, "partial_bundled", uid not in excluded[label])
        if neuron["cls"] in multi_day:
            for unit in units:
                _add(counts, "singleton", group_size[group_of[unit]] == 1)
            groups = {group_of[u] for u in units}
            complete = len(groups) == 1 and group_size[groups.pop()] == len(
                units
            )
            for label in (neuron["cls"], "all"):
                _add(counts, f"identity_complete:{label}", complete)

    counts["recording_count_mismatch"] = [0, 0]
    for day in dataset.days:
        for uid, planted_per_member in day.planted_counts.items():
            _add(
                counts,
                "recording_count_mismatch",
                recording_counts[day.label][uid] != planted_per_member,
            )

    counts["sessions_detected_mismatch"] = [0, 0]
    counts["matching_inputs_mismatch"] = [0, 0]
    days = {day.label: day for day in dataset.days}
    for tu in tracked:
        sessions = {
            days[label].session_names[m]
            for label, _, uid in tu["members"]
            for m, n_spikes in enumerate(days[label].planted_counts[uid])
            if n_spikes > 0
        }
        n_inputs = len({label for label, _, _ in tu["members"]})
        _add(
            counts,
            "sessions_detected_mismatch",
            tu["n_sessions_detected"] != len(sessions),
        )
        _add(
            counts,
            "matching_inputs_mismatch",
            tu["n_matching_inputs"] != n_inputs,
        )
    return counts


def _jsonable_neurons(neurons) -> list[dict]:
    out = []
    for n in neurons:
        out.append(
            {
                **{k: v for k, v in n.items() if k not in ("location_um",)},
                "location_um": {
                    str(d): [float(x) for x in loc]
                    for d, loc in n["location_um"].items()
                },
                "amplitude_scale": {
                    str(d): v for d, v in n["amplitude_scale"].items()
                },
                "active_members": {
                    str(d): v for d, v in n["active_members"].items()
                },
                "unit_ids": {str(d): v for d, v in n["unit_ids"].items()},
            }
        )
    return out


def run_one(scenario: str, seed: int, out_dir) -> dict:
    """Build the dataset, bundle, match, track and score one seed.

    Bundles are written to ``out_dir/<scenario>/seed<seed>/<day label>`` and
    the record to ``run.json`` beside them.
    """
    from spyglass.spikesorting.v2._matcher_graph import (
        canonicalize_match_pairs,
    )

    run_dir = Path(out_dir) / scenario / f"seed{seed}"
    t0 = time.perf_counter()
    dataset = make_dataset(scenario, seed)
    labels = [day.label for day in dataset.days]
    dirs, excluded = [], {}
    # UnitMatchPy prints progress on every save and match; keep it out of the
    # summary.
    with contextlib.redirect_stdout(io.StringIO()):
        t1 = time.perf_counter()
        for day in dataset.days:
            d = run_dir / day.label
            excluded[day.label] = build_bundle(d, day)
            dirs.append(d)
        t2 = time.perf_counter()
        raw_pairs, fitted = match_days(dirs, labels)
    t3 = time.perf_counter()
    oriented = canonicalize_match_pairs(
        raw_pairs, {(label, 0): index for index, label in enumerate(labels)}
    )
    tracked, recording_counts = track_units(dataset, oriented)
    counts = score_run(dataset, oriented, tracked, excluded, recording_counts)
    t4 = time.perf_counter()
    record = {
        "scenario": scenario,
        "seed": seed,
        "units_per_day": {
            day.label: len(day.sorting.get_unit_ids()) for day in dataset.days
        },
        "member_spans": {day.label: day.member_spans for day in dataset.days},
        "exclusions": {day.label: list(day.exclusion) for day in dataset.days},
        "statistics_spans": {
            day.label: day.statistics_spans for day in dataset.days
        },
        "neurons": _jsonable_neurons(dataset.neurons),
        "excluded_unit_ids": excluded,
        "pairs": [
            [
                p["session_a_sorting_id"],
                p["unit_a_id"],
                p["session_b_sorting_id"],
                p["unit_b_id"],
                p["match_probability"],
            ]
            for p in oriented
        ],
        "tracked_units": [
            {
                "members": [[m[0], m[2]] for m in tu["members"]],
                "n_sessions_detected": tu["n_sessions_detected"],
                "n_matching_inputs": tu["n_matching_inputs"],
                "median_match_probability": tu["median_match_probability"],
            }
            for tu in tracked
            if len(tu["members"]) > 1
        ],
        "fitted": fitted,
        "counts": counts,
        "timings_s": {
            "dataset": t1 - t0,
            "bundles": t2 - t1,
            "match": t3 - t2,
            "track_and_score": t4 - t3,
        },
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "run.json", "w") as f:
        json.dump(record, f, indent=1)
    return record


# ----------------------------------------------------------------------------
# Pooling and summary
# ----------------------------------------------------------------------------
def pooled_counts(records, scenario: str) -> dict | None:
    """Sum every ``[numerator, denominator]`` over one scenario's seeds."""
    runs = [r for r in records if r["scenario"] == scenario]
    if not runs:
        return None
    keys = sorted({k for r in runs for k in r["counts"]})
    return {
        k: [
            sum(r["counts"].get(k, [0, 0])[0] for r in runs),
            sum(r["counts"].get(k, [0, 0])[1] for r in runs),
        ]
        for k in keys
    }


def rate(count) -> float:
    """``count[0] / count[1]``; NaN for an empty pool."""
    return count[0] / count[1] if count[1] else float("nan")


def _frac(count) -> str:
    return f"{count[0]}/{count[1]} ({rate(count):.3f})"


# ----------------------------------------------------------------------------
# Gate derivation (development seeds only)
# ----------------------------------------------------------------------------
def pooled_rate_se(records, scenario: str, metric: str) -> float:
    """Seed-bootstrap standard error of one metric's pooled rate.

    Resamples the scenario's seeds with replacement (``N_BOOTSTRAP`` draws,
    generator seeded with ``BOOTSTRAP_SEED``) and recomputes the pooled
    ``sum(numerators) / sum(denominators)`` of each draw, so seeds with more
    pairs weigh more, exactly as in the pooled value itself.
    """
    runs = [r for r in records if r["scenario"] == scenario]
    num = np.array([r["counts"][metric][0] for r in runs], dtype=float)
    den = np.array([r["counts"][metric][1] for r in runs], dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(runs), size=(N_BOOTSTRAP, len(runs)))
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = num[draws].sum(axis=1) / den[draws].sum(axis=1)
    return float(np.nanstd(boot, ddof=1))


def derive_gate(
    records, scenario: str, metric: str, comparison: str, n_held_out: int
) -> dict:
    """Apply the margin rule to one metric's development distribution.

    ``margin = max(MARGIN_SE_MULTIPLIER * se_diff, MARGIN_FLOOR)``, where
    ``se_diff = se_dev * sqrt(1 + n_dev / n_held_out)`` is the standard error
    of the difference between a held-out pooled rate and the development
    pooled rate (``se_dev`` from :func:`pooled_rate_se`; the held-out pool's
    error scales with ``1 / sqrt(n_held_out)``). A ``>=`` gate is the
    development pooled rate minus the margin, rounded DOWN to 0.01; a ``<=``
    gate is the pooled rate plus the margin, rounded UP to 0.01 (after
    rounding to nine decimals, so float noise never moves a bound). A ``>=``
    bound below ``MIN_GATED_RECALL`` (or a ``<=`` bound above
    ``1 - MIN_GATED_RECALL``) leaves the metric ungated (``gated`` False):
    the cut-off is a policy threshold for how weak a gate is still worth
    asserting, not a chance level (random pairing recovers about
    ``1 / n_units`` of the pairs).

    Returns
    -------
    dict
        ``pooled`` (count), ``pooled_rate``, per-seed ``min`` / ``median`` /
        ``max``, ``n_dev``, ``se_dev``, ``se_diff``, ``margin``, ``bound``
        and ``gated``.
    """
    runs = [r for r in records if r["scenario"] == scenario]
    per_seed = [rate(r["counts"][metric]) for r in runs]
    per_seed = [v for v in per_seed if not np.isnan(v)]
    pooled = pooled_counts(runs, scenario)[metric]
    pooled_rate = rate(pooled)
    se_dev = pooled_rate_se(runs, scenario, metric)
    se_diff = se_dev * float(np.sqrt(1.0 + len(runs) / n_held_out))
    margin = max(MARGIN_SE_MULTIPLIER * se_diff, MARGIN_FLOOR)
    if comparison == ">=":
        bound = max(0.0, np.floor(round((pooled_rate - margin) * 100, 9)) / 100)
        gated = bound >= MIN_GATED_RECALL
    elif comparison == "<=":
        bound = min(1.0, np.ceil(round((pooled_rate + margin) * 100, 9)) / 100)
        gated = bound <= 1.0 - MIN_GATED_RECALL
    else:
        raise ValueError(f"unknown comparison {comparison!r}")
    return {
        "metric": metric,
        "comparison": comparison,
        "pooled": pooled,
        "pooled_rate": pooled_rate,
        "min": float(np.min(per_seed)),
        "median": float(np.median(per_seed)),
        "max": float(np.max(per_seed)),
        "n_dev": len(runs),
        "se_dev": se_dev,
        "se_diff": se_diff,
        "margin": margin,
        "bound": float(bound),
        "gated": bool(gated),
    }


@dataclass(frozen=True)
class Gate:
    """One gate outcome; ``passed`` is ``None`` when it cannot be evaluated."""

    gate_id: str
    scenario: str
    metric: str
    value: float
    threshold: float
    comparison: str
    passed: bool | None
    detail: str


def evaluate_gates(records, scenario: str) -> list[Gate]:
    """Evaluate one scenario's gates and invariants on the pooled records.

    Each gate compares the pooled rate ``sum(numerators) /
    sum(denominators)`` over every record of ``scenario`` with its
    threshold, in exact :class:`fractions.Fraction` arithmetic (a value on
    the threshold passes). An invariant passes when its pooled violation
    count is zero. A metric with an empty pool (no record, or a zero
    denominator) is not evaluable: ``passed`` is ``None``.
    """
    import fractions

    pooled = pooled_counts(records, scenario) or {}
    specs = list(GATES[scenario]) + [
        GateSpec(f"invariant-{metric}", metric, "<=", 0.0)
        for metric in INVARIANT_METRICS
    ]
    gates = []
    for spec in specs:
        count = pooled.get(spec.metric, [0, 0])
        if not count[1]:
            gates.append(
                Gate(
                    spec.gate_id,
                    scenario,
                    spec.metric,
                    float("nan"),
                    spec.threshold,
                    spec.comparison,
                    None,
                    "empty pool",
                )
            )
            continue
        exact = fractions.Fraction(count[0], count[1])
        threshold = fractions.Fraction(str(spec.threshold))
        passed = (
            exact >= threshold
            if spec.comparison == ">="
            else exact <= threshold
        )
        gates.append(
            Gate(
                spec.gate_id,
                scenario,
                spec.metric,
                float(exact),
                spec.threshold,
                spec.comparison,
                bool(passed),
                f"{count[0]}/{count[1]}",
            )
        )
    return gates


def format_gate_line(gate: Gate) -> str:
    """Render one :class:`Gate` as a single line."""
    verdict = {True: "PASS", False: "FAIL", None: "N/A"}[gate.passed]
    return (
        f"- {verdict} {gate.gate_id} [{gate.scenario}] {gate.metric}: "
        f"{gate.value:.4f} {gate.comparison} {gate.threshold} ({gate.detail})"
    )


def format_derivation(records, scenario: str) -> list[str]:
    """Markdown table of :func:`derive_gate` over the gate candidates."""
    lines = [
        "| metric | pooled | seed min / median / max | se_dev | se_diff | "
        "margin | derived bound | gated |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for metric, comparison in GATE_CANDIDATES[scenario]:
        d = derive_gate(records, scenario, metric, comparison, HELD_OUT_SEEDS)
        lines.append(
            f"| {metric} | {_frac(d['pooled'])} | {d['min']:.3f} / "
            f"{d['median']:.3f} / {d['max']:.3f} | {d['se_dev']:.4f} | "
            f"{d['se_diff']:.4f} | {d['margin']:.4f} | {comparison} "
            f"{d['bound']:.2f} | {'yes' if d['gated'] else 'no'} |"
        )
    return lines


def format_summary(records) -> str:
    """Render pooled and per-seed tables as markdown."""
    lines = ["# UnitMatch daily-concatenation benchmark", ""]
    for scenario in SCENARIOS:
        runs = sorted(
            (r for r in records if r["scenario"] == scenario),
            key=lambda r: r["seed"],
        )
        if not runs:
            continue
        pooled = pooled_counts(records, scenario)
        seeds = [r["seed"] for r in runs]
        lines += [
            f"## {scenario}",
            "",
            f"Seeds {seeds} (n={len(seeds)}); units per day "
            f"{runs[0]['units_per_day']}; pass = both directed probabilities "
            f"> {MATCH_THRESHOLD}; tracked-unit edge threshold "
            f"{TRACKED_UNIT_THRESHOLD}.",
            "",
            "| metric | pooled | per seed |",
            "|---|---|---|",
        ]
        for key, count in pooled.items():
            per_seed = ", ".join(
                (
                    f"{r['counts'][key][0]}/{r['counts'][key][1]}"
                    if key in r["counts"]
                    else "-"
                )
                for r in runs
            )
            lines.append(f"| {key} | {_frac(count)} | {per_seed} |")
        lines += [
            "",
            "| seed | pairs emitted | tracked units >1 | fitted prior | "
            "excluded units | bundles s | match s |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in runs:
            prior = (
                f"{r['fitted']['match_class_prior']:.4f}"
                if r["fitted"]
                else "n/a"
            )
            n_excluded = sum(len(v) for v in r["excluded_unit_ids"].values())
            lines.append(
                f"| {r['seed']} | {len(r['pairs'])} | "
                f"{len(r['tracked_units'])} | {prior} | {n_excluded} | "
                f"{r['timings_s']['bundles']:.1f} | "
                f"{r['timings_s']['match']:.1f} |"
            )
        lines += ["", f"### Gates (pooled over seeds {seeds})", ""]
        lines += [format_gate_line(g) for g in evaluate_gates(runs, scenario)]
        for metric, reason in UNGATED_DIAGNOSTICS.get(scenario, {}).items():
            lines.append(
                f"- not gated: {metric} {_frac(pooled[metric])} -- {reason}"
            )
        lines += [
            "",
            "### Margin rule applied to these seeds",
            "",
            "Meaningful only for development seeds; the committed gates come "
            f"from seeds {DEVELOPMENT_FIRST_SEED}.."
            f"{DEVELOPMENT_FIRST_SEED + DEVELOPMENT_SEEDS - 1}.",
            "",
            *format_derivation(runs, scenario),
            "",
        ]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main(argv=None) -> None:
    """Run the benchmark from the command line."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenario", choices=SCENARIOS, required=True)
    ap.add_argument(
        "--first-seed",
        type=int,
        default=DEVELOPMENT_FIRST_SEED,
        help="first seed (runs first-seed .. first-seed + seeds - 1)",
    )
    ap.add_argument(
        "--seeds", type=int, default=DEVELOPMENT_SEEDS, help="seed count"
    )
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    if args.seeds < 1:
        ap.error("--seeds must be >= 1")
    warnings.filterwarnings("ignore")
    # save_avg_waveforms changes the working directory; keep paths absolute.
    out_dir = (
        Path(tempfile.mkdtemp(prefix="unitmatch_daily_concat_"))
        if args.out_dir is None
        else args.out_dir
    ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"out-dir: {out_dir}", flush=True)

    records = []
    for seed in range(args.first_seed, args.first_seed + args.seeds):
        t0 = time.perf_counter()
        records.append(run_one(args.scenario, seed, out_dir))
        print(
            f"{args.scenario} seed {seed} done in "
            f"{time.perf_counter() - t0:.1f} s",
            flush=True,
        )

    summary = format_summary(records)
    with open(out_dir / "results.json", "w") as f:
        json.dump({"records": records}, f, indent=1)
    (out_dir / "summary.md").write_text(summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()

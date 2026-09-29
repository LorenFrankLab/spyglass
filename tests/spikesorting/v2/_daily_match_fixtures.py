"""Two recording days of the same planted neurons, for daily-concat matching.

Each day is one polymer-shank session (one NWB, the Frank-lab daily shape)
cut into two member intervals separated by an acquisition gap, with a manual
artifact exclusion inside one member. The same 24 neurons -- identical
templates at identical locations (``planted_polymer_recordings``) -- are
planted on both days, each firing on its own Poisson train per day:

- 21 ``shared`` neurons fire in both members of both days;
- one ``partial`` neuron fires on both days, but on day 1 only in member 1;
- one ``day1_only`` and one ``day2_only`` distractor fire on one day and are
  silent on the other.

That is 23 units per day, about the unit count at which UnitMatchPy's
per-call fit is known to work (it fails outright with about 12 units per
session). Neurons sit on 28 um slots along the shank; the slots next to
each distractor are left empty, so a distractor is at least 56 um from every
other neuron.

Day 1 carries the planted rigid zigzag (+/-25 um, 16 s period): 0 um at
16 s, +25 um at 20 s, 0 um at 24 s, -25 um at 28 s. Day 2 is static. The
16 s period was chosen so that day 1's corrected position averages to day
2's: per-day motion correction registers each day to its own mean position,
not across days. In DB-free trials of this design (production bundle,
UnitMatchPy backend and tracked-unit graph), a rigid offset between the days
of 3 / 6 / 12 um recovered 22 / 12 / 1 of 24 neurons, and one drift period
per session (the drift fixture's default, leaving the corrected days about
12 um apart) recovered 6 of 24. The unit count, spacing, distractor
clearance and drift period were fixed after those trials, on these seeds and
templates; the tests' recovery floor was fixed before them. So these days
are co-registered by construction: they exercise the workflow on days whose
corrected positions agree, not recovery across days that moved relative to
each other.

No spike is planted within ``GUARD_S`` of a member edge or an exclusion
edge, so which member a spike belongs to and whether it is excluded are
unambiguous; spikes deeper inside an exclusion are in the raw data and are
masked by the artifact detection.

Everything here is DB-free; the ingesting fixture is
``planted_matching_days`` in ``conftest.py``.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np

from tests.spikesorting.v2._motion_fixtures import SAMPLING_FREQUENCY

#: Team owning both days' session groups (its own owner, so no other
#: fixture's group cleanup removes them).
DAILY_MATCH_TEAM = "daily_match_team"
#: Length of each day's session (s). Member intervals start at or after
#: 16 s so both members' first timestamps share one float64 binade
#: ([16, 32) s), which the concatenation's sampling-rate check needs.
DAY_DURATION_S = 32.0
#: Neuron slots along the shank (um apart), starting 30 um below the tip
#: contact; 28 slots span 756 um of the 806 um shank.
SLOT_PITCH_UM = 28.0
N_SLOTS = 28
#: Role of each occupied slot; the slots next to a distractor stay empty.
_SLOT_ROLES = {7: "day1_only", 13: "partial", 20: "day2_only"}
_EMPTY_SLOTS = {6, 8, 19, 21}
SLOTS = tuple(k for k in range(N_SLOTS) if k not in _EMPTY_SLOTS)
#: Neuron roles, by neuron index (neurons are the occupied slots in order).
ROLES = tuple(_SLOT_ROLES.get(k, "shared") for k in SLOTS)
N_NEURONS = len(ROLES)
#: Neurons present on both days (every one should be matched).
BOTH_DAYS = tuple(i for i, r in enumerate(ROLES) if r in ("shared", "partial"))
FIRING_RATE_HZ = 10.0
REFRACTORY_S = 0.004
#: No spike is planted this close to a member or exclusion edge (s).
GUARD_S = 0.010
#: Template parameters are drawn once from this seed (identical both days).
TEMPLATE_SEED = 20260929
#: Template amplitude range (SpikeInterface's ``alpha``); the upper part of
#: SpikeInterface's default range, so every neuron is well above the noise.
ALPHA_RANGE = (350.0, 500.0)
#: Distance of every neuron from the shank plane (um).
DEPTH_UM = 20.0
#: Period of day 1's planted rigid zigzag (s); chosen so day 1's corrected
#: position averages to day 2's (see the module docstring).
DRIFT_PERIOD_S = 16.0


@dataclass(frozen=True)
class DaySpec:
    """One recording day.

    Attributes
    ----------
    label : str
        ``"day1"`` or ``"day2"``.
    session_start : datetime.datetime
        The NWB session start (UTC).
    members_s : tuple of (float, float)
        Each member interval, seconds on the session clock (from the first
        raw timestamp).
    exclusion : (int, (float, float))
        ``(member_index, (start_s, stop_s))`` manual exclusion, seconds on
        the session clock.
    drifting : bool
        Whether the day carries the planted rigid zigzag.
    noise_seed : int
        Background-noise seed.
    train_seed : int
        Spike-train seed.
    unit_ids : dict[int, int]
        ``{neuron index: unit id}`` of the day's planted sort. The two days
        use disjoint ids, so an id never identifies a neuron.
    """

    label: str
    session_start: dt.datetime
    members_s: tuple
    exclusion: tuple
    drifting: bool
    noise_seed: int
    train_seed: int
    unit_ids: dict


def _present(role: str, day: str) -> bool:
    return not (
        (role == "day1_only" and day == "day2")
        or (role == "day2_only" and day == "day1")
    )


DAYS = (
    DaySpec(
        label="day1",
        session_start=dt.datetime(2023, 7, 1, 12, tzinfo=dt.timezone.utc),
        members_s=((16.2, 23.6), (24.4, 31.8)),
        exclusion=(0, (18.0, 19.0)),
        drifting=True,
        noise_seed=11,
        train_seed=21,
        unit_ids={
            i: i + 1 for i in range(N_NEURONS) if _present(ROLES[i], "day1")
        },
    ),
    DaySpec(
        label="day2",
        session_start=dt.datetime(2023, 7, 2, 12, tzinfo=dt.timezone.utc),
        members_s=((16.4, 23.2), (24.2, 31.6)),
        exclusion=(1, (27.0, 28.0)),
        drifting=False,
        noise_seed=12,
        train_seed=22,
        unit_ids={
            i: 100 + (5 * i) % N_NEURONS
            for i in range(N_NEURONS)
            if _present(ROLES[i], "day2")
        },
    ),
)


def active_members(neuron: int, day: DaySpec) -> tuple[int, ...]:
    """Member indexes in which ``neuron`` fires on ``day`` (may be empty)."""
    role = ROLES[neuron]
    if not _present(role, day.label):
        return ()
    if role == "partial" and day.label == "day1":
        return (1,)
    return (0, 1)


def unit_locations() -> np.ndarray:
    """``(N_NEURONS, 3)`` neuron locations (um): alternating +/-6 um across
    the shank, on their slots along it, ``DEPTH_UM`` from it."""
    slots = np.asarray(SLOTS)
    return np.column_stack(
        [
            np.where(slots % 2 == 0, 6.0, -6.0),
            -(30.0 + SLOT_PITCH_UM * slots),
            np.full(N_NEURONS, DEPTH_UM),
        ]
    )


def unit_params() -> dict:
    """``generate_templates`` parameters per neuron, drawn once.

    Every key of SpikeInterface's ``default_unit_params_range``, drawn
    uniformly from its default range, except ``alpha`` from
    ``ALPHA_RANGE``.
    """
    from spikeinterface.core.generate import default_unit_params_range

    rng = np.random.default_rng(TEMPLATE_SEED)
    params = {}
    for key, (low, high) in default_unit_params_range.items():
        if key == "alpha":
            low, high = ALPHA_RANGE
        params[key] = rng.uniform(low, high, N_NEURONS)
    return params


def _poisson_train(rng, start_s: float, stop_s: float) -> np.ndarray:
    """Poisson spike times in ``[start_s, stop_s)`` with a refractory
    period (s)."""
    times = []
    t = start_s
    while True:
        t += REFRACTORY_S + rng.exponential(1.0 / FIRING_RATE_HZ)
        if t >= stop_s:
            return np.asarray(times)
        times.append(t)


def _frames(times_s: np.ndarray) -> np.ndarray:
    return np.round(np.asarray(times_s) * SAMPLING_FREQUENCY).astype(np.int64)


def planted_trains(day: DaySpec) -> list[np.ndarray]:
    """Each neuron's spike frames on ``day``'s session clock (0 = first raw
    sample).

    A neuron fires only inside its active members, never within
    ``GUARD_S`` of a member or exclusion edge; spikes deeper inside the
    exclusion are kept (they are in the raw data).
    """
    rng = np.random.default_rng(day.train_seed)
    excluded_member, (ex_start, ex_stop) = day.exclusion
    trains = []
    for neuron in range(N_NEURONS):
        # Draw over the whole session so each neuron's draw does not depend
        # on the others' roles, then keep only its active members.
        times = _poisson_train(rng, 0.0, DAY_DURATION_S)
        keep = np.zeros(times.size, dtype=bool)
        for member in active_members(neuron, day):
            start, stop = day.members_s[member]
            keep |= (times >= start + GUARD_S) & (times < stop - GUARD_S)
        near_edge = (np.abs(times - ex_start) < GUARD_S) | (
            np.abs(times - ex_stop) < GUARD_S
        )
        trains.append(_frames(times[keep & ~near_edge]))
    return trains


def expected_member_frames(day: DaySpec) -> dict[int, list[np.ndarray]]:
    """The spikes a sorter of ``day``'s masked concatenation returns.

    ``{unit_id: [session-clock frames in member 0, in member 1]}`` -- the
    planted frames of each present neuron outside the exclusion.
    """
    trains = planted_trains(day)
    excluded_member, (ex_start, ex_stop) = day.exclusion
    ex = _frames([ex_start, ex_stop])
    expected = {}
    for neuron, unit_id in day.unit_ids.items():
        per_member = []
        for member, (start, stop) in enumerate(day.members_s):
            lo, hi = _frames([start, stop])
            frames = trains[neuron][
                (trains[neuron] >= lo) & (trains[neuron] < hi)
            ]
            if member == excluded_member:
                frames = frames[(frames < ex[0]) | (frames >= ex[1])]
            per_member.append(frames)
        expected[unit_id] = per_member
    return expected


def excluded_planted_frames(day: DaySpec) -> np.ndarray:
    """Planted frames inside ``day``'s exclusion (masked, never sorted)."""
    trains = planted_trains(day)
    _member, (ex_start, ex_stop) = day.exclusion
    lo, hi = _frames([ex_start, ex_stop])
    return np.sort(
        np.concatenate([t[(t >= lo) & (t < hi)] for t in trains])
    ).astype(np.int64)


def day_recording(day: DaySpec):
    """``day``'s raw recording: drifting for a drifting day, else static."""
    from tests.spikesorting.v2._motion_fixtures import (
        planted_polymer_recordings,
    )

    drifting, static = planted_polymer_recordings(
        unit_trains=planted_trains(day),
        unit_locations=unit_locations(),
        unit_params=unit_params(),
        duration_s=DAY_DURATION_S,
        drift_period_s=DRIFT_PERIOD_S,
        seed=day.noise_seed,
    )
    return drifting if day.drifting else static

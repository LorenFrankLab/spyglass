"""DB-free graph logic for cross-session unit tracking.

Pure transforms the ``unit_matching`` tables delegate to so the
matcher-output validation and the tracked-unit derivation are unit-testable
without a database:

- :func:`canonicalize_match_pairs` validates raw :class:`MatchPair` records
  against the explicitly pinned matching-input curations and orients each by
  ascending ``input_index``. It rejects pairs whose curation is not pinned,
  same-input pairs (which include self-pairs), and reversed/duplicate pairs --
  so ``(A, B)`` and ``(B, A)`` can never both be inserted into ``UnitMatch.Pair``.
- :func:`input_set_hash` / :func:`chronological_input_order` content-address
  and order the frozen matching inputs of a selection.
- :func:`count_recording_spikes` counts each parent unit's spikes inside each
  constituent recording of its matching input.
- :func:`derive_tracked_units` seeds a graph from the full curated-unit universe,
  enforces the strict node budget, and partitions the units into tracked units
  via a greedy maximal-clique cover (each unit in exactly one tracked unit; the
  strongest overlapping clique wins), with isolated units surfacing as singletons.

None of them imports DataJoint or SpikeInterface, so the module loads without a
database connection.
"""

from __future__ import annotations

from datetime import datetime, timezone
from itertools import combinations
from statistics import median
from typing import TYPE_CHECKING

import numpy as np

from spyglass.spikesorting.v2._core.selection_identity import sha256_json
from spyglass.spikesorting.v2.exceptions import (
    TrackedUnitBudgetExceededError,
)

if TYPE_CHECKING:
    from spyglass.spikesorting.v2.matcher_protocol import MatchPair


def input_set_hash(input_rows, recording_rows) -> str:
    """Content-address a UnitMatch selection's frozen matching inputs.

    The sha256 hex digest over the inputs in ascending ``input_index`` order
    (the frozen chronological order), each contributing its pinned curation
    (``sorting_id``, ``curation_id``, ``curation_uuid``), its source
    (``source_kind``, ``source_id``, ``motion_corrected_recording_id``) and,
    per constituent recording in ``recording_index`` order, the
    ``recording_id``, ``recording_content_hash``, its frozen
    ``session_start_time`` (as UTC, :func:`utc_datetime`), the recording's
    ``[start_sample, end_sample)`` frames in the sort and its kept
    ``valid_times``. An input's ``input_start_time`` is not hashed on its own:
    :func:`frozen_order_errors` requires it to equal the earliest of its
    recordings' session start times. A corrected
    ``Session.session_start_time`` therefore gives a new hash, and selecting
    the same curations again gives a new selection. It is the
    ``input_set_hash`` stored on every ``UnitMatchSelection`` master and the
    single source of truth for that selection's identity:
    ``insert_inputs`` calls it to mint the hash from the part rows it is about
    to insert, and both ``_find_existing_pk`` and ``UnitMatch.make_fetch``
    call it to re-derive the hash from the stored ``Input`` /
    ``InputRecording`` rows and reject a selection whose stored hash
    disagrees with its parts.

    Parameters
    ----------
    input_rows : iterable of dict
        ``UnitMatchSelection.Input`` rows (``input_index``, ``sorting_id``,
        ``curation_id``, ``curation_uuid``, ``source_kind``, ``source_id``,
        ``motion_corrected_recording_id``). UUID values may be ``uuid.UUID``
        or ``str``; both hash identically.
    recording_rows : iterable of dict
        ``UnitMatchSelection.InputRecording`` rows (``input_index``,
        ``recording_index``, ``recording_id``, ``recording_content_hash``,
        ``session_start_time`` (``datetime``; naive is read as UTC, so a
        naive and an aware value of one instant hash identically),
        ``start_sample``, ``end_sample``, ``valid_times`` as an
        ``(n_intervals, 2)`` array of seconds; each float enters the digest
        exactly). Rows whose ``input_index`` has no
        input row do not enter the digest; the caller rejects them with
        :func:`input_part_structure_errors`.

    Returns
    -------
    str
        The 64-character sha256 hex digest.
    """
    recordings_by_input: dict[int, list] = {}
    for row in recording_rows:
        recordings_by_input.setdefault(int(row["input_index"]), []).append(
            [
                int(row["recording_index"]),
                str(row["recording_id"]),
                str(row["recording_content_hash"]),
                utc_datetime(row["session_start_time"]).isoformat(),
                int(row["start_sample"]),
                int(row["end_sample"]),
                np.asarray(row["valid_times"], dtype=np.float64)
                .reshape(-1, 2)
                .tolist(),
            ]
        )
    canonical = []
    for row in sorted(input_rows, key=lambda r: int(r["input_index"])):
        index = int(row["input_index"])
        corrected_id = row.get("motion_corrected_recording_id")
        canonical.append(
            {
                "input_index": index,
                "sorting_id": str(row["sorting_id"]),
                "curation_id": int(row["curation_id"]),
                "curation_uuid": str(row["curation_uuid"]),
                "source_kind": str(row["source_kind"]),
                "source_id": str(row["source_id"]),
                "motion_corrected_recording_id": (
                    None if corrected_id is None else str(corrected_id)
                ),
                "recordings": sorted(recordings_by_input.get(index, [])),
            }
        )
    return sha256_json(canonical, separators=(", ", ": "))


def input_part_structure_errors(input_rows, recording_rows) -> list[str]:
    """Describe structural defects of a selection's ``Input`` parts.

    A selection written by ``insert_inputs`` numbers its inputs ``0..n-1``,
    gives every input at least one constituent recording numbered
    ``0..k-1``, and has no recording row without an input. A raw insert can
    break any of these; the hash alone cannot see a recording row whose input
    is missing, so the selection boundary and ``UnitMatch.make_fetch`` check
    the structure explicitly.

    Parameters
    ----------
    input_rows : iterable of dict
        ``UnitMatchSelection.Input`` rows (only ``input_index`` is read).
    recording_rows : iterable of dict
        ``UnitMatchSelection.InputRecording`` rows (``input_index``,
        ``recording_index``).

    Returns
    -------
    list[str]
        One message per defect; empty when the parts are well formed.
    """
    errors: list[str] = []
    indexes = sorted(int(row["input_index"]) for row in input_rows)
    if indexes != list(range(len(indexes))):
        errors.append(
            f"input_index values {indexes} are not 0..{len(indexes) - 1}"
        )
    recordings_by_input: dict[int, list[int]] = {}
    for row in recording_rows:
        recordings_by_input.setdefault(int(row["input_index"]), []).append(
            int(row["recording_index"])
        )
    orphans = sorted(set(recordings_by_input) - set(indexes))
    if orphans:
        errors.append(f"recording rows for missing input_index {orphans}")
    for index in indexes:
        recording_indexes = sorted(recordings_by_input.get(index, []))
        if not recording_indexes or recording_indexes != list(
            range(len(recording_indexes))
        ):
            errors.append(
                f"input_index {index} has recording_index values "
                f"{recording_indexes}, not 0..k-1 with k >= 1"
            )
    return errors


def utc_datetime(value: datetime) -> datetime:
    """Return ``value`` as a timezone-aware UTC datetime.

    DataJoint returns MySQL ``datetime`` columns naive; a naive value is
    treated as UTC, the convention ``Session.session_start_time`` follows.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def chronological_input_order(inputs: list[dict]) -> list[dict]:
    """Order matching inputs chronologically, the order ``input_index`` freezes.

    Sorts by ``(input_start_time, str(sorting_id), curation_id)``:
    ``input_start_time`` is the earliest session start among an input's
    constituent recordings, and the sorting and curation ids break ties
    deterministically, so any caller order of the same inputs yields the same
    order. UnitMatchPy aligns each session's drift to the previous session in
    feed order, so the matcher is fed in this order.

    Parameters
    ----------
    inputs : list of dict
        Each carrying ``input_start_time`` (``datetime``; naive is read as
        UTC), ``sorting_id`` and ``curation_id``.

    Returns
    -------
    list of dict
        The same dicts, chronologically ordered.
    """
    return sorted(
        inputs,
        key=lambda item: (
            utc_datetime(item["input_start_time"]),
            str(item["sorting_id"]),
            int(item["curation_id"]),
        ),
    )


#: The only tracked-unit policy shipped today (greedy maximal-clique cover).
#: Future policies are pure inserts into ``TrackedUnit.policy_used`` -- no
#: schema migration.
STRICT_POLICY = "strict"

#: Node identity for the tracked-unit graph: one curated unit.
CuratedUnit = tuple  # (sorting_id: str, curation_id: int, unit_id: int)


def frozen_order_errors(input_rows, recording_rows) -> list[str]:
    """Describe frozen start times or input numbering that disagree.

    Each input's ``input_start_time`` must be the earliest
    ``session_start_time`` among its recordings, and ``input_index`` must
    follow :func:`chronological_input_order` of those frozen times -- the
    order the matcher is fed and pairs are oriented by.

    Parameters
    ----------
    input_rows : iterable of dict
        ``UnitMatchSelection.Input`` rows (``input_index``, ``sorting_id``,
        ``curation_id``, ``input_start_time``).
    recording_rows : iterable of dict
        ``UnitMatchSelection.InputRecording`` rows (``input_index``,
        ``session_start_time``); every input has at least one.

    Returns
    -------
    list[str]
        One message per defect; empty when the frozen order holds.
    """
    input_rows = list(input_rows)
    starts_by_input: dict[int, list] = {}
    for row in recording_rows:
        starts_by_input.setdefault(int(row["input_index"]), []).append(
            utc_datetime(row["session_start_time"])
        )
    errors = []
    for row in input_rows:
        index = int(row["input_index"])
        earliest = min(starts_by_input[index])
        if utc_datetime(row["input_start_time"]) != earliest:
            errors.append(
                f"input_index {index} input_start_time "
                f"{utc_datetime(row['input_start_time']).isoformat()} is not "
                f"its earliest session start {earliest.isoformat()}"
            )
    stored = sorted(int(row["input_index"]) for row in input_rows)
    chronological = [
        int(row["input_index"]) for row in chronological_input_order(input_rows)
    ]
    if chronological != stored:
        errors.append(
            f"input_index order {stored} is not chronological by the frozen "
            f"start times (chronological order {chronological})"
        )
    return errors


def count_recording_spikes(unit_spike_trains: dict, spans: list) -> dict:
    """Count each unit's spikes inside each constituent recording's span.

    A matching input's parent unit (a unit of a concatenation sort) covers
    every constituent recording and may fire in only some of them. Its
    spikes are split by the recordings' frozen ``[start_sample,
    end_sample)`` frame spans with
    :func:`spyglass.spikesorting.v2._recording.concat.split_spike_frames_by_spans`, which conserves
    every spike: a spike outside all spans raises rather than going
    uncounted, so the per-recording counts always sum to the unit's total.

    Parameters
    ----------
    unit_spike_trains : dict[int, numpy.ndarray]
        ``{unit_id: spike frames in the input sort's frame space}``.
    spans : list of (int, int)
        ``(start_sample, end_sample)`` of each constituent recording, in
        ``recording_index`` order, contiguous from frame 0.

    Returns
    -------
    dict[int, list[int]]
        ``{unit_id: [n_spikes in recording 0, n_spikes in recording 1,
        ...]}``.

    Raises
    ------
    ConcatSplitError
        If the spans are not contiguous from frame 0 or a spike falls
        outside them.
    """
    from spyglass.spikesorting.v2._recording.concat import (
        split_spike_frames_by_spans,
    )

    per_recording = split_spike_frames_by_spans(unit_spike_trains, spans)
    return {
        int(unit_id): [len(frames[unit_id]) for frames in per_recording]
        for unit_id in unit_spike_trains
    }


def canonicalize_match_pairs(
    pairs: "list[MatchPair]",
    input_index_by_curation: "dict[tuple[str, int], int]",
    *,
    eligible_unit_ids_by_curation: (
        dict[tuple[str, int], set[int]] | None
    ) = None,
) -> list[dict]:
    """Validate and canonically orient raw matcher output before insertion.

    Each :class:`MatchPair` carries ``(sorting_id, curation_id)`` per side; this
    maps both sides to their pinned matching input's ``input_index`` (via
    ``input_index_by_curation``) and:

    - rejects a side whose ``(sorting_id, curation_id)`` is not one of the
      pinned ``UnitMatchSelection.Input`` curations (the matcher returned a key
      it was never fed);
    - rejects a pair whose two sides share an ``input_index`` (a same-input
      pair, which includes self-pairs) -- match pairs must span two distinct
      matching inputs;
    - requires integer curation and unit identifiers without coercion, and
      when prepared unit sets are supplied, rejects units outside those sets;
    - orients each pair so side A is the lower ``input_index``, then rejects a
      second pair that orients to the same ``(input_a, unit_a, input_b,
      unit_b)`` identity (a reversed duplicate), so ``UnitMatch.Pair`` cannot
      hold both orientations.

    Parameters
    ----------
    pairs : list[MatchPair]
        Raw matcher output.
    input_index_by_curation : dict[(str, int), int]
        ``(sorting_id, curation_id) -> input_index`` for the pinned curations.
    eligible_unit_ids_by_curation : dict[(str, int), set[int]], optional
        Units retained by each input's preparer. Compute supplies this after
        exclusions; excluded units remain in the frozen tracking universe but
        cannot occur in a match pair.

    Returns
    -------
    list[dict]
        Oriented pair dicts (deterministically ordered) with the
        ``session_a_*`` / ``session_b_*`` / ``unit_*`` / ``match_probability`` /
        ``drift_estimate_um`` / ``fdr_estimate`` keys matching the
        ``UnitMatch.Pair`` projection, plus ``input_a`` / ``input_b`` indices.

    Raises
    ------
    ValueError
        On invalid identifiers, an ineligible unit, an unpinned curation, a
        same-input pair, or a reversed duplicate.
    """
    from spyglass.spikesorting.v2._core.lookup_validation import lossless_int

    oriented: dict[tuple, dict] = {}
    for pair in pairs:
        side_a = (
            str(pair.session_a_sorting_id),
            lossless_int(pair.session_a_curation_id, "session_a_curation_id"),
        )
        side_b = (
            str(pair.session_b_sorting_id),
            lossless_int(pair.session_b_curation_id, "session_b_curation_id"),
        )
        unit_a = lossless_int(pair.unit_a_id, "unit_a_id")
        unit_b = lossless_int(pair.unit_b_id, "unit_b_id")
        for side, unit in ((side_a, unit_a), (side_b, unit_b)):
            if side not in input_index_by_curation:
                raise ValueError(
                    "UnitMatch.make: matcher returned a pair referencing "
                    f"curation {side} that is not one of the pinned "
                    "UnitMatchSelection.Input curations "
                    f"{sorted(input_index_by_curation)}. The matcher emitted a "
                    "key it was never fed; this is a backend contract violation."
                )
            if (
                eligible_unit_ids_by_curation is not None
                and unit not in eligible_unit_ids_by_curation[side]
            ):
                raise ValueError(
                    "UnitMatch.make: matcher returned unit "
                    f"{unit} from curation {side} that is not eligible in its "
                    "prepared matching input. Units excluded by preparation "
                    "or absent from the frozen matchable universe cannot "
                    "occur in match pairs."
                )
        input_a = input_index_by_curation[side_a]
        input_b = input_index_by_curation[side_b]
        if input_a == input_b:
            raise ValueError(
                "UnitMatch.make: matcher returned a same-input pair (both "
                f"sides are input_index {input_a}; this includes self-pairs). "
                "Match pairs must span two distinct matching inputs."
            )
        # Orient so side A is the lower input_index (canonical orientation).
        if input_a <= input_b:
            low, low_unit, high, high_unit = (
                side_a,
                unit_a,
                side_b,
                unit_b,
            )
            low_input, high_input = input_a, input_b
        else:
            low, low_unit, high, high_unit = (
                side_b,
                unit_b,
                side_a,
                unit_a,
            )
            low_input, high_input = input_b, input_a
        identity = (low_input, low_unit, high_input, high_unit)
        if identity in oriented:
            raise ValueError(
                "UnitMatch.make: matcher returned a reversed/duplicate pair for "
                f"input units {identity}; (A, B) and (B, A) cannot both be "
                "inserted. The backend must emit each unordered pair once."
            )
        fdr = pair.fdr_estimate
        oriented[identity] = {
            "session_a_sorting_id": low[0],
            "session_a_curation_id": low[1],
            "unit_a_id": low_unit,
            "session_b_sorting_id": high[0],
            "session_b_curation_id": high[1],
            "unit_b_id": high_unit,
            "match_probability": float(pair.match_probability),
            "drift_estimate_um": float(pair.drift_estimate_um),
            "fdr_estimate": None if fdr is None else float(fdr),
            "input_a": low_input,
            "input_b": high_input,
        }
    return [oriented[identity] for identity in sorted(oriented)]


def assert_disjoint_input_sessions(nwb_files_by_input: dict) -> None:
    """Reject matching inputs that share a recording session.

    Cross-session matching tracks a unit ACROSS recording sessions, so no two
    matching inputs may draw on the same session (``nwb_file_name``). This
    rejects two single-recording sorts of one nwb, a concatenation together
    with a sort of one of its own members, two concatenations sharing a
    constituent session, and any overlap in acquisition, since two spans of
    one acquisition share its nwb. Matching within one session is out of
    scope.

    Parameters
    ----------
    nwb_files_by_input : dict
        ``{input_label: iterable of nwb_file_name}``; the label names the
        input in the error (e.g. its ``(sorting_id, curation_id)``).

    Raises
    ------
    SameSessionMatchError
        If an ``nwb_file_name`` belongs to two or more inputs.
    """
    from spyglass.spikesorting.v2.exceptions import SameSessionMatchError

    inputs_by_session: dict = {}
    for label, nwb_file_names in nwb_files_by_input.items():
        for nwb_file_name in set(nwb_file_names):
            inputs_by_session.setdefault(nwb_file_name, []).append(label)
    collisions = {
        nwb: labels
        for nwb, labels in inputs_by_session.items()
        if len(labels) > 1
    }
    if collisions:
        detail = "; ".join(
            f"{nwb!r}: inputs {labels}"
            for nwb, labels in sorted(collisions.items())
        )
        raise SameSessionMatchError(
            "UnitMatchSelection: cross-session matching requires every "
            "matching input to come from its own recording sessions, but "
            "these inputs share the same recording session (nwb_file_name) "
            f"-- {detail}. A concatenation already contains its members' "
            "sessions; match it against sorts of other sessions only."
        )


def divergent_electrode_space_members(signatures_by_member) -> list:
    """Return member indexes whose electrode signature differs from the anchor.

    Channel GEOMETRY (positions) can coincide across two physically distinct
    probes / sort groups, so a geometry match alone does not prove two members
    are the same chronic implant. This compares each member's electrode IDENTITY
    signature -- e.g. the ``(electrode_group_name, electrode_id, region)`` tuple
    the concat path uses -- against the anchor (lowest ``member_index``).

    The result is ADVISORY, not a rejection. Unlike concatenation (which reads
    members in one electrode frame, so a mismatch corrupts data and is rejected
    outright), UnitMatch does not stitch recordings, and electrode-group names /
    ids come from each NWB file's ``ElectrodeGroup`` and are NOT guaranteed
    stable across labs' ingestion. So the UnitMatch caller WARNS on a divergence
    rather than blocking a possibly-legitimate chronic match -- a genuine
    distinct-probe mix-up still surfaces as poor matcher AUC / few pairs.

    Parameters
    ----------
    signatures_by_member : dict
        ``{member_index: electrode_signature}``. The signature is any
        equality-comparable value (the caller builds it from the DB).

    Returns
    -------
    list
        Member indexes (sorted, excluding the anchor) whose signature differs
        from the anchor's. Empty when all match or there are fewer than two
        members.
    """
    if len(signatures_by_member) < 2:
        return []
    ordered = sorted(signatures_by_member)
    anchor_signature = signatures_by_member[ordered[0]]
    return [
        member_index
        for member_index in ordered[1:]
        if signatures_by_member[member_index] != anchor_signature
    ]


def derive_tracked_units(
    node_universe: "list[CuratedUnit]",
    edges: "list[tuple[CuratedUnit, CuratedUnit, float]]",
    *,
    threshold: float,
    max_strict_nodes: int,
    policy: str = STRICT_POLICY,
    input_by_node: "dict | None" = None,
    detected_sessions_by_node: "dict | None" = None,
) -> list[dict]:
    """Group curated units into tracked (biological) units via strict cliques.

    The graph is seeded with EVERY node in ``node_universe`` so a unit the
    matcher emitted no pair for still surfaces as a singleton tracked unit. An
    edge is added only when its probability is strictly above ``threshold``.

    Strict tracked units are a **partition**: each curated unit belongs to
    exactly one tracked unit (mirroring UnitMatch's conservative unique-id
    assignment, which keeps one group id per unit). Maximal cliques
    (``networkx.find_cliques``) can overlap -- e.g. edges ``a1-b1``, ``a1-b2``,
    ``a2-b1`` yield three size-2 cliques sharing ``a1``/``b1`` -- so emitting
    every clique would assign one unit to several biological identities. Instead
    the graph is covered greedily by maximal clique, largest first
    (deterministic tie-break by sorted members): each clique contributes only its
    not-yet-claimed members, so a fully-connected group stays whole while
    overlap is resolved into singletons. Among equal-size overlapping cliques the
    strongest (highest median edge probability) is assigned first, so a weaker
    identity never preempts a stronger one (mirroring UnitMatch's
    descending-probability assignment). Isolated nodes are singletons.

    Parameters
    ----------
    node_universe : list of (sorting_id, curation_id, unit_id)
        The full curated-unit universe (one node per matchable curated unit).
    edges : list of ((node_a), (node_b), probability)
        Candidate match edges; only those above ``threshold`` join the graph.
    threshold : float
        Pair-probability cutoff for seeding an edge.
    max_strict_nodes : int
        Hard upper bound on graph size. A larger universe raises
        :class:`TrackedUnitBudgetExceededError` before the (exponential) clique
        search runs.
    policy : str, optional
        Persisted on every row. Only ``"strict"`` ships today.
    input_by_node : dict, optional
        ``{node: matching-input key}`` (e.g. the node's ``input_index``);
        ``n_matching_inputs`` counts distinct values among a tracked unit's
        members. When ``None``, a node's input is its ``(sorting_id,
        curation_id)``, since an input pins one curation per sorting.
    detected_sessions_by_node : dict, optional
        ``{node: iterable of session}``: the original recording sessions
        (``nwb_file_name``) in which the node's unit has at least one spike.
        A concatenation input's parent unit covers several recordings and
        lists only those it fired in; two intervals of one nwb are one
        session, and a unit with no spikes lists none (a node absent from
        the map is detected nowhere). ``n_sessions_detected`` counts the
        distinct sessions over a tracked unit's members, so two sortings
        from one nwb (different sort groups of the same day) count once --
        a within-session match cannot inflate to multi-session. When
        ``None``, each node counts as detected in its own input.

    Returns
    -------
    list[dict]
        One dict per tracked unit (deterministically ordered) with ``members``
        (sorted node tuples), ``n_sessions_detected``, ``n_matching_inputs``,
        ``median_match_probability`` (``None`` for singletons), and
        ``policy_used``.

    Raises
    ------
    TrackedUnitBudgetExceededError
        If ``len(node_universe)`` exceeds ``max_strict_nodes``.
    ValueError
        If ``policy`` is not a shipped policy.
    """
    import networkx as nx

    if policy != STRICT_POLICY:
        raise ValueError(
            f"derive_tracked_units: policy {policy!r} is not supported; only "
            f"{STRICT_POLICY!r} (greedy maximal-clique cover) ships today."
        )

    nodes = [tuple(node) for node in node_universe]
    # Enforce the node budget BEFORE building the graph / running find_cliques --
    # the cap is the guard against the exponential maximal-clique blow-up.
    if len(nodes) > max_strict_nodes:
        raise TrackedUnitBudgetExceededError(
            f"TrackedUnit.make: the curated-unit graph has {len(nodes)} nodes, "
            f"exceeding max_strict_nodes={max_strict_nodes}. Shrink the session "
            "group or raise MatcherParameters.params['max_strict_nodes'] "
            "intentionally (the strict maximal-clique search is exponential in "
            "the worst case)."
        )

    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    node_set = set(nodes)
    for node_a, node_b, probability in edges:
        edge = (tuple(node_a), tuple(node_b))
        # networkx ``add_edge`` silently CREATES missing endpoints, which would
        # smuggle a unit past the node budget and into the partition without it
        # being part of the declared curated-unit universe. Require every edge
        # endpoint to be a seeded node instead.
        if edge[0] not in node_set or edge[1] not in node_set:
            raise ValueError(
                f"derive_tracked_units: edge {edge} references a node absent "
                "from node_universe; edges must be a subset of the curated-unit "
                "universe (seed every matchable unit as a node)."
            )
        if float(probability) > threshold:
            graph.add_edge(*edge, probability=float(probability))

    def _clique_edge_probs(members: list) -> list[float]:
        return [
            graph[u][v]["probability"]
            for u, v in combinations(members, 2)
            if graph.has_edge(u, v)
        ]

    # Greedy maximal-clique cover -> partition. Largest cliques first (so a
    # fully-connected group is emitted whole); among equal-size cliques the
    # STRONGEST wins (highest median edge probability), mirroring UnitMatch's
    # descending-probability assignment so a weaker identity never preempts a
    # stronger overlapping one. Members break any remaining tie deterministically.
    # Each clique contributes only its still-unclaimed members.
    def _clique_strength(members: list) -> float:
        probs = _clique_edge_probs(members)
        return median(probs) if probs else 0.0

    cliques = sorted(
        (sorted(clique) for clique in nx.find_cliques(graph)),
        key=lambda members: (
            -len(members),
            -_clique_strength(members),
            members,
        ),
    )

    def _node_input(node):
        if input_by_node is None:
            return (node[0], node[1])
        return input_by_node[node]

    claimed: set = set()
    tracked: list[dict] = []
    for clique_members in cliques:
        members = [node for node in clique_members if node not in claimed]
        if not members:
            continue
        claimed.update(members)
        inputs = {_node_input(node) for node in members}
        # Count distinct RECORDING SESSIONS in which a member unit fired, not
        # inputs or recordings: two sortings (or two concatenated intervals)
        # of one nwb are one session, and a member with an empty train in a
        # recording is not detected there.
        if detected_sessions_by_node is None:
            sessions = inputs
        else:
            sessions = {
                session
                for node in members
                for session in detected_sessions_by_node.get(node, ())
            }
        edge_probs = _clique_edge_probs(members)
        median_prob = float(median(edge_probs)) if edge_probs else None
        tracked.append(
            {
                "members": members,
                "n_sessions_detected": len(sessions),
                "n_matching_inputs": len(inputs),
                "median_match_probability": median_prob,
                "policy_used": policy,
            }
        )
    # Deterministic order: by the sorted member tuple of each tracked unit.
    tracked.sort(key=lambda tu: tu["members"])
    return tracked

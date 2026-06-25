"""DB-free graph logic for cross-session unit tracking.

Two pure transforms the ``unit_matching`` tables delegate to so the
matcher-output validation and the tracked-unit derivation are unit-testable
without a database:

- :func:`canonicalize_match_pairs` validates raw :class:`MatchPair` records
  against the explicitly pinned per-member curations and orients each by
  ascending ``member_index``. It rejects pairs whose curation is not pinned,
  same-member pairs (which include self-pairs), and reversed/duplicate pairs --
  so ``(A, B)`` and ``(B, A)`` can never both be inserted into ``UnitMatch.Pair``.
- :func:`derive_tracked_units` seeds a graph from the full curated-unit universe,
  enforces the strict node budget, and partitions the units into tracked units
  via a greedy maximal-clique cover (each unit in exactly one tracked unit; the
  strongest overlapping clique wins), with isolated units surfacing as singletons.

Neither function imports DataJoint or SpikeInterface, so the module loads without
a database connection.
"""

from __future__ import annotations

import hashlib
import json
from itertools import combinations
from statistics import median
from typing import TYPE_CHECKING

from spyglass.spikesorting.v2.exceptions import (
    TrackedUnitBudgetExceededError,
)

if TYPE_CHECKING:
    from spyglass.spikesorting.v2.matcher_protocol import MatchPair


def curation_set_hash(member_choices) -> str:
    """Content-address a UnitMatch selection's per-member curation choices.

    The sha256 hex digest over the canonical, ascending-``member_index``
    ordered ``(member_index, sorting_id, curation_id)`` triples. It is the
    ``curation_set_hash`` stored on every ``UnitMatchSelection`` master and
    is the single source of truth for that selection's identity:
    ``insert_selection`` calls it to MINT the hash, and ``UnitMatch.make_fetch``
    calls it to RE-DERIVE the hash from the pinned ``MemberCuration`` rows and
    reject a row whose stored hash disagrees with its parts (a raw-insert
    bypass).

    Parameters
    ----------
    member_choices : iterable of (member_index, sorting_id, curation_id)
        One entry per ``SessionGroup`` member. ``sorting_id`` may be a
        ``uuid.UUID`` or its ``str`` form -- both hash identically because it
        is stringified here, so a freshly-passed str and a DB-fetched UUID
        agree. ``member_index`` / ``curation_id`` are coerced to ``int``.
        Order does not matter: entries are sorted by ``member_index`` first.

    Returns
    -------
    str
        The 64-char sha256 hex digest, byte-compatible with the value stored
        on existing ``UnitMatchSelection`` rows.
    """
    ordered = sorted(
        (
            [int(member_index), str(sorting_id), int(curation_id)]
            for member_index, sorting_id, curation_id in member_choices
        ),
        key=lambda entry: entry[0],
    )
    return hashlib.sha256(
        json.dumps(ordered, sort_keys=True).encode("utf-8")
    ).hexdigest()

#: The only tracked-unit policy shipped today (greedy maximal-clique cover).
#: Future policies are pure inserts into ``TrackedUnit.policy_used`` -- no
#: schema migration.
STRICT_POLICY = "strict"

#: Node identity for the tracked-unit graph: one curated unit.
CuratedUnit = tuple  # (sorting_id: str, curation_id: int, unit_id: int)


def chronological_member_order(member_plan: list[dict]) -> list[dict]:
    """Order member-plan dicts chronologically for the matcher.

    Sorts by ``(recording_date, member_index)`` so a sequential matcher (e.g.
    UnitMatch's drift correction, which aligns each session to the previous one)
    sees sessions in recording order; ``member_index`` breaks ties for same-day
    members. ``recording_date`` is the UTC ISO 8601 string resolved upstream, so
    plain lexicographic order is chronological.
    """
    return sorted(
        member_plan,
        key=lambda plan: (plan["recording_date"], plan["member_index"]),
    )


def canonicalize_match_pairs(
    pairs: "list[MatchPair]",
    member_index_by_curation: "dict[tuple[str, int], int]",
) -> list[dict]:
    """Validate and canonically orient raw matcher output before insertion.

    Each :class:`MatchPair` carries ``(sorting_id, curation_id)`` per side; this
    maps both sides to their pinned ``member_index`` (via
    ``member_index_by_curation``) and:

    - rejects a side whose ``(sorting_id, curation_id)`` is not one of the pinned
      ``UnitMatchSelection.MemberCuration`` rows (the matcher returned a key it
      was never fed);
    - rejects a pair whose two sides share a ``member_index`` (a same-member
      pair, which includes self-pairs) -- cross-session matching pairs must span
      two distinct members;
    - orients each pair so side A is the lower ``member_index``, then rejects a
      second pair that orients to the same ``(member_a, unit_a, member_b,
      unit_b)`` identity (a reversed duplicate), so ``UnitMatch.Pair`` cannot
      hold both orientations.

    Parameters
    ----------
    pairs : list[MatchPair]
        Raw matcher output.
    member_index_by_curation : dict[(str, int), int]
        ``(sorting_id, curation_id) -> member_index`` for the pinned curations.

    Returns
    -------
    list[dict]
        Oriented pair dicts (deterministically ordered) with the
        ``session_a_*`` / ``session_b_*`` / ``unit_*`` / ``match_probability`` /
        ``drift_estimate_um`` / ``fdr_estimate`` keys matching the
        ``UnitMatch.Pair`` projection, plus ``member_a`` / ``member_b`` indices.

    Raises
    ------
    ValueError
        On an unpinned curation, a same-member pair, or a reversed duplicate.
    """
    oriented: dict[tuple, dict] = {}
    for pair in pairs:
        side_a = (str(pair.session_a_sorting_id), int(pair.session_a_curation_id))
        side_b = (str(pair.session_b_sorting_id), int(pair.session_b_curation_id))
        for side in (side_a, side_b):
            if side not in member_index_by_curation:
                raise ValueError(
                    "UnitMatch.make: matcher returned a pair referencing "
                    f"curation {side} that is not one of the pinned "
                    "UnitMatchSelection.MemberCuration rows "
                    f"{sorted(member_index_by_curation)}. The matcher emitted a "
                    "key it was never fed; this is a backend contract violation."
                )
        member_a = member_index_by_curation[side_a]
        member_b = member_index_by_curation[side_b]
        if member_a == member_b:
            raise ValueError(
                "UnitMatch.make: matcher returned a same-member pair (both "
                f"sides are member_index {member_a}; this includes self-pairs). "
                "Cross-session match pairs must span two distinct "
                "SessionGroup.Member rows."
            )
        # Orient so side A is the lower member_index (canonical orientation).
        if member_a <= member_b:
            low, low_unit, high, high_unit = (
                side_a,
                int(pair.unit_a_id),
                side_b,
                int(pair.unit_b_id),
            )
            low_member, high_member = member_a, member_b
        else:
            low, low_unit, high, high_unit = (
                side_b,
                int(pair.unit_b_id),
                side_a,
                int(pair.unit_a_id),
            )
            low_member, high_member = member_b, member_a
        identity = (low_member, low_unit, high_member, high_unit)
        if identity in oriented:
            raise ValueError(
                "UnitMatch.make: matcher returned a reversed/duplicate pair for "
                f"member units {identity}; (A, B) and (B, A) cannot both be "
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
            "member_a": low_member,
            "member_b": high_member,
        }
    return [oriented[identity] for identity in sorted(oriented)]


def derive_tracked_units(
    node_universe: "list[CuratedUnit]",
    edges: "list[tuple[CuratedUnit, CuratedUnit, float]]",
    *,
    threshold: float,
    max_strict_nodes: int,
    policy: str = STRICT_POLICY,
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

    Returns
    -------
    list[dict]
        One dict per tracked unit (deterministically ordered) with ``members``
        (sorted node tuples), ``n_sessions_observed``, ``median_match_probability``
        (``None`` for singletons), and ``policy_used``.

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
        key=lambda members: (-len(members), -_clique_strength(members), members),
    )
    claimed: set = set()
    tracked: list[dict] = []
    for clique_members in cliques:
        members = [node for node in clique_members if node not in claimed]
        if not members:
            continue
        claimed.update(members)
        sessions = {(node[0], node[1]) for node in members}
        edge_probs = _clique_edge_probs(members)
        median_prob = float(median(edge_probs)) if edge_probs else None
        tracked.append(
            {
                "members": members,
                "n_sessions_observed": len(sessions),
                "median_match_probability": median_prob,
                "policy_used": policy,
            }
        )
    # Deterministic order: by the sorted member tuple of each tracked unit.
    tracked.sort(key=lambda tu: tu["members"])
    return tracked

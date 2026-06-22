"""Pure graph utilities for skeleton validation and duplicate detection."""

import re
from difflib import SequenceMatcher

import datajoint as dj
import networkx as nx

from spyglass.utils import logger


def normalize_label(s: str) -> str:
    """Normalize a body part label for matching/comparison.

    Handles camelCase, snake_case, kebab-case, and whitespace.

    Parameters
    ----------
    s : str
        Raw label string.

    Returns
    -------
    str
        Lowercased, space-separated label with camelCase expanded.

    Examples
    --------
    >>> normalize_label('greenLED')
    'green led'
    >>> normalize_label('green_led')
    'green led'
    """
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    s = s.replace("_", " ").replace("-", " ")
    return " ".join(s.strip().lower().split())


def build_canonical_map(known_names: "list[str]") -> "dict[str, str]":
    """Map each normalized label to its canonical (raw) spelling.

    The canonical spelling is the exact string as it appears in *known_names*
    (e.g. the curated ``BodyPart`` table). The returned mapping is the bridge
    used to resolve arbitrary surface forms back to a single canonical
    identity, without ever storing the normalized form as an identifier.

    Parameters
    ----------
    known_names : list[str]
        Canonical body part spellings, e.g. ``BodyPart().fetch('bodypart')``.

    Returns
    -------
    dict[str, str]
        ``{normalize_label(name): name}``.

    Raises
    ------
    datajoint.errors.DataJointError
        If two distinct spellings collapse to the same normalized key, which
        would make the canonical identity ambiguous.

    Examples
    --------
    >>> build_canonical_map(['greenLED', 'earR'])
    {'green led': 'greenLED', 'ear r': 'earR'}
    """
    canon_map: "dict[str, str]" = {}
    for name in known_names:
        key = normalize_label(name)
        existing = canon_map.get(key)
        if existing is not None and existing != name:
            raise dj.DataJointError(
                f"Body part name collision: {existing!r} and {name!r} both "
                f"normalize to {key!r}. Canonical spelling is ambiguous."
            )
        canon_map[key] = name
    return canon_map


def canonicalize(
    name: str, canon_map: "dict[str, str]", default: "str | None" = None
) -> "str | None":
    """Resolve a surface form to its canonical spelling.

    Looks *name* up by its normalized key in *canon_map*. The result is always
    a canonical spelling from the map (or *default* on a miss) -- never the
    normalized form itself.

    Parameters
    ----------
    name : str
        Arbitrary surface form, e.g. ``'EarR'`` or ``'green_led'``.
    canon_map : dict[str, str]
        Mapping from :func:`build_canonical_map`.
    default : str or None, optional
        Returned when *name* has no canonical match, by default None.

    Returns
    -------
    str or None
        The canonical spelling, or *default* if unresolved.

    Examples
    --------
    >>> canonicalize('EarR', {'ear r': 'earR'})
    'earR'
    """
    return canon_map.get(normalize_label(name), default)


def fuzzy_equal(a: str, b: str, threshold: float = 0.85) -> bool:
    """Return True if normalized *a* and *b* are similar above *threshold*.

    Parameters
    ----------
    a, b : str
        Labels to compare.
    threshold : float, optional
        SequenceMatcher ratio threshold, by default 0.85.

    Returns
    -------
    bool
    """
    return (
        SequenceMatcher(None, normalize_label(a), normalize_label(b)).ratio()
        >= threshold
    )


def norm_edges(
    edges: "list[tuple[str, str]]",
) -> "list[tuple[str, str]] | None":
    """Return edges with normalized labels and sorted tuples.

    Handles two non-standard forms found in DLC configs written by Position V1:

    - **Nested groups**: an element is itself a list of pairs, e.g.
      ``[['a','b'], [['c','c'],['d','d']]]``.  One level of nesting is
      flattened and a warning is emitted.
    - **Empty slots**: an element is ``[]`` or has fewer than 2 items.  These
      are silently dropped.

    Parameters
    ----------
    edges : list[tuple[str, str]]
        Raw edge pairs, possibly containing nested groups or empty slots.

    Returns
    -------
    list[tuple[str, str]] or None
        Sorted, deduplicated edge list with normalized labels, or None if
        *edges* is empty/falsy after cleaning.
    """
    if not edges:
        return None

    flat: list = []
    malformed: list = []
    for edge in edges:
        if not edge:  # drop empty slots []
            continue
        if isinstance(edge[0], (list, tuple)):  # nested group
            malformed.append(edge)
            flat.extend(e for e in edge if e)  # skip empty inner slots
        else:
            flat.append(edge)

    if malformed:
        logger.warning(
            f"Skeleton edges contained {len(malformed)} nested group(s) that "
            f"were flattened: {malformed}. "
            "The source DLC config.yaml has a non-standard skeleton format."
        )

    if not flat:
        return None

    return sorted(
        {
            tuple(sorted((normalize_label(u), normalize_label(v))))
            for (u, v) in flat
        }
    )


def shape_hash_from_edges(
    labels: "list[str]", edges: "list[tuple[str, str]] | None"
) -> str:
    """Compute a topology hash on the unlabeled graph defined by *labels* and *edges*.

    Parameters
    ----------
    labels : list[str]
        Body part names (node labels).
    edges : list[tuple[str, str]] or None
        Edge pairs, or None/empty for an edgeless graph.

    Returns
    -------
    str
        Weisfeiler-Lehman graph hash (topology only, labels ignored).

    Raises
    ------
    dj.DataJointError
        If duplicate labels exist after normalization, or an edge references
        an unknown label.
    """
    labels_norm = [normalize_label(x) for x in labels]
    if len(labels_norm) != len(set(labels_norm)):
        raise dj.DataJointError(
            "Duplicate body part names after normalization."
        )

    idx_of = {label: i for i, label in enumerate(labels_norm)}

    if edges:
        for left, right in norm_edges(edges):
            if left not in idx_of or right not in idx_of:
                raise dj.DataJointError(
                    f"Edge ({left!r}, {right!r}) "
                    + "references a label not in bodyparts."
                )

    this_graph = nx.Graph()
    this_graph.add_nodes_from(range(len(labels_norm)))
    this_graph.add_edges_from(
        (idx_of[normalize_label(u)], idx_of[normalize_label(v)])
        for u, v in (edges or [])
    )
    return nx.weisfeiler_lehman_graph_hash(
        this_graph, node_attr=None, edge_attr=None
    )


def build_labeled_graph(
    labels: "list[str]", edges: "list[tuple[str, str]]"
) -> nx.Graph:
    """Build a networkx Graph with normalized node labels.

    Parameters
    ----------
    labels : list[str]
        Body part names.
    edges : list[tuple[str, str]]
        Edge pairs (may be empty or None).

    Returns
    -------
    nx.Graph
        Graph with normalized labels as node names and node attribute ``label``.

    Raises
    ------
    ValueError
        If duplicate labels exist after normalization, or an edge references
        an unknown label.
    """
    labels_norm = [normalize_label(x) for x in labels]
    if len(labels_norm) != len(set(labels_norm)):
        raise ValueError("Duplicate body part names after normalization.")

    this_graph = nx.Graph()
    for label in labels_norm:
        this_graph.add_node(label, label=label)

    if not edges:
        return this_graph

    label_set = set(labels_norm)
    for u, v in norm_edges(edges):
        if u not in label_set or v not in label_set:
            raise ValueError(
                f"Edge uses label not in bodyparts: ({u!r}, {v!r})"
            )

    for u, v in edges:
        this_graph.add_edge(normalize_label(u), normalize_label(v))

    return this_graph


def validate_skeleton_graph(
    bodyparts: "list[str]", edges: "list[tuple[str, str]]"
) -> None:
    """Raise if *bodyparts* / *edges* do not form a valid skeleton description.

    Parameters
    ----------
    bodyparts : list[str]
        Node labels.
    edges : list[tuple[str, str]]
        Pairs of node labels defining connections.

    Raises
    ------
    ValueError
        When *bodyparts* is empty or an edge references an unknown bodypart.
    """
    if not bodyparts:
        raise ValueError("bodyparts must not be empty")
    # Normalize both sides so this check works whether edges carry original
    # labels or have already been passed through norm_edges().
    bp_set = {normalize_label(bp) for bp in bodyparts}
    for edge in edges:
        try:
            a, b = edge
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Each edge must be a (bodypart, bodypart) pair, got: {edge!r}"
            ) from e
        if normalize_label(a) not in bp_set or normalize_label(b) not in bp_set:
            raise ValueError(
                f"Edge ({a!r}, {b!r}) references bodypart(s) "
                + f"not in {sorted(bodyparts)}"
            )


def is_duplicate_skeleton(
    bodyparts_new: "list[str]",
    edges_new: "list[tuple[str, str]]",
    bodyparts_existing: "list[str]",
    edges_existing: "list[tuple[str, str]]",
    threshold: float = 0.85,
) -> bool:
    """Return True if the two skeletons are graph-isomorphic with similar labels.

    Uses :mod:`networkx` graph isomorphism with fuzzy node-label matching.

    Parameters
    ----------
    bodyparts_new, edges_new : list
        Candidate skeleton to test.
    bodyparts_existing, edges_existing : list
        Skeleton already in the database.
    threshold : float, optional
        SequenceMatcher ratio threshold for node-label similarity. Default 0.85.

    Returns
    -------
    bool
    """

    def _build(bps, eds):
        g = nx.Graph()
        for bp in bps:
            g.add_node(bp, label=bp.lower())
        for a, b in eds:
            g.add_edge(a, b)
        return g

    graph_new = _build(bodyparts_new, edges_new)
    graph_old = _build(bodyparts_existing, edges_existing)

    if graph_new.number_of_nodes() != graph_old.number_of_nodes():
        return False
    if graph_new.number_of_edges() != graph_old.number_of_edges():
        return False

    def node_match(a, b):
        ratio = SequenceMatcher(
            None, a.get("label", ""), b.get("label", "")
        ).ratio()
        return ratio >= threshold

    graph_match = nx.algorithms.isomorphism.GraphMatcher(
        graph_new, graph_old, node_match=node_match
    )
    return graph_match.is_isomorphic()

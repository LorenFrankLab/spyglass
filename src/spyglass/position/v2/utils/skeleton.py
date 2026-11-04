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

    Looks *name* up by its normalized key in *canon_map*. The result is a
    canonical spelling from the map, or *default* on a miss, never the
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


def clean_edges(
    edges: "list[tuple[str, str]]",
) -> "list[tuple[str, str]] | None":
    """Structurally clean an edge list, preserving labels verbatim.

    Handles two non-standard forms found in DLC configs written by Position V1,
    without touching the label text (so callers can canonicalize the labels
    into whatever namespace they store):

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
        Sorted, deduplicated edge list with **original** labels, or None if
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
            f"were flattened:\n{malformed}. \n"
            "The source DLC config.yaml has a non-standard skeleton format."
        )

    if not flat:
        return None

    return sorted({tuple(sorted((u, v))) for (u, v) in flat})


def norm_edges(
    edges: "list[tuple[str, str]]",
) -> "list[tuple[str, str]] | None":
    """Return structurally-cleaned edges with **normalized** labels.

    Thin wrapper over :func:`clean_edges` that additionally normalizes labels.
    Used for topology hashing and validation, where labels must collapse to a
    canonical comparison key; callers that *store* edges should instead clean
    structurally and canonicalize into their own namespace.

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
    cleaned = clean_edges(edges)
    if not cleaned:
        return None

    return sorted(
        {
            tuple(sorted((normalize_label(u), normalize_label(v))))
            for (u, v) in cleaned
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
                + f"not in {sorted(bodyparts)}."
                + "\nThis is an edge in your skeleton with invalid body parts."
            )


def is_duplicate_skeleton(
    bodyparts_new: "list[str]",
    edges_new: "list[tuple[str, str]]",
    bodyparts_existing: "list[str]",
    edges_existing: "list[tuple[str, str]]",
    threshold: float = 0.85,
) -> bool:
    """Return True if the skeletons are isomorphic with equal node labels.

    Uses :mod:`networkx` graph isomorphism, matching nodes on their
    normalized labels.

    Parameters
    ----------
    bodyparts_new, edges_new : list
        Candidate skeleton to test.
    bodyparts_existing, edges_existing : list
        Skeleton already in the database.
    threshold : float, optional
        Accepted for callers that still pass it; node matching is exact, so
        it does not affect the result.

    Returns
    -------
    bool
    """
    # Two skeletons are duplicates only if they name the same body parts.
    # Topology isomorphism alone over-merges distinct skeletons whose labels
    # happen to fuzzy-match (e.g. redLED_L vs redLED_R), so require equal
    # normalized node sets first.
    if {normalize_label(b) for b in bodyparts_new} != {
        normalize_label(b) for b in bodyparts_existing
    }:
        return False

    def _build(bps, eds):
        g = nx.Graph()
        for bp in bps:
            g.add_node(bp, label=normalize_label(bp))
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
        # Exact on the normalized label. A similarity ratio made
        # `redLED_L`/`redLED_R` match at 0.889, so skeletons wired to opposite
        # sides compared as duplicates and insertion reused the wrong topology.
        return a.get("label", "") == b.get("label", "")

    graph_match = nx.algorithms.isomorphism.GraphMatcher(
        graph_new, graph_old, node_match=node_match
    )
    return graph_match.is_isomorphic()


def draw_skeleton(skeleton_id: str, bodyparts: list, edges: list):
    """Render a skeleton as a labelled graph.

    Pure rendering: the caller fetches the row and passes its contents, so
    this holds no database access.

    Parameters
    ----------
    skeleton_id : str
        Identifier, used in the title.
    bodyparts : list of str
        Node labels, in their stored (canonical) spelling.
    edges : list
        Pairs of body-part labels. May be empty for a single-part skeleton.

    Returns
    -------
    matplotlib.figure.Figure
        The composed figure. The caller shows or saves it.

    Raises
    ------
    ImportError
        If matplotlib is unavailable.
    """
    try:
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "Skeleton visualization requires matplotlib and networkx. "
            "Install with: pip install matplotlib"
        ) from e

    this_graph = build_labeled_graph(bodyparts, edges)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect("equal")

    # Calculate positions with expanded coordinate space
    if len(bodyparts) == 1:
        pos = {normalize_label(bodyparts[0]): (0, 0)}
    else:
        # Spring layout with larger separation (k parameter)
        pos = nx.spring_layout(this_graph, k=3, iterations=150, seed=42)

    # Create mapping from normalized labels back to original labels
    label_map = {normalize_label(bp): bp for bp in bodyparts}

    # Style settings for zoomed-out view
    node_color = "#E8F4FD"  # Light blue like DataJoint tables
    edge_color = "#2E86AB"  # DataJoint blue
    circle_radius = 0.25  # Smaller circles for better proportions

    # Draw edges first (behind nodes)
    for u, v in this_graph.edges():  # G has normalized node names
        if u in pos and v in pos:
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            ax.plot(
                [x1, x2],
                [y1, y2],
                color=edge_color,
                linewidth=2,
                alpha=0.7,
                zorder=1,
            )

    # Draw bodypart nodes using original names for display
    for norm_label, (x, y) in pos.items():
        original_label = label_map[norm_label]

        # Draw circle
        circle = patches.Circle(
            (x, y),
            circle_radius,
            facecolor=node_color,
            edgecolor=edge_color,
            linewidth=1.5,
            zorder=3,
        )
        ax.add_patch(circle)

        # Add label with appropriate font size
        fontsize = max(8, min(11, int(100 / max(len(original_label), 1))))
        ax.text(
            x,
            y,
            original_label,
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=fontsize,
            fontweight="bold",
            zorder=4,
        )

    # Set plot limits with margins for better zoom
    if pos:
        x_coords = [pos[node][0] for node in pos]
        y_coords = [pos[node][1] for node in pos]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        # Add margins around the skeleton
        x_margin = max(0.5, (x_max - x_min) * 0.2)
        y_margin = max(0.5, (y_max - y_min) * 0.2)
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
    else:
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
    ax.set_title(
        f"Skeleton: {skeleton_id}\n{len(bodyparts)} bodyparts, "
        + f"{len(edges)} connections",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.axis("off")

    # Add legend with appropriately sized elements
    legend_elements = [
        patches.Circle(
            (0, 0),
            0.05,  # Smaller legend circle
            facecolor=node_color,
            edgecolor=edge_color,
            label="Bodypart",
        ),
    ]
    if edges:
        legend_elements.append(
            plt.Line2D(
                [0], [0], color=edge_color, linewidth=2, label="Connection"
            )
        )
    ax.legend(handles=legend_elements, loc="upper right")

    # Add bodypart list as simple text box at bottom
    bodypart_text = f"Bodyparts: {', '.join(sorted(bodyparts))}"
    ax.text(
        0.02,
        0.02,
        bodypart_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
    )

    plt.tight_layout()
    # Don't call plt.show() to avoid double display in notebooks
    return fig


def centroid_method_for(bodyparts) -> str:
    """Choose a centroid method for a set of body parts.

    Pure: the caller fetches the skeleton's parts and passes them, so this
    holds no database access.

    Parameters
    ----------
    bodyparts : iterable of str
        Body-part labels, in any spelling; they are normalized here.

    Returns
    -------
    str
        ``"4pt"`` when all four Frank Lab LEDs are present, ``"2pt"`` for two
        or more parts, ``"1pt"`` for one.

    Raises
    ------
    ValueError
        If *bodyparts* is empty.
    """
    if not bodyparts:
        raise ValueError("No bodyparts to choose a centroid method from")

    # Normalize bodypart names for consistent matching
    normalized_bodyparts = {normalize_label(bp) for bp in bodyparts}

    # Required bodyparts for 4-point Frank Lab method
    frank_lab_parts = {
        normalize_label("greenLED"),
        normalize_label("redLED_C"),
        normalize_label("redLED_L"),
        normalize_label("redLED_R"),
    }

    # All four Frank Lab LEDs: 4pt
    if frank_lab_parts.issubset(normalized_bodyparts):
        return "4pt"

    # Check for minimal LED setup (2pt method)
    led_keywords = ["led", "green", "red", "marker", "point"]
    led_count = 0
    for bp in normalized_bodyparts:
        if any(keyword in bp.lower() for keyword in led_keywords):
            led_count += 1

    # Two or more LED-like parts: 2pt
    if led_count >= 2:
        return "2pt"

    # Two or more parts of any kind: 2pt
    if len(bodyparts) >= 2:
        return "2pt"

    # One part: 1pt
    return "1pt"

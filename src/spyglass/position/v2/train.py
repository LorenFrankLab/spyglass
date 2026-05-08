"""DataJoint tables for training and managing pose estimation models.

NOTE: This approach departs from position v1 by choosing not to store
information already on disk. v1 maintained copies of files in the event that
a user or DLC modified them so that the database always reflected the 'ground
truth'. In practice, file modification is rare, and storing copies is
inefficient.
"""

import re
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Optional, Tuple, Union

import datajoint as dj
import networkx as nx
import pandas as pd
from pynwb import NWBHDF5IO

# Register NWB file in AnalysisNwbfile using any available parent file
from spyglass.common import AnalysisNwbfile, LabMember, Nwbfile, VideoFile
from spyglass.position.utils import suppress_print_from_package
from spyglass.position.utils.path_helpers import resolve_model_path
from spyglass.position.utils.path_helpers import (
    to_stored_path as _to_stored_path,
)
from spyglass.position.utils.protocols import default_pk_name
from spyglass.position.utils.tool_strategies import ToolStrategyFactory
from spyglass.position.utils.yaml_io import load_yaml
from spyglass.position.v2.video import VidFileGroup
from spyglass.utils import SpyglassMixin

# ------------------------------ Optional imports ------------------------------
with suppress_print_from_package():
    try:
        # Only import evaluation function for remaining Model.evaluate()
        from deeplabcut import evaluate_network
    except (ImportError, RecursionError):
        # RecursionError can occur when TF 2.16+ (Keras 3) is installed without
        # tf-keras; the TF compat layer loader recurses infinitely. Set
        # TF_USE_LEGACY_KERAS=1 and install tf-keras to restore TF-backend support.
        evaluate_network = None

# -------------------------------- Module setup --------------------------------
warnings.filterwarnings("ignore", category=UserWarning, module="networkx")

schema = dj.schema(
    "cbroz_position_v2_train"
)  # TODO: remove cbroz prefix  # pylint: disable=fixme


# ----------------------------------- Helpers ----------------------------------


# Parameter dataclasses for methods with 5+ parameters
@dataclass
class ModelMetadata:
    """Parameter object for _register_model_metadata method."""

    model_id: str
    model_path: str
    project_path: Path
    config_path: Path
    params: dict
    config: dict
    latest_model: dict
    skeleton_id: str
    parent_id: Optional[str] = None


# ----------------------- Training-history pure helpers -----------------------

_CSV_PATTERNS = [
    "**/learning_stats.csv",
    "**/log.csv",
    "**/*training*.csv",
]


def discover_training_csvs(model_dir: Path) -> "list[Path]":
    """Return unique CSV files that may contain training loss data.

    Searches *model_dir* and up to two parent directories using the
    standard DLC CSV filename patterns.  Results are deduplicated while
    preserving discovery order.

    Parameters
    ----------
    model_dir : Path
        Directory where the trained model weights live.

    Returns
    -------
    list[Path]
        Unique paths to candidate CSV files.  Empty list if none found.
    """
    search_roots = [model_dir, model_dir.parent, model_dir.parent.parent]
    seen: set = set()
    found: list = []
    for root in search_roots:
        if not root.exists():
            continue
        for pattern in _CSV_PATTERNS:
            for p in root.glob(pattern):
                if p not in seen:
                    seen.add(p)
                    found.append(p)
        if found:
            break  # stop once any root yields results
    return found


def parse_training_csv(path: Path) -> "pd.DataFrame | None":
    """Parse a single training-history CSV into a normalised DataFrame.

    Handles both header-less (DLC ``learning_stats.csv``) and header-bearing
    formats.  Returns ``None`` when the file is empty or has fewer than 2
    columns (not parseable as training data).

    Parameters
    ----------
    path : Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame or None
        Columns: ``iteration``, ``loss``, optionally ``learning_rate``,
        and ``source_file``.  ``None`` on parse failure or insufficient data.
    """
    try:
        # Always read raw (no header) first to detect format
        df_raw = pd.read_csv(path, header=None)
    except Exception:
        return None

    if df_raw.empty or df_raw.shape[1] < 2:
        return None

    # Detect whether the first row is a header (any non-numeric value)
    def _is_numeric(val):
        try:
            float(val)
            return True
        except (TypeError, ValueError):
            return False

    first_row_numeric = all(_is_numeric(v) for v in df_raw.iloc[0])
    if first_row_numeric:
        # Headerless format (DLC learning_stats.csv)
        df = df_raw
    else:
        # Re-read letting pandas use first row as header
        try:
            df = pd.read_csv(path)
        except Exception:
            return None
        if df.empty or df.shape[1] < 2:
            return None

    col_map: dict = {}
    cols = list(df.columns)

    # Map first two columns to canonical names if needed
    if str(cols[0]) != "iteration":
        col_map[cols[0]] = "iteration"
    if str(cols[1]) != "loss":
        col_map[cols[1]] = "loss"

    # Detect learning-rate column by name or position
    lr_candidates = [
        c
        for c in cols
        if isinstance(c, str)
        and ("learning_rate" in c.lower() or c.lower() == "lr")
    ]
    if lr_candidates and str(lr_candidates[0]) != "learning_rate":
        col_map[lr_candidates[0]] = "learning_rate"
    elif (
        not lr_candidates and len(cols) >= 3 and str(cols[2]) != "learning_rate"
    ):
        col_map[cols[2]] = "learning_rate"

    if col_map:
        df = df.rename(columns=col_map)

    df["source_file"] = str(path)
    return df


def aggregate_training_stats(dfs: "list[pd.DataFrame]") -> "pd.DataFrame":
    """Combine a list of per-file training DataFrames into one sorted table.

    Parameters
    ----------
    dfs : list[pd.DataFrame]
        DataFrames produced by :func:`parse_training_csv`.

    Returns
    -------
    pd.DataFrame
        Concatenated data sorted by ``iteration`` (if present).  An empty
        DataFrame is returned when *dfs* is empty.
    """

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    if "iteration" in combined.columns:
        combined = combined.sort_values("iteration").reset_index(drop=True)
    return combined


# ----------------------- Skeleton pure helpers -----------------------


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
    bp_set = set(bodyparts)
    for edge in edges:
        try:
            a, b = edge
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Each edge must be a (bodypart, bodypart) pair, got: {edge!r}"
            ) from e
        if a not in bp_set or b not in bp_set:
            raise ValueError(
                f"Edge ({a!r}, {b!r}) references bodypart(s) "
                + f"not in {sorted(bp_set)}"
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
        SequenceMatcher ratio threshold for node-label similarity, by default 0.85.

    Returns
    -------
    bool
    """

    def _build(bps, eds):
        this_graph = nx.Graph()
        for bp in bps:
            this_graph.add_node(bp, label=bp.lower())
        for a, b in eds:
            this_graph.add_edge(a, b)
        return this_graph

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


# ---------------------------------- Tables -----------------------------------
@schema
class BodyPart(SpyglassMixin, dj.Lookup):
    """Accepted body parts for pose estimation."""

    definition = """
    bodypart: varchar(32)
    """

    # Note: This reflects existing FrankLab body parts, rather than model zoo
    contents = [
        # LEDs
        ["greenLED"],
        ["redLED_C"],
        ["redLED_L"],
        ["redLED_R"],
        ["whiteLED"],
        # Drive
        ["driveBack"],
        ["driveFront"],
        # Head
        ["head"],
        ["nose"],
        ["earL"],
        ["earR"],
        # Body
        ["forelimbL"],
        ["forelimbR"],
        ["hindlimbL"],
        ["hindlimbR"],
        *[[f"spine{i}"] for i in range(1, 6)],
        # Tail
        ["tailBase"],
        ["tailMid"],
        ["tailTip"],
        # DLC example project
        *[[f"bodypart{i}"] for i in range(1, 4)],
        ["objectA"],
    ]

    def insert1(self, key, warn=True, **kwargs):
        """Insert body part into the database."""
        if self & {"bodypart": key["bodypart"]}:
            if warn:
                self._warn_msg(f"Body part {key['bodypart']} already exists")
            return
        if not LabMember().user_is_admin:
            raise PermissionError("Only admins can insert body parts")
        super().insert1(key, **kwargs)


@schema
class Skeleton(SpyglassMixin, dj.Lookup):
    definition = """
    skeleton_id: varchar(32)
    ---
    bodyparts=NULL: blob   # list of body part names, List[str]
    edges=NULL: blob       # list of edge pairs, List[Tuple[str, str]]
    hash=NULL : varchar(32) # hash of graph for duplicate detection
    """

    class BodyPart(dj.Part):
        definition = """
        -> Skeleton
        -> BodyPart
        """

    # Edges are optional to allow unconnected body parts

    contents = [
        ["1LED", "whiteLED", None, "1042f79ba7ecbe933e43d9e23b8bb3e2"],
        [
            "2LED",
            [("greenLED", "redLED_C")],
            None,
            "3bac2dea32a574e9d97917dc8ea8480a",
        ],
        [
            "4LED",
            [
                ("greenLED", "redLED_C"),
                ("greenLED", "redLED_L"),
                ("greenLED", "redLED_R"),
            ],
            None,
            "cfe386902f8c79d228e8ae628079945e",
        ],
    ]

    # Bodyparts for default skeletons (populated via _populate_default_bodyparts)
    _default_bodyparts = {
        "1LED": ["whiteLED"],
        "2LED": ["greenLED", "redLED_C"],
        "4LED": ["greenLED", "redLED_C", "redLED_L", "redLED_R"],
    }

    @classmethod
    def _populate_default_bodyparts(cls):
        """Populate BodyPart part table for default skeleton contents."""
        for skeleton_id, bodyparts in cls._default_bodyparts.items():
            if not (cls & {"skeleton_id": skeleton_id}):
                continue  # Skip if skeleton doesn't exist yet
            for bp in bodyparts:
                cls.BodyPart.insert1(
                    {"skeleton_id": skeleton_id, "bodypart": bp},
                    skip_duplicates=True,
                )

    def get_bodyparts(self, skeleton_id: str = None) -> List[str]:
        """Fetch bodyparts for a skeleton from the BodyPart part table.

        Parameters
        ----------
        skeleton_id : str, optional
            Skeleton ID to fetch bodyparts for. If None, uses current restriction.

        Returns
        -------
        List[str]
            List of bodypart names for this skeleton.
        """
        if skeleton_id:
            query = self.BodyPart & {"skeleton_id": skeleton_id}
        else:
            query = self.BodyPart & self.restriction
        return list(query.fetch("bodypart"))

    # ----------------- static helpers -----------------
    @staticmethod
    def _normalize_label(s: str) -> str:
        """Normalize a body part label for matching/comparison.

        Handles camelCase, snake_case, kebab-case, and whitespace.
        Examples:
            'FirstSecond' -> 'first second'
            'greenLED' -> 'green led'
            'green_led' -> 'green led'
            'green-led' -> 'green led'
        """
        # Insert space before uppercase letters that follow lowercase letters
        # This handles camelCase: 'FirstSecond' -> 'First Second'
        s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
        # Replace underscores and hyphens with spaces
        s = s.replace("_", " ").replace("-", " ")
        # Lowercase, strip, and normalize multiple spaces to single space
        return " ".join(s.strip().lower().split())

    def _fuzzy_equal(self, a: str, b: str, threshold: float = 0.85) -> bool:
        """Fuzzy string equality at given similarity threshold [0..1]."""
        a_norm = self._normalize_label(a)
        b_norm = self._normalize_label(b)
        return SequenceMatcher(None, a_norm, b_norm).ratio() >= threshold

    def _norm_edges(
        self, edges: list[tuple[str, str]]
    ) -> list[tuple[str, str]]:
        """Return edges with normalized labels and sorted tuples."""
        if not edges:
            return None
        return sorted(
            {
                tuple(
                    sorted((self._normalize_label(u), self._normalize_label(v)))
                )
                for (u, v) in edges
            }
        )

    def _validate_bodyparts(self, labels: set[str]) -> bool:
        """Validate that all normalized labels exist in BodyPart table.

        Parameters
        ----------
        labels : set[str]
            Set of normalized body part labels to validate.

        Raises
        ------
        dj.DataJointError
            If any label is not found in the BodyPart table.

        Returns
        -------
        bool
            True if all labels are valid.
        """
        all_parts = BodyPart().fetch("bodypart")
        valid = {self._normalize_label(x) for x in all_parts}
        missing = set(self._normalize_label(x) for x in labels) - valid
        if missing:
            raise dj.DataJointError(
                f"Unknown bodypart name(s) (not in BodyPart): {sorted(missing)}"
                + "\nPlease either change to existing name or "
                + "consult with admin to add it."
            )
        return True

    def _shape_hash_from_edges(
        self, labels: list[str], edges: Union[list[tuple[str, str]], None]
    ) -> str:
        """Compute a topology hash on an unlabeled graph."""

        # Map normalized label -> deterministic node id
        labels_norm = [self._normalize_label(x) for x in labels]
        if len(labels_norm) != len(set(labels_norm)):
            raise dj.DataJointError(
                "Duplicate body part names after normalization."
            )

        idx_of = {label: i for i, label in enumerate(labels_norm)}
        # Verify edges reference known labels
        if edges:
            for left, right in self._norm_edges(edges):
                if left not in idx_of or right not in idx_of:
                    raise dj.DataJointError(
                        f"Edge ({left!r}, {right!r}) "
                        + "references a label not in bodyparts."
                    )

        this_graph = nx.Graph()
        this_graph.add_nodes_from(range(len(labels_norm)))
        this_graph.add_edges_from(
            (
                idx_of[self._normalize_label(u)],
                idx_of[self._normalize_label(v)],
            )
            for u, v in edges
        )
        return nx.weisfeiler_lehman_graph_hash(
            this_graph, node_attr=None, edge_attr=None
        )

    def _build_labeled_graph(
        self, labels: list[str], edges: list[tuple[str, str]]
    ) -> nx.Graph:
        """Build a graph with labeled nodes with normalized labels."""

        labels_norm = [self._normalize_label(x) for x in labels]
        if len(labels_norm) != len(set(labels_norm)):
            raise ValueError("Duplicate body part names after normalization.")

        this_graph = nx.Graph()
        for label in labels_norm:
            this_graph.add_node(label, label=label)

        if not edges:
            return this_graph

        label_set = set(labels_norm)
        # Edge endpoints must belong to the declared label set
        for u, v in self._norm_edges(edges):
            if u not in label_set or v not in label_set:
                raise ValueError(
                    f"Edge uses label not in bodyparts: ({u!r}, {v!r})"
                )

        for u, v in edges:
            this_graph.add_edge(
                self._normalize_label(u), self._normalize_label(v)
            )

        return this_graph

    # ----------------- main insert -----------------

    def insert1(
        self,
        key: dict,
        name_similarity: float = 0.85,
        check_duplicates: bool = True,
        **kwargs,
    ):
        """Insert the skeleton if no duplicate exists.

        Duplicate criteria: same topology and similarly labeled nodes.

        Parameters
        ----------
        key : dict
            Dictionary with 'bodyparts' (List[str]) and 'edges'
            (List[Tuple[str, str]]). Or DLC config.
        name_similarity : float, optional
            Fuzzy matching threshold for body part names [0..1], by default 0.85
        check_duplicates : bool, optional
            Whether to run duplicate detection, by default True
        """
        bodyparts: List[str] = key.get("bodyparts", [])
        edges: List[Tuple[str, str]] = key.get("edges") or key.get(
            "skeleton", []
        )

        if not bodyparts or edges is None:
            raise dj.DataJointError(
                f"Key must include 'bodyparts' and 'edges' fields: {key}"
            )

        # Validate body parts against the reference table (DB query, raises first)
        labels_norm = [self._normalize_label(x) for x in bodyparts]
        _ = self._validate_bodyparts(set(labels_norm))

        # Validate edge structure (pure, no DB — runs after bodypart check so
        # "unknown bodypart" errors surface before structural edge errors)
        validate_skeleton_graph(bodyparts, edges)

        shape_hash = self._shape_hash_from_edges(bodyparts, edges)

        if check_duplicates:
            for row in self & dict(hash=shape_hash):
                row_bodyparts = self.get_bodyparts(row["skeleton_id"])
                if is_duplicate_skeleton(
                    bodyparts,
                    edges,
                    row_bodyparts,
                    row["edges"],
                    name_similarity,
                ):
                    if row.get("bodyparts") is None:
                        super().update1(
                            dict(
                                skeleton_id=row["skeleton_id"],
                                bodyparts=bodyparts,
                            )
                        )
                    return dict(skeleton_id=row["skeleton_id"])

        skeleton_id = default_pk_name("skel", key)
        insert_pk = dict(skeleton_id=skeleton_id)
        super().insert1(
            dict(insert_pk, bodyparts=bodyparts, edges=edges, hash=shape_hash),
            **kwargs,
        )
        self.BodyPart.insert(
            [{"skeleton_id": skeleton_id, "bodypart": bp} for bp in bodyparts],
            skip_duplicates=True,
        )
        return insert_pk

    def show_skeleton(self, skeleton_id: str = None):
        """Display skeleton graph visualization similar to datajoint.Diagram.

        This method visualizes the skeleton structure showing bodyparts as nodes
        and their connections as edges, similar to how DataJoint's Diagram shows
        table relationships.

        Parameters
        ----------
        skeleton_id : str, optional
            Skeleton ID to visualize. If None, uses current restriction.
            If multiple skeletons match, shows the first one.

        Returns
        -------
        matplotlib figure
            Interactive plot showing the skeleton graph structure.

        Examples
        --------
        >>> # Show a specific skeleton
        >>> Skeleton().show_skeleton("4LED")

        >>> # Show skeleton from current restriction
        >>> (Skeleton & {"skeleton_id": "4LED"}).show_skeleton()
        """
        try:
            import matplotlib.patches as patches
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "Skeleton visualization requires matplotlib and networkx. "
                "Install with: pip install matplotlib"
            ) from e

        # Get skeleton data
        if skeleton_id:
            skeleton_data = (self & {"skeleton_id": skeleton_id}).fetch1()
        else:
            skeleton_data = self.fetch1() if len(self) == 1 else self.fetch()[0]

        skeleton_id = skeleton_data["skeleton_id"]
        bodyparts = skeleton_data["bodyparts"]
        edges = skeleton_data["edges"] or []

        if not bodyparts:
            # Use existing method to get bodyparts
            bodyparts = self.get_bodyparts(skeleton_id)

        if not bodyparts:
            print(f"No bodyparts found for skeleton '{skeleton_id}'")
            return None

        # Use existing method to build graph for positioning
        this_graph = self._build_labeled_graph(bodyparts, edges)

        # Create figure with better sizing for zoomed-out view
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_aspect("equal")

        # Calculate positions with expanded coordinate space
        if len(bodyparts) == 1:
            pos = {self._normalize_label(bodyparts[0]): (0, 0)}
        else:
            # Spring layout with larger separation (k parameter)
            pos = nx.spring_layout(this_graph, k=3, iterations=150, seed=42)

        # Create mapping from normalized labels back to original labels
        label_map = {self._normalize_label(bp): bp for bp in bodyparts}

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
            f"Skeleton: {skeleton_id}\n{len(bodyparts)} bodyparts, {len(edges)} connections",
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

    def get_centroid_method(self, skeleton_id: str = None) -> str:
        """Suggest appropriate centroid method for this skeleton.

        Analyzes the bodyparts in the skeleton to recommend the best
        centroid calculation method for position estimation.

        Parameters
        ----------
        skeleton_id : str, optional
            Skeleton ID to analyze. If None, uses current restriction.

        Returns
        -------
        str
            Recommended centroid method ('1pt', '2pt', or '4pt')

        Examples
        --------
        >>> # Get centroid method suggestion
        >>> method = (Skeleton & {"skeleton_id": "4LED"}).get_centroid_method()
        >>> print(f"Recommended method: {method}")
        """
        # Get skeleton data
        if skeleton_id:
            skeleton_data = (self & {"skeleton_id": skeleton_id}).fetch1()
        else:
            skeleton_data = self.fetch1() if len(self) == 1 else self.fetch()[0]

        bodyparts = skeleton_data["bodyparts"]

        if not bodyparts:
            bodyparts = self.get_bodyparts(skeleton_data["skeleton_id"])

        if not bodyparts:
            raise ValueError("No bodyparts found for skeleton")

        # Normalize bodypart names for consistent matching
        normalized_bodyparts = {self._normalize_label(bp) for bp in bodyparts}

        # Required bodyparts for 4-point Frank Lab method
        frank_lab_parts = {
            self._normalize_label("greenLED"),
            self._normalize_label("redLED_C"),
            self._normalize_label("redLED_L"),
            self._normalize_label("redLED_R"),
        }

        # Check if we have all Frank Lab LEDs (4pt method)
        if frank_lab_parts.issubset(normalized_bodyparts):
            return "4pt"

        # Check for minimal LED setup (2pt method)
        led_keywords = ["led", "green", "red", "marker", "point"]
        led_count = 0
        for bp in normalized_bodyparts:
            if any(keyword in bp.lower() for keyword in led_keywords):
                led_count += 1

        # If we have 2+ LED-like bodyparts, suggest 2pt
        if led_count >= 2:
            return "2pt"

        # If we have multiple bodyparts but not LED-like, still suggest 2pt
        if len(bodyparts) >= 2:
            return "2pt"

        # Single bodypart - use 1pt method
        return "1pt"


@schema
class ModelParams(SpyglassMixin, dj.Lookup):
    definition = """
    model_params_id: varchar(32)
    tool: varchar(32)
    ---
    params: json
    -> [nullable] Skeleton
    params_hash: varchar(64) # hash of parameters
    unique index (tool, params_hash)
    """

    # Strategy pattern replaced the tool_info structure
    # Use ToolStrategyFactory to get tool-specific parameters and validation

    default_entries_data = [
        {
            "model_params_id": "dlc_default",
            "tool": "DLC",
            "params": {"shuffle": 1, "trainingsetindex": 0, "model_prefix": ""},
            "skeleton_id": None,
        },
        {
            "model_params_id": "sleap_default",
            "tool": "SLEAP",
            "params": {
                "model_type": "single_instance",
                "backbone": "unet",
                "max_epochs": 200,
                "batch_size": 4,
            },
            "skeleton_id": None,
        },
    ]

    contents = [
        (
            entry["model_params_id"],
            entry["tool"],
            entry["params"],
            entry["skeleton_id"],
            dj.hash.key_hash(entry["params"]),
        )
        for entry in default_entries_data
    ]

    @classmethod
    def tool_info(cls) -> dict:
        """Return tool information using strategy pattern.

        Returns
        -------
        dict
            Tool information with structure:
            {
                tool_name: {
                    "required": set,
                    "accepted": set,
                    "skipped": set,
                    "aliases": dict
                }
            }
        """
        if hasattr(cls, "_cached_tool_info"):
            return cls._cached_tool_info

        tool_info = {}

        try:
            # Build tool_info for each registered strategy
            for tool_name in ToolStrategyFactory.get_available_tools():
                try:
                    strategy = ToolStrategyFactory.create_strategy(tool_name)
                    tool_info[tool_name] = {
                        "required": strategy.get_required_params(),
                        "accepted": strategy.get_accepted_params(),
                        "skipped": strategy.get_skipped_params(),
                        "aliases": strategy.get_parameter_aliases(),
                    }
                except Exception:
                    # Skip tools that fail to initialize
                    continue
        except Exception:
            # Return empty dict if anything fails
            tool_info = {}

        cls._cached_tool_info = tool_info
        return tool_info

    def get_accepted_params(self, tool: str) -> set:
        """Return all accepted parameters for specified tool using strategy pattern.

        Parameters
        ----------
        tool : str
            Tool name ("DLC", "SLEAP", "ndx-pose")

        Returns
        -------
        set
            Set of accepted parameter names
        """
        strategy = ToolStrategyFactory.create_strategy(tool)
        return strategy.get_accepted_params()

    def _append_aliases(self, tool: str, params: dict) -> dict:
        """Append parameter aliases using strategy pattern.

        Parameters
        ----------
        tool : str
            Tool name
        params : dict
            Original parameters

        Returns
        -------
        dict
            Parameters with aliases added
        """
        strategy = ToolStrategyFactory.create_strategy(tool)
        return strategy.append_aliases(params)

    def insert1(self, key, **kwargs):
        """Insert model parameters with auto-generated name.

        1. Check if key is a dictionary
        2. Check if tool is supported
        3. Check if all required parameters are present, remove skipped
        4. Check if params already exist
        5. Always use default_pk_name for consistent naming
        """
        if not isinstance(key, dict):
            raise TypeError("Key must be a dictionary")

        this_tool = key.get("tool", "UNSPECIFIED").strip()
        try:
            strategy = ToolStrategyFactory.create_strategy(this_tool)
        except ValueError as e:
            raise ValueError(f"Tool not supported: {this_tool}") from e

        # Remove any skipped params (strategy pattern doesn't use skipped params)
        params = key["params"].copy()

        # Filter out skipped parameters using tool_info
        tool_info = self.tool_info()
        if this_tool in tool_info:
            skipped = tool_info[this_tool]["skipped"]
            params = {k: v for k, v in params.items() if k not in skipped}

        # Append aliases using strategy
        params = self._append_aliases(this_tool, params)

        # Validate required parameters using strategy
        strategy.validate_params(params)

        # Check for existing entry with same params hash
        params_hash_dict = dict(params_hash=dj.hash.key_hash(params))
        if dupe := (self & params_hash_dict & dict(tool=key["tool"])):
            dupe_key = dupe.fetch1("KEY")
            self._warn_msg(f"Entry exists with same params: {dupe_key}")
            return dupe_key

        model_params_id: str = (key.get("model_params_id") or "").strip()
        if not model_params_id:
            # Always use default_pk_name for consistent naming
            model_params_id = default_pk_name(
                "mp", dict(tool=key["tool"], params=params)
            )
        key["model_params_id"] = model_params_id

        if "skeleton_id" in key and not (
            Skeleton() & dict(skeleton_id=key["skeleton_id"])
        ):
            raise dj.DataJointError(
                f"Skeleton ID not in the Skeleton table: {key['skeleton_id']}"
            )

        insert_kwargs = dict(kwargs)
        super().insert1(
            dict(key, params=params, **params_hash_dict), **insert_kwargs
        )

        return dict(model_params_id=key["model_params_id"], tool=key["tool"])


@schema
class ModelSelection(SpyglassMixin, dj.Manual):
    """Represents a paring of model parameters and video group for training."""

    definition = """
    -> ModelParams
    -> VidFileGroup
    model_selection_id: varchar(32)
    ---
    parent_id=NULL: varchar(32) # ID of parent model, if any
    """


@schema
class Model(SpyglassMixin, dj.Computed):
    """Information to uniquely identify a trained model.

    Note: DeepLabCut 'projects' contain multiple models, each trained with
    different parameters (e.g., different trainFraction, shuffle, etc).
    Each such model is represented by a row in ModelSelection.
    """

    definition = """
    model_id: varchar(32)
    ---
    -> ModelSelection
    -> [nullable] AnalysisNwbfile
    model_path         : varchar(255)
    evaluation=NULL    : json          # tool-specific evaluation metrics dict
    """

    key_source = ModelSelection  # one entry per selection, ensures unique id

    def make(self, key):
        """Train a new model based on ModelSelection entry.

        Performs the following:
        1. Fetches model parameters and video group information
        2. Creates training dataset (if needed)
        3. Trains the model using the specified tool (DLC, SLEAP, etc.)
        4. Stores model metadata in NWB file
        5. Inserts entry into Model table

        Parameters
        ----------
        key : dict
            Primary key from ModelSelection table containing:
            - model_params_id
            - tool
            - vid_group_id
            - parent_id (optional, for continued training)

        Raises
        ------
        NotImplementedError
            If the tool is not supported for training
        ValueError
            If required parameters or data are missing
        """
        self._info_msg(f"Training model for selection: {key}")

        # Fetch selection details
        sel_entry = (ModelSelection() & key).fetch1()
        params_key = {
            "model_params_id": sel_entry["model_params_id"],
            "tool": sel_entry["tool"],
        }
        params_entry = (ModelParams() & params_key).fetch1()
        tool = params_entry["tool"]
        params = params_entry["params"]
        skeleton_id = params_entry.get("skeleton_id")

        # Fetch video group
        vid_group_key = {"vid_group_id": sel_entry["vid_group_id"]}
        vid_group = (VidFileGroup() & vid_group_key).fetch1()

        self._info_msg(f"Training {tool} model with params: {params_key}")

        # Dispatch to tool-specific training method using strategy pattern
        strategy = ToolStrategyFactory.create_strategy(tool)

        # Early validation using supports_training for better error messages
        if not strategy.supports_training:
            raise NotImplementedError(
                f"Training not supported for {tool}. "
                f"Use a different tool that supports training."
            )

        try:
            model_result = strategy.train_model(
                key, params, skeleton_id, vid_group, sel_entry, self
            )
        except NotImplementedError as e:
            raise NotImplementedError(
                f"Training not implemented for {tool}: {e}"
            ) from e

        # Insert into Model table
        self.insert1(model_result)
        self._info_msg(f"Model training complete: {model_result['model_id']}")

    def _register_model_metadata(self, metadata: ModelMetadata) -> str:
        """Create NWB file with model metadata directly in project directory.

        Parameters
        ----------
        metadata : ModelMetadata
            Consolidated metadata for model registration

        Returns
        -------
        str
            NWB file name for direct reference (stored in project directory)
        """
        from pynwb import NWBFile

        nwb_file_name = f"{metadata.model_id}_model.nwb"

        # Store in analysis directory for AnalysisNwbfile compatibility

        analysis_dir = (
            AnalysisNwbfile()._analysis_dir
        )  # pylint: disable=protected-access
        nwb_path = Path(analysis_dir) / nwb_file_name

        # Create basic NWB file
        nwbfile = NWBFile(
            session_description=f"DLC model training: {metadata.model_id}",
            identifier=f"model_{datetime.now(timezone.utc):%Y%m%d%H%M%S}",
            session_start_time=datetime.now(timezone.utc),
        )

        # Store training metadata in scratch space
        # Store training metadata as JSON string (NWB-compatible)
        import json

        training_metadata = {
            "model_id": metadata.model_id,
            "tool": "DLC",
            "project_path": str(metadata.project_path),
            "config_path": str(metadata.config_path),
            "model_path": _to_stored_path(metadata.model_path),
            "shuffle": metadata.params.get("shuffle", 1),
            "trainingsetindex": metadata.params.get("trainingsetindex", 0),
            "iteration": metadata.latest_model["iteration"],
            "trainFraction": metadata.latest_model["trainFraction"],
            "snapshot": metadata.latest_model.get("snapshot", ""),
            "trained_date": metadata.latest_model["date_trained"].isoformat(),
            "parent_id": metadata.parent_id or "",
            "skeleton_id": metadata.skeleton_id,
        }

        # Convert to JSON string for NWB storage
        nwbfile.add_scratch(
            data=json.dumps(training_metadata),
            name="model_training_metadata",
            description="Training metadata for position estimation model",
        )

        # Write NWB file
        with NWBHDF5IO(str(nwb_path), mode="w") as nwb_io:
            nwb_io.write(nwbfile)

        self._info_msg(f"Model metadata saved to NWB: {nwb_path}")

        # Get any available NWB file as a dummy parent (foreign key requirement)
        available_parents = Nwbfile().fetch("nwb_file_name")
        if len(available_parents) == 0:
            raise ValueError(
                "No NWB files available to use as parent for AnalysisNwbfile"
            )

        dummy_parent = available_parents[0]
        self._info_msg(
            f"Using '{dummy_parent}' as dummy parent for analysis file"
        )

        # Register the already-created NWB file in AnalysisNwbfile
        try:
            AnalysisNwbfile().add(dummy_parent, nwb_file_name)
            self._info_msg(f"Registered analysis file: {nwb_file_name}")
        except Exception as e:
            # File may already be registered, check if it exists
            if (
                len(AnalysisNwbfile() & {"analysis_file_name": nwb_file_name})
                > 0
            ):
                self._info_msg(
                    f"Analysis file already registered: {nwb_file_name}"
                )
            else:
                raise e

        return nwb_file_name

    def _make_dlc_model(
        self,
        key: dict,
        params: dict,
        skeleton_id: str,
        vid_group: dict,
        sel_entry: dict,
    ) -> dict:
        """Legacy DLC training method - now delegates to strategy pattern.

        This method is deprecated and maintained for backwards compatibility.
        New code should use the strategy pattern via make() method.
        """
        from spyglass.common.common_usage import ActivityLog
        from spyglass.position.utils.tool_strategies import DLCStrategy

        ActivityLog().deprecate_log(
            "_make_dlc_model",
            alt="Model.make() with tool='DLC'",
        )

        strategy = DLCStrategy()
        return strategy.train_model(
            key, params, skeleton_id, vid_group, sel_entry, self
        )

    def train(
        self,
        model_key: dict,
        maxiters: Union[int, None] = None,
        **kwargs,
    ) -> dict:
        """Continue training an existing model or train with new parameters.

        This method creates a new ModelSelection entry with parent_id pointing
        to the original model, then triggers populate() to train the new model.

        Parameters
        ----------
        model_key : dict
            Primary key for existing Model entry (must include 'model_id')
        maxiters : Union[int, None], optional
            Additional training iterations. If None, uses default from params.
        **kwargs
            Additional parameters to override in ModelParams. Can include:
            - shuffle : int, new shuffle index
            - trainingsetindex : int, new training set fraction
            - displayiters : int, display frequency
            - saveiters : int, save frequency

        Returns
        -------
        dict
            Primary key for the new Model entry

        Raises
        ------
        ValueError
            If model_key doesn't exist in Model table

        Examples
        --------
        >>> # Continue training with 50k more iterations
        >>> model_key = {"model_id": "my_dlc_model"}
        >>> new_model_key = model.train(model_key, maxiters=50000)
        >>>
        >>> # Train new shuffle from same model
        >>> new_model_key = model.train(model_key, shuffle=2)
        """
        # Validate model exists
        if not (self & model_key):
            raise ValueError(
                f"Model not found in database: {model_key}. "
                "Cannot continue training from non-existent model."
            )

        # Fetch existing model info
        model_entry = (self & model_key).fetch1()
        old_sel_key = {
            "model_params_id": model_entry["model_params_id"],
            "tool": model_entry["tool"],
            "vid_group_id": model_entry["vid_group_id"],
        }

        self._info_msg(
            f"Creating new training session from model: {model_key['model_id']}"
        )

        # Fetch original params
        params_entry = (ModelParams() & old_sel_key).fetch1()
        old_params = params_entry["params"].copy()

        # Update params with new values
        if maxiters is not None:
            old_params["maxiters"] = maxiters

        for k, v in kwargs.items():
            if k in ModelParams().get_accepted_params(params_entry["tool"]):
                old_params[k] = v
            else:
                self._warn_msg(f"Ignoring unknown parameter: {k}")

        # Create new ModelParams entry if params changed
        if old_params != params_entry["params"]:
            new_params_key = ModelParams().insert1(
                dict(
                    tool=params_entry["tool"],
                    params=old_params,
                    skeleton_id=params_entry.get("skeleton_id"),
                ),
                skip_duplicates=True,
            )
            if not new_params_key:
                # Params already exist, fetch the key
                params_hash = dj.hash.key_hash(old_params)
                new_params_key = (
                    ModelParams()
                    & {
                        "tool": params_entry["tool"],
                        "params_hash": params_hash,
                    }
                ).fetch1("KEY")
        else:
            new_params_key = {
                "model_params_id": params_entry["model_params_id"],
                "tool": params_entry["tool"],
            }

        # Create new ModelSelection with parent_id
        new_sel_key = dict(
            new_params_key,
            vid_group_id=old_sel_key["vid_group_id"],
            parent_id=model_key["model_id"],
        )

        # Generate model_selection_id if not provided
        if "model_selection_id" not in new_sel_key:
            new_sel_key["model_selection_id"] = default_pk_name(
                "ms-train", {"parent_id": model_key["model_id"]}
            )

        self._info_msg(
            f"Inserting new ModelSelection with parent_id: {model_key['model_id']}"
        )
        ModelSelection().insert1(new_sel_key, skip_duplicates=True)

        # Populate to trigger make()
        self._info_msg("Triggering model training...")
        self.populate(new_sel_key)

        # Fetch and return new model key
        new_model = (self & new_sel_key).fetch1("KEY")
        self._info_msg(f"New model trained: {new_model}")

        return new_model

    def evaluate(
        self,
        model_key: dict,
        plotting: bool = False,
        show_errors: bool = True,
        **kwargs,
    ) -> dict:
        """Evaluate a trained model on test dataset.

        Runs DLC's evaluate_network to compute test/train errors and optionally
        create labeled images. Results are stored in the evaluation-results
        directory within the DLC project.

        Parameters
        ----------
        model_key : dict
            Primary key for Model entry (must include 'model_id')
        plotting : bool, optional
            Whether to create labeled comparison images, by default False
        show_errors : bool, optional
            Whether to display errors in logger, by default True
        **kwargs
            Additional parameters for evaluate_network. Can include:
            - Shuffles : list, which shuffles to evaluate (default: [1])
            - trainingsetindex : int, training set fraction index
            - comparisonbodyparts : list, specific bodyparts to evaluate

        Returns
        -------
        dict
            Evaluation results with keys:
            - train_error : float, mean training error in pixels
            - test_error : float, mean test error in pixels
            - train_error_p : float, training error with p-cutoff
            - test_error_p : float, test error with p-cutoff
            - p_cutoff : float, p-value cutoff used
            - results_path : str, path to results CSV file

        Raises
        ------
        ValueError
            If model_key doesn't exist in Model table
        ImportError
            If DLC is not installed or evaluate_network not available
        FileNotFoundError
            If model or config file not found

        Examples
        --------
        >>> # Evaluate model
        >>> results = model.evaluate({"model_id": "my_dlc_model"})
        >>> print(f"Test error: {results['test_error']:.2f} px")
        >>>
        >>> # Evaluate with labeled images
        >>> results = model.evaluate(
        ...     {"model_id": "my_dlc_model"},
        ...     plotting=True
        ... )
        """
        # Validate model exists FIRST (before checking DLC import)
        if not (self & model_key):
            raise ValueError(
                f"Model not found in database: {model_key}. "
                "Cannot evaluate non-existent model."
            )

        if evaluate_network is None:
            raise ImportError(
                "DeepLabCut evaluate_network is required for evaluation. "
                "Install with: pip install deeplabcut>=3.0"
            )

        # Fetch model info
        model_entry = (self & model_key).fetch1()
        sel_key = {
            "model_params_id": model_entry["model_params_id"],
            "tool": model_entry["tool"],
            "vid_group_id": model_entry["vid_group_id"],
        }

        params_entry = (ModelParams() & sel_key).fetch1()
        tool = params_entry["tool"]

        self._info_msg(f"Evaluating model: {model_key['model_id']}")

        # Use strategy pattern for tool-specific evaluation
        strategy = ToolStrategyFactory.create_strategy(tool)

        # Early validation using supports_training for better error messages
        if not strategy.supports_training:
            self._warn_msg(
                f"Evaluation not supported for {tool}. "
                "Evaluation is only available for tools that support training."
            )
            return None

        results = strategy.evaluate_model(
            model_entry, params_entry, self, plotting, show_errors, **kwargs
        )

        if results:
            (self & model_key).update1(
                {"model_id": model_key["model_id"], "evaluation": results}
            )

        return results

    def _evaluate_dlc_model(
        self,
        _model_entry: dict,
        params_entry: dict,
        plotting: bool,
        show_errors: bool,
        **kwargs,
    ) -> dict:
        """Evaluate a DLC model.

        Parameters
        ----------
        _model_entry : dict
            Model table entry (unused; kept for API consistency)
        params_entry : dict
            ModelParams entry
        plotting : bool
            Create labeled images
        show_errors : bool
            Display errors
        **kwargs
            Additional DLC evaluation parameters

        Returns
        -------
        dict
            Evaluation results
        """
        params = params_entry["params"]

        # Get project path
        if "project_path" not in params:
            raise ValueError(
                "DLC evaluation requires 'project_path' in ModelParams"
            )

        project_path = Path(params["project_path"])
        config_path = project_path / "config.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"DLC config not found: {config_path}")

        # Get evaluation parameters
        shuffle = kwargs.get("Shuffles", [params.get("shuffle", 1)])
        if not isinstance(shuffle, list):
            shuffle = [shuffle]

        trainingsetindex = kwargs.get(
            "trainingsetindex", params.get("trainingsetindex", 0)
        )

        self._info_msg(f"Running DLC evaluation on shuffles: {shuffle}")

        # Run DLC evaluation
        try:
            with suppress_print_from_package():
                evaluate_network(
                    str(config_path),
                    Shuffles=shuffle,
                    trainingsetindex=trainingsetindex,
                    plotting=plotting,
                    show_errors=show_errors,
                )
        except Exception as e:
            self._err_msg(f"DLC evaluation failed: {e}")
            raise

        self._info_msg("Evaluation completed successfully")

        # Parse evaluation results
        results = self._parse_dlc_evaluation_results(
            project_path, shuffle[0], trainingsetindex
        )

        if show_errors and results:
            self._info_msg(
                f"Train error: {results['train_error']:.2f} px, "
                f"Test error: {results['test_error']:.2f} px"
            )

        return results

    def _parse_dlc_evaluation_results(
        self, project_path: Path, shuffle: int, _trainingsetindex: int
    ) -> dict:
        """Parse DLC evaluation results from CSV file.

        Parameters
        ----------
        project_path : Path
            DLC project path
        shuffle : int
            Shuffle number
        trainingsetindex : int
            Training set index

        Returns
        -------
        dict
            Parsed evaluation results
        """
        # Find evaluation results
        # Pattern: evaluation-results/iteration-X/TASK-trainsetYshuffleZ/*-results.csv
        eval_dir = project_path / "evaluation-results"

        if not eval_dir.exists():
            self._warn_msg(f"No evaluation results found in {eval_dir}")
            return {}

        # Find most recent iteration
        iteration_dirs = sorted(eval_dir.glob("iteration-*"))
        if not iteration_dirs:
            self._warn_msg("No iteration directories found")
            return {}

        latest_iter = iteration_dirs[-1]

        # Find results CSV for this shuffle
        results_csvs = list(latest_iter.rglob("*-results.csv"))
        results_csvs = [
            f
            for f in results_csvs
            if f"shuffle{shuffle}" in str(f)
            and "CombinedEvaluation" not in str(f)
        ]

        if not results_csvs:
            self._warn_msg(f"No results CSV found for shuffle {shuffle}")
            return {}

        # Use most recent results file
        results_csv = max(results_csvs, key=lambda p: p.stat().st_mtime)
        self._logger.debug("Reading evaluation results: %s", results_csv)

        # Parse CSV
        df = pd.read_csv(results_csv)

        # Get last row (latest snapshot)
        if len(df) == 0:
            return {}

        last_row = df.iloc[-1]

        return {
            "train_error": float(last_row["Train error(px)"]),
            "test_error": float(last_row["Test error(px)"]),
            "train_error_p": float(last_row["Train error with p-cutoff"]),
            "test_error_p": float(last_row["Test error with p-cutoff"]),
            "p_cutoff": float(last_row["p-cutoff used"]),
            "training_iterations": int(last_row["Training iterations:"]),
            "shuffle": int(last_row["Shuffle number"]),
            "train_fraction": int(last_row["%Training dataset"]),
            "results_path": str(results_csv),
        }

    def get_training_history(
        self, model_key: dict
    ) -> Union["pd.DataFrame", None]:
        """Extract training history (loss curves) for a model.

        Reads CSV files from DLC training output with enhanced discovery.
        Supports multiple CSV patterns and combines data from multiple files.

        Parameters
        ----------
        model_key : dict
            Primary key for Model entry

        Returns
        -------
        pd.DataFrame or None
            DataFrame with columns: iteration, loss, learning_rate, source_file
            Returns None if training history not found

        Examples
        --------
        >>> history = model.get_training_history({"model_id": "my_dlc_model"})
        >>> if history is not None:
        ...     print(f"Final loss: {history['loss'].iloc[-1]:.4f}")
        """
        # Validate model exists
        if not (self & model_key):
            raise ValueError(f"Model not found: {model_key}")

        model_entry = (self & model_key).fetch1()
        model_path = resolve_model_path(str(model_entry["model_path"]))

        if model_path.name != "config.yaml":
            return None

        csv_files = discover_training_csvs(model_path)
        if not csv_files:
            self._warn_msg(
                f"No training stats found near: {model_path}. "
                "Training may have been too short (displayiters > 0 required) or incomplete."
            )
            return None

        self._info_msg(f"Found {len(csv_files)} training CSV file(s)")

        parsed = [parse_training_csv(p) for p in csv_files]
        valid = [df for df in parsed if df is not None]
        if not valid:
            self._warn_msg(
                "No valid training data found in CSV files. "
                "Ensure displayiters > 0 and check training logs for errors."
            )
            return None

        combined = aggregate_training_stats(valid)

        if "loss" in combined.columns and len(combined) > 0:
            initial_loss = combined["loss"].iloc[0]
            final_loss = combined["loss"].iloc[-1]
            improvement = (
                ((initial_loss - final_loss) / initial_loss) * 100
                if initial_loss > 0
                else 0
            )
            self._info_msg(
                f"Training improvement: {improvement:.1f}%"
                f" ({initial_loss:.6f} → {final_loss:.6f})"
            )

        self._info_msg(
            f"Combined {len(combined)} training records from {len(valid)} file(s)"
        )
        return combined

    def plot_training_history(
        self, model_key: dict, save_path: Union[Path, str, None] = None
    ):
        """Plot training loss curves for a model with enhanced visualization.

        Displays training progress with enhanced visualization including
        loss curves, improvement statistics, optional learning rate plotting,
        and support for multiple data sources.

        Parameters
        ----------
        model_key : dict
            Primary key for Model entry
        save_path : Union[Path, str, None], optional
            Path to save plot. If None, displays plot, by default None

        Returns
        -------
        matplotlib.figure.Figure
            The figure object

        Examples
        --------
        >>> fig = model.plot_training_history({"model_id": "my_dlc_model"})
        >>> # Or save to file
        >>> model.plot_training_history(
        ...     {"model_id": "my_dlc_model"},
        ...     save_path="training_loss.png"
        ... )
        """
        import matplotlib.pyplot as plt

        # Get training history with enhanced discovery
        df = self.get_training_history(model_key)

        if df is None:
            raise ValueError("No training history found for this model")

        # Check available columns for flexible plotting
        has_iteration = "iteration" in df.columns
        has_loss = "loss" in df.columns
        has_lr = "learning_rate" in df.columns
        has_source = "source_file" in df.columns

        if not has_loss:
            raise ValueError("No loss data found to plot")

        # Setup plot with subplots if learning rate available
        n_plots = 2 if has_lr else 1
        fig, axes = plt.subplots(
            n_plots, 1, figsize=(10, 8 if n_plots == 2 else 6)
        )
        if n_plots == 1:
            axes = [axes]  # Make it consistent

        # X-axis: iteration if available, otherwise index
        x_data = df["iteration"] if has_iteration else range(len(df))
        x_label = "Training Iteration" if has_iteration else "Step"

        # Plot 1: Loss curve
        axes[0].plot(x_data, df["loss"], "b-", linewidth=2, alpha=0.8)
        axes[0].set_xlabel(x_label, fontsize=12)
        axes[0].set_ylabel("Loss", fontsize=12)
        axes[0].set_title(
            f"Training Loss Curve: {model_key['model_id']}",
            fontsize=14,
            fontweight="bold",
        )
        axes[0].grid(True, alpha=0.3)

        # Add improvement statistics to loss plot
        if len(df) > 1 and has_loss:
            initial_loss = df["loss"].iloc[0]
            final_loss = df["loss"].iloc[-1]
            min_loss = df["loss"].min()
            improvement = (
                ((initial_loss - final_loss) / initial_loss) * 100
                if initial_loss > 0
                else 0
            )

            # Add text box with statistics
            stats_text = (
                f"Initial: {initial_loss:.6f}\n"
                f"Final: {final_loss:.6f}\n"
                f"Best: {min_loss:.6f}\n"
                f"Improvement: {improvement:.1f}%"
            )
            axes[0].text(
                0.02,
                0.98,
                stats_text,
                transform=axes[0].transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
                fontsize=9,
            )

            # Add final loss annotation (original behavior)
            if has_iteration:
                final_iter = df["iteration"].iloc[-1]
                axes[0].annotate(
                    f"Final: {final_loss:.4f}",
                    xy=(final_iter, final_loss),
                    xytext=(10, 10),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.7),
                    arrowprops=dict(
                        arrowstyle="->", connectionstyle="arc3,rad=0"
                    ),
                )

        # Plot 2: Learning rate (if available)
        if has_lr and n_plots == 2:
            axes[1].plot(
                x_data, df["learning_rate"], "r-", linewidth=2, alpha=0.8
            )
            axes[1].set_xlabel(x_label, fontsize=12)
            axes[1].set_ylabel("Learning Rate", fontsize=12)
            axes[1].set_title(
                "Learning Rate Schedule", fontsize=14, fontweight="bold"
            )
            axes[1].grid(True, alpha=0.3)
            axes[1].set_yscale("log")  # Learning rate often better on log scale

        # Add source file information if available
        if has_source:
            unique_sources = df["source_file"].nunique()
            if unique_sources > 1:
                source_text = f"Data from {unique_sources} files: {', '.join(df['source_file'].unique())}"
                fig.suptitle(source_text, fontsize=10, y=0.02)

        plt.tight_layout()

        # Add some spacing if we have subtitle
        if has_source:
            plt.subplots_adjust(bottom=0.08)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            self._info_msg(f"Training plot saved to: {save_path}")
            plt.close(fig)
        else:
            plt.show()

        return fig

    def create_project(
        self,
        project_name: str,
        bodyparts: "List[str]",
        video_list: "List[dict]",
        frames_per_video: int = 20,
        project_directory: Union[str, None] = None,
        **kwargs,
    ) -> dict:
        """Create a new DLC project from videos in the Spyglass database.

        Creates the DLC project folder on disk, extracts frames for labeling,
        and inserts a ``Skeleton`` entry.  Does **not** insert into
        ``ModelSelection`` or trigger training — return here with
        ``Model.load(config_path)`` after labeling frames externally.

        .. note::
            There is no ``Project`` table in V2.  The DLC project folder and
            its ``config.yaml`` are the on-disk record.  ``config.yaml`` is
            stored in ``ModelParams`` only after training is complete via
            ``Model.load()``.

        Parameters
        ----------
        project_name : str
            Human-readable name for the DLC project (passed to
            ``deeplabcut.create_new_project`` as ``project``).
        bodyparts : list of str
            Body parts to track. Every entry must already exist in the
            ``BodyPart`` lookup table (admins can add new ones).
        video_list : list of dict
            Training videos.  Each element is a dict with at least
            ``nwb_file_name`` and ``epoch`` keys to look up an existing
            ``VideoFile`` entry.  Partial keys (without ``video_file_num``)
            expand to all camera angles for that epoch.
        frames_per_video : int, optional
            Number of frames to extract per video (``numframes2pick`` in
            DLC). Default 20.
        project_directory : str, optional
            Directory in which to create the project folder. Defaults to
            ``spyglass.settings.pose_project_dir``.
        **kwargs
            Passed through to ``deeplabcut.create_new_project()`` and/or
            ``deeplabcut.extract_frames()`` based on each function's
            signature.  For example, pass ``algo='uniform'`` to override
            the default ``'kmeans'`` frame-extraction algorithm.

        Returns
        -------
        dict
            ``{"config_path": str, "skeleton_id": str, "vid_group_id": str}``

        Raises
        ------
        ImportError
            If DeepLabCut is not installed.
        PermissionError
            If a bodypart in *bodyparts* does not yet exist in the
            ``BodyPart`` table.  Adding new body parts is restricted to
            admins; this error is raised by ``Skeleton().insert1()`` (which
            delegates to ``BodyPart.insert1()``), not by this method.
        FileNotFoundError
            If a video path cannot be resolved from ``VideoFile`` or does not
            exist on disk.
        ValueError
            If *video_list* is empty or no valid videos are found.

        Examples
        --------
        Train from videos already registered in Spyglass::

            project_info = Model().create_project(
                project_name="rat_head_tracking",
                bodyparts=["greenLED", "redLED_C"],
                video_list=[
                    {"nwb_file_name": "subject_20240101_.nwb", "epoch": 1},
                ],
            )
            # → label frames externally, then:
            model_key = Model().load(project_info["config_path"])

        """
        try:
            from deeplabcut import create_new_project, extract_frames
        except ImportError as e:
            raise ImportError(
                "DeepLabCut is required for Model.create_project(). "
                "Install with: pip install deeplabcut"
            ) from e

        from spyglass.position.utils import (
            get_param_names,
            sanitize_filename,
        )
        from spyglass.position.utils.dlc_io import read_yaml, save_yaml
        from spyglass.settings import pose_project_dir

        # Split kwargs by function signature so callers control both steps
        # without us needing to hard-code every parameter name.
        create_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in get_param_names(create_new_project)
        }
        extract_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in get_param_names(extract_frames)
        }
        extract_kwargs.setdefault("algo", "uniform")
        extract_kwargs.setdefault("userfeedback", False)

        # ── 1. Resolve video paths via VideoFile; build a VidFileGroup ────────
        # Each dict in video_list is a partial VideoFile key; get_abs_paths
        # expands multi-camera epochs automatically.
        resolved_videos = [
            path
            for item in video_list
            for path in VideoFile.get_abs_paths(item)
        ]

        if not resolved_videos:
            raise ValueError(
                "No valid training videos found. Check video_list entries."
            )

        vid_group_key = VidFileGroup.create_from_files(
            video_files=resolved_videos,
            description=f"Training videos: {project_name}",
        )

        # ── 2. Create the DLC project folder ─────────────────────────────────
        project_directory = str(
            project_directory
            or pose_project_dir
            or Path.home() / "dlc_projects"
        )

        config_path = Path(
            create_new_project(
                project=project_name,
                experimenter=sanitize_filename(project_name),
                videos=resolved_videos,
                working_directory=project_directory,
                copy_videos=True,
                multianimal=False,
                **create_kwargs,
            )
        )
        self._info_msg(f"DLC project created: {config_path}")

        # ── 3. Patch numframes2pick via dlc_io utilities ──────────────────────
        _, cfg = read_yaml(config_path.parent)
        cfg["numframes2pick"] = frames_per_video
        config_path = Path(
            save_yaml(config_path.parent, cfg, filename="config")
        )

        # ── 4. Extract frames ─────────────────────────────────────────────────
        algo = extract_kwargs["algo"]
        self._info_msg(
            f"Extracting {frames_per_video} frames/video (algo='{algo}')…"
        )
        extract_frames(str(config_path), **extract_kwargs)
        self._info_msg("Frame extraction complete.")

        # ── 5. Insert (or retrieve) Skeleton ──────────────────────────────────
        # BodyPart validation is deferred to Skeleton().insert1(): unknown
        # body parts cause PermissionError for non-admin users.
        sk_key = Skeleton().insert1(
            {"bodyparts": bodyparts, "edges": []},
            skip_duplicates=True,
        )

        self._info_msg(f"Skeleton: {sk_key['skeleton_id']}")

        return {
            "config_path": str(config_path),
            "skeleton_id": sk_key["skeleton_id"],
            "vid_group_id": vid_group_key["vid_group_id"],
        }

    def get_latest(self, config: dict) -> dict:
        """Get latest trained model information for the tool used by this entry.

        Fetches the tool name from the linked ``ModelParams`` row and delegates
        to the matching strategy's ``get_latest_model_info`` implementation.

        Parameters
        ----------
        config : dict
            Tool configuration dictionary containing at minimum
            ``project_path``.

        Returns
        -------
        dict
            Model metadata dict whose keys depend on the tool (e.g. ``path``,
            ``iteration``, ``trainFraction``, ``shuffle``, ``date_trained``
            for DLC).  Returns ``{}`` if no trained models exist or if the
            strategy has no discovery logic.

        Raises
        ------
        ValueError
            If ``self`` is not restricted to exactly one entry.
        FileNotFoundError
            If the ``project_path`` in *config* does not exist.
        """
        tool = (ModelParams & self).fetch1("tool")
        strategy = ToolStrategyFactory.create_strategy(tool)
        return strategy.get_latest_model_info(config)

    def load(
        self,
        model_path: Union[Path, str],
        tool: str = None,
        skeleton_id: Union[str, None] = None,
        model_params_id: Union[str, None] = None,
        model_id: Union[str, None] = None,
        model_name: Union[str, None] = None,
        **kwargs,
    ):
        """Import an existing trained model into the database.

        Parameters
        ----------
        model_path : Union[Path, str]
            Path to the model file or configuration (e.g., DLC .yml file or
            ndx-pose NWB file)
        tool : str, optional
            Tool used to train the model. If None, auto-detect from file type.
            Options: "DLC", "ndx-pose"
        skeleton_id : Union[str, None], optional
            Skeleton ID to associate with the model. If None, auto-generate
            default
        model_params_id : Union[str, None], optional
            Model parameters ID to associate with the model. If None,
            auto-generate
        model_id : Union[str, None], optional
            Model ID to assign. If None, auto-generate
        model_name : Union[str, None], optional
            For NWB files with multiple models, specify which model to import.
            Required if NWB contains multiple PoseEstimation objects.
        **kwargs
            Additional parameters for specific tools. Common parameters:
            - nwb_file_name: Parent NWB file name for linking

        Returns
        -------
        dict
            Primary key for the created Model entry

        Raises
        ------
        FileNotFoundError
            If the model path does not exist.
        ValueError
            If file type cannot be determined or multiple models found without
            model_name specified.
        NotImplementedError
            If the tool is not supported or import not implemented.
        """
        # Validate model path
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        kwargs.update(
            dict(
                kwargs,
                skeleton_id=skeleton_id,
                model_params_id=model_params_id,
                model_id=model_id,
                model_name=model_name,
            )
        )

        # Auto-detect tool from file extension if not provided
        tool_suffix_map = {"yml": "DLC", "yaml": "DLC"}
        suffix = model_path.suffix.lstrip(".").lower()
        if tool is None:
            tool = tool_suffix_map.get(suffix)
            if tool is None:
                if suffix == "nwb":
                    raise ValueError(
                        "ndx-pose NWB files should be ingested via "
                        "ImportedPose, not Model.load(). "
                        "Use:\n"
                        "  ImportedPose().insert_from_nwbfile(nwb_file_name)\n"
                        "Pass import_to_v2=True to also register skeleton and "
                        "model metadata in the V2 pipeline tables."
                    )
                raise ValueError(
                    f"Cannot auto-detect tool from file extension: {suffix}. "
                    "Please specify 'tool' parameter."
                )

        # Use strategy pattern for tool-specific imports
        strategy = ToolStrategyFactory.create_strategy(tool)
        return strategy.load(model_path, self, **kwargs)

    def import_from_v1(self, v1_key: dict, **kwargs) -> dict:
        """Import a V1 DLCModel into the V2 pipeline.

        Looks up the V1 model's ``project_path`` from the
        ``spyglass.position.v1`` schema and delegates to :meth:`load` with the
        resolved ``config.yaml`` path.

        Parameters
        ----------
        v1_key : dict
            Primary-key fields that uniquely identify a row in the V1
            ``DLCModel`` table, e.g.::

                {
                    "project_name": "Wtrack_WhiteLED",
                    "dlc_model_name": "Wtrack_WhiteLED_ms_stim_wtrack_00",
                    "dlc_model_params_name": "default",
                }

        **kwargs
            Forwarded to :meth:`load` (e.g. ``skeleton_id``,
            ``model_params_id``).

        Returns
        -------
        dict
            V2 ``Model`` primary key returned by :meth:`load`.

        Raises
        ------
        ImportError
            If the V1 position schema is not available in the current
            environment.
        FileNotFoundError
            If the V1 ``project_path`` does not contain a ``config.yaml``.
        KeyError
            If *v1_key* does not match any row in ``DLCModel``.
        """
        try:
            from spyglass.position.v1.position_dlc_model import DLCModel
        except Exception as exc:
            raise ImportError(
                "Could not import the V1 DLCModel table.  Make sure the "
                "spyglass.position.v1 schema is available and that the V1 "
                "tables have been created in your database."
            ) from exc

        row = (DLCModel & v1_key).fetch1()
        project_path = Path(row["project_path"])
        available = sorted(project_path.glob("*config.y*ml"))
        if not available:
            raise FileNotFoundError(
                f"No config.yaml found in V1 project_path: {project_path}"
            )
        # Prefer a DJ-managed config (prefixed with 'dj_dlc') when present
        dj_configs = [p for p in available if "dj_dlc" in p.name]
        config_path = dj_configs[0] if dj_configs else available[0]
        self._info_msg(
            f"Importing V1 model '{row['dlc_model_name']}' "
            f"from {config_path}"
        )
        return self.load(config_path, **kwargs)

    def _import_dlc_model(self, model_path: Path, **kwargs):
        if model_path.suffix not in [".yml", ".yaml"]:
            raise ValueError("DLC model path must be a .yml or .yaml file")

        config = load_yaml(model_path)

        # Step 1: Extract and insert skeleton from DLC config
        skeleton_config = {
            "bodyparts": config.get("bodyparts", []),
            "skeleton": config.get("skeleton", []),
        }
        skeleton_key = Skeleton().insert1(
            skeleton_config,
            check_duplicates=True,
            skip_duplicates=True,
        )
        self._info_msg(f"Skeleton: {skeleton_key['skeleton_id']}")

        # Step 2: Create or retrieve ModelParams entry
        model_params_key = ModelParams().insert1(
            dict(
                tool="DLC",
                params=config,
                model_params_id=kwargs.get("model_params_id"),
                skeleton_id=skeleton_key["skeleton_id"],
            ),
            skip_duplicates=True,
        )
        self._info_msg(f"ModelParams: {model_params_key['model_params_id']}")

        # Step 3: Create VidFileGroup linked to registered Spyglass session.
        # Raises ValueError if no session matches the DLC config's video paths.
        # Register the session with insert_sessions() before calling load().
        vid_group_key = VidFileGroup.create_from_dlc_config(model_path)
        self._info_msg(f"VidFileGroup: {vid_group_key['vid_group_id']}")

        # Step 4: Create ModelSelection entry (no skeleton_id — it lives in
        # ModelParams, not ModelSelection)
        sel_key = {**model_params_key, **vid_group_key}
        # Generate model_selection_id if not provided
        if "model_selection_id" not in sel_key:
            sel_key["model_selection_id"] = default_pk_name(
                "ms-dlc",
                {"model_params_id": model_params_key["model_params_id"]},
            )
        ModelSelection().insert1(sel_key, skip_duplicates=True)

        # Return the existing Model entry if one already exists for this path.
        # model_path is the most stable identifier; default_pk_name embeds
        # today's date so model_id changes across days.
        stored_path = _to_stored_path(model_path)
        if existing := (self & {"model_path": stored_path}):
            return existing.fetch1()

        # Step 5: Generate model_id and insert directly into Model
        task = config.get("Task", "DLCTask")
        date = config.get(
            "date", datetime.now(timezone.utc).strftime("%Y-%m-%d")
        )
        model_id = kwargs.get("model_id") or default_pk_name(
            f"DLC-{task}-{date}",
            dict(tool="DLC", model_path=stored_path),
        )
        model_key = {
            **sel_key,
            "model_id": model_id,
            "model_path": stored_path,
        }
        self.insert1(model_key, allow_direct_insert=True)
        self._info_msg(f"Model imported: {model_id}")

        return model_key

    def verify(
        self,
        model_key: dict,
        check_inference: bool = False,
    ) -> dict:
        """Verify a model is valid and accessible.

        Performs comprehensive validation checks on a model entry including:
        1. Model exists in database
        2. Model file/path exists on disk
        3. Skeleton is valid (has bodyparts and edges)
        4. ModelParams are valid
        5. (Optional) Can run basic inference test

        Parameters
        ----------
        model_key : dict
            Primary key for Model entry (must include 'model_id')
        check_inference : bool, optional
            Whether to test that inference can be initialized (doesn't actually
            run inference, just validates the model can be loaded),
            by default False

        Returns
        -------
        dict
            Verification results with keys:
            - valid : bool, overall validation status
            - checks : dict, individual check results (True/False)
            - errors : list, error messages for failed checks
            - warnings : list, warning messages for non-critical issues
            - model_info : dict, model metadata if validation passed

        Examples
        --------
        >>> # Basic verification
        >>> results = Model().verify({"model_id": "my_dlc_model"})
        >>> if results['valid']:
        ...     print("Model is valid!")
        >>> else:
        ...     print(f"Errors: {results['errors']}")
        >>>
        >>> # Full verification with inference check
        >>> results = Model().verify(
        ...     {"model_id": "my_dlc_model"},
        ...     check_inference=True
        ... )
        """
        errors = []
        warn_list = []
        checks = {
            "model_exists": False,
            "model_path_exists": False,
            "skeleton_valid": False,
            "params_valid": False,
            "inference_ready": False,
        }
        model_info = {}

        # Check 1: Model exists in database
        if not (self & model_key):
            errors.append(
                f"Model not found in database: {model_key}. "
                "Use Model.load() to import a model first."
            )
            return {
                "valid": False,
                "checks": checks,
                "errors": errors,
                "warnings": warn_list,
                "model_info": model_info,
            }
        checks["model_exists"] = True

        # Fetch model information
        try:
            model_entry = (self & model_key).fetch1()
            model_info = dict(model_entry)
        except Exception as e:
            errors.append(f"Error fetching model entry: {e}")
            return {
                "valid": False,
                "checks": checks,
                "errors": errors,
                "warnings": warn_list,
                "model_info": model_info,
            }

        # Check 2: Model path exists
        model_path = Path(model_entry["model_path"])
        if not model_path.exists():
            errors.append(f"Model file not found: {model_path}")
        else:
            checks["model_path_exists"] = True

            # For DLC models, verify it's a valid config file
            if model_entry["tool"] == "DLC":
                if model_path.suffix not in [".yaml", ".yml"]:
                    warn_list.append(
                        f"DLC model path should be config.yaml, got: {model_path.name}"
                    )
                else:
                    # Try to read config to verify it's valid YAML
                    try:
                        config = load_yaml(model_path)
                        if "project_path" not in config:
                            warn_list.append(
                                "DLC config missing 'project_path' field"
                            )
                    except Exception as e:
                        warn_list.append(f"Could not parse DLC config: {e}")

        # Check 3: Skeleton is valid
        try:
            params_key = {
                "model_params_id": model_entry["model_params_id"],
                "tool": model_entry["tool"],
            }
            params_entry = (ModelParams() & params_key).fetch1()
            skeleton_id = params_entry["skeleton_id"]

            if skeleton_id and (Skeleton() & {"skeleton_id": skeleton_id}):
                skeleton_tbl = Skeleton() & {"skeleton_id": skeleton_id}
                skeleton_entry = skeleton_tbl.fetch1()

                bodyparts = skeleton_tbl.get_bodyparts(skeleton_id)
                edges = skeleton_entry["edges"]

                if not bodyparts or len(bodyparts) == 0:
                    errors.append("Skeleton has no bodyparts")
                elif len(bodyparts) < 2:
                    warn_list.append(
                        f"Skeleton has only {len(bodyparts)} bodypart(s)"
                    )
                else:
                    checks["skeleton_valid"] = True

                if edges is None or len(edges) == 0:
                    warn_list.append("Skeleton has no edges defined")
            else:
                errors.append(f"Skeleton not found: {skeleton_id}")

        except Exception as e:
            errors.append(f"Error validating skeleton: {e}")

        # Check 4: ModelParams are valid
        try:
            # Verify required params exist for the tool
            tool = model_entry["tool"]
            params = params_entry["params"]

            if tool == "DLC":
                # Check if this is an NDX-pose import (has nwb_file param)
                if "nwb_file" in params:
                    # NDX-pose imports don't need traditional DLC config params
                    checks["params_valid"] = True
                else:
                    # Traditional DLC model - check for required config params
                    required = ["net_type"]  # Minimal requirement
                    missing = [p for p in required if p not in params]
                    if missing:
                        errors.append(f"Missing required DLC params: {missing}")
                    else:
                        checks["params_valid"] = True
            else:
                # For non-DLC tools, just check params exist
                if params and isinstance(params, dict):
                    checks["params_valid"] = True
                else:
                    errors.append(f"Invalid params for tool {tool}: {params}")

        except Exception as e:
            errors.append(f"Error validating ModelParams: {e}")

        # Check 5: (Optional) Inference readiness using strategy pattern
        if check_inference and checks["model_path_exists"]:
            tool = model_entry["tool"]
            try:
                strategy = ToolStrategyFactory.create_strategy(tool)

                # Only run verification if the tool supports training
                if strategy.supports_training:
                    tool_checks, tool_warnings = strategy.verify_model(
                        model_path, check_inference=True
                    )
                    checks.update(tool_checks)
                    warn_list.extend(tool_warnings)
                else:
                    warn_list.append(
                        f"Model verification not supported for tool: {tool} "
                        "(tool does not support training)"
                    )
            except ValueError:
                warn_list.append(
                    f"Inference check not supported for tool: {tool}"
                )

        # Overall validation status
        valid = (
            checks["model_exists"]
            and checks["model_path_exists"]
            and checks["skeleton_valid"]
            and checks["params_valid"]
            and len(errors) == 0
        )

        result = {
            "valid": valid,
            "checks": checks,
            "errors": errors,
            "warnings": warn_list,
            "model_info": model_info,
        }

        # Log results
        if valid:
            self._info_msg(f"Model verification PASSED: {model_key}")
            if warn_list:
                self._warn_msg(f"Warnings: {warn_list}")
        else:
            self._err_msg(f"Model verification FAILED: {model_key}")
            self._err_msg(f"Errors: {errors}")

        return result

    def run_inference(
        self,
        model_key: dict,
        video_path: Union[Path, str, list],
        save_as_csv: bool = False,
        destfolder: Union[Path, str, None] = None,
        **kwargs,
    ):
        """Validate inputs and delegate to PoseEstim.run_inference().

        Parameters
        ----------
        model_key : dict
            Primary key for the Model entry (must include 'model_id').
        video_path : Union[Path, str, list]
            Path(s) to video file(s) for inference.
        save_as_csv : bool, optional
            Whether to save output as CSV in addition to h5, by default False.
        destfolder : Union[Path, str, None], optional
            Destination folder for output files. If None, saves alongside
            video, by default None.
        **kwargs
            Additional parameters passed to the underlying inference function.

        Raises
        ------
        ValueError
            If model_key is not in the database, or if the stored model_path is
            not a DLC config.yaml file (e.g. when imported from an NWB).
        FileNotFoundError
            If video_path does not exist on disk.
        """
        if not (self & model_key):
            raise ValueError(f"Model not found in database: {model_key}")

        video_paths = (
            video_path
            if isinstance(video_path, (list, tuple))
            else [video_path]
        )
        for vp in video_paths:
            if not Path(vp).exists():
                raise FileNotFoundError(f"Video not found: {vp}")

        model_info = (self & model_key).fetch1()
        resolved = resolve_model_path(model_info["model_path"])
        if resolved.suffix not in (".yaml", ".yml"):
            raise ValueError(
                f"DLC inference requires a config.yaml file; "
                f"got: {resolved}. "
                "Re-import the model from its DLC project directory."
            )

        from spyglass.position.v2.estim import PoseEstim

        return PoseEstim().run_inference(
            model_key, video_path, save_as_csv, destfolder, **kwargs
        )

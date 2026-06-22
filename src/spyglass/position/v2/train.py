"""DataJoint tables for training and managing pose estimation models.

NOTE: This approach departs from position v1 by choosing not to store
information already on disk. v1 maintained copies of files in the event that
a user or DLC modified them so that the database always reflected the 'ground
truth'. In practice, file modification is rare, and storing copies is
inefficient.
"""

import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
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
from spyglass.position.v2.utils.skeleton import (
    build_canonical_map,
    build_labeled_graph,
    canonicalize,
    is_duplicate_skeleton,
    norm_edges,
    normalize_label,
    shape_hash_from_edges,
    validate_skeleton_graph,
)
from spyglass.position.v2.utils.training_io import (
    aggregate_training_stats,
    discover_training_csvs,
    parse_training_csv,
)
from spyglass.position.v2.video import VidFileGroup
from spyglass.utils import SpyglassMixin


def prompt_default(key: str, default, abort_value: str = "n") -> str:
    """Prompt the user for a value, returning the default on empty input.

    Parameters
    ----------
    key : str
        Label shown in the prompt.
    default : any
        Default value returned when the user presses Enter without input.
    abort_value : str, optional
        Input that triggers a RuntimeError ("Aborted by user"), by default "n".

    Returns
    -------
    str
        The user's response, or *default* cast to str when input is blank.

    Raises
    ------
    RuntimeError
        If the user enters *abort_value*.
    """
    response = input(f"{key} [{default}]: ").strip()
    if response == abort_value:
        raise RuntimeError("Aborted by user")
    return response if response else default


# ------------------------------ Optional imports ------------------------------
with suppress_print_from_package():
    try:
        # Only import evaluation function for remaining Model.evaluate()
        from deeplabcut import evaluate_network
    except (ImportError, RecursionError):
        # RecursionError can occur when TF 2.16+ (Keras 3) is installed without
        # tf-keras; the TF compat layer loader recurses infinitely. Set
        # TF_USE_LEGACY_KERAS=1 and install tf-keras to restore TF-backend
        evaluate_network = None

# -------------------------------- Module setup --------------------------------
warnings.filterwarnings("ignore", category=UserWarning, module="networkx")

schema = dj.schema("cbroz_position_v3_train")  # TODO: remove cbroz prefix


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
        """Insert body part into the database.

        A concept may have only one canonical spelling. A new surface form that
        normalizes to an existing entry (e.g. ``greenLed`` vs ``greenLED``) is
        rejected rather than inserted, so the table stays an unambiguous source
        of canonical names.
        """
        bodypart = key["bodypart"]
        if self & {"bodypart": bodypart}:
            if warn:
                self._warn_msg(f"Body part {bodypart} already exists")
            return
        # Reject a different spelling of an existing concept. Scan only for a
        # clash with this name so unrelated legacy duplicates don't block it.
        norm = normalize_label(bodypart)
        collision = next(
            (
                bp
                for bp in self.fetch("bodypart")
                if bp != bodypart and normalize_label(bp) == norm
            ),
            None,
        )
        if collision is not None:
            raise dj.DataJointError(
                f"Body part {bodypart!r} collides with existing "
                f"{collision!r} (both normalize to {norm!r}). Use the existing "
                "spelling."
            )
        if not LabMember().user_is_admin:
            raise PermissionError("Only admins can insert body parts")
        super().insert1(key, **kwargs)


@schema
class Skeleton(SpyglassMixin, dj.Lookup):
    """Defines skeletons as body parts and their connections (edges)."""

    definition = """
    skeleton_id: varchar(32)
    ---
    bodyparts=NULL: blob   # list of body part names, List[str]
    edges=NULL: blob       # list of edge pairs, List[Tuple[str, str]]
    hash=NULL : varchar(32) # hash of graph for duplicate detection
    """

    class BodyPart(dj.Part):
        """Body parts belonging to a skeleton."""

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

    # Bodyparts for default skeletons
    _default_bodyparts = {
        "1LED": ["whiteLED"],
        "2LED": ["greenLED", "redLED_C"],
        "4LED": ["greenLED", "redLED_C", "redLED_L", "redLED_R"],
    }

    @classmethod
    def _populate_default_bodyparts(cls):
        """Populate BodyPart part table for default skeleton contents."""
        for skeleton_id, bodyparts in cls._default_bodyparts.items():
            if not cls & {"skeleton_id": skeleton_id}:
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
            Skeleton ID to fetch bodyparts for. If None, uses cls.restriction.

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

    # Thin wrappers so callers (including tests) can use instance methods.
    # Implementations live in spyglass.position.v2.utils.skeleton.
    def _normalize_label(self, s: str) -> str:
        return normalize_label(s)

    def _fuzzy_equal(self, a: str, b: str, threshold: float = 0.85) -> bool:
        from spyglass.position.v2.utils.skeleton import fuzzy_equal

        return fuzzy_equal(a, b, threshold)

    def _norm_edges(self, edges):
        return norm_edges(edges)

    def _shape_hash_from_edges(self, labels, edges):
        return shape_hash_from_edges(labels, edges)

    def _build_labeled_graph(self, labels, edges):
        return build_labeled_graph(labels, edges)

    def _validate_bodyparts(self, labels: "set[str]") -> bool:
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
        valid = {normalize_label(x) for x in all_parts}
        missing = {normalize_label(x) for x in labels} - valid
        if missing:
            raise dj.DataJointError(
                f"Unknown bodypart name(s) (not in BodyPart): {sorted(missing)}"
                + "\nPlease either change to existing name or "
                + "consult with admin to add it."
            )
        return True

    # ----------------- main insert -----------------

    def insert1(
        self,
        key: dict,
        name_similarity: float = 0.85,
        check_duplicates: bool = True,
        accept_default: bool = False,
        accept_new_bodyparts: bool = False,
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
        accept_new_bodyparts : bool, optional
            When True, any bodypart names absent from the ``BodyPart`` reference
            table are inserted automatically with a warning rather than raising
            an error.  Intended for V1 model imports where project-specific
            body part names may not yet be in the curated list.  Default False.
        """
        bodyparts: List[str] = key.get("bodyparts", [])
        edges: List[Tuple[str, str]] = key.get("edges") or key.get(
            "skeleton", []
        )

        if not bodyparts or edges is None:
            raise dj.DataJointError(
                f"Key must include 'bodyparts' and 'edges' fields: {key}"
            )

        # Validate body parts against the reference table first so unknown-
        # bodypart errors surface before structural edge errors.
        labels_norm = [normalize_label(x) for x in bodyparts]
        if accept_new_bodyparts:
            all_parts = {
                normalize_label(x) for x in BodyPart().fetch("bodypart")
            }
            novel = sorted(
                {normalize_label(bp) for bp in bodyparts} - all_parts
            )
            if novel:
                self._warn_msg(
                    f"Auto-inserting {len(novel)} novel bodypart(s) into "
                    f"BodyPart: {novel}"
                )
                BodyPart().insert(
                    [{"bodypart": bp} for bp in novel],
                    skip_duplicates=True,
                    allow_direct_insert=True,
                )
        else:
            _ = self._validate_bodyparts(set(labels_norm))

        # Normalise edges: flatten nested groups, drop empty slots, normalize
        # labels.  norm_edges emits a warning when the input was non-standard.
        edges = norm_edges(edges) or []

        # Validate edge structure against the (now normalized) bodypart set.
        validate_skeleton_graph(bodyparts, edges)

        shape_hash = shape_hash_from_edges(bodyparts, edges)

        explicit_id = key.get("skeleton_id")
        if check_duplicates and not explicit_id:
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

        skeleton_id = key.get("skeleton_id") or default_pk_name("skel", key)
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
        this_graph = build_labeled_graph(bodyparts, edges)

        # Create figure with better sizing for zoomed-out view
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
        normalized_bodyparts = {normalize_label(bp) for bp in bodyparts}

        # Required bodyparts for 4-point Frank Lab method
        frank_lab_parts = {
            normalize_label("greenLED"),
            normalize_label("redLED_C"),
            normalize_label("redLED_L"),
            normalize_label("redLED_R"),
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
    """Model parameters for training and evaluation."""

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
        """Return all accepted parameters for specified tool using strategy

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

    def insert1(self, key, accept_default: bool = False, **kwargs):
        """Insert model parameters after validation and duplicate check.

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

        # Remove any skipped params
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
        if dupe := self & params_hash_dict & dict(tool=key["tool"]):
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
    """Represents a pairing of model parameters and video group for training."""

    definition = """
    -> ModelParams
    -> VidFileGroup
    model_selection_id: varchar(32)
    ---
    parent_id=NULL              : varchar(32)   # parent model for fine-tuning
    training_labels_path=NULL   : varchar(255)  # SLEAP .slp labels file; NULL for DLC
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

        lm = (
            metadata.latest_model
        )  # may be {} if get_latest_model_info missed dir
        training_metadata = {
            "model_id": metadata.model_id,
            "tool": "DLC",
            "project_path": str(metadata.project_path),
            "config_path": str(metadata.config_path),
            "model_path": _to_stored_path(metadata.model_path),
            "shuffle": metadata.params.get("shuffle", 1),
            "trainingsetindex": metadata.params.get("trainingsetindex", 0),
            "iteration": lm.get("iteration", 0),
            "trainFraction": lm.get("trainFraction", 0.0),
            "snapshot": lm.get("snapshot", ""),
            "trained_date": (
                lm["date_trained"].isoformat() if lm.get("date_trained") else ""
            ),
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
        if not self & model_key:
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

        self._info_msg(f"Inserting new ModelSelection: {model_key['model_id']}")
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
        if not self & model_key:
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
        # Pattern: evaluation-results/iteration-X/
        #          TASK-trainsetYshuffleZ/*-results.csv
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
        if not self & model_key:
            raise ValueError(f"Model not found: {model_key}")

        model_entry = (self & model_key).fetch1()
        model_path = resolve_model_path(str(model_entry["model_path"]))

        if model_path.name != "config.yaml":
            return None

        csv_files = discover_training_csvs(model_path)
        if not csv_files:
            self._warn_msg(
                f"No training stats found near: {model_path}. "
                "Training may have been too short (displayiters > 0 required)"
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
            f"Combined {len(combined)} training records from "
            + f"{len(valid)} file(s)"
        )
        return combined

    def plot_training_history(
        self,
        model_key: dict,
        save_path: Union[Path, str, None] = None,
        detailed: bool = False,
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
        detailed : bool, optional
            If True, renders additional diagnostics including a smoothed
            loss trend and optional validation loss panel, by default False

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

        # X-axis: iteration if available, otherwise index
        x_data = df["iteration"] if has_iteration else range(len(df))
        x_label = "Training Iteration" if has_iteration else "Step"

        fig, axes = self._setup_training_history_figure(
            plt=plt, detailed=detailed, has_lr=has_lr
        )

        self._plot_training_loss_panel(
            ax=axes[0],
            df=df,
            x_data=x_data,
            x_label=x_label,
            model_id=model_key["model_id"],
            has_iteration=has_iteration,
            detailed=detailed,
        )

        if detailed:
            self._plot_validation_panel(
                ax=axes[1],
                df=df,
                x_data=x_data,
            )
            if has_lr:
                self._plot_learning_rate_panel(
                    ax=axes[2],
                    x_data=x_data,
                    learning_rate=df["learning_rate"],
                    x_label=x_label,
                    title_fontsize=13,
                )
            else:
                axes[1].set_xlabel(x_label, fontsize=12)
        elif has_lr:
            self._plot_learning_rate_panel(
                ax=axes[1],
                x_data=x_data,
                learning_rate=df["learning_rate"],
                x_label=x_label,
                title_fontsize=14,
            )

        # Add source file information if available
        if has_source:
            self._add_training_source_summary(fig=fig, df=df)

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

    def _setup_training_history_figure(self, plt, detailed: bool, has_lr: bool):
        """Create figure and axes layout for training-history plots."""
        if detailed:
            n_plots = 3 if has_lr else 2
            fig, axes = plt.subplots(
                n_plots,
                1,
                figsize=(11, 10 if has_lr else 8),
                sharex=True,
            )
        else:
            n_plots = 2 if has_lr else 1
            fig, axes = plt.subplots(
                n_plots,
                1,
                figsize=(10, 8 if n_plots == 2 else 6),
            )

        if n_plots == 1:
            return fig, [axes]

        return fig, list(axes)

    def _plot_training_loss_panel(
        self,
        ax,
        df: pd.DataFrame,
        x_data,
        x_label: str,
        model_id: str,
        has_iteration: bool,
        detailed: bool,
    ):
        """Plot primary loss panel and overlay summary diagnostics."""
        ax.plot(x_data, df["loss"], "b-", linewidth=2, alpha=0.8)
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title(
            f"Training Loss Curve: {model_id}",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)

        if detailed and len(df) > 10:
            # Show trend independently of noisy step-to-step changes.
            window = max(10, len(df) // 20)
            smoothed_loss = df["loss"].rolling(window=window).mean()
            ax.plot(
                x_data,
                smoothed_loss,
                "g-",
                linewidth=2,
                alpha=0.8,
                label=f"Smoothed Loss (window={window})",
            )
            ax.legend(loc="upper right", fontsize=9)

        if len(df) > 1:
            self._annotate_training_loss_stats(
                ax=ax,
                df=df,
                has_iteration=has_iteration,
            )

    def _annotate_training_loss_stats(
        self,
        ax,
        df: pd.DataFrame,
        has_iteration: bool,
    ):
        """Add summary statistics and final-point annotation to loss panel."""
        initial_loss = df["loss"].iloc[0]
        final_loss = df["loss"].iloc[-1]
        min_loss = df["loss"].min()
        improvement = (
            ((initial_loss - final_loss) / initial_loss) * 100
            if initial_loss > 0
            else 0
        )

        stats_text = (
            f"Initial: {initial_loss:.6f}\n"
            f"Final: {final_loss:.6f}\n"
            f"Best: {min_loss:.6f}\n"
            f"Improvement: {improvement:.1f}%"
        )
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            fontsize=9,
        )

        if has_iteration:
            final_iter = df["iteration"].iloc[-1]
            ax.annotate(
                f"Final: {final_loss:.4f}",
                xy=(final_iter, final_loss),
                xytext=(10, 10),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.7),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
            )

    def _plot_validation_panel(self, ax, df: pd.DataFrame, x_data):
        """Plot validation loss panel or fallback message."""
        if "val_loss" in df.columns:
            ax.plot(x_data, df["val_loss"], "m-", linewidth=2, alpha=0.85)
        else:
            ax.text(
                0.5,
                0.5,
                "Validation data not available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        ax.set_ylabel("Validation Loss", fontsize=12)
        ax.set_title("Validation Loss", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)

    def _plot_learning_rate_panel(
        self,
        ax,
        x_data,
        learning_rate,
        x_label: str,
        title_fontsize: int,
    ):
        """Plot learning-rate schedule panel with log-scaled y-axis."""
        ax.plot(x_data, learning_rate, "r-", linewidth=2, alpha=0.8)
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel("Learning Rate", fontsize=12)
        ax.set_title(
            "Learning Rate Schedule",
            fontsize=title_fontsize,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

    def _add_training_source_summary(self, fig, df: pd.DataFrame):
        """Add data-source summary subtitle when multiple CSV sources exist."""
        unique_sources = df["source_file"].nunique()
        if unique_sources > 1:
            source_text = (
                f"Data from {unique_sources} files: "
                + f"{', '.join(df['source_file'].unique())}"
            )
            fig.suptitle(source_text, fontsize=10, y=0.02)

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
            ``spyglass.settings.pose_project_dir`` (set via ``pose_dirs.project``
            in ``dj_local_conf.json``).  Falls back to ``~/dlc_projects`` if
            that setting is not configured.
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
            If a video path cannot be resolved from ``VideoFile``, does not
            exist on disk, or is on an inaccessible network mount.
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

        if frames_per_video < 1:
            raise ValueError("frames_per_video must be >= 1")

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

        missing = [p for p in resolved_videos if not Path(p).exists()]
        if missing:
            raise FileNotFoundError(
                "The following training videos could not be found on disk "
                "(check that the relevant network mounts are accessible):\n"
                + "\n".join(f"  {p}" for p in missing)
            )

        # Try to infer a safe frame budget from the shortest video.  If this
        # fails (e.g., codec issues), we keep the user's requested value and
        # rely on DLC's own validation below.
        effective_frames_per_video = frames_per_video
        try:
            import cv2

            frame_counts = []
            for video_path in resolved_videos:
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    continue
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                if frame_count > 0:
                    frame_counts.append(frame_count)

            if frame_counts:
                shortest_video = min(frame_counts)
                if frames_per_video > shortest_video:
                    effective_frames_per_video = shortest_video
                    self._warn_msg(
                        "Requested "
                        + f"{frames_per_video} frames/video, but shortest "
                        + f"video has {shortest_video} frames. Using "
                        + f"{effective_frames_per_video}."
                    )
        except Exception:
            # Non-fatal; DLC may still be able to read and sample frames.
            pass

        vid_group_key = VidFileGroup.create_from_files(
            video_files=resolved_videos,
            description=f"Training videos: {project_name}",
        )

        # ── 2. Create or reuse the DLC project folder ─────────────────────────
        # Idempotency: search for an existing project under project_directory
        # whose config.yaml lists the same video basenames.  If found, skip
        # creation and frame extraction and return the existing config.
        project_directory = str(
            project_directory
            or pose_project_dir
            or Path.home() / "dlc_projects"
        )

        resolved_video_basenames = {Path(v).name for v in resolved_videos}
        config_path = None
        for candidate in sorted(
            Path(project_directory).glob(f"{project_name}-*-*/config.yaml")
        ):
            try:
                _, existing_cfg = read_yaml(candidate.parent)
            except Exception:
                continue
            existing_videos = {
                Path(v).name for v in existing_cfg.get("video_sets", {}).keys()
            }
            # Match if existing config's videos are a non-empty subset of
            # resolved_videos — DLC may silently drop some invalid files from
            # video_sets even though it copies them, so strict equality would
            # produce false negatives.
            if existing_videos and existing_videos.issubset(
                resolved_video_basenames
            ):
                config_path = candidate.resolve()
                self._info_msg(f"Reusing existing DLC project: {config_path}")
                break

        if config_path is None:
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
            ).resolve()
            self._info_msg(f"DLC project created: {config_path}")

        # ── 3. Patch numframes2pick via dlc_io utilities ──────────────────────
        _, cfg = read_yaml(config_path.parent)
        cfg["numframes2pick"] = effective_frames_per_video
        cfg["bodyparts"] = bodyparts
        config_path = Path(
            save_yaml(config_path.parent, cfg, filename="config")
        )

        # ── 4. Extract frames (skip if frames already exist) ──────────────────
        labeled_data_dir = config_path.parent / "labeled-data"
        already_extracted = (
            any(
                list(d.glob("img*.png"))
                for d in labeled_data_dir.iterdir()
                if d.is_dir()
            )
            if labeled_data_dir.exists()
            else False
        )

        if already_extracted:
            self._info_msg("Frames already extracted — skipping.")
        else:
            algo = extract_kwargs["algo"]
            self._info_msg(
                "Extracting "
                + f"{effective_frames_per_video} frames/video "
                + f"(algo='{algo}')…"
            )
            try:
                extract_frames(str(config_path), **extract_kwargs)
            except ValueError as exc:
                if "larger sample than population" in str(exc):
                    raise ValueError(
                        "DLC could not sample the requested number of frames "
                        + "from at least one training video. Try reducing "
                        + "frames_per_video (or use longer videos). "
                        + f"Requested: {effective_frames_per_video}."
                    ) from exc
                raise
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

    @staticmethod
    def _extract_bodypart_names(names) -> list:
        """Coerce supported inputs into a list of body-part names.

        Tool-agnostic: accepts an iterable of names, or a parsed config dict
        keyed by either the DLC ``"bodyparts"`` field or the ndx-pose / SLEAP
        ``"nodes"`` field. Reading a tool-specific model file from disk is the
        responsibility of that tool's import path, which parses it and passes
        the names (or parsed dict) here -- e.g. ``load_yaml(path)`` for DLC.

        Raises
        ------
        TypeError
            If given a bare ``str`` or :class:`~pathlib.Path`, which is
            ambiguous (a single name vs. a file to parse) and tool-specific.
        """
        if isinstance(names, (str, Path)):
            raise TypeError(
                "Pass body-part names as a list, or a parsed config dict; a "
                "bare string/path is ambiguous and tool-specific. For a DLC "
                "config, pass load_yaml(path)."
            )
        if isinstance(names, dict):
            names = names.get("bodyparts") or names.get("nodes") or []
        return list(names)

    def canonicalize_bodyparts(self, names) -> dict:
        """Resolve body-part surface forms to their canonical spellings.

        Uses the ``BodyPart`` table as the source of truth for canonical
        spelling. Names matching an existing part -- ignoring camelCase,
        snake_case, kebab-case, and whitespace differences -- map to that
        part's exact spelling; names with no match are returned as unresolved
        for the caller to decide what to do (e.g. error, or rewrite the DLC
        project with ``normalize_names=True``).

        Parameters
        ----------
        names : list[str] or dict
            Names to resolve. Accepts an iterable of names, or a parsed config
            dict keyed by either ``"bodyparts"`` (DLC) or ``"nodes"``
            (ndx-pose / SLEAP). A single name must be wrapped in a list.

        Returns
        -------
        dict
            ``{"mapping": {surface_form: canonical}, "unresolved": [...]}``.
            ``mapping`` holds only resolved names; ``unresolved`` preserves
            input order.
        """
        names = self._extract_bodypart_names(names)
        known = [str(bp) for bp in BodyPart().fetch("bodypart")]
        canon_map = build_canonical_map(known)
        mapping, unresolved = {}, []
        for name in names:
            canonical = canonicalize(name, canon_map)
            if canonical is None:
                unresolved.append(name)
            else:
                mapping[name] = canonical
        return {"mapping": mapping, "unresolved": unresolved}

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
            accept_new_bodyparts=False,  # TODO: discuss with team
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
        if existing := self & {"model_path": stored_path}:
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
        if not self & model_key:
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
                        "DLC model path should be config.yaml, got: "
                        + f"{model_path.name}"
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
        if not self & model_key:
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

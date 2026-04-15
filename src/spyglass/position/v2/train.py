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
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

if TYPE_CHECKING:
    import pandas as pd

import datajoint as dj
import networkx as nx
import numpy as np
import yaml
from datajoint.errors import DuplicateError
from pynwb import NWBHDF5IO

from spyglass.common import AnalysisNwbfile, LabMember, VideoFile
from spyglass.position.utils import get_most_recent_file, get_param_names
from spyglass.position.utils.tool_strategies import ToolStrategyFactory
from spyglass.position.utils_dlc import suppress_print_from_package
from spyglass.position.v2.video import VidFileGroup
from spyglass.settings import dlc_project_dir
from spyglass.utils import SpyglassMixin


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


def resolve_model_path(stored_path: str) -> Path:
    """Resolve a stored model_path to an absolute Path.

    Stored paths may be absolute or relative to ``dlc_project_dir``.
    Absolute paths are returned unchanged. Relative paths are resolved
    against ``dlc_project_dir`` if it is configured, otherwise against
    the current working directory.

    Parameters
    ----------
    stored_path : str
        Value stored in the Model.model_path column.

    Returns
    -------
    Path
        Absolute path to the model file.
    """
    p = Path(stored_path)
    if p.is_absolute():
        return p
    base = dlc_project_dir or Path.cwd()
    return Path(base) / p


def _to_stored_path(path: Path) -> str:
    """Convert an absolute model path to a stored (relative if possible) string.

    If ``path`` falls under ``dlc_project_dir``, the relative portion is
    returned so that entries remain portable across base-directory changes.
    Otherwise the absolute path is stored unchanged (same as V1 behaviour).

    Parameters
    ----------
    path : Path
        Absolute path to the model file.

    Returns
    -------
    str
        Path string to store in the database.
    """
    if dlc_project_dir:
        try:
            return str(path.relative_to(dlc_project_dir))
        except ValueError:
            pass
    return str(path)


# ------------------------------ Optional imports ------------------------------
try:
    import ndx_pose
except ImportError:
    ndx_pose = None

with suppress_print_from_package():
    try:
        from deeplabcut import create_training_dataset, train_network
        from deeplabcut.core.engine import Engine

        try:
            from deeplabcut import evaluate_network
        except (ImportError, RecursionError):
            evaluate_network = None
    except (ImportError, RecursionError):
        # RecursionError can occur when TF 2.16+ (Keras 3) is installed without
        # tf-keras; the TF compat layer loader recurses infinitely. Set
        # TF_USE_LEGACY_KERAS=1 and install tf-keras to restore TF-backend support.
        create_training_dataset, train_network, Engine = [None] * 3
        evaluate_network = None

# -------------------------------- Module setup --------------------------------
warnings.filterwarnings("ignore", category=UserWarning, module="networkx")

schema = dj.schema("cbroz_position_v2_train")  # TODO: remove cbroz prefix


# ----------------------------------- Helpers ----------------------------------
def default_pk_name(
    prefix: str,
    params: dict = None,
    limit: int = 32,
    include_hash: bool = True,
) -> str:
    """Generate a default name based on prefix and params.

    Format: PREFIX-YYYYMMDD-HASH where...
    - YYYYMMDD is the current date
    - HASH is a short hash of the params

    Parameters
    ----------
    prefix : str
        Prefix for the name, e.g. 'mp' for model params
    params : dict
        Dictionary of data to hash
    limit : int, optional
        Maximum length of the returned name, by default 32
    include_hash : bool, optional
        Whether to include the hash in the name (default: True). If False, only
        prefix and date. Relevant for tables that store a hash in another field.
    """
    # TODO: Migrate this to general utils for use with other params tables?

    when = datetime.utcnow()
    h = dj.hash.key_hash(params or dict()) if include_hash else ""
    return f"{prefix}-{when:%Y%m%d}-{h}"[:limit]


def prompt_default(primary_key: str, default: str) -> str:
    """Prompt user for input with a default value.

    Parameters
    ----------
    primary_key : str
        Prompt message
    default : str
        Default value if user presses enter

    Returns
    -------
    str
        User input or default
    """
    choice = input(
        f"Found no value for {primary_key}. Please provide one. "
        + f"['enter' for default: {default!r}, 'n' to abort]: "
    ).strip()
    if choice.lower() == "n":
        raise RuntimeError("Aborted by user.")
    return choice if choice else default


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

    def insert1(self, key, **kwargs):
        """Insert body part into the database."""
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

    def _validate_bodyparts(self, labels: set[str]):
        """Validate that all normalized labels exist in BodyPart table."""
        all_parts = BodyPart().fetch("bodypart")
        valid = {self._normalize_label(x) for x in all_parts}
        missing = set(self._normalize_label(x) for x in labels) - valid
        if missing:
            raise dj.DataJointError(
                f"Unknown bodypart name(s) (not in BodyPart): {sorted(missing)}"
                + "\nPlease either change to existing name or "
                + "consult with admin to add it."
            )

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

        G = nx.Graph()
        G.add_nodes_from(range(len(labels_norm)))
        G.add_edges_from(
            (
                idx_of[self._normalize_label(u)],
                idx_of[self._normalize_label(v)],
            )
            for u, v in edges
        )
        return nx.weisfeiler_lehman_graph_hash(
            G, node_attr=None, edge_attr=None
        )

    def _build_labeled_graph(
        self, labels: list[str], edges: list[tuple[str, str]]
    ) -> nx.Graph:
        """Build a graph with labeled nodes with normalized labels."""

        labels_norm = [self._normalize_label(x) for x in labels]
        if len(labels_norm) != len(set(labels_norm)):
            raise ValueError("Duplicate body part names after normalization.")

        G = nx.Graph()
        for label in labels_norm:
            G.add_node(label, label=label)

        if not edges:
            return G

        label_set = set(labels_norm)
        # Edge endpoints must belong to the declared label set
        for u, v in self._norm_edges(edges):
            if u not in label_set or v not in label_set:
                raise ValueError(
                    f"Edge uses label not in bodyparts: ({u!r}, {v!r})"
                )

        for u, v in edges:
            G.add_edge(self._normalize_label(u), self._normalize_label(v))

        return G

    # ----------------- main insert -----------------

    def insert1(
        self,
        key: dict,
        name_similarity: float = 0.85,
        accept_default: bool = False,
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
        accept_default : bool, optional
            Whether to accept default skeleton_id without prompting, by default
            False
        check_duplicates : bool, optional
            Whether to run duplicate detection, by default True
        """
        skeleton_id: str = key.get("skeleton_id", "").strip()
        bodyparts: List[str] = key.get("bodyparts", [])
        edges: List[Tuple[str, str]] = key.get("edges") or key.get(
            "skeleton", []
        )

        if skeleton_id in self.fetch("skeleton_id"):
            self._warn_msg(f"Skeleton ID {skeleton_id} already exists")
            return (self & dict(skeleton_id=skeleton_id)).fetch1("KEY")
        if not bodyparts or edges is None:
            raise dj.DataJointError(
                f"Key must include 'bodyparts' and 'edges' fields: {key}"
            )

        # Validate body parts against the reference table
        labels_norm = [self._normalize_label(x) for x in bodyparts]
        _ = self._validate_bodyparts(set(labels_norm))

        # If dupe error, compute labeled graph.
        # Skip topology-based dedup when the caller provides an explicit
        # skeleton_id — they want to create a named entry, not reuse an
        # existing topologically-equivalent one.
        shape_hash = self._shape_hash_from_edges(bodyparts, edges)
        G_new, checks = None, []
        if check_duplicates and not skeleton_id:
            G_new = self._build_labeled_graph(bodyparts, edges)
            checks = self & dict(hash=shape_hash)

        # Check for graph matches
        for row in checks:
            row_bodyparts = self.get_bodyparts(row["skeleton_id"])
            G_old = self._build_labeled_graph(row_bodyparts, row["edges"])

            def node_match(a, b):
                return self._fuzzy_equal(
                    a.get("label", ""), b.get("label", ""), name_similarity
                )

            GM = nx.algorithms.isomorphism.GraphMatcher(
                G_new, G_old, node_match=node_match
            )
            if GM.is_isomorphic():
                # Back-fill bodyparts blob if missing (schema migration)
                if row.get("bodyparts") is None:
                    super().update1(
                        dict(
                            skeleton_id=row["skeleton_id"], bodyparts=bodyparts
                        )
                    )
                return dict(skeleton_id=row["skeleton_id"])

        # Prompt user for skeleton id if not provided
        # Likely case for inserting via DLC config
        if not skeleton_id:
            skeleton_id = default_pk_name("skel", key)
            if not accept_default and not self._test_mode:
                skeleton_id = prompt_default("skeleton_id", skeleton_id)

        insert_pk = dict(skeleton_id=skeleton_id)
        super().insert1(
            dict(
                insert_pk,
                bodyparts=bodyparts,
                edges=edges,
                hash=shape_hash,
            ),
            **kwargs,
        )

        # Insert bodyparts into part table
        bodypart_entries = [
            {"skeleton_id": skeleton_id, "bodypart": bp} for bp in bodyparts
        ]
        self.BodyPart.insert(bodypart_entries, skip_duplicates=True)

        return insert_pk

    def get_centroid_method(self):
        raise NotImplementedError("Return func based graph")


@schema
class ModelParams(SpyglassMixin, dj.Lookup):
    definition = """
    model_params_id: varchar(32)
    tool: varchar(32)
    ---
    params: blob
    -> [nullable] Skeleton
    params_hash: varchar(64) # hash of params
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
    @property
    def tool_info(cls) -> dict:
        """Return tool information using strategy pattern for backward compatibility.

        Returns
        -------
        dict
            Tool information with structure:
            {tool_name: {"required": set, "accepted": set, "skipped": set, "aliases": dict}}
        """
        from ..utils.tool_strategies import ToolStrategyFactory

        tool_info = {}

        # Define skipped params per tool (params handled by Spyglass, not tool)
        # These are typically paths, project settings, and metadata that
        # Spyglass manages rather than passing to the tool
        skipped_params = {
            "DLC": {
                "video_sets",
                "model_path",
                "analysis_file_id",
            },
            "SLEAP": {
                "project_path",
                "video_sets",
                "model_path",
                "analysis_file_id",
            },
            "ndx-pose": {"model_path", "analysis_file_id", "nwb_file_path"},
        }

        # Build tool_info for each registered strategy
        for tool_name in ToolStrategyFactory._strategies:
            try:
                strategy = ToolStrategyFactory.create_strategy(tool_name)
                tool_info[tool_name] = {
                    "required": strategy.get_required_params(),
                    "accepted": strategy.get_accepted_params(),
                    "skipped": skipped_params.get(tool_name, set()),
                    "aliases": strategy.get_parameter_aliases(),
                }
            except Exception as e:
                # Skip tools that fail to initialize
                # Use warnings instead of self._warn_msg since this is a class method
                import warnings

                warnings.warn(
                    f"Could not initialize strategy for {tool_name}: {e}"
                )
                continue

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

    def insert1(self, key, accept_default=False, **kwargs):
        """Insert model parameters into the database.

        1. Check if key is a dictionary
        2. Check if tool is supported
        3. Check if all required parameters are present, remove skipped
        4. Check if params already exist
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
        tool_info = self.tool_info
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
            model_params_id = default_pk_name(
                "mp", dict(tool=key["tool"], params=params)
            )
            if not accept_default and not self._test_mode:
                model_params_id = prompt_default(
                    "model_params_id", model_params_id
                )
        key["model_params_id"] = model_params_id

        if "skeleton_id" in key and not (
            Skeleton() & dict(skeleton_id=key["skeleton_id"])
        ):
            raise dj.DataJointError(
                f"Skeleton ID not in the Skeleton table: {key['skeleton_id']}"
            )

        super().insert1(dict(key, params=params, **params_hash_dict), **kwargs)

        return dict(model_params_id=key["model_params_id"], tool=key["tool"])


@schema
class ModelSelection(SpyglassMixin, dj.Manual):
    definition = """
    -> ModelParams
    -> VidFileGroup
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
    model_path    : varchar(255)
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

        try:
            model_result = strategy.train_model(
                key, params, skeleton_id, vid_group, sel_entry, self
            )
        except NotImplementedError as e:
            raise NotImplementedError(
                f"Training not implemented for {tool}: {e}"
            )

        # Insert into Model table
        self.insert1(model_result)
        self._info_msg(f"Model training complete: {model_result['model_id']}")

    def _prepare_training_dataset(
        self, config_path: Path, params: dict, config: dict
    ) -> None:
        """Create DLC training dataset if needed.

        Parameters
        ----------
        config_path : Path
            Path to DLC config.yaml file
        params : dict
            Training parameters
        config : dict
            Loaded DLC config dictionary
        """
        shuffle = params.get("shuffle", 1)
        training_dataset_path = (
            config_path.parent
            / "training-datasets"
            / f"iteration-{config.get('iteration', 0)}"
        )

        if not training_dataset_path.exists():
            self._info_msg("Creating training dataset...")
            try:
                with suppress_print_from_package():
                    create_training_dataset(
                        str(config_path),
                        num_shuffles=shuffle,
                        Shuffles=[shuffle],
                        trainIndices=None,
                        testIndices=None,
                        net_type=params.get("net_type"),
                        augmenter_type=params.get(
                            "augmenter_type", config.get("default_augmenter")
                        ),
                    )
                self._info_msg("Training dataset created successfully")
            except Exception as e:
                self._warn_msg(
                    f"Training dataset creation failed or already exists: {e}"
                )
                # May already exist, continue

    def _execute_training(self, config_path: Path, params: dict) -> None:
        """Execute DLC network training.

        Parameters
        ----------
        config_path : Path
            Path to DLC config.yaml file
        params : dict
            Training parameters
        """
        train_params = {
            "config": str(config_path),
            "shuffle": params.get("shuffle", 1),
            "trainingsetindex": params.get("trainingsetindex", 0),
            "displayiters": params.get("displayiters"),
            "saveiters": params.get("saveiters"),
            "maxiters": params.get("maxiters"),
        }

        # Remove None values
        train_params = {k: v for k, v in train_params.items() if v is not None}

        self._info_msg("Starting model training...")
        self._info_msg(f"Training parameters: {train_params}")

        try:
            with suppress_print_from_package():
                train_network(**train_params)
        except Exception as e:
            self._err_msg(f"Training failed: {e}")
            raise

        self._info_msg("Training completed successfully")

    def _localize_trained_model(self, config: dict) -> tuple[str, str]:
        """Find and validate the trained model.

        Parameters
        ----------
        config : dict
            Loaded DLC config dictionary

        Returns
        -------
        tuple[str, str]
            Model path and generated model_id
        """
        latest_model = self._get_latest_dlc_model_info(config)
        if not latest_model:
            raise ValueError(
                "No trained model found after training. Check DLC output."
            )

        model_path = latest_model["path"]
        self._info_msg(f"Trained model path: {model_path}")

        # Generate model_id
        task = config.get("Task", "DLCTask")
        date = config.get("date", datetime.utcnow().strftime("%Y-%m-%d"))

        model_id = default_pk_name(
            f"DLC-{task}-{date}",
            dict(
                tool="DLC",
                shuffle=latest_model["shuffle"],
                iteration=latest_model["iteration"],
                trainFraction=latest_model["trainFraction"],
            ),
        )

        return model_path, model_id

    def _register_model_metadata(self, metadata: ModelMetadata) -> str:
        """Create NWB file with model metadata and register in AnalysisNwbfile.

        Parameters
        ----------
        metadata : ModelMetadata
            Consolidated metadata for model registration

        Returns
        -------
        str
            NWB file name for AnalysisNwbfile
        """
        from pynwb import NWBHDF5IO, NWBFile

        nwb_file_name = f"{metadata.model_id}_model.nwb"
        nwb_path = metadata.project_path / nwb_file_name

        # Create basic NWB file
        nwbfile = NWBFile(
            session_description=f"DLC model training: {metadata.model_id}",
            identifier=f"model_{datetime.utcnow():%Y%m%d%H%M%S}",
            session_start_time=datetime.utcnow(),
        )

        # Store training metadata in scratch space
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
            "snapshot": metadata.latest_model.get("snapshot"),
            "trained_date": metadata.latest_model["date_trained"].isoformat(),
            "parent_id": metadata.parent_id,
            "skeleton_id": metadata.skeleton_id,
        }

        nwbfile.add_scratch(training_metadata, name="model_training_metadata")

        # Write NWB file
        with NWBHDF5IO(str(nwb_path), mode="w") as nwb_io:
            nwb_io.write(nwbfile)

        self._info_msg(f"Model metadata saved to NWB: {nwb_path}")

        # Register NWB file in AnalysisNwbfile
        analysis_key = AnalysisNwbfile().create(nwb_file_name)
        if not analysis_key:
            # File may already be registered
            analysis_key = (
                AnalysisNwbfile() & {"analysis_file_name": nwb_file_name}
            ).fetch1("KEY")

        return nwb_file_name

    def _make_dlc_model(
        self,
        key: dict,
        params: dict,
        skeleton_id: str,
        vid_group: dict,
        sel_entry: dict,
    ) -> dict:
        """Train a DLC model using focused helper methods.

        Parameters
        ----------
        key : dict
            ModelSelection key
        params : dict
            Training parameters from ModelParams
        skeleton_id : str
            Skeleton ID for this model
        vid_group : dict
            VidFileGroup entry
        sel_entry : dict
            Full ModelSelection entry

        Returns
        -------
        dict
            Model table entry with model_id, model_path, analysis_file_name
        """
        if create_training_dataset is None or train_network is None:
            raise ImportError(
                "DeepLabCut is required for training. "
                "Install with: pip install deeplabcut>=3.0"
            )

        # Validate project configuration
        if "project_path" not in params:
            raise ValueError(
                "DLC training requires 'project_path' in ModelParams. "
                "Please specify the DLC project directory."
            )

        project_path = Path(params["project_path"])
        if not project_path.exists():
            raise FileNotFoundError(f"DLC project not found: {project_path}")

        config_path = project_path / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(
                f"DLC config.yaml not found in: {project_path}"
            )

        self._info_msg(f"Using DLC project: {project_path}")

        # Load config
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Check for continued training
        parent_id = sel_entry.get("parent_id")
        if parent_id:
            self._info_msg(
                f"Continuing training from parent model: {parent_id}"
            )

        # Execute training pipeline
        self._prepare_training_dataset(config_path, params, config)
        self._execute_training(config_path, params)

        # Localize and register trained model
        model_path, model_id = self._localize_trained_model(config)
        latest_model = self._get_latest_dlc_model_info(config)

        nwb_file_name = self._register_model_metadata(
            ModelMetadata(
                model_id=model_id,
                model_path=model_path,
                project_path=project_path,
                config_path=config_path,
                params=params,
                config=config,
                latest_model=latest_model,
                skeleton_id=skeleton_id,
                parent_id=parent_id,
            )
        )

        # Return Model entry
        return dict(
            key,
            model_id=model_id,
            analysis_file_name=nwb_file_name,
            model_path=_to_stored_path(model_path),
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

        # Dispatch to tool-specific evaluation
        if tool == "DLC":
            return self._evaluate_dlc_model(
                model_entry, params_entry, plotting, show_errors, **kwargs
            )
        else:
            raise NotImplementedError(
                f"Evaluation not implemented for tool: {tool}"
            )

    def _evaluate_dlc_model(
        self,
        model_entry: dict,
        params_entry: dict,
        plotting: bool,
        show_errors: bool,
        **kwargs,
    ) -> dict:
        """Evaluate a DLC model.

        Parameters
        ----------
        model_entry : dict
            Model table entry
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
        self, project_path: Path, shuffle: int, trainingsetindex: int
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
        self._logger.debug(f"Reading evaluation results: {results_csv}")

        # Parse CSV
        import pandas as pd

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

        Reads the learning_stats.csv file from DLC training output.

        # TODO: Generalize to other tools

        Parameters
        ----------
        model_key : dict
            Primary key for Model entry

        Returns
        -------
        pd.DataFrame or None
            DataFrame with columns: iteration, loss, learning_rate
            Returns None if training history not found

        Examples
        --------
        >>> history = model.get_training_history({"model_id": "my_dlc_model"})
        >>> if history is not None:
        ...     print(f"Final loss: {history['loss'].iloc[-1]:.4f}")
        """
        import pandas as pd

        # Validate model exists
        if not (self & model_key):
            raise ValueError(f"Model not found: {model_key}")

        # Get model path
        model_entry = (self & model_key).fetch1()
        model_path = Path(model_entry["model_path"])

        # model_path is the train/ directory (set by _make_dlc_model via
        # _get_latest_dlc_model_info which returns pose_cfg.parent).
        # learning_stats.csv lives in the same train/ directory.
        model_path = resolve_model_path(str(model_path))
        stats_path = model_path / "learning_stats.csv"

        if not stats_path.exists():
            self._warn_msg(f"Training stats not found: {stats_path}")
            return None

        # Read CSV (format: iteration, loss, learning_rate)
        df = pd.read_csv(
            stats_path,
            header=None,
            names=["iteration", "loss", "learning_rate"],
        )

        self._logger.debug(
            f"Loaded {len(df)} training iterations from {stats_path}"
        )
        return df

    def plot_training_history(
        self, model_key: dict, save_path: Union[Path, str, None] = None
    ):
        """Plot training loss curve for a model.

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

        history = self.get_training_history(model_key)

        if history is None:
            raise ValueError("No training history found for this model")

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(history["iteration"], history["loss"], linewidth=2)
        ax.set_xlabel("Training Iteration", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title(
            f"Training Loss Curve: {model_key['model_id']}",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)

        # Add final loss annotation
        final_loss = history["loss"].iloc[-1]
        final_iter = history["iteration"].iloc[-1]
        ax.annotate(
            f"Final: {final_loss:.4f}",
            xy=(final_iter, final_loss),
            xytext=(10, 10),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.7),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            self._info_msg(f"Training plot saved to: {save_path}")
            plt.close(fig)
        else:
            plt.show()

        return fig

    def import_model(
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
        tool_suffix_map = {"yml": "DLC", "yaml": "DLC", "nwb": "ndx-pose"}
        suffix = model_path.suffix.lstrip(".").lower()
        if tool is None:
            tool = tool_suffix_map.get(suffix)
            if tool is None:
                raise ValueError(
                    f"Cannot auto-detect tool from file extension: {suffix}. "
                    "Please specify 'tool' parameter."
                )

        # Dispatch to appropriate import method
        if tool == "DLC":
            return self._import_dlc_model(model_path, **kwargs)
        elif tool == "ndx-pose":
            return self._import_ndx_pose_model(model_path, **kwargs)
        else:
            raise NotImplementedError(
                f"Model import not implemented for {tool}"
            )

    def _import_dlc_model(self, model_path: Path, **kwargs):
        if model_path.suffix not in [".yml", ".yaml"]:
            raise ValueError("DLC model path must be a .yml or .yaml file")

        with open(model_path, "r") as f:
            config = yaml.safe_load(f)

        # Step 1: Extract and insert skeleton from DLC config
        skeleton_config = {
            "bodyparts": config.get("bodyparts", []),
            "skeleton": config.get("skeleton", []),
        }
        if kwargs.get("skeleton_id"):
            skeleton_config["skeleton_id"] = kwargs["skeleton_id"]

        skeleton_key = Skeleton().insert1(
            skeleton_config,
            check_duplicates=True,
            skip_duplicates=True,
            accept_default=True,
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
            accept_default=True,
        )
        self._info_msg(f"ModelParams: {model_params_key['model_params_id']}")

        # Step 3: Create VidFileGroup linked to registered Spyglass session.
        # Raises ValueError if no session matches the DLC config's video paths.
        # Register the session with insert_sessions() before calling import_model().
        vid_group_key = VidFileGroup.create_from_dlc_config(model_path)
        self._info_msg(f"VidFileGroup: {vid_group_key['vid_group_id']}")

        # Step 4: Create ModelSelection entry (no skeleton_id — it lives in
        # ModelParams, not ModelSelection)
        sel_key = {**model_params_key, **vid_group_key}
        ModelSelection().insert1(sel_key, skip_duplicates=True)

        # Return the existing Model entry if one already exists for this path.
        # model_path is the most stable identifier; default_pk_name embeds
        # today's date so model_id changes across days.
        stored_path = _to_stored_path(model_path)
        if existing := (self & {"model_path": stored_path}):
            return existing.fetch1()

        # Step 5: Generate model_id and insert directly into Model
        task = config.get("Task", "DLCTask")
        date = config.get("date", datetime.utcnow().strftime("%Y-%m-%d"))
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

    def _import_ndx_pose_model(self, model_path: Path, **kwargs):
        """Import a trained model from an ndx-pose NWB file.

        Parameters
        ----------
        model_path : Path
            Path to the NWB file containing ndx-pose data
        **kwargs
            Additional parameters:
            - nwb_file_name: Parent NWB file name for linking
            - model_name: Name of specific PoseEstimation to import (required
              if multiple models in file)
            - skeleton_id: Override skeleton ID
            - model_params_id: Override model params ID
            - model_id: Override model ID

        Returns
        -------
        dict
            Primary key for the created Model entry

        Raises
        ------
        ImportError
            If ndx-pose is not installed
        ValueError
            If NWB file has no pose data or multiple models without model_name
        """
        if ndx_pose is None:
            raise ImportError(
                "ndx-pose is required to import pose estimation models from NWB"
                "Install with: pip install ndx-pose>=0.2.0"
            )

        if model_path.suffix != ".nwb":
            raise ValueError("ndx-pose model path must be an .nwb file")

        self._info_msg(f"Importing ndx-pose model from {model_path}")

        # Step 1: Read NWB file and extract pose data
        with NWBHDF5IO(str(model_path), mode="r") as nwb_io:
            nwbfile = nwb_io.read()

            # Step 2: Find behavior processing module
            if "behavior" not in nwbfile.processing:
                raise ValueError(
                    f"No 'behavior' processing module in NWB file: {model_path}"
                )

            # Step 3: Find PoseEstimation objects in behavior processing module

            behavior_module = nwbfile.processing["behavior"]
            pose_estimations = {
                name: obj
                for name, obj in behavior_module.data_interfaces.items()
                if isinstance(obj, ndx_pose.PoseEstimation)
            }

            if not pose_estimations:
                raise ValueError(
                    f"No PoseEstimation objects found in NWB file: {model_path}"
                )

            # Step 4: Select which PoseEstimation to import
            model_name = kwargs.get("model_name")
            if len(pose_estimations) > 1 and not model_name:
                raise ValueError(
                    f"NWB file contains {len(pose_estimations)} mation models. "
                    f"Please specify 'model_name' parameter. Available models: "
                    f"{list(pose_estimations.keys())}"
                )

            if model_name:
                if model_name not in pose_estimations:
                    raise ValueError(
                        f"Model '{model_name}' not found in NWB file. "
                        f"Available models: {list(pose_estimations.keys())}"
                    )
                pose_estimation = pose_estimations[model_name]
            else:
                # Single model, use it
                pose_estimation = list(pose_estimations.values())[0]
                model_name = list(pose_estimations.keys())[0]

            self._info_msg(f"Importing model: {model_name}")

            # Step 5: Extract skeleton from PoseEstimation
            skeleton_obj = pose_estimation.skeleton
            if skeleton_obj is None:
                raise ValueError(
                    f"PoseEstimation '{model_name}' has no skeleton defined"
                )

            # Step 6: Extract bodyparts (nodes) and edges
            bodyparts = list(skeleton_obj.nodes)
            edges_array = skeleton_obj.edges

            # Convert edge indices to bodypart name tuples
            edges = []
            for edge in edges_array:
                node1_idx, node2_idx = int(edge[0]), int(edge[1])
                edges.append((bodyparts[node1_idx], bodyparts[node2_idx]))

            self._info_msg(
                f"Skeleton has {len(bodyparts)} bodyparts, {len(edges)} edges"
            )

            # Step 7: Extract metadata from PoseEstimation
            source_software = getattr(
                pose_estimation, "source_software", "ndx-pose"
            )
            source_version = getattr(
                pose_estimation, "source_software_version", None
            )

            # Map source_software to our tool naming
            tool_mapping = {"deeplabcut": "DLC", "sleap": "SLEAP"}
            tool = tool_mapping.get(source_software.lower(), "ndx-pose")

            # Build params dict from available metadata
            params = {
                "source_software": source_software,
                "model_name": model_name,
                "nwb_file": str(model_path),
            }
            if source_version:
                params["source_software_version"] = source_version
            if pose_estimation.description:
                params["description"] = pose_estimation.description
            if pose_estimation.original_videos:
                params["original_videos"] = list(
                    pose_estimation.original_videos
                )
            if pose_estimation.labeled_videos:
                params["labeled_videos"] = list(pose_estimation.labeled_videos)
            if pose_estimation.dimensions is not None:
                # Convert to numpy array first (could be HDF5 Dataset)
                params["dimensions"] = np.array(
                    pose_estimation.dimensions
                ).tolist()

            # Add default required params for DLC if importing from DLC
            if tool == "DLC" and "project_path" not in params:
                params["project_path"] = str(
                    model_path.parent
                )  # Use model directory

        # Step 8: Create skeleton config dict for Skeleton.insert1()
        skeleton_config = {
            "bodyparts": bodyparts,
            "skeleton": edges,
        }
        # Add skeleton_id to config dict if provided
        if "skeleton_id" in kwargs and kwargs["skeleton_id"] is not None:
            skeleton_config["skeleton_id"] = kwargs["skeleton_id"]

        # Step 9: Create or retrieve Skeleton entry
        skeleton_key = Skeleton().insert1(
            skeleton_config,
            check_duplicates=True,
            skip_duplicates=True,
        )
        self._info_msg(f"Skeleton: {skeleton_key['skeleton_id']}")

        # Step 10: Create or retrieve ModelParams entry
        self._info_msg("Creating or retrieving ModelParams entry")
        model_params_key = ModelParams().insert1(
            dict(
                tool=tool,
                params=params,
                model_params_id=kwargs.get("model_params_id"),
                skeleton_id=skeleton_key["skeleton_id"],
            ),
            skip_duplicates=True,
            accept_default=True,
        )

        if not model_params_key:
            # Fetch existing params
            params_hash = dj.hash.key_hash(params)
            model_params_key = (
                ModelParams() & {"tool": tool, "params_hash": params_hash}
            ).fetch1("KEY")

        self._info_msg(f"ModelParams: {model_params_key['model_params_id']}")

        # Step 11: Create VidFileGroup placeholder
        vid_group_id = f"ndx-pose-{model_name}"
        vid_group_key = VidFileGroup().insert1(
            {
                "vid_group_id": vid_group_id,
                "description": f"Videos from {model_name} in {model_path.name}",
            },
            skip_duplicates=True,
        )
        if not vid_group_key:
            vid_group_key = (
                VidFileGroup() & {"vid_group_id": vid_group_id}
            ).fetch1("KEY")

        self._info_msg(f"VidFileGroup: {vid_group_key['vid_group_id']}")

        # Step 11b: Link original_videos from NWB to existing VideoFile entries
        matched_files = []
        for vid_path in params.get("original_videos", []):
            try:
                vf_key = VideoFile().fetch_key_from_path(str(vid_path))
                matched_files.append(vf_key)
            except Exception:
                self._logger.debug(
                    f"No VideoFile match for ndx-pose video: {vid_path}"
                )
        if matched_files:
            VidFileGroup().add_files(
                vid_group_key["vid_group_id"], matched_files
            )
            self._info_msg(
                f"Linked {len(matched_files)} video(s) to VidFileGroup "
                f"'{vid_group_key['vid_group_id']}'"
            )
        else:
            self._info_msg(
                "No matching VideoFile entries found for original_videos. "
                "VidFileGroup created as placeholder — call add_files() or "
                "register sessions before running PoseEstim.populate()."
            )

        # Step 12: Create ModelSelection entry
        sel_key = {**model_params_key, **vid_group_key}
        ModelSelection().insert1(sel_key, skip_duplicates=True)

        # Return the existing Model entry if one already exists for this path.
        if existing := (self & {"model_path": str(model_path)}):
            return existing.fetch1()

        # Step 13: Create Model entry
        model_id = kwargs.get("model_id")
        if not model_id:
            # Generate default model_id using helper (max 32 chars)
            model_id = default_pk_name(
                "mdl", {"model_name": model_name, "nwb_file": str(model_path)}
            )

        # For ndx-pose imports the source NWB file IS the model artifact —
        # analysis_file_name is intentionally nullable here. Unlike DLC
        # imports (which create a fresh AnalysisNwbfile to store training
        # metadata), the ndx-pose file already contains all model info and
        # is tracked via model_path. An AnalysisNwbfile link can be added
        # after import if the file is later registered in a Spyglass session.
        nwb_file_name = kwargs.get("nwb_file_name")
        if nwb_file_name and (
            AnalysisNwbfile & {"analysis_file_name": nwb_file_name}
        ):
            analysis_file_name = nwb_file_name
        else:
            analysis_file_name = None
            if nwb_file_name:
                self._warn_msg(
                    f"nwb_file_name {nwb_file_name!r} not found in "
                    "AnalysisNwbfile; analysis_file_name link skipped."
                )
            else:
                self._info_msg(
                    "ndx-pose import: analysis_file_name=None (model_path is "
                    "the NWB reference). Pass nwb_file_name= to link a "
                    "registered AnalysisNwbfile."
                )

        model_key = {
            **sel_key,
            "model_id": model_id,
            "model_path": str(model_path),
        }
        if analysis_file_name:
            model_key["analysis_file_name"] = analysis_file_name

        self.insert1(model_key, skip_duplicates=True, allow_direct_insert=True)
        self._info_msg(f"Model imported: {model_id}")

        # skeleton_id is a secondary attribute of ModelParams, not a Model column.
        # Include it in the return value for convenience.
        return {**model_key, "skeleton_id": skeleton_key["skeleton_id"]}

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
        warnings = []
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
                "Use Model.import_model() to import a model first."
            )
            return {
                "valid": False,
                "checks": checks,
                "errors": errors,
                "warnings": warnings,
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
                "warnings": warnings,
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
                    warnings.append(
                        f"DLC model path should be config.yaml, got: {model_path.name}"
                    )
                else:
                    # Try to read config to verify it's valid YAML
                    try:
                        import yaml

                        with open(model_path) as f:
                            config = yaml.safe_load(f)
                            if "project_path" not in config:
                                warnings.append(
                                    "DLC config missing 'project_path' field"
                                )
                    except Exception as e:
                        warnings.append(f"Could not parse DLC config: {e}")

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
                    warnings.append(
                        f"Skeleton has only {len(bodyparts)} bodypart(s)"
                    )
                else:
                    checks["skeleton_valid"] = True

                if edges is None or len(edges) == 0:
                    warnings.append("Skeleton has no edges defined")
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

        # Check 5: (Optional) Inference readiness
        if check_inference and checks["model_path_exists"]:
            tool = model_entry["tool"]

            # TODO: Separate into `_verify_dlc`
            if tool == "DLC":
                # Check if DLC is available
                if create_training_dataset is None or train_network is None:
                    warnings.append(
                        "DeepLabCut not installed - cannot verify inference readiness"
                    )
                else:
                    # Verify model directory structure
                    if model_path.suffix in [".yaml", ".yml"]:
                        model_dir = model_path.parent
                        # Look for trained model directories
                        train_dirs = list(model_dir.rglob("**/train")) + list(
                            model_dir.rglob("**/dlc-models")
                        )
                        if train_dirs:
                            checks["inference_ready"] = True
                        else:
                            warnings.append(
                                "No trained model directories found - "
                                "model may not be trained yet"
                            )
            elif tool == "ndx-pose":
                # For ndx-pose models, they're stored in NWB so can't run inference
                warnings.append(
                    "ndx-pose models cannot run inference directly - "
                    "use original DLC/SLEAP project"
                )
            else:
                warnings.append(
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
            "warnings": warnings,
            "model_info": model_info,
        }

        # Log results
        if valid:
            self._info_msg(f"Model verification PASSED: {model_key}")
            if warnings:
                self._warn_msg(f"Warnings: {warnings}")
        else:
            self._err_msg(f"Model verification FAILED: {model_key}")
            self._err_msg(f"Errors: {errors}")

        return result

    def _get_latest_dlc_model_info(self, config: dict) -> dict:
        """Given a DLC project path, return available model info

        Parameters
        ----------
        config : dict
            DLC project configuration dictionary, including project_path.

        Returns
        -------
        OrderedDict
            Dictionary with model paths as keys and dictionaries as values. Each
            value dictionary contains 'iteration', 'trainFraction', 'shuffle',
            and 'date_trained'.
        """

        project_path = Path(config["project_path"])
        if not project_path.exists() or not project_path.is_dir():
            raise FileNotFoundError(f"Invalid project path: {project_path}")

        # Regex to extract iteration, train fraction, and shuffle
        # EG: dlc-models/iteration-0/TESTSep8-trainset80shuffle1/
        re_pattern = re.compile(r"iteration-(\d+)/.*-trainset(\d+)shuffle(\d+)")

        latest_model = None
        for pose_cfg in project_path.rglob("pose_cfg.yaml"):
            try:
                # Extract metadata from the relative path
                match = re.search(re_pattern, str(pose_cfg.parent))
                if not match:
                    continue
                date_trained = datetime.fromtimestamp(pose_cfg.stat().st_mtime)
                if (
                    latest_model
                    and date_trained <= latest_model["date_trained"]
                ):
                    continue  # Skip if not the latest model
                iteration, train_fraction, shuffle = map(int, match.groups())
                latest_model = {
                    "path": pose_cfg.parent,
                    "iteration": iteration,
                    "trainFraction": train_fraction / 100,
                    "shuffle": shuffle,
                    "date_trained": date_trained,
                }
            except (IndexError, ValueError):
                continue  # Skip files that don't match the expected pattern
        return latest_model or dict()

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

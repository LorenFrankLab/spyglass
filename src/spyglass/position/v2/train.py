"""DataJoint tables for training and managing pose estimation models.

NOTE: This approach departs from position v1 by choosing not to store
information already on disk. v1 maintained copies of files in the event that
a user or DLC modified them so that the database always reflected the 'ground
truth'. In practice, file modification is rare, and storing copies is
inefficient.
"""

import re
import warnings
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Tuple, Union

import datajoint as dj
import networkx as nx
import numpy as np
import yaml
from datajoint.errors import DuplicateError
from pynwb import NWBHDF5IO

from spyglass.common import AnalysisNwbfile, LabMember
from spyglass.position.utils import get_most_recent_file, get_param_names
from spyglass.position.v2.video import VidFileGroup
from spyglass.utils import SpyglassMixin, logger

# ------------------------------ Optional imports ------------------------------
try:
    import ndx_pose
except ImportError:
    ndx_pose = None

try:
    from deeplabcut import create_training_dataset, train_network
    from deeplabcut.core.engine import Engine
except ImportError:
    create_training_dataset, train_network, Engine = [None] * 3

# -------------------------------- Module setup --------------------------------
warnings.filterwarnings("ignore", category=UserWarning, module="networkx")

schema = dj.schema("cbroz_position_v2_train")  # TODO: remove cbroz prefix


# ----------------------------------- Helpers ----------------------------------
def default_pk_name(prefix: str, params: dict, limit: int = 32) -> str:
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
    """
    when = datetime.utcnow()
    h = dj.hash.key_hash(params)
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
    bodyparts : blob # list of bodypart names
    edges=NULL: blob # list of edge pairs, List[Tuple[str, str]]
    hash=NULL : varchar(32) # hash of graph for duplicate detection
    """

    # TODO: Should BodyPart be a part table to foreign key reference?
    # Edges are optional to allow unconnected body parts

    contents = [
        ["1LED", ["whiteLED"], "", "1042f79ba7ecbe933e43d9e23b8bb3e2"],
        [
            "2LED",
            ["greenLED", "redLED_C"],
            [("greenLED", "redLED_C")],
            "3bac2dea32a574e9d97917dc8ea8480a",
        ],
        [
            "4LED",
            ["greenLED", "redLED_C", "redLED_L", "redLED_R"],
            [
                ("greenLED", "redLED_C"),
                ("greenLED", "redLED_L"),
                ("greenLED", "redLED_R"),
            ],
            "cfe386902f8c79d228e8ae628079945e",
        ],
    ]

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
            logger.warning(f"Skeleton ID {skeleton_id} already exists")
            return (self & dict(skeleton_id=skeleton_id)).fetch1("KEY")
        if not bodyparts or edges is None:
            raise dj.DataJointError(
                f"Key must include 'bodyparts' and 'edges' fields: {key}"
            )

        # Validate body parts against the reference table
        labels_norm = [self._normalize_label(x) for x in bodyparts]
        _ = self._validate_bodyparts(set(labels_norm))

        # If dupe error, compute labeled graph
        shape_hash = self._shape_hash_from_edges(bodyparts, edges)
        G_new, checks = None, []
        if check_duplicates:
            G_new = self._build_labeled_graph(bodyparts, edges)
            checks = self & dict(hash=shape_hash)

        # Check for graph matches
        for row in checks:
            G_old = self._build_labeled_graph(row["bodyparts"], row["edges"])

            def node_match(a, b):
                return self._fuzzy_equal(
                    a.get("label", ""), b.get("label", ""), name_similarity
                )

            GM = nx.algorithms.isomorphism.GraphMatcher(
                G_new, G_old, node_match=node_match
            )
            if GM.is_isomorphic():
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

    tool_info = {  # Dict[tool-name, Dict[required, skipped, accepted, aliases]]
        "DLC": {
            "required": [
                "net_type",
            ],
            "skipped": ["project_path", "video_sets"],
            "accepted": set(
                [
                    *get_param_names(train_network),
                    *get_param_names(create_training_dataset),
                ]
            ),
            "aliases": {  # alternate names for same param
                "net_type": ["default_net_type"]
            },
        },
        "ndx-pose": {
            "required": [
                "source_software",
                "model_name",
            ],
            "skipped": [],
            "accepted": {
                "source_software",
                "source_software_version",
                "model_name",
                "nwb_file",
                "description",
                "original_videos",
                "labeled_videos",
                "dimensions",
                "scorer",
                "nodes",
                "edges",
            },
            "aliases": {},
        },
    }
    default_dlc_params = {
        "params": {},
        "shuffle": 1,
        "trainingsetindex": 0,
        "model_prefix": "",
    }
    contents = [
        (
            "dlc_default",
            "DLC",
            default_dlc_params,
            None,
            dj.hash.key_hash(default_dlc_params),
        )
    ]

    def get_accepted_params(self, tool):
        """Return all accepted parameters for DLC model training."""
        if tool not in self.tool_info:
            raise ValueError(f"Tool {tool} not supported")
        return self.tool_info[tool]["accepted"]

    def _append_aliases(self, tool, params):
        """Append any parameter aliases to the params dictionary."""
        if tool not in self.tool_info:
            raise ValueError(f"Tool {tool} not supported")
        aliases = self.tool_info[tool].get("aliases", {})
        for primary, alias_list in aliases.items():
            if primary in params:
                for alias in alias_list:
                    if alias not in params:
                        params[alias] = params[primary]
        return params

    def insert1(self, key, accept_default=False, **kwargs):
        """Insert model parameters into the database.

        1. Check if key is a dictionary
        2. Check if tool is supported
        3. Check if all required parameters are present, remove skipped
        4. Check if params already exist
        """
        if not isinstance(key, dict):
            raise TypeError("Key must be a dictionary")

        this_tool = key.get("tool", "UNKNOWN")
        if this_tool not in self.tool_info:
            raise ValueError(f"Tool not supported {this_tool}")

        tool_info = self.tool_info[this_tool]
        params = {  # Drop skipped params
            k: v
            for k, v in key["params"].items()
            if k not in tool_info["skipped"]
        }

        params = self._append_aliases(this_tool, params)

        tool_required = tool_info["required"]  # Ensure all required params
        if not all(r in params for r in tool_required):
            raise ValueError(
                f"Missing required params: {tool_required - params.keys()}"
            )

        params_hash_dict = dict(params_hash=dj.hash.key_hash(params))
        if dupe := (self & params_hash_dict & dict(tool=key["tool"])):
            logger.warning(f"Entry exists with same params: \n{dupe}")
            return dupe.fetch1("KEY")

        model_params_id: str = key.get("model_params_id", "").strip()
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
        pass

    def train(self, key):
        raise NotImplementedError("add selection w/key as parent, populate")

    def evaluate(self):
        raise NotImplementedError("Evaluate trained model")

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

        config = kwargs.update(config) or config

        skeleton = Skeleton().insert1(
            config, check_duplicates=False, skip_duplicates=True
        )
        model_params = ModelParams().insert1(
            dict(tool="DLC", params=config),
            raise_on_duplicate=False,
        )
        model_id = kwargs.get("model_id") or default_pk_name(
            f"DLC-{config['Task']}{config['date']}",
            dict(tool="DLC", model_path=str(model_path)),
        )
        sel_key = dict(skeleton, **model_params)
        ModelSelection().insert1(sel_key)

        raise NotImplementedError("Need to generate NWB file")

        self.insert1(
            dict(sel_key, model_id=model_id, model_path=str(model_path))
        )

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

        logger.info(f"Importing ndx-pose model from {model_path}")

        # Step 1: Read NWB file and extract pose data
        with NWBHDF5IO(str(model_path), mode="r") as io:
            nwbfile = io.read()

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

            logger.info(f"Importing model: {model_name}")

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

            logger.info(
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

        # Step 8: Create skeleton config dict for Skeleton.insert1()
        skeleton_config = {
            "bodyparts": bodyparts,
            "skeleton": edges,
        }

        # Step 9: Create or retrieve Skeleton entry
        skeleton_key = Skeleton().insert1(
            skeleton_config,
            check_duplicates=True,
            skip_duplicates=True,
            skeleton_id=kwargs.get("skeleton_id"),
        )
        logger.info(f"Skeleton: {skeleton_key['skeleton_id']}")

        # Step 10: Create or retrieve ModelParams entry
        logger.info("Creating or retrieving ModelParams entry")
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

        logger.info(f"ModelParams: {model_params_key['model_params_id']}")

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

        logger.info(f"VidFileGroup: {vid_group_key['vid_group_id']}")

        # Step 12: Create ModelSelection entry
        sel_key = {**model_params_key, **vid_group_key}
        ModelSelection().insert1(sel_key, skip_duplicates=True)

        # Step 13: Create Model entry
        model_id = kwargs.get("model_id")
        if not model_id:
            # Generate default model_id
            model_id = f"ndx-pose-{model_name}-{model_path.stem}"

        # Note: We don't create a new NWB file - we reference the source NWB
        # The analysis_file_name will point to the source NWB file
        nwb_file_name = kwargs.get("nwb_file_name")
        if nwb_file_name:
            # Use provided parent NWB file
            analysis_file_name = nwb_file_name
        else:
            # Use the source NWB file itself
            analysis_file_name = model_path.name
            logger.info(
                f"No nwb_file_name provided. Using source NWB: {analysis_file_name}"
            )

        model_key = {
            **sel_key,
            "model_id": model_id,
            "analysis_file_name": analysis_file_name,
            "model_path": str(model_path),
        }

        self.insert1(model_key, skip_duplicates=True)
        logger.info(f"Model imported: {model_id}")

        return model_key

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

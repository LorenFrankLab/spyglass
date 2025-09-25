"""DataJoint tables for training and managing pose estimation models.

NOTE: This approach departs from position v1 by choosing not to store
information already on disk. v1 maintained copies of files in the event that
a user or DLC modified them so that the database always reflected the 'ground
truth'. In practice, file modification is rare, and storing copies is
inefficient.

"""

import hashlib
import warnings
from collections import OrderedDict
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Tuple, Union

import datajoint as dj
import networkx as nx
import yaml
from datajoint.errors import DuplicateError

from spyglass.common import AnalysisNwbfile, LabMember
from spyglass.position.utils import get_most_recent_file, get_param_names
from spyglass.position.v2.video import VidFileGroup
from spyglass.utils import SpyglassMixin, logger

try:
    from deeplabcut import create_training_dataset, train_network
    from deeplabcut.core.engine import Engine
except ImportError:
    create_training_dataset, train_network, Engine = [None] * 3

warnings.filterwarnings("ignore", category=UserWarning, module="networkx")


schema = dj.schema("cbroz_position_v2_train")  # TODO: remove cbroz prefix


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
        """Normalize a body part label for matching/comparison."""
        return " ".join(
            s.strip().lower().replace("_", " ").replace("-", " ").split()
        )

    @staticmethod
    def _fuzzy_equal(a: str, b: str, threshold: float = 0.85) -> bool:
        """Fuzzy string equality at given similarity threshold [0..1]."""
        return SequenceMatcher(None, a, b).ratio() >= threshold

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

        label_set = set(labels_norm)
        # Edge endpoints must belong to the declared label set
        for u, v in self._norm_edges(edges):
            if u not in label_set or v not in label_set:
                raise ValueError(
                    f"Edge uses label not in bodyparts: ({u!r}, {v!r})"
                )

        G = nx.Graph()
        for label in labels_norm:
            G.add_node(label, label=label)
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
            return (self & dict(skeleton_id=skeleton_id)).fetch1()
        if not bodyparts or not edges:
            raise dj.DataJointError(
                f"Key must include 'bodyparts' and 'edges' fields: {key}"
            )

        # Validate body parts against the reference table
        labels_norm = [self._normalize_label(x) for x in bodyparts]
        _ = self._validate_bodyparts(set(labels_norm))

        # If dupe error, compute labeled graph
        shape_hash = self._shape_hash_from_edges(bodyparts, edges)
        G_new = None
        if check_duplicates:
            G_new = self._build_labeled_graph(bodyparts, edges)

        # Check for graph matches
        checks = self & dict(hash=shape_hash) if check_duplicates else []
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
                if check_duplicates:
                    raise dj.DataJointError(
                        "Duplicate skeleton detected. "
                        + f"Key matches skeleton_id={row['skeleton_id']})"
                    )
                return row

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

        if key["tool"] not in self.tool_info:
            raise ValueError(f"Tool {key['tool']} not supported")

        tool_info = self.tool_info[key["tool"]]
        params = {  # Drop skipped params
            k: v
            for k, v in key["params"].items()
            if k not in tool_info["skipped"]
        }

        params = self._append_aliases(key["tool"], params)

        tool_required = tool_info["required"]  # Ensure all required params
        if not all(k in tool_required for k in params):
            raise ValueError(
                f"Missing required params: {tool_required - params.keys()}"
            )

        params_hash_dict = dict(params_hash=dj.hash.key_hash(params))
        if dupe := (self & params_hash_dict & dict(tool=key["tool"])):
            logger.warning(f"Entry exists with same params: {dupe}")
            return dupe

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
    -> ModelSelection
    ---
    -> AnalysisNwbfile # TODO: store model here? Or at next step?
    model_path    : varchar(255)
    """

    def make(self, key):
        pass

    def train(self, key):
        raise NotImplementedError("add selection w/key as parent, populate")

    def evaluate(self):
        raise NotImplementedError("Evaluate trained model")

    def import_model(
        self,
        model_path: Union[Path, str],
        tool: str = "DLC",
        skeleton_id: Union[str, None] = None,
        model_params_id: Union[str, None] = None,
        model_id: Union[str, None] = None,
        **kwargs,
    ):
        """Import an existing trained model into the database.

        Parameters
        ----------
        model_path : Union[Path, str]
            Path to the model file or configuration (e.g., DLC .yml file)
        tool : str, optional
            Tool used to train the model, by default "DLC"
        skeleton_id : Union[str, None], optional
            Skeleton ID to associate with the model. If None, auto-generate
            default
        model_params_id : Union[str, None], optional
            Model parameters ID to associate with the model. If None,
            auto-generate
        model_id : Union[str, None], optional
            Model ID to assign. If None, auto-generate
        **kwargs
            Additional parameters for specific tools.

        Raises
        ------
        FileNotFoundError
            If the model path does not exist.
        NotImplementedError
            If the tool is not supported or import not implemented.
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        kwargs.update(
            dict(
                kwargs,
                skeleton_id=skeleton_id,
                model_params_id=model_params_id,
                model_id=model_id,
            )
        )

        if tool == "DLC":
            self._import_dlc_model(model_path, **kwargs)
        raise NotImplementedError(f"Model import not implemented for {tool}")

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

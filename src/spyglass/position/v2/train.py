import hashlib
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Tuple

import datajoint as dj
import networkx as nx
import yaml
from datajoint.errors import DuplicateError

from spyglass.common import AnalysisNwbfile, LabMember
from spyglass.position.utils import get_param_names
from spyglass.position.v2.video import VidFileGroup

try:
    from deeplabcut import create_training_dataset, train_network
except ImportError:
    create_training_dataset, train_network = None, None

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
class BodyPart(dj.Lookup):
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
        [[f"bodypart{i}"] for i in range(1, 3)],
        ["objectA"],
    ]

    def insert1(self, key, **kwargs):
        """Insert body part into the database."""
        if not LabMember().user_is_admin():
            raise PermissionError("Only admins can insert body parts")
        super().insert1(key, **kwargs)


@schema
class Skeleton(dj.Lookup):
    definition = """
    skeleton_id: varchar(32)
    ---
    bodyparts : blob # list of bodypart names
    edges=NULL: blob # list of edge pairs, List[Tuple[str, str]]
    hash: varchar(128) # hash of graph for duplicate detection
    """

    # TODO: Should BodyPart be a part table to foreign key reference?

    contents = [
        ["1LED", ["whiteLED"], ""],
        ["2LED", ["greenLED", "redLED_C"], [("greenLED", "redLED_C")]],
        [
            "4LED",
            ["greenLED", "redLED_C", "redLED_L", "redLED_R"],
            [
                ("greenLED", "redLED_C"),
                ("greenLED", "redLED_L"),
                ("greenLED", "redLED_R"),
            ],
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
        valid = {self._normalize_label(x) for x in BodyPart.fetch("bodypart")}
        missing = set(self._normalize_label(x) for x in labels) - valid
        if missing:
            raise dj.DataJointError(
                f"Unknown bodypart name(s) (not in BodyPart): {sorted(missing)}"
                + "\nPlease either change to existing name or "
                + "consult with admin to add it."
            )

    def _shape_hash_from_edges(
        self, labels: list[str], edges: list[tuple[str, str]]
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
        for u, v in self._norm_edges(edges):
            if u0 not in idx_of or v0 not in idx_of:
                raise dj.DataJointError(
                    f"Edge ({u!r}, {v!r}) references a label not in body_parts."
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
                    f"Edge uses label not in body_parts: ({u!r}, {v!r})"
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
        raise_on_duplicate: bool = True,
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
        raise_on_duplicate : bool, optional
            Whether to run duplicate detection, by default True
        """
        skeleton_id: str = key.get("skeleton_id", "").strip()
        body_parts: List[str] = key.get("bodyparts", [])
        edges: List[Tuple[str, str]] = key.get("edges", [])

        if skeleton_id in self.fetch("skeleton_id"):
            logger.warning(f"Skeleton ID {skeleton_id} already exists")
            return (self & dict(skeleton_id=skeleton_id)).fetch1()
        if not body_parts or not edges:
            raise dj.DataJointError(
                f"Key must include 'bodyparts' and 'edges' fields: {key}"
            )

        # Validate body parts against the reference table
        labels_norm = [self._normalize_label(x) for x in body_parts]
        _ = self._validate_bodyparts(set(labels_norm))

        # If dupe error, compute shape hash and labeled graph
        shp, G_new = None, None
        if raise_on_duplicate:
            shp = self._shape_hash_from_edges(body_parts, edges)
            G_new = self._build_labeled_graph(body_parts, edges)

        # Deep check via isomorphism with fuzzy node labels
        checks = self & dict(shape_hash=shp) if raise_on_duplicate else []
        for row in checks:
            G_old = self._build_labeled_graph(row["body_parts"], row["edges"])

            def node_match(a, b):
                return self._fuzzy_equal(
                    a.get("label", ""), b.get("label", ""), name_similarity
                )

            GM = nx.algorithms.isomorphism.GraphMatcher(
                G_new, G_old, node_match=node_match
            )
            if GM.is_isomorphic():
                if raise_on_duplicate:
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
        self.insert1(
            dict(
                insert_pk,
                body_parts=body_parts,
                edges=edges,
                shape_hash=shp,
            ),
            skip_duplicates=False,
        )

        return insert_pk

    def get_centroid_method(self):
        raise NotImplementedError("Return func based graph")


@schema
class ModelParams(dj.Lookup):
    definition = """
    model_params_id: varchar(32)
    tool: varchar(32)
    ---
    params: blob
    -> [nullable] Skeleton
    params_hash: varchar(64) # hash of params
    unique index (tool, params_hash)
    """

    tool_info = {  # Dict[tool-name, Dict[required, skipped, accepted]]
        "DLC": {
            "required": [
                "shuffle",
                "trainingsetindex",
                "net_type",
                "gputouse",
            ],
            "skipped": ["project_path", "video_sets"],
            "accepted": set(
                [
                    *get_param_names(train_network),
                    *get_param_names(create_training_dataset),
                ]
            ),
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

        tool_required = tool_info["required"]  # Ensure all required params
        if not all(k in tool_required for k in params.keys()):
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
class ModelSelection(dj.Manual):
    definition = """
    -> ModelParams
    -> VidFileGroup
    ---
    parent_id=NULL: varchar(32) # ID of parent model, if any
    """
    tool_info = {"DLC": "some_info"}

    def import_model(self, model_path: str, tool="DLC"):
        if tool not in self.tool_info:
            raise ValueError(f"Tool {tool} not supported")
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        raise NotImplementedError("Import trained model")


@schema
class Model(dj.Computed):
    definition = """
    model_id: varchar(32)
    -> ModelSelection
    ---
    -> AnalysisNwbfile # TODO: store model here? Or at next step?
    model_path    : varchar(255)
    """

    def make(self, key):
        pass

    def train(self):
        raise NotImplementedError("add selection w/key as parent, populate")

    def evaluate(self):
        raise NotImplementedError("Evaluate trained model")

    def import_model(self, model_path: Union[Path, str], tool:str="DLC"):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        if tool = "DLC":
            self._import_dlc_model(model_path)
        raise NotImplementedError(f"Model import not implemented for {tool}")

    def _import_dlc_model(self, model_path: Path):
        if model_path.suffix not in ["yml", "yaml"]:
            raise ValueError("DLC model path must be a .yml or .yaml file")
        with open(model_path, "r") as f:
            config = yaml.safe_load(f)
        skeleton = Skeleton().insert1(config, raise_on_duplicate=False)
        model_params = ModelParams().insert1(
            dict(tool="DLC", params=config),
            raise_on_duplicate=False,
        )

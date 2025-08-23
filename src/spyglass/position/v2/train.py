from pathlib import Path
from typing import List, Tuple

import datajoint as dj
from datajoint.errors import DuplicateError

from spyglass.common import AnalysisNwbfile, LabMember
from spyglass.position.utils import get_param_names
from spyglass.position.v2.video import VidFileGroup

try:
    from deeplabcut import create_training_dataset, train_network
except ImportError:
    create_training_dataset, train_network = None, None

schema = dj.schema("cbroz_position_v2_train")


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
        ["spine1"],
        ["spine2"],
        ["spine3"],
        ["spine4"],
        ["spine5"],
        # Tail
        ["tailBase"],
        ["tailMid"],
        ["tailTip"],
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
    """

    contents = [
        ["1LED", ["whiteLED"], ""],
        ["2LED", ["greenLED", "redLED_C"], ""],
        ["4LED", ["greenLED", "redLED_C", "redLED_L", "redLED_R"], ""],
    ]

    def _similar_exists(self, key):
        # return entry
        raise NotImplementedError("Check if similar skeleton exists")

    def _check_missing_parts(
        self,
        bodyparts: List[str],
        edges: List[Tuple[str, str]] = None,
        **kwargs,
    ):
        """Check if all bodyparts are in edges."""
        if not edges:
            return True
        edges_set = set([part for edge in edges for part in edge])
        if missing := edges_set - set(bodyparts):
            raise ValueError(
                f"Missing bodyparts in edges: {missing}\n\t"
                f"Edges: {edges}\n\tBodyparts: {bodyparts}"
            )
        return True

    def _check_valid_parts(self, bodyparts: List[str], **kwargs):
        """Check if all bodyparts are in BodyPart table."""
        valid_parts = set(BodyPart().fetch("bodypart"))
        if missing := set(bodyparts) - valid_parts:
            raise ValueError(
                f"Entry has bodyparts not in BodyPart table: {missing}\n"
                "Please rename, or contact a database admin to add them."
            )
        return True

    def insert1(self, key, force=False, **kwargs):
        """Insert skeleton into the database, as admin only."""
        if not LabMember().user_is_admin():
            raise PermissionError("Only admins can insert skeletons")

        existing = self._similar_exists(key)
        if existing and not force:
            raise DuplicateError(
                f"Skeleton with similar structure already exists: {existing}"
                "Use force=True to add anyway"
            )

        _ = self._check_missing_parts(**key)
        _ = self._check_valid_parts(**key)

        super().insert1(key, **kwargs)

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

    def insert1(self, key, **kwargs):
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
            raise DuplicateError(f"Entry exists with same params: {dupe}")

        super().insert1(dict(key, params=params, **params_hash_dict), **kwargs)


@schema
class ModelSelection(dj.Manual):
    definition = """
    -> ModelParams
    -> VidFileGroup
    ---
    parent_id=NULL: varchar(64) # ID of parent model, if any
    model_path    : varchar(255)
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
    ---
    -> AnalysisNwbfile # TODO: store model here? Or at next step?
    -> [nullable] ModelSelection # null for imported models
    """

    def make(self, key):
        pass

    def train(self):
        raise NotImplementedError("add selection w/key as parent, populate")

    def evaluate(self):
        raise NotImplementedError("Evaluate trained model")

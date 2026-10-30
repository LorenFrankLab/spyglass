"""DataJoint tables for training and managing pose estimation models.

NOTE: This approach departs from position v1 by choosing not to store
information already on disk. v1 maintained copies of files in the event that
a user or DLC modified them so that the database always reflected the 'ground
truth'. In practice, file modification is rare, and storing copies is
inefficient.
"""

import getpass
import shutil
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import datajoint as dj
import pandas as pd
from pynwb import NWBHDF5IO

from spyglass.common import AnalysisNwbfile, LabMember, VideoFile
from spyglass.position.utils import suppress_print_from_package
from spyglass.position.utils.path_helpers import (
    resolve_model_path,
)
from spyglass.position.utils.path_helpers import (
    to_stored_path as _to_stored_path,
)
from spyglass.position.utils.protocols import ModelMetadata, default_pk_name
from spyglass.position.utils.tool_strategies import ToolStrategyFactory
from spyglass.position.utils.yaml_io import load_yaml
from spyglass.position.v2.utils.config_io import _rename_leaves
from spyglass.position.v2.utils.model import (
    rank_model_rows,
    redundant_model_msg,
    reuse_candidates,
)
from spyglass.position.v2.utils.skeleton import (
    build_canonical_map,
    canonicalize,
    centroid_method_for,
    clean_edges,
    draw_skeleton,
    is_duplicate_skeleton,
    normalize_label,
    shape_hash_from_edges,
    validate_skeleton_graph,
)
from spyglass.position.v2.utils.training_plots import (
    build_training_history_figure,
)
from spyglass.position.v2.utils.training_io import (
    aggregate_training_stats,
    discover_training_csvs,
    parse_training_csv,
)
from spyglass.position.v2.video import VidFileGroup
from spyglass.utils import SpyglassMixin, logger

# -------------------------------- Module setup --------------------------------
warnings.filterwarnings("ignore", category=UserWarning, module="networkx")

schema = dj.schema("cbroz_position_v3_train")  # TODO: remove cbroz prefix


# ----------------------------------- Helpers ----------------------------------


# Parameter dataclasses for methods with 5+ parameters
# ---------------------------------- Tables -----------------------------------
@schema
class BodyPart(SpyglassMixin, dj.Lookup):
    """Accepted body parts for pose estimation."""

    definition = """
    bodypart: varchar(32)
    ---
    source='curated': enum('curated','imported')  # imported = tool/zoo vocab
    """

    # Note: This reflects existing FrankLab body parts, rather than model zoo
    _curated = [
        # LEDs
        "greenLED",
        "redLED_C",
        "redLED_L",
        "redLED_R",
        "whiteLED",
        # Drive
        "driveBack",
        "driveFront",
        # Head
        "head",
        "nose",
        "earL",
        "earR",
        # Body
        "forelimbL",
        "forelimbR",
        "hindlimbL",
        "hindlimbR",
        *[f"spine{i}" for i in range(1, 6)],
        # Tail
        "tailBase",
        "tailMid",
        "tailTip",
        # DLC example project
        *[f"bodypart{i}" for i in range(1, 4)],
        "objectA",
    ]

    # Dicts, not tuples: `source` has a default, and a dict lets these rows
    # omit it. Positional rows would have to repeat 'curated' 27 times, and a
    # short tuple silently breaks Lookup population when an attribute is added.
    contents = [{"bodypart": bp} for bp in _curated]

    @classmethod
    def canon_map(cls) -> dict:
        """Return ``{normalized: canonical}`` for all body parts.

        This is the single source of truth used to resolve any surface form to
        its canonical spelling. If the curated table itself holds two spellings
        of one concept, raise a clear, admin-actionable error -- the
        inconsistency is in the reference table, not in the caller's input.

        Returns
        -------
        dict
            Mapping from normalized label to canonical (stored) spelling.

        Raises
        ------
        datajoint.errors.DataJointError
            If two ``BodyPart`` rows normalize to the same key.
        """
        names = [str(bp) for bp in cls().fetch("bodypart")]
        try:
            return build_canonical_map(names)
        except dj.DataJointError as err:
            raise dj.DataJointError(
                "The BodyPart reference table is inconsistent: it holds two "
                f"spellings of the same body part. {err} This is a "
                "data-integrity problem in the curated BodyPart table -- not "
                "in your model or config -- so it cannot be fixed by renaming "
                "your bodyparts. An admin must reconcile the duplicate: choose "
                "the canonical spelling, repoint any references to it, then "
                "remove the other entry."
            ) from err

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

    # Dicts, not positional rows: a later column with a default can then be
    # omitted here. Each `hash` is shape_hash_from_edges(bodyparts, edges),
    # matching what insert1 computes.
    contents = [
        {
            "skeleton_id": "1LED",
            "bodyparts": ["whiteLED"],
            "edges": [],
            "hash": "8999452509fdfa98335c5ba44ed05498",
        },
        {
            "skeleton_id": "2LED",
            "bodyparts": ["greenLED", "redLED_C"],
            "edges": [("greenLED", "redLED_C")],
            "hash": "3ba4ffe16dfe637510ed1c3676ec6cb0",
        },
        {
            "skeleton_id": "4LED",
            "bodyparts": [
                "greenLED",
                "redLED_C",
                "redLED_L",
                "redLED_R",
            ],
            "edges": [
                ("greenLED", "redLED_C"),
                ("greenLED", "redLED_L"),
                ("greenLED", "redLED_R"),
            ],
            "hash": "cde6b48ed870286595c1455af7aff8bd",
        },
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

        # `contents` populates the master without the part table, so the
        # default skeletons have no rows until asked for. Same lazy idiom as
        # V1's insert_default accessors.
        if not query and (
            skeleton_id is None or skeleton_id in self._default_bodyparts
        ):
            self._populate_default_bodyparts()

        return list(query.fetch("bodypart"))

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
        # Curated only: a zoo import must not widen what a hand-built lab
        # project may use. `canon_map` deliberately still reads every row, so
        # an imported skeleton can resolve its own parts ("reads lenient,
        # writes strict").
        curated = (BodyPart() & {"source": "curated"}).fetch("bodypart")
        valid = {normalize_label(x) for x in curated}
        missing = {normalize_label(x) for x in labels} - valid
        if missing:
            raise dj.DataJointError(
                f"Unknown bodypart name(s) (not in BodyPart): {sorted(missing)}"
                + "\nPlease either change to existing name or "
                + "consult with admin to add it."
            )
        return True

    # ----------------- main insert -----------------

    def _register_novel_bodyparts(
        self, unresolved: List[str], bodyparts: List[str], accept: bool
    ) -> None:
        """Add incoming body part names that are absent from ``BodyPart``.

        Parameters
        ----------
        unresolved : List[str]
            Incoming names ``canonicalize`` could not resolve.
        bodyparts : List[str]
            The full incoming list, named in the error raised when *accept*
            is False.
        accept : bool
            Register the names rather than raise.

        Raises
        ------
        datajoint.DataJointError
            When *accept* is False, naming the unknown parts; or when two
            unresolved names normalize alike.
        """
        if not accept:
            self._validate_bodyparts({normalize_label(bp) for bp in bodyparts})
            return

        # Check the batch against itself. The flag's safety argument covers
        # collisions with existing rows only: two novel spellings of one
        # concept are both unresolvable, so both would commit and
        # canon_map() would then raise on every later read.
        by_norm: Dict[str, List[str]] = {}
        for bp in unresolved:
            by_norm.setdefault(normalize_label(bp), []).append(bp)
        clashes = {n: names for n, names in by_norm.items() if len(names) > 1}
        if clashes:
            detail = "; ".join(
                f"{names} all normalize to {norm!r}"
                for norm, names in clashes.items()
            )
            raise dj.DataJointError(
                f"Novel bodyparts collide with each other: {detail}. "
                "Use one spelling per concept."
            )

        # Register the incoming spelling itself as canonical so downstream
        # tool output, which uses this spelling, matches.
        self._warn_msg(
            f"Auto-inserting {len(unresolved)} novel bodypart(s) into "
            f"BodyPart: {unresolved}"
        )
        # Plural insert, deliberately: the flag exists so a non-admin can
        # import a model whose parts are novel. See
        # TestAcceptNewBodypartsScope.
        BodyPart().insert(
            [{"bodypart": bp, "source": "imported"} for bp in unresolved],
            skip_duplicates=True,
            allow_direct_insert=True,
        )

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

        # Resolve incoming names to the canonical BodyPart spelling so the
        # stored bodyparts blob, the edges blob, and the part-table FK all
        # share one namespace. (Previously validation used normalized names
        # but the FK insert used raw names, so a variant spelling could pass
        # validation and then fail the FK with an opaque IntegrityError.)
        canon_map = BodyPart.canon_map()
        unresolved = [
            bp for bp in bodyparts if canonicalize(bp, canon_map) is None
        ]
        if unresolved:
            self._register_novel_bodyparts(
                unresolved, bodyparts, accept=accept_new_bodyparts
            )
            canon_map = BodyPart.canon_map()

        bodyparts = [
            canonicalize(bp, canon_map, default=bp) for bp in bodyparts
        ]

        # Structurally clean edges (flatten nested groups, drop empty slots),
        # then canonicalize labels into the same namespace as bodyparts.
        edges = sorted(
            {
                tuple(
                    sorted(
                        (
                            canonicalize(u, canon_map, default=u),
                            canonicalize(v, canon_map, default=v),
                        )
                    )
                )
                for (u, v) in (clean_edges(edges) or [])
            }
        )

        # Validate edge structure against the (canonical) bodypart set.
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

        Fetches the row, then delegates the figure to
        :func:`~spyglass.position.v2.utils.skeleton.draw_skeleton`.

        Parameters
        ----------
        skeleton_id : str, optional
            Skeleton ID to visualize. If None, uses current restriction.
            If multiple skeletons match, shows the first one.

        Returns
        -------
        matplotlib.figure.Figure or None
            The skeleton graph, or None when the skeleton has no bodyparts.

        Examples
        --------
        >>> # Show a specific skeleton
        >>> Skeleton().show_skeleton("4LED")

        >>> # Show skeleton from current restriction
        >>> (Skeleton & {"skeleton_id": "4LED"}).show_skeleton()
        """
        if skeleton_id:
            skeleton_data = (self & {"skeleton_id": skeleton_id}).fetch1()
        else:
            skeleton_data = self.fetch1() if len(self) == 1 else self.fetch()[0]

        skeleton_id = skeleton_data["skeleton_id"]
        bodyparts = skeleton_data["bodyparts"] or self.get_bodyparts(
            skeleton_id
        )

        if not bodyparts:
            self._warn_msg(f"No bodyparts found for skeleton '{skeleton_id}'")
            return None

        return draw_skeleton(
            skeleton_id, bodyparts, skeleton_data["edges"] or []
        )

    def get_centroid_method(self, skeleton_id: str = None) -> str:
        """Suggest appropriate centroid method for this skeleton.

        Fetches the skeleton's body parts, then delegates the choice to
        :func:`~spyglass.position.v2.utils.skeleton.centroid_method_for`.

        Parameters
        ----------
        skeleton_id : str, optional
            Skeleton ID to analyze. If None, uses current restriction.

        Returns
        -------
        str
            Recommended centroid method ('1pt', '2pt', or '4pt')

        Raises
        ------
        ValueError
            If the skeleton names no body parts.

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

        return centroid_method_for(bodyparts)


@schema
class ModelParams(SpyglassMixin, dj.Lookup):
    """Model parameters for training and evaluation."""

    definition = """
    model_params_id: varchar(32)
    tool: varchar(32)
    ---
    params: json
    -> [nullable] Skeleton
    params_hash: varchar(64) # hash of params and skeleton together
    unique index (tool, params_hash)
    """

    # Strategy pattern replaced the tool_info structure
    # Use ToolStrategyFactory to get tool-specific parameters and validation

    # NOTE: ``params`` feeds ``params_hash`` (see ``contents`` below). The DLC
    # default carries the PyTorch-native length knobs (``epochs``/
    # ``save_epochs``, matching DeepLabCut's 3.x defaults) so a fresh run's
    # training budget is explicit rather than implicit. These mirror the subset
    # of ``DLCStrategy.get_default_params`` (the fuller, non-hashed template).
    default_entries_data = [
        {
            "model_params_id": "dlc_default",
            "tool": "DLC",
            "params": {
                "shuffle": 1,
                "trainingsetindex": 0,
                "model_prefix": "",
                "epochs": 200,  # PyTorch engine (DLC 3.x default) length knob
                "save_epochs": 25,  # PyTorch snapshot cadence
            },
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

    @staticmethod
    def get_params_hash(params: dict, skeleton_id: Union[str, None]) -> str:
        """Hash a parameter set together with the skeleton it applies to.

        Feeds ``params_hash``, which ``unique index (tool, params_hash)``
        enforces. A skeleton inside the hash lets that index separate two
        skeletons sharing parameters, including the null-skeleton default rows
        that a nullable index column cannot constrain (MySQL treats NULLs as
        distinct).

        Parameters
        ----------
        params : dict
            Training parameters.
        skeleton_id : str or None
            Skeleton the parameters apply to.

        Returns
        -------
        str
            Hash over both.
        """
        return dj.hash.key_hash({"params": params, "skeleton_id": skeleton_id})

    # A loop, not a comprehension: a comprehension body runs in its own scope
    # and cannot see `get_params_hash`, which is called here at class-creation
    # time.
    contents = []
    for _entry in default_entries_data:
        contents.append(
            (
                _entry["model_params_id"],
                _entry["tool"],
                _entry["params"],
                _entry["skeleton_id"],
                get_params_hash(_entry["params"], _entry["skeleton_id"]),
            )
        )
    del _entry

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

        # Build tool_info for each registered strategy. A registered tool that
        # fails to instantiate or introspect is a real defect (broken/mis-
        # registered strategy), so it must surface loudly — never be silently
        # skipped or collapsed into an empty result.
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
                logger.error(
                    f"Failed to build parameter info for registered tool "
                    f"'{tool_name}'; its strategy is broken or mis-registered."
                )
                raise

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
        params = strategy.append_aliases(params)

        # Validate required parameters using strategy
        strategy.validate_params(params)

        # Check for an existing entry with the same identity. The hash covers
        # the skeleton, so a shared parameter set under a second skeleton is a
        # distinct row rather than a match on the first skeleton's key.
        skeleton_id = key.get("skeleton_id")
        params_hash_dict = dict(
            params_hash=self.get_params_hash(params, skeleton_id)
        )
        if dupe := self & params_hash_dict & dict(tool=key["tool"]):
            dupe_key = dupe.fetch1("KEY")
            self._warn_msg(f"Entry exists with same params: {dupe_key}")
            return dupe_key

        model_params_id: str = (key.get("model_params_id") or "").strip()
        if not model_params_id:
            # Always use default_pk_name for consistent naming. The skeleton
            # reaches the generated id too, so two skeletons sharing params do
            # not collide on the primary key.
            model_params_id = default_pk_name(
                "mp",
                dict(
                    tool=key["tool"],
                    params=params,
                    skeleton_id=skeleton_id,
                ),
            )
        key["model_params_id"] = model_params_id

        # `skeleton_id` is nullable, and _derive_model passes it explicitly as
        # None for the default parameter rows. Only check a non-null value.
        if key.get("skeleton_id") and not (
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

    @classmethod
    def _subjects(cls, key) -> set:
        """Subjects (animals) behind a selection's training video group.

        Parameters
        ----------
        key : dict
            A selection key (or any dict carrying ``vid_group_id``).

        Returns
        -------
        set of str
            The ``Session.subject_id`` values in the group (Nones dropped).
        """
        from spyglass.common import Session
        from spyglass.position.v2.video import VidFileGroup

        vid_group_id = (
            key["vid_group_id"]
            if isinstance(key, dict) and "vid_group_id" in key
            else (cls & key).fetch1("vid_group_id")
        )
        files = VidFileGroup.File & {"vid_group_id": vid_group_id}
        subs = (Session & files).fetch("subject_id")
        return {s for s in subs if s is not None}

    def insert1(self, row, *, allow_redundant_model=False, **kwargs):
        """Insert one selection, guarding against redundant model declaration.

        Blocks declaring a new model whose skeleton already has trained models
        (to promote reuse) unless it is a fine-tune (``parent_id`` set) or
        ``allow_redundant_model=True``. See :meth:`Model.reusable_for`.
        """
        self._guard_redundant_model(row, allow_redundant_model)
        super().insert1(row, **kwargs)

    @staticmethod
    def _skeleton_for_params(params_id, tool=None):
        """Skeleton id for a ModelParams row, or None if unset/missing.

        Parameters
        ----------
        params_id : str
            The ``model_params_id`` to look up.
        tool : str, optional
            Restrict to this tool. ``ModelParams`` is keyed by
            ``(model_params_id, tool)``, so an id alone can match rows for
            another tool with a different skeleton.
        """
        restriction = {"model_params_id": params_id}
        if tool:
            restriction["tool"] = tool
        ids = (ModelParams & restriction).fetch("skeleton_id")
        return ids[0] if len(ids) and ids[0] else None

    @classmethod
    def _own_model_ids(cls, row):
        """model_id(s) already belonging to this exact selection (re-run).

        A named touchpoint rather than an inline fetch: `TestGuardBranches`
        stubs it, alongside `_skeleton_for_params` and `_subjects`, to test
        the guard's decision logic without a database.
        """
        return list((Model & row).fetch("model_id"))

    def _guard_redundant_model(self, row, allow_redundant_model):
        """Raise if *row* would declare a model redundant with existing ones."""
        if allow_redundant_model or not isinstance(row, dict):
            return
        if row.get("parent_id"):  # fine-tune lineage is intentional
            return
        params_id = row.get("model_params_id")
        if not params_id:
            return
        skeleton_id = self._skeleton_for_params(params_id, row.get("tool"))
        if not skeleton_id:  # no skeleton → nothing to match on
            return
        candidates = Model.reusable_for(
            skeleton_id,
            subjects=self._subjects(row),
            exclude=self._own_model_ids(row),
        )
        if not candidates:
            return
        raise ValueError(redundant_model_msg(skeleton_id, candidates))


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
    snapshot_path=NULL : varchar(255)  # this row's own checkpoint/weights file
    model_name=NULL    : varchar(64)   # human-readable; DLC Task, or zoo name
    training_mode='resumable': enum('resumable','weights_only','inference_only')
    evaluation=NULL    : json          # tool-specific evaluation metrics dict
    """

    key_source = ModelSelection  # one entry per selection, ensures unique id

    # ── Reuse discovery (promote reusing models over declaring new ones) ──
    @classmethod
    def with_skeleton(cls, skeleton_id: str):
        """Models whose training selection uses *skeleton_id*.

        Parameters
        ----------
        skeleton_id : str
            Skeleton (body-part set) to match.

        Returns
        -------
        datajoint expression
            The ``Model`` rows sharing that skeleton.
        """
        return cls & (
            ModelSelection & (ModelParams & {"skeleton_id": skeleton_id})
        )

    @classmethod
    def for_subject(cls, subject_id: str):
        """Models trained on a video group containing *subject_id* (an animal).

        Parameters
        ----------
        subject_id : str
            The ``Session.subject_id`` (animal) to match.

        Returns
        -------
        datajoint expression
            The ``Model`` rows trained on that subject.
        """
        from spyglass.common import Session
        from spyglass.position.v2.video import VidFileGroup

        groups = VidFileGroup.File & (Session & {"subject_id": subject_id})
        return cls & (ModelSelection & groups)

    # ── Inference-time model nudge (rank same-skeleton models by score) ──

    @classmethod
    def skeleton_of(cls, model_id: str) -> str:
        """Return the ``skeleton_id`` a model was trained against.

        Parameters
        ----------
        model_id : str
            The model to look up.

        Returns
        -------
        str
            The ``skeleton_id`` from the model's training selection.
        """
        return (
            cls * ModelSelection * ModelParams & {"model_id": model_id}
        ).fetch1("skeleton_id")

    @classmethod
    def rank_for_skeleton(cls, skeleton_id: str) -> list:
        """Rank existing models sharing *skeleton_id*, best-scoring first.

        Fetches the same-skeleton models and delegates ordering to
        :func:`~spyglass.position.v2.utils.model.rank_model_rows`.

        Parameters
        ----------
        skeleton_id : str
            Skeleton (body-part set) to match.

        Returns
        -------
        list[dict]
            Ranked ``{"model_id", "metric", "value", "iterations"}`` entries.
        """
        rows = cls.with_skeleton(skeleton_id).fetch(
            "model_id", "evaluation", as_dict=True
        )
        return rank_model_rows(rows)

    @classmethod
    def reusable_for(
        cls, skeleton_id: str, subjects=None, exclude=None
    ) -> list:
        """Existing models sharing *skeleton_id*, annotated with their subjects.

        Uses one skeleton-restricted join (not a per-model loop), so the large
        ``VideoFile``/``Session`` tables are only touched for the few
        same-skeleton models.

        Parameters
        ----------
        skeleton_id : str
            Skeleton to match. Returns ``[]`` if falsy.
        subjects : iterable of str, optional
            Subjects of the prospective new model, for the ``overlap`` field.
        exclude : iterable of str, optional
            ``model_id`` values to omit (e.g. the selection's own model on a
            re-run).

        Returns
        -------
        list of dict
            ``{"model_id", "subjects", "overlap"}`` per candidate model.
        """
        if not skeleton_id:
            return []
        from spyglass.common import Session
        from spyglass.position.v2.video import VidFileGroup

        same = cls * ModelSelection * ModelParams & {"skeleton_id": skeleton_id}
        ids = set(same.fetch("model_id"))
        rows = (same * VidFileGroup.File * Session).fetch(
            "model_id", "subject_id", as_dict=True
        )
        # Keep models whose group has no resolvable subject.
        for mid in ids:
            rows.append({"model_id": mid, "subject_id": None})
        excl = set(exclude or [])
        return [
            c
            for c in reuse_candidates(rows, subjects)
            if c["model_id"] not in excl
        ]

    def _warn_existing_models_for_videos(self, video_list) -> None:
        """Warn (never raise) if models already exist for these animals.

        A best-effort early nudge for ``create_project``: resolves the
        subjects behind *video_list* and, if any existing model was trained on
        them, suggests reuse. Silent on any lookup failure.
        """
        try:
            from spyglass.common import Session

            subjects = set()
            for item in video_list:
                if isinstance(item, dict) and "nwb_file_name" in item:
                    subjects |= set(
                        (
                            Session & {"nwb_file_name": item["nwb_file_name"]}
                        ).fetch("subject_id")
                    )
            subjects = {s for s in subjects if s is not None}

            existing = set()
            for subject in subjects:
                existing |= set(Model.for_subject(subject).fetch("model_id"))
            if existing:
                self._warn_msg(
                    f"{len(existing)} existing model(s) were trained on the "
                    f"same subject(s) {sorted(subjects)}: {sorted(existing)}. "
                    "Consider reusing an existing model instead of training a "
                    "new one (see Model.reusable_for / Model.for_subject)."
                )
        except Exception as e:
            # Nudge is best-effort; never blocks project creation, but a
            # failed lookup is downgraded to a visible warning, not silenced.
            self._warn_msg(
                f"Could not check for existing models on the same subject(s): "
                f"{e}. Skipping the model-reuse nudge."
            )

    def make_fetch(self, key, device=None):
        """Read the selection, parameters, and video group for training.

        Read-only: the tri-part contract forbids writes here, and DataJoint
        re-runs this method inside the transaction to verify (via ``DeepHash``)
        that upstream data did not change during ``make_compute``.

        When ``parent_id`` is set, the parent model's ``model_path`` and
        ``snapshot_path`` are resolved here and stashed on the returned
        ``sel_entry`` as ``parent_model_path`` and ``parent_snapshot_path``,
        so the strategy's fine-tune path does not have to query the database
        from within ``make_compute``.

        Parameters
        ----------
        key : dict
            Primary key from ModelSelection table containing:
            - model_params_id
            - tool
            - vid_group_id
            - parent_id (optional, for continued training)
        device : str, optional
            Compute device for training (e.g. ``"cuda:0"``, ``"cpu"``).
            This is a *runtime* selection, not a stored/hashed parameter:
            it is injected into a runtime copy of the fetched params and
            never persisted, so it does not fork otherwise-identical
            parameter sets. Pass it via ``populate``'s ``make_kwargs``, e.g.
            ``Model().populate(sel_key, make_kwargs={"device": "cuda:0"})``.
            DataJoint routes ``make_kwargs`` to ``make_fetch`` only, so the
            value is threaded through the return value to
            :meth:`make_compute`. When ``None`` (default), the tool
            auto-selects a device.

        Returns
        -------
        list
            ``[sel_entry, params_entry, vid_group, device]``, passed as
            positional arguments to :meth:`make_compute`.
        """
        sel_entry = (ModelSelection() & key).fetch1()
        params_key = {
            "model_params_id": sel_entry["model_params_id"],
            "tool": sel_entry["tool"],
        }
        params_entry = (ModelParams() & params_key).fetch1()

        vid_group_key = {"vid_group_id": sel_entry["vid_group_id"]}
        vid_group = (VidFileGroup() & vid_group_key).fetch1()

        # Sessions behind the group, for the AnalysisNwbfile parent. Resolved
        # here so training does not run for hours before registration fails,
        # and sorted so the chosen parent does not depend on row order. A
        # group may span sessions; `get_nwb_file` asserts one and so is not
        # usable here.
        vid_group["nwb_file_names"] = sorted(
            set((VidFileGroup.File & vid_group_key).fetch("nwb_file_name"))
        )
        if not vid_group["nwb_file_names"]:
            raise ValueError(
                f"Video group '{sel_entry['vid_group_id']}' resolves no "
                "session, so a trained model has no analysis-file parent. "
                "Add VidFileGroup.File entries for registered videos."
            )

        # Pre-resolve the parent model directory and checkpoint so the
        # fine-tune path in make_compute stays database-free.
        if parent_id := sel_entry.get("parent_id"):
            parent = self.__class__ & {"model_id": parent_id}
            parent_path, parent_snapshot = (
                parent.fetch1("model_path", "snapshot_path")
                if parent
                else (None, None)
            )
            sel_entry["parent_model_path"] = parent_path
            sel_entry["parent_snapshot_path"] = parent_snapshot

        return [sel_entry, params_entry, vid_group, device]

    def make_compute(self, key, sel_entry, params_entry, vid_group, device):
        """Train the model with the tool-specific strategy.

        Pure with respect to the database: no queries or inserts. Writes
        training artifacts (checkpoints, the model-metadata NWB file) to disk,
        which the tri-part contract permits — only database access is barred.

        Parameters
        ----------
        key : dict
            Primary key from ModelSelection table.
        sel_entry : dict
            ModelSelection entry from :meth:`make_fetch`.
        params_entry : dict
            ModelParams entry from :meth:`make_fetch`.
        vid_group : dict
            VidFileGroup entry from :meth:`make_fetch`.
        device : str or None
            Compute device override threaded through from :meth:`make_fetch`.

        Returns
        -------
        list
            ``[model_result]``, passed as a positional argument to
            :meth:`make_insert`.

        Raises
        ------
        NotImplementedError
            If the tool is not supported for training
        ValueError
            If required parameters or data are missing
        """
        self._info_msg(f"Training model for selection: {key}")

        tool = params_entry["tool"]
        params = params_entry["params"]
        skeleton_id = params_entry.get("skeleton_id")
        params_key = {
            "model_params_id": sel_entry["model_params_id"],
            "tool": sel_entry["tool"],
        }

        # Device is a runtime override, not a stored/hashed param: inject it
        # into a copy so the fetched (persisted) params are never mutated.
        if device is not None:
            params = {**(params or {}), "device": device}

        self._info_msg(f"Training {tool} model with params: {params_key}")

        # Dispatch to tool-specific training method using strategy pattern
        strategy = ToolStrategyFactory.create_strategy(tool)

        # Early validation using supports_training for better error messages
        if not strategy.supports_training:
            raise NotImplementedError(
                f"Training not supported for {tool}. "
                f"Use a different tool that supports training."
            )

        # Route continued training (parent_id set) to the fine-tune path,
        # which resumes the parent's weights. Fresh selections train_model.
        parent_id = sel_entry.get("parent_id")
        try:
            if parent_id:
                self._info_msg(
                    f"Continuing training from parent model: {parent_id}"
                )
                model_result = strategy.continue_training(
                    key, params, skeleton_id, vid_group, sel_entry, self
                )
            else:
                model_result = strategy.train_model(
                    key, params, skeleton_id, vid_group, sel_entry, self
                )
        except NotImplementedError as e:
            verb = "Continuation" if parent_id else "Training"
            raise NotImplementedError(
                f"{verb} not implemented for {tool}: {e}"
            ) from e

        return [model_result, vid_group["nwb_file_names"][0]]

    def make_insert(self, key, model_result, analysis_parent):
        """Register the metadata file and insert the trained model row.

        Runs inside a transaction; the only database writes in the tri-part
        sequence. The ``AnalysisNwbfile`` row is created here rather than
        during :meth:`make_compute` so a failed insert leaves no orphan.

        Parameters
        ----------
        key : dict
            Primary key from ModelSelection table.
        model_result : dict
            Row assembled by the tool strategy in :meth:`make_compute`.
        analysis_parent : str
            Session that owns the metadata file, from :meth:`make_fetch`.
        """
        if analysis_file := model_result.get("analysis_file_name"):
            self._register_analysis_file(analysis_file, analysis_parent)
        self.insert1(model_result)
        self._info_msg(f"Model training complete: {model_result['model_id']}")

    def _write_model_metadata(self, metadata: ModelMetadata) -> str:
        """Write an NWB file holding the model's training metadata.

        Writes to disk only. :meth:`make_insert` registers the result in
        ``AnalysisNwbfile``, keeping that row inside the insertion
        transaction per the tri-part contract.

        Parameters
        ----------
        metadata : ModelMetadata
            Consolidated metadata for model registration.

        Returns
        -------
        str
            NWB file name, for the row's ``analysis_file_name``.
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
            # All of them: only the first becomes the FK parent, so the rest
            # would otherwise be unrecorded for a cross-session model.
            "nwb_file_names": list(metadata.session_files),
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

        return nwb_file_name

    def _register_analysis_file(self, analysis_file_name: str, parent: str):
        """Register a written metadata file under its parent session.

        Parameters
        ----------
        analysis_file_name : str
            File written by :meth:`_write_model_metadata`.
        parent : str
            ``nwb_file_name`` of the session that owns it.

        Raises
        ------
        ValueError
            If *parent* is empty.
        """
        if not parent:
            raise ValueError(
                f"No parent session for analysis file {analysis_file_name}. "
                "A model's metadata file belongs to the session(s) its "
                "training videos came from."
            )

        try:
            AnalysisNwbfile().add(parent, analysis_file_name)
            self._info_msg(f"Registered analysis file: {analysis_file_name}")
        except Exception as e:
            # Re-running a selection rewrites the same file name.
            if AnalysisNwbfile() & {"analysis_file_name": analysis_file_name}:
                self._info_msg(
                    f"Analysis file already registered: {analysis_file_name}"
                )
            else:
                raise e

    def train(
        self,
        key: dict,
        *,
        epochs: Union[int, None] = None,
        **overrides,
    ) -> dict:
        """Train a model — either continuing an existing one or a fresh run.

        A single, tool-agnostic, state-driven verb. The path is chosen by
        ``bool(Model & key)``:

        - **Continue / fine-tune** (a ``Model`` row already matches *key*):
          derive a new ``ModelSelection`` with ``parent_id`` set to that
          model, resume its weights, train ``epochs`` more, and return the new
          parent-linked ``Model`` key. ``epochs`` is normalized to each tool's
          native length knob (DLC PyTorch → ``epochs``; DLC TF → ``maxiters``;
          SLEAP → ``max_epochs``). SLEAP resumes from the parent's checkpoint
          (``best.ckpt``/``best_model.h5``) via ``sleap-train
          --base_checkpoint``, rewriting a config copy to carry the epoch
          budget (sleap-train reads length from the config, not params).
        - **Fresh train** (no ``Model`` row yet, but a ``ModelSelection``
          matches *key*): ``populate(key)`` runs ``make`` → ``train_model``;
          return the resulting ``Model`` key.
        - **Neither**: raise ``ValueError``.

        The rule works for either key shape — a ``{'model_id': ...}`` key, or a
        full ``ModelSelection`` key (which resolves a ``Model`` only once it has
        been populated).

        Parameters
        ----------
        key : dict
            A ``Model`` key (e.g. ``{'model_id': ...}``) to continue, or a
            ``ModelSelection`` key to train fresh.
        epochs : int or None, optional
            Additional training length for the continue path (ignored when a
            fresh ``ModelSelection`` only needs populating). Mapped to the
            tool's native knob.
        **overrides
            Parameter overrides applied to the derived ``ModelParams`` on the
            continue path (e.g. ``shuffle``, ``trainingsetindex``). Unknown
            names are warned about and ignored.

        Returns
        -------
        dict
            Primary key for the trained (new) ``Model`` entry.

        Raises
        ------
        ValueError
            If *key* matches neither a ``Model`` nor a ``ModelSelection`` row.
            A key carrying an attribute ``ModelSelection`` does not declare,
            such as an unknown ``model_id``, reports that separately.

        Examples
        --------
        >>> # Continue training an existing model for 50 more epochs
        >>> new_model_key = Model().train({"model_id": "my_dlc_model"},
        ...                               epochs=50)
        >>> # Fine-tune a new shuffle from the same model
        >>> new_model_key = Model().train({"model_id": "my_dlc_model"},
        ...                               shuffle=2)
        >>> # Train a freshly declared selection
        >>> model_key = Model().train(selection_key)
        """
        if self & key:  # a Model row exists → continue / fine-tune
            model_entry = (self & key).fetch1()
            return self._derive_model(
                model_entry, epochs=epochs, overrides=overrides
            )

        # A restriction naming an attribute the table lacks matches every row,
        # so `model_id` must not reach this branch: it would populate every
        # pending selection and then fail on the fetch1 below.
        foreign = set(key) - set(ModelSelection.heading.names)

        if key and not foreign and (ModelSelection() & key):
            if epochs is not None or overrides:
                self._warn_msg(
                    "epochs/overrides are ignored for a fresh ModelSelection; "
                    "bake them into ModelParams before populating."
                )
            self._info_msg(f"Training fresh model for selection: {key}")
            self.populate(key)
            return (self & key).fetch1("KEY")

        if foreign:
            raise ValueError(
                f"No Model matches {key}, and {sorted(foreign)} is not part "
                "of ModelSelection, so this key cannot name a selection to "
                "train fresh. Pass the id of an existing model to continue "
                "training, or a full ModelSelection key to train a new one."
            )

        raise ValueError(
            f"Nothing to train for key: {key}. No matching Model or "
            "ModelSelection row exists. Declare a ModelSelection first, or "
            "pass an existing model_id to continue training."
        )

    def _derive_model(
        self,
        parent_entry: dict,
        *,
        epochs: Union[int, None] = None,
        overrides: Union[dict, None] = None,
    ) -> dict:
        """Derive and train a child model from an existing parent model.

        Shared plumbing for the continue/fine-tune path: builds the child's
        ``ModelParams`` (a new row only if the normalized params actually
        changed, otherwise the parent's is reused to keep content-addressing
        clean), creates a ``ModelSelection`` with ``parent_id`` set — carrying
        any tool-specific selection fields (SLEAP ``training_labels_path``) —
        then populates ``make`` and returns the new ``Model`` key.

        Parameters
        ----------
        parent_entry : dict
            Full parent ``Model`` row (``fetch1()``).
        epochs : int or None, optional
            Additional training length, mapped to the tool's native knob via
            the strategy's ``apply_epochs``.
        overrides : dict, optional
            Validated parameter overrides for the child ``ModelParams``.

        Returns
        -------
        dict
            Primary key for the new (child) ``Model`` entry.
        """
        overrides = dict(overrides or {})
        parent_id = parent_entry["model_id"]
        parent_sel_key = {
            "model_params_id": parent_entry["model_params_id"],
            "tool": parent_entry["tool"],
            "vid_group_id": parent_entry["vid_group_id"],
            "model_selection_id": parent_entry["model_selection_id"],
        }
        params_key = {
            "model_params_id": parent_entry["model_params_id"],
            "tool": parent_entry["tool"],
        }
        params_entry = (ModelParams() & params_key).fetch1()
        tool = params_entry["tool"]
        strategy = ToolStrategyFactory.create_strategy(tool)

        self._info_msg(f"Deriving new training session from model: {parent_id}")

        # Normalize epochs → the tool's native length knob.
        new_params = params_entry["params"].copy()
        new_params = strategy.apply_epochs(
            new_params, epochs, config=strategy.project_config(new_params)
        )

        # Apply validated overrides.
        accepted = ModelParams().get_accepted_params(tool)
        for k, v in overrides.items():
            if k in accepted:
                new_params[k] = v
            else:
                self._warn_msg(f"Ignoring unknown parameter: {k}")

        # run_name names the trainer's output directory, so the child gets its
        # own, keyed on parent_id to match model_selection_id below.
        if new_params.get("run_name") and "run_name" not in overrides:
            new_params["run_name"] = default_pk_name(
                "run", {"parent_id": parent_id}
            )

        # New ModelParams only if the params actually changed.
        if new_params == params_entry["params"]:
            new_params_key = params_key
        else:
            new_params_key = ModelParams().insert1(
                dict(
                    tool=tool,
                    params=new_params,
                    skeleton_id=params_entry.get("skeleton_id"),
                ),
                skip_duplicates=True,
            )

        # New ModelSelection with parent_id, carrying tool-specific fields.
        new_sel_key = dict(
            new_params_key,
            vid_group_id=parent_entry["vid_group_id"],
            parent_id=parent_id,
        )
        labels_path = (ModelSelection() & parent_sel_key).fetch1(
            "training_labels_path"
        )
        if labels_path:  # SLEAP .slp labels — carry forward for populate()
            new_sel_key["training_labels_path"] = labels_path

        new_sel_key["model_selection_id"] = default_pk_name(
            "ms-train", {"parent_id": parent_id}
        )

        self._info_msg(f"Inserting new ModelSelection from parent: {parent_id}")
        ModelSelection().insert1(new_sel_key, skip_duplicates=True)

        self._info_msg("Triggering model training...")
        self.populate(new_sel_key)

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

        Dispatches to the model's tool strategy to compute test/train errors
        and optionally create labeled images, then stores the results on the
        ``Model`` row. DLC writes them to the evaluation-results directory
        within its project.

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
            From the DLC path only, when DeepLabCut is unavailable. A SLEAP
            model evaluates through ``SLEAPStrategy`` without it.
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

        # Imports accept any DLC config suffix, including V1's
        # dj_dlc_config.yaml, so gate on tool and suffix rather than the name.
        if model_entry.get(
            "tool"
        ) != "DLC" or model_path.suffix.lower() not in (
            ".yml",
            ".yaml",
        ):
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
        """Plot training loss curves for a model.

        Fetches the history, then delegates the figure to
        :func:`~spyglass.position.v2.utils.training_plots.build_training_history_figure`.

        Parameters
        ----------
        model_key : dict
            Primary key for Model entry.
        save_path : Union[Path, str, None], optional
            Where to write the figure. Shown instead when None, by default
            None.
        detailed : bool, optional
            If True, renders additional diagnostics including a smoothed
            loss trend and optional validation loss panel, by default False.

        Returns
        -------
        matplotlib.figure.Figure
            The plotted figure.

        Raises
        ------
        ValueError
            If the model has no training history, or none with loss data.

        Examples
        --------
        >>> fig = model.plot_training_history({"model_id": "my_dlc_model"})
        >>> model.plot_training_history(
        ...     {"model_id": "my_dlc_model"}, save_path="history.png"
        ... )
        """
        import matplotlib.pyplot as plt

        df = self.get_training_history(model_key)
        if df is None:
            raise ValueError("No training history found for this model")

        fig = build_training_history_figure(
            df, model_id=model_key["model_id"], detailed=detailed
        )

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
        experimenter: Union[str, None] = None,
        **kwargs,
    ) -> dict:
        """Create a new DLC project from videos in the Spyglass database.

        Creates the DLC project folder on disk, extracts frames for labeling,
        and inserts a ``Skeleton`` entry.  Does **not** insert into
        ``ModelSelection`` or trigger training — label frames externally, then
        ``Model.load(config_path)`` registers the project and returns a
        ``ModelSelection`` key for ``populate`` to train.

        .. note::
            There is no ``Project`` table in V2.  The DLC project folder and
            its ``config.yaml`` are the on-disk record.  ``config.yaml``
            reaches ``ModelParams`` when ``Model.load()`` registers the
            project, before training.

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
        experimenter : str, optional
            Name recorded as the DLC project's ``experimenter`` -- becomes
            the middle component of the folder name DLC creates:
            ``{project_name}-{experimenter}-{date}``. Defaults to the
            current OS user (``getpass.getuser()``) if not given. Must be
            passed here, not via ``**kwargs``: ``deeplabcut.create_new_project``
            already receives an explicit ``experimenter``, so forwarding one
            through ``**kwargs`` collides with it (``TypeError: got multiple
            values for keyword argument 'experimenter'``).
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
            # → label frames externally, then register the project. With no
            # snapshot yet, this returns a ModelSelection key:
            sel_key = Model().load(project_info["config_path"])
            Model().populate(sel_key)  # trains; creates the Model row

        """
        try:
            from deeplabcut import create_new_project, extract_frames
        except ImportError as e:
            raise ImportError(
                "DeepLabCut is required for Model.create_project(). "
                "Install with: pip install deeplabcut"
            ) from e

        from spyglass.position.utils import get_param_names, sanitize_filename
        from spyglass.settings import pose_project_dir, pose_video_dir

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

        # Early nudge: warn (don't block) if models already exist for these
        # animals, before the user invests in labeling. The hard skeleton-based
        # guard lives at model declaration (ModelSelection.insert1).
        self._warn_existing_models_for_videos(video_list)

        # Convert any raw video streams (h264/h265/hevc) to mp4 so DeepLabCut
        # can read them — DLC's frame extraction fails on raw streams. V1 did
        # this at project creation. The original paths (which map to VideoFile)
        # are kept for the VidFileGroup below; DLC gets the converted paths.
        convert_dir = (
            pose_video_dir
            or pose_project_dir
            or str(Path.home() / "dlc_projects")
        )
        Path(convert_dir).mkdir(parents=True, exist_ok=True)
        from spyglass.position.utils.general import ensure_mp4

        dlc_videos = ensure_mp4(resolved_videos, convert_dir)

        # Try to infer a safe frame budget from the shortest video.  If this
        # fails (e.g., codec issues), we keep the user's requested value and
        # rely on DLC's own validation below.
        effective_frames_per_video = frames_per_video
        try:
            import cv2

            frame_counts = []
            for video_path in dlc_videos:
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
        except Exception as e:
            # Non-fatal frame-count optimization; DLC may still read and
            # sample frames. Downgraded to a visible warning, not silenced.
            self._warn_msg(
                f"Could not pre-check video frame counts ({e}); using the "
                f"requested {frames_per_video} frames/video and relying on "
                "DLC's own validation."
            )

        vid_group_key = VidFileGroup.create_from_files(
            video_files=resolved_videos,
            description=f"Training videos: {project_name}",
        )

        # ── 2. Create or update the DLC project (proactive config mgmt) ───────
        # DLCStrategy.ensure_project owns the create-or-update + config edits:
        # it reuses a matching project, creates one if absent, copies in the
        # converted videos, and re-points video_sets/bodyparts/numframes2pick —
        # so we never depend on create_new_project returning a fresh config for
        # an existing folder.
        project_directory = str(
            project_directory
            or pose_project_dir
            or Path.home() / "dlc_projects"
        )
        from spyglass.position.utils.tool_strategies import ToolStrategyFactory

        config_path = ToolStrategyFactory.create_strategy("DLC").ensure_project(
            project_name=project_name,
            project_directory=project_directory,
            videos=dlc_videos,
            bodyparts=bodyparts,
            numframes2pick=effective_frames_per_video,
            sanitize=sanitize_filename,
            experimenter=experimenter or getpass.getuser(),
            create_kwargs=create_kwargs,
            model_instance=self,
        )

        # ── 3. Extract frames for each video that has none ────────────────────
        # Per video, not per project: `ensure_project` reuses a project whose
        # videos are a subset of those requested, so one populated
        # `labeled-data` subdirectory is no evidence for a newly added video.
        # DLC names each subdirectory after the video's stem.
        labeled_data_dir = config_path.parent / "labeled-data"
        project_videos = config_path.parent / "videos"
        pending = [
            video
            for video in dlc_videos
            if not list((labeled_data_dir / Path(video).stem).glob("img*.png"))
        ]

        if not pending:
            self._info_msg("Frames already extracted — skipping.")
        else:
            algo = extract_kwargs["algo"]
            # Name the videos where DLC supports it, so a reused project's
            # already-extracted videos are left alone. `video_sets` holds the
            # copies under the project, which is what DLC matches on.
            if "videos_list" in get_param_names(extract_frames):
                extract_kwargs["videos_list"] = [
                    str(project_videos / Path(video).name) for video in pending
                ]
            elif len(pending) < len(dlc_videos):
                self._warn_msg(
                    f"Installed DeepLabCut's extract_frames takes no "
                    f"`videos_list`, so all {len(dlc_videos)} videos are "
                    f"extracted rather than the {len(pending)} missing."
                )
            self._info_msg(
                "Extracting "
                + f"{effective_frames_per_video} frames/video for "
                + f"{len(pending)} video(s) (algo='{algo}')…"
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

        # ── 4. Insert (or retrieve) Skeleton ──────────────────────────────────
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
        canon_map = BodyPart.canon_map()
        mapping, unresolved = {}, []
        for name in names:
            canonical = canonicalize(name, canon_map)
            if canonical is None:
                unresolved.append(name)
            else:
                mapping[name] = canonical
        return {"mapping": mapping, "unresolved": unresolved}

    def _canonicalize_dlc_project(self, config_path) -> dict:
        """Rewrite a DLC project's ``config.yaml`` names to canonical spelling.

        Opt-in (via ``normalize_names=True`` on :meth:`load`) rewrite of the
        user's project files so their body-part and skeleton names match the
        curated ``BodyPart`` namespace. The original config is copied to a
        timestamped backup (``config.yaml.<timestamp>.bak``) beside it before
        any write, so repeated rewrites never overwrite an earlier backup.
        Names with no canonical match raise -- a rewrite cannot invent a
        curated part. A project holding ``CollectedData_*`` labels raises too,
        since their columns name body parts and are not rewritten here. A
        config that is already fully canonical is left untouched (no backup
        written), labels or not.

        Parameters
        ----------
        config_path : str or pathlib.Path
            Path to the DLC ``config.yaml``.

        Returns
        -------
        dict
            ``{surface_form: canonical}`` for the names that were rewritten
            (empty if the config was already canonical).

        Raises
        ------
        datajoint.DataJointError
            If a name has no canonical match, or if the project holds label
            files that the rewrite would leave inconsistent. Nothing is
            written in either case.
        """
        from spyglass.position.utils.yaml_io import dump_yaml

        config_path = Path(config_path)
        config = load_yaml(config_path)

        resolved = self.canonicalize_bodyparts(config)
        if resolved["unresolved"]:
            raise dj.DataJointError(
                f"Cannot canonicalize DLC project: body part(s) "
                f"{resolved['unresolved']} have no canonical match in "
                "BodyPart. Correct the names or ask an admin to add them."
            )

        mapping = resolved["mapping"]
        changes = {k: v for k, v in mapping.items() if k != v}
        if not changes:
            return {}  # already canonical: no-op, no backup churn

        # Labels carry body-part names as column headers, and nothing here
        # rewrites them, so a rewritten config would disagree with the labels
        # continued training reads. Checked after `changes`, since an already
        # canonical project needs no rewrite and so cannot desync.
        labels = sorted(
            (config_path.parent / "labeled-data").rglob("CollectedData_*")
        )
        if labels:
            shown = "\n".join(
                f"    {p.name} in {p.parent.name}/" for p in labels[:5]
            )
            if len(labels) > 5:
                shown += f"\n    ... and {len(labels) - 5} more"
            raise dj.DataJointError(
                f"Cannot canonicalize DLC project: {len(labels)} label "
                "file(s) hold body-part columns that the rewritten config "
                f"would no longer match:\n{shown}\n\nNothing was written. "
                "Either...\n"
                "1. Import without `normalize_names`. Variant spellings "
                f"({', '.join(sorted(changes))}) resolve to canonical names "
                "in the database regardless, so the project files need no "
                "change.\n"
                "2. To reconcile the project on disk, rename the body-part "
                "columns in the CollectedData CSVs, re-run "
                "`deeplabcut.convertcsv2h5(config_path)`, then import."
            )

        # Timestamped backup so repeated rewrites never overwrite an earlier
        # one (microseconds guarantee a unique name within the same second).
        ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        backup = config_path.with_name(f"{config_path.name}.{ts}.bak")
        shutil.copy2(config_path, backup)

        config["bodyparts"] = [
            mapping.get(bp, bp) for bp in config.get("bodyparts", [])
        ]
        config["skeleton"] = _rename_leaves(config.get("skeleton", []), mapping)
        dump_yaml(config_path, config)
        self._info_msg(
            f"Canonicalized {len(changes)} name(s) in {config_path.name}; "
            f"original backed up to {backup.name}"
        )
        return changes

    def _augment_name_error(self, err, skeleton_config):
        """Return *err* with guidance on how to resolve unknown body parts.

        If every offending name is a variant spelling of an existing part,
        point the user at ``normalize_names=True``; otherwise the names are
        genuinely novel and an admin must add them.
        """
        resolved = self.canonicalize_bodyparts(skeleton_config)
        if resolved["unresolved"]:
            return type(err)(
                f"{err}\n\nBody part(s) {resolved['unresolved']} are not in "
                "BodyPart and have no canonical match; an admin must add them "
                "before this project can be imported."
            )
        return type(err)(
            f"{err}\n\nThese appear to be variant spellings of existing body "
            "parts. Re-run Model().load(..., normalize_names=True) to rewrite "
            "the DLC project to canonical names before import."
        )

    def load(
        self,
        model_path: Union[Path, str],
        tool: str = None,
        skeleton_id: Union[str, None] = None,
        model_params_id: Union[str, None] = None,
        model_id: Union[str, None] = None,
        model_name: Union[str, None] = None,
        normalize_names: bool = False,
        allow_redundant_model: bool = False,
        external_videos: bool = False,
        vid_file_matches: Optional[Dict[str, Optional[Dict]]] = None,
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
            Human-readable label stored on the row. Defaults to the DLC
            ``Task`` for a project import, or the zoo dataset name for a zoo
            import. ``model_id`` is a date+hash, so this is what makes
            ``Model()`` previews legible.
        external_videos : bool, optional
            Import even when the project's training videos are not registered
            in ``VideoFile``, recording them as ``VidFileGroup.ExternalVideo``
            provenance instead. Default False.

            For pretrained models whose training media are not Spyglass
            sessions -- pilot/scoping footage, or public models whose videos are
            unavailable. The resulting group has no session link, so it cannot
            back inference; that guard is unchanged. Reuse discovery by subject
            will not surface the model, since its animals are unknown.
        vid_file_matches : dict, optional
            Explicit ``VideoFile`` keys for config video paths that automatic
            matching cannot resolve, forwarded to
            ``VidFileGroup.create_from_dlc_config``. Keys are paths exactly as
            they appear in the config's ``video_sets``; values are primary-key
            dicts. The ``ValueError`` raised for unresolved paths lists the
            paths needing an entry.
        normalize_names : bool, optional
            For DLC imports, when True rewrite the project's ``config.yaml``
            body-part and skeleton names to their canonical ``BodyPart``
            spelling before import (original copied to a timestamped
            ``config.yaml.<timestamp>.bak`` beside it). Default False. Offered
            as a backup when an import is blocked by variant-spelling names.
        allow_redundant_model : bool, optional
            When False (default), declaring a model whose skeleton already has
            trained models raises with a list of reusable models (to curb model
            proliferation). Set True to declare a new model anyway. Fine-tunes
            (``parent_id`` set) are always exempt. See
            :meth:`Model.reusable_for`.
        **kwargs
            Additional parameters for specific tools. Common parameters:
            - nwb_file_name: Parent NWB file name for linking

        Returns
        -------
        dict
            Primary key for the created Model entry. For a DLC project that
            holds no trained snapshot, a ``ModelSelection`` key instead: there
            is nothing to run inference with, so no ``Model`` row is created.
            Pass that key to ``populate`` to train it. The two are told apart
            by the presence of ``model_id``.

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
        # A string that is not a path may name a DLC Model Zoo entry. Checked
        # before the FileNotFoundError so `load` is one door for "import a
        # pretrained model", whatever its provenance.
        if not Path(model_path).exists():
            from spyglass.position.v2.utils.fetch_dlc_zoo import (
                is_zoo_model,
                zoo_catalog,
            )

            if is_zoo_model(str(model_path)):
                return self.load_from_dlc_zoo(
                    str(model_path),
                    model_params_id=model_params_id,
                    model_id=model_id,
                    allow_redundant_model=allow_redundant_model,
                    **kwargs,
                )
            raise FileNotFoundError(
                f"Model path does not exist, and is not a DLC Model Zoo "
                f"entry: {str(model_path)!r}. "
                f"Zoo models: {sorted(zoo_catalog())}"
            )

        model_path = Path(model_path)

        kwargs.update(
            dict(
                kwargs,
                skeleton_id=skeleton_id,
                model_params_id=model_params_id,
                model_id=model_id,
                model_name=model_name,
                normalize_names=normalize_names,
                allow_redundant_model=allow_redundant_model,
                external_videos=external_videos,
                vid_file_matches=vid_file_matches,
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

        # The project config is shared by every model trained from it. V1's
        # unique index is (task, date, iteration, shuffle, snapshotindex,
        # trainingsetindex), so these are what separate one V1 model from
        # another, and they reach V2's params rather than the config alone.
        extra_params = {
            field: row[field]
            for field in ("shuffle", "trainingsetindex", "snapshotindex")
            if field in row
        }
        if config_template := row.get("config_template"):
            extra_params["config_template"] = config_template
        if model_prefix := row.get("model_prefix"):
            extra_params["model_prefix"] = model_prefix
        kwargs.setdefault("extra_params", {})
        kwargs["extra_params"] = {**extra_params, **kwargs["extra_params"]}

        self._info_msg(
            f"Importing V1 model '{row['dlc_model_name']}' "
            f"from {config_path}"
        )
        return self.load(config_path, **kwargs)

    def load_from_dlc_zoo(
        self,
        dataset: str,
        model_name: Union[str, None] = None,
        model_params_id: Union[str, None] = None,
        model_id: Union[str, None] = None,
        allow_redundant_model: bool = False,
        **kwargs,
    ):
        """Import a DeepLabCut Model Zoo (SuperAnimal) model.

        Reachable as ``Model().load("superanimal_topviewmouse")`` -- ``load``
        dispatches here when the string is not a path.

        Zoo models have no DLC project and no training videos: DLC ships their
        configs with ``video_sets:`` empty, and the weights are public
        checkpoints. So the import records provenance rather than sessions, and
        the model is ``weights_only`` -- DLC fine-tunes a SuperAnimal by
        *seeding a new project* (``build_weight_init``), never by resuming it.

        Parameters
        ----------
        dataset : str
            Zoo dataset, e.g. ``'superanimal_topviewmouse'``.
        model_name : str, optional
            Backbone. Defaults to the first the dataset offers -- these differ
            per dataset, so there is no single valid default.
        model_params_id, model_id : str, optional
            Override the generated ids.
        allow_redundant_model : bool, optional
            Bypass the reuse guard, e.g. to import a second backbone for a
            skeleton that already has models.

        Returns
        -------
        dict
            The inserted ``Model`` key.

        Notes
        -----
        Downloads the weights (a few hundred MB) on first use and copies them
        under ``pose_project_dir``. DLC caches them *inside its own package*,
        which is per conda environment -- storing that path would yield a row
        valid only for the importing user.
        """
        from spyglass.settings import pose_project_dir

        # The zoo is the tool's: the strategy resolves the backbone, fetches
        # the weights and copies them somewhere this database can reach. The
        # table rows below are this method's work.
        zoo = ToolStrategyFactory.create_strategy("DLC").resolve_zoo_model(
            dataset, model_name, Path(pose_project_dir) / "modelzoo"
        )
        backbone, config = zoo["backbone"], zoo["config"]

        # Step 1: Skeleton. Zoo vocabulary is registered as `imported`, so it
        # never widens what a hand-built lab project may use (BodyPart.source).
        skeleton_key = Skeleton().insert1(
            {
                # `or []`, not a get-default: these keys exist in the zoo
                # configs with a null value, and Skeleton.insert1 rejects None
                # edges. SuperAnimal ships bodyparts but no skeleton graph.
                "bodyparts": config.get("bodyparts") or [],
                "edges": config.get("skeleton") or [],
            },
            check_duplicates=True,
            skip_duplicates=True,
            accept_new_bodyparts=True,
        )
        self._info_msg(f"Skeleton: {skeleton_key['skeleton_id']}")

        # Step 2: ModelParams
        model_params_key = ModelParams().insert1(
            dict(
                tool="DLC",
                params={
                    **config,
                    "superanimal_name": dataset,
                    "zoo_backbone": backbone,
                },
                model_params_id=model_params_id,
                skeleton_id=skeleton_key["skeleton_id"],
            ),
            skip_duplicates=True,
        )

        # Step 3: a group with no File rows -- the training videos are not
        # Spyglass sessions and never will be. `get_nwb_file` still refuses it,
        # so this model can never back inference on unregistered data.
        vid_group_key = VidFileGroup().insert1(
            {
                "vid_group_id": default_pk_name("zoo-vg", {"ds": dataset}),
                "description": f"DLC Model Zoo: {dataset}",
                "external_videos": [
                    {
                        "path": None,
                        "source": f"dlc-modelzoo:{dataset}",
                        "note": "training videos not publicly available",
                    }
                ],
            },
            skip_duplicates=True,
        )
        self._info_msg(f"VidFileGroup: {vid_group_key['vid_group_id']}")

        # Step 4: ModelSelection
        sel_key = {**model_params_key, **vid_group_key}
        sel_key.setdefault(
            "model_selection_id",
            default_pk_name("ms-zoo", {"ds": dataset, "bb": backbone}),
        )
        ModelSelection().insert1(
            sel_key,
            skip_duplicates=True,
            allow_redundant_model=allow_redundant_model,
        )

        # Step 5: the weights the strategy placed for us.
        stored_path = _to_stored_path(zoo["weights"])

        existing = self & {"model_path": stored_path}
        if existing:
            key = existing.fetch("KEY", order_by="model_id DESC", limit=1)[0]
            return (self & key).fetch1()

        model_key = {
            **sel_key,
            "model_id": model_id
            or default_pk_name("zoo", {"ds": dataset, "bb": backbone}),
            "model_path": stored_path,
            "model_name": dataset,
            "training_mode": zoo["training_mode"],
        }
        self.insert1(model_key, allow_direct_insert=True)
        self._info_msg(f"Zoo model imported: {model_key['model_id']}")
        return model_key

    def _import_dlc_model(self, model_path: Path, **kwargs):
        # This method is the DLC import path, reached through
        # `DLCStrategy.load`. The strategy answers the project-capability
        # questions below.
        dlc = ToolStrategyFactory.create_strategy("DLC")
        normalize_names = kwargs.pop("normalize_names", False)
        allow_redundant_model = kwargs.pop("allow_redundant_model", False)
        external_videos = kwargs.pop("external_videos", False)
        vid_file_matches = kwargs.pop("vid_file_matches", None)
        extra_params = kwargs.pop("extra_params", None) or {}
        if model_path.suffix not in [".yml", ".yaml"]:
            raise ValueError("DLC model path must be a .yml or .yaml file")

        # Opt-in: rewrite the on-disk project to canonical names first, so the
        # config read below (and any later re-training) uses canonical spelling.
        if normalize_names:
            self._canonicalize_dlc_project(model_path)

        # A DLC project holds several models, distinguished by training
        # choices the config does not record. `extra_params` carries them, so
        # the params hash and the model identity differ per model.
        config = {**load_yaml(model_path), **extra_params}

        # Step 1: Extract and insert skeleton from DLC config
        skeleton_config = {
            "bodyparts": config.get("bodyparts", []),
            "skeleton": config.get("skeleton", []),
        }
        try:
            skeleton_key = Skeleton().insert1(
                skeleton_config,
                check_duplicates=True,
                skip_duplicates=True,
                accept_new_bodyparts=False,  # TODO: discuss with team
            )
        except (PermissionError, dj.DataJointError) as err:
            # Surface actionable guidance: either an admin must add genuinely
            # novel parts, or normalize_names=True can reconcile variant
            # spellings on disk. (When normalize_names was already True the
            # project is canonical, so let the original error propagate.)
            if normalize_names:
                raise
            raise self._augment_name_error(err, skeleton_config) from err
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
        # Raises ValueError if no session matches the DLC config's video paths,
        # unless external_videos=True, which records them as ExternalVideo
        # provenance instead. Register sessions with insert_sessions() first
        # when the training videos *are* Spyglass sessions.
        vid_group_key = VidFileGroup.create_from_dlc_config(
            model_path,
            external_videos=external_videos,
            vid_file_matches=vid_file_matches,
        )
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
        # Return an existing Model entry if one already exists for this path
        # and these training parameters, before declaring the selection.
        # model_selection_id embeds the date, so re-importing the same config
        # on a later day is a new prospective selection that the redundancy
        # guard would reject; matching on the rest of the selection keeps that
        # import idempotent. Matching on `model_path` alone would not: several
        # models share one project's config.yaml, so it would return whichever
        # was imported first. Never assume a single match -- ``fetch1`` on a
        # multi-row set would raise.
        stored_path = _to_stored_path(model_path)
        dated = {"model_selection_id"}
        existing = (
            self
            & {"model_path": stored_path}
            & {k: v for k, v in sel_key.items() if k not in dated}
        )
        if existing:
            key = existing.fetch("KEY", order_by="model_id DESC", limit=1)[0]
            return (self & key).fetch1()

        ModelSelection().insert1(
            sel_key,
            skip_duplicates=True,
            allow_redundant_model=allow_redundant_model,
        )

        # Step 4b: A project with no snapshot has nothing to run inference
        # with, so it gets no Model row. The selection above is what
        # `populate` trains.
        if not dlc.project_has_weights(config, model_path):
            self._info_msg(
                f"Project has no trained weights: {model_path}. Registered "
                "ModelSelection "
                f"{sel_key['model_selection_id']} instead of a Model. Train "
                "it with `Model().populate(<the returned key>)`."
            )
            return sel_key

        # Step 5: Generate model_id and insert directly into Model. The
        # prefix stays short so the identifying hash of (task, model_path)
        # survives the 32-char limit: a longer prefix pushes the hash out and
        # two distinct models sharing a task and date collide.
        task = config.get("Task", "DLCTask")
        model_id = kwargs.get("model_id") or default_pk_name(
            "DLC", dict(tool="DLC", task=task, model_path=str(stored_path))
        )
        model_key = {
            **sel_key,
            "model_id": model_id,
            "model_path": stored_path,
            "model_name": kwargs.get("model_name") or task,
            "training_mode": dlc.project_training_mode(config, model_path),
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

        # Check 2: Model path exists. DLC paths under pose_project_dir are
        # stored relative by _to_stored_path, so they resolve through
        # resolve_model_path; SLEAP output dirs stay cwd-relative.
        model_path = (
            resolve_model_path(str(model_entry["model_path"]))
            if model_entry["tool"] == "DLC"
            else Path(model_entry["model_path"])
        )
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
                else:
                    # Single-point skeletons are supported, including the 1LED
                    # default, so one bodypart is valid and only warns.
                    if len(bodyparts) < 2:
                        warn_list.append(
                            f"Skeleton has only {len(bodyparts)} bodypart(s)"
                        )
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

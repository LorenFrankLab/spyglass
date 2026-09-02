"""Tool strategy pattern for pose estimation model training.

Provides a pluggable architecture for different pose estimation tools
(DLC, SLEAP, etc.) with consistent interfaces and parameter management.
"""

import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Set

import datajoint as dj
import yaml

from .protocols import FileSystemProtocol, RealFileSystem


class PoseToolStrategy(ABC):
    """Abstract base class for pose estimation tool strategies.

    Each tool (DLC, SLEAP, etc.) implements this interface to provide
    consistent parameter validation, training, and model management.
    """

    def __init__(self, filesystem: FileSystemProtocol = None):
        """Initialize with optional filesystem dependency.

        Parameters
        ----------
        filesystem : FileSystemProtocol, optional
            File system implementation. If None, uses RealFileSystem.
        """
        self._fs = filesystem or RealFileSystem()

    @property
    @abstractmethod
    def tool_name(self) -> str:
        """Name of the pose estimation tool."""

    @property
    def source_software(self) -> str:
        """Display name for source_software in ndx-pose NWB objects.

        Override in concrete strategies to return the canonical software name
        (e.g. ``"DeepLabCut"`` rather than ``"DLC"``). Defaults to tool_name.
        """
        return self.tool_name

    @property
    @abstractmethod
    def supports_training(self) -> bool:
        """Whether this tool supports model training.

        Used to avoid Liskov substitution principle violations where
        callers would need to check the specific strategy type to know
        if training methods are available.

        Returns
        -------
        bool
            True if train_model, evaluate_model, verify_model are functional.
        """

    @abstractmethod
    def get_required_params(self) -> Set[str]:
        """Get parameters required for training with this tool."""

    @abstractmethod
    def get_accepted_params(self) -> Set[str]:
        """Get all parameters accepted by this tool."""

    @abstractmethod
    def get_default_params(self) -> Dict[str, Any]:
        """Get default parameter values for this tool."""

    @abstractmethod
    def get_parameter_aliases(self) -> Dict[str, list]:
        """Get parameter aliases (alternative names)."""

    @abstractmethod
    def validate_params(self, params: dict) -> None:
        """Validate parameters for this tool.

        Parameters
        ----------
        params : dict
            Parameters to validate

        Raises
        ------
        ValueError
            If parameters are invalid for this tool
        """

    @abstractmethod
    def get_skipped_params(self) -> Set[str]:
        """Get parameters that Spyglass handles internally (not passed to tool).

        Returns
        -------
        Set[str]
            Set of parameter names that are handled by Spyglass infrastructure
            rather than being passed directly to the tool (e.g. paths, analysis_file_id)
        """

    @abstractmethod
    def train_model(
        self,
        key: dict,
        params: dict,
        skeleton_id: str,
        vid_group: dict,
        sel_entry: dict,
        model_instance,
    ) -> dict:
        """Train a model using this tool's training pipeline.

        Parameters
        ----------
        key : dict
            ModelSelection key
        params : dict
            Training parameters
        skeleton_id : str
            Skeleton ID for the model
        vid_group : dict
            VidFileGroup entry
        sel_entry : dict
            Full ModelSelection entry
        model_instance
            Model table instance for logging/utilities

        Returns
        -------
        dict
            Model table entry with model_id, analysis_file_name, model_path
        """

    @abstractmethod
    def evaluate_model(
        self,
        model_entry: dict,
        params_entry: dict,
        model_instance,
        plotting: bool = True,
        show_errors: bool = True,
        **kwargs,
    ) -> dict:
        """Evaluate a trained model.

        Parameters
        ----------
        model_entry : dict
            Model table entry
        params_entry : dict
            ModelParams entry
        model_instance
            Model table instance for utilities
        plotting : bool
            Whether to generate evaluation plots
        show_errors : bool
            Whether to display error information
        **kwargs
            Additional tool-specific options

        Returns
        -------
        dict
            Evaluation results
        """

    @abstractmethod
    def find_output_files(
        self,
        video_paths: list,
        output_dir: str = "",
        output_file_info: Any = None,
    ) -> list:
        """Find tool output files using tool-specific naming patterns.

        Parameters
        ----------
        video_paths : list
            List of video file paths that were analyzed
        output_dir : str, optional
            User-specified output directory (may be empty)
        output_file_info : Any, optional
            Direct output from inference run (paths or metadata)

        Returns
        -------
        list
            List of output file paths found
        """

    @abstractmethod
    def get_output_file_patterns(self) -> Dict[str, str]:
        """Get tool-specific output file naming patterns.

        Returns
        -------
        Dict[str, str]
            Dictionary mapping pattern types to glob patterns
            e.g., {"primary": "*DLC_*.h5", "fallback": "*.h5"}
        """

    def get_default_output_location(
        self, video_path: str, output_dir: str = ""
    ) -> str:
        """Get default output location for this tool.

        Parameters
        ----------
        video_path : str
            Path to input video
        output_dir : str, optional
            User-specified output directory

        Returns
        -------
        str
            Default output directory path
        """
        # Default behavior: use output_dir if provided, else video directory
        from pathlib import Path

        if output_dir:
            return output_dir
        return str(Path(video_path).parent)

    def load(self, model_path: Path, model_instance, **kwargs) -> dict:
        """Import a pre-trained model (default: not supported).

        Parameters
        ----------
        model_path : Path
            Path to the model files
        model_instance
            Model table instance for utilities
        **kwargs
            Additional import options

        Returns
        -------
        dict
            Model entry information

        Raises
        ------
        NotImplementedError
            If tool doesn't support model import
        """
        raise NotImplementedError(
            f"Model import not implemented for {self.tool_name}"
        )

    def verify_model(
        self, model_path: Path, check_inference: bool = True
    ) -> tuple[dict, list]:
        """Verify model integrity and readiness (default: basic checks).

        Parameters
        ----------
        model_path : Path
            Path to the model files
        check_inference : bool
            Whether to check inference readiness

        Returns
        -------
        tuple[dict, list]
            Checks results dict and warnings list
        """
        checks = {}
        warnings = []

        # Basic existence check (all tools)
        checks["model_exists"] = model_path.exists()
        if not checks["model_exists"]:
            warnings.append(f"Model path does not exist: {model_path}")

        return checks, warnings

    def apply_import_defaults(self, params: dict, model_path: Path) -> dict:
        """Apply tool-specific defaults during model import.

        Parameters
        ----------
        params : dict
            Current parameters
        model_path : Path
            Path to model being imported

        Returns
        -------
        dict
            Parameters with tool-specific defaults applied
        """
        return params

    def get_latest_model_info(self, config: dict) -> dict:
        """Get information about the latest trained model in this project.

        Override in tool-specific strategies to scan the project directory
        and return metadata for the most recently modified trained model.

        Parameters
        ----------
        config : dict
            Tool configuration dictionary (must include ``project_path``)

        Returns
        -------
        dict
            Model metadata, or empty dict if no trained models exist.
        """
        return {}

    def append_aliases(self, params: dict) -> dict:
        """Append parameter aliases to params dictionary.

        Handles bidirectional aliasing:
        - If primary key exists, add aliases
        - If alias exists, add primary key

        Parameters
        ----------
        params : dict
            Original parameters

        Returns
        -------
        dict
            Parameters with aliases added
        """
        aliases = self.get_parameter_aliases()
        expanded_params = params.copy()

        for primary, alias_list in aliases.items():
            # If primary exists, add all aliases
            if primary in params:
                for alias in alias_list:
                    expanded_params[alias] = params[primary]
            # If any alias exists, add primary
            else:
                for alias in alias_list:
                    if alias in params:
                        expanded_params[primary] = params[alias]
                        break

        return expanded_params

    def apply_epochs(self, params: dict, epochs, config: dict = None) -> dict:
        """Map a generic ``epochs`` budget onto this tool's native knob.

        ``epochs`` is the tool-agnostic "how much more training" unit used by
        :meth:`Model.train`. Each strategy translates it to whatever length
        parameter its trainer actually consumes. The base implementation is a
        no-op, returning *params* unchanged.

        Parameters
        ----------
        params : dict
            Training parameters to update (never mutated in place by
            overrides).
        epochs : int or None
            Requested training length. ``None`` leaves *params* unchanged.
        config : dict, optional
            Tool configuration, used by strategies (e.g. DLC) that resolve the
            native knob from an engine setting.

        Returns
        -------
        dict
            Parameters with the native length knob set (or *params* unchanged).
        """
        return params

    def continue_training(
        self,
        key: dict,
        params: dict,
        skeleton_id: str,
        vid_group: dict,
        sel_entry: dict,
        model_instance,
        *,
        epochs=None,
    ) -> dict:
        """Resume training from a parent model's weights (fine-tune).

        Mirrors :meth:`train_model` but wires the parent model's latest
        snapshot into the trainer so training resumes from those weights
        rather than starting fresh. The base implementation is unsupported;
        concrete strategies override where true weight-resume is available.

        Parameters
        ----------
        key : dict
            ModelSelection key for the new (child) model.
        params : dict
            Training parameters from ModelParams.
        skeleton_id : str
            Skeleton ID for this model.
        vid_group : dict
            VidFileGroup entry.
        sel_entry : dict
            Full ModelSelection entry (carries ``parent_id``).
        model_instance
            Model table instance for logging/utilities.
        epochs : int or None, optional
            Additional training length; when ``None`` the native knob already
            baked into *params* is used.

        Returns
        -------
        dict
            Model table entry with model_id, analysis_file_name, model_path.

        Raises
        ------
        NotImplementedError
            If this tool does not support continued training.
        """
        raise NotImplementedError(
            f"Continuation not supported for {self.tool_name}"
        )


class DLCStrategy(PoseToolStrategy):
    """DeepLabCut tool strategy implementation."""

    # Above this, a PyTorch ``epochs`` value is almost certainly a TensorFlow
    # ``maxiters`` value carried across engines (see ``validate_params``).
    _EPOCHS_SANITY_LIMIT = 10_000

    @property
    def tool_name(self) -> str:
        return "DLC"

    @property
    def source_software(self) -> str:
        """Canonical software name used in ndx-pose NWB objects."""
        return "DeepLabCut"

    @property
    def supports_training(self) -> bool:
        return True

    def get_required_params(self) -> Set[str]:
        return {"project_path"}  # project_path required for DLC training

    def get_skipped_params(self) -> Set[str]:
        return {
            "video_sets",
            "model_path",
            "analysis_file_id",
            # Compute-device selection is a runtime concern, not part of a
            # param set's identity — pass it via
            # ``populate(make_kwargs={"device": ...})`` instead. Stripped here so
            # it never enters the content-addressed params hash.
            "device",
            "gputouse",
        }

    def get_accepted_params(self) -> Set[str]:
        return {
            # Core DLC parameters
            "project_path",
            "shuffle",
            "trainingsetindex",
            "maxiters",
            "epochs",  # DLC 3.x PyTorch length knob
            "save_epochs",  # DLC 3.x PyTorch snapshot cadence
            "displayiters",
            "saveiters",
            "net_type",
            "augmenter_type",
            "warmup_epochs",
            "snapshots_epoch",
            "adam_lr",
            "decay_steps",
            "decay_factor",
            "global_scale",
            "location_refinement",
            "locref_stdev",
            "locref_loss_weight",
            "locref_huber_loss",
            "intermediate_supervision",
            "intermediate_supervision_layer",
            "regularize",
            "weight_decay",
            "mirror",
            "crop_pad",
            "scoremap_dir",
            "dataset_type",
            "deterministic",
            "allow_growth",
            "init_weights",
            "multi_step",
            # Project metadata
            "model_prefix",
            "Task",
            "scorer",
            "date",
            "iteration",
            "snapshotindex",
            "batch_size",
            "cropping",
            "x1",
            "x2",
            "y1",
            "y2",
            "corner2move2",
            "move2corner",
            "bodyparts",
            "skeleton",
            "numframes2pick",
            "TrainingFraction",
        }

    def get_default_params(self) -> Dict[str, Any]:
        # DLC 3.x defaults to the PyTorch engine, so the PyTorch-native length
        # knobs (``epochs``/``save_epochs``) are the primary defaults here.
        # ``epochs=200``/``save_epochs=25`` mirror DeepLabCut's own PyTorch
        # defaults -- a good fit for the small (~100-frame) Frank Lab training
        # sets. Do NOT carry a TensorFlow ``maxiters`` value (the old TF default
        # was 1,030,000 *iterations*) into ``epochs``: one epoch is a full pass
        # over the training set (many gradient steps), so ~1e6 epochs is a
        # runaway. The TensorFlow knobs below are retained only for models
        # explicitly trained with ``engine: tensorflow`` (see ``apply_epochs``).
        #
        # This is the fuller informational template; the hashed seed row is
        # the ``dlc_default`` subset in ``ModelParams.default_entries_data``.
        # Both share ``shuffle``/``trainingsetindex``/``model_prefix``/
        # ``epochs``/``save_epochs``; only this template adds the TF knobs and
        # ``net_type``.
        return {
            "shuffle": 1,
            "trainingsetindex": 0,
            "epochs": 200,  # PyTorch engine (DLC 3.x default) length knob
            "save_epochs": 25,  # PyTorch snapshot cadence
            "maxiters": None,  # TensorFlow engine only; None => DLC default
            "displayiters": None,  # TensorFlow engine only
            "saveiters": None,  # TensorFlow engine only
            "model_prefix": "",
            "net_type": "resnet_50",
        }

    def get_parameter_aliases(self) -> Dict[str, list]:
        return {
            "model_type": ["approach"],
            "backbone": ["backbone_type", "net_type"],
            "shuffle": ["shuffle_idx"],
            "maxiters": ["max_iters", "training_iterations"],
        }

    def validate_params(self, params: dict) -> None:
        """Validate DLC-specific parameters."""
        required = self.get_required_params()
        missing = [k for k in required if k not in params]
        if missing:
            raise ValueError(
                f"DLC training missing required parameters: {missing}"
            )

        # Validate numeric parameters
        for param, param_type in [("shuffle", int), ("trainingsetindex", int)]:
            if param in params and not isinstance(params[param], param_type):
                raise ValueError(
                    f"DLC parameter '{param}' must be {param_type.__name__}"
                )

        # Guardrail for the TF->PyTorch migration trap: ``epochs`` is a full
        # pass over the training set (many gradient steps), so a value in the
        # TF-``maxiters`` range (the old TF default was 1,030,000 *iterations*)
        # is almost certainly a knob carried across engines. Warn, do not fail
        # -- a user may legitimately want a long run.
        epochs = params.get("epochs")
        if isinstance(epochs, (int, float)) and not isinstance(epochs, bool):
            if epochs > self._EPOCHS_SANITY_LIMIT:
                warnings.warn(
                    f"DLC 'epochs' is {int(epochs)}, which is implausibly "
                    "large for the PyTorch engine (one epoch is a full pass "
                    "over the training set, not a single iteration). This "
                    "looks like a TensorFlow 'maxiters' value carried into "
                    "'epochs'; ~200 epochs is typical. Set 'maxiters' instead "
                    "for a TensorFlow-engine model.",
                    stacklevel=2,
                )

    @staticmethod
    def _is_tf_engine(config: dict) -> bool:
        """Whether the project config selects the TensorFlow engine.

        DLC 3.x defaults to the PyTorch engine
        (``deeplabcut.compat.DEFAULT_ENGINE = Engine.PYTORCH``); an explicit
        ``engine: tensorflow`` (or legacy ``tf``) opts back into TensorFlow.

        Parameters
        ----------
        config : dict
            DLC project configuration.

        Returns
        -------
        bool
            True for the TensorFlow engine, False for PyTorch (the default).
        """
        engine = str((config or {}).get("engine", "pytorch")).lower()
        return "tensorflow" in engine or engine == "tf"

    def apply_epochs(self, params: dict, epochs, config: dict = None) -> dict:
        """Map ``epochs`` onto DLC's engine-specific length knob.

        Resolves the engine from ``config['engine']``: the PyTorch engine
        (DLC 3.x default) consumes ``epochs``; the TensorFlow engine consumes
        ``maxiters``.

        Parameters
        ----------
        params : dict
            Training parameters.
        epochs : int or None
            Requested training length; ``None`` returns *params* unchanged.
        config : dict, optional
            DLC project configuration (for engine resolution).

        Returns
        -------
        dict
            Copy of *params* with the resolved native knob set.
        """
        if epochs is None:
            return params
        params = dict(params)
        if self._is_tf_engine(config):
            params["maxiters"] = int(epochs)
        else:  # PyTorch is the DLC 3.x default
            params["epochs"] = int(epochs)
        return params

    @staticmethod
    def _build_resume_kwargs(*, snapshot_path, epochs=None) -> dict:
        """Build the PyTorch trainer kwargs that resume from a parent snapshot.

        DLC 3.x PyTorch resumes from a checkpoint via ``snapshot_path`` (and
        optionally ``epochs``). TensorFlow-engine weight-resume is not
        supported and is rejected earlier in :meth:`continue_training`
        (``init_weights`` is not a ``train_network`` kwarg, so it would be
        silently dropped), so this helper is PyTorch-only.

        Parameters
        ----------
        snapshot_path : str or Path
            Path to the parent model's latest snapshot.
        epochs : int or None, optional
            Additional training length. When ``None``, only the weight path is
            set (length comes from the params already baked by ``apply_epochs``).

        Returns
        -------
        dict
            Keyword arguments to merge into the training parameters.
        """
        kwargs = {"snapshot_path": str(snapshot_path)}
        if epochs is not None:
            kwargs["epochs"] = int(epochs)
        return kwargs

    def _resolve_parent_snapshot(self, config: dict, *, shuffle=None):
        """Locate the parent project's latest training snapshot.

        Uses :meth:`get_latest_model_info` to find the most recent train
        directory, then returns its newest PyTorch snapshot file
        (``snapshot-*.pt``). Returns ``None`` when no resumable snapshot is
        available — e.g. the requested ``shuffle`` differs from the trained
        one, so weight-resume is invalid and the caller should degrade to a
        fresh (parent-linked) train.

        Parameters
        ----------
        config : dict
            DLC project configuration (must include ``project_path``).
        shuffle : int, optional
            Requested shuffle; if it differs from the latest trained model's
            shuffle, no snapshot is returned.

        Returns
        -------
        Path or None
            Newest snapshot path, or ``None`` if none is resumable.
        """
        info = self.get_latest_model_info(config)
        if not info:
            return None
        if shuffle is not None and info.get("shuffle") != shuffle:
            return None  # shuffle changed → weights can't be reused
        train_dir = Path(info["path"])
        # PyTorch-only: TF-engine continuation is rejected in
        # ``continue_training`` before this is ever reached, so a ``*.index``
        # (TensorFlow) fallback here would be dead code.
        snaps = sorted(train_dir.glob("snapshot-*.pt"))
        if not snaps:
            return None
        return max(snaps, key=lambda p: self._fs.getmtime(p))

    def continue_training(
        self,
        key: dict,
        params: dict,
        skeleton_id: str,
        vid_group: dict,
        sel_entry: dict,
        model_instance,
        *,
        epochs=None,
    ) -> dict:
        """Resume DLC training from the parent model's latest snapshot.

        Resolves the parent snapshot from the project and wires it into the
        PyTorch trainer via ``snapshot_path`` so training continues from those
        weights. When no resumable snapshot is available (missing snapshot, or
        a ``shuffle`` override that invalidates weight-resume) this degrades to
        a fresh — but still ``parent_id`` linked — train, keeping the lineage
        while starting weights from scratch. The caller never has to decide
        this.

        The TensorFlow engine is not supported for continuation: its resume
        mechanism (``init_weights`` in ``pose_cfg``) is not a ``train_network``
        kwarg, so it cannot be wired here without risk of a silent no-op. A
        clear ``NotImplementedError`` is raised instead.

        See :meth:`PoseToolStrategy.continue_training` for parameters.

        Raises
        ------
        NotImplementedError
            If the project uses the DLC TensorFlow engine.
        """
        config = self._fs.read_yaml(
            Path(params["project_path"]) / "config.yaml"
        )
        if self._is_tf_engine(config):
            raise NotImplementedError(
                "DLC TensorFlow-engine continuation (weight-resume) is not "
                "implemented: `init_weights` is not a `train_network` kwarg "
                "and would be silently dropped. Use the PyTorch engine (the "
                "DLC 3.x default) to continue training, or retrain fresh."
            )
        shuffle = int(params.get("shuffle", config.get("shuffle", 1)))
        snapshot = self._resolve_parent_snapshot(config, shuffle=shuffle)

        if snapshot is None:
            model_instance._warn_msg(
                "No resumable parent snapshot found (or shuffle changed); "
                "training fresh but keeping the parent lineage."
            )
            return self.train_model(
                key, params, skeleton_id, vid_group, sel_entry, model_instance
            )

        # epochs: explicit kwarg wins; else the length knob already baked into
        # params by apply_epochs (Model.train's epochs= arrives here as
        # params['epochs'], since make() does not thread the kwarg through).
        # Mirrors SLEAPStrategy.continue_training's params.get("max_epochs").
        eff_epochs = epochs if epochs is not None else params.get("epochs")
        resume = self._build_resume_kwargs(
            snapshot_path=snapshot, epochs=eff_epochs
        )
        model_instance._info_msg(
            f"Resuming DLC training from parent snapshot: {snapshot}"
        )
        aug_params = dict(params, **resume)
        return self.train_model(
            key, aug_params, skeleton_id, vid_group, sel_entry, model_instance
        )

    def train_model(
        self,
        key: dict,
        params: dict,
        skeleton_id: str,
        vid_group: dict,
        sel_entry: dict,
        model_instance,
    ) -> dict:
        """Train DLC model with tool-specific logic.

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
        model_instance
            Model table instance for logging/utilities

        Returns
        -------
        dict
            Model table entry with model_id, model_path, analysis_file_name
        """
        from pathlib import Path

        import yaml

        # Import DLC functions (defer to avoid startup dependency)
        try:
            from deeplabcut import create_training_dataset, train_network
        except ImportError:
            raise ImportError(
                "DeepLabCut is required for training. "
                "Install with: pip install deeplabcut>=3.0"
            )

        # Import utility functions from Model class
        from spyglass.position.utils import suppress_print_from_package
        from spyglass.position.v2.train import (
            ModelMetadata,
            _to_stored_path,
            default_pk_name,
        )

        # Validate project configuration
        if "project_path" not in params:
            raise ValueError(
                "DLC training requires 'project_path' in ModelParams. "
                "Please specify the DLC project directory."
            )

        project_path = Path(params["project_path"])
        if not self._fs.exists(project_path):
            raise FileNotFoundError(f"DLC project not found: {project_path}")

        config_path = project_path / "config.yaml"
        if not self._fs.exists(config_path):
            raise FileNotFoundError(
                f"DLC config.yaml not found in: {project_path}"
            )

        model_instance._info_msg(f"Using DLC project: {project_path}")

        # Load config
        from spyglass.position.utils.yaml_io import load_yaml

        config = self._fs.read_yaml(config_path)

        # Check for continued training
        parent_id = sel_entry.get("parent_id")
        if parent_id:
            model_instance._info_msg(
                f"Continuing training from parent model: {parent_id}"
            )

        # Execute training pipeline
        self._prepare_training_dataset(
            config_path, params, config, model_instance
        )
        self._execute_training(
            config_path, params, model_instance, config=config
        )

        # Localize and register trained model. Pass the selection *key* so the
        # generated model_id is unique per selection: default_pk_name("mdl")
        # with no params is deterministic per day, which collides when two
        # models are trained the same day (e.g. a parent and its continuation).
        model_path, model_id = self._localize_trained_model(
            config, model_instance, key=key
        )
        latest_model = self.get_latest_model_info(config)

        nwb_file_name = model_instance._register_model_metadata(
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

    def ensure_project(
        self,
        *,
        project_name: str,
        project_directory: str,
        videos: list,
        bodyparts: list,
        numframes2pick: int,
        sanitize,
        experimenter: str,
        create_kwargs: dict = None,
        model_instance=None,
    ) -> Path:
        """Create or update a DLC training project; return its config path.

        Proactive **create-or-update**: finds an existing project by name,
        creates one if absent, copies in any converted video DLC does not yet
        reference, and rewrites ``video_sets`` / ``bodyparts`` /
        ``numframes2pick`` to the requested values. This never relies on
        DeepLabCut's ``create_new_project`` returning a fresh config for an
        existing folder — it returns the existing (possibly stale) config
        unchanged, which is why a project first built from raw h264 otherwise
        keeps pointing at the h264 after conversion.

        Parameters
        ----------
        project_name : str
            DLC project name (folder prefix).
        project_directory : str
            Directory under which the ``{project_name}-*-*`` folder lives.
        videos : list
            The (converted) video paths the project should reference.
        bodyparts : list
            Body-part names to write into the config.
        numframes2pick : int
            Frames-per-video budget for extraction.
        sanitize : callable
            Filesystem-safety sanitizer applied to *experimenter* (matches
            the folder naming used at creation time).
        experimenter : str
            Name recorded as the project's experimenter -- the middle
            component of the ``{project_name}-{experimenter}-{date}`` folder
            DLC creates. Must be a real identifier (e.g. the calling user),
            not derived from *project_name* -- doing so used to duplicate
            the project name into the folder (see issue #1676).
        create_kwargs : dict, optional
            Extra keyword args forwarded to ``create_new_project``.
        model_instance : optional
            Model table instance, used only for ``_info_msg`` logging.

        Returns
        -------
        pathlib.Path
            Path to the reconciled ``config.yaml``.
        """
        from deeplabcut import add_new_videos, create_new_project

        from spyglass.position.utils.dlc_config import DlcConfig

        create_kwargs = create_kwargs or {}
        want = {Path(v).name for v in videos}

        def _log(msg):
            if model_instance is not None:
                model_instance._info_msg(msg)

        # Reuse an existing project whose video_sets are a subset of the
        # requested (converted) videos — DLC may silently drop invalid files,
        # so a subset (not equality) avoids false negatives.
        config_path = None
        for candidate in sorted(
            Path(project_directory).glob(f"{project_name}-*-*/config.yaml")
        ):
            try:
                existing = DlcConfig.read(candidate.parent).video_names()
            except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError):
                # Unreadable / malformed candidate config → skip it (visibly),
                # then try the next one. Unexpected errors still propagate.
                _log(f"Skipping unreadable DLC project candidate: {candidate}")
                continue
            if existing and existing.issubset(want):
                config_path = candidate.resolve()
                _log(f"Reusing existing DLC project: {config_path}")
                break

        if config_path is None:
            config_path = Path(
                create_new_project(
                    project=project_name,
                    experimenter=sanitize(experimenter),
                    videos=list(videos),
                    working_directory=project_directory,
                    copy_videos=True,
                    multianimal=False,
                    **create_kwargs,
                )
            ).resolve()
            _log(f"DLC project created: {config_path}")

        project_dir = config_path.parent
        videos_dir = project_dir / "videos"
        cfg = DlcConfig.read(project_dir)

        # Copy in converted videos DLC does not already reference on disk.
        have = cfg.video_names()
        to_add = [
            v
            for v in videos
            if Path(v).name not in have
            or not (videos_dir / Path(v).name).exists()
        ]
        if to_add:
            add_new_videos(
                config=str(config_path), videos=to_add, copy_videos=True
            )
            cfg = DlcConfig.read(project_dir)

        # Re-point video_sets at exactly the converted videos and set params.
        cfg.keep_videos(want).set_bodyparts(bodyparts).set(
            "numframes2pick", numframes2pick
        )
        return Path(cfg.write())

    def evaluate_model(
        self,
        model_entry: dict,
        params_entry: dict,
        model_instance,
        plotting: bool = True,
        show_errors: bool = True,
        **kwargs,
    ) -> dict:
        """Evaluate DLC model using existing _evaluate_dlc_model method."""
        return model_instance._evaluate_dlc_model(
            model_entry, params_entry, plotting, show_errors, **kwargs
        )

    def find_output_files(
        self,
        video_paths: list,
        output_dir: str = "",
        output_file_info: Any = None,
    ) -> list:
        """Find DLC output files using DLC naming conventions.

        DLC saves files as: {video_stem}DLC_{network}_{shuffle}_{snapshot}.h5
        If no destfolder specified, saves to video directory.
        """
        # If we got direct file info from inference, use it
        if output_file_info:
            if isinstance(output_file_info, str):
                return [output_file_info]
            return output_file_info

        from pathlib import Path

        output_files = []

        for video_path in video_paths:
            video_path = Path(video_path)

            # Determine search directory using tool's default logic
            search_dir = Path(
                self.get_default_output_location(str(video_path), output_dir)
            )

            if not self._fs.exists(search_dir):
                continue

            # Look for DLC output files for this video
            patterns = self.get_output_file_patterns()
            dlc_pattern = patterns["primary"].format(video_stem=video_path.stem)
            # Convert to full path pattern for filesystem glob
            full_pattern = str(search_dir / dlc_pattern)
            matching_files = [Path(p) for p in self._fs.glob(full_pattern)]

            if matching_files:
                # Use most recent if multiple files
                latest_file = max(
                    matching_files, key=lambda p: p.stat().st_mtime
                )
                output_files.append(str(latest_file))
            else:
                # Fallback: look for any h5 files
                fallback_pattern = patterns["fallback"]
                full_fallback_pattern = str(search_dir / fallback_pattern)
                h5_files = [
                    Path(p) for p in self._fs.glob(full_fallback_pattern)
                ]
                if h5_files:
                    latest_h5 = max(h5_files, key=lambda p: p.stat().st_mtime)
                    output_files.append(str(latest_h5))

        return output_files

    def get_output_file_patterns(self) -> Dict[str, str]:
        """Get DLC-specific output file naming patterns."""
        return {
            "primary": "{video_stem}DLC_*.h5",  # Standard DLC output pattern
            "fallback": "*.h5",  # Any H5 file as last resort
        }

    def load(self, model_path: Path, model_instance, **kwargs) -> dict:
        """Import DLC model using existing _import_dlc_model method."""
        return model_instance._import_dlc_model(model_path, **kwargs)

    def verify_model(
        self, model_path: Path, check_inference: bool = True
    ) -> tuple[dict, list]:
        """Verify DLC model integrity and readiness."""
        checks, warnings = super().verify_model(model_path, check_inference)

        if not checks["model_exists"]:
            return checks, warnings

        # DLC-specific checks
        if check_inference:
            try:
                # Import DLC dynamically to avoid circular dependencies
                from deeplabcut import create_training_dataset, train_network

                # Verify model directory structure for trained models
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
                else:
                    warnings.append(
                        "DLC model path should be a .yaml/.yml file"
                    )
            except ImportError:
                warnings.append(
                    "DeepLabCut not installed - cannot verify inference readiness"
                )

        return checks, warnings

    def apply_import_defaults(self, params: dict, model_path: Path) -> dict:
        """Apply DLC-specific defaults during import."""
        result_params = params.copy()

        # Add default project_path if importing from DLC and not provided
        if "project_path" not in result_params:
            result_params["project_path"] = str(model_path.parent)

        return result_params

    def get_latest_model_info(self, config: dict) -> dict:
        """Get latest trained DLC model information from project directory.

        Discovers trained models in the DLC project's dlc-models directory
        structure. Returns information about the most recently modified model,
        or empty dict if no trained models are found.

        Parameters
        ----------
        config : dict
            DLC configuration dictionary containing 'project_path' key

        Returns
        -------
        dict
            Dictionary with keys:
            - path : str - Path to the model's train directory
            - iteration : int - Model iteration number
            - trainFraction : float - Training fraction (0.0-1.0)
            - shuffle : int - Shuffle number
            - date_trained : datetime - Date model was last modified

            Returns empty dict {} if no trained models exist.

        Raises
        ------
        FileNotFoundError
            If the project_path does not exist
        """
        import re
        from datetime import datetime
        from pathlib import Path

        from spyglass.position.utils.yaml_io import load_yaml

        # Validate project exists
        project_path = Path(config["project_path"])
        if not self._fs.exists(project_path):
            raise FileNotFoundError(f"DLC project not found: {project_path}")

        # DLC 3.x PyTorch saves to dlc-models-pytorch/; TF saves to dlc-models/.
        # Search both; prefer PyTorch if found.
        iteration_dirs = []
        for models_subdir in ("dlc-models-pytorch", "dlc-models"):
            dlc_models_dir = project_path / models_subdir
            if self._fs.exists(dlc_models_dir):
                iteration_dirs = sorted(dlc_models_dir.glob("iteration-*"))
                if iteration_dirs:
                    break
        if not iteration_dirs:
            return {}

        # Collect all model directories with their metadata
        models_found = []

        for iter_dir in iteration_dirs:
            # Extract iteration number from directory name
            iter_match = re.search(r"iteration-(\d+)", iter_dir.name)
            if not iter_match:
                continue
            iteration = int(iter_match.group(1))

            # Find model folders (e.g., TESTv2-Nov12-trainset80shuffle1)
            # Pattern: TASK-trainsetXshuffleY
            model_dirs = list(iter_dir.glob("*trainset*shuffle*"))

            for model_dir in model_dirs:
                # Extract trainFraction and shuffle from directory name
                # e.g., "TESTv2-Nov12-trainset80shuffle1" -> trainset=80%, shuffle=1
                shuffle_match = re.search(r"shuffle(\d+)", model_dir.name)
                trainset_match = re.search(r"trainset(\d+)", model_dir.name)

                if not (shuffle_match and trainset_match):
                    continue

                shuffle = int(shuffle_match.group(1))
                trainset_pct = int(trainset_match.group(1))
                train_fraction = trainset_pct / 100.0

                # Check for train directory containing a trainer config.
                # DLC 3.x PyTorch writes ``pytorch_config.yaml`` here; the
                # TensorFlow engine writes ``pose_cfg.yaml``. Accept either so
                # a real PyTorch-trained model is discoverable (otherwise
                # weight-resume degrades to a fresh train).
                train_dir = model_dir / "train"
                if not train_dir.exists():
                    continue

                marker = next(
                    (
                        train_dir / name
                        for name in ("pytorch_config.yaml", "pose_cfg.yaml")
                        if (train_dir / name).exists()
                    ),
                    None,
                )
                if marker is None:
                    continue

                # Modification time of the trainer config as the training date
                mtime = marker.stat().st_mtime
                date_trained = datetime.fromtimestamp(mtime)

                models_found.append(
                    {
                        "path": str(train_dir),
                        "iteration": iteration,
                        "trainFraction": train_fraction,
                        "shuffle": shuffle,
                        "date_trained": date_trained,
                        "mtime": mtime,  # For sorting
                    }
                )

        if not models_found:
            return {}

        # Return the most recently modified model
        latest = max(models_found, key=lambda x: x["mtime"])

        # Remove the temporary sorting key
        result = {
            "path": latest["path"],
            "iteration": latest["iteration"],
            "trainFraction": latest["trainFraction"],
            "shuffle": latest["shuffle"],
            "date_trained": latest["date_trained"],
        }

        return result

    def _prepare_training_dataset(
        self, config_path: Path, params: dict, config: dict, model_instance
    ) -> None:
        """Prepare DLC training dataset using create_training_dataset."""
        from spyglass.position.utils import get_param_names, test_mode_suppress

        # Check DLC version for Engine enum support
        try:
            from deeplabcut.core.engine import Engine as _Engine

            _dlc3 = True
        except ImportError:
            _dlc3 = False

        def _to_engine(val):
            """Convert string engine name to Engine enum for DLC 3.x."""
            if _dlc3 and isinstance(val, str):
                return _Engine(val)
            return val

        # Import create_training_dataset with deferred import
        from deeplabcut import create_training_dataset

        # Filter parameters to only those accepted by create_training_dataset
        training_dataset_kwargs = {
            k: v
            for k, v in params.items()
            if k in get_param_names(create_training_dataset)
        }

        # Convert engine parameter for DLC 3.x
        if "engine" in training_dataset_kwargs:
            training_dataset_kwargs["engine"] = _to_engine(
                training_dataset_kwargs["engine"]
            )

        # Always suppress interactive prompts; DLC's default userfeedback=True
        # blocks on input() when the model folder already exists.
        training_dataset_kwargs.setdefault("userfeedback", False)

        model_instance._info_msg("Creating DLC training dataset...")

        with test_mode_suppress():
            create_training_dataset(str(config_path), **training_dataset_kwargs)

    def _execute_training(
        self, config_path: Path, params: dict, model_instance, config=None
    ) -> None:
        """Execute DLC model training using train_network.

        Parameters
        ----------
        config_path : Path
            Path to the DLC project ``config.yaml``.
        params : dict
            Training parameters from ModelParams.
        model_instance
            Model table instance for logging.
        config : dict, optional
            The already-loaded DLC project configuration. Used only to resolve
            the engine (PyTorch vs TensorFlow) for GPU-selection routing. When
            ``None`` (e.g. direct unit-test callers) the engine defaults to
            PyTorch, so ``gputouse`` routing stays active.
        """
        from spyglass.position.utils import (
            get_param_names,
            suppress_print_from_package,
            test_mode_suppress,
        )

        # Check DLC version for Engine enum support
        try:
            from deeplabcut.core.engine import Engine as _Engine

            _dlc3 = True
        except ImportError:
            _dlc3 = False

        def _to_engine(val):
            """Convert string engine name to Engine enum for DLC 3.x."""
            if _dlc3 and isinstance(val, str):
                return _Engine(val)
            return val

        # Import train_network with deferred import
        from deeplabcut import train_network

        # Filter parameters to only those accepted by train_network
        train_network_kwargs = {
            k: v
            for k, v in params.items()
            if k in get_param_names(train_network)
        }

        # Convert engine parameter for DLC 3.x
        if "engine" in train_network_kwargs:
            train_network_kwargs["engine"] = _to_engine(
                train_network_kwargs["engine"]
            )

        # Convert string parameters to integers
        for k in ["shuffle", "trainingsetindex", "maxiters"]:
            if value := train_network_kwargs.get(k):
                train_network_kwargs[k] = int(value)

        # Test mode adjustments
        test_mode = params.get("test_mode", False)
        if test_mode:
            train_network_kwargs["maxiters"] = 2
            if _dlc3:  # DLC 3.x PyTorch uses epochs instead of maxiters
                train_network_kwargs.setdefault("epochs", 1)
                train_network_kwargs.setdefault("save_epochs", 1)

        # GPU selection: route a legacy `gputouse` to the v2 `device` selector,
        # but ONLY for the PyTorch engine. `train_network` accepts `gputouse`
        # in its signature; the PyTorch engine ignores it (it selects the GPU
        # via `device`), so routing corrects a silent no-op there. The
        # TensorFlow engine, however, DOES honor `gputouse` — popping it and
        # setting `device` (which TF ignores) would silently change GPU
        # selection — so leave TF-engine kwargs untouched.
        from spyglass.position.utils.dlc_io import route_gputouse_to_device

        if not self._is_tf_engine(config):
            route_gputouse_to_device(
                params,
                train_network_kwargs,
                model_instance._warn_msg,
                context="training",
            )

        model_instance._info_msg("Starting DLC model training...")

        try:
            with suppress_print_from_package():
                train_network(str(config_path), **train_network_kwargs)
        except KeyboardInterrupt:  # pragma: no cover
            model_instance._info_msg(
                "DLC training stopped via Keyboard Interrupt"
            )
        except RuntimeError as e:
            msg = str(e)
            # TF1 signals end-of-training by raising CancelledError mid-queue
            hit_end_of_train = ("CancelledError" in msg) and (
                "fifo_queue_enqueue" in msg
            )
            if not hit_end_of_train:
                raise

    def _localize_trained_model(
        self, config: dict, model_instance, key: dict = None
    ) -> tuple[Path, str]:
        """Localize the trained model and generate model ID.

        Parameters
        ----------
        config : dict
            DLC configuration (must include ``project_path``).
        model_instance
            Model table instance for logging.
        key : dict, optional
            ModelSelection key. Seeds the generated ``model_id`` hash so it is
            unique per selection (a parent and its same-day continuation get
            distinct ids). When ``None``, the id is deterministic per day
            (legacy behavior, retained for direct unit-test callers).

        Returns
        -------
        tuple of (Path, str)
            The model config path and the generated ``model_id``.
        """
        from deeplabcut.utils import get_model_folder
        from deeplabcut.utils.auxiliaryfunctions import read_config

        from spyglass.position.utils.protocols import default_pk_name

        project_path = Path(config["project_path"])
        config_path = project_path / "config.yaml"

        # Read the DLC config to get the correct structure
        dlc_config = read_config(str(config_path))

        # Try PyTorch engine first (DLC 3.x), fall back to TF.
        try:
            from deeplabcut.core.engine import Engine as _Engine

            _engines = [_Engine.PYTORCH, _Engine.TF]
        except ImportError:
            _engines = [None]  # older DLC — get_model_folder ignores engine

        train_dir = None
        for _engine in _engines:
            _kwargs = dict(
                trainFraction=dlc_config.get("TrainingFraction", [0.95])[0],
                shuffle=dlc_config.get("shuffle", 1),
                cfg=dlc_config,
                modelprefix=dlc_config.get("modelprefix", ""),
            )
            if _engine is not None:
                _kwargs["engine"] = _engine
            _model_dir = project_path / get_model_folder(**_kwargs)
            _train_dir = _model_dir / "train"
            if _train_dir.exists():
                train_dir = _train_dir
                break

        if train_dir is None:
            raise FileNotFoundError(
                f"Training directory not found under: {project_path}"
            )

        # Find latest snapshot: PyTorch uses *.pt, TF uses *index*.
        snapshots = list(train_dir.glob("snapshot-*.pt")) or list(
            train_dir.glob("*index*")
        )

        if not snapshots:
            # In test mode or if training failed, there may be no snapshots
            model_instance._warn_msg("No snapshot files found after training")
            latest_snapshot = 0
        else:
            # Find most recently modified snapshot
            latest_snapshot = 0
            max_modified_time = 0
            for snapshot in snapshots:
                modified_time = self._fs.getmtime(snapshot)
                if modified_time > max_modified_time:
                    # Extract snapshot number from filename (e.g. "snapshot-010")
                    import re as _re

                    _m = _re.search(r"(\d+)", snapshot.stem)
                    latest_snapshot = int(_m.group(1)) if _m else 0
                    max_modified_time = modified_time

        # Generate model ID. ``default_pk_name`` supplies the date segment,
        # so the prefix must stay date-free to avoid a doubled date. Hashing
        # the selection *key* keeps ids unique across models trained the same
        # day (e.g. a parent and its continuation share the date segment but
        # differ by model_selection_id).
        model_id = default_pk_name("mdl", key)

        model_instance._info_msg(
            f"Located trained model - snapshot: {latest_snapshot}, "
            f"model_id: {model_id}"
        )

        # Return the config path as the model path (standard DLC pattern)
        config_path = project_path / "config.yaml"
        return config_path, model_id


class SLEAPStrategy(PoseToolStrategy):
    """SLEAP tool strategy implementation."""

    @property
    def tool_name(self) -> str:
        return "SLEAP"

    @property
    def source_software(self) -> str:
        return "SLEAP"

    @property
    def supports_training(self) -> bool:
        return True

    def get_required_params(self) -> Set[str]:
        return {
            "model_type",  # single_instance, centroid, etc.
        }

    def get_skipped_params(self) -> Set[str]:
        return {
            "project_path",
            "video_sets",
            "model_path",
            "analysis_file_id",
        }

    def get_accepted_params(self) -> Set[str]:
        return {
            # Model architecture
            "model_type",
            "backbone",
            "max_stride",
            "output_stride",
            # Training parameters
            "max_epochs",
            "batch_size",
            "learning_rate",
            "save_freq",
            "val_size",
            # Data augmentation
            "augmentation_config",
            "rotation",
            "scale",
            "translate",
            # Optimization
            "optimizer",
            "lr_schedule",
            "early_stopping",
            # Model-specific
            "sigma",
            "peak_threshold",
            "integral_patch_size",
            # Paths
            "model_name",
            "run_name",
            "initial_config",
            # SLEAP version info
            "sleap_version",
            "training_labels",
        }

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "model_type": "single_instance",
            "backbone": "unet",
            "max_epochs": 200,
            "batch_size": 4,
            "learning_rate": 1e-4,
        }

    def get_parameter_aliases(self) -> Dict[str, list]:
        return {
            "model_type": ["approach"],
            "backbone": ["backbone_type"],
            "max_epochs": ["epochs", "training_epochs"],
            "batch_size": ["batch"],
        }

    def validate_params(self, params: dict) -> None:
        """Validate SLEAP-specific parameters."""
        required = self.get_required_params()
        missing = [k for k in required if k not in params]
        if missing:
            raise ValueError(f"missing required parameters: {missing}")

        # Validate model_type
        valid_model_types = [
            "single_instance",
            "centroid",
            "multi_instance",
            "topdown",
            "bottom_up",
        ]
        model_type = params.get("model_type")
        if model_type not in valid_model_types:
            raise ValueError(
                f"Invalid SLEAP model_type: {model_type}. "
                f"Must be one of: {valid_model_types}"
            )

        # Validate numeric parameters
        for param, param_type in [("max_epochs", int), ("batch_size", int)]:
            if param in params and not isinstance(params[param], param_type):
                raise ValueError(
                    f"SLEAP parameter '{param}' must be {param_type.__name__}"
                )

    def apply_epochs(self, params: dict, epochs, config: dict = None) -> dict:
        """Map ``epochs`` onto SLEAP's ``max_epochs`` (plus aliases).

        SLEAP's native training length is ``max_epochs``; it is aliased to
        ``epochs``/``training_epochs`` via :meth:`get_parameter_aliases`, so
        this sets ``max_epochs`` and expands the aliases for downstream config
        assembly.

        Parameters
        ----------
        params : dict
            Training parameters.
        epochs : int or None
            Requested training length; ``None`` returns *params* unchanged.
        config : dict, optional
            Unused (present for interface symmetry).

        Returns
        -------
        dict
            Copy of *params* with ``max_epochs`` and its aliases set.
        """
        if epochs is None:
            return params
        params = dict(params)
        params["max_epochs"] = int(epochs)
        return self.append_aliases(params)

    def train_model(
        self,
        key: dict,
        params: dict,
        skeleton_id: str,
        vid_group: dict,
        sel_entry: dict,
        model_instance,
    ) -> dict:
        """Train a SLEAP model via the sleap-train CLI.

        Thin wrapper over :meth:`_run_training` (a fresh train wires no base
        checkpoint). SLEAP reads its training length from the ``initial_config``
        file, not from ``params``; ``epochs``/``max_epochs`` baked into
        ``params`` are inert here (they only take effect on the continuation
        path via :meth:`continue_training`, which rewrites a config copy).

        Parameters
        ----------
        key : dict
            ModelSelection key
        params : dict
            Training parameters from ModelParams. Must include
            ``initial_config`` (path to SLEAP training config JSON/YAML).
            Optional: ``run_name``, ``output_dir``, ``batch_size``,
            ``peak_threshold``, ``integral_patch_size``.
        skeleton_id : str
            Skeleton ID (not used by SLEAP; stored for DB consistency)
        vid_group : dict
            VidFileGroup entry (not used directly by SLEAP CLI)
        sel_entry : dict
            Full ModelSelection entry. Must include
            ``training_labels_path`` — path to .slp labels file.
        model_instance
            Model table instance for logging.

        Returns
        -------
        dict
            Model table entry with model_id, analysis_file_name, model_path.

        Raises
        ------
        ValueError
            If ``sel_entry["training_labels_path"]`` is None or empty.
        FileNotFoundError
            If the labels file or config file does not exist.
        subprocess.CalledProcessError
            If ``sleap-train`` exits with a non-zero status.
        """
        return self._run_training(key, params, sel_entry, model_instance)

    @staticmethod
    def _build_train_cmd(
        config_path,
        labels_path,
        run_name: str,
        output_dir: str = "",
        base_checkpoint=None,
    ) -> list:
        """Assemble the ``sleap-train`` command line.

        Shared by :meth:`train_model` (fresh) and :meth:`continue_training`
        (weight-resume) so the two never drift. Passing *base_checkpoint* adds
        ``--base_checkpoint <file>``, which sleap-train wires into
        ``model_config.pretrained_backbone_weights``/``pretrained_head_weights``
        to initialise from those weights.

        Parameters
        ----------
        config_path : str, Path, or None
            SLEAP training config (JSON/YAML). Omitted from the command if
            falsy.
        labels_path : str or Path
            Path to the ``.slp`` training labels file.
        run_name : str
            Value for ``--run_name``.
        output_dir : str, optional
            Value for ``--output_dir`` (omitted if empty).
        base_checkpoint : str, Path, or None, optional
            Path to the parent checkpoint **file** (``best.ckpt`` or, for
            legacy SLEAP <=1.4, ``best_model.h5``) to resume from. Despite the
            CLI help wording ("directory containing best_model.h5"), the modern
            sleap-nn trainer requires a checkpoint *file* whose extension is
            ``.ckpt`` or ``.h5`` — a directory is rejected — so this must be the
            file path, not its parent directory.

        Returns
        -------
        list of str
            The command argument vector for ``subprocess.run``.
        """
        cmd = ["sleap-train"]
        if config_path:
            cmd.append(str(config_path))
        cmd.extend([str(labels_path), "--run_name", run_name])
        if output_dir:
            cmd.extend(["--output_dir", str(output_dir)])
        if base_checkpoint:
            cmd.extend(["--base_checkpoint", str(base_checkpoint)])
        return cmd

    def _run_training(
        self,
        key: dict,
        params: dict,
        sel_entry: dict,
        model_instance,
        *,
        base_checkpoint=None,
        config_override=None,
    ) -> dict:
        """Run ``sleap-train`` and register the resulting model.

        The single implementation behind both fresh training and continuation.
        *base_checkpoint* and *config_override* are the only continuation knobs;
        when both are ``None`` this is a plain fresh train.

        Parameters
        ----------
        key : dict
            ModelSelection key (seeds the generated ``model_id``).
        params : dict
            Training parameters from ModelParams.
        sel_entry : dict
            Full ModelSelection entry; must carry ``training_labels_path``.
        model_instance
            Model table instance for logging.
        base_checkpoint : str, Path, or None, optional
            Parent checkpoint file to resume from (see :meth:`_build_train_cmd`).
        config_override : str, Path, or None, optional
            Config to pass instead of ``params['initial_config']`` (used to
            inject an ``epochs`` budget on the continuation path).

        Returns
        -------
        dict
            Model table entry with model_id, analysis_file_name, model_path.

        Raises
        ------
        ValueError
            If ``sel_entry['training_labels_path']`` is None or empty.
        FileNotFoundError
            If the labels file or config file does not exist.
        subprocess.CalledProcessError
            If ``sleap-train`` exits with a non-zero status.
        """
        import subprocess

        from spyglass.position.utils.protocols import default_pk_name

        labels_path = sel_entry.get("training_labels_path")
        if not labels_path:
            raise ValueError(
                "SLEAP training requires 'training_labels_path' in "
                "ModelSelection. Set it to the path of your .slp labels file."
            )

        labels_path = Path(labels_path)
        if not labels_path.exists():
            raise FileNotFoundError(
                f"SLEAP labels file not found: {labels_path}"
            )

        config_path = config_override or params.get("initial_config")
        if config_path:
            config_path = Path(config_path)
            if not config_path.exists():
                raise FileNotFoundError(
                    f"SLEAP training config not found: {config_path}"
                )

        run_name = params.get("run_name") or default_pk_name(
            "sleap-run", key or {}
        )
        output_dir = params.get("output_dir", "")

        cmd = self._build_train_cmd(
            config_path,
            labels_path,
            run_name,
            output_dir,
            base_checkpoint=base_checkpoint,
        )

        model_instance._info_msg(f"Running SLEAP training: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        # Locate model output directory
        model_dir = self._find_model_output_dir(
            run_name, output_dir or "models"
        )
        model_instance._info_msg(f"SLEAP model saved to: {model_dir}")

        model_id = default_pk_name("sleap-mdl", key or {})

        return dict(
            key,
            model_id=model_id,
            analysis_file_name=None,
            model_path=str(model_dir),
        )

    def _resolve_parent_model_dir(self, sel_entry: dict, model_instance):
        """Resolve the parent model's on-disk directory from the DB.

        Reads ``model_path`` from the parent ``Model`` row identified by
        ``sel_entry['parent_id']``. SLEAP stores an absolute model directory as
        ``model_path`` (unlike DLC's stored-relative config path), so it is used
        as-is.

        Parameters
        ----------
        sel_entry : dict
            ModelSelection entry carrying ``parent_id``.
        model_instance
            Child ``Model`` table instance; its class is queried for the parent
            row.

        Returns
        -------
        pathlib.Path or None
            The parent model directory, or ``None`` if no parent is set or the
            row/path cannot be resolved.
        """
        parent_id = sel_entry.get("parent_id")
        if not parent_id:
            return None
        if "parent_model_path" in sel_entry:
            # Pre-resolved by Model.make_fetch so the tri-part make_compute
            # stays database-free. Present-but-None means the parent row or
            # its path was missing; caller degrades to fresh either way.
            model_path = sel_entry["parent_model_path"]
        else:
            try:
                model_path = (
                    model_instance.__class__ & {"model_id": parent_id}
                ).fetch1("model_path")
            except dj.errors.DataJointError:
                # missing row / no DB — caller degrades to fresh
                return None
        return Path(model_path) if model_path else None

    def _resolve_parent_checkpoint(self, parent_dir):
        """Find the resumable checkpoint file in a parent model directory.

        The modern sleap-nn trainer writes ``<model_dir>/best.ckpt``; legacy
        SLEAP (<=1.4) wrote ``best_model.h5``. Either is a valid base checkpoint
        (the loader dispatches on the ``.ckpt``/``.h5`` extension), so both are
        accepted, ``best.ckpt`` preferred. Also searches one directory level
        down, since a stored ``model_path`` may point at the run's parent; when
        several one-level-down matches exist, the most recently modified wins
        (parity with :meth:`DLCStrategy._resolve_parent_snapshot`).

        Parameters
        ----------
        parent_dir : str, Path, or None
            The parent model directory.

        Returns
        -------
        pathlib.Path or None
            Path to the checkpoint file, or ``None`` when none is present (the
            caller then degrades to a fresh, parent-linked train).
        """
        if parent_dir is None:
            return None
        parent_dir = Path(parent_dir)
        if not self._fs.exists(parent_dir):
            return None
        for name in ("best.ckpt", "best_model.h5"):
            cand = parent_dir / name
            if self._fs.exists(cand):
                return cand
        for name in ("best.ckpt", "best_model.h5"):
            matches = self._fs.glob(str(parent_dir / "*" / name))
            if matches:
                return Path(max(matches, key=self._fs.getmtime))
        return None

    def _write_epochs_config(self, config_path, epochs: int, model_instance):
        """Write a config copy with the training length set to *epochs*.

        SLEAP's trainer reads its length from the config file, not the CLI or
        ``params`` — so honouring an ``epochs`` budget means editing a copy of
        the config and pointing ``sleap-train`` at it (the user's original file
        is never touched). The epoch field differs by format: modern sleap-nn
        YAML uses ``trainer_config.max_epochs``; legacy SLEAP JSON uses
        ``optimization.epochs``.

        Parameters
        ----------
        config_path : str, Path, or None
            The ``initial_config`` to base the copy on. If falsy, no config is
            available to carry the budget and a warning is emitted.
        epochs : int
            Training length to write.
        model_instance
            Model table instance for logging.

        Returns
        -------
        pathlib.Path or None
            Path to the rewritten config copy, or ``None`` if the budget could
            not be applied (no/invalid config) — in which case the caller falls
            back to the original config and the length baked into it.
        """
        import json
        import tempfile

        if not config_path:
            model_instance._warn_msg(
                "SLEAP continuation received an epochs budget but no "
                "'initial_config' to write it into. sleap-train reads training "
                "length from the config, not from params, so the epochs value "
                "cannot be applied — set max_epochs in a SLEAP training config. "
                "Ignoring epochs."
            )
            return None

        config_path = Path(config_path)
        try:
            suffix = config_path.suffix.lower()
            if suffix in (".yaml", ".yml"):
                import yaml

                data = yaml.safe_load(config_path.read_text()) or {}
                data.setdefault("trainer_config", {})["max_epochs"] = epochs
                payload = yaml.safe_dump(data)
                out_suffix = suffix
            else:  # treat anything else as legacy SLEAP JSON
                data = json.loads(config_path.read_text())
                data.setdefault("optimization", {})["epochs"] = epochs
                payload = json.dumps(data, indent=2)
                out_suffix = ".json"
        except Exception as err:  # malformed config — don't break the resume
            model_instance._warn_msg(
                f"Could not rewrite SLEAP config for epochs={epochs} "
                f"({config_path}): {err}. Falling back to the config's own "
                "training length."
            )
            return None

        # delete=False is deliberate, not a leak: this rewritten config is
        # consumed by the out-of-process ``sleap-train`` subprocess launched in
        # ``_run_training`` (via ``config_override``), so it must outlive this
        # function's scope — a ``finally: unlink`` would pull it out from under
        # the trainer. The subprocess reads it, and the file is a small config
        # left in the system temp dir for post-mortem inspection of the exact
        # length used; OS temp reaping cleans it up.
        out = tempfile.NamedTemporaryFile(
            prefix="sleap_cont_", suffix=out_suffix, delete=False
        )
        out.write(payload.encode())
        out.close()
        new_path = Path(out.name)
        model_instance._info_msg(
            f"Wrote SLEAP continuation config with epochs={epochs}: {new_path}"
        )
        return new_path

    def continue_training(
        self,
        key: dict,
        params: dict,
        skeleton_id: str,
        vid_group: dict,
        sel_entry: dict,
        model_instance,
        *,
        epochs=None,
    ) -> dict:
        """Resume SLEAP training from the parent model's weights.

        Locates the parent model's checkpoint (``best.ckpt`` or legacy
        ``best_model.h5``) and passes it to ``sleap-train --base_checkpoint`` so
        the backbone and head layers initialise from those weights instead of
        random init. When no checkpoint is found this degrades to a fresh — but
        still ``parent_id`` linked — train (mirroring :class:`DLCStrategy`),
        keeping the lineage while starting from scratch; the caller never has to
        decide this.

        Training length ("more epochs") is honoured by writing a copy of the
        ``initial_config`` with the epoch field set and pointing the trainer at
        it, because ``sleap-train`` reads length from the config rather than
        params/CLI. The epochs value is taken from the explicit *epochs* kwarg
        when given, otherwise from ``params['max_epochs']`` (where
        :meth:`Model.train`'s ``epochs=`` lands after ``apply_epochs``). If no
        config is available the epochs value is warned about, never silently
        dropped.

        See :meth:`PoseToolStrategy.continue_training` for parameters.
        """
        parent_dir = self._resolve_parent_model_dir(sel_entry, model_instance)
        checkpoint = self._resolve_parent_checkpoint(parent_dir)

        if checkpoint is None:
            model_instance._warn_msg(
                "No resumable SLEAP checkpoint (best.ckpt / best_model.h5) "
                f"found under parent model dir {parent_dir}; training fresh "
                "but keeping the parent lineage."
            )
            return self.train_model(
                key, params, skeleton_id, vid_group, sel_entry, model_instance
            )

        # epochs: explicit kwarg wins; else the max_epochs baked into params by
        # apply_epochs (Model.train's epochs= arrives here as params max_epochs,
        # since make() does not thread the kwarg through).
        eff_epochs = epochs if epochs is not None else params.get("max_epochs")
        config_override = None
        if eff_epochs is not None:
            config_override = self._write_epochs_config(
                params.get("initial_config"), int(eff_epochs), model_instance
            )

        model_instance._info_msg(
            f"Resuming SLEAP training from parent checkpoint: {checkpoint}"
        )
        return self._run_training(
            key,
            params,
            sel_entry,
            model_instance,
            base_checkpoint=checkpoint,
            config_override=config_override,
        )

    def _find_model_output_dir(self, run_name: str, output_dir: str) -> Path:
        """Locate the model directory created by sleap-train.

        Parameters
        ----------
        run_name : str
            The ``--run_name`` passed to sleap-train.
        output_dir : str
            The ``--output_dir`` passed to sleap-train (or ``"models"``).

        Returns
        -------
        Path
            Path to the model directory.
        """
        candidate = Path(output_dir) / run_name
        if candidate.exists():
            return candidate
        # Fallback: search cwd
        cwd_candidate = Path.cwd() / "models" / run_name
        if cwd_candidate.exists():
            return cwd_candidate
        return candidate  # return even if absent; caller handles missing

    def evaluate_model(
        self,
        model_entry: dict,
        params_entry: dict,
        model_instance,
        plotting: bool = True,
        show_errors: bool = True,
        **kwargs,
    ) -> dict:
        """Evaluate a trained SLEAP model via the sleap-eval CLI.

        Parameters
        ----------
        model_entry : dict
            Model table entry; must include ``model_path``.
        params_entry : dict
            ModelParams entry; ``params`` may include
            ``training_labels`` (path to .slp labels for evaluation).
        model_instance
            Model table instance for logging.
        plotting : bool
            Currently unused for SLEAP (no equivalent CLI flag).
        show_errors : bool
            If True, log evaluation metrics to the model instance logger.
        **kwargs
            Additional options forwarded to sleap-eval.

        Returns
        -------
        dict
            Evaluation results with at least ``oks`` and ``mAP`` keys.

        Raises
        ------
        FileNotFoundError
            If the model directory does not exist.
        subprocess.CalledProcessError
            If ``sleap-eval`` exits with a non-zero status.
        """
        import subprocess
        import tempfile

        model_path = Path(model_entry.get("model_path", ""))
        if not model_path.exists():
            raise FileNotFoundError(
                f"SLEAP model directory not found: {model_path}"
            )

        params = params_entry.get("params", {})
        labels_path = params.get("training_labels") or kwargs.get("labels_path")

        # Use a temp file if the caller supplied an explicit output path
        eval_output = params.get("eval_output") or kwargs.get("eval_output")
        _tmp = None
        if eval_output is None:
            _tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
            eval_output = _tmp.name
            _tmp.close()

        cmd = ["sleap-eval", "--model", str(model_path)]
        if labels_path:
            cmd.extend(["--labels", str(labels_path)])
        cmd.extend(["--output", str(eval_output)])

        model_instance._info_msg(f"Running SLEAP evaluation: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        results = self._parse_eval_results(eval_output)

        if show_errors and results:
            model_instance._info_msg(
                f"SLEAP eval — OKS: {results.get('oks', 'N/A')}, "
                f"mAP: {results.get('mAP', 'N/A')}"
            )

        return results

    def _parse_eval_results(self, output_path: str) -> dict:
        """Parse JSON evaluation results written by sleap-eval.

        Parameters
        ----------
        output_path : str
            Path to the JSON file written by sleap-eval.

        Returns
        -------
        dict
            Evaluation metrics; guaranteed to include ``oks`` and ``mAP``.
        """
        import json

        output_path = Path(output_path)
        if not output_path.exists():
            return {"oks": None, "mAP": None}

        with open(output_path) as f:
            data = json.load(f)

        # Normalise key names across SLEAP versions
        results = {}
        for out_key, candidates in [
            ("oks", ["oks", "OKS", "mean_oks"]),
            ("mAP", ["mAP", "map", "mean_ap"]),
        ]:
            for k in candidates:
                if k in data:
                    results[out_key] = data[k]
                    break
            else:
                results[out_key] = data.get(out_key)

        # Pass through any extra metrics
        for k, v in data.items():
            if k not in results:
                results[k] = v

        return results

    def find_output_files(
        self,
        video_paths: list,
        output_dir: str = "",
        output_file_info: Any = None,
    ) -> list:
        """Find SLEAP output files in output_dir.

        Prefers ``*.analysis.h5`` (SLEAP analysis export); falls back to
        ``*.predictions.slp`` (SLEAP project with embedded predictions).
        Mirrors the structure of DLCStrategy.find_output_files().
        """
        if output_file_info:
            if isinstance(output_file_info, str):
                return [output_file_info]
            return output_file_info

        from pathlib import Path

        output_files = []
        patterns = self.get_output_file_patterns()

        for video_path in video_paths:
            video_path = Path(video_path)
            search_dir = Path(
                self.get_default_output_location(str(video_path), output_dir)
            )

            if not self._fs.exists(search_dir):
                continue

            # Primary: *.analysis.h5
            analysis_files = [
                Path(p)
                for p in self._fs.glob(str(search_dir / patterns["primary"]))
            ]
            if analysis_files:
                latest = max(analysis_files, key=lambda p: p.stat().st_mtime)
                output_files.append(str(latest))
                continue

            # Fallback: *.predictions.slp
            slp_files = [
                Path(p)
                for p in self._fs.glob(str(search_dir / patterns["fallback"]))
            ]
            if slp_files:
                latest = max(slp_files, key=lambda p: p.stat().st_mtime)
                output_files.append(str(latest))

        return output_files

    def get_output_file_patterns(self) -> Dict[str, str]:
        """Get SLEAP-specific output file patterns."""
        return {
            "primary": "*.analysis.h5",  # SLEAP analysis export (preferred)
            "fallback": "*.predictions.slp",  # SLEAP project with predictions
        }


class ToolStrategyFactory:
    """Factory for creating tool strategy instances."""

    _strategies = {
        "DLC": DLCStrategy,
        "SLEAP": SLEAPStrategy,
    }

    @classmethod
    def create_strategy(cls, tool: str) -> PoseToolStrategy:
        """Create strategy instance for specified tool.

        Parameters
        ----------
        tool : str
            Tool name ("DLC", "SLEAP")

        Returns
        -------
        PoseToolStrategy
            Strategy instance for the tool

        Raises
        ------
        ValueError
            If tool is not supported
        """
        if tool not in cls._strategies:
            available = list(cls._strategies.keys())
            raise ValueError(
                f"Unsupported tool: {tool}. Available: {available}"
            )

        return cls._strategies[tool]()

    @classmethod
    def get_available_tools(cls) -> list[str]:
        """Get list of available tool names."""
        return list(cls._strategies.keys())

    @classmethod
    def register_strategy(cls, tool: str, strategy_class: type) -> None:
        """Register a new tool strategy (for extensibility).

        Parameters
        ----------
        tool : str
            Tool name
        strategy_class : type
            Strategy class implementing PoseToolStrategy
        """
        if not issubclass(strategy_class, PoseToolStrategy):
            raise ValueError(
                "Strategy class must implement PoseToolStrategy interface"
            )
        cls._strategies[tool] = strategy_class

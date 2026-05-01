"""Tool strategy pattern for pose estimation model training.

Provides a pluggable architecture for different pose estimation tools
(DLC, SLEAP, etc.) with consistent interfaces and parameter management.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Set

import datajoint as dj

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


class DLCStrategy(PoseToolStrategy):
    """DeepLabCut tool strategy implementation."""

    @property
    def tool_name(self) -> str:
        return "DLC"

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
        }

    def get_accepted_params(self) -> Set[str]:
        return {
            # Core DLC parameters
            "project_path",
            "shuffle",
            "trainingsetindex",
            "maxiters",
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
        return {
            "shuffle": 1,
            "trainingsetindex": 0,
            "maxiters": None,  # Use DLC default
            "displayiters": None,
            "saveiters": None,
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
        self._execute_training(config_path, params, model_instance)

        # Localize and register trained model
        model_path, model_id = self._localize_trained_model(
            config, model_instance
        )
        latest_model = self._get_latest_dlc_model_info(config)

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

    def _get_latest_dlc_model_info(self, config: dict) -> dict:
        """Get latest trained DLC model information from project directory.

        Discovers trained models in the DLC project's dlc-models directory structure.
        Returns information about the most recently modified model, or empty dict if
        no trained models are found.

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

        # Look for dlc-models directory
        dlc_models_dir = project_path / "dlc-models"
        if not self._fs.exists(dlc_models_dir):
            return {}

        # Find all iteration directories (iteration-0, iteration-1, etc.)
        iteration_dirs = sorted(dlc_models_dir.glob("iteration-*"))
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

                # Check for train directory containing pose_cfg.yaml
                train_dir = model_dir / "train"
                if not train_dir.exists():
                    continue

                pose_cfg = train_dir / "pose_cfg.yaml"
                if not pose_cfg.exists():
                    continue

                # Get modification time of pose_cfg.yaml as training date
                mtime = pose_cfg.stat().st_mtime
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

        model_instance._info_msg("Creating DLC training dataset...")

        with test_mode_suppress():
            create_training_dataset(str(config_path), **training_dataset_kwargs)

    def _execute_training(
        self, config_path: Path, params: dict, model_instance
    ) -> None:
        """Execute DLC model training using train_network."""
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
        self, config: dict, model_instance
    ) -> tuple[Path, str]:
        """Localize the trained model and generate model ID."""
        from datetime import datetime

        from deeplabcut.utils import get_model_folder
        from deeplabcut.utils.auxiliaryfunctions import read_config

        from spyglass.position.utils.protocols import default_pk_name

        project_path = Path(config["project_path"])
        config_path = project_path / "config.yaml"

        # Read the DLC config to get the correct structure
        dlc_config = read_config(str(config_path))

        # Get the model directory from DLC
        model_dir = project_path / get_model_folder(
            trainFraction=dlc_config.get("TrainingFraction", [0.95])[
                0
            ],  # Get first training fraction with default
            shuffle=dlc_config.get("shuffle", 1),  # Default shuffle number
            cfg=dlc_config,
            modelprefix=dlc_config.get("modelprefix", ""),
        )

        train_dir = model_dir / "train"
        if not train_dir.exists():
            raise FileNotFoundError(
                f"Training directory not found: {train_dir}"
            )

        # Find latest snapshot file
        snapshots = list(train_dir.glob("*index*"))

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
                    # Extract snapshot number from filename (skip "snapshot-" prefix)
                    latest_snapshot = int(snapshot.stem[9:])
                    max_modified_time = modified_time

        # Generate model ID
        timestamp = datetime.now().strftime("%Y%m%d")
        model_id = default_pk_name(f"mdl-{timestamp}")

        model_instance._info_msg(
            f"Located trained model - snapshot: {latest_snapshot}, "
            f"model_id: {model_id}"
        )

        # Return the config path as the model path (standard DLC pattern)
        config_path = project_path / "config.yaml"
        return config_path, model_id


class SLEAPStrategy(PoseToolStrategy):
    """SLEAP tool strategy implementation (placeholder/future)."""

    @property
    def tool_name(self) -> str:
        return "SLEAP"

    @property
    def supports_training(self) -> bool:
        return False  # SLEAP training not yet implemented

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

    def train_model(
        self,
        key: dict,
        params: dict,
        skeleton_id: str,
        vid_group: dict,
        sel_entry: dict,
        model_instance,
    ) -> dict:
        """Train SLEAP model (not yet implemented)."""
        raise NotImplementedError(
            "SLEAP training not yet implemented. "
            "Please use DLC or import pre-trained SLEAP models via load()."
        )

    def evaluate_model(
        self,
        model_entry: dict,
        params_entry: dict,
        model_instance,
        plotting: bool = True,
        show_errors: bool = True,
        **kwargs,
    ) -> dict:
        """Evaluate SLEAP model (not yet implemented)."""
        raise NotImplementedError("SLEAP evaluation not yet implemented.")

    def find_output_files(
        self,
        video_paths: list,
        output_dir: str = "",
        output_file_info: Any = None,
    ) -> list:
        """Find SLEAP output files (placeholder implementation)."""
        # TODO: Implement SLEAP-specific output discovery
        from pathlib import Path

        output_files = []

        for video_path in video_paths:
            video_path = Path(video_path)
            search_dir = Path(
                self.get_default_output_location(str(video_path), output_dir)
            )

            if search_dir.exists():
                # Look for SLEAP output files (.slp files)
                patterns = self.get_output_file_patterns()
                sleap_files = list(search_dir.glob(patterns["primary"]))
                if sleap_files:
                    latest_file = max(
                        sleap_files, key=lambda p: p.stat().st_mtime
                    )
                    output_files.append(str(latest_file))

        return output_files

    def get_output_file_patterns(self) -> Dict[str, str]:
        """Get SLEAP-specific output file patterns."""
        return {
            "primary": "*.slp",  # SLEAP project files
            "predictions": "*.h5",  # SLEAP prediction files
            "fallback": "*.slp",
        }


class NDXPoseStrategy(PoseToolStrategy):
    """NDX-Pose import strategy for pre-trained models."""

    @property
    def tool_name(self) -> str:
        return "ndx-pose"

    @property
    def supports_training(self) -> bool:
        return False  # NDX-Pose is for importing existing models only

    def get_required_params(self) -> Set[str]:
        return {"nwb_file", "model_name"}  # Required for NDX-Pose import

    def get_skipped_params(self) -> Set[str]:
        return {
            "model_path",
            "analysis_file_id",
            "nwb_file_path",
        }

    def get_accepted_params(self) -> Set[str]:
        return {
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
        }

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "source_software": "unknown",
            "description": "Imported from ndx-pose NWB file",
        }

    def get_parameter_aliases(self) -> Dict[str, list]:
        return {
            "nwb_file": ["nwb_path", "file_path"],
            "model_name": ["model_id"],
        }

    def validate_params(self, params: dict) -> None:
        """Validate NDX-Pose import parameters."""
        required = self.get_required_params()
        missing = [k for k in required if k not in params]
        if missing:
            raise ValueError(
                f"NDX-Pose import missing required parameters: {missing}"
            )

        # Validate NWB file exists
        nwb_path = Path(params["nwb_file"])
        if not nwb_path.exists():
            raise ValueError(f"NWB file does not exist: {nwb_path}")

    def train_model(
        self,
        key: dict,
        params: dict,
        skeleton_id: str,
        vid_group: dict,
        sel_entry: dict,
        model_instance,
    ) -> dict:
        """Import model from NDX-Pose NWB (training not applicable)."""
        raise NotImplementedError(
            "NDX-Pose strategy is for importing existing models, not training. "
            "Use Model.load() for NDX-Pose NWB files."
        )

    def evaluate_model(
        self,
        model_entry: dict,
        params_entry: dict,
        model_instance,
        plotting: bool = True,
        show_errors: bool = True,
        **kwargs,
    ) -> dict:
        """Evaluate NDX-Pose model (not applicable for imported models)."""
        raise NotImplementedError(
            "NDX-Pose models are pre-trained imports and cannot be evaluated. "
            "Evaluation should be done in the original training environment."
        )

    def load(self, model_path: Path, model_instance, **kwargs) -> dict:
        """Import NDX-Pose model using existing _import_ndx_pose_model method."""
        return model_instance._import_ndx_pose_model(model_path, **kwargs)

    def verify_model(
        self, model_path: Path, check_inference: bool = True
    ) -> tuple[dict, list]:
        """Verify NDX-Pose model integrity."""
        checks, warnings = super().verify_model(model_path, check_inference)

        if check_inference and checks.get("model_exists"):
            # ndx-pose models cannot run inference directly
            warnings.append(
                "ndx-pose models cannot run inference directly - "
                "use original DLC/SLEAP project"
            )

        return checks, warnings

    def find_output_files(
        self,
        video_paths: list,
        output_dir: str = "",
        output_file_info: Any = None,
    ) -> list:
        """Find NDX-Pose output files (not applicable for imports)."""
        # NDX-Pose models are imported from existing NWB files, not generated
        raise NotImplementedError(
            "NDX-Pose models are imported from NWB files, not generated from inference"
        )

    def get_output_file_patterns(self) -> Dict[str, str]:
        """Get NDX-Pose output patterns (not applicable)."""
        return {
            "primary": "*.nwb",  # NWB files contain the pose data
            "fallback": "*.nwb",
        }


class ToolStrategyFactory:
    """Factory for creating tool strategy instances."""

    _strategies = {
        "DLC": DLCStrategy,
        "SLEAP": SLEAPStrategy,
        "ndx-pose": NDXPoseStrategy,
    }

    @classmethod
    def create_strategy(cls, tool: str) -> PoseToolStrategy:
        """Create strategy instance for specified tool.

        Parameters
        ----------
        tool : str
            Tool name ("DLC", "SLEAP", "ndx-pose")

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

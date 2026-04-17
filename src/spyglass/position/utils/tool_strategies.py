"""Tool strategy pattern for pose estimation model training.

Provides a pluggable architecture for different pose estimation tools
(DLC, SLEAP, etc.) with consistent interfaces and parameter management.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Set

import datajoint as dj


class PoseToolStrategy(ABC):
    """Abstract base class for pose estimation tool strategies.

    Each tool (DLC, SLEAP, etc.) implements this interface to provide
    consistent parameter validation, training, and model management.
    """

    @property
    @abstractmethod
    def tool_name(self) -> str:
        """Name of the pose estimation tool."""

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
        """Train DLC model using existing _make_dlc_model method."""
        return model_instance._make_dlc_model(
            key, params, skeleton_id, vid_group, sel_entry
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

            if not search_dir.exists():
                continue

            # Look for DLC output files for this video
            patterns = self.get_output_file_patterns()
            dlc_pattern = patterns["primary"].format(video_stem=video_path.stem)
            matching_files = list(search_dir.glob(dlc_pattern))

            if matching_files:
                # Use most recent if multiple files
                latest_file = max(
                    matching_files, key=lambda p: p.stat().st_mtime
                )
                output_files.append(str(latest_file))
            else:
                # Fallback: look for any h5 files
                fallback_pattern = patterns["fallback"]
                h5_files = list(search_dir.glob(fallback_pattern))
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


class SLEAPStrategy(PoseToolStrategy):
    """SLEAP tool strategy implementation (placeholder/future)."""

    @property
    def tool_name(self) -> str:
        return "SLEAP"

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

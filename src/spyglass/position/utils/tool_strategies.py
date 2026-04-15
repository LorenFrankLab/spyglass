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


class SLEAPStrategy(PoseToolStrategy):
    """SLEAP tool strategy implementation (placeholder/future)."""

    @property
    def tool_name(self) -> str:
        return "SLEAP"

    def get_required_params(self) -> Set[str]:
        return {
            "model_type",  # single_instance, centroid, etc.
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
            "Please use DLC or import pre-trained SLEAP models via import_model()."
        )


class NDXPoseStrategy(PoseToolStrategy):
    """NDX-Pose import strategy for pre-trained models."""

    @property
    def tool_name(self) -> str:
        return "ndx-pose"

    def get_required_params(self) -> Set[str]:
        return {"nwb_file", "model_name"}  # Required for NDX-Pose import

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
            "Use Model.import_model() for NDX-Pose NWB files."
        )


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

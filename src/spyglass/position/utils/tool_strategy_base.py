"""The pose-tool strategy interface.

Its own module so each tool's strategy can import the base class without
importing the others: `tool_strategies` imports all of them, so a strategy
importing from there would be a cycle.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Set

from .protocols import RealFileSystem


class PoseToolStrategy(ABC):
    """Abstract base class for pose estimation tool strategies.

    Each tool (DLC, SLEAP, etc.) implements this interface to provide
    consistent parameter validation, training, and model management.
    """

    def __init__(self, filesystem=None):
        """Initialize with optional filesystem dependency.

        Parameters
        ----------
        filesystem : optional
            Anything with RealFileSystem's four methods; defaults to it.
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

        Lets a caller branch on the capability rather than the strategy type.

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
            One entry per entry in ``video_paths``, positionally aligned:
            the output file path, or ``None`` placeholder where that video has
            no output.
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

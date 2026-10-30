"""The pose-tool strategy interface.

Its own module so each tool's strategy can import the base class without
importing the others: `tool_strategies` imports all of them, so a strategy
importing from there would be a cycle.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Set, Union

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

    #: Suffixes, without the dot, that name this tool's project
    #: configuration. Empty where a tool's models are not identified by a
    #: file extension, so it never claims another tool's file.
    project_suffixes: Set[str] = set()

    def split_project_kwargs(self, kwargs: dict) -> tuple:
        """Divide caller kwargs between project creation and extraction.

        Override per tool, which knows its own entry points. The default
        claims none of them.

        Parameters
        ----------
        kwargs : dict
            Extra keyword arguments the caller passed.

        Returns
        -------
        tuple of (dict, dict)
            Arguments for creating the project, and for extracting frames.
        """
        return {}, {}

    def extract_frames(
        self, config_path, videos: list, extract_kwargs: dict, model_instance
    ) -> list:
        """Extract labelling frames for any video that has none.

        Override per tool. The default raises: a tool that cannot extract
        frames must say so rather than report success having done nothing.

        Parameters
        ----------
        config_path : pathlib.Path
            The project configuration file.
        videos : list
            Video paths the project should hold frames for.
        extract_kwargs : dict
            Tool arguments from :meth:`split_project_kwargs`.
        model_instance
            ``Model`` table instance, used for logging.

        Returns
        -------
        list
            The videos frames were extracted for, empty when all were
            already present.

        Raises
        ------
        NotImplementedError
            Always, unless a subclass implements it.
        """
        raise NotImplementedError(
            f"{type(self).__name__} cannot extract labelling frames. "
            "Prepare the project with the tool directly."
        )

    @staticmethod
    def resolve_stored_path(stored_path) -> Path:
        """Turn a stored ``model_path`` into a path on this filesystem.

        Override where the tool's paths are stored relative to a configured
        root. The default treats the stored value as usable as-is.

        Parameters
        ----------
        stored_path : str or pathlib.Path

        Returns
        -------
        pathlib.Path
        """
        return Path(stored_path)

    @staticmethod
    def verify_model_path(model_path) -> tuple:
        """Check that a model path looks like this tool's model.

        Override where the tool expects a particular file. The default makes
        no claim about the path's contents.

        Parameters
        ----------
        model_path : pathlib.Path
            An existing path, already confirmed present.

        Returns
        -------
        tuple of (bool, list of str)
            Whether it looks valid, and any warnings.
        """
        return True, []

    @staticmethod
    def verify_params(params: dict) -> tuple:
        """Check that stored parameters are usable for this tool.

        Override where the tool requires particular keys. The default asks
        only that parameters exist.

        Parameters
        ----------
        params : dict
            The ``ModelParams`` row's parameters.

        Returns
        -------
        tuple of (bool, list of str)
            Whether they are valid, and any errors.
        """
        if params and isinstance(params, dict):
            return True, []
        return False, [f"Invalid params: {params}"]

    def read_v1_model(self, v1_key: dict) -> dict:
        """Locate a V1 model's project and the parameters that identify it.

        Override where the tool has a V1 table in spyglass. The default
        raises: only DeepLabCut does.

        Parameters
        ----------
        v1_key : dict
            Primary-key fields identifying one row in the tool's V1 table.

        Returns
        -------
        dict
            ``config_path``, ``extra_params`` and ``name`` (for logging).

        Raises
        ------
        NotImplementedError
            Always, unless a subclass implements it.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no V1 table to import from."
        )

    def read_project(self, model_path, extra_params: dict = None) -> dict:
        """Read a trained project and return what the tables need from it.

        Override per tool: the file type, the configuration layout and the
        key holding body-part names are all tool-specific. The default
        raises rather than guessing a format.

        Parameters
        ----------
        model_path : pathlib.Path
            The project's configuration file.
        extra_params : dict, optional
            Training parameters the configuration does not record, merged
            over it. One project can hold several models distinguished only
            by these, so they reach the parameter hash.

        Returns
        -------
        dict
            ``config``, the merged parameters, and ``skeleton``, a dict of
            ``bodyparts`` and ``skeleton`` for ``Skeleton.insert1``.

        Raises
        ------
        NotImplementedError
            Always, unless a subclass implements it.
        """
        raise NotImplementedError(
            f"{type(self).__name__} cannot read a project from disk. "
            "Import this tool's models another way."
        )

    def canonicalize_project(
        self, config_path, config: dict, mapping: dict, model_instance
    ) -> dict:
        """Rewrite a project's files to canonical body-part spelling.

        Override where the tool stores names in files on disk. The default
        raises: rewriting a user's project is a capability a strategy has to
        claim, and silently doing nothing would report success.

        The caller resolves the names, since that reads the ``BodyPart``
        table, and passes the resulting *mapping* here.

        Parameters
        ----------
        config_path : str or pathlib.Path
            The project configuration file.
        config : dict
            Its already-parsed contents.
        mapping : dict
            ``{surface_form: canonical}`` for every name, resolved by the
            caller.
        model_instance
            ``Model`` table instance, used for logging.

        Returns
        -------
        dict
            The subset of *mapping* that actually changed, empty when the
            project was already canonical.

        Raises
        ------
        NotImplementedError
            Always, unless a subclass implements it.
        """
        raise NotImplementedError(
            f"{type(self).__name__} cannot rewrite project files. Correct "
            "the body-part names in the project by hand, or import without "
            "`normalize_names`."
        )

    def resolve_zoo_model(self, dataset: str, model_name, dest_dir):
        """Resolve a pretrained zoo model and place its weights on disk.

        Override where the tool publishes a model zoo. The default raises:
        a zoo is a capability a strategy has to claim, and a stub returning
        nothing would let `Model.load` build a row around absent weights.

        Parameters
        ----------
        dataset : str
            Zoo dataset name.
        model_name : str or None
            Backbone to use, or None for the dataset's default.
        dest_dir : pathlib.Path
            Directory to copy the weights into, so the stored path resolves
            for every user of the database rather than one environment's
            package cache.

        Returns
        -------
        dict
            ``backbone``, ``config``, ``weights`` (a ``Path``) and
            ``training_mode``.

        Raises
        ------
        NotImplementedError
            Always, unless a subclass implements it.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no model zoo. Import a trained "
            "project with `Model.load(<config path>)` instead."
        )

    @staticmethod
    def project_has_weights(config: dict, config_path) -> bool:
        """Does this tool's project hold trained weights?

        Override where the tool keeps weights in a discoverable layout. The
        default answers False: a strategy that does not claim weights must
        not have them presumed. `Model.load` reads this to decide between a
        `Model` row and a pending `ModelSelection`, so a True default would
        register a model that cannot run inference.

        Parameters
        ----------
        config : dict
            Tool configuration.
        config_path : str or pathlib.Path
            Its location.

        Returns
        -------
        bool
            False unless a subclass finds weights.
        """
        return False

    @staticmethod
    def project_training_mode(config: dict, config_path) -> str:
        """Can this imported project be trained further?

        Override where the tool distinguishes the cases. The default answers
        `inference_only`, the weakest of the three: continuing training is a
        capability a strategy has to claim, not one it inherits.

        Note this is the opposite of `Model.training_mode`'s column default,
        which is `resumable` because a row written by `Model.make` was
        trained in place. This hook answers for an *imported* project.

        Parameters
        ----------
        config : dict
            Tool configuration.
        config_path : str or pathlib.Path
            Its location.

        Returns
        -------
        str
            One of ``resumable``, ``weights_only``, ``inference_only``.
        """
        return "inference_only"

    @staticmethod
    def project_config(params: dict) -> Union[dict, None]:
        """Load a project's configuration, for engine resolution.

        Override where the tool has a project config on disk. The default
        returns None, meaning no configuration is available.

        Parameters
        ----------
        params : dict
            Training parameters.

        Returns
        -------
        dict or None
        """
        return None

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

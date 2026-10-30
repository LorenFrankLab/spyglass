"""DeepLabCut implementation of the pose-tool strategy.

DeepLabCut imports are deferred into the method that needs them, or guarded
at module level, so this module loads without DeepLabCut installed.
"""

import warnings
from pathlib import Path
from typing import Any, Dict, Set, Union

import yaml

from .dlc_io import read_dlc_model_eval
from .path_helpers import to_stored_path as _to_stored_path
from .protocols import ModelMetadata, default_pk_name
from .yaml_io import load_yaml
from .tool_strategy_base import PoseToolStrategy

from spyglass.utils import logger

from spyglass.position.utils import suppress_print_from_package

# Guarded rather than deferred so `evaluate_network is None` stays testable.
with suppress_print_from_package():
    try:
        from deeplabcut import evaluate_network
    except (ImportError, RecursionError):
        # RecursionError can occur when TF 2.16+ (Keras 3) is installed
        # without tf-keras; the TF compat loader recurses infinitely. Set
        # TF_USE_LEGACY_KERAS=1 and install tf-keras to restore the TF
        # backend.
        evaluate_network = None


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

    @staticmethod
    def _shuffle_of(snapshot: Path) -> Union[int, None]:
        """Read the shuffle number out of a snapshot's model-folder name.

        DLC names model folders ``<task>-<date>-trainset<pct>shuffle<n>``, so
        the shuffle a checkpoint belongs to is recoverable from its path.

        Parameters
        ----------
        snapshot : Path
            Path to a snapshot file, nested under ``.../train/``.

        Returns
        -------
        int or None
            The shuffle number, or ``None`` if no ancestor encodes one.
        """
        import re as _re

        for parent in snapshot.parents:
            if match := _re.search(r"shuffle(\d+)", parent.name):
                return int(match.group(1))
        return None

    def _resolve_parent_snapshot(
        self, config: dict, *, shuffle=None, parent_snapshot=None
    ):
        """Locate the snapshot to resume the parent model from.

        Uses *parent_snapshot*, the checkpoint recorded on the parent ``Model``
        row, when it is still on disk. Otherwise falls back to
        :meth:`get_latest_model_info`, which returns the most recently
        modified train directory anywhere in the project; that path serves
        rows predating ``snapshot_path``.

        Returns ``None`` when no resumable snapshot is available, e.g. the
        requested ``shuffle`` differs from the trained one, so weight-resume
        is invalid and the caller should degrade to a fresh (parent-linked)
        train.

        Parameters
        ----------
        config : dict
            DLC project configuration (must include ``project_path``).
        shuffle : int, optional
            Requested shuffle; if it differs from the parent model's shuffle,
            no snapshot is returned.
        parent_snapshot : str or Path, optional
            Snapshot stored on the parent ``Model`` row, as a stored-relative
            or absolute path.

        Returns
        -------
        Path or None
            Snapshot path, or ``None`` if none is resumable.
        """
        from .path_helpers import resolve_model_path

        if parent_snapshot:
            snapshot = resolve_model_path(parent_snapshot)
            if self._fs.exists(snapshot):
                if shuffle is not None and (
                    self._shuffle_of(snapshot) not in (None, shuffle)
                ):
                    return None  # shuffle changed → weights can't be reused
                return snapshot

        info = self.get_latest_model_info(config)
        if not info:
            return None
        if shuffle is not None and info.get("shuffle") != shuffle:
            return None  # shuffle changed → weights can't be reused
        train_dir = Path(info["path"])
        # PyTorch-only: ``continue_training`` rejects TF-engine continuation
        # before this is reached, so only ``snapshot-*.pt`` is searched
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
        """Resume DLC training from the parent model's own snapshot.

        Takes the checkpoint recorded on the parent ``Model`` row (passed
        through ``sel_entry['parent_snapshot_path']``) and wires it into the
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
        snapshot = self._resolve_parent_snapshot(
            config,
            shuffle=shuffle,
            parent_snapshot=sel_entry.get("parent_snapshot_path"),
        )

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

        from spyglass.position.utils import suppress_print_from_package

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
        model_path, model_id, snapshot_path = self._localize_trained_model(
            config, model_instance, key=key, params=params
        )
        latest_model = self.get_latest_model_info(config)

        nwb_file_name = model_instance._write_model_metadata(
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
                session_files=vid_group.get("nwb_file_names", []),
            )
        )

        # Return Model entry
        return dict(
            key,
            model_id=model_id,
            analysis_file_name=nwb_file_name,
            model_path=_to_stored_path(model_path),
            snapshot_path=(
                _to_stored_path(snapshot_path) if snapshot_path else None
            ),
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
            Name recorded as the project's experimenter, and the middle
            component of the ``{project_name}-{experimenter}-{date}`` folder
            DLC creates. Must be a real identifier (e.g. the calling user).
            Deriving it from *project_name* duplicates the name into the
            folder (see issue #1676).
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
        """Evaluate a DLC model.

        Parameters
        ----------
        model_entry : dict
            Model table entry (unused; kept for API consistency)
        params_entry : dict
            ModelParams entry
        model_instance
            `Model` table instance, used for logging
        plotting : bool
            Create labeled images
        show_errors : bool
            Display errors
        **kwargs
            Additional DLC evaluation parameters

        Returns
        -------
        dict
            Evaluation results

        Raises
        ------
        ImportError
            If DeepLabCut is unavailable. Checked here rather than in
            :meth:`evaluate` so a SLEAP model still evaluates through
            ``SLEAPStrategy``.
        ValueError
            If ``project_path`` is absent from the parameters.
        FileNotFoundError
            If the project's ``config.yaml`` is missing.
        """
        if evaluate_network is None:
            raise ImportError(
                "DeepLabCut evaluate_network is required for evaluation. "
                "Install with: pip install deeplabcut>=3.0"
            )

        params = params_entry["params"]

        # Get project path
        if "project_path" not in params:
            raise ValueError(
                "DLC evaluation requires 'project_path' in ModelParams"
            )

        project_path = Path(params["project_path"])
        config_path = project_path / "config.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"DLC config not found: {config_path}")

        # Get evaluation parameters
        shuffle = kwargs.get("Shuffles", [params.get("shuffle", 1)])
        if not isinstance(shuffle, list):
            shuffle = [shuffle]

        trainingsetindex = kwargs.get(
            "trainingsetindex", params.get("trainingsetindex", 0)
        )

        model_instance._info_msg(
            f"Running DLC evaluation on shuffles: {shuffle}"
        )

        # Run DLC evaluation
        try:
            with suppress_print_from_package():
                evaluate_network(
                    str(config_path),
                    Shuffles=shuffle,
                    trainingsetindex=trainingsetindex,
                    plotting=plotting,
                    show_errors=show_errors,
                )
        except Exception as e:
            model_instance._err_msg(f"DLC evaluation failed: {e}")
            raise

        model_instance._info_msg("Evaluation completed successfully")

        # Parse evaluation results
        results = self._parse_dlc_evaluation_results(
            project_path, shuffle[0], trainingsetindex
        )

        if show_errors and results:
            model_instance._info_msg(
                f"Train error: {results['train_error']:.2f} px, "
                f"Test error: {results['test_error']:.2f} px"
            )

        return results

    def _parse_dlc_evaluation_results(
        self, project_path: Path, shuffle: int, trainingsetindex: int
    ) -> dict:
        """Parse DLC evaluation results from CSV file.

        Parameters
        ----------
        project_path : Path
            DLC project path
        shuffle : int
            Shuffle number
        trainingsetindex : int
            Training set index

        Returns
        -------
        dict
            Parsed evaluation results
        """
        try:
            raw = read_dlc_model_eval(
                project_path,
                load_yaml(project_path / "config.yaml"),
                shuffle,
                trainingsetindex,
            )
        except (RuntimeError, KeyError, IndexError) as e:
            logger.warning(f"No evaluation results for shuffle {shuffle}: {e}")
            return {}

        def _num(field, cast=float):
            """Cast a normalized field, tolerating one the CSV omits."""
            value = raw.get(field)
            return None if value in (None, "") else cast(value)

        return {
            "train_error": _num(" Train error(px)"),
            "test_error": _num(" Test error(px)"),
            "train_error_p": _num("Train error with p-cutoff"),
            "test_error_p": _num("Test error with p-cutoff"),
            "p_cutoff": _num("p-cutoff used"),
            "training_iterations": _num("Training iterations:", int),
            "shuffle": int(shuffle),
            "train_fraction": _num("%Training dataset", int),
            "results_path": raw.get("results_path"),
        }

    def find_output_files(
        self,
        video_paths: list,
        output_dir: str = "",
        output_file_info: Any = None,
    ) -> list:
        """Find DLC output files using DLC naming conventions.

        DLC saves files as: {video_stem}DLC_{network}_{shuffle}_{snapshot}.h5
        If no destfolder specified, saves to video directory.

        Returns one entry per video, ``None`` where a video has no output.
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
                output_files.append(None)
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
                continue

            # Fallback: look for any h5 files
            fallback_pattern = patterns["fallback"]
            full_fallback_pattern = str(search_dir / fallback_pattern)
            h5_files = [Path(p) for p in self._fs.glob(full_fallback_pattern)]
            if h5_files:
                latest_h5 = max(h5_files, key=lambda p: p.stat().st_mtime)
                output_files.append(str(latest_h5))
            else:
                output_files.append(None)

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

    def read_project(self, model_path, extra_params: dict = None) -> dict:
        """Read a DLC `config.yaml` and the skeleton it declares.

        See :meth:`PoseToolStrategy.read_project` for parameters.

        Raises
        ------
        ValueError
            If *model_path* is not a YAML file.
        """
        model_path = Path(model_path)
        if model_path.suffix not in [".yml", ".yaml"]:
            raise ValueError("DLC model path must be a .yml or .yaml file")

        config = {**load_yaml(model_path), **(extra_params or {})}
        return {
            "config": config,
            "skeleton": {
                "bodyparts": config.get("bodyparts", []),
                "skeleton": config.get("skeleton", []),
            },
        }

    def canonicalize_project(
        self, config_path, config: dict, mapping: dict, model_instance
    ) -> dict:
        """Rewrite a DLC project's `config.yaml` to canonical spelling.

        The original is copied to a timestamped
        ``config.yaml.<timestamp>.bak`` beside it before any write, so
        repeated rewrites never overwrite an earlier backup.

        A project holding ``CollectedData_*`` labels raises instead: those
        files carry body-part names as column headers and are not rewritten
        here, so continued training would read columns the config no longer
        names. Checked after the no-op test, since an already canonical
        project performs no rewrite and so cannot fall out of step.

        See :meth:`PoseToolStrategy.canonicalize_project` for parameters.

        Raises
        ------
        datajoint.DataJointError
            If the project holds label files. Nothing is written.
        """
        import shutil
        from datetime import datetime

        import datajoint as dj

        from spyglass.position.v2.utils.config_io import _rename_leaves
        from spyglass.position.utils.yaml_io import dump_yaml

        config_path = Path(config_path)
        changes = {k: v for k, v in mapping.items() if k != v}
        if not changes:
            return {}  # already canonical: no-op, no backup churn

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

        # Microseconds guarantee a unique backup name within the same second.
        ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        backup = config_path.with_name(f"{config_path.name}.{ts}.bak")
        shutil.copy2(config_path, backup)

        config["bodyparts"] = [
            mapping.get(bp, bp) for bp in config.get("bodyparts", [])
        ]
        config["skeleton"] = _rename_leaves(config.get("skeleton", []), mapping)
        dump_yaml(config_path, config)
        model_instance._info_msg(
            f"Canonicalized {len(changes)} name(s) in {config_path.name}; "
            f"original backed up to {backup.name}"
        )
        return changes

    def resolve_zoo_model(self, dataset: str, model_name, dest_dir):
        """Resolve a SuperAnimal backbone and copy its weights out of cache.

        DLC caches zoo weights inside its own package, which is per conda
        environment, so a stored path pointing there would be valid only for
        the importing user. The weights are copied to *dest_dir* first.

        See :meth:`PoseToolStrategy.resolve_zoo_model` for parameters.
        """
        import shutil

        from spyglass.position.v2.utils import fetch_dlc_zoo

        backbone = fetch_dlc_zoo.resolve_backbone(dataset, model_name)
        cached = fetch_dlc_zoo.fetch(dataset, backbone)
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / cached.name
        if not dest.exists():
            shutil.copy2(cached, dest)

        return {
            "backbone": backbone,
            "config": fetch_dlc_zoo.project_config(dataset),
            "weights": dest,
            "training_mode": self.zoo_training_mode(backbone),
        }

    @staticmethod
    def project_has_weights(config: dict, config_path) -> bool:
        """Does this DLC project hold trained weights?

        Covers both engines: PyTorch writes ``snapshot-*.pt``, TensorFlow
        writes a ``*.index`` triple. Distinct from
        :meth:`project_training_mode`, which reports ``inference_only`` both
        for a TensorFlow project that *can* infer and for a project with no
        weights at all, which can do neither.

        Parameters
        ----------
        config : dict
            Parsed DLC ``config.yaml``.
        config_path : str or pathlib.Path
            Its location, used when the config carries no ``project_path``.

        Returns
        -------
        bool
            True when any snapshot is present.
        """
        project = Path(config.get("project_path") or Path(config_path).parent)
        return bool(
            next(project.rglob("snapshot-*.pt"), None)
            or next(project.rglob("*.index"), None)
        )

    @staticmethod
    def project_training_mode(config: dict, config_path) -> str:
        """Can this imported DLC project be trained further?

        Declared once at import, so ``Model`` rows can be filtered without
        touching the filesystem. It is a *claim*, not enforcement --
        ``Model.train`` still verifies the snapshot exists, because paths go
        stale.

        Parameters
        ----------
        config : dict
            Parsed DLC ``config.yaml``.
        config_path : str or pathlib.Path
            Its location, used to find the project's ``dlc-models`` tree.

        Returns
        -------
        str
            - ``resumable`` — a PyTorch ``snapshot-*.pt`` is present.
            - ``inference_only`` — TensorFlow engine (DLC-TF continuation is a
              documented won't-do), or no snapshot to resume from.
        """
        if str(config.get("engine", "pytorch")).lower() != "pytorch":
            return "inference_only"  # DLC-TF continuation: NotImplementedError

        project = Path(config.get("project_path") or Path(config_path).parent)
        if next(project.rglob("snapshot-*.pt"), None):
            return "resumable"
        return "inference_only"

    @staticmethod
    def zoo_training_mode(model_name: str) -> str:
        """Continuation capability for a zoo backbone.

        Zoo models are never *resumed*: DLC's fine-tune entry point is
        ``build_weight_init(super_animal=...)``, which seeds a **new** project
        from SuperAnimal weights rather than continuing the imported model.

        Returns
        -------
        str
            ``inference_only`` for TensorFlow backbones (``dlcrnet``, whose
            continuation is a won't-do here), ``weights_only`` otherwise.
        """
        from spyglass.position.v2.utils.fetch_dlc_zoo import backbone_framework

        if backbone_framework(model_name) == "tensorflow":
            return "inference_only"
        return "weights_only"

    @staticmethod
    def project_config(params: dict) -> Union[dict, None]:
        """Load the DLC project's config, for engine resolution.

        Parameters
        ----------
        params : dict
            Training parameters (may contain ``project_path``).

        Returns
        -------
        dict or None
            The project configuration, or ``None`` when unavailable.
        """
        if not params.get("project_path"):
            return None
        cfg_path = Path(params["project_path"]) / "config.yaml"
        try:
            return load_yaml(cfg_path)
        except OSError:
            # Config genuinely unavailable (missing / unreadable) → best-effort
            # None so the PyTorch default applies. A malformed-but-present
            # config (YAMLError) is a real defect and is left to propagate.
            return None

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
        self,
        config: dict,
        model_instance,
        key: dict = None,
        params: dict = None,
    ) -> tuple[Path, str, Union[Path, None]]:
        """Localize the trained model and generate model ID.

        The train directory is resolved from *params* where they specify the
        model, falling back to the project config. ``shuffle`` and
        ``trainingsetindex`` are training parameters, so a model trained with
        an override is located in the folder that override produced.

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
        params : dict, optional
            Training parameters. ``shuffle``, ``trainingsetindex`` and
            ``modelprefix`` are read from here first.

        Returns
        -------
        tuple of (Path, str, Path or None)
            The model config path, the generated ``model_id``, and this
            model's newest snapshot file (``None`` when training produced no
            snapshot).

        Raises
        ------
        FileNotFoundError
            If no train directory exists under the project.
        IndexError
            If ``trainingsetindex`` exceeds the project's
            ``TrainingFraction`` list.
        """
        from deeplabcut.utils import get_model_folder
        from deeplabcut.utils.auxiliaryfunctions import read_config

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

        params = params or {}
        fractions = dlc_config.get("TrainingFraction", [0.95])
        set_index = int(params.get("trainingsetindex", 0))
        if set_index >= len(fractions):
            raise IndexError(
                f"trainingsetindex {set_index} exceeds the project's "
                f"{len(fractions)} TrainingFraction entries: {fractions}"
            )
        train_fraction = fractions[set_index]
        shuffle = int(params.get("shuffle", dlc_config.get("shuffle", 1)))
        model_prefix = params.get(
            "modelprefix", dlc_config.get("modelprefix", "")
        )

        train_dir = None
        for _engine in _engines:
            _kwargs = dict(
                trainFraction=train_fraction,
                shuffle=shuffle,
                cfg=dlc_config,
                modelprefix=model_prefix,
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
            snapshot_path = None
        else:
            # Find most recently modified snapshot
            latest_snapshot = 0
            snapshot_path = None
            max_modified_time = 0
            for snapshot in snapshots:
                modified_time = self._fs.getmtime(snapshot)
                if modified_time > max_modified_time:
                    # Extract snapshot number from filename (e.g. "snapshot-010")
                    import re as _re

                    _m = _re.search(r"(\d+)", snapshot.stem)
                    latest_snapshot = int(_m.group(1)) if _m else 0
                    snapshot_path = snapshot
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
        return config_path, model_id, snapshot_path

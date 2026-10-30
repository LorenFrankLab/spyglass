"""SLEAP implementation of the pose-tool strategy.

Every SLEAP import is deferred into the method that needs it, so this module
loads without SLEAP installed.
"""

from pathlib import Path
from typing import Any, Dict, Set

import datajoint as dj
import yaml

from .protocols import default_pk_name
from .tool_strategy_base import PoseToolStrategy


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
        checkpoint). SLEAP reads its training length from the config file, so
        a ``max_epochs``/``epochs`` budget in *params* is applied by writing a
        config copy and training against that.

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
        epochs=None,
    ) -> dict:
        """Run ``sleap-train`` and register the resulting model.

        The single implementation behind both fresh training and
        continuation, and the one place an epoch budget is turned into a
        config copy. With no *base_checkpoint* this is a plain fresh train.

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
        epochs : int or None, optional
            Training length. Takes precedence over ``params['max_epochs']``
            and ``params['epochs']``. When all are absent the config's own
            length is used.

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

        # SLEAP reads its training length from the config file, so a budget
        # is applied by writing a config copy and training against that.
        # An explicit *epochs* wins over one stored in *params*.
        eff_epochs = (
            epochs
            if epochs is not None
            else params.get("max_epochs", params.get("epochs"))
        )
        config_override = (
            self._write_epochs_config(
                params.get("initial_config"), int(eff_epochs), model_instance
            )
            if eff_epochs is not None
            else None
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
        checkpoint = self._resolve_parent_checkpoint(model_dir)

        return dict(
            key,
            model_id=model_id,
            analysis_file_name=None,
            model_path=str(model_dir),
            snapshot_path=str(checkpoint) if checkpoint else None,
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

        # delete=False: ``_run_training`` passes this config to the
        # out-of-process ``sleap-train`` subprocess, so it must outlive this
        # function. OS temp reaping cleans it up.
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

        model_instance._info_msg(
            f"Resuming SLEAP training from parent checkpoint: {checkpoint}"
        )
        # epochs: Model.train's epochs= arrives as params max_epochs, since
        # make() does not thread the kwarg through.
        return self._run_training(
            key,
            params,
            sel_entry,
            model_instance,
            base_checkpoint=checkpoint,
            epochs=epochs,
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
        cwd_candidate = Path.cwd() / "models" / run_name  # Fallback: search cwd
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

        Returns one entry per video, ``None`` where a video has no output.
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
                output_files.append(None)
                continue

            # Primary: {video_stem}*.analysis.h5
            analysis_files = [
                Path(p)
                for p in self._fs.glob(
                    str(
                        search_dir
                        / patterns["primary"].format(video_stem=video_path.stem)
                    )
                )
            ]
            if analysis_files:
                latest = max(analysis_files, key=self._fs.getmtime)
                output_files.append(str(latest))
                continue

            slp_files = [  # Fallback: {video_stem}*.predictions.slp
                Path(p)
                for p in self._fs.glob(
                    str(
                        search_dir
                        / patterns["fallback"].format(
                            video_stem=video_path.stem
                        )
                    )
                )
            ]
            if slp_files:
                latest = max(slp_files, key=self._fs.getmtime)
                output_files.append(str(latest))
            else:
                output_files.append(None)

        return output_files

    def get_output_file_patterns(self) -> Dict[str, str]:
        """Get SLEAP-specific output file patterns.

        Each pattern anchors on the video stem followed by a dot. Videos
        usually share one output directory, and without the dot ``vid1`` also
        matches ``vid10.analysis.h5``. SLEAP's own names
        (``<video>.mp4.predictions.slp``) still match.
        """
        return {
            "primary": "{video_stem}.*analysis.h5",  # SLEAP analysis export
            "fallback": "{video_stem}.*predictions.slp",  # embedded predictions
        }

"""DataJoint tables for pose estimation inference and pose output processing.

Typical workflow
----------------
1. Configure inference parameters and select a model + video group::

       PoseEstimParams.insert_params({...})          # or use default
       PoseEstimSelection().insert1({
           "model_id": ..., "vid_group_id": ...,
           "pose_estim_params_id": "default",
           "task_mode": "trigger", "output_dir": "/path/to/output"
       })
       PoseEstim().populate()

2. Configure pose processing (centroid, orientation, smoothing) and populate::

       PoseParams.insert_default()                   # or custom
       PoseSelection().insert1({
           **pose_estim_key, "pose_params_id": "default"
       })
       PoseV2().populate()

Results are registered in ``PositionOutput`` automatically and accessible via
``PositionOutput.fetch1_dataframe()``.
"""

import dataclasses
import subprocess
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Union

import datajoint as dj
import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO

from spyglass.common import AnalysisNwbfile
from spyglass.common.common_behav import VideoFile
from spyglass.position.utils.dlc_io import parse_dlc_h5_output
from spyglass.position.utils.validation import (
    validate_centroid_params,
    validate_orientation_params,
    validate_smoothing_params,
)
from spyglass.position.v2.train import (
    Model,
    ModelParams,
    ModelSelection,
    Skeleton,
    default_pk_name,
    resolve_model_path,
)
from spyglass.position.v2.utils.nwb_io import (
    NDXPoseBuilder,
    PoseInferenceRunner,
    _populate_nwb_pose_estimation,
)
from spyglass.position.v2.utils.params import (
    CentroidParams,
    OrientationParams,
    PoseParameterSet,
    SmoothingParams,
)
from spyglass.position.v2.video import VidFileGroup
from spyglass.settings import pose_output_dir
from spyglass.utils import SpyglassMixin, logger
from spyglass.utils.nwb_helper_fn import get_nwb_file

# ----------------------------- Optional imports ------------------------------
try:
    import ndx_pose
except ImportError:  # pragma: no cover
    ndx_pose = None  # pragma: no cover

schema = dj.schema("cbroz_position_v2_estim")


@schema
class PoseEstimParams(SpyglassMixin, dj.Lookup):
    """Parameters for pose estimation inference.

    Stores tool-specific inference parameters (batch_size, device, etc.)
    as reusable named parameter sets.

    Attributes
    ----------
    pose_estim_params_id : str
        Unique identifier for this parameter set (max 32 chars)
    params : json
        Tool-specific inference parameters dict
    params_hash : str
        Hash of params for deduplication
    """

    definition = """
    pose_estim_params_id: varchar(32)
    ---
    params: json          # tool-specific inference params
    params_hash: varchar(64)  # hash of parameters
    unique index (params_hash)
    """

    default_params = {}
    contents = [
        (
            "default",
            default_params,
            dj.hash.key_hash(default_params),
        ),
    ]

    @classmethod
    def insert_params(
        cls,
        params,
        params_id=None,
        skip_duplicates=True,
    ):
        """Insert a new inference parameter set.

        Parameters
        ----------
        params : dict
            Tool-specific inference parameters (batch_size, device, etc.)
        params_id : str, optional
            Name for this parameter set. If None, auto-generated using
            default_pk_name convention, by default None
        skip_duplicates : bool, optional
            Skip if entry already exists, by default True

        Returns
        -------
        dict
            Key with pose_estim_params_id
        """
        params_hash = dj.hash.key_hash(params)

        # Check if params already exist by hash
        existing = cls() & {"params_hash": params_hash}
        if existing:
            return existing.fetch1("KEY")

        if params_id is None:  # Use `pep-{YYYYMMDD}`
            params_id = default_pk_name(prefix="pep", include_hash=False)

        key = {
            "pose_estim_params_id": params_id,
            "params": params,
            "params_hash": params_hash,
        }
        cls.insert1(key, skip_duplicates=skip_duplicates)
        logger.info_msg(msg=f"Inserted PoseEstimParams: {params_id}")
        return {"pose_estim_params_id": params_id}


@schema
class PoseEstimSelection(SpyglassMixin, dj.Manual):
    """Selection table for pose estimation tasks.

    Pairs a trained Model with a VidFileGroup and inference parameters,
    specifying whether to trigger new inference or load existing results.

    Attributes
    ----------
    model_id : str
        Foreign key to Model table
    vid_group_id : str
        Foreign key to VidFileGroup table
    pose_estim_params_id : str
        Foreign key to PoseEstimParams table
    task_mode : enum
        'trigger' to run inference, 'load' to load existing tool-native output
        files (e.g. DLC .h5, SLEAP output) already on disk. Use 'load' when
        inference was run outside Spyglass but with a supported tool; use
        ``ImportedPose`` instead when results already exist in NWB format.
    output_dir : str
        Results directory (load: existing, trigger: destination)
    """

    definition = """
    -> Model
    -> VidFileGroup
    -> PoseEstimParams
    ---
    task_mode='trigger': enum('load', 'trigger')
    output_dir='': varchar(255)       # results dir
    """

    def insert_estimation_task(
        self,
        key,
        task_mode="trigger",
        params=None,
        output_dir=None,
        skip_duplicates=True,
    ):
        """Insert a pose estimation task.

        Parameters
        ----------
        key : dict
            Must include model_id and vid_group_id. May include
            pose_estim_params_id; if not, one is resolved from params.
        task_mode : str, optional
            'trigger' to run inference, 'load' to load existing results,
            by default 'trigger'
        params : dict, optional
            Tool-specific inference parameters (batch_size, device, etc.).
            Ignored if key already contains pose_estim_params_id.
        output_dir : str, optional
            Output directory. If None, inferred from model and video info
        skip_duplicates : bool, optional
            Skip if entry already exists, by default True

        Returns
        -------
        dict
            The full key (for chaining into populate)
        """
        # Resolve params to a PoseEstimParams entry
        if "pose_estim_params_id" not in key:
            params = params or {}
            params_key = PoseEstimParams.insert_params(
                params, skip_duplicates=True
            )
            key = {**key, **params_key}

        if output_dir is None:
            output_dir = self._infer_output_dir(key)

        if "vid_group_id" not in key:  # pragma: no cover
            raise ValueError(  # pragma: no cover
                "key must include 'vid_group_id'. "
                "Use VidFileGroup().insert1() to create a group first."
            )
        if not (
            VidFileGroup() & {"vid_group_id": key["vid_group_id"]}
        ):  # pragma: no cover
            raise ValueError(  # pragma: no cover
                f"VidFileGroup '{key['vid_group_id']}' not found. "
                "Create it first with VidFileGroup().insert1() or "
                "VidFileGroup.create_from_files()."
            )

        self.insert1(
            {
                **key,
                "task_mode": task_mode,
                "output_dir": str(output_dir),
            },
            skip_duplicates=skip_duplicates,
        )

        self._info_msg("Inserted entry into PoseEstimSelection")
        return {**key, "task_mode": task_mode}

    @staticmethod
    def _infer_output_dir(key):
        """Infer output directory from model and video group info.

        Parameters
        ----------
        key : dict
            Must include model_id and vid_group_id

        Returns
        -------
        str
            Inferred output directory path
        """
        model_id = key.get("model_id", "unknown_model")

        # Try to use the configured pose_output_dir
        base_dir = pose_output_dir
        if not base_dir:
            base_dir = Path.cwd() / "pose_estimation_output"

        output_dir = (
            Path(base_dir) / f"v2_{model_id}_{key.get('vid_group_id', '')}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        return str(output_dir)


@schema
class PoseEstim(SpyglassMixin, dj.Computed):
    """Pose estimation results from model inference on video files.

    Computed from PoseEstimSelection entries. Runs inference (trigger)
    or loads existing results (load), stores in NWB via AnalysisNwbfile.

    Uses helper classes for separation of concerns:
    - PoseInferenceRunner: DLC/SLEAP inference execution
    - NDXPoseBuilder: NWB serialization

    Attributes
    ----------
    model_id : str
        Foreign key via PoseEstimSelection -> Model
    vid_group_id : str
        Foreign key via PoseEstimSelection -> VidFileGroup
    analysis_file_name : str
        Foreign key to AnalysisNwbfile (nullable)
    """

    definition = """
    -> PoseEstimSelection
    ---
    -> AnalysisNwbfile
    """

    # Dependency injection - default to real implementations
    # Tests can override these to inject stubs without unittest.mock.patch
    _inference_runner_cls = None  # Will be set to PoseInferenceRunner
    _nwb_builder_cls = None  # Will be set to NDXPoseBuilder

    @classmethod
    def _get_inference_runner_cls(cls):
        """Get the inference runner class, defaulting to PoseInferenceRunner."""
        if cls._inference_runner_cls is None:
            return PoseInferenceRunner
        return cls._inference_runner_cls

    @classmethod
    def _get_nwb_builder_cls(cls):
        """Get the NWB builder class, defaulting to NDXPoseBuilder."""
        if cls._nwb_builder_cls is None:
            return NDXPoseBuilder
        return cls._nwb_builder_cls

    def insert1(self, row, **kwargs):
        """Insert a single row, allowing direct inserts outside of populate.

        Defaults pose_estim_params_id to 'default' when not provided. Also
        creates a PoseEstimSelection entry (task_mode='load') if one doesn't
        already exist, so callers can insert with just {model_id, vid_group_id}.
        """
        kwargs.setdefault("allow_direct_insert", True)
        row = dict(row)
        row.setdefault("pose_estim_params_id", "default")

        sel_fields = ["model_id", "vid_group_id", "pose_estim_params_id"]
        sel_key = {k: row[k] for k in sel_fields if k in row}
        if len(sel_key) == 3 and not PoseEstimSelection() & sel_key:
            PoseEstimSelection().insert1(
                {**sel_key, "task_mode": "load", "output_dir": ""},
                skip_duplicates=True,
            )

        super().insert1(row, **kwargs)

    @staticmethod
    def load_dlc_output(
        dlc_output_path: Union[Path, str],
        nwb_file_name: Union[Path, str, None] = None,
        pose_estimation_name: str = "PoseEstimation",
        timestamps: np.ndarray = None,
    ) -> Path:
        """Load DLC inference output (h5 or csv) into NWB file with ndx-pose.

        Parameters
        ----------
        dlc_output_path : Union[Path, str]
            Path to DLC output file (.h5 or .csv)
        nwb_file_name : Union[Path, str, None], optional
            Name of NWB file to create/update. If None, generates a name based
            on the DLC output file, by default None
        pose_estimation_name : str, optional
            Name for the PoseEstimation object in NWB. Default: "PoseEstimation"
        Returns
        -------
        Path
            Path to the created/updated NWB file

        Raises
        ------
        ImportError
            If ndx-pose is not installed
        FileNotFoundError
            If dlc_output_path doesn't exist
        ValueError
            If file format is not supported

        Examples
        --------
        >>> # Load DLC output into new NWB file
        >>> nwb_path = PoseEstim.load_dlc_output(
        ...     "video1DLC_resnet50_model_shuffle1_100.h5"
        ... )
        >>>
        >>> # Load into specific NWB file
        >>> nwb_path = PoseEstim.load_dlc_output(
        ...     "video1DLC_resnet50_model_shuffle1_100.h5",
        ...     nwb_file_name="analysis_20250101.nwb"
        ... )
        """
        if ndx_pose is None:  # pragma: no cover
            raise ImportError(  # pragma: no cover
                "ndx-pose is required to load DLC output into NWB. "
                "Install with: pip install ndx-pose>=0.2.0"
            )

        dlc_output_path = Path(dlc_output_path)
        if not dlc_output_path.exists():
            raise FileNotFoundError(f"DLC output not found: {dlc_output_path}")

        logger.info_msg(f"Loading DLC output: {dlc_output_path}")

        # Load DLC output data using consolidated parser
        df, scorer, bodyparts = parse_dlc_h5_output(dlc_output_path)

        logger.info_msg(
            f"DLC data: {len(bodyparts)} bodyparts, {len(df)} frames, "
            f"scorer: {scorer}"
        )

        # Determine NWB file path
        if nwb_file_name is None:
            nwb_file_name = (
                f"{dlc_output_path.stem}_analysis_{datetime.now():%Y%m%d}.nwb"
            )

        nwb_path = (
            Path(nwb_file_name)
            if Path(nwb_file_name).is_absolute()
            else dlc_output_path.parent / nwb_file_name
        )

        # Create or load NWB file
        if nwb_path.exists():
            logger.info_msg(f"Updating existing NWB file: {nwb_path}")
            io_mode = "r+"
        else:
            logger.info_msg(f"Creating new NWB file: {nwb_path}")
            io_mode = "w"

        # Resolve timestamps before opening the file (raises early on error)
        if df.index.name == "time":
            _timestamps = df.index.values
        elif timestamps is not None:
            _timestamps = timestamps
        else:
            raise ValueError(
                "No 'time' index in DLC output and no timestamps "
                "provided to load_dlc_output; refusing to fall back to "
                "frame indices. Pass timestamps fetched from the "
                "registered VideoFile NWB."
            )

        with NWBHDF5IO(
            str(nwb_path), mode=io_mode, load_namespaces=(io_mode != "w")
        ) as io:
            if io_mode == "r+":
                nwbfile = io.read()
            else:
                from pynwb import NWBFile

                nwbfile = NWBFile(
                    session_description=(
                        f"Pose estimation from DLC {dlc_output_path.name}"
                    ),
                    identifier=f"pose_{datetime.now():%Y%m%d%H%M%S}",
                    session_start_time=datetime.now(),
                )

            _populate_nwb_pose_estimation(
                nwbfile,
                df,
                scorer,
                bodyparts,
                pose_estimation_name,
                dlc_output_path,
                _timestamps,
            )
            io.write(nwbfile)

        logger.info_msg(f"DLC output loaded into NWB: {nwb_path}")
        return nwb_path

    @staticmethod
    def load_from_nwb(
        nwb_file_path: Union[Path, str],
        pose_estimation_name: str = None,
    ) -> dict:
        """Load pose estimation data from existing ndx-pose NWB file.

        This method validates and extracts metadata from NWB files that already
        contain ndx-pose PoseEstimation data (e.g., from SLEAP, pre-processed
        data, or previously created files).

        Parameters
        ----------
        nwb_file_path : Union[Path, str]
            Path to NWB file containing ndx-pose PoseEstimation data
        pose_estimation_name : str, optional
            Name of specific PoseEstimation object to load. If None, uses the
            first PoseEstimation found, by default None

        Returns
        -------
        dict
            Dictionary with keys:
            - nwb_file_path: Path to the NWB file
            - pose_estimation_name: Name of the PoseEstimation object
            - bodyparts: List of bodypart names
            - n_frames: Number of frames
            - scorer: Scorer/software name
            - source_software: Source software (e.g., "DeepLabCut", "SLEAP")

        Raises
        ------
        ImportError
            If ndx-pose is not installed
        FileNotFoundError
            If nwb_file_path doesn't exist
        ValueError
            If NWB file doesn't contain valid ndx-pose data

        Examples
        --------
        >>> # Load pose data from SLEAP NWB file
        >>> metadata = PoseEstim.load_from_nwb("sleap_output.nwb")
        >>> print(f"Found {len(metadata['bodyparts'])} bodyparts")
        >>>
        >>> # Load specific PoseEstimation by name
        >>> metadata = PoseEstim.load_from_nwb(
        ...     "multi_pose.nwb",
        ...     pose_estimation_name="PoseEstimation_camera1"
        ... )
        """
        if ndx_pose is None:  # pragma: no cover
            raise ImportError(  # pragma: no cover
                "ndx-pose is required to load pose data from NWB. "
                "Install with: pip install ndx-pose>=0.2.0"
            )

        nwb_file_path = Path(nwb_file_path)
        logger.info_msg(f"Loading pose data from NWB: {nwb_file_path}")

        nwbfile = get_nwb_file(str(nwb_file_path))

        if "behavior" not in nwbfile.processing:  # pragma: no cover
            raise ValueError(  # pragma: no cover
                f"No behavior module in NWB file: {nwb_file_path}. "
                "Expected ndx-pose PoseEstimation data in behavior module."
            )

        behavior_module = nwbfile.processing["behavior"]

        # Find PoseEstimation objects
        pose_estimations = {
            name: obj
            for name, obj in behavior_module.data_interfaces.items()
            if isinstance(obj, ndx_pose.PoseEstimation)
        }

        if not pose_estimations:  # pragma: no cover
            raise ValueError(  # pragma: no cover
                f"No PoseEstimation objects found in NWB: {nwb_file_path}. "
                "File must contain ndx-pose.PoseEstimation data."
            )

        # Select PoseEstimation
        if pose_estimation_name is not None:
            if pose_estimation_name not in pose_estimations:  # pragma: no cover
                available = ", ".join(
                    pose_estimations.keys()
                )  # pragma: no cover
                raise ValueError(  # pragma: no cover
                    f"PoseEstimation '{pose_estimation_name}' not found in NWB."
                    f" Available: {available}"
                )
            pose_estimation = pose_estimations[pose_estimation_name]
            selected_name = pose_estimation_name
        else:
            selected_name = list(pose_estimations.keys())[0]
            pose_estimation = pose_estimations[selected_name]
            if len(pose_estimations) > 1:
                logger.warn_msg(
                    f"Multiple PoseEstimation objects found. "
                    f"Using '{selected_name}'. "
                    f"Available: {', '.join(pose_estimations.keys())}"
                )

        scorer = getattr(pose_estimation, "scorer", "unknown")
        source_software = getattr(pose_estimation, "source_software", "unknown")

        bodyparts = []
        n_frames = None

        for series in pose_estimation.pose_estimation_series.values():
            bodypart = series.name.replace("_pose", "")
            bodyparts.append(bodypart)
            if n_frames is None:
                n_frames = series.data.shape[0]

        metadata = {
            "nwb_file_path": nwb_file_path,
            "pose_estimation_name": selected_name,
            "bodyparts": bodyparts,
            "n_frames": n_frames,
            "scorer": scorer,
            "source_software": source_software,
        }

        logger.info_msg(
            f"Loaded pose data: {len(bodyparts)} bodyparts, {n_frames} frames, "
            f"source: {source_software}"
        )

        return metadata

    def fetch1_dataframe(self):
        """Fetch pose estimation data as a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            Pose data with MultiIndex columns [bodypart, coords] where coords
            are [x, y, likelihood]

        Raises
        ------
        ImportError
            If ndx-pose is not installed
        """
        if ndx_pose is None:  # pragma: no cover
            raise ImportError(  # pragma: no cover
                "ndx-pose is required to fetch pose data. "
                "Install with: pip install ndx-pose>=0.2.0"
            )

        nwb_file_name = self.fetch1("analysis_file_name")
        self._logger.debug("Fetching pose data from NWB: %s", nwb_file_name)

        # Use standard Spyglass path resolution (handles DANDI streaming,
        # logging, and caching) instead of a custom AnalysisNwbfile lookup.
        nwb_data = self.fetch_nwb("analysis_file_name")[0]
        nwbfile = get_nwb_file(nwb_data["nwb2load_filepath"])

        if "behavior" not in nwbfile.processing:
            raise ValueError(
                f"No behavior module in NWB: {nwb_file_name}. "
                "The analysis file may be corrupted or was created by an "
                "older pipeline. Delete the PoseEstim entry and re-populate: "
                "``(PoseEstim & key).delete(); populate(key)``"
            )

        behavior_module = nwbfile.processing["behavior"]

        # Find PoseEstimation objects
        pose_estimations = {
            name: obj
            for name, obj in behavior_module.data_interfaces.items()
            if isinstance(obj, ndx_pose.PoseEstimation)
        }

        if not pose_estimations:
            raise ValueError(
                f"No PoseEstimation in NWB: {nwb_file_name}. "
                "Expected an ndx-pose PoseEstimation object in the behavior "
                "processing module. Delete the PoseEstim entry and "
                "re-populate: ``(PoseEstim & key).delete(); populate(key)``"
            )

        # Use first PoseEstimation (or could select by name)
        pose_estimation = list(pose_estimations.values())[0]
        scorer = getattr(pose_estimation, "scorer", "DLC_scorer")

        # Build DataFrame from PoseEstimationSeries
        data_dict = {}

        for series in pose_estimation.pose_estimation_series.values():
            bodypart = series.name.replace("_pose", "")

            # Extract x, y data (shape: n_frames, 2)
            pose_data = series.data[:]
            x_data = pose_data[:, 0]
            y_data = pose_data[:, 1]

            # Extract confidence
            confidence = series.confidence[:]

            # Add to dictionary with MultiIndex keys
            data_dict[(scorer, bodypart, "x")] = x_data
            data_dict[(scorer, bodypart, "y")] = y_data
            data_dict[(scorer, bodypart, "likelihood")] = confidence

        # Create DataFrame with MultiIndex columns
        df = pd.DataFrame(data_dict)
        df.columns = pd.MultiIndex.from_tuples(
            df.columns, names=["scorer", "bodyparts", "coords"]
        )

        self._logger.debug(f"Fetched {len(df)} frames of pose data")

        return df

    def make(self, key):
        """Run or load pose estimation for a Model + VidFileGroup pairing.

        Flow:
        1. Fetch task_mode, output_dir, inference_params from Selection
        2. If trigger: run inference via self.run_inference()
        3. Find DLC h5 output in output_dir
        4. Load into ndx-pose NWB via _store_estimation_nwb()
        5. Insert entry with analysis_file_name

        Parameters
        ----------
        key : dict
            Primary key from PoseEstimSelection
        """
        if ndx_pose is None:  # pragma: no cover
            raise ImportError(  # pragma: no cover
                "ndx-pose is required for pose estimation. "
                "Install with: pip install ndx-pose>=0.2.0"
            )

        # Fetch selection entry
        task_mode, output_dir = (PoseEstimSelection & key).fetch1(
            "task_mode", "output_dir"
        )

        # Fetch model info to determine tool type (use only model_id)
        model_key = {"model_id": key["model_id"]}
        model_info = (Model() & model_key).fetch1()
        model_params_key = {
            "model_params_id": model_info["model_params_id"],
            "tool": model_info["tool"],
        }
        tool_info = (ModelParams() & model_params_key).fetch1()
        tool = tool_info["tool"]

        # Fetch inference params from PoseEstimParams
        inference_params = (PoseEstimParams & key).fetch1("params")
        inference_params = inference_params or {}

        self._info_msg(
            "PoseEstim.make: "
            + f"tool={tool}, mode={task_mode}, output_dir={output_dir}"
        )

        # Resolve video paths (needed for trigger mode inference)
        video_paths = []
        for vf_key in (VidFileGroup.File & key).fetch("KEY"):
            vf_entry = (VideoFile & vf_key).fetch1()
            if vf_entry.get("path"):
                video_paths.append(vf_entry["path"])

        # Resolve nwb_file_name via the VidFileGroup → Session link.
        # Returns None when no common NWB parent exists (e.g. tutorial mode).
        nwb_file_name = None
        try:
            nwb_file_name = VidFileGroup().get_nwb_file(key["vid_group_id"])[
                "nwb_file_name"
            ]
        except ValueError:
            pass

        # Step 1: Fail fast if no NWB parent (before any expensive I/O)
        if nwb_file_name is None:
            raise ValueError(
                "Cannot store pose estimation: no registered Nwbfile linked "
                f"to VidFileGroup '{key['vid_group_id']}'. Ensure the video "
                "files in this group are registered in VideoFile and linked "
                "to a Session via TaskEpoch. Use "
                "VidFileGroup().get_nwb_file() to verify before populating."
            )

        # Step 2: Trigger inference if needed and get output file info
        output_file_info = None
        if task_mode == "trigger":
            if not video_paths:
                raise ValueError(
                    f"No video files registered for vid_group_id="
                    f"{key['vid_group_id']}. Cannot trigger inference "
                    "without registered video files in VideoFile table."
                )
            self._info_msg("Triggering inference...")
            model_key = {"model_id": key["model_id"]}
            destfolder = output_dir if output_dir else None
            output_file_info = self.run_inference(
                model_key,
                video_paths if len(video_paths) > 1 else video_paths[0],
                destfolder=destfolder,
                **inference_params,
            )

        # Step 3: Find output files using tool-specific logic
        output_files = self._find_output_files(
            tool, video_paths, output_dir, output_file_info
        )

        if not output_files:
            raise FileNotFoundError(
                f"No output files found for {tool} inference. "
                f"Video paths: {video_paths}, output_dir: {output_dir}. "
                "For DLC, check that analyze_videos completed without error "
                "and that output_dir matches the destfolder used during "
                "inference. You can re-trigger by setting task_mode='trigger' "
                "in PoseEstimSelection and calling populate() again."
            )

        # Use the most recent output file
        primary_output_file = max(
            output_files, key=lambda p: Path(p).stat().st_mtime
        )
        self._info_msg(f"Loading {tool} output: {primary_output_file}")

        # Load pose data using tool-specific loader
        pose_df, scorer, bodyparts = self._load_pose_data(
            tool, primary_output_file
        )
        self._info_msg(
            f"DLC data: {len(bodyparts)} bodyparts, {len(pose_df)} frames, "
            f"scorer: {scorer}"
        )

        # Step 5: Store data in AnalysisNwbfile
        analysis_file_name = self._store_estimation_nwb(
            key, pose_df, bodyparts, scorer, nwb_file_name
        )
        self.insert1({**key, "analysis_file_name": analysis_file_name})
        self._info_msg("PoseEstim entry inserted.")

    def _find_output_files(
        self,
        tool: str,
        video_paths: list,
        output_dir: str,
        output_file_info: Union[str, list, None] = None,
    ) -> list:
        """Find output files using tool-specific strategy pattern.

        Parameters
        ----------
        tool : str
            Tool name (e.g., 'DLC')
        video_paths : list
            List of video file paths that were analyzed
        output_dir : str
            User-specified output directory (may be empty)
        output_file_info : Union[str, list, None]
            Output from run_inference (actual paths created)

        Returns
        -------
        list
            List of output file paths
        """
        # Import strategy factory
        from spyglass.position.utils.tool_strategies import ToolStrategyFactory

        # Get strategy for this tool
        try:
            strategy = ToolStrategyFactory.create_strategy(tool)
        except ValueError as e:
            available_tools = ToolStrategyFactory.get_available_tools()
            raise ValueError(
                f"Tool '{tool}' not supported. Available: {available_tools}"
            ) from e

        # Use strategy's output file discovery
        return strategy.find_output_files(
            video_paths=video_paths,
            output_dir=output_dir,
            output_file_info=output_file_info,
        )

    def _load_pose_data(self, tool: str, output_file: str):
        """Load pose data using tool-specific loaders.

        Parameters
        ----------
        tool : str
            Tool name (e.g., 'DLC')
        output_file : str
            Path to output file

        Returns
        -------
        tuple
            (pose_df, scorer, bodyparts)
        """
        if tool == "DLC":
            return parse_dlc_h5_output(output_file, return_metadata=True)
        else:
            raise ValueError(f"Unsupported tool for data loading: {tool}")

    def run_inference(
        self,
        model_key: dict,
        video_path: Union[Path, str, list],
        save_as_csv: bool = False,
        destfolder: Union[Path, str, None] = None,
        **kwargs,
    ) -> Union[str, list]:
        """Run pose estimation inference on video(s) using a trained model.

        Parameters
        ----------
        model_key : dict
            Primary key for the Model table entry (must include 'model_id')
        video_path : Union[Path, str, list]
            Path to video file(s) for inference
        save_as_csv : bool, optional
            Whether to save output as CSV, by default False
        destfolder : Union[Path, str, None], optional
            Destination folder for output files, by default None
        **kwargs
            Additional parameters for underlying inference function

        Returns
        -------
        Union[str, list]
            Path(s) to output file(s)
        """
        if not Model() & model_key:
            raise ValueError(
                f"Model not found in database: {model_key}. "
                "Please import model first using Model.load()"
            )

        model_info = (Model() & model_key).fetch1()
        model_params_key = {
            "model_params_id": model_info["model_params_id"],
            "tool": model_info["tool"],
        }
        params_info = (ModelParams() & model_params_key).fetch1()
        tool = params_info["tool"]

        self._info_msg(f"Running inference with {tool} model: {model_key}")

        if tool == "DLC":
            runner = self._get_inference_runner_cls()()
            return runner.run_dlc_inference(
                model_info, video_path, save_as_csv, destfolder, **kwargs
            )
        elif tool == "ndx-pose":
            raise NotImplementedError(
                "Inference from ndx-pose NWB files not yet supported. "
                "Please use the original DLC project for inference."
            )
        else:
            raise NotImplementedError(
                f"Inference not supported for tool: {tool}"
            )

    @staticmethod
    def _fetch_video_timestamps(key: dict) -> np.ndarray:
        """Fetch per-frame timestamps from the first VideoFile in the group.

        Parameters
        ----------
        key : dict
            Must include vid_group_id.

        Returns
        -------
        np.ndarray
            Per-frame timestamps in seconds.

        Raises
        ------
        ValueError
            If no VideoFile entries exist or timestamps cannot be retrieved.
        """
        vf_keys = (VidFileGroup.File & key).fetch("KEY", as_dict=True)
        if not vf_keys:
            raise ValueError(
                f"No VideoFile entries found for vid_group_id="
                f"'{key['vid_group_id']}'. Cannot fetch real timestamps."
            )
        try:
            nwb_data = (VideoFile & vf_keys[0]).fetch_nwb()[0]
            ts = np.asarray(nwb_data["video_file"].timestamps)
        except (OSError, KeyError, TypeError, AttributeError) as exc:
            raise ValueError(
                f"Could not fetch timestamps from VideoFile for "
                f"vid_group_id='{key['vid_group_id']}': {exc}"
            ) from exc
        if ts is None or len(ts) == 0:
            raise ValueError(
                f"VideoFile for vid_group_id='{key['vid_group_id']}' has "
                "empty or null timestamps."
            )
        return ts

    def _store_estimation_nwb(
        self,
        key: dict,
        pose_df: pd.DataFrame,
        bodyparts: list,
        scorer: str,
        nwb_file_name: str,
    ) -> str:
        """Store pose estimation data in AnalysisNwbfile using helper classes.

        Parameters
        ----------
        key : dict
            Primary key
        pose_df : pd.DataFrame
            DLC output DataFrame with MultiIndex columns
        bodyparts : list
            List of bodypart names
        scorer : str
            Scorer/model name from DLC output
        nwb_file_name : str
            Name of the source NWB file

        Returns
        -------
        str
            The analysis_file_name for the created NWB file
        """
        # Create analysis NWB file
        analysis_file_name = AnalysisNwbfile().create(nwb_file_name)
        nwb_analysis_file = AnalysisNwbfile()

        # Fetch skeleton info from ModelParams via Model join
        model_info = (
            Model * ModelParams & {"model_id": key["model_id"]}
        ).fetch1()
        skeleton_key = {"skeleton_id": model_info["skeleton_id"]}
        skeleton_entry = (Skeleton & skeleton_key).fetch1()
        skeleton_edges = skeleton_entry.get("edges", [])

        # Get video information for pose estimation
        video_keys = [str(p) for p in (VidFileGroup.File & key).fetch("KEY")]

        # Build pose estimation using helper class
        vid_timestamps = self._fetch_video_timestamps(key)
        builder = self._get_nwb_builder_cls()()

        # Derive canonical software name from the active tool strategy so
        # ndx-pose NWB metadata reflects the actual tool used (e.g.
        # "DeepLabCut" for DLC, not the internal abbreviation).
        from spyglass.position.utils.tool_strategies import ToolStrategyFactory

        try:
            active_strategy = ToolStrategyFactory.create_strategy(
                model_info["tool"]
            )
            source_sw = active_strategy.source_software
        except (ValueError, KeyError):
            source_sw = model_info.get("tool", "unknown")

        pose_estimation, nwb_skeleton = builder.build_pose_estimation(
            pose_df=pose_df,
            bodyparts=bodyparts,
            scorer=scorer,
            model_id=key["model_id"],
            skeleton_edges=skeleton_edges,
            description=f"Pose estimation from model {key['model_id']}",
            original_videos=video_keys,
            timestamps=vid_timestamps,
            source_software=source_sw,
        )

        # original_videos now set in builder, no need to set again here

        # Store to NWB using helper class
        analysis_abs_path = nwb_analysis_file.get_abs_path(analysis_file_name)
        builder.store_to_nwb(pose_estimation, nwb_skeleton, analysis_abs_path)

        nwb_analysis_file.add(
            nwb_file_name=nwb_file_name,
            analysis_file_name=analysis_file_name,
        )

        self._info_msg(f"Stored pose estimation in NWB: {analysis_file_name}")
        return analysis_file_name


@schema
class PoseParams(SpyglassMixin, dj.Lookup):
    """Parameters for pose processing (orientation, centroid, smoothing).

    Attributes
    ----------
    pose_params_id : str
        Name for this parameter set
    orient : dict
        Orientation calculation parameters
    centroid : dict
        Centroid calculation parameters
    smoothing : dict
        Smoothing and interpolation parameters
    """

    definition = """
    pose_params_id: varchar(32)
    ---
    orient: json  # Orientation calculation params
    centroid: json  # Centroid calculation params
    smoothing: json  # Smoothing/interpolation params
    """

    # -- Supported orient methods and their required keys --
    orient_methods = {
        "two_pt": ["bodypart1", "bodypart2"],
        "bisector": ["led1", "led2", "led3"],
        "none": [],
    }

    # -- Supported centroid methods: n_points, required point keys, extra keys
    centroid_methods = {
        "1pt": {
            "n_points": 1,
            "required_point_keys": None,  # any single key
            "extra_required": [],
        },
        "2pt": {
            "n_points": 2,
            "required_point_keys": None,  # any two keys
            "extra_required": ["max_LED_separation"],
        },
        "4pt": {
            "n_points": 4,
            "required_point_keys": {
                "greenLED",
                "redLED_C",
                "redLED_L",
                "redLED_R",
            },
            "extra_required": ["max_LED_separation"],
        },
    }

    # -- Smoothing methods (validated at runtime via SMOOTHING_METHODS) --
    smoothing_required = ["likelihood_thresh"]

    @classmethod
    def insert_default(cls, **kwargs):
        """Insert default parameter set for 2-LED tracking (green + red).

        This is the standard Frank Lab configuration with:
        - Two-point orientation (green → red)
        - Two-point centroid (average of green and red)
        - Moving average smoothing with interpolation
        """
        default_params = {
            "pose_params_id": "default",
            "orient": {
                "method": "two_pt",
                "bodypart1": "greenLED",
                "bodypart2": "redLED_C",
                "interpolate": True,
                "smooth": True,
                "smoothing_params": {
                    "std_dev": 0.001,  # 1ms Gaussian smoothing
                },
            },
            "centroid": {
                "method": "2pt",
                "points": {
                    "point1": "greenLED",
                    "point2": "redLED_C",
                },
                "max_LED_separation": 15.0,  # cm
            },
            "smoothing": {
                "interpolate": True,
                "interp_params": {
                    "max_pts_to_interp": 10,
                    "max_cm_to_interp": 15.0,
                },
                "smooth": True,
                "smoothing_params": {
                    "method": "moving_avg",
                    "smoothing_duration": 0.05,  # 50ms window
                },
                "likelihood_thresh": 0.95,
            },
        }
        cls.insert1(default_params, **kwargs)

    @classmethod
    def insert_4LED_default(cls, **kwargs):
        """Insert default for 4-LED tracking (1 green + 3 red).

        Standard Frank Lab 4-LED configuration:
        - Bisector orientation (perpendicular to left/right LEDs)
        - 4-point centroid (priority-based averaging)
        - Moving average smoothing with interpolation
        """
        params_4led = {
            "pose_params_id": "4LED_default",
            "orient": {
                "method": "bisector",
                "led1": "redLED_L",
                "led2": "redLED_R",
                "led3": "greenLED",
                "interpolate": True,
                "smooth": True,
                "smoothing_params": {
                    "std_dev": 0.001,  # 1ms Gaussian smoothing
                },
            },
            "centroid": {
                "method": "4pt",
                "points": {
                    "greenLED": "greenLED",
                    "redLED_C": "redLED_C",
                    "redLED_L": "redLED_L",
                    "redLED_R": "redLED_R",
                },
                "max_LED_separation": 15.0,  # cm
            },
            "smoothing": {
                "interpolate": True,
                "interp_params": {
                    "max_pts_to_interp": 10,
                    "max_cm_to_interp": 15.0,
                },
                "smooth": True,
                "smoothing_params": {
                    "method": "moving_avg",
                    "smoothing_duration": 0.05,  # 50ms window
                },
                "likelihood_thresh": 0.95,
            },
        }
        cls.insert1(params_4led, **kwargs)

    @classmethod
    def insert_single_LED(cls, **kwargs):
        """Insert params for single LED tracking (no orientation).

        For cases where only position tracking is needed:
        - No orientation calculation
        - 1-point centroid (passthrough)
        - Savitzky-Golay smoothing
        """
        params_single = {
            "pose_params_id": "single_LED",
            "orient": {
                "method": "none",
            },
            "centroid": {
                "method": "1pt",
                "points": {
                    "point1": "LED",
                },
            },
            "smoothing": {
                "interpolate": True,
                "interp_params": {
                    "max_pts_to_interp": 5,
                    "max_cm_to_interp": 10.0,
                },
                "smooth": True,
                "smoothing_params": {
                    "method": "savgol",
                    "window_length": 11,
                    "polyorder": 3,
                },
                "likelihood_thresh": 0.95,
            },
        }
        cls.insert1(params_single, **kwargs)

    @classmethod
    def insert_no_smoothing(cls, **kwargs):
        """Insert params with no smoothing (raw pose only).

        For testing or when smoothing is not desired:
        - Two-point orientation (no smoothing)
        - Two-point centroid
        - No interpolation or smoothing
        """
        params_raw = {
            "pose_params_id": "no_smoothing",
            "orient": {
                "method": "two_pt",
                "bodypart1": "greenLED",
                "bodypart2": "redLED_C",
                "interpolate": False,
                "smooth": False,
            },
            "centroid": {
                "method": "2pt",
                "points": {
                    "point1": "greenLED",
                    "point2": "redLED_C",
                },
                "max_LED_separation": 15.0,
            },
            "smoothing": {
                "interpolate": False,
                "smooth": False,
                "likelihood_thresh": 0.95,
            },
        }
        cls.insert1(params_raw, **kwargs)

    def insert1(self, key, **kwargs):
        """Insert pose parameters with validation.

        Standard DataJoint insert1 interface with added parameter validation.
        Validates orientation, centroid, and smoothing params before insertion.

        Parameters
        ----------
        key : dict
            Parameter dictionary with:
            - pose_params_id: str (name for parameter set)
            - orient: dict (orientation parameters)
            - centroid: dict (centroid parameters)
            - smoothing: dict (smoothing parameters)
        **kwargs
            Additional DataJoint insertion options

        Examples
        --------
        >>> # Standard DataJoint interface
        >>> PoseParams().insert1({
        ...     "pose_params_id": "my_config",
        ...     "orient": {
        ...         "method": "two_pt", "bodypart1": "nose", "bodypart2": "tail"
        ...     },
        ...     "centroid": {"method": "1pt", "points": {"point1": "nose"}},
        ...     "smoothing": {
        ...         "interpolate": False,
        ...         "smooth": False,
        ...         "likelihood_thresh": 0.9,
        ...     },
        ... })

        >>> # Or use class method
        >>> PoseParams.insert1({
        ...     "pose_params_id": "my_config",
        ...     "orient": {"method": "none"},
        ...     "centroid": {"method": "1pt", "points": {"point1": "nose"}},
        ...     "smoothing": {
        ...         "interpolate": False,
        ...         "smooth": False,
        ...         "likelihood_thresh": 0.9,
        ...     }
        ... })
        """
        # Validate parameter structure
        required_keys = {"pose_params_id", "orient", "centroid", "smoothing"}
        missing_keys = required_keys - set(key.keys())
        if missing_keys:
            raise ValueError(f"Missing required keys: {missing_keys}")

        # Extract and validate each component using validation.py functions.
        # Each field may be a typed dataclass or a plain dict; validate_*
        # functions accept both and _to_dict_if_dataclass normalises.
        orient = key["orient"]
        centroid = key["centroid"]
        smoothing = key["smoothing"]

        if orient is not None:
            validate_orientation_params(orient)
        if centroid is not None:
            validate_centroid_params(centroid)
        if smoothing is not None:
            validate_smoothing_params(smoothing)

        # Serialize typed dataclasses to plain dicts before DB insertion.
        key = dict(key)
        for field_name in ("orient", "centroid", "smoothing"):
            val = key[field_name]
            if (
                val is not None
                and dataclasses.is_dataclass(val)
                and not isinstance(val, type)
            ):
                key[field_name] = val.to_dict()

        # Call parent insert1 with validated parameters
        # Use the DataJoint Table base class directly
        return dj.Table.insert1(self, key, **kwargs)

    @classmethod
    def insert_custom(cls, params_name, orient, centroid, smoothing, **kwargs):
        """Convenience method for parameter insertion with individual arguments.

        Accepts plain dicts or typed dataclasses from
        :mod:`spyglass.position.utils.validation`
        (e.g. :class:`~TwoPtOrientParams`, :class:`~TwoPtCentroidParams`,
        :class:`~SmoothingParams`).  Typed dataclasses are automatically
        serialized to dicts before validation and database insertion.

        Parameters
        ----------
        params_name : str
            Name for this parameter set
        orient : dict or OrientParams
            Orientation calculation parameters
        centroid : dict or CentroidParams
            Centroid calculation parameters
        smoothing : dict or SmoothingParams
            Smoothing and interpolation parameters
        **kwargs
            Additional DataJoint insertion options

        Examples
        --------
        >>> # Plain dict API (backward compatible)
        >>> PoseParams.insert_custom(
        ...     params_name="my_config",
        ...     orient={
        ...         "method": "two_pt", "bodypart1": "nose", "bodypart2": "tail"
        ...     },
        ...     centroid={"method": "1pt", "points": {"point1": "nose"}},
        ...     smoothing={
        ...         "interpolate": False,
        ...         "smooth": False,
        ...         "likelihood_thresh": 0.9,
        ...     },
        ... )

        >>> # Typed dataclass API
        >>> from spyglass.position.utils.validation import (
        ...     TwoPtOrientParams, OnePtCentroidParams, SmoothingParams
        ... )
        >>> PoseParams.insert_custom(
        ...     params_name="typed_config",
        ...     orient=TwoPtOrientParams(bodypart1="nose", bodypart2="tail"),
        ...     centroid=OnePtCentroidParams(points={"point1": "nose"}),
        ...     smoothing=SmoothingParams(
        ...         likelihood_thresh=0.9, interpolate=False, smooth=False
        ...     ),
        ... )
        """
        params_dict = {
            "pose_params_id": params_name,
            "orient": orient,
            "centroid": centroid,
            "smoothing": smoothing,
        }
        kwargs.setdefault("skip_duplicates", True)
        cls.insert1(params_dict, **kwargs)

    @classmethod
    def insert_custom_dict(
        cls, params_name, orient, centroid, smoothing, **kwargs
    ):
        """Alias for insert_custom for test compatibility."""
        return cls.insert_custom(
            params_name, orient, centroid, smoothing, **kwargs
        )


@schema
class PoseSelection(SpyglassMixin, dj.Manual):
    """Pairing of PoseEstim and PoseParams for pose processing."""

    definition = """
    -> PoseEstim
    -> PoseParams
    """

    def insert1(self, key, **kwargs):
        """Insert pose selection with automatic centroid method validation.

        Validates that the centroid method in PoseParams is appropriate
        for the skeleton associated with the PoseEstim's model.

        Parameters
        ----------
        key : dict
            Primary key with pose_estim_id and pose_params_id entries
        **kwargs
            Additional DataJoint insertion options
        """
        # Validate centroid method against skeleton
        self._validate_centroid_method(key)

        # Proceed with normal insertion
        super().insert1(key, **kwargs)

    def _validate_centroid_method(self, key):
        """Validate centroid method against associated skeleton.

        Parameters
        ----------
        key : dict
            Contains pose_estim_id and pose_params_id for validation
        """
        # Get skeleton_id by joining through the pipeline:
        # PoseEstim -> PoseEstimSelection -> Model -> ModelSelection ...
        # -> ModelParams -> skeleton_id
        # Use projection to avoid analysis_file_name conflicts
        pose_estim_key = {
            k: v
            for k, v in key.items()
            if k in PoseEstimSelection.heading.names
        }

        # First get the model_id from PoseEstimSelection
        selection_info = (PoseEstimSelection & pose_estim_key).fetch1()
        model_id = selection_info["model_id"]

        # Then get skeleton_id from Model -> ModelSelection -> ModelParams
        model_info = (Model & {"model_id": model_id}).fetch1()
        model_selection_key = {
            k: v
            for k, v in model_info.items()
            if k in ModelSelection.heading.names
        }

        # Get skeleton_id from ModelParams via ModelSelection
        params_info = (
            ModelParams & (ModelSelection & model_selection_key)
        ).fetch1()
        skeleton_id = params_info["skeleton_id"]

        # Get centroid params from PoseParams
        pose_params_key = {
            k: v for k, v in key.items() if k in ["pose_params_id"]
        }
        params_info = (PoseParams & pose_params_key).fetch1()
        centroid_params = params_info["centroid"]

        if not centroid_params:
            return  # No centroid params to validate

        # Check if skeleton exists and get suggestion
        skeleton_entry = Skeleton & {"skeleton_id": skeleton_id}
        if len(skeleton_entry) == 0:
            self._warn_msg(
                "Skeleton '%s' not found. Skipping validation.",
                skeleton_id,
            )
            return

        suggested_method = skeleton_entry.get_centroid_method()
        actual_method = centroid_params.get("method")

        if actual_method is None:
            self._warn_msg(
                "No centroid method specified in '%s'. Skeleton '%s' use: '%s'",
                pose_params_key["pose_params_id"],
                skeleton_id,
                suggested_method,
            )
        elif actual_method != suggested_method:
            bodyparts = skeleton_entry.get_bodyparts()
            self._warn_msg(
                "Centroid method '%s' in '%s' may not be optimal for "
                + "skeleton '%s' (bodyparts: %s). Suggested method: '%s'",
                actual_method,
                pose_params_key["pose_params_id"],
                skeleton_id,
                bodyparts,
                suggested_method,
            )

        self._info_msg(
            "Centroid method '%s' in '%s' matches skeleton '%s' suggestion",
            actual_method,
            pose_params_key["pose_params_id"],
            skeleton_id,
        )


@schema
class PoseV2(SpyglassMixin, dj.Computed):
    """Processed pose derivatives (orientation, centroid, velocity)"""

    definition = """
    -> PoseSelection
    ---
    -> AnalysisNwbfile
    orient_object_id='': varchar(40)
    centroid_object_id='': varchar(40)
    velocity_object_id='': varchar(40)
    smoothed_pose_object_id='': varchar(40)
    """

    def fetch1_dataframe(self):
        """Fetch processed pose data as pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with processed pose data. Columns:
            - time: timestamps (index)
            - position_x: centroid x-coordinate (cm)
            - position_y: centroid y-coordinate (cm)
            - orientation: head orientation (radians)
            - velocity: speed (cm/s)

        Examples
        --------
        >>> df = (PoseV2 & key).fetch1_dataframe()
        >>> print(df.head())
                position_x  position_y  orientation  velocity
        time
        0.0          10.5        12.3         1.57       0.0
        0.033        10.6        12.4         1.58       1.2
        ...

        Notes
        -----
        - Returns smoothed centroid position (not raw)
        - Orientation is in radians, range [-π, π]
        - Velocity is Euclidean speed in cm/s
        - First velocity value is typically NaN
        """
        _ = self.ensure_single_entry()

        nwb_data = self.fetch_nwb(
            "orient_object_id",
            "centroid_object_id",
            "velocity_object_id",
            "analysis_file_name",
        )[0]
        centroid_series = nwb_data["centroid"].spatial_series["centroid"]
        orient_series = nwb_data["orient"].spatial_series["orientation"]
        velocity_series = nwb_data["velocity"].time_series["velocity"]

        # Get timestamps from centroid (should be same for all)
        timestamps = centroid_series.timestamps[:]

        # Extract data arrays
        centroid_data = centroid_series.data[:]  # (n_frames, 2)
        orientation_data = orient_series.data[:]  # (n_frames,)
        velocity_data = velocity_series.data[:]  # (n_frames,)

        # Build DataFrame
        df = pd.DataFrame(
            {
                "position_x": centroid_data[:, 0],
                "position_y": centroid_data[:, 1],
                "orientation": orientation_data,
                "velocity": velocity_data,
            },
            index=pd.Index(timestamps, name="time"),
        )

        return df

    def fetch_pose_dataframe(self):
        """Fetch raw bodypart pose coordinates from the linked PoseEstim entry.

        Returns
        -------
        pd.DataFrame
            MultiIndex DataFrame with (scorer, bodypart, coord) columns,
            as produced by PoseEstim.fetch1_dataframe().
        """
        _ = self.ensure_single_entry()
        key = self.fetch1("KEY")
        return (PoseEstim & key).fetch1_dataframe()

    def fetch_video_paths(self) -> list:
        """Return all valid video paths in the VidFileGroup.

        Returns
        -------
        list of str
            Absolute paths to all existing video files in the group.

        Raises
        ------
        ValueError
            If no valid video paths are found for this VidFileGroup.
        """
        _ = self.ensure_single_entry()
        vid_group_id = self.fetch1("vid_group_id")
        video_rows = (
            VidFileGroup.File & {"vid_group_id": vid_group_id}
        ) * VideoFile
        paths = video_rows.fetch("path")
        valid_paths = [p for p in paths if p and Path(p).exists()]
        if not valid_paths:
            raise ValueError(
                f"No valid video paths found for vid_group_id='{vid_group_id}'."
            )
        return valid_paths

    def make_video(
        self,
        output_path: Union[Path, str, None] = None,
        cm_to_pixels: float = 1.0,
        percent_frames: float = 1.0,
        **kwargs,
    ) -> str:
        """Generate a video with pose overlay from a PoseV2 entry.

        Fetches processed centroid/orientation from this PoseV2 entry, raw
        bodypart positions and likelihoods from the linked PoseEstim entry,
        and the source video from VidFileGroup → VideoFile. Renders each
        frame with matplotlib overlays using
        ``spyglass.position.utils.make_video.VideoMaker``.

        Parameters
        ----------
        output_path : Union[Path, str, None], optional
            Path for the output video file. Defaults to
            ``<video_stem>_pose.mp4`` next to the source video.
        cm_to_pixels : float, optional
            Conversion factor from centimeters to pixels, by default 1.0.
        percent_frames : float, optional
            Fraction of video frames to process (0.0–1.0), by default 1.0.
        **kwargs
            Additional keyword arguments forwarded to VideoMaker (e.g.
            ``batch_size``, ``crop``, ``debug``).

        Returns
        -------
        str
            Absolute path to the output video file.

        Raises
        ------
        ValueError
            If the table restriction does not resolve to exactly one entry,
            or if no video path is registered for the VidFileGroup.
        FileNotFoundError
            If the source video file does not exist on disk.

        Examples
        --------
        >>> output = (PoseV2() & key).make_video()
        >>> print(output)
        '/data/videos/session_pose.mp4'

        >>> # Crop to a region and process half the frames
        >>> output = (PoseV2() & key).make_video(
        ...     output_path="/tmp/cropped.mp4",
        ...     crop=(100, 600, 50, 450),
        ...     percent_frames=0.5,
        ... )
        """
        from datajoint.hash import key_hash

        from spyglass.position.utils.make_video import VideoMaker

        _ = self.ensure_single_entry()

        # --- processed pose data ---
        df = self.fetch1_dataframe()
        timestamps = df.index.values
        position_mean = df[["position_x", "position_y"]].values
        orientation_mean = df["orientation"].values

        # --- source video path ---
        entry = self.fetch1()
        vid_group_id = entry["vid_group_id"]

        video_rows = (
            VidFileGroup.File & {"vid_group_id": vid_group_id}
        ) * VideoFile
        paths = video_rows.fetch("path")
        valid_paths = [p for p in paths if p and Path(p).exists()]
        if not valid_paths:
            raise ValueError(
                f"No valid video path found for vid_group_id='{vid_group_id}'. "
                "Ensure VideoFile entries have a 'path' field populated."
            )
        video_path = valid_paths[0]
        if len(valid_paths) > 1:
            self._warn_msg(
                f"Multiple videos in group '{vid_group_id}'; "
                f"using first: {video_path}"
            )

        # --- fps from video (needed for time display in VideoMaker) ---
        # Timestamps stored in V2 NWB are frame indices (0, 1, 2, ...) because
        # DLC does not embed absolute timestamps. fps converts them to seconds.
        ret = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v",
                "-show_entries",
                "stream=r_frame_rate",
                "-of",
                "csv=p=0",
                str(video_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        if ret.returncode != 0:
            raise FileNotFoundError(
                f"ffprobe failed on {video_path}: {ret.stderr}"
            )
        fps = float(Fraction(ret.stdout.strip()))  # e.g. "30000/1001" → 29.97

        # video_frame_inds: timestamps ARE frame indices (0, 1, 2, ...) since
        # DLC runs inference on every frame. Mirrors V1's video_frame_ind column
        video_frame_inds = timestamps.astype(int)

        # position_time: convert frame indices to seconds for VideoMaker display
        position_time = timestamps / fps

        # --- raw bodypart positions + likelihoods from PoseEstim ---
        centroids: dict = {}
        likelihoods: dict = {}
        estim_restriction = {
            k: entry[k]
            for k in ("model_id", "vid_group_id", "pose_estim_params_id")
            if k in entry
        }
        if PoseEstim() & estim_restriction:
            try:
                pose_df = (PoseEstim() & estim_restriction).fetch1_dataframe()
                idx = pd.IndexSlice
                scorer = pose_df.columns.get_level_values(0)[0]
                bodyparts = pose_df.columns.get_level_values(1).unique()
                for bp in bodyparts:
                    centroids[bp] = pose_df.loc[
                        :, idx[scorer, bp, ["x", "y"]]
                    ].values
                    likelihoods[bp] = pose_df.loc[
                        :, idx[scorer, bp, "likelihood"]
                    ].values
            except (KeyError, IndexError, ValueError, TypeError) as e:
                self._warn_msg(
                    f"Could not load raw pose for overlay: {e}. "
                    "Proceeding without bodypart dots."
                )

        # --- output path ---
        if output_path is None:
            src = Path(video_path)
            output_path = src.parent / f"{src.stem}_pose.mp4"
        output_path = str(output_path)

        VideoMaker(
            video_filename=str(video_path),
            position_mean=position_mean,
            orientation_mean=orientation_mean,
            centroids=centroids,
            position_time=position_time,
            video_frame_inds=video_frame_inds,
            likelihoods=likelihoods or None,
            output_video_filename=output_path,
            cm_to_pixels=cm_to_pixels,
            percent_frames=percent_frames,
            key_hash=key_hash(entry),
            **kwargs,
        )
        return output_path

    def make(self, key):
        """Process raw pose estimation with orientation, centroid, and smoothing

        1. Fail fast: verify NWB link
        2. Fetch raw pose DataFrame and parameters
        3. Delegate computation to compute_pose_outputs (pure, no DB)
        4. Store results in NWB and insert row

        Parameters
        ----------
        key : dict
            Primary key from PoseSelection
        """
        from spyglass.position.utils.pose_processing import compute_pose_outputs

        nwb_file_name = self._get_nwb_file_name(key)

        self._info_msg(f"Processing pose for: {key['model_id']}")
        pose_df = (PoseEstim & key).fetch1_dataframe()
        params = (PoseParams & key).fetch1()

        outputs = compute_pose_outputs(
            pose_df,
            params["orient"],
            params["centroid"],
            params["smoothing"],
        )

        self._info_msg("Storing results in NWB...")
        analysis_file_name, obj_ids = self._store_pose_nwb(
            {**key, "nwb_file_name": nwb_file_name},
            outputs["orientation"],
            outputs["centroid"],
            outputs["velocity"],
            outputs["timestamps"],
            outputs["sampling_rate"],
        )
        self.insert1(
            {
                **key,
                "analysis_file_name": analysis_file_name,
                "orient_object_id": obj_ids["orient"],
                "centroid_object_id": obj_ids["centroid"],
                "velocity_object_id": obj_ids["velocity"],
                "smoothed_pose_object_id": obj_ids["smoothed_pose"],
            }
        )

        from spyglass.position.position_merge import PositionOutput

        PositionOutput._merge_insert(
            [key],
            part_name="PoseV2",
            skip_duplicates=True,
        )
        self._info_msg("Pose processing complete!")

    def _apply_likelihood_threshold(
        self, pose_df: pd.DataFrame, likelihood_thresh: float
    ) -> pd.DataFrame:
        from spyglass.position.utils.pose_processing import (
            apply_likelihood_threshold,
        )

        return apply_likelihood_threshold(pose_df, likelihood_thresh)

    @staticmethod
    def _get_nwb_file_name(key: dict):
        """Return the common nwb_file_name for this VidFileGroup, or None.

        Delegates to VidFileGroup.get_nwb_file(), which joins VideoFile →
        TaskEpoch → Session to confirm all videos share one NWB parent.
        Returns None when no such parent exists (e.g. tutorial / no registered
        videos), so callers can fall back to standalone NWB storage.

        Parameters
        ----------
        key : dict
            Must include vid_group_id

        Returns
        -------
        str or None
        """
        try:
            return VidFileGroup().get_nwb_file(key["vid_group_id"])[
                "nwb_file_name"
            ]
        except ValueError:
            return None

    def _calculate_orientation(
        self,
        pose_df: pd.DataFrame,
        orient_params: dict,
        timestamps: np.ndarray,
        sampling_rate: float,
    ) -> np.ndarray:
        """Calculate orientation from pose data.

        Parameters
        ----------
        pose_df : pd.DataFrame
            Pose DataFrame
        orient_params : dict
            Orientation parameters
        timestamps : np.ndarray
            Timestamps for each frame
        sampling_rate : float
            Sampling rate in Hz

        Returns
        -------
        np.ndarray
            Orientation in radians, shape (n_frames,)
        """
        from spyglass.position.utils.orientation import (
            bisector_orientation,
            no_orientation,
            smooth_orientation,
            two_pt_orientation,
        )

        method = orient_params["method"]

        # Flatten MultiIndex to single-level for utility functions
        pose_flat = self._flatten_multiindex(pose_df)

        # Calculate raw orientation based on method
        if method == "two_pt":
            orientation = two_pt_orientation(
                pose_flat,
                point1=orient_params["bodypart1"],
                point2=orient_params["bodypart2"],
            )
        elif method == "bisector":
            orientation = bisector_orientation(
                pose_flat,
                led1=orient_params["led1"],
                led2=orient_params["led2"],
                led3=orient_params["led3"],
            )
        elif method == "none":
            orientation = no_orientation(pose_flat)
        else:
            raise ValueError(
                f"Unknown orientation method: {method}"
            )  # pragma: no cover

        # Apply smoothing if requested
        if orient_params.get("smooth", False):
            smooth_params = orient_params.get("smoothing_params", {})
            std_dev = smooth_params.get("std_dev", 0.001)
            interpolate = orient_params.get("interpolate", True)

            orientation = smooth_orientation(
                orientation, timestamps, std_dev, interpolate
            )

        return orientation

    def _calculate_centroid(
        self, pose_df: pd.DataFrame, centroid_params: dict
    ) -> np.ndarray:
        """Calculate centroid from pose data.

        Parameters
        ----------
        pose_df : pd.DataFrame
            Pose DataFrame
        centroid_params : dict
            Centroid parameters

        Returns
        -------
        np.ndarray
            Centroid positions, shape (n_frames, 2)
        """
        from spyglass.position.utils.centroid import calculate_centroid

        # Flatten MultiIndex
        pose_flat = self._flatten_multiindex(pose_df)

        # Calculate centroid
        max_sep = centroid_params.get("max_LED_separation", None)
        centroid = calculate_centroid(
            pose_flat, centroid_params["points"], max_sep
        )

        return centroid

    def _smooth_position(
        self,
        position: np.ndarray,
        timestamps: np.ndarray,
        sampling_rate: float,
        smooth_params: dict,
    ) -> np.ndarray:
        """Apply interpolation and smoothing to position data.

        Parameters
        ----------
        position : np.ndarray
            Position array, shape (n_frames, 2)
        timestamps : np.ndarray
            Timestamps
        sampling_rate : float
            Sampling rate in Hz
        smooth_params : dict
            Smoothing parameters

        Returns
        -------
        np.ndarray
            Smoothed position, shape (n_frames, 2)
        """
        from spyglass.position.utils.interpolation import (
            get_smoothing_function,
            interp_position,
        )
        from spyglass.position.utils.orientation import get_span_start_stop

        # Create DataFrame for processing
        pos_df = pd.DataFrame(position, columns=["x", "y"], index=timestamps)

        # Step 1: Interpolation
        if smooth_params.get("interpolate", False):
            interp_params = smooth_params.get("interp_params", {})

            # Find NaN spans
            is_nan = np.isnan(pos_df["x"]) | np.isnan(pos_df["y"])
            if np.any(is_nan):
                nan_indices = np.where(is_nan)[0]
                nan_spans = get_span_start_stop(nan_indices)

                pos_df = interp_position(
                    pos_df,
                    nan_spans,
                    max_pts_to_interp=interp_params.get(
                        "max_pts_to_interp", None
                    ),
                    max_cm_to_interp=interp_params.get(
                        "max_cm_to_interp", None
                    ),
                )

        # Step 2: Smoothing
        if smooth_params.get("smooth", False):
            smoothing_params = smooth_params["smoothing_params"]
            method = smoothing_params["method"]
            smooth_func = get_smoothing_function(method)

            # Apply smoothing based on method
            if method == "moving_avg":
                pos_df = smooth_func(
                    pos_df,
                    smoothing_duration=smoothing_params["smoothing_duration"],
                    sampling_rate=sampling_rate,
                )
            elif method == "savgol":
                pos_df = smooth_func(
                    pos_df,
                    window_length=smoothing_params["window_length"],
                    polyorder=smoothing_params.get("polyorder", 3),
                )
            elif method == "gaussian":
                pos_df = smooth_func(
                    pos_df,
                    std_dev=smoothing_params["std_dev"],
                    sampling_rate=sampling_rate,
                )

        return pos_df[["x", "y"]].values

    @staticmethod
    def _calculate_velocity(
        position: np.ndarray, timestamps: np.ndarray, sampling_rate: float
    ) -> np.ndarray:
        from spyglass.position.utils.pose_processing import calculate_velocity

        return calculate_velocity(position, timestamps, sampling_rate)

    @staticmethod
    def _flatten_multiindex(pose_df: pd.DataFrame) -> pd.DataFrame:
        from spyglass.position.utils.general import flatten_multiindex

        return flatten_multiindex(pose_df)

    def _store_pose_nwb(
        self,
        key: dict,
        orientation: np.ndarray,
        centroid: np.ndarray,
        velocity: np.ndarray,
        timestamps: np.ndarray,
        sampling_rate: float,
    ) -> tuple:
        """Store processed pose data in NWB file.

        Parameters
        ----------
        key : dict
            Primary key
        orientation : np.ndarray
            Orientation in radians
        centroid : np.ndarray
            Centroid positions (n_frames, 2)
        velocity : np.ndarray
            Velocity in cm/s
        timestamps : np.ndarray
            Timestamps
        sampling_rate : float
            Sampling rate in Hz

        Returns
        -------
        tuple
            (analysis_file_name, obj_ids) where obj_ids is dict with keys:
            'orient', 'centroid', 'velocity', 'smoothed_pose'
        """
        import pynwb

        # Constant for unit conversion
        METERS_PER_CM = 0.01

        # Create or get analysis NWB file
        a_fname = AnalysisNwbfile().create(key["nwb_file_name"])
        nwb_analysis_file = AnalysisNwbfile()

        # Create Position object for centroid
        position = pynwb.behavior.Position(name="centroid")
        position.create_spatial_series(
            name="centroid",
            timestamps=timestamps,
            data=centroid,
            reference_frame="(0,0) is top left",
            conversion=METERS_PER_CM,
            description="Centroid position (x, y) in cm",
            comments="Calculated from pose estimation bodyparts",
        )

        # Create BehavioralTimeSeries for orientation
        orientation_ts = pynwb.behavior.CompassDirection(name="orientation")
        orientation_ts.create_spatial_series(
            name="orientation",
            timestamps=timestamps,
            data=orientation,
            conversion=1.0,
            unit="radians",
            description="Animal orientation in radians",
            comments="Counter-clockwise from positive x-axis, range [-π, π]",
            # reference_frame=??,
        )

        # Create BehavioralTimeSeries for velocity
        velocity_ts = pynwb.behavior.BehavioralTimeSeries(name="velocity")
        velocity_ts.create_timeseries(
            name="velocity",
            timestamps=timestamps,
            data=velocity,
            unit="cm/s",
            description="Speed in cm/s",
            comments="Euclidean distance traveled per unit time",
        )

        # Create Position object for smoothed pose (centroid with metadata)
        smoothed_pose = pynwb.behavior.Position(name="smoothed_pose")
        smoothed_pose.create_spatial_series(
            name="smoothed_position",
            timestamps=timestamps,
            data=centroid,
            reference_frame="(0,0) is top left",
            conversion=METERS_PER_CM,
            description="Smoothed position (x, y) in cm",
            comments="Interpolated and smoothed centroid position",
        )

        # Add objects to NWB file and get their IDs
        obj_ids = {
            "orient": nwb_analysis_file.add_nwb_object(a_fname, orientation_ts),
            "centroid": nwb_analysis_file.add_nwb_object(a_fname, position),
            "velocity": nwb_analysis_file.add_nwb_object(a_fname, velocity_ts),
            "smoothed_pose": nwb_analysis_file.add_nwb_object(
                a_fname, smoothed_pose
            ),
        }

        # Register the relationship between raw and analysis NWB files
        nwb_analysis_file.add(
            nwb_file_name=key["nwb_file_name"],
            analysis_file_name=a_fname,
        )

        return a_fname, obj_ids

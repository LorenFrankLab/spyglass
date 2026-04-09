import contextlib
import io
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Union

import datajoint as dj
import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO

from spyglass.common import AnalysisNwbfile
from spyglass.common.common_behav import VideoFile
from spyglass.position.utils_dlc import suppress_print_from_package
from spyglass.position.v2.train import (
    Model,
    ModelParams,
    Skeleton,
    default_pk_name,
    resolve_model_path,
)
from spyglass.position.v2.video import VidFileGroup
from spyglass.settings import dlc_output_dir
from spyglass.utils import logger

# ----------------------------- Optional imports ------------------------------
try:
    import ndx_pose
except ImportError:
    ndx_pose = None

schema = dj.schema("cbroz_position_v2_estim")


@schema
class PoseEstimParams(dj.Lookup):
    """Parameters for pose estimation inference.

    Stores tool-specific inference parameters (batch_size, device, etc.)
    as reusable named parameter sets.

    Attributes
    ----------
    pose_estim_params_id : str
        Unique identifier for this parameter set (max 32 chars)
    params : blob
        Tool-specific inference parameters dict
    params_hash : str
        Hash of params for deduplication
    """

    definition = """
    pose_estim_params_id: varchar(32)
    ---
    params: blob          # tool-specific inference params
    params_hash: varchar(64)  # hash of params
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

        if params_id is None:
            params_id = default_pk_name("pep", params)

        key = {
            "pose_estim_params_id": params_id,
            "params": params,
            "params_hash": params_hash,
        }
        cls.insert1(key, skip_duplicates=skip_duplicates)
        logger.info(f"Inserted PoseEstimParams: {params_id}")
        return {"pose_estim_params_id": params_id}


@schema
class PoseEstimSelection(dj.Manual):
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
        'trigger' to run inference, 'load' to load existing results
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

        if "vid_group_id" not in key:
            raise ValueError(
                "key must include 'vid_group_id'. "
                "Use VidFileGroup().insert1() to create a group first."
            )
        if not (VidFileGroup() & {"vid_group_id": key["vid_group_id"]}):
            raise ValueError(
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

        logger.info("Inserted entry into PoseEstimSelection")
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

        # Try to use the configured dlc_output_dir
        base_dir = dlc_output_dir
        if not base_dir:
            base_dir = Path.cwd() / "pose_estimation_output"

        output_dir = (
            Path(base_dir) / f"v2_{model_id}_{key.get('vid_group_id', '')}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        return str(output_dir)


@schema
class PoseEstim(dj.Computed):
    """Pose estimation results from model inference on video files.

    Computed from PoseEstimSelection entries. Runs inference (trigger)
    or loads existing results (load), stores in NWB via AnalysisNwbfile.

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
    -> [nullable] AnalysisNwbfile
    """

    @staticmethod
    def load_dlc_output(
        dlc_output_path: Union[Path, str],
        nwb_file_name: Union[Path, str, None] = None,
        pose_estimation_name: str = "PoseEstimation",
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
            Name for the PoseEstimation object in NWB, by default "PoseEstimation"

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
        if ndx_pose is None:
            raise ImportError(
                "ndx-pose is required to load DLC output into NWB. "
                "Install with: pip install ndx-pose>=0.2.0"
            )

        dlc_output_path = Path(dlc_output_path)
        if not dlc_output_path.exists():
            raise FileNotFoundError(f"DLC output not found: {dlc_output_path}")

        logger.info(f"Loading DLC output: {dlc_output_path}")

        # Load DLC output data
        if dlc_output_path.suffix == ".h5":
            df = pd.read_hdf(dlc_output_path)
        elif dlc_output_path.suffix == ".csv":
            df = pd.read_csv(dlc_output_path, header=[0, 1, 2], index_col=0)
        else:
            raise ValueError(
                f"Unsupported file format: {dlc_output_path.suffix}. "
                "Must be .h5 or .csv"
            )

        # Extract metadata from DLC output
        # DLC columns are MultiIndex: [scorer, bodypart, coords]
        scorer = df.columns.get_level_values(0)[0]
        bodyparts = df.columns.get_level_values(1).unique().tolist()

        logger.info(
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
            logger.info(f"Updating existing NWB file: {nwb_path}")
            io_mode = "r+"
        else:
            logger.info(f"Creating new NWB file: {nwb_path}")
            io_mode = "w"

        # Build ndx-pose structure
        with NWBHDF5IO(str(nwb_path), mode=io_mode) as io:
            if io_mode == "r+":
                nwbfile = io.read()
            else:
                from pynwb import NWBFile

                nwbfile = NWBFile(
                    session_description=f"Pose estimation from {dlc_output_path.name}",
                    identifier=f"pose_{datetime.now():%Y%m%d%H%M%S}",
                    session_start_time=datetime.now(),
                )

            # Create behavior processing module if it doesn't exist
            if "behavior" not in nwbfile.processing:
                behavior_module = nwbfile.create_processing_module(
                    name="behavior",
                    description="Behavioral pose estimation data",
                )
            else:
                behavior_module = nwbfile.processing["behavior"]

            # Create skeleton
            # Note: We don't have edge information from DLC output alone
            skeleton = ndx_pose.Skeleton(
                name=f"{pose_estimation_name}_skeleton",
                nodes=bodyparts,
                edges=np.array([], dtype="uint8").reshape(
                    0, 2
                ),  # No edges from DLC output
            )

            # Create Skeletons container if it doesn't exist
            if "Skeletons" not in behavior_module.data_interfaces:
                skeletons = ndx_pose.Skeletons(skeletons=[skeleton])
                behavior_module.add_data_interface(skeletons)
            else:
                # Add skeleton to existing container
                skeletons = behavior_module.data_interfaces["Skeletons"]
                skeletons.skeletons.append(skeleton)

            # Create PoseEstimationSeries for each bodypart
            pose_series_list = []

            # Get timestamps (frame indices if not provided)
            if "time" in df.columns:
                timestamps = df["time"].values
            else:
                # Use frame indices as timestamps
                timestamps = np.arange(len(df), dtype=float)

            for bodypart in bodyparts:
                # Extract x, y, likelihood for this bodypart
                x = df[(scorer, bodypart, "x")].values
                y = df[(scorer, bodypart, "y")].values
                likelihood = df[(scorer, bodypart, "likelihood")].values

                # Stack x, y as (n_frames, 2)
                pose_data = np.column_stack([x, y])

                # Create PoseEstimationSeries
                series = ndx_pose.PoseEstimationSeries(
                    name=f"{bodypart}_pose",
                    description=f"Pose estimation for {bodypart}",
                    data=pose_data,
                    unit="pixels",
                    reference_frame="(0,0) is top-left corner",
                    timestamps=timestamps,
                    confidence=likelihood,
                    confidence_definition="DLC likelihood score",
                )
                pose_series_list.append(series)

            # Create PoseEstimation container
            pose_estimation = ndx_pose.PoseEstimation(
                name=pose_estimation_name,
                pose_estimation_series=pose_series_list,
                description=f"Pose estimation from DLC: {dlc_output_path.name}",
                original_videos=[str(dlc_output_path.stem)],
                source_software="DeepLabCut",
                skeleton=skeleton,
                scorer=scorer,
            )

            # Add to behavior module
            behavior_module.add(pose_estimation)

            # Write NWB file
            io.write(nwbfile)

        logger.info(f"DLC output loaded into NWB: {nwb_path}")
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
        if ndx_pose is None:
            raise ImportError(
                "ndx-pose is required to load pose data from NWB. "
                "Install with: pip install ndx-pose>=0.2.0"
            )

        nwb_file_path = Path(nwb_file_path)
        if not nwb_file_path.exists():
            raise FileNotFoundError(f"NWB file not found: {nwb_file_path}")

        logger.info(f"Loading pose data from NWB: {nwb_file_path}")

        # Read NWB file and validate structure
        with NWBHDF5IO(str(nwb_file_path), mode="r") as io:
            nwbfile = io.read()

            if "behavior" not in nwbfile.processing:
                raise ValueError(
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

            if not pose_estimations:
                raise ValueError(
                    f"No PoseEstimation objects found in NWB: {nwb_file_path}. "
                    "File must contain ndx-pose.PoseEstimation data."
                )

            # Select PoseEstimation
            if pose_estimation_name is not None:
                if pose_estimation_name not in pose_estimations:
                    available = ", ".join(pose_estimations.keys())
                    raise ValueError(
                        f"PoseEstimation '{pose_estimation_name}' not found in NWB. "
                        f"Available: {available}"
                    )
                pose_estimation = pose_estimations[pose_estimation_name]
                selected_name = pose_estimation_name
            else:
                # Use first PoseEstimation
                selected_name = list(pose_estimations.keys())[0]
                pose_estimation = pose_estimations[selected_name]
                if len(pose_estimations) > 1:
                    logger.warning(
                        f"Multiple PoseEstimation objects found. "
                        f"Using '{selected_name}'. "
                        f"Available: {', '.join(pose_estimations.keys())}"
                    )

            # Extract metadata
            scorer = getattr(pose_estimation, "scorer", "unknown")
            source_software = getattr(
                pose_estimation, "source_software", "unknown"
            )

            # Extract bodyparts from PoseEstimationSeries
            bodyparts = []
            n_frames = None

            for series in pose_estimation.pose_estimation_series.values():
                bodypart = series.name.replace("_pose", "")
                bodyparts.append(bodypart)

                # Get frame count from first series
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

        logger.info(
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
        ValueError
            If analysis_file_name is not set
        """
        if ndx_pose is None:
            raise ImportError(
                "ndx-pose is required to fetch pose data. "
                "Install with: pip install ndx-pose>=0.2.0"
            )

        # Fetch entry
        entry = self.fetch1()
        nwb_file_name = entry["analysis_file_name"]

        if nwb_file_name is None:
            raise ValueError(
                "Cannot fetch pose data: analysis_file_name is not set. "
                "Re-populate with a VidFileGroup linked to a registered "
                "Nwbfile so the entry references an AnalysisNwbfile."
            )

        logger.debug(f"Fetching pose data from NWB: {nwb_file_name}")

        # Get NWB file path
        nwb_file_matches = AnalysisNwbfile() & {
            "analysis_file_name": nwb_file_name
        }
        if not nwb_file_matches:
            raise ValueError(
                f"AnalysisNwbfile not found: {nwb_file_name}. "
                "Ensure the NWB file is registered in AnalysisNwbfile table."
            )
        nwb_path = nwb_file_matches.fetch1("analysis_file_abs_path")

        # Read pose data from NWB
        with NWBHDF5IO(str(nwb_path), mode="r") as io:
            nwbfile = io.read()

            if "behavior" not in nwbfile.processing:
                raise ValueError(f"No behavior module in NWB: {nwb_file_name}")

            behavior_module = nwbfile.processing["behavior"]

            # Find PoseEstimation objects
            pose_estimations = {
                name: obj
                for name, obj in behavior_module.data_interfaces.items()
                if isinstance(obj, ndx_pose.PoseEstimation)
            }

            if not pose_estimations:
                raise ValueError(f"No PoseEstimation in NWB: {nwb_file_name}")

            # Use first PoseEstimation (or could select by name)
            pose_estimation = list(pose_estimations.values())[0]
            scorer = getattr(pose_estimation, "scorer", "DLC_scorer")

            # Build DataFrame from PoseEstimationSeries
            data_dict = {}

            for series in pose_estimation.pose_estimation_series:
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

        logger.debug(f"Fetched {len(df)} frames of pose data")
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
        if ndx_pose is None:
            raise ImportError(
                "ndx-pose is required for pose estimation. "
                "Install with: pip install ndx-pose>=0.2.0"
            )

        # Fetch selection entry
        task_mode, output_dir = (PoseEstimSelection & key).fetch1(
            "task_mode", "output_dir"
        )

        # Fetch inference params from PoseEstimParams
        inference_params = (PoseEstimParams & key).fetch1("params")
        inference_params = inference_params or {}

        logger.info(
            f"PoseEstim.make: mode={task_mode}, " f"output_dir={output_dir}"
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

        # Step 1: Trigger inference if needed
        if task_mode == "trigger":
            if not video_paths:
                raise ValueError(
                    f"No video files registered for vid_group_id="
                    f"{key['vid_group_id']}. Cannot trigger inference "
                    "without registered video files in VideoFile table."
                )
            logger.info("Triggering inference...")
            model_key = {"model_id": key["model_id"]}
            destfolder = output_dir if output_dir else None
            self.run_inference(
                model_key,
                video_paths if len(video_paths) > 1 else video_paths[0],
                destfolder=destfolder,
                **inference_params,
            )

        # Step 2: Find DLC output file(s) in output_dir
        output_path = Path(output_dir)
        if not output_path.exists():
            raise FileNotFoundError(
                f"Output directory not found: {output_dir}. "
                "Ensure inference has been run or output_dir is correct."
            )

        # Look for DLC h5 output files
        h5_files = sorted(output_path.glob("*DLC*.h5"))
        if not h5_files:
            # Also check for generic h5 files
            h5_files = sorted(output_path.glob("*.h5"))
        if not h5_files:
            raise FileNotFoundError(
                f"No h5 output files found in {output_dir}. "
                "Expected DLC output (.h5) files."
            )

        # Use most recent h5 file
        dlc_output_path = max(h5_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"Loading DLC output: {dlc_output_path}")

        # Step 3: Read DLC output into DataFrame
        df = pd.read_hdf(dlc_output_path)
        scorer = df.columns.get_level_values(0)[0]
        bodyparts = df.columns.get_level_values(1).unique().tolist()

        logger.info(
            f"DLC data: {len(bodyparts)} bodyparts, {len(df)} frames, "
            f"scorer: {scorer}"
        )

        # Step 4: Store data in AnalysisNwbfile (requires registered Nwbfile).
        if nwb_file_name is None:
            raise ValueError(
                "Cannot store pose estimation: no registered Nwbfile linked "
                f"to VidFileGroup '{key['vid_group_id']}'. Ensure the video "
                "files in this group are registered in VideoFile and linked "
                "to a Session via TaskEpoch. Use "
                "VidFileGroup().get_nwb_file() to verify before populating."
            )

        analysis_file_name = self._store_estimation_nwb(
            key, df, bodyparts, scorer, nwb_file_name
        )
        self.insert1({**key, "analysis_file_name": analysis_file_name})
        logger.info("PoseEstim entry inserted.")

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
            Path to video file(s) for inference. Can be a single path, list of
            paths, or directory containing videos.
        save_as_csv : bool, optional
            Whether to save output as CSV in addition to h5, by default False
        destfolder : Union[Path, str, None], optional
            Destination folder for output files. If None, saves alongside
            video, by default None
        **kwargs
            Additional parameters passed to the underlying inference function
            (e.g., DLC's analyze_videos). Common options:
            - shuffle : int, shuffle index (default: 1)
            - trainingsetindex : int, training set fraction index (default: 0)
            - batch_size : int, batch size for inference
            - device : str, device for inference ('cpu', 'cuda', etc.)

        Returns
        -------
        Union[str, list]
            Path(s) to output file(s). Returns single path for single video
            input, list of paths for multiple videos.

        Raises
        ------
        ValueError
            If model_key doesn't exist in Model table
        FileNotFoundError
            If video_path doesn't exist
        NotImplementedError
            If model tool is not supported for inference

        Examples
        --------
        >>> model_key = {"model_id": "my_dlc_model"}
        >>> output = PoseEstim().run_inference(model_key, "/path/to/video.mp4")
        """
        if not (Model() & model_key):
            raise ValueError(
                f"Model not found in database: {model_key}. "
                "Please import model first using Model.import_model()"
            )

        model_info = (Model() & model_key).fetch1()
        model_params_key = {
            "model_params_id": model_info["model_params_id"],
            "tool": model_info["tool"],
        }
        params_info = (ModelParams() & model_params_key).fetch1()
        tool = params_info["tool"]

        logger.info(f"Running inference with {tool} model: {model_key}")

        if tool == "DLC":
            return self._run_dlc_inference(
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

    def _run_dlc_inference(
        self,
        model_info: dict,
        video_path: Union[Path, str, list],
        save_as_csv: bool,
        destfolder: Union[Path, str, None],
        **kwargs,
    ) -> Union[str, list]:
        """Run DLC inference on video(s).

        Parameters
        ----------
        model_info : dict
            Model table entry with model_path and other metadata
        video_path : Union[Path, str, list]
            Video path(s) for inference
        save_as_csv : bool
            Save output as CSV
        destfolder : Union[Path, str, None]
            Destination folder for outputs
        **kwargs
            Additional DLC analyze_videos parameters

        Returns
        -------
        Union[str, list]
            Output file path(s)
        """
        if not isinstance(video_path, (list, tuple)):
            video_path = [video_path]

        for vp in video_path:
            if not Path(vp).exists():
                raise FileNotFoundError(f"Video not found: {vp}")
        videos = [str(vp) for vp in video_path]

        model_path = resolve_model_path(model_info["model_path"])
        if not model_path.exists():
            raise FileNotFoundError(f"Model config not found: {model_path}")
        if model_path.suffix not in [".yaml", ".yml"]:
            raise ValueError(
                f"DLC model path must be a config.yaml file, got: {model_path}"
            )

        analyze_params = {
            "config": str(model_path),
            "videos": [videos] if isinstance(videos, str) else videos,
            "save_as_csv": save_as_csv,
            "destfolder": str(destfolder) if destfolder else None,
        }
        dlc_params = [
            "shuffle",
            "trainingsetindex",
            "videotype",
            "in_random_order",
            "snapshot_index",
            "device",
            "batch_size",
            "dynamic",
            "modelprefix",
            "robust_nframes",
            "cropping",
        ]
        for param in dlc_params:
            if param in kwargs:
                analyze_params[param] = kwargs[param]

        logger.info(f"Running DLC inference on {videos}")
        logger.debug(f"DLC parameters: {analyze_params}")

        try:
            from deeplabcut import analyze_videos
        except ImportError:
            raise ImportError(
                "DeepLabCut is required for inference. "
                "Install with: pip install deeplabcut>=3.0"
            )

        # suppress_print_from_package filters stack-based prints;
        # redirect_stdout catches anything that bypasses it.
        try:
            with (
                suppress_print_from_package(),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                analyze_videos(**analyze_params)
        except Exception as e:
            logger.error(f"DLC inference failed: {e}")
            raise

        output_folder = Path(destfolder) if destfolder else None

        output_paths = []
        for vid_path in videos:
            vid_path = Path(vid_path)
            output_dir = output_folder if output_folder else vid_path.parent
            output_files = list(output_dir.glob(f"{vid_path.stem}DLC_*.h5"))
            if output_files:
                output_paths.append(
                    str(max(output_files, key=lambda p: p.stat().st_mtime))
                )
        logger.info(f"Inference complete. Output files: {output_paths}")
        return output_paths if len(output_paths) != 1 else output_paths[0]

    def _store_estimation_nwb(
        self, key, pose_df, bodyparts, scorer, nwb_file_name
    ):
        """Store pose estimation data in an AnalysisNwbfile.

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
            Name of the source NWB file (for AnalysisNwbfile.create)

        Returns
        -------
        str
            The analysis_file_name for the created NWB file
        """
        # Create analysis NWB file
        analysis_file_name = AnalysisNwbfile().create(nwb_file_name)
        nwb_analysis_file = AnalysisNwbfile()

        # Fetch skeleton info from Model -> Skeleton
        model_info = (Model & key).fetch1()
        skeleton_key = {"skeleton_id": model_info["skeleton_id"]}
        skeleton_entry = (Skeleton & skeleton_key).fetch1()
        skeleton_entry.get("nodes", bodyparts)
        skeleton_edges = skeleton_entry.get("edges", [])

        # Build ndx-pose Skeleton
        edge_array = (
            np.array(skeleton_edges, dtype="uint8")
            if skeleton_edges
            else np.array([], dtype="uint8").reshape(0, 2)
        )
        nwb_skeleton = ndx_pose.Skeleton(
            name=f"skeleton_{key['model_id']}",
            nodes=bodyparts,
            edges=edge_array,
        )

        # Build PoseEstimationSeries for each bodypart
        timestamps = np.arange(len(pose_df), dtype=float)
        pose_series_list = []

        for bodypart in bodyparts:
            x = pose_df[(scorer, bodypart, "x")].values
            y = pose_df[(scorer, bodypart, "y")].values
            likelihood = pose_df[(scorer, bodypart, "likelihood")].values
            pose_data = np.column_stack([x, y])

            series = ndx_pose.PoseEstimationSeries(
                name=f"{bodypart}_pose",
                description=f"Pose estimation for {bodypart}",
                data=pose_data,
                unit="pixels",
                reference_frame="(0,0) is top-left corner",
                timestamps=timestamps,
                confidence=likelihood,
                confidence_definition="DLC likelihood score",
            )
            pose_series_list.append(series)

        # Build PoseEstimation container
        pose_estimation = ndx_pose.PoseEstimation(
            name="PoseEstimation",
            pose_estimation_series=pose_series_list,
            description=(f"Pose estimation from model {key['model_id']}"),
            original_videos=[
                str(p) for p in (VidFileGroup.File & key).fetch("KEY")
            ],
            source_software="DeepLabCut",
            skeleton=nwb_skeleton,
            scorer=scorer,
        )

        # Create a pynwb processing module container
        import pynwb

        behavior_module = pynwb.ProcessingModule(
            name="behavior",
            description="Behavioral pose estimation data",
        )

        # Add skeleton container and pose estimation
        skeletons = ndx_pose.Skeletons(skeletons=[nwb_skeleton])
        behavior_module.add(skeletons)
        behavior_module.add(pose_estimation)

        # Add to analysis NWB and register
        nwb_analysis_file.add_nwb_object(analysis_file_name, behavior_module)
        nwb_analysis_file.add(
            nwb_file_name=nwb_file_name,
            analysis_file_name=analysis_file_name,
        )

        logger.info(f"Stored pose estimation in NWB: {analysis_file_name}")
        return analysis_file_name


@schema
class PoseParams(dj.Lookup):
    """Parameters for pose processing (orientation, centroid, smoothing).

    Attributes
    ----------
    pose_params : str
        Name for this parameter set
    orient : dict
        Orientation calculation parameters
    centroid : dict
        Centroid calculation parameters
    smoothing : dict
        Smoothing and interpolation parameters
    """

    definition = """
    pose_params: varchar(32)
    ---
    orient: blob  # Orientation calculation params
    centroid: blob  # Centroid calculation params
    smoothing: blob  # Smoothing/interpolation params
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
            "pose_params": "default",
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
            "pose_params": "4LED_default",
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
            "pose_params": "single_LED",
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
            "pose_params": "no_smoothing",
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

    @classmethod
    def insert_custom(
        cls,
        params_name: str,
        orient: dict,
        centroid: dict,
        smoothing: dict,
        **kwargs,
    ):
        """Insert custom parameter set.

        Parameters
        ----------
        params_name : str
            Name for this parameter set (max 32 chars)
        orient : dict
            Orientation parameters. Must include 'method' key.
            For two_pt: bodypart1, bodypart2
            For bisector: led1, led2, led3
            For none: no additional keys needed
            Optional: interpolate, smooth, smoothing_params
        centroid : dict
            Centroid parameters. Must include 'method' and 'points' keys.
            For 1pt: points={'point1': 'bodypart_name'}
            For 2pt: points={'point1': 'name1', 'point2': 'name2'},
                     max_LED_separation
            For 4pt: points={'greenLED': ..., 'redLED_C': ...,
                            'redLED_L': ..., 'redLED_R': ...},
                     max_LED_separation
        smoothing : dict
            Smoothing parameters.
            Keys: interpolate, interp_params, smooth, smoothing_params,
                  likelihood_thresh

        Raises
        ------
        ValueError
            If required parameters are missing or invalid

        Examples
        --------
        >>> PoseParams.insert_custom(
        ...     params_name="my_config",
        ...     orient={
        ...         "method": "two_pt",
        ...         "bodypart1": "nose",
        ...         "bodypart2": "tail_base",
        ...     },
        ...     centroid={
        ...         "method": "1pt",
        ...         "points": {"point1": "nose"},
        ...     },
        ...     smoothing={
        ...         "interpolate": True,
        ...         "interp_params": {"max_cm_to_interp": 20.0},
        ...         "smooth": True,
        ...         "smoothing_params": {
        ...             "method": "gaussian",
        ...             "std_dev": 0.01,
        ...         },
        ...         "likelihood_thresh": 0.9,
        ...     },
        ... )
        """
        # Validate parameters
        cls._validate_orient_params(orient)
        cls._validate_centroid_params(centroid)
        cls._validate_smoothing_params(smoothing)

        params = {
            "pose_params": params_name,
            "orient": orient,
            "centroid": centroid,
            "smoothing": smoothing,
        }
        cls.insert1(params, **kwargs)

    @classmethod
    def _validate_orient_params(cls, params: dict):
        """Validate orientation parameters."""
        if "method" not in params:
            raise ValueError("orient params must include 'method'")

        method = params["method"]
        if method not in cls.orient_methods:
            raise ValueError(
                f"Invalid orient method: {method}. "
                f"Must be one of: {list(cls.orient_methods.keys())}"
            )

        required = cls.orient_methods[method]
        missing = [k for k in required if k not in params]
        if missing:
            raise ValueError(
                f"{method} orientation requires: {required}. "
                f"Missing: {missing}"
            )

    @classmethod
    def _validate_centroid_params(cls, params: dict):
        """Validate centroid parameters."""
        if "method" not in params:
            raise ValueError("centroid params must include 'method'")
        if "points" not in params:
            raise ValueError("centroid params must include 'points'")

        method = params["method"]
        points = params["points"]

        if method not in cls.centroid_methods:
            raise ValueError(
                f"Invalid centroid method: {method}. "
                f"Must be one of: {list(cls.centroid_methods.keys())}"
            )

        spec = cls.centroid_methods[method]

        if len(points) != spec["n_points"]:
            raise ValueError(
                f"{method} centroid requires exactly "
                f"{spec['n_points']} point(s)"
            )

        if spec["required_point_keys"] is not None:
            if set(points.keys()) != spec["required_point_keys"]:
                raise ValueError(
                    f"{method} centroid requires points: "
                    f"{spec['required_point_keys']}. "
                    f"Got: {set(points.keys())}"
                )

        for k in spec["extra_required"]:
            if k not in params:
                raise ValueError(f"{method} centroid requires {k}")

    @classmethod
    def _validate_smoothing_params(cls, params: dict):
        """Validate smoothing parameters."""
        for k in cls.smoothing_required:
            if k not in params:
                raise ValueError(f"smoothing params must include '{k}'")

        if params.get("interpolate", False):
            if "interp_params" not in params:
                raise ValueError("interpolate=True requires interp_params")

        if params.get("smooth", False):
            if "smoothing_params" not in params:
                raise ValueError("smooth=True requires smoothing_params")
            smooth_params = params["smoothing_params"]
            if "method" not in smooth_params:
                raise ValueError("smoothing_params must include 'method'")

            from spyglass.position.utils.interpolation import SMOOTHING_METHODS

            method = smooth_params["method"]
            if method not in SMOOTHING_METHODS:
                raise ValueError(
                    f"Invalid smoothing method: {method}. "
                    f"Must be one of: "
                    f"{list(SMOOTHING_METHODS.keys())}"
                )


@schema
class PoseSelection(dj.Manual):
    definition = """
    -> PoseEstim
    -> PoseParams
    """


@schema
class PoseV2(dj.Computed):
    definition = """
    -> PoseSelection
    ---
    -> [nullable] AnalysisNwbfile
    orient_obj_id='': varchar(40)
    centroid_obj_id='': varchar(40)
    velocity_obj_id='': varchar(40)
    smoothed_pose_id='': varchar(40)
    """

    def fetch_obj(self, objects=None):
        """Fetch processed pose objects from NWB file.

        Parameters
        ----------
        objects : str or list of str, optional
            Which objects to fetch. Options: "orient", "centroid",
            "velocity", "smoothed_pose". If None, fetches all objects.
            Default: None

        Returns
        -------
        dict
            Dictionary with requested objects. Keys are object names,
            values are pynwb objects:
            - "orient": BehavioralTimeSeries with orientation data
            - "centroid": Position with centroid spatial series
            - "velocity": BehavioralTimeSeries with velocity data
            - "smoothed_pose": Position with smoothed position data

        Examples
        --------
        >>> # Fetch all objects
        >>> objs = (PoseV2 & key).fetch_obj()
        >>> orientation = objs["orient"].time_series["orientation"]
        >>> centroid = objs["centroid"].spatial_series["centroid"]
        >>>
        >>> # Fetch specific object
        >>> objs = (PoseV2 & key).fetch_obj("centroid")
        >>> centroid_data = objs["centroid"].spatial_series["centroid"].data[:]

        Notes
        -----
        - Uses fetch_nwb() to retrieve objects from AnalysisNwbfile
        - Object IDs are stored in table columns: orient_obj_id,
          centroid_obj_id, velocity_obj_id, smoothed_pose_id
        - Requires exactly one entry in the restriction
        """
        # Ensure single entry
        _ = self.ensure_single_entry()

        # Map user-friendly names to database column names
        object_map = {
            "orient": "orient_obj_id",
            "centroid": "centroid_obj_id",
            "velocity": "velocity_obj_id",
            "smoothed_pose": "smoothed_pose_id",
        }

        # Determine which objects to fetch
        if objects is None:
            # Fetch all objects
            fetch_objects = list(object_map.keys())
        elif isinstance(objects, str):
            # Single object requested
            fetch_objects = [objects]
        else:
            # List of objects requested
            fetch_objects = list(objects)

        # Validate requested objects
        invalid_objects = set(fetch_objects) - set(object_map.keys())
        if invalid_objects:
            raise ValueError(
                f"Invalid objects requested: {invalid_objects}. "
                f"Valid options: {list(object_map.keys())}"
            )

        # Fetch NWB objects using object ID attributes
        nwb_attrs = [object_map[obj] for obj in fetch_objects]
        nwb_data = self.fetch_nwb(*nwb_attrs)[0]

        # Build result dictionary with user-friendly names
        result = {}
        for obj_name in fetch_objects:
            db_col = object_map[obj_name]
            result[obj_name] = nwb_data[db_col]

        return result

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
        import pandas as pd

        # Ensure single entry
        _ = self.ensure_single_entry()

        # Fetch NWB objects
        objs = self.fetch_obj()

        # Extract data from NWB objects
        centroid_series = objs["centroid"].spatial_series["centroid"]
        orient_series = objs["orient"].time_series["orientation"]
        velocity_series = objs["velocity"].time_series["velocity"]

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
            Conversion factor from centimetres to pixels, by default 1.0.
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
            logger.warning(
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
        )
        if ret.returncode != 0:
            raise FileNotFoundError(
                f"ffprobe failed on {video_path}: {ret.stderr}"
            )
        fps = eval(ret.stdout.strip())  # e.g. "30000/1001" → 29.97

        # video_frame_inds: timestamps ARE frame indices (0, 1, 2, ...) since
        # DLC runs inference on every frame. Mirrors V1's video_frame_ind column.
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
            except Exception as e:
                logger.warning(
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
        """Process raw pose estimation with orientation, centroid, and smoothing.

        This method:
        1. Fetches raw pose data from PoseEstim
        2. Applies likelihood thresholding
        3. Calculates orientation
        4. Calculates centroid
        5. Applies interpolation and smoothing
        6. Calculates velocity
        7. Stores results in NWB file

        Parameters
        ----------
        key : dict
            Primary key from PoseSelection
        """
        from spyglass.position.utils.centroid import calculate_centroid
        from spyglass.position.utils.interpolation import (
            get_smoothing_function,
            interp_position,
        )
        from spyglass.position.utils.orientation import (
            bisector_orientation,
            get_span_start_stop,
            no_orientation,
            smooth_orientation,
            two_pt_orientation,
        )

        # Fail fast: verify Nwbfile link before any expensive computation.
        nwb_file_name = self._get_nwb_file_name(key)
        estim_entry = (PoseEstim & key).fetch1()
        if not estim_entry.get("analysis_file_name") or nwb_file_name is None:
            raise ValueError(
                "Cannot store processed pose: the parent PoseEstim entry "
                f"for VidFileGroup '{key['vid_group_id']}' has no "
                "AnalysisNwbfile. Ensure the video group is linked to a "
                "registered Nwbfile before populating."
            )

        # Fetch raw pose data
        logger.info(f"Processing pose for: {key}")
        pose_df = (PoseEstim & key).fetch1_dataframe()

        # Fetch parameters
        params = (PoseParams & key).fetch1()
        orient_params = params["orient"]
        centroid_params = params["centroid"]
        smooth_params = params["smoothing"]

        # Get sampling rate from timestamps
        timestamps = pose_df.index.values
        sampling_rate = 1 / np.median(np.diff(timestamps))

        # Step 1: Apply likelihood thresholding
        likelihood_thresh = smooth_params.get("likelihood_thresh", 0.95)
        pose_df = self._apply_likelihood_threshold(pose_df, likelihood_thresh)

        # Step 2: Calculate orientation
        logger.info("Calculating orientation...")
        orientation = self._calculate_orientation(
            pose_df, orient_params, timestamps, sampling_rate
        )

        # Step 3: Calculate centroid
        logger.info("Calculating centroid...")
        centroid = self._calculate_centroid(pose_df, centroid_params)

        # Step 4: Apply interpolation and smoothing to centroid
        logger.info("Smoothing centroid...")
        centroid_smooth = self._smooth_position(
            centroid, timestamps, sampling_rate, smooth_params
        )

        # Step 5: Calculate velocity
        logger.info("Calculating velocity...")
        velocity = self._calculate_velocity(
            centroid_smooth, timestamps, sampling_rate
        )

        # Step 6: Store results (nwb_file_name validated at top of make())
        logger.info("Storing results in NWB...")
        analysis_file_name, obj_ids = self._store_pose_nwb(
            {**key, "nwb_file_name": nwb_file_name},
            orientation,
            centroid_smooth,
            velocity,
            timestamps,
            sampling_rate,
        )
        self.insert1(
            {
                **key,
                "analysis_file_name": analysis_file_name,
                "orient_obj_id": obj_ids["orient"],
                "centroid_obj_id": obj_ids["centroid"],
                "velocity_obj_id": obj_ids["velocity"],
                "smoothed_pose_id": obj_ids["smoothed_pose"],
            }
        )
        logger.info("Pose processing complete!")

    @staticmethod
    def _apply_likelihood_threshold(
        pose_df: pd.DataFrame, likelihood_thresh: float
    ) -> pd.DataFrame:
        """Set position to NaN where likelihood is below threshold.

        Parameters
        ----------
        pose_df : pd.DataFrame
            Pose DataFrame with MultiIndex columns (scorer, bodypart, coord)
        likelihood_thresh : float
            Likelihood threshold (0-1)

        Returns
        -------
        pd.DataFrame
            DataFrame with low-likelihood positions set to NaN
        """
        # MultiIndex structure: (scorer, bodypart, coord)
        # coord can be 'x', 'y', 'likelihood'
        idx = pd.IndexSlice

        # Get all bodyparts (level 1 in MultiIndex)
        bodyparts = pose_df.columns.get_level_values(1).unique()

        for bodypart in bodyparts:
            # Check if likelihood column exists
            try:
                likelihood = pose_df.loc[:, idx[:, bodypart, "likelihood"]]
                # Set x, y to NaN where likelihood < threshold
                low_likelihood = likelihood.values.flatten() < likelihood_thresh
                pose_df.loc[low_likelihood, idx[:, bodypart, ["x", "y"]]] = (
                    np.nan
                )
            except KeyError:
                # No likelihood column for this bodypart, skip
                logger.warning(
                    f"No likelihood column for {bodypart}, skipping threshold"
                )
                continue

        return pose_df

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
            raise ValueError(f"Unknown orientation method: {method}")

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
        """Calculate velocity from position.

        Parameters
        ----------
        position : np.ndarray
            Position array, shape (n_frames, 2)
        timestamps : np.ndarray
            Timestamps
        sampling_rate : float
            Sampling rate in Hz

        Returns
        -------
        np.ndarray
            Velocity in cm/s, shape (n_frames,)
        """
        # Calculate displacement
        dx = np.diff(position[:, 0])
        dy = np.diff(position[:, 1])
        displacement = np.sqrt(dx**2 + dy**2)

        # Calculate time differences
        dt = np.diff(timestamps)

        # Velocity = displacement / time
        velocity = displacement / dt

        # Pad with NaN at the beginning to match length
        velocity = np.concatenate([[np.nan], velocity])

        return velocity

    @staticmethod
    def _flatten_multiindex(pose_df: pd.DataFrame) -> pd.DataFrame:
        """Flatten MultiIndex columns to single level.

        Converts (scorer, bodypart, coord) to (bodypart, coord)

        Parameters
        ----------
        pose_df : pd.DataFrame
            DataFrame with MultiIndex columns

        Returns
        -------
        pd.DataFrame
            DataFrame with (bodypart, coord) columns
        """
        # Check if already flattened
        if not isinstance(pose_df.columns, pd.MultiIndex):
            return pose_df

        # Flatten by dropping scorer level (level 0)
        if pose_df.columns.nlevels == 3:
            # Drop scorer level, keep bodypart and coord
            pose_df.columns = pose_df.columns.droplevel(0)

        return pose_df

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
        analysis_file_name = AnalysisNwbfile().create(key["nwb_file_name"])
        nwb_analysis_file = AnalysisNwbfile()

        # Create Position object for centroid
        position = pynwb.behavior.Position()
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
        orientation_ts = pynwb.behavior.BehavioralTimeSeries()
        orientation_ts.create_timeseries(
            name="orientation",
            timestamps=timestamps,
            data=orientation,
            unit="radians",
            description="Head orientation in radians",
            comments="Counter-clockwise from positive x-axis, range [-π, π]",
        )

        # Create BehavioralTimeSeries for velocity
        velocity_ts = pynwb.behavior.BehavioralTimeSeries()
        velocity_ts.create_timeseries(
            name="velocity",
            timestamps=timestamps,
            data=velocity,
            unit="cm/s",
            description="Speed in cm/s",
            comments="Euclidean distance traveled per unit time",
        )

        # Create Position object for smoothed pose (centroid with metadata)
        smoothed_pose = pynwb.behavior.Position()
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
            "orient": nwb_analysis_file.add_nwb_object(
                analysis_file_name, orientation_ts
            ),
            "centroid": nwb_analysis_file.add_nwb_object(
                analysis_file_name, position
            ),
            "velocity": nwb_analysis_file.add_nwb_object(
                analysis_file_name, velocity_ts
            ),
            "smoothed_pose": nwb_analysis_file.add_nwb_object(
                analysis_file_name, smoothed_pose
            ),
        }

        # Register the relationship between raw and analysis NWB files
        nwb_analysis_file.add(
            nwb_file_name=key["nwb_file_name"],
            analysis_file_name=analysis_file_name,
        )

        return analysis_file_name, obj_ids

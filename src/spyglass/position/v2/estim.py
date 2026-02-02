from datetime import datetime
from pathlib import Path
from typing import Union

import datajoint as dj
import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO

from spyglass.common import AnalysisNwbfile
from spyglass.position.v2.train import Model, ModelParams, Skeleton
from spyglass.position.v2.video import VidFileGroup
from spyglass.utils import logger

# ----------------------------- Optional imports ------------------------------
try:
    import ndx_pose
except ImportError:
    ndx_pose = None

schema = dj.schema("cbroz_position_v2_estim")


@schema
class PoseEstim(dj.Manual):
    definition = """
    -> Model
    -> VidFileGroup
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
                "PoseEstim entry must reference an AnalysisNwbfile containing "
                "pose estimation data."
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

    @staticmethod
    def _validate_orient_params(params: dict):
        """Validate orientation parameters."""
        if "method" not in params:
            raise ValueError("orient params must include 'method'")

        method = params["method"]
        if method not in ["two_pt", "bisector", "none"]:
            raise ValueError(
                f"Invalid orient method: {method}. "
                "Must be 'two_pt', 'bisector', or 'none'"
            )

        if method == "two_pt":
            required = ["bodypart1", "bodypart2"]
            missing = [k for k in required if k not in params]
            if missing:
                raise ValueError(
                    f"two_pt orientation requires: {required}. Missing: {missing}"
                )
        elif method == "bisector":
            required = ["led1", "led2", "led3"]
            missing = [k for k in required if k not in params]
            if missing:
                raise ValueError(
                    f"bisector orientation requires: {required}. Missing: {missing}"
                )

    @staticmethod
    def _validate_centroid_params(params: dict):
        """Validate centroid parameters."""
        if "method" not in params:
            raise ValueError("centroid params must include 'method'")
        if "points" not in params:
            raise ValueError("centroid params must include 'points'")

        method = params["method"]
        points = params["points"]

        if method not in ["1pt", "2pt", "4pt"]:
            raise ValueError(
                f"Invalid centroid method: {method}. "
                "Must be '1pt', '2pt', or '4pt'"
            )

        # Validate number of points matches method
        if method == "1pt" and len(points) != 1:
            raise ValueError("1pt centroid requires exactly 1 point")
        elif method == "2pt":
            if len(points) != 2:
                raise ValueError("2pt centroid requires exactly 2 points")
            if "max_LED_separation" not in params:
                raise ValueError("2pt centroid requires max_LED_separation")
        elif method == "4pt":
            if len(points) != 4:
                raise ValueError("4pt centroid requires exactly 4 points")
            required_keys = {"greenLED", "redLED_C", "redLED_L", "redLED_R"}
            if set(points.keys()) != required_keys:
                raise ValueError(
                    f"4pt centroid requires points: {required_keys}. "
                    f"Got: {set(points.keys())}"
                )
            if "max_LED_separation" not in params:
                raise ValueError("4pt centroid requires max_LED_separation")

    @staticmethod
    def _validate_smoothing_params(params: dict):
        """Validate smoothing parameters."""
        if "likelihood_thresh" not in params:
            raise ValueError(
                "smoothing params must include 'likelihood_thresh'"
            )

        # Validate interpolation params
        if params.get("interpolate", False):
            if "interp_params" not in params:
                raise ValueError("interpolate=True requires interp_params")

        # Validate smoothing params
        if params.get("smooth", False):
            if "smoothing_params" not in params:
                raise ValueError("smooth=True requires smoothing_params")
            smooth_params = params["smoothing_params"]
            if "method" not in smooth_params:
                raise ValueError("smoothing_params must include 'method'")

            # Import here to avoid circular dependency
            from spyglass.position.utils.interpolation import SMOOTHING_METHODS

            method = smooth_params["method"]
            if method not in SMOOTHING_METHODS:
                raise ValueError(
                    f"Invalid smoothing method: {method}. "
                    f"Must be one of: {list(SMOOTHING_METHODS.keys())}"
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
    -> AnalysisNwbfile
    orient_obj_id: varchar(40)
    centroid_obj_id: varchar(40)
    velocity_obj_id: varchar(40)
    smoothed_pose_id: varchar(40)
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

    def make_video(self):
        raise NotImplementedError("Make video from pose")

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

        # Step 6: Store in NWB
        logger.info("Storing results in NWB...")
        nwb_file_name, obj_ids = self._store_pose_nwb(
            key,
            orientation,
            centroid_smooth,
            velocity,
            timestamps,
            sampling_rate,
        )

        # Step 7: Insert into table
        self.insert1(
            {
                **key,
                "analysis_file_name": nwb_file_name,
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

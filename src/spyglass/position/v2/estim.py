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
    -> AnalysisNwbfile
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
        if ndx_pose is None:
            raise ImportError(
                "ndx-pose is required to fetch pose data. "
                "Install with: pip install ndx-pose>=0.2.0"
            )

        # Fetch entry
        entry = self.fetch1()
        nwb_file_name = entry["analysis_file_name"]

        logger.debug(f"Fetching pose data from NWB: {nwb_file_name}")

        # Get NWB file path
        nwb_path = (
            AnalysisNwbfile() & {"analysis_file_name": nwb_file_name}
        ).fetch1("analysis_file_abs_path")

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
    definition = """
    pose_params: varchar(32)
    ---
    # TODO: also moseq/PCA?
    orient: blob
    centroid: blob
    smoothing: blob
    """


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

    def fetch_obj(self, object=["orient", "centroid", "velocity", "smoothing"]):
        raise NotImplementedError("Return obj(s)")

    def make_video(self):
        raise NotImplementedError("Make video from pose")

    def make(self, key):
        """Make pose estimation for each entry in PoseSelection."""
        raise NotImplementedError(
            "Make pose estimation for each entry in PoseSelection."
        )

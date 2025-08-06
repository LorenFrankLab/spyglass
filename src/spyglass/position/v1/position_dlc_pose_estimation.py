from datetime import datetime
from pathlib import Path

import datajoint as dj
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynwb
from IPython.display import display

from spyglass.common.common_behav import (  # noqa: F401
    RawPosition,
    VideoFile,
    convert_epoch_interval_name_to_position_interval_name,
)
from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.position.v1.dlc_utils import (
    file_log,
    find_mp4,
    get_video_info,
    infer_output_dir,
)
from spyglass.position.v1.position_dlc_model import DLCModel
from spyglass.settings import test_mode
from spyglass.utils import SpyglassMixin, logger

from . import dlc_reader

schema = dj.schema("position_v1_dlc_pose_estimation")


@schema
class DLCPoseEstimationSelection(SpyglassMixin, dj.Manual):
    definition = """
    -> VideoFile                           # Session -> Recording + File part table
    -> DLCModel                                    # Must specify a DLC project_path
    ---
    task_mode='load' : enum('load', 'trigger')  # load results or trigger computation
    video_path : varchar(120)                   # path to video file
    pose_estimation_output_dir='': varchar(255) # output dir relative to the root dir
    pose_estimation_params=null  : longblob     # analyze_videos params, if not default
    """
    log_path = None

    @classmethod
    def get_video_crop(cls, video_path, crop_input=None):
        """
        Queries the user to determine the cropping parameters for a given video

        Parameters
        ----------
        video_path : str
            path to the video file

        Returns
        -------
        crop_ints : list
            list of 4 integers [x min, x max, y min, y max]
        crop_input : str, optional
            input string to determine cropping parameters. If None, query user
        """
        import cv2

        video_path = str(video_path)
        cap = cv2.VideoCapture(video_path)
        _, frame = cap.read()
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.imshow(frame)
        xlims = ax.get_xlim()
        ylims = ax.get_ylim()
        ax.set_xticks(np.arange(xlims[0], xlims[-1], 50))
        ax.set_yticks(np.arange(ylims[0], ylims[-1], -50))
        ax.grid(visible=True, color="white", lw=0.5, alpha=0.5)
        display(fig)

        if crop_input is None and not test_mode:
            crop_input = input(  # pragma: no cover
                "Please enter the crop parameters for your video in format "
                + "xmin, xmax, ymin, ymax, or 'none'\n"
            )

        plt.close()
        if crop_input is None or crop_input.lower() == "none":
            return None
        crop_ints = [int(val) for val in crop_input.split(",")]
        if not all(isinstance(val, int) and val >= 0 for val in crop_ints):
            raise ValueError("Crop parameters must be non-negative integers.")
        return crop_ints

    def insert_estimation_task(
        self,
        key,
        task_mode="trigger",  # load or trigger
        params: dict = None,
        check_crop=None,
        skip_duplicates=True,
    ):
        """
        Insert PoseEstimationTask in inferred output dir.
        From Datajoint Elements

        Parameters
        ----------
        key: dict
            DataJoint key specifying a pairing of VideoRecording and Model.
        task_mode: bool, optional
            Default 'trigger' computation. Or 'load' existing results.
        params (dict): Optional. Parameters passed to DLC's analyze_videos:
            videotype, gputouse, save_as_csv, batchsize, cropping,
            TFGPUinference, dynamic, robust_nframes, allow_growth, use_shelve
        """
        output_dir = infer_output_dir(key)
        self.log_path = Path(output_dir) / "log.log"
        self._insert_est_with_log(
            key, task_mode, params, check_crop, skip_duplicates, output_dir
        )
        logger.info("inserted entry into Pose Estimation Selection")
        return {**key, "task_mode": task_mode}

    @file_log(logger, console=False)
    def _insert_est_with_log(
        self, key, task_mode, params, check_crop, skip_duplicates, output_dir
    ):
        v_path, v_fname, _, _ = get_video_info(key)
        if not v_path:
            raise FileNotFoundError(f"Video file not found for {key}")
        logger.info("Pose Estimation Selection")
        logger.info(f"video_dir: {v_path}")
        v_path = find_mp4(video_path=Path(v_path), video_filename=v_fname)
        if check_crop:
            params["cropping"] = self.get_video_crop(
                video_path=v_path.as_posix()
            )
        self.insert1(
            {
                **key,
                "task_mode": task_mode,
                "pose_estimation_params": params,
                "video_path": v_path,
                "pose_estimation_output_dir": output_dir,
            },
            skip_duplicates=skip_duplicates,
        )


@schema
class DLCPoseEstimation(SpyglassMixin, dj.Computed):
    definition = """
    -> DLCPoseEstimationSelection
    ---
    pose_estimation_time: datetime  # time of generation of this set of DLC results
    meters_per_pixel : double       # conversion of meters per pixel for analyzed video
    """

    class BodyPart(SpyglassMixin, dj.Part):
        definition = """ # uses DeepLabCut h5 output for body part position
        -> DLCPoseEstimation
        -> DLCModel.BodyPart
        ---
        -> AnalysisNwbfile
        dlc_pose_estimation_position_object_id : varchar(80)
        dlc_pose_estimation_likelihood_object_id : varchar(80)
        """

        _nwb_table = AnalysisNwbfile
        log_path = None

        def fetch1_dataframe(self) -> pd.DataFrame:
            """Fetch a single bodypart dataframe."""
            _ = self.ensure_single_entry()
            nwb_data = self.fetch_nwb()[0]
            index = pd.Index(
                np.asarray(
                    nwb_data["dlc_pose_estimation_position"]
                    .get_spatial_series()
                    .timestamps
                ),
                name="time",
            )
            COLUMNS = [
                "video_frame_ind",
                "x",
                "y",
                "likelihood",
            ]
            return pd.DataFrame(
                np.concatenate(
                    (
                        np.asarray(
                            nwb_data["dlc_pose_estimation_likelihood"]
                            .time_series["video_frame_ind"]
                            .data,
                            dtype=int,
                        )[:, np.newaxis],
                        np.asarray(
                            nwb_data["dlc_pose_estimation_position"]
                            .get_spatial_series()
                            .data
                        ),
                        np.asarray(
                            nwb_data["dlc_pose_estimation_likelihood"]
                            .time_series["likelihood"]
                            .data
                        )[:, np.newaxis],
                    ),
                    axis=1,
                ),
                columns=COLUMNS,
                index=index,
            )

    def make(self, key):
        """.populate() method will launch training for each PoseEstimationTask"""
        self.log_path = (
            Path(infer_output_dir(key=key, makedir=False)) / "log.log"
        )
        self._logged_make(key)

    @file_log(logger, console=True)
    def _logged_make(self, key):
        METERS_PER_CM = 0.01

        logger.info("----------------------")
        logger.info("Pose Estimation")
        # ID model and directories
        dlc_model = (DLCModel & key).fetch1()
        bodyparts = (DLCModel.BodyPart & key).fetch("bodypart")
        task_mode, analyze_video_params, video_path, output_dir = (
            DLCPoseEstimationSelection & key
        ).fetch1(
            "task_mode",
            "pose_estimation_params",
            "video_path",
            "pose_estimation_output_dir",
        )
        analyze_video_params = analyze_video_params or {}

        project_path = dlc_model["project_path"]

        # Trigger PoseEstimation
        if task_mode == "trigger":
            dlc_reader.do_pose_estimation(
                video_path,
                dlc_model,
                project_path,
                output_dir,
                **analyze_video_params,
            )

        dlc_result = dlc_reader.PoseEstimation(output_dir)
        creation_time = datetime.fromtimestamp(
            dlc_result.creation_time
        ).strftime("%Y-%m-%d %H:%M:%S")

        logger.info("getting raw position")
        interval_list_name = (
            convert_epoch_interval_name_to_position_interval_name(
                {
                    "nwb_file_name": key["nwb_file_name"],
                    "epoch": key["epoch"],
                },
            )
        )
        if interval_list_name:
            spatial_series = (
                RawPosition()
                & {**key, "interval_list_name": interval_list_name}
            ).fetch_nwb()[0]["raw_position"]
        else:
            spatial_series = None  # # pragma: no cover

        _, _, meters_per_pixel, video_time = get_video_info(key)
        key["meters_per_pixel"] = meters_per_pixel

        # TODO: should get timestamps from VideoFile, but need the
        # video_frame_ind from RawPosition, which also has timestamps

        # Insert entry into DLCPoseEstimation
        logger.info(
            "Inserting %s, epoch %02d into DLCPoseEsimation",
            key["nwb_file_name"],
            key["epoch"],
        )
        self.insert1({**key, "pose_estimation_time": creation_time})

        meters_per_pixel = key.pop("meters_per_pixel")
        body_parts = dlc_result.df.columns.levels[0]
        body_parts_df = {}
        # Insert dlc pose estimation into analysis NWB file for
        # each body part.
        for body_part in bodyparts:
            if body_part in body_parts:
                body_parts_df[body_part] = pd.DataFrame.from_dict(
                    {
                        c: dlc_result.df.get(body_part).get(c).values
                        for c in dlc_result.df.get(body_part).columns
                    }
                )
        idx = pd.IndexSlice
        for body_part, part_df in body_parts_df.items():
            logger.info("converting to cm")
            part_df = convert_to_cm(part_df, meters_per_pixel)
            logger.info("adding timestamps to DataFrame")
            part_df = add_timestamps(
                part_df,
                pos_time=getattr(spatial_series, "timestamps", video_time),
                video_time=video_time,
            )
            key["bodypart"] = body_part
            key["analysis_file_name"] = AnalysisNwbfile().create(
                key["nwb_file_name"]
            )
            position = pynwb.behavior.Position()
            likelihood = pynwb.behavior.BehavioralTimeSeries()
            position.create_spatial_series(
                name="position",
                timestamps=part_df.time.to_numpy(),
                conversion=METERS_PER_CM,
                data=part_df.loc[:, idx[("x", "y")]].to_numpy(),
                reference_frame=getattr(spatial_series, "reference_frame", ""),
                comments=getattr(spatial_series, "comments", "no comments"),
                description="x_position, y_position",
            )
            likelihood.create_timeseries(
                name="likelihood",
                timestamps=part_df.time.to_numpy(),
                data=part_df.loc[:, idx["likelihood"]].to_numpy(),
                unit="likelihood",
                comments="no comments",
                description="likelihood",
            )
            likelihood.create_timeseries(
                name="video_frame_ind",
                timestamps=part_df.time.to_numpy(),
                data=part_df.loc[:, idx["video_frame_ind"]].to_numpy(),
                unit="index",
                comments="no comments",
                description="video_frame_ind",
            )
            nwb_analysis_file = AnalysisNwbfile()

            key["dlc_pose_estimation_position_object_id"] = (
                nwb_analysis_file.add_nwb_object(
                    analysis_file_name=key["analysis_file_name"],
                    nwb_object=position,
                )
            )
            key["dlc_pose_estimation_likelihood_object_id"] = (
                nwb_analysis_file.add_nwb_object(
                    analysis_file_name=key["analysis_file_name"],
                    nwb_object=likelihood,
                )
            )
            nwb_analysis_file.add(
                nwb_file_name=key["nwb_file_name"],
                analysis_file_name=key["analysis_file_name"],
            )
            self.BodyPart.insert1(key)

    def fetch_dataframe(self, *attrs, **kwargs) -> pd.DataFrame:
        """Fetch a concatenated dataframe of all bodyparts."""
        entries = (self.BodyPart & self).fetch("KEY", log_export=False)
        nwb_data_dict = {
            entry["bodypart"]: (self.BodyPart() & entry).fetch_nwb()[0]
            for entry in entries
        }
        index = pd.Index(
            np.asarray(
                nwb_data_dict[entries[0]["bodypart"]][
                    "dlc_pose_estimation_position"
                ]
                .get_spatial_series()
                .timestamps
            ),
            name="time",
        )
        COLUMNS = ["video_frame_ind", "x", "y", "likelihood"]
        return pd.concat(
            {
                entry["bodypart"]: pd.DataFrame(
                    np.concatenate(
                        (
                            np.asarray(
                                nwb_data_dict[entry["bodypart"]][
                                    "dlc_pose_estimation_likelihood"
                                ]
                                .time_series["video_frame_ind"]
                                .data,
                                dtype=int,
                            )[:, np.newaxis],
                            np.asarray(
                                nwb_data_dict[entry["bodypart"]][
                                    "dlc_pose_estimation_position"
                                ]
                                .get_spatial_series()
                                .data
                            ),
                            np.asarray(
                                nwb_data_dict[entry["bodypart"]][
                                    "dlc_pose_estimation_likelihood"
                                ]
                                .time_series["likelihood"]
                                .data
                            )[:, np.newaxis],
                        ),
                        axis=1,
                    ),
                    columns=COLUMNS,
                    index=index,
                )
                for entry in entries
            },
            axis=1,
        )

    def fetch_video_path(self, key: dict = True) -> str:
        """Return the video path for pose estimate

        Parameters
        ----------
        key : dict, optional
            DataJoint key, by default dict()
        Returns
        -------
        str
            absolute path to video file
        """
        key = (self & key).fetch1("KEY", log_export=False)
        return (DLCPoseEstimationSelection & key).fetch1("video_path")


def convert_to_cm(df, meters_to_pixels):
    """Converts x and y columns from pixels to cm"""
    CM_TO_METERS = 100
    idx = pd.IndexSlice
    df.loc[:, idx[("x", "y")]] *= meters_to_pixels * CM_TO_METERS
    return df


def add_timestamps(
    df: pd.DataFrame, pos_time: np.ndarray, video_time: np.ndarray
) -> pd.DataFrame:
    """
    Takes timestamps from raw_pos_df and adds to df,
    which is returned with timestamps and their matching video frame index

    Parameters
    ----------
    df : pd.DataFrame
        pose estimation dataframe to add timestamps
    pos_time : np.ndarray
        numpy array containing timestamps from the raw position object
    video_time: np.ndarray
        numpy array containing timestamps from the video file

    Returns
    -------
    pd.DataFrame
        original df with timestamps and video_frame_ind as new columns
    """
    first_video_frame = np.searchsorted(video_time, pos_time[0])
    video_frame_ind = np.arange(first_video_frame, len(video_time))
    time_df = pd.DataFrame(
        index=video_frame_ind,
        data=video_time[first_video_frame:],
        columns=["time"],
    )
    df = df.join(time_df)
    # Drop indices where time is NaN
    df = df.dropna(subset=["time"])
    # Add video_frame_ind as column
    df = df.rename_axis("video_frame_ind").reset_index()
    return df

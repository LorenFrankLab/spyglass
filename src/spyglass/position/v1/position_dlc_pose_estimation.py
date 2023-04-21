import os
from datetime import datetime

import cv2
import datajoint as dj
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from ...common.common_behav import RawPosition, VideoFile  # noqa: F401
from ...common.common_nwbfile import AnalysisNwbfile
from ...utils.dj_helper_fn import fetch_nwb
from .dlc_utils import OutputLogger, infer_output_dir
from .position_dlc_model import DLCModel

schema = dj.schema("position_v1_dlc_pose_estimation")


@schema
class DLCPoseEstimationSelection(dj.Manual):
    definition = """
    -> VideoFile                           # Session -> Recording + File part table
    -> DLCModel                                    # Must specify a DLC project_path
    ---
    task_mode='load' : enum('load', 'trigger')  # load results or trigger computation
    video_path : varchar(120)                   # path to video file
    pose_estimation_output_dir='': varchar(255) # output dir relative to the root dir
    pose_estimation_params=null  : longblob     # analyze_videos params, if not default
    """

    @classmethod
    def get_video_crop(cls, video_path):
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
        """

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
        crop_input = input(
            "Please enter the crop parameters for your video in format xmin, xmax, ymin, ymax, or 'none'\n"
        )
        plt.close()
        if crop_input.lower() == "none":
            return None
        crop_ints = [int(val) for val in crop_input.split(",")]
        assert all(isinstance(val, int) for val in crop_ints)
        return crop_ints

    @classmethod
    def insert_estimation_task(
        cls,
        key,
        task_mode="trigger",
        params: dict = None,
        check_crop=None,
        skip_duplicates=True,
    ):
        """
        Insert PoseEstimationTask in inferred output dir.
        From Datajoint Elements

        Parameters
        ----------
        key: DataJoint key specifying a pairing of VideoRecording and Model.
        task_mode (bool): Default 'trigger' computation. Or 'load' existing results.
        params (dict): Optional. Parameters passed to DLC's analyze_videos:
            videotype, gputouse, save_as_csv, batchsize, cropping, TFGPUinference,
            dynamic, robust_nframes, allow_growth, use_shelve
        """
        from .dlc_utils import check_videofile, get_video_path

        video_path, video_filename, _, _ = get_video_path(key)
        output_dir = infer_output_dir(key)
        with OutputLogger(
            name=f"{key['nwb_file_name']}_{key['epoch']}_{key['dlc_model_name']}_log",
            path=f"{output_dir.as_posix()}/log.log",
        ) as logger:
            logger.logger.info("Pose Estimation Selection")
            video_dir = os.path.dirname(video_path) + "/"
            logger.logger.info("video_dir: %s", video_dir)
            video_path = check_videofile(
                video_path=video_dir, video_filename=video_filename
            )[0]
            if check_crop is not None:
                params["cropping"] = cls.get_video_crop(
                    video_path=video_path.as_posix()
                )
            cls.insert1(
                {
                    **key,
                    "task_mode": task_mode,
                    "pose_estimation_params": params,
                    "video_path": video_path,
                    "pose_estimation_output_dir": output_dir,
                },
                skip_duplicates=skip_duplicates,
            )
        logger.logger.info("inserted entry into Pose Estimation Selection")
        return {**key, "task_mode": task_mode}


@schema
class DLCPoseEstimation(dj.Computed):
    definition = """
    -> DLCPoseEstimationSelection
    ---
    pose_estimation_time: datetime  # time of generation of this set of DLC results
    meters_per_pixel : double       # conversion of meters per pixel for analyzed video
    """

    class BodyPart(dj.Part):
        definition = """ # uses DeepLabCut h5 output for body part position
        -> DLCPoseEstimation
        -> DLCModel.BodyPart
        ---
        -> AnalysisNwbfile
        dlc_pose_estimation_object_id : varchar(80)
        """

        def fetch_nwb(self, *attrs, **kwargs):
            return fetch_nwb(
                self, (AnalysisNwbfile, "analysis_file_abs_path"), *attrs, **kwargs
            )

        def fetch1_dataframe(self):
            return self.fetch_nwb()[0]["dlc_pose_estimation"].set_index("time")

    def make(self, key):
        """.populate() method will launch training for each PoseEstimationTask"""
        from . import dlc_reader

        output_dir = infer_output_dir(key=key, makedir=False)
        with OutputLogger(
            name=f"{key['nwb_file_name']}_{key['epoch']}_{key['dlc_model_name']}_log",
            path=f"{output_dir.as_posix()}/log.log",
        ) as logger:
            logger.logger.info("----------------------")
            logger.logger.info("Pose Estimation")
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
            creation_time = datetime.fromtimestamp(dlc_result.creation_time).strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            logger.logger.info("getting raw position")
            interval_list_name = f"pos {key['epoch']-1} valid times"
            raw_position = (
                RawPosition()
                & {
                    "nwb_file_name": key["nwb_file_name"],
                    "interval_list_name": interval_list_name,
                }
            ).fetch_nwb()[0]
            raw_pos_df = pd.DataFrame(
                data=raw_position["raw_position"].data,
                index=pd.Index(raw_position["raw_position"].timestamps, name="time"),
                columns=raw_position["raw_position"].description.split(", "),
            )
            # TODO: should get timestamps from VideoFile, but need the video_frame_ind from RawPosition,
            # which also has timestamps
            key["meters_per_pixel"] = raw_position["raw_position"].conversion

            # Insert entry into DLCPoseEstimation
            logger.logger.info(
                "Inserting %s, epoch %02d into DLCPoseEsimation",
                key["nwb_file_name"],
                key["epoch"],
            )
            self.insert1({**key, "pose_estimation_time": creation_time})
            meters_per_pixel = key["meters_per_pixel"]
            del key["meters_per_pixel"]
            body_parts = dlc_result.df.columns.levels[0]
            body_parts_df = {}
            # Insert dlc pose estimation into analysis NWB file for each body part.
            for body_part in bodyparts:
                if body_part in body_parts:
                    body_parts_df[body_part] = pd.DataFrame.from_dict(
                        {
                            c: dlc_result.df.get(body_part).get(c).values
                            for c in dlc_result.df.get(body_part).columns
                        }
                    )
            for body_part, part_df in body_parts_df.items():
                logger.logger.info("converting to cm")
                part_df = convert_to_cm(part_df, meters_per_pixel)
                logger.logger.info("adding timestamps to DataFrame")
                part_df = add_timestamps(part_df, raw_pos_df.copy())
                key["bodypart"] = body_part
                key["analysis_file_name"] = AnalysisNwbfile().create(
                    key["nwb_file_name"]
                )
                nwb_analysis_file = AnalysisNwbfile()
                key["dlc_pose_estimation_object_id"] = nwb_analysis_file.add_nwb_object(
                    analysis_file_name=key["analysis_file_name"],
                    nwb_object=part_df,
                )
                nwb_analysis_file.add(
                    nwb_file_name=key["nwb_file_name"],
                    analysis_file_name=key["analysis_file_name"],
                )
                self.BodyPart.insert1(key)

    def fetch_dataframe(self, *attrs, **kwargs):
        entries = (self.BodyPart & self).fetch("KEY")
        return pd.concat(
            {
                entry["bodypart"]: (self.BodyPart & entry).fetch_nwb()[0][
                    "dlc_pose_estimation"
                ]
                for entry in entries
            },
            axis=1,
        )


def convert_to_cm(df, meters_to_pixels):
    CM_TO_METERS = 100
    idx = pd.IndexSlice
    df.loc[:, idx[("x", "y")]] *= meters_to_pixels * CM_TO_METERS
    return df


def add_timestamps(df: pd.DataFrame, raw_pos_df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes timestamps from raw_pos_df and adds to df,
    which is returned with timestamps and their matching video frame index

    Parameters
    ----------
    df : pd.DataFrame
        pose estimation dataframe to add timestamps
    raw_pos_df : pd.DataFrame
        dataframe fetched from RawPosition that
        contains timestamps and video frame inds

    Returns
    -------
    pd.DataFrame
        original df with timestamps and video_frame_ind as new columns
    """

    raw_pos_df = raw_pos_df.drop(
        columns=[
            column for column in raw_pos_df.columns if column not in ["video_frame_ind"]
        ]
    )
    raw_pos_df["time"] = raw_pos_df.index
    raw_pos_df.set_index("video_frame_ind", inplace=True)
    df = df.join(raw_pos_df)
    # Drop indices where time is NaN
    df = df.dropna(subset=["time"])
    # Add video_frame_ind as column
    df = df.rename_axis("video_frame_ind").reset_index()
    return df

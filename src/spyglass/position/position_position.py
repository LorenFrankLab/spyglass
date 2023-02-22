import os
import datajoint as dj
import pandas as pd
import numpy as np
from typing import Dict
from pathlib import Path
import pynwb
from tqdm import tqdm as tqdm
from ..common.dj_helper_fn import fetch_nwb
from ..common.common_behav import VideoFile, RawPosition
from ..common.common_nwbfile import AnalysisNwbfile
from ..common.common_interval import IntervalList

from .position_dlc_pose_estimation import (
    DLCPoseEstimation,
    DLCPoseEstimationSelection,
)
from .dlc_utils import get_video_path, check_videofile
from .position_dlc_selection import DLCPos
from .position_trodes_position import TrodesPos
from ..common.common_position import IntervalPositionInfo as CommonIntervalPositionInfo

schema = dj.schema("position_position")

_valid_data_sources = ["DLC", "Trodes", "Common"]


@schema
class FinalPosition(dj.Manual):
    """
    Table to identify source of Position Information from upstream options
    (e.g. DLC, Trodes, etc...) To add another upstream option, a new Part table
    should be added in the same syntax as DLCPos and TrodesPos and

    Note: all part tables need to be named using the source+"Pos" convention
    i.e. if the source='DLC', then the table is DLCPos
    """

    definition = """
    -> IntervalList
    source: varchar(40)
    position_id: int
    ---
    """

    class DLCPos(dj.Part):
        """
        Table to pass-through upstream DLC Pose Estimation information
        """

        definition = """
        -> FinalPosition
        -> DLCPos
        ---
        -> AnalysisNwbfile
        position_object_id : varchar(80)
        orientation_object_id : varchar(80)
        velocity_object_id : varchar(80)
        """

        def fetch_nwb(self, *attrs, **kwargs):
            return fetch_nwb(
                self, (AnalysisNwbfile, "analysis_file_abs_path"), *attrs, **kwargs
            )

    class TrodesPos(dj.Part):
        """
        Table to pass-through upstream Trodes Position Tracking information
        """

        definition = """
        -> FinalPosition
        -> TrodesPos
        ---
        -> AnalysisNwbfile
        position_object_id : varchar(80)
        orientation_object_id : varchar(80)
        velocity_object_id : varchar(80)
        """

        def fetch_nwb(self, *attrs, **kwargs):
            return fetch_nwb(
                self, (AnalysisNwbfile, "analysis_file_abs_path"), *attrs, **kwargs
            )

    class CommonPos(dj.Part):
        """
        Table to pass-through upstream Trodes Position Tracking information
        """

        definition = """
        -> FinalPosition
        -> CommonIntervalPositionInfo
        ---
        -> AnalysisNwbfile
        position_object_id : varchar(80)
        orientation_object_id : varchar(80)
        velocity_object_id : varchar(80)
        """

        def fetch_nwb(self, *attrs, **kwargs):
            return fetch_nwb(
                self, (AnalysisNwbfile, "analysis_file_abs_path"), *attrs, **kwargs
            )

    def insert1(self, key, params: Dict = None, **kwargs):
        """Overrides insert1 to also insert into specific part table.

        Parameters
        ----------
        key : Dict
            key specifying the entry to insert
        params : Dict, optional
            A dictionary containing all table entries
            not specified by the parent table (PosMerge)
        """
        assert (
            key["source"] in _valid_data_sources
        ), f"source needs to be one of {_valid_data_sources}"
        position_id = key.get("position_id", None)
        if position_id is None:
            key["position_id"] = (
                dj.U().aggr(self & key, n="max(position_id)").fetch1("n") or 0
            ) + 1
        else:
            id = (self & key).fetch("position_id")
            if len(id) > 0:
                position_id = max(id) + 1
            else:
                position_id = max(0, position_id)
            key["position_id"] = position_id
        super().insert1(key, **kwargs)
        source = key["source"]
        part_table = getattr(self, f"{source}Pos")
        # TODO: The parent table to refer to is hard-coded here, expecting it to be the second
        # Table in the definition. This could be more flexible.
        if params:
            table_query = (
                dj.FreeTable(dj.conn(), full_table_name=part_table.parents()[1])
                & key
                & params
            )
        else:
            table_query = (
                dj.FreeTable(dj.conn(), full_table_name=part_table.parents()[1]) & key
            )
        if any("head" in col for col in list(table_query.fetch().dtype.fields.keys())):
            (
                analysis_file_name,
                position_object_id,
                orientation_object_id,
                velocity_object_id,
            ) = table_query.fetch1(
                "analysis_file_name",
                "head_position_object_id",
                "head_orientation_object_id",
                "head_velocity_object_id",
            )
        else:
            (
                analysis_file_name,
                position_object_id,
                orientation_object_id,
                velocity_object_id,
            ) = table_query.fetch1(
                "analysis_file_name",
                "position_object_id",
                "orientation_object_id",
                "velocity_object_id",
            )
        part_table.insert1(
            {
                **key,
                "analysis_file_name": analysis_file_name,
                "position_object_id": position_object_id,
                "orientation_object_id": orientation_object_id,
                "velocity_object_id": velocity_object_id,
                **params,
            },
        )

    def fetch1_dataframe(self):
        source = self.fetch1("source")
        part_table = getattr(self, f"{source}Pos") & self
        nwb_data = part_table.fetch_nwb()[0]

        index = pd.Index(
            np.asarray(nwb_data["position"].get_spatial_series().timestamps),
            name="time",
        )
        if "video_frame_ind" in nwb_data["velocity"].fields["time_series"].keys():
            COLUMNS = [
                "video_frame_ind",
                "position_x",
                "position_y",
                "orientation",
                "velocity_x",
                "velocity_y",
                "speed",
            ]
            return pd.DataFrame(
                np.concatenate(
                    (
                        np.asarray(
                            nwb_data["velocity"].get_timeseries("video_frame_ind").data,
                            dtype=int,
                        )[:, np.newaxis],
                        np.asarray(nwb_data["position"].get_spatial_series().data),
                        np.asarray(nwb_data["orientation"].get_spatial_series().data)[
                            :, np.newaxis
                        ],
                        np.asarray(
                            nwb_data["velocity"].get_timeseries("velocity").data
                        ),
                    ),
                    axis=1,
                ),
                columns=COLUMNS,
                index=index,
            )
        else:
            COLUMNS = [
                "position_x",
                "position_y",
                "orientation",
                "velocity_x",
                "velocity_y",
                "speed",
            ]
            return pd.DataFrame(
                np.concatenate(
                    (
                        np.asarray(nwb_data["position"].get_spatial_series().data),
                        np.asarray(nwb_data["orientation"].get_spatial_series().data)[
                            :, np.newaxis
                        ],
                        np.asarray(nwb_data["velocity"].get_timeseries().data),
                    ),
                    axis=1,
                ),
                columns=COLUMNS,
                index=index,
            )


@schema
class PositionVideoSelection(dj.Manual):
    definition = """
    nwb_file_name       : varchar(255)                 # name of the NWB file
    interval_list_name  : varchar(200)                 # descriptive name of this interval list
    trodes_position_id  : int
    dlc_position_id     : int
    plot                : enum("DLC", "Trodes", "All") # Which position info to overlay on video file
    output_dir          : varchar(255)                 # directory where to save output video
    ---
    """


@schema
class PositionVideo(dj.Computed):
    """Creates a video of the computed head position and orientation as well as
    the original LED positions overlayed on the video of the animal.

    Use for debugging the effect of position extraction parameters."""

    definition = """
    -> PositionVideoSelection
    ---
    """

    def make(self, key):
        M_TO_CM = 100

        print("Loading position data...")
        raw_position_df = (
            RawPosition()
            & {
                "nwb_file_name": key["nwb_file_name"],
                "interval_list_name": key["interval_list_name"],
            }
        ).fetch1_dataframe()
        if key["plot"] == "DLC":
            pos_df = (
                FinalPosition()
                & {
                    "nwb_file_name": key["nwb_file_name"],
                    "interval_list_name": key["interval_list_name"],
                    "source": "DLC",
                    "position_id": key["dlc_position_id"],
                }
            ).fetch1_dataframe()
        elif key["plot"] == "Trodes":
            pos_df = (
                FinalPosition()
                & {
                    "nwb_file_name": key["nwb_file_name"],
                    "interval_list_name": key["interval_list_name"],
                    "source": "Trodes",
                    "position_id": key["trodes_position_id"],
                }
            ).fetch1_dataframe()
        elif key["plot"] == "All":
            dlc_df = (
                (
                    FinalPosition()
                    & {
                        "nwb_file_name": key["nwb_file_name"],
                        "interval_list_name": key["interval_list_name"],
                        "source": "DLC",
                        "position_id": key["dlc_position_id"],
                    }
                )
                .fetch1_dataframe()
                .drop(columns=["velocity_x", "velocity_y", "speed"])
            )
            trodes_df = (
                (
                    FinalPosition()
                    & {
                        "nwb_file_name": key["nwb_file_name"],
                        "interval_list_name": key["interval_list_name"],
                        "source": "Trodes",
                        "position_id": key["trodes_position_id"],
                    }
                )
                .fetch1_dataframe()
                .drop(columns=["velocity_x", "velocity_y", "speed"])
            )
            pos_df = dlc_df.merge(
                trodes_df,
                left_index=True,
                right_index=True,
                suffixes=["_DLC", "_Trodes"],
            )
        print("Loading video data...")
        epoch = (
            int(
                key["interval_list_name"]
                .replace("pos ", "")
                .replace(" valid times", "")
            )
            + 1
        )

        video_path, video_filename, meters_per_pixel, video_time = get_video_path(
            {"nwb_file_name": key["nwb_file_name"], "epoch": epoch}
        )
        video_dir = os.path.dirname(video_path) + "/"
        video_frame_col_name = [
            col for col in pos_df.columns if "video_frame_ind" in col
        ]
        video_frame_inds = pos_df[video_frame_col_name[0]].astype(int).to_numpy()
        if key["plot"] in ["DLC", "All"]:
            dlc_model_name = (FinalPosition.DLCPos & key).fetch1("dlc_model_name")
            video_path = (
                DLCPoseEstimationSelection & {"dlc_model_name": dlc_model_name, **key}
            ).fetch1("video_path")
        else:
            video_path = check_videofile(video_dir, key["output_dir"], video_filename)[
                0
            ]

        nwb_base_filename = key["nwb_file_name"].replace(".nwb", "")
        output_video_filename = Path(
            f"{Path(key['output_dir']).as_posix()}/{nwb_base_filename}_{epoch:02d}_"
            f"{key['plot']}_pos_overlay.mp4"
        ).as_posix()

        # centroids = {'red': np.asarray(raw_position_df[['xloc', 'yloc']]),
        #              'green':  np.asarray(raw_position_df[['xloc2', 'yloc2']])}
        position_mean_dict = {}
        orientation_mean_dict = {}
        if key["plot"] in ["DLC", "Trodes"]:
            position_mean_dict[key["plot"]] = np.asarray(
                pos_df[["position_x", "position_y"]]
            )
            orientation_mean_dict[key["plot"]] = np.asarray(pos_df[["orientation"]])
        elif key["plot"] == "All":
            position_mean_dict["DLC"] = np.asarray(
                pos_df[["position_x_DLC", "position_y_DLC"]]
            )
            orientation_mean_dict["DLC"] = np.asarray(pos_df[["orientation_DLC"]])
            position_mean_dict["Trodes"] = np.asarray(
                pos_df[["position_x_Trodes", "position_y_Trodes"]]
            )
            orientation_mean_dict["Trodes"] = np.asarray(pos_df[["orientation_Trodes"]])
        position_time = np.asarray(pos_df.index)
        cm_per_pixel = meters_per_pixel * M_TO_CM
        print("Making video...")
        self.make_video(
            video_path,
            video_frame_inds,
            position_mean_dict,
            orientation_mean_dict,
            video_time,
            position_time,
            output_video_filename=output_video_filename,
            cm_to_pixels=cm_per_pixel,
            disable_progressbar=False,
        )
        self.insert1(key)

    @staticmethod
    def convert_to_pixels(data, frame_size, cm_to_pixels=1.0):
        """Converts from cm to pixels and flips the y-axis.
        Parameters
        ----------
        data : ndarray, shape (n_time, 2)
        frame_size : array_like, shape (2,)
        cm_to_pixels : float
        Returns
        -------
        converted_data : ndarray, shape (n_time, 2)
        """
        return data / cm_to_pixels

    @staticmethod
    def fill_nan(variable, video_time, variable_time):
        video_ind = np.digitize(variable_time, video_time[1:])

        n_video_time = len(video_time)
        try:
            n_variable_dims = variable.shape[1]
            filled_variable = np.full((n_video_time, n_variable_dims), np.nan)
        except IndexError:
            filled_variable = np.full((n_video_time,), np.nan)
        filled_variable[video_ind] = variable

        return filled_variable

    def make_video(
        self,
        video_filename,
        video_frame_inds,
        position_mean_dict,
        orientation_mean_dict,
        video_time,
        position_time,
        output_video_filename="output.mp4",
        cm_to_pixels=1.0,
        disable_progressbar=False,
        arrow_radius=20,
        circle_radius=6,
    ):
        import cv2

        RGB_PINK = (234, 82, 111)
        RGB_YELLOW = (253, 231, 76)
        RGB_WHITE = (255, 255, 255)
        RGB_BLUE = (30, 144, 255)
        RGB_ORANGE = (255, 127, 80)
        #     "#29ff3e",
        #     "#ff0073",
        #     "#ff291a",
        #     "#1e2cff",
        #     "#b045f3",
        #     "#ffe91a",
        # ]
        video = cv2.VideoCapture(video_filename)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        frame_size = (int(video.get(3)), int(video.get(4)))
        frame_rate = video.get(5)
        n_frames = len(video_frame_inds)

        out = cv2.VideoWriter(
            output_video_filename, fourcc, frame_rate, frame_size, True
        )
        print(f"video_output: {output_video_filename}")

        # centroids = {
        #     color: self.fill_nan(data, video_time, position_time)
        #     for color, data in centroids.items()
        # }
        position_mean_dict = {
            key: self.fill_nan(position_mean_dict[key], video_time, position_time)
            for key in position_mean_dict.keys()
        }
        orientation_mean_dict = {
            key: self.fill_nan(orientation_mean_dict[key], video_time, position_time)
            for key in orientation_mean_dict.keys()
        }
        for time_ind in range(video_frame_inds[0]):
            is_grabbed, frame = video.read()
            if is_grabbed:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame)
            else:
                break

        for time_ind in tqdm(
            video_frame_inds, desc="frames", disable=disable_progressbar
        ):
            is_grabbed, frame = video.read()
            if is_grabbed:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # red_centroid = centroids["red"][time_ind]
                # green_centroid = centroids["green"][time_ind]
                for key in position_mean_dict.keys():
                    position = position_mean_dict[key][time_ind]
                    position = self.convert_to_pixels(
                        position, frame_size, cm_to_pixels
                    )
                    orientation = orientation_mean_dict[key][time_ind]
                    if key == "DLC":
                        color = RGB_BLUE
                    if key == "Trodes":
                        color = RGB_ORANGE
                    if np.all(~np.isnan(position)) & np.all(~np.isnan(orientation)):
                        arrow_tip = (
                            int(position[0] + arrow_radius * np.cos(orientation)),
                            int(position[1] + arrow_radius * np.sin(orientation)),
                        )
                        cv2.arrowedLine(
                            img=frame,
                            pt1=tuple(position.astype(int)),
                            pt2=arrow_tip,
                            color=color,
                            thickness=4,
                            line_type=8,
                            shift=cv2.CV_8U,
                            tipLength=0.25,
                        )

                    if np.all(~np.isnan(position)):
                        cv2.circle(
                            img=frame,
                            center=tuple(position.astype(int)),
                            radius=circle_radius,
                            color=color,
                            thickness=-1,
                            shift=cv2.CV_8U,
                        )
                # if np.all(~np.isnan(red_centroid)):
                #     cv2.circle(
                #         img=frame,
                #         center=tuple(red_centroid.astype(int)),
                #         radius=circle_radius,
                #         color=RGB_YELLOW,
                #         thickness=-1,
                #         shift=cv2.CV_8U,
                #     )

                # if np.all(~np.isnan(green_centroid)):
                #     cv2.circle(
                #         img=frame,
                #         center=tuple(green_centroid.astype(int)),
                #         radius=circle_radius,
                #         color=RGB_PINK,
                #         thickness=-1,
                #         shift=cv2.CV_8U,
                #     )

                # if np.all(~np.isnan(head_position)) & np.all(
                #     ~np.isnan(head_orientation)
                # ):
                #     arrow_tip = (
                #         int(head_position[0] + arrow_radius * np.cos(head_orientation)),
                #         int(head_position[1] + arrow_radius * np.sin(head_orientation)),
                #     )
                #     cv2.arrowedLine(
                #         img=frame,
                #         pt1=tuple(head_position.astype(int)),
                #         pt2=arrow_tip,
                #         color=RGB_WHITE,
                #         thickness=4,
                #         line_type=8,
                #         shift=cv2.CV_8U,
                #         tipLength=0.25,
                #     )

                # if np.all(~np.isnan(head_position)):
                #     cv2.circle(
                #         img=frame,
                #         center=tuple(head_position.astype(int)),
                #         radius=circle_radius,
                #         color=RGB_WHITE,
                #         thickness=-1,
                #         shift=cv2.CV_8U,
                #     )

                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame)
            else:
                break

        video.release()
        out.release()
        cv2.destroyAllWindows()

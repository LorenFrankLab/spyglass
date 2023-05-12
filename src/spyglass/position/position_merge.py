import os
from pathlib import Path
from typing import Dict

import datajoint as dj
import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm

from ..common.common_interval import IntervalList
from ..common.common_nwbfile import AnalysisNwbfile
from ..common.common_position import IntervalPositionInfo as CommonIntervalPositionInfo
from ..utils.dj_helper_fn import fetch_nwb
from .v1.dlc_utils import check_videofile, get_video_path, make_video
from .v1.position_dlc_pose_estimation import DLCPoseEstimationSelection
from .v1.position_dlc_selection import DLCPosV1
from .v1.position_trodes_position import TrodesPosV1

schema = dj.schema("position_merge")

_valid_data_sources = ["DLC", "Trodes", "Common"]


@schema
class PositionOutput(dj.Manual):
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
    version: int
    position_id: int
    ---
    """

    class DLCPosV1(dj.Part):
        """
        Table to pass-through upstream DLC Pose Estimation information
        """

        definition = """
        -> PositionOutput
        -> DLCPosV1
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

    class TrodesPosV1(dj.Part):
        """
        Table to pass-through upstream Trodes Position Tracking information
        """

        definition = """
        -> PositionOutput
        -> TrodesPosV1
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
        -> PositionOutput
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
        if source in ["Common"]:
            table_name = f"{source}Pos"
        else:
            version = key["version"]
            table_name = f"{source}PosV{version}"
        part_table = getattr(self, table_name)
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
        if source in ["Common"]:
            table_name = f"{source}Pos"
        else:
            version = self.fetch1("version")
            table_name = f"{source}PosV{version}"
        part_table = getattr(self, table_name) & self
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
    nwb_file_name           : varchar(255)                 # name of the NWB file
    interval_list_name      : varchar(200)                 # descriptive name of this interval list
    plot_id                 : int
    plot                    : varchar(40) # Which position info to overlay on video file
    ---
    position_ids            : mediumblob
    output_dir              : varchar(255)                 # directory where to save output video
    """

    def insert1(self, key, **kwargs):
        key["plot_id"] = self.get_plotid(key)
        super().insert1(key, **kwargs)

    def get_plotid(self, key):
        fields = list(self.primary_key)
        temp_key = {k: val for k, val in key.items() if k in fields}
        plot_id = temp_key.get("plot_id", None)
        if plot_id is None:
            plot_id = (
                dj.U().aggr(self & temp_key, n="max(plot_id)").fetch1("n") or 0
            ) + 1
        else:
            id = (self & temp_key).fetch("plot_id")
            if len(id) > 0:
                plot_id = max(id) + 1
            else:
                plot_id = max(0, plot_id)
        return plot_id


@schema
class PositionVideo(dj.Computed):
    """Creates a video of the computed head position and orientation as well as
    the original LED positions overlaid on the video of the animal.

    Use for debugging the effect of position extraction parameters."""

    definition = """
    -> PositionVideoSelection
    ---
    """

    def make(self, key):
        assert key["plot"] in ["DLC", "Trodes", "Common", "All"]
        M_TO_CM = 100
        output_dir, position_ids = (PositionVideoSelection & key).fetch1(
            "output_dir", "position_ids"
        )

        print("Loading position data...")
        # raw_position_df = (
        #     RawPosition()
        #     & {
        #         "nwb_file_name": key["nwb_file_name"],
        #         "interval_list_name": key["interval_list_name"],
        #     }
        # ).fetch1_dataframe()
        if key["plot"] == "DLC":
            assert position_ids["dlc_position_id"]
            pos_df = (
                PositionOutput()
                & {
                    "nwb_file_name": key["nwb_file_name"],
                    "interval_list_name": key["interval_list_name"],
                    "source": "DLC",
                    "position_id": position_ids["dlc_position_id"],
                }
            ).fetch1_dataframe()
        elif key["plot"] == "Trodes":
            assert position_ids["trodes_position_id"]
            pos_df = (
                PositionOutput()
                & {
                    "nwb_file_name": key["nwb_file_name"],
                    "interval_list_name": key["interval_list_name"],
                    "source": "Trodes",
                    "position_id": position_ids["trodes_position_id"],
                }
            ).fetch1_dataframe()
        elif key["plot"] == "All":
            assert position_ids["trodes_position_id"]
            assert position_ids["dlc_position_id"]
            dlc_df = (
                (
                    PositionOutput()
                    & {
                        "nwb_file_name": key["nwb_file_name"],
                        "interval_list_name": key["interval_list_name"],
                        "source": "DLC",
                        "position_id": position_ids["dlc_position_id"],
                    }
                )
                .fetch1_dataframe()
                .drop(columns=["velocity_x", "velocity_y", "speed"])
            )
            trodes_df = (
                (
                    PositionOutput()
                    & {
                        "nwb_file_name": key["nwb_file_name"],
                        "interval_list_name": key["interval_list_name"],
                        "source": "Trodes",
                        "position_id": position_ids["trodes_position_id"],
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
            temp_key = (PositionOutput.DLCPosV1 & key).fetch1("KEY")
            video_path = (DLCPoseEstimationSelection & temp_key).fetch1("video_path")
        else:
            video_path = check_videofile(video_dir, key["output_dir"], video_filename)[
                0
            ]

        nwb_base_filename = key["nwb_file_name"].replace(".nwb", "")
        output_video_filename = Path(
            f"{Path(output_dir).as_posix()}/{nwb_base_filename}{epoch:02d}_"
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

        make_video(
            video_path,
            video_frame_inds,
            position_mean_dict,
            orientation_mean_dict,
            video_time,
            position_time,
            processor="opencv",
            output_video_filename=output_video_filename,
            cm_to_pixels=cm_per_pixel,
            disable_progressbar=False,
        )
        self.insert1(key)

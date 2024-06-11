from pathlib import Path
from uuid import uuid4

import datajoint as dj
import numpy as np
import pandas as pd

from spyglass.position.position_merge import PoseOutput
from spyglass.utils import SpyglassMixin, SpyglassMixinPart

schema = dj.schema("behavior_core_v1")


@schema
class PoseGroup(SpyglassMixin, dj.Manual):
    definition = """
    pose_group_name: varchar(80)
    ----
    bodyparts = NULL: longblob # list of body parts to include in the pose
    """

    class Pose(SpyglassMixinPart):
        definition = """
        -> PoseGroup
        -> PoseOutput.proj(pose_merge_id='merge_id')
        """

    def create_group(
        self,
        group_name: str,
        merge_ids: list[str],
        bodyparts: list[str] = None,
    ):
        """create a group of pose information

        Parameters
        ----------
        group_name : str
            name of the group
        keys : list[dict]
            list of keys from PoseOutput to include in the group
        bodyparts : list[str], optional
            body parts to include in the group, by default None includes all from every set
        """
        group_key = {
            "pose_group_name": group_name,
        }
        self.insert1(
            {
                **group_key,
                "bodyparts": bodyparts,
            },
            skip_duplicates=True,
        )
        for merge_id in merge_ids:
            self.Pose.insert1(
                {
                    **group_key,
                    "pose_merge_id": merge_id,
                },
                skip_duplicates=True,
            )

    def fetch_pose_datasets(
        self, key: dict = None, format_for_moseq: bool = False
    ):
        """fetch pose information for a group of videos

        Parameters
        ----------
        key : dict
            group key
        format_for_moseq : bool, optional
            format for MoSeq, by default False

        Returns
        -------
        dict
            dictionary of video name to pose dataset
        """
        if key is None:
            key = {}

        bodyparts = (self & key).fetch1("bodyparts")
        datasets = {}
        for merge_key in (self.Pose & key).proj(merge_id="pose_merge_id"):
            video_name = (
                Path((PoseOutput & merge_key).fetch_video_name()).stem + ".mp4"
            )
            bodyparts_df = (PoseOutput & merge_key).fetch_dataframe()
            if bodyparts is None:
                bodyparts = (
                    bodyparts_df.keys().get_level_values(0).unique().values
                )
            bodyparts_df = bodyparts_df[bodyparts]
            datasets[video_name] = bodyparts_df
        if format_for_moseq:
            datasets = format_dataset_for_moseq(datasets, bodyparts)
        return datasets

    def fetch_video_paths(self, key: dict = None):
        """fetch video paths for a group of videos

        Parameters
        ----------
        key : dict
            group key

        Returns
        -------
        list[Path]
            list of video paths
        """
        if key is None:
            key = {}
        video_paths = [
            Path((PoseOutput & merge_key).fetch_video_name())
            for merge_key in (self.Pose & key).proj(merge_id="pose_merge_id")
        ]
        return video_paths


def format_dataset_for_moseq(
    datasets: dict[str, pd.DataFrame],
    bodyparts: list[str],
    coordinate_axes: list[str] = ["x", "y"],
):
    """format pose datasets for MoSeq

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        dictionary of video name to pose dataset
    bodyparts : list[str]
        list of body parts to include in the pose

    Returns
    -------
    tuple[dict[str, np.ndarray], dict[str, np.ndarray]
        coordinates and confidences for each video
    """
    num_keypoints = len(bodyparts)
    num_dimensions = len(coordinate_axes)
    coordinates = {}
    confidences = {}

    for video_name, bodyparts_df in datasets.items():
        coordinates_i = None
        confidences_i = None
        for i, bodypart in enumerate(bodyparts):
            part_df = bodyparts_df[bodypart]
            print(len(part_df))
            if coordinates_i is None:
                num_frames = len(part_df)
                coordinates_i = np.empty(
                    (num_frames, num_keypoints, num_dimensions)
                )
                confidences_i = np.empty((num_frames, num_keypoints))
            coordinates_i[:, i, :] = part_df[coordinate_axes].values
            confidences_i[:, i] = part_df["likelihood"].values
        coordinates[video_name] = coordinates_i
        confidences[video_name] = confidences_i
    return coordinates, confidences
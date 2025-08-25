import warnings
from pathlib import Path
from typing import List

import datajoint as dj
import numpy as np
import pandas as pd

from spyglass.position.position_merge import PositionOutput
from spyglass.utils import SpyglassMixin, SpyglassMixinPart

schema = dj.schema("behavior_v1_core")


@schema
class PoseGroup(SpyglassMixin, dj.Manual):
    """Groups one or more entries of keypoint pose information"""

    definition = """
    pose_group_name: varchar(80)
    ----
    bodyparts = NULL: longblob # list of body parts to include in the pose
    """

    class Pose(SpyglassMixinPart):
        definition = """
        -> PoseGroup
        -> PositionOutput.proj(pose_merge_id='merge_id')
        """

    def create_group(
        self,
        group_name: str,
        merge_ids: List[str],
        bodyparts: List[str] = None,
    ) -> None:
        """Create a group of pose information

        Parameters
        ----------
        group_name : str
            Name of the group
        merge_ids : List[str]
            list of keys from PositionOutput to include in the group
        bodyparts : List[str], optional
            body parts to include in the group, by default None includes all from every set
        """
        group_key = {"pose_group_name": group_name}
        if self & group_key:
            warnings.warn(
                f"Pose group {group_name} already exists. Skipping insertion."
            )
            return
        self.insert1(
            {
                **group_key,
                "bodyparts": bodyparts,
            }
        )
        for merge_id in merge_ids:
            self.Pose.insert1(
                {
                    **group_key,
                    "pose_merge_id": merge_id,
                }
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
        self.ensure_single_entry(key)
        query = self & key
        bodyparts = query.fetch1("bodyparts")
        datasets = {}
        for merge_key in (self.Pose & query).proj(merge_id="pose_merge_id"):
            video_name = Path(
                (PositionOutput & merge_key).fetch_video_path()
            ).name
            bodyparts_df = (PositionOutput & merge_key).fetch_pose_dataframe()
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
        List[Path]
            list of video paths
        """
        self.ensure_single_entry(key)
        key = (self & key).fetch1("KEY")
        return [
            Path((PositionOutput & merge_key).fetch_video_path())
            for merge_key in (self.Pose & key).proj(merge_id="pose_merge_id")
        ]


def format_dataset_for_moseq(
    datasets: dict[str, pd.DataFrame],
    bodyparts: List[str],
    coordinate_axes: List[str] = ["x", "y"],
):
    """format pose datasets for MoSeq

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        dictionary of video name to pose dataset
    bodyparts : List[str]
        list of body parts to include in the pose
    coordinate_axes : List[str], optional
        list of coordinate axes to include, by default ["x", "y"]

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
        num_frames = len(bodyparts_df[bodyparts[0]])
        coordinates_i = np.empty((num_frames, num_keypoints, num_dimensions))
        confidences_i = np.empty((num_frames, num_keypoints))

        for i, bodypart in enumerate(bodyparts):
            part_df = bodyparts_df[bodypart]
            coordinates_i[:, i, :] = part_df[coordinate_axes].values
            confidences_i[:, i] = part_df["likelihood"].values
        coordinates[video_name] = coordinates_i
        confidences[video_name] = confidences_i
    return coordinates, confidences


def results_to_df(results):
    for key in results:
        column_names, data = [], []

        if "syllable" in results[key].keys():
            column_names.append(["syllable"])
            data.append(results[key]["syllable"].reshape(-1, 1))

        if "centroid" in results[key].keys():
            d = results[key]["centroid"].shape[1]
            column_names.append(["centroid x", "centroid y", "centroid z"][:d])
            data.append(results[key]["centroid"])

        if "heading" in results[key].keys():
            column_names.append(["heading"])
            data.append(results[key]["heading"].reshape(-1, 1))

        if "latent_state" in results[key].keys():
            latent_dim = results[key]["latent_state"].shape[1]
            column_names.append(
                [f"latent_state_{i}" for i in range(latent_dim)]
            )
            data.append(results[key]["latent_state"])

        dfs = [
            pd.DataFrame(arr, columns=cols)
            for arr, cols in zip(data, column_names)
        ]
        df = pd.concat(dfs, axis=1)

        for col in df.select_dtypes(include=[np.floating]).columns:
            df[col] = df[col].astype(float).round(4)

        return df

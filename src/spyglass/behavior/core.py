from pathlib import Path

import datajoint as dj

from spyglass.position.position_merge import PoseOutput
from spyglass.utils import SpyglassMixin, SpyglassMixinPart

schema = dj.schema("behavior_core_v1")


@schema
class PoseGroup(SpyglassMixin, dj.Manual):
    definition = """
    pose_group_name: varchar(80)
    ----
    body_parts = NULL: longblob # list of body parts to include in the pose
    """

    class Pose(SpyglassMixinPart):
        definition = """
        -> PoseGroup
        -> PoseOutput.proj(pose_merge_id='merge_id')
        """

    def create_group(
        self,
        group_name: str,
        keys: list[dict],
        pose_variables: list[str] = None,
    ):
        """create a group of pose information

        Parameters
        ----------
        group_name : str
            name of the group
        keys : list[dict]
            list of keys from PoseOutput to include in the group
        pose_variables : list[str], optional
            body parts to include in the group, by default None includes all from every set
        """
        group_key = {
            "pose_group_name": group_name,
        }
        self.insert1(
            {
                **group_key,
                "pose_variables": pose_variables,
            },
            skip_duplicates=True,
        )
        for key in keys:
            self.Pose.insert1(
                {
                    **key,
                    **group_key,
                },
                skip_duplicates=True,
            )

    def fetch_pose_datasets(self, key: dict):
        """fetch pose information for a group of videos

        Parameters
        ----------
        key : dict
            group key

        Returns
        -------
        dict
            dictionary of video name to pose dataset
        """
        datasets = {}
        for merge_key in self.Pose & key:
            video_name = Path((PoseOutput & merge_key).fetch_video_name()).stem
            datasets[video_name] = (
                PoseOutput & merge_key
            ).fetch_dataset()  # TODO: restrict returned dataframe to bodyparts of interest
        return datasets

import datajoint as dj
from datajoint.utils import to_camel_case
from pandas import DataFrame

from spyglass.common.common_position import IntervalPositionInfo as CommonPos
from spyglass.position.v1.imported_pose import ImportedPose
from spyglass.position.v1.position_dlc_pose_estimation import DLCPoseEstimation
from spyglass.position.v1.position_dlc_selection import DLCPosV1
from spyglass.position.v1.position_trodes_position import TrodesPosV1
from spyglass.utils import SpyglassMixin, _Merge

schema = dj.schema("position_merge")

source_class_dict = {
    "IntervalPositionInfo": CommonPos,
    "DLCPosV1": DLCPosV1,
    "TrodesPosV1": TrodesPosV1,
    "ImportedPose": ImportedPose,
    "DLCPoseEstimation": DLCPoseEstimation,
}


@schema
class PositionOutput(_Merge, SpyglassMixin):
    """
    Table to identify source of Position Information from upstream options
    (e.g. DLC, Trodes, etc...) To add another upstream option, a new Part table
    should be added in the same syntax as DLCPos and TrodesPos.
    """

    definition = """
    merge_id: uuid
    ---
    source: varchar(32)
    """

    class DLCPosV1(SpyglassMixin, dj.Part):
        """
        Table to pass-through upstream DLC Pose Estimation information
        """

        definition = """
        -> PositionOutput
        ---
        -> DLCPosV1
        """

    class TrodesPosV1(SpyglassMixin, dj.Part):
        """
        Table to pass-through upstream Trodes Position Tracking information
        """

        definition = """
        -> PositionOutput
        ---
        -> TrodesPosV1
        """

    class CommonPos(SpyglassMixin, dj.Part):
        """
        Table to pass-through upstream Trodes Position Tracking information
        """

        definition = """
        -> PositionOutput
        ---
        -> CommonPos
        """

    class ImportedPose(SpyglassMixin, dj.Part):
        """
        Table to pass-through upstream Pose information from NWB file
        """

        definition = """
        -> PositionOutput
        ---
        -> ImportedPose
        """

    def fetch1_dataframe(self) -> DataFrame:
        """Fetch a single dataframe from the merged table."""
        query = (
            source_class_dict[to_camel_case(self.get_parent_table().table_name)]
            & self.get_part_table()[0]
        )
        return query.fetch1_dataframe()

    def fetch_pose_dataframe(self):
        """Fetch a single dataframe of pose bodypart coordinates.

        This will work only for sources that have pose information.
        """
        query = (
            source_class_dict[to_camel_case(self.get_parent_table().table_name)]
            & self.get_part_table()[0]
        )
        return query.fetch_pose_dataframe()

    def fetch_video_path(self, key=dict()):
        """Fetch the video path associated with the position information."""
        restricted = self & key
        query = (
            source_class_dict[
                to_camel_case(restricted.get_parent_table().table_name)
            ]
            & restricted.get_part_table()[0]
        )
        return query.fetch_video_path()

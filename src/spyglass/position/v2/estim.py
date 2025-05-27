import datajoint as dj

from spyglass.common import AnalysisNwbfile
from spyglass.position.v2.train import Model
from spyglass.position.v2.video import VidFileGroup

schema = dj.schema("cbroz_position_v2_estim")


@schema
class PoseEstim(dj.Manual):
    definition = """
    -> Model
    -> VidFileGroup
    ---
    -> AnalysisNwbfile
    """

    def fetch1_dataframe(self):
        raise NotImplementedError


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

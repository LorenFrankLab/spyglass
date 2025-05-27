import datajoint as dj

from spyglass.common import VideoFile

schema = dj.schema("cbroz_position_v2_video")


# TODO: Common or Position schema?
# 1. VidFileGroup could be used across multiple pipelines
# 2. Calibration is specific to Position pipeline
# Separating would require a separate many-to-one Calibration table


@schema
class VidFileGroup(dj.Manual):
    definition = """
    vid_group_id: int
    ---
    description: varchar(255)
    """

    class File(dj.Part):
        definition = """
        -> master
        -> VideoFile
        """

    class Calibration(dj.Part):
        definition = """
        -> VidFileGroup
        calibration_id: int
        ---
        # What other fields are needed?
        path: varchar(255)
        """

    def insert1(self, key, tool="DLC", **kwargs):
        raise NotImplementedError("Define calibration tool at insert")

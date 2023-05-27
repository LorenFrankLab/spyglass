import datajoint as dj
from spikesorting.v1 import CuratedSpikeSortingV1  # noqa F401
from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.utils.dj_helper_fn import fetch_nwb

schema = dj.schema("spikesorting_merge")


class CuratedSpikeSortingOutput:
    definition = """
    spikesorting_id: uuid
    source: varchar(40)
    version: int
    ---
    """

    class CuratedSpikeSortingV1(dj.Part):
        """
        Table to pass-through upstream DLC Pose Estimation information
        """

        definition = """
        -> CuratedSpikeSortingOutput
        -> CuratedSpikeSortingV1
        ---
        -> AnalysisNwbfile
        units_object_id: varchar(40)
        """

        def fetch_nwb(self, *attrs, **kwargs):
            return fetch_nwb(
                self, (AnalysisNwbfile, "analysis_file_abs_path"), *attrs, **kwargs
            )

import datajoint as dj

from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.high_multiunit_detection.v1.high_multiunit_detection import (
    HighMultiunitTimesV1,
)  # noqa F401
from spyglass.utils.dj_helper_fn import fetch_nwb

schema = dj.schema("ripple_merge")


@schema
class HighMultiunitTimesOutput(dj.Manual):
    definition = """
    high_multiunit_times_id: uuid
    ---
    source: varchar(40)
    version: int
    """

    class HighMultiunitTimesV1(dj.Part):
        definition = """
        -> HighMultiunitTimesOutput
        -> HighMultiunitTimesV1
        ---
        -> AnalysisNwbfile
        high_multiunit_times_object_id : varchar(40)
        """

    def fetch_nwb(self, *attrs, **kwargs):
        return fetch_nwb(
            self, (AnalysisNwbfile, "analysis_file_abs_path"), *attrs, **kwargs
        )

    def fetch1_dataframe(self):
        return self.fetch_dataframe()[0]

    def fetch_dataframe(self):
        return [data["high_multiunit_times"] for data in self.fetch_nwb()]

import datajoint as dj

from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.ripple.v1.ripple import RippleTimesV1  # noqa F401
from spyglass.utils.dj_helper_fn import fetch_nwb

schema = dj.schema("ripple_merge")


@schema
class RippleTimesOutput(dj.Manual):
    definition = """
    ripple_id: uuid
    ---
    source: varchar(40)
    version: int

    """

    class RippleTimesV1(dj.Part):
        definition = """
        -> RippleTimesOutput
        -> RippleTimesV1
        ---
        -> AnalysisNwbfile
        ripple_times_object_id : varchar(40)
        """

        def fetch_nwb(self, *attrs, **kwargs):
            return fetch_nwb(
                self,
                (AnalysisNwbfile, "analysis_file_abs_path"),
                *attrs,
                **kwargs,
            )

        def fetch1_dataframe(self):
            """Convenience function for returning the marks in a readable format"""
            return self.fetch_dataframe()[0]

        def fetch_dataframe(self):
            return [data["ripple_times"] for data in self.fetch_nwb()]

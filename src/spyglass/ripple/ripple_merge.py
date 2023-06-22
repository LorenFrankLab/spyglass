import datajoint as dj

from spyglass.ripple.v1.ripple import RippleTimesV1  # noqa F401
from spyglass.utils.dj_merge_tables import Merge

schema = dj.schema("ripple_merge")


@schema
class RippleTimesOutput(dj.Manual):
    definition = """
    merge_id: uuid
    ---
    source: varchar(32)
    """

    class RippleTimesV1(dj.Part):
        definition = """
        -> master
        ---
        -> RippleTimesV1
        """

    def fetch1_dataframe(self):
        """Convenience function for returning the marks in a readable format"""
        return self.fetch_dataframe()[0]

    def fetch_dataframe(self):
        return [data["ripple_times"] for data in self.fetch_nwb()]

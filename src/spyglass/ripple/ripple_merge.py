import datajoint as dj

from spyglass.common.common_ripple import RippleTimes as CommonRipple
from .v1.ripple import RippleTimesV1  # noqa F401
from spyglass.utils.dj_merge_tables import _Merge

schema = dj.schema("ripple_merge")


@schema
class RippleTimesOutput(_Merge):
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

    class CommonRipple(dj.Part):
        """Table to pass-through legacy RippleTimes"""

        definition = """
        -> master
        ---
        -> CommonRipple
        """

    def fetch1_dataframe(self):
        """Convenience function for returning the marks in a readable format"""
        return self.fetch_dataframe()[0]

    def fetch_dataframe(self):
        # Note: `proj` below facilitates operator syntax eg Table & restrict
        return [data["ripple_times"] for data in self.fetch_nwb(self.proj())]

import datajoint as dj

from spyglass.linearization.v0.main import (  # noqa F401
    IntervalLinearizedPosition as LinearizedPositionV0,
)
from spyglass.linearization.v1.main import LinearizedPositionV1  # noqa F401
from spyglass.utils import SpyglassMixin, _Merge

schema = dj.schema("position_linearization_merge")


@schema
class LinearizedPositionOutput(_Merge, SpyglassMixin):
    definition = """
    merge_id: uuid
    ---
    source: varchar(32)
    """

    class LinearizedPositionV0(SpyglassMixin, dj.Part):  # noqa: F811
        definition = """
        -> LinearizedPositionOutput
        ---
        -> LinearizedPositionV0
        """

    class LinearizedPositionV1(SpyglassMixin, dj.Part):  # noqa: F811
        definition = """
        -> LinearizedPositionOutput
        ---
        -> LinearizedPositionV1
        """

    def fetch1_dataframe(self):
        """Fetch a single dataframe from the merged table."""
        _ = self.ensure_single_entry()
        return self.fetch_nwb(self.proj())[0]["linearized_position"].set_index(
            "time"
        )

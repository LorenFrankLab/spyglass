import datajoint as dj

from spyglass.position_linearization.v1.linearization import (  # noqa F401
    LinearizedPositionV1,
)
from spyglass.utils.dj_merge_tables import _Merge
from spyglass.utils.dj_mixin import SpyglassMixin

schema = dj.schema("position_linearization_merge")


@schema
class LinearizedPositionOutput(_Merge, SpyglassMixin):
    definition = """
    merge_id: uuid
    ---
    source: varchar(32)
    """

    class LinearizedPositionV1(SpyglassMixin, dj.Part):  # noqa: F811
        definition = """
        -> LinearizedPositionOutput
        ---
        -> LinearizedPositionV1
        """

    def fetch1_dataframe(self):
        return self.fetch_nwb(self.proj())[0]["linearized_position"].set_index(
            "time"
        )

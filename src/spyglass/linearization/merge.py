import datajoint as dj

from spyglass.linearization.v0.main import LinearizedV0  # noqa F401
from spyglass.linearization.v1.main import LinearizedV1  # noqa F401

from ..utils.dj_merge_tables import _Merge

schema = dj.schema("linearization_merge")


@schema
class LinearizedOutput(_Merge):
    definition = """
    merge_id: uuid
    ---
    source: varchar(32)
    """

    class LinearizedV0(dj.Part):  # noqa F811
        definition = """
        -> master
        ---
        -> LinearizedV0
        """

    class LinearizedV1(dj.Part):  # noqa F811
        definition = """
        -> master
        ---
        -> LinearizedV1
        """

    def fetch1_dataframe(self):
        return self.fetch_nwb(self.proj())[0]["linearized_position"].set_index(
            "time"
        )

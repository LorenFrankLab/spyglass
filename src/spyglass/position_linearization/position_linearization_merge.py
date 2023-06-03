import datajoint as dj

from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.position_linearization.v1.linearization import (
    LinearizedPositionV1,
)  # noqa F401
from spyglass.utils.dj_helper_fn import fetch_nwb

schema = dj.schema("position_linearization_merge")


@schema
class LinearizedPositionOutput(dj.Manual):
    definition = """
    linearized_position_id: uuid
    ---
    source: varchar(40)
    version: int

    """

    class LinearizedPositionV1(dj.Part):
        definition = """
        -> LinearizedPositionOutput
        -> LinearizedPositionV1
        ---
        -> AnalysisNwbfile
        linearized_position_object_id: varchar(40)
        """

        def fetch_nwb(self, *attrs, **kwargs):
            return fetch_nwb(
                self,
                (AnalysisNwbfile, "analysis_file_abs_path"),
                *attrs,
                **kwargs,
            )

        def fetch1_dataframe(self):
            return self.fetch_nwb()[0]["linearized_position"].set_index("time")

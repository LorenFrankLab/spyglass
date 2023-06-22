import datajoint as dj
import pandas as pd

from spyglass.common.common_ephys import LFP as CommonLFP  # noqa: F401
from spyglass.common.common_filter import FirFilterParameters  # noqa: F401
from spyglass.common.common_interval import IntervalList  # noqa: F401
from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.lfp.v1.lfp import LFPV1, ImportedLFPV1  # noqa: F401
from spyglass.utils.dj_helper_fn import fetch_nwb
from spyglass.utils.dj_merge_tables import Merge

schema = dj.schema("lfp_merge")


@schema
class LFPOutput(Merge):
    definition = """
    merge_id: uuid
    ---
    source: varchar(32)
    """

    class LFPV1(dj.Part):
        definition = """
        -> master
        ---
        -> LFPV1
        """

    class ImportedLFPV1(dj.Part):
        definition = """
        -> master
        ---
        -> ImportedLFPV1
        """

    class CommonLFP(dj.Part):
        """Table to pass-through legacy LFP"""

        definition = """
        -> master
        ---
        -> CommonLFP
        """

    def fetch_nwb(self, *attrs, **kwargs):
        part_parents = self._merge_restrict_parents(
            restriction=self.restriction, return_empties=False
        )

        if len(part_parents) == 1:
            return fetch_nwb(
                part_parents[0],
                (AnalysisNwbfile, "analysis_file_abs_path"),
                *attrs,
                **kwargs,
            )
        else:
            raise ValueError(
                f"{len(part_parents)} possible sources found in Merge Table"
                + part_parents
            )

    def fetch1_dataframe(self, *attrs, **kwargs):
        nwb_lfp = self.fetch_nwb()[0]
        return pd.DataFrame(
            nwb_lfp["lfp"].data,
            index=pd.Index(nwb_lfp["lfp"].timestamps, name="time"),
        )

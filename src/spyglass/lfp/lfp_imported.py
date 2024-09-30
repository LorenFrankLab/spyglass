import datajoint as dj

from spyglass.common.common_interval import IntervalList  # noqa: F401
from spyglass.common.common_nwbfile import AnalysisNwbfile  # noqa: F401
from spyglass.common.common_session import Session  # noqa: F401
from spyglass.lfp.lfp_electrode import LFPElectrodeGroup  # noqa: F401
from spyglass.utils.dj_mixin import SpyglassMixin

schema = dj.schema("lfp_imported")


@schema
class ImportedLFP(SpyglassMixin, dj.Imported):
    definition = """
    -> Session                      # the session to which this LFP belongs
    -> LFPElectrodeGroup            # the group of electrodes to be filtered
    -> IntervalList                 # the original set of times to be filtered
    lfp_object_id: varchar(40)      # object ID for loading from the NWB file
    ---
    lfp_sampling_rate: float        # the sampling rate, in samples/sec
    -> AnalysisNwbfile
    """

    def make(self, key):
        """Placeholder for importing LFP."""
        raise NotImplementedError(
            "For `insert`, use `allow_direct_insert=True`"
        )

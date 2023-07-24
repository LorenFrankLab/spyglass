import datajoint as dj

from spyglass.common.common_interval import IntervalList
from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.common.common_session import Session
from spyglass.lfp.lfp_electrode import LFPElectrodeGroup

schema = dj.schema("lfp_imported")


@schema
class ImportedLFP(dj.Imported):
    definition = """
    -> Session                      # the session to which this LFP belongs
    -> LFPElectrodeGroup            # the group of electrodes to be filtered
    -> IntervalList # the original set of times to be filtered
    lfp_object_id: varchar(40)      # the NWB object ID for loading this object from the file
    ---
    lfp_sampling_rate: float        # the sampling rate, in samples/sec
    -> AnalysisNwbfile
    """

    def make(self, key):
        raise NotImplementedError(
            "For `insert`, use `allow_direct_insert=True`"
        )

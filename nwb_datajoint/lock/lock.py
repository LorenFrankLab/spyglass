import datajoint as dj

from ..common.common_nwbfile import Nwbfile, AnalysisNwbfile

schema = dj.schema('lock')

@schema
class NwbfileLock(dj.Manual):
    definition = """
    -> Nwbfile
    """

@schema
class AnalysisNwbfileLock(dj.Manual):
    definition = """
    -> AnalysisNwbfile
    """
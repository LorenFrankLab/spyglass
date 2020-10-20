import datajoint as dj
import os

from ..common import Nwbfile, AnalysisNwbfile

schema = dj.schema('lock')

@schema
class NwbfileLock(dj.Manual):
    definition = """
    -> Nwbfile
    """
    def populate_from_lock_file(self):
        """Reads from the NWB_LOCK_FILE (defined by an environment variable), adds the entries to this schema, and then removes the file
        """
        try:
            lock_file = open(os.getenv('NWB_LOCK_FILE'), 'r')
            for line in lock_file:
                self.insert1(line.strip())
        except:
            pass

@schema
class AnalysisNwbfileLock(dj.Manual):
    definition = """
    -> AnalysisNwbfile
    """
    def populate_from_lock_file(self):
        """Reads from the ANALYSIS_LOCK_FILE (defined by an environment variable), adds the entries to this schema, and then removes the file
        """
        try:
            lock_file = open(os.getenv('ANALYSIS_LOCK_FILE'), 'r')
            for line in lock_file:
                self.insert1(line.strip())
        except:
            pass
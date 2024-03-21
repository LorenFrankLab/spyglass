import os

import datajoint as dj

from spyglass.utils import SpyglassMixin, logger

schema = dj.schema("file_lock")

from spyglass.common.common_nwbfile import AnalysisNwbfile, Nwbfile


@schema
class NwbfileLock(SpyglassMixin, dj.Manual):
    definition = """
    -> Nwbfile
    """

    def populate_from_lock_file(self):
        """
        Reads from the NWB_LOCK_FILE (defined by an environment variable),
        adds the entries to this schema, and then removes the file
        """
        if os.path.exists(os.getenv("NWB_LOCK_FILE")):
            lock_file = open(os.getenv("NWB_LOCK_FILE"), "r")
            for line in lock_file:
                logger.info(line)
                key = {"nwb_file_name": line.strip()}
                self.insert1(key, skip_duplicates="True")
            lock_file.close()
            os.remove(os.getenv("NWB_LOCK_FILE"))


@schema
class AnalysisNwbfileLock(SpyglassMixin, dj.Manual):
    definition = """
    -> AnalysisNwbfile
    """

    def populate_from_lock_file(self):
        """Reads from the ANALYSIS_LOCK_FILE (defined by an environment variable), adds the entries to this schema, and
        then removes the file
        """
        if os.path.exists(os.getenv("ANALYSIS_LOCK_FILE")):
            lock_file = open(os.getenv("ANALYSIS_LOCK_FILE"), "r")
            for line in lock_file:
                key = {"analysis_file_name": line.strip()}
                self.insert1(key, skip_duplicates="True")
            os.remove(os.getenv("ANALYSIS_LOCK_FILE"))

import datajoint as dj
import numpy as np

schema = dj.schema("common_backup")


@schema
class SpikeSortingBackUp(dj.Manual):
    definition = """
    nwb_file_name: varchar(500)
    sort_group_id: int
    sort_interval_name: varchar(500)
    filter_parameter_set_name: varchar(500)
    sorter_name: varchar(500)
    spikesorter_parameter_set_name: varchar(500)
    ---
    sorting_id: varchar(500)
    analysis_file_name: varchar(1000)
    time_of_sort: int   # in Unix time, to the nearest second
    units_object_id: varchar(100)
    """

    def insert_from_backup(self, backup_file):
        """backup file lives in /common/backup_keys/

        Parameters
        ----------
        backup_file : str
            path to npy pickle file containing keys
        """
        backup_keys = np.load(backup_file, allow_pickle=True)
        self.insert(backup_keys, skip_duplicates=True)


@schema
class CuratedSpikeSortingBackUp(dj.Manual):
    definition = """
    nwb_file_name: varchar(500)
    sort_group_id: int
    sort_interval_name: varchar(500)
    filter_parameter_set_name: varchar(500)
    sorting_id: varchar(500)
    ---
    analysis_file_name: varchar(1000)
    units_object_id: varchar(100)
    """

    def insert_from_backup(self, backup_file):
        backup_keys = np.load(backup_file, allow_pickle=True)
        self.insert(backup_keys, skip_duplicates=True)

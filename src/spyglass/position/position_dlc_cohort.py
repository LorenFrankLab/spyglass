import datajoint as dj
from ..common.dj_helper_fn import fetch_nwb
from ..common.common_nwbfile import AnalysisNwbfile
from .position_dlc_pose_estimation import DLCPoseEstimation
from .position_dlc_position import DLCSmoothInterp

schema = dj.schema("position_dlc_cohort")


@schema
class DLCSmoothInterpCohortSelection(dj.Manual):
    """
    Table to specify which combination of bodyparts from DLCSmoothInterp
    get combined into a cohort
    """

    definition = """
    dlc_si_cohort_selection_name : varchar(120)
    -> DLCPoseEstimation
    ---
    bodyparts_params_dict   : blob      # Dict with bodypart as key and desired dlc_si_params_name as value
    """


@schema
class DLCSmoothInterpCohort(dj.Computed):
    """
    Table to combine multiple bodyparts from DLCSmoothInterp
    to enable centroid/orientation calculations
    """

    # Need to ensure that nwb_file_name/epoch/interval list name endure as primary keys
    definition = """
    -> DLCSmoothInterpCohortSelection
    ---
    """

    class BodyPart(dj.Part):
        definition = """
        -> DLCSmoothInterpCohort
        -> DLCSmoothInterp
        ---
        -> AnalysisNwbfile
        dlc_smooth_interp_object_id : varchar(80)
        """

        def fetch_nwb(self, *attrs, **kwargs):
            return fetch_nwb(
                self, (AnalysisNwbfile, "analysis_file_abs_path"), *attrs, **kwargs
            )

        def fetch1_dataframe(self):
            return self.fetch_nwb()[0]["dlc_smooth_interp"].set_index("time")

    def make(self, key):
        # from Jen Guidera
        self.insert1(key)
        cohort_selection = (DLCSmoothInterpCohortSelection & key).fetch1()
        table_entries = []
        bodyparts_params_dict = cohort_selection.pop("bodyparts_params_dict")
        temp_key = cohort_selection.copy()
        for bodypart, params in bodyparts_params_dict.items():
            temp_key["bodypart"] = bodypart
            temp_key["dlc_si_params_name"] = params
            table_entries.append((DLCSmoothInterp & temp_key).fetch())
        assert len(table_entries) == len(
            bodyparts_params_dict
        ), "more entries found in DLCSmoothInterp than specified in bodyparts_params_dict"
        table_column_names = list(table_entries[0].dtype.fields.keys())
        for table_entry in table_entries:
            entry_key = {
                **{k: v for k, v in zip(table_column_names, table_entry[0])},
                **key,
            }
            DLCSmoothInterpCohort.BodyPart.insert1(entry_key, skip_duplicates=True)

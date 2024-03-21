import datajoint as dj
import numpy as np
import pandas as pd

from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.position.v1.position_dlc_pose_estimation import (  # noqa: F401
    DLCPoseEstimation,
)
from spyglass.position.v1.position_dlc_position import DLCSmoothInterp
from spyglass.utils.dj_mixin import SpyglassMixin

schema = dj.schema("position_v1_dlc_cohort")


@schema
class DLCSmoothInterpCohortSelection(SpyglassMixin, dj.Manual):
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
class DLCSmoothInterpCohort(SpyglassMixin, dj.Computed):
    """
    Table to combine multiple bodyparts from DLCSmoothInterp
    to enable centroid/orientation calculations
    """

    # Need to ensure that nwb_file_name/epoch/interval list name endure as primary keys
    definition = """
    -> DLCSmoothInterpCohortSelection
    ---
    """

    class BodyPart(SpyglassMixin, dj.Part):
        definition = """
        -> DLCSmoothInterpCohort
        -> DLCSmoothInterp
        ---
        -> AnalysisNwbfile
        dlc_smooth_interp_position_object_id : varchar(80)
        dlc_smooth_interp_info_object_id : varchar(80)
        """

        def fetch1_dataframe(self):
            nwb_data = self.fetch_nwb()[0]
            index = pd.Index(
                np.asarray(
                    nwb_data["dlc_smooth_interp_position"]
                    .get_spatial_series()
                    .timestamps
                ),
                name="time",
            )
            COLUMNS = [
                "video_frame_ind",
                "x",
                "y",
            ]
            return pd.DataFrame(
                np.concatenate(
                    (
                        np.asarray(
                            nwb_data["dlc_smooth_interp_info"]
                            .time_series["video_frame_ind"]
                            .data,
                            dtype=int,
                        )[:, np.newaxis],
                        np.asarray(
                            nwb_data["dlc_smooth_interp_position"]
                            .get_spatial_series()
                            .data
                        ),
                    ),
                    axis=1,
                ),
                columns=COLUMNS,
                index=index,
            )

    def make(self, key):
        from .dlc_utils import OutputLogger, infer_output_dir

        output_dir = infer_output_dir(key=key, makedir=False)
        with OutputLogger(
            name=f"{key['nwb_file_name']}_{key['epoch']}_{key['dlc_model_name']}_log",
            path=f"{output_dir.as_posix()}/log.log",
            print_console=False,
        ) as logger:
            logger.logger.info("-----------------------")
            logger.logger.info("Bodypart Cohort")
            # from Jen Guidera
            self.insert1(key)
            cohort_selection = (DLCSmoothInterpCohortSelection & key).fetch1()
            table_entries = []
            bodyparts_params_dict = cohort_selection.pop(
                "bodyparts_params_dict"
            )
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
                    **{
                        k: v for k, v in zip(table_column_names, table_entry[0])
                    },
                    **key,
                }
                DLCSmoothInterpCohort.BodyPart.insert1(
                    entry_key, skip_duplicates=True
                )
        logger.logger.info("Inserted entry into DLCSmoothInterpCohort")

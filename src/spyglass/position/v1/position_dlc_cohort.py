from pathlib import Path

import datajoint as dj
import numpy as np
import pandas as pd

from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.position.v1.dlc_utils import file_log, infer_output_dir
from spyglass.position.v1.position_dlc_pose_estimation import (  # noqa: F401
    DLCPoseEstimation,
)
from spyglass.position.v1.position_dlc_position import DLCSmoothInterp
from spyglass.utils import SpyglassMixin, logger

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
    """
    log_path = None

    class BodyPart(SpyglassMixin, dj.Part):
        definition = """
        -> DLCSmoothInterpCohort
        -> DLCSmoothInterp
        ---
        -> AnalysisNwbfile
        dlc_smooth_interp_position_object_id : varchar(80)
        dlc_smooth_interp_info_object_id : varchar(80)
        """

        def fetch1_dataframe(self) -> pd.DataFrame:
            """Fetch a single dataframe."""
            _ = self.ensure_single_entry()
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
        """Populate DLCSmoothInterpCohort table with the combined bodyparts.

        Calls _logged_make to log the process to a log.log file while...
        1. Fetching the cohort selection and smooted interpolated data for each
              bodypart.
        2. Ensuring the number of bodyparts match across data and parameters.
        3. Inserting the combined bodyparts into DLCSmoothInterpCohort.
        """
        output_dir = infer_output_dir(key=key, makedir=False)
        self.log_path = Path(output_dir) / "log.log"
        self._logged_make(key)
        logger.info("Inserted entry into DLCSmoothInterpCohort")

    @file_log(logger, console=False)
    def _logged_make(self, key):
        logger.info("-----------------------")
        logger.info("Bodypart Cohort")

        cohort_selection = (DLCSmoothInterpCohortSelection & key).fetch1()
        table_entries = []
        bp_params_dict = cohort_selection.pop("bodyparts_params_dict")
        if len(bp_params_dict) == 0:
            logger.warn("No bodyparts specified in bodyparts_params_dict")
            self.insert1(key)
            return
        temp_key = cohort_selection.copy()
        for bodypart, params in bp_params_dict.items():
            temp_key.update(dict(bodypart=bodypart, dlc_si_params_name=params))
            query = DLCSmoothInterp & temp_key
            if len(query):  # added to prevent appending empty array
                table_entries.append((DLCSmoothInterp & temp_key).fetch())

        if not len(table_entries) == len(bp_params_dict):
            raise ValueError(
                f"Mismatch: DLCSmoothInterp {len(table_entries)} vs "
                + f"bodyparts_params_dict {len(bp_params_dict)}"
            )

        table_column_names = DLCSmoothInterp.heading.names
        part_keys = [
            {
                **{k: v for k, v in zip(table_column_names, table_entry[0])},
                **key,
            }
            for table_entry in table_entries
        ]

        self.insert1(key)
        self.BodyPart.insert(part_keys, skip_duplicates=True)

import datajoint as dj
import numpy as np
import pandas as pd
import pynwb
from position_tools.core import gaussian_smooth

from spyglass.common.common_behav import RawPosition
from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.position.v1.dlc_utils import (
    get_span_start_stop,
    interp_orientation,
    no_orientation,
    red_led_bisector_orientation,
    two_pt_head_orientation,
)
from spyglass.utils import SpyglassMixin

from .position_dlc_cohort import DLCSmoothInterpCohort

schema = dj.schema("position_v1_dlc_orient")

# Add new functions for orientation calculation here
_key_to_func_dict = {
    "none": no_orientation,
    "red_green_orientation": two_pt_head_orientation,
    "red_led_bisector": red_led_bisector_orientation,
}


@schema
class DLCOrientationParams(SpyglassMixin, dj.Manual):
    """
    Params for determining and smoothing the orientation of a set of BodyParts

    Attributes
    ----------
    dlc_orientation_params_name : str
        Name for this set of parameters
    params : dict
        Dictionary of parameters, including...
        orient_method : str
            Method for determining orientation. Options are:
            'none': No orientation calculation
            'red_green_orientation': Two-point head orientation calculation
            'red_led_bisector': Red LED bisector orientation calculation
        bodypart1 : str
            First bodypart to use for orientation calculation
        bodypart2 : str
            Second bodypart to use for orientation calculation
        orientation_smoothing_std_dev : float
            Standard deviation for Gaussian smoothing of the orientation data
    """

    definition = """
    dlc_orientation_params_name: varchar(80) # name for this set of parameters
    ---
    params: longblob
    """

    @classmethod
    def insert_params(cls, params_name: str, params: dict, **kwargs):
        """Insert a set of parameters for orientation calculation"""
        cls.insert1(
            {"dlc_orientation_params_name": params_name, "params": params},
            **kwargs,
        )

    @classmethod
    def insert_default(cls, **kwargs):
        """Insert the default set of parameters for orientation calculation"""
        params = {
            "orient_method": "red_green_orientation",
            "bodypart1": "greenLED",
            "bodypart2": "redLED_C",
            "orientation_smoothing_std_dev": 0.001,
        }
        cls.insert1(
            {"dlc_orientation_params_name": "default", "params": params},
            **kwargs,
        )

    @classmethod
    def get_default(cls):
        """Return the default set of parameters for orientation calculation"""
        query = cls & {"dlc_orientation_params_name": "default"}
        if not len(query) > 0:
            cls().insert_default(skip_duplicates=True)
            default = (
                cls & {"dlc_orientation_params_name": "default"}
            ).fetch1()
        else:
            default = query.fetch1()
        return default


@schema
class DLCOrientationSelection(SpyglassMixin, dj.Manual):
    definition = """
    -> DLCSmoothInterpCohort
    -> DLCOrientationParams
    ---
    """


@schema
class DLCOrientation(SpyglassMixin, dj.Computed):
    """
    Determines and smooths orientation of a set of bodyparts given a specified method
    """

    definition = """
    -> DLCOrientationSelection
    ---
    -> AnalysisNwbfile
    dlc_orientation_object_id : varchar(80)
    """

    def _get_pos_df(self, key):
        cohort_entries = DLCSmoothInterpCohort.BodyPart & key
        pos_df = (
            pd.concat(
                {
                    bodypart: (
                        DLCSmoothInterpCohort.BodyPart
                        & {**key, **{"bodypart": bodypart}}
                    ).fetch1_dataframe()
                    for bodypart in cohort_entries.fetch("bodypart")
                },
                axis=1,
            )
            if cohort_entries
            else pd.DataFrame()
        )
        return pos_df

    def make(self, key):
        """Populate the DLCOrientation table.

        1. Fetch parameters and position data from DLCOrientationParams and
            DLCSmoothInterpCohort.BodyPart tables, respectively.
        2. Apply chosen orientation method to position data.
        3. Generate a CompassDirection object and add it to the AnalysisNwbfile.
        4. Insert the key into the DLCOrientation table.
        """
        # Get labels to smooth from Parameters table
        pos_df = self._get_pos_df(key)

        params = (DLCOrientationParams() & key).fetch1("params")
        orientation_smoothing_std_dev = params.pop(
            "orientation_smoothing_std_dev", None
        )
        sampling_rate = 1 / np.median(np.diff(pos_df.index.to_numpy()))
        orient_func = _key_to_func_dict[params["orient_method"]]
        orientation = orient_func(pos_df, **params)

        # TODO: Absorb this into the `no_orientation` function
        if not params["orient_method"] == "none":
            # Smooth orientation
            is_nan = np.isnan(orientation)
            unwrap_orientation = orientation.copy()
            # Only unwrap non nan values, while keeping nans in dataset for interpolation
            unwrap_orientation[~is_nan] = np.unwrap(orientation[~is_nan])
            unwrap_df = pd.DataFrame(
                unwrap_orientation, columns=["orientation"], index=pos_df.index
            )
            nan_spans = get_span_start_stop(np.where(is_nan)[0])
            orient_df = interp_orientation(
                unwrap_df,
                nan_spans,
            )
            orientation = gaussian_smooth(
                orient_df["orientation"].to_numpy(),
                orientation_smoothing_std_dev,
                sampling_rate,
                axis=0,
                truncate=8,
            )
            # convert back to between -pi and pi
            orientation = np.angle(np.exp(1j * orientation))

        final_df = pd.DataFrame(
            orientation, columns=["orientation"], index=pos_df.index
        )
        key["analysis_file_name"] = AnalysisNwbfile().create(
            key["nwb_file_name"]
        )
        # if spatial series exists, get metadata from there
        if query := (RawPosition & key):
            spatial_series = query.fetch_nwb()[0]["raw_position"]
        else:
            spatial_series = None  # pragma: no cover

        orientation = pynwb.behavior.CompassDirection()
        orientation.create_spatial_series(
            name="orientation",
            timestamps=final_df.index.to_numpy(),
            conversion=1.0,
            data=final_df["orientation"].to_numpy(),
            reference_frame=getattr(spatial_series, "reference_frame", ""),
            comments=getattr(spatial_series, "comments", "no comments"),
            description="orientation",
        )
        nwb_analysis_file = AnalysisNwbfile()
        key["dlc_orientation_object_id"] = nwb_analysis_file.add_nwb_object(
            key["analysis_file_name"], orientation
        )

        nwb_analysis_file.add(
            nwb_file_name=key["nwb_file_name"],
            analysis_file_name=key["analysis_file_name"],
        )

        self.insert1(key)

    def fetch1_dataframe(self) -> pd.DataFrame:
        """Fetch a single dataframe"""
        _ = self.ensure_single_entry()
        nwb_data = self.fetch_nwb()[0]
        index = pd.Index(
            np.asarray(
                nwb_data["dlc_orientation"].get_spatial_series().timestamps
            ),
            name="time",
        )
        COLUMNS = ["orientation"]
        return pd.DataFrame(
            np.asarray(nwb_data["dlc_orientation"].get_spatial_series().data)[
                :, np.newaxis
            ],
            columns=COLUMNS,
            index=index,
        )

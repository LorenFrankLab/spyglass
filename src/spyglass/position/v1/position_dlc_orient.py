import numpy as np
import pandas as pd
import datajoint as dj
import pynwb
from position_tools.core import gaussian_smooth
from ...common.dj_helper_fn import fetch_nwb
from ...common.common_nwbfile import AnalysisNwbfile
from .position_dlc_cohort import DLCSmoothInterpCohort
from .position_dlc_project import BodyPart
from ...common.common_behav import RawPosition

schema = dj.schema("position_dlc_orient")


@schema
class DLCOrientationParams(dj.Manual):
    """
    Parameters for determining and smoothing the orientation of a set of BodyParts
    """

    definition = """
    dlc_orientation_params_name: varchar(80) # name for this set of parameters
    ---
    params: longblob
    """

    @classmethod
    def insert_params(cls, params_name: str, params: dict, **kwargs):
        cls.insert1(
            {"dlc_orientation_params_name": params_name, "params": params}, **kwargs
        )

    @classmethod
    def insert_default(cls, **kwargs):
        params = {
            "orient_method": "red_green_orientation",
            "bodypart1": "greenLED",
            "bodypart2": "redLED_C",
            "orientation_smoothing_std_dev": 0.001,
        }
        cls.insert1(
            {"dlc_orientation_params_name": "default", "params": params}, **kwargs
        )

    @classmethod
    def get_default(cls):
        query = cls & {"dlc_orientation_params_name": "default"}
        if not len(query) > 0:
            cls().insert_default(skip_duplicates=True)
            default = (cls & {"dlc_orientation_params_name": "default"}).fetch1()
        else:
            default = query.fetch1()
        return default


@schema
class DLCOrientationSelection(dj.Manual):
    """ """

    definition = """
    -> DLCSmoothInterpCohort
    -> DLCOrientationParams
    ---
    """


@schema
class DLCOrientation(dj.Computed):
    """
    Determines and smooths orientation of a set of bodyparts given a specified method
    """

    definition = """
    -> DLCOrientationSelection
    ---
    -> AnalysisNwbfile
    dlc_orientation_object_id : varchar(80)
    """

    def make(self, key):
        # Get labels to smooth from Parameters table
        cohort_entries = DLCSmoothInterpCohort.BodyPart & key
        pos_df = pd.concat(
            {
                bodypart: (
                    DLCSmoothInterpCohort.BodyPart & {**key, **{"bodypart": bodypart}}
                ).fetch1_dataframe()
                for bodypart in cohort_entries.fetch("bodypart")
            },
            axis=1,
        )
        params = (DLCOrientationParams() & key).fetch1("params")
        orientation_smoothing_std_dev = params.pop(
            "orientation_smoothing_std_dev", None
        )
        dt = np.median(np.diff(pos_df.index.to_numpy()))
        sampling_rate = 1 / dt
        orient_func = _key_to_func_dict[params["orient_method"]]
        orientation = orient_func(pos_df, **params)
        if not params["orient_method"] == "none":
            # Smooth orientation
            is_nan = np.isnan(orientation)
            # Unwrap orientation before smoothing
            orientation[~is_nan] = np.unwrap(orientation[~is_nan])
            orientation[~is_nan] = gaussian_smooth(
                orientation[~is_nan],
                orientation_smoothing_std_dev,
                sampling_rate,
                axis=0,
                truncate=8,
            )
            # convert back to between -pi and pi
            orientation[~is_nan] = np.angle(np.exp(1j * orientation[~is_nan]))
        final_df = pd.DataFrame(
            orientation, columns=["orientation"], index=pos_df.index
        )
        key["analysis_file_name"] = AnalysisNwbfile().create(key["nwb_file_name"])
        spatial_series = (RawPosition() & key).fetch_nwb()[0]["raw_position"]
        orientation = pynwb.behavior.CompassDirection()
        orientation.create_spatial_series(
            name="orientation",
            timestamps=final_df.index.to_numpy(),
            conversion=1.0,
            data=final_df["orientation"].to_numpy(),
            reference_frame=spatial_series.reference_frame,
            comments=spatial_series.comments,
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

    def fetch_nwb(self, *attrs, **kwargs):
        return fetch_nwb(
            self, (AnalysisNwbfile, "analysis_file_abs_path"), *attrs, **kwargs
        )

    def fetch1_dataframe(self):
        nwb_data = self.fetch_nwb()[0]
        index = pd.Index(
            np.asarray(nwb_data["dlc_orientation"].get_spatial_series().timestamps),
            name="time",
        )
        COLUMNS = [
            "orientation",
        ]
        return pd.DataFrame(
            np.asarray(nwb_data["dlc_orientation"].get_spatial_series().data)[
                :, np.newaxis
            ],
            columns=COLUMNS,
            index=index,
        )


def two_pt_head_orientation(pos_df: pd.DataFrame, **params):
    """Determines orientation based on vector between two points"""
    BP1 = params.pop("bodypart1", None)
    BP2 = params.pop("bodypart2", None)
    orientation = np.arctan2(
        (pos_df[BP1]["y"] - pos_df[BP2]["y"]),
        (pos_df[BP1]["x"] - pos_df[BP2]["x"]),
    )
    return orientation


def no_orientation(pos_df: pd.DataFrame, **params):
    fill_value = params.pop("fill_with", np.nan)
    n_frames = len(pos_df)
    orientation = np.full(shape=(n_frames), fill_value=fill_value, dtype=np.float16)
    return orientation


def red_led_bisector_orientation(pos_df: pd.DataFrame, **params):
    """Determines orientation based on 2 equally-spaced identifiers
    that are assumed to be perpendicular to the orientation direction.
    A third object is needed to determine forward/backward
    """
    LED1 = params.pop("led1", None)
    LED2 = params.pop("led2", None)
    LED3 = params.pop("led3", None)
    orientation = []
    for index, row in pos_df.iterrows():
        x_vec = row[LED1]["x"] - row[LED2]["x"]
        y_vec = row[LED1]["y"] - row[LED2]["y"]
        if y_vec == 0:
            if (row[LED3]["y"] > row[LED1]["y"]) & (row[LED3]["y"] > row[LED2]["y"]):
                orientation.append(np.pi / 2)
            elif (row[LED3]["y"] < row[LED1]["y"]) & (row[LED3]["y"] < row[LED2]["y"]):
                orientation.append(-(np.pi / 2))
            else:
                raise Exception("Cannot determine head direction from bisector")
        else:
            length = np.sqrt(y_vec * y_vec + x_vec * x_vec)
            norm = np.array([-y_vec / length, x_vec / length])
            orientation.append(np.arctan2(norm[1], norm[0]))
        if index + 1 == len(pos_df):
            break
    return np.array(orientation)


# Add new functions for orientation calculation here

_key_to_func_dict = {
    "none": no_orientation,
    "red_green_orientation": two_pt_head_orientation,
    "red_led_bisector": red_led_bisector_orientation,
}

from itertools import groupby
from operator import itemgetter
import numpy as np
import pandas as pd
import datajoint as dj
import bottleneck as bn
from position_tools import (
    get_speed,
    interpolate_nan,
)
from ..common.dj_helper_fn import fetch_nwb
from ..common.common_nwbfile import AnalysisNwbfile
from ..common.common_behav import RawPosition
from .position_dlc_pose_estimation import DLCPoseEstimation
from .position_dlc_project import BodyPart

schema = dj.schema("position_dlc_position")


@schema
class DLCSmoothInterpParams(dj.Manual):
    """
     Parameters for extracting the smoothed head position.

    Parameters
    ----------
    smoothing_params : dict
        smoothing_duration : float, default 0.05
            number of frames to smooth over: sampling_rate*smoothing_duration = num_frames
    interp_params : dict
        likelihood_thresh : float, default 0.95
            likelihood below which to NaN and interpolate over
    sampling_rate : int
        sampling rate of the recording
    max_plausible_speed : float, default 300.0
        fastest possible speed (m/s) bodypart could move,
        above which is NaN
    speed_smoothing_std_dev : float, default 0.1
        standard deviation of gaussian kernel to smooth speed
    """

    definition = """
    dlc_si_params_name : varchar(80) # name for this set of parameters
    ---
    params: longblob # dictionary of parameters
    """

    @classmethod
    def insert_params(cls, params_name: str, params: dict, skip_duplicates=True):

        cls.insert1(
            {"dlc_si_params_name": params_name, "params": params},
            skip_duplicates=skip_duplicates,
        )

    @classmethod
    def insert_default(cls):
        default_params = {
            "smoothing_params": {
                "smoothing_duration": 0.05,
            },
            "interp_params": {
                "likelihood_thresh": 0.95,
            },
            "max_plausible_speed": 300.0,
            "speed_smoothing_std_dev": 0.100,
        }
        cls.insert1(
            {"dlc_si_params_name": "default", "params": default_params},
            skip_duplicates=True,
        )

    @classmethod
    def get_default(cls):
        default = (cls & {"dlc_si_params_name": "default"}).fetch1()
        if not len(default) > 0:
            cls().insert_default(skip_duplicates=True)
            default = (cls & {"dlc_si_params_name": "default"}).fetch1()
        return default

    # def delete(self, key, **kwargs):
    #     super().delete(key, **kwargs)


@schema
class DLCSmoothInterpSelection(dj.Manual):
    definition = """
    -> DLCPoseEstimation.BodyPart
    -> DLCSmoothInterpParams
    ---

    """


@schema
class DLCSmoothInterp(dj.Computed):
    """
    Interpolates across low likelihood periods and smooths the position
    Can take a few minutes.
    """

    definition = """
    -> DLCSmoothInterpSelection
    ---
    -> AnalysisNwbfile
    dlc_smooth_interp_object_id : varchar(80)
    """

    def make(self, key):
        # Get labels to smooth from Parameters table
        params = (DLCSmoothInterpParams() & key).fetch1("params")
        max_plausible_speed = params.pop("max_plausible_speed", None)
        speed_smoothing_std_dev = params.pop("speed_smoothing_std_dev", None)
        # Get DLC output dataframe
        print("fetching Pose Estimation Dataframe")
        dlc_df = (DLCPoseEstimation.BodyPart() & key).fetch1_dataframe()
        dt = np.median(np.diff(dlc_df.index.to_numpy()))
        sampling_rate = 1 / dt
        # Calculate speed
        idx = pd.IndexSlice
        print("Calculating speed")
        LED_speed = get_speed(
            dlc_df.loc[:, idx[("x", "y")]],
            dlc_df.index.to_numpy(),
            sigma=speed_smoothing_std_dev,
            sampling_frequency=sampling_rate,
        )
        # Set points to NaN where the speed is too fast
        is_too_fast = LED_speed > max_plausible_speed
        dlc_df.loc[idx[is_too_fast], idx[("x", "y")]] = np.nan
        # Interpolate the NaN points
        dlc_df.loc[:, idx[("x", "y")]] = interpolate_nan(
            position=dlc_df.loc[:, idx[("x", "y")]].to_numpy(),
            time=dlc_df.index.to_numpy(),
        )
        # get interpolated points
        print("interpolating across low likelihood times")
        interp_df = interp_pos(dlc_df, **params["interp_params"])
        if "smoothing_duration" in params["smoothing_params"]:
            smoothing_duration = params["smoothing_params"].pop("smoothing_duration")
        dt = np.median(np.diff(dlc_df.index.to_numpy()))
        sampling_rate = 1 / dt
        print("smoothing position")
        smooth_df = smooth_pos(
            interp_df,
            smoothing_duration=smoothing_duration,
            sampling_rate=sampling_rate,
            **params["smoothing_params"],
        )
        final_df = smooth_df.drop(["likelihood"], axis=1)
        final_df = final_df.rename_axis("time").reset_index()
        key["analysis_file_name"] = AnalysisNwbfile().create(key["nwb_file_name"])
        # Add dataframe to AnalysisNwbfile
        nwb_analysis_file = AnalysisNwbfile()
        key["dlc_smooth_interp_object_id"] = nwb_analysis_file.add_nwb_object(
            analysis_file_name=key["analysis_file_name"],
            nwb_object=final_df,
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
        return self.fetch_nwb()[0]["dlc_smooth_interp"].set_index("time")


def interp_pos(dlc_df, **kwargs):

    idx = pd.IndexSlice
    subthresh_inds = get_subthresh_inds(
        dlc_df, likelihood_thresh=kwargs.pop("likelihood_thresh")
    )
    subthresh_spans = get_span_start_stop(subthresh_inds)
    for ind, (span_start, span_stop) in enumerate(subthresh_spans):
        x = [dlc_df["x"].iloc[span_start - 1], dlc_df["x"].iloc[span_stop + 1]]
        y = [dlc_df["y"].iloc[span_start - 1], dlc_df["y"].iloc[span_stop + 1]]
        span_len = int(span_stop - span_start + 1)
        # TODO: determine if necessary to allow for these parameters
        if "max_pts_to_interp" in kwargs:
            if span_len > kwargs["max_pts_to_interp"]:
                if "max_cm_to_interp" in kwargs:
                    if (
                        np.linalg.norm(np.array([x[0], y[0]]) - np.array([x[1], y[1]]))
                        < kwargs["max_cm_to_interp"]
                    ):
                        change = np.linalg.norm(
                            np.array([x[0], y[0]]) - np.array([x[1], y[1]])
                        )
                        print(
                            f"ind: {ind} length: "
                            f"{span_len} interpolated because minimal change:\n {change}cm"
                        )
                    else:
                        dlc_df.loc[idx[span_start:span_stop], idx["x"]] = np.nan
                        dlc_df.loc[idx[span_start:span_stop], idx["y"]] = np.nan
                        print(f"ind: {ind} length: {span_len} " f"not interpolated")
                        continue
        start_time = dlc_df.index[span_start]
        stop_time = dlc_df.index[span_stop]
        xnew = np.interp(
            x=dlc_df.index[span_start : span_stop + 1],
            xp=[start_time, stop_time],
            fp=[x[0], x[-1]],
        )
        ynew = np.interp(
            x=dlc_df.index[span_start : span_stop + 1],
            xp=[start_time, stop_time],
            fp=[y[0], y[-1]],
        )
        dlc_df.loc[idx[start_time:stop_time], idx["x"]] = xnew
        dlc_df.loc[idx[start_time:stop_time], idx["y"]] = ynew
    return dlc_df


def get_subthresh_inds(dlc_df: pd.DataFrame, likelihood_thresh: float):
    # Need to get likelihood threshold from kwargs or make it a specified argument
    df_filter = dlc_df["likelihood"] < likelihood_thresh
    sub_thresh_inds = np.where(~np.isnan(dlc_df["likelihood"].where(df_filter)))[0]
    sub_thresh_percent = (len(sub_thresh_inds) / len(dlc_df)) * 100
    # TODO: add option to return sub_thresh_percent
    return sub_thresh_inds


def get_span_start_stop(sub_thresh_inds):

    sub_thresh_spans = []
    for k, g in groupby(enumerate(sub_thresh_inds), lambda x: x[1] - x[0]):
        group = list(map(itemgetter(1), g))
        sub_thresh_spans.append((group[0], group[-1]))
    return sub_thresh_spans


def smooth_pos(interp_df, smoothing_duration: float, sampling_rate: int, **kwargs):
    idx = pd.IndexSlice
    moving_avg_window = int(smoothing_duration * sampling_rate)
    xy_arr = interp_df.loc[:, idx[("x", "y")]].values
    smoothed_xy_arr = bn.move_mean(
        xy_arr, window=moving_avg_window, axis=0, min_count=1
    )
    interp_df.loc[:, idx["x"]], interp_df.loc[:, idx["y"]] = [
        *zip(*smoothed_xy_arr.tolist())
    ]
    return interp_df

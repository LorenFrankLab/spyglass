from itertools import groupby
from operator import itemgetter
import numpy as np
import pandas as pd
import datajoint as dj
import bottleneck as bn
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
    def insert_params(cls, params_name: str, params: dict, **kwargs):

        cls.insert1(
            {"dlc_si_params_name": params_name, "params": params},
            **kwargs,
        )

    @classmethod
    def insert_default(cls, **kwargs):
        default_params = {
            "smoothing_params": {
                "smoothing_duration": 0.05,
                "smooth_method": "moving_avg",
            },
            "interp_params": {
                "likelihood_thresh": 0.95,
            },
            "max_cm_between_pts": 20,
            "speed_smoothing_std_dev": 0.100,
        }
        cls.insert1(
            {"dlc_si_params_name": "default", "params": default_params}, **kwargs
        )

    @classmethod
    def get_default(cls):
        query = cls & {"dlc_si_params_name": "default"}
        if not len(query) > 0:
            cls().insert_default(skip_duplicates=True)
            default = (cls & {"dlc_si_params_name": "default"}).fetch1()
        else:
            default = query.fetch1()
        return default

    @staticmethod
    def get_available_methods():
        return _key_to_smooth_func_dict.keys()

    def insert1(self, key, **kwargs):
        if "params" in key:
            if "smoothing_params" in key["params"]:
                if "smooth_method" in key["params"]["smoothing_params"]:
                    smooth_method = key["params"]["smoothing_params"]["smooth_method"]
                    if smooth_method not in _key_to_smooth_func_dict:
                        raise KeyError(
                            f"smooth_method: {smooth_method} not an available method."
                        )
                if not "smoothing_duration" in key["params"]["smoothing_params"]:
                    raise KeyError(
                        "smoothing_duration must be passed as a smoothing_params within key['params']"
                    )
                else:
                    assert isinstance(
                        key["params"]["smoothing_params"]["smoothing_duration"],
                        (float, int),
                    ), "smoothing_duration must be a float or int"
            else:
                raise ValueError("smoothing_params not in key['params']")
            if "interp_params" in key["params"]:
                if "likelihood_thresh" in key["params"]["interp_params"]:
                    assert isinstance(
                        key["params"]["interp_params"]["likelihood_thresh"], float
                    ), "likelihood_thresh must be a float"
                    assert (
                        0 < key["params"]["interp_params"]["likelihood_thresh"] < 1
                    ), "likelihood_thresh must be between 0 and 1"
                else:
                    raise ValueError(
                        "likelihood_thresh must be passed as an interp_params within key['params']"
                    )
            else:
                raise ValueError("interp_params not in key['params']")
        else:
            raise KeyError("'params' must be in key")
        super().insert1(key, **kwargs)

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
        # Get DLC output dataframe
        print("fetching Pose Estimation Dataframe")
        dlc_df = (DLCPoseEstimation.BodyPart() & key).fetch1_dataframe()
        dt = np.median(np.diff(dlc_df.index.to_numpy()))
        sampling_rate = 1 / dt
        idx = pd.IndexSlice
        if "max_cm_between_pts" in params:
            bad_inds = get_jump_points(dlc_df, params["max_cm_between_pts"])
            pts_to_nan = dlc_df.index[bad_inds]
            print(
                f"{len(pts_to_nan)} pts with jump more than {params['max_cm_between_pts']} cm"
            )
            dlc_df.loc[idx[pts_to_nan], idx[("x", "y")]] = np.nan
        subthresh_inds = get_subthresh_inds(
            dlc_df.copy(),
            likelihood_thresh=params["interp_params"].pop("likelihood_thresh"),
        )
        subthresh_spans = get_span_start_stop(subthresh_inds)
        # Check to make sure all inds from bad_inds are in final subthresh_inds
        if len(bad_inds) > 0:
            all_subthresh_inds = [
                item
                for entry in subthresh_spans
                for item in list(range(entry[0], entry[1] + 1))
            ]
            assert all(
                item in all_subthresh_inds for item in bad_inds
            ), "not all indices from get_jump_points in subthresh_spans"
        # get interpolated points
        print("interpolating across low likelihood times")
        interp_df = interp_pos(
            dlc_df.copy(), subthresh_spans, **params["interp_params"]
        )
        if "smoothing_duration" in params["smoothing_params"]:
            smoothing_duration = params["smoothing_params"].pop("smoothing_duration")
        dt = np.median(np.diff(dlc_df.index.to_numpy()))
        sampling_rate = 1 / dt
        print("smoothing position")
        smooth_func = _key_to_smooth_func_dict[
            params["smoothing_params"]["smooth_method"]
        ]
        smooth_df = smooth_func(
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


def interp_pos(dlc_df, spans_to_interp, **kwargs):

    idx = pd.IndexSlice
    for ind, (span_start, span_stop) in enumerate(spans_to_interp):
        if (span_stop + 1) >= len(dlc_df):
            dlc_df.loc[idx[span_start:span_stop], idx["x"]] = np.nan
            dlc_df.loc[idx[span_start:span_stop], idx["y"]] = np.nan
            print(f"ind: {ind} has no endpoint with which to interpolate")
            continue
        if span_start < 1:
            dlc_df.loc[idx[span_start:span_stop], idx["x"]] = np.nan
            dlc_df.loc[idx[span_start:span_stop], idx["y"]] = np.nan
            print(f"ind: {ind} has no startpoint with which to interpolate")
            continue
        x = [dlc_df["x"].iloc[span_start - 1], dlc_df["x"].iloc[span_stop + 1]]
        y = [dlc_df["y"].iloc[span_start - 1], dlc_df["y"].iloc[span_stop + 1]]
        span_len = int(span_stop - span_start + 1)
        start_time = dlc_df.index[span_start]
        stop_time = dlc_df.index[span_stop]
        # TODO: determine if necessary to allow for these parameters
        if "max_pts_to_interp" in kwargs:
            if span_len > kwargs["max_pts_to_interp"]:
                dlc_df.loc[idx[span_start:span_stop], idx["x"]] = np.nan
                dlc_df.loc[idx[span_start:span_stop], idx["y"]] = np.nan
                print(f"ind: {ind} length: {span_len} " f"not interpolated")
        if "max_cm_to_interp" in kwargs:
            if (
                np.linalg.norm(np.array([x[0], y[0]]) - np.array([x[1], y[1]]))
                < kwargs["max_cm_to_interp"]
            ):
                change = np.linalg.norm(np.array([x[0], y[0]]) - np.array([x[1], y[1]]))
            else:
                dlc_df.loc[idx[start_time:stop_time], idx["x"]] = np.nan
                dlc_df.loc[idx[start_time:stop_time], idx["y"]] = np.nan
                print(f"ind: {ind} length: {span_len} " f"not interpolated")
                continue
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


def get_jump_points(dlc_df: pd.DataFrame, max_dist_between):

    idx = pd.IndexSlice
    total_diff = np.zeros(shape=(len(dlc_df)))
    total_diff[1:] = np.sqrt(
        np.sum(
            np.abs(np.diff(dlc_df.loc[:, idx[("x", "y")]].to_numpy(), axis=0)) ** 2,
            axis=1,
        )
    )
    too_much_jump_inds = np.where(total_diff > max_dist_between)[0]
    inds_that_jump = []
    # Get spans of consecutive indices that jump
    for k, g in groupby(enumerate(too_much_jump_inds), lambda x: x[1] - x[0]):
        group = list(map(itemgetter(1), g))
        inds_that_jump.append((group[0], group[-1]))
    # see if spans are within 2 inds of each other and combine
    modified_jump_spans = []
    for (start1, stop1), (start2, stop2) in zip(
        inds_that_jump[:-1], inds_that_jump[1:]
    ):
        check_existing = [
            entry
            for entry in modified_jump_spans
            if start1 in range(entry[0] - 2, entry[1] + 2)
        ]
        if len(check_existing) > 0:
            modify_ind = modified_jump_spans.index(check_existing[0])
            if (start2 - stop1) <= 2:
                modified_jump_spans[modify_ind] = (check_existing[0][0], stop2)
            else:
                modified_jump_spans[modify_ind] = (check_existing[0][0], stop1)
                modified_jump_spans.append((start2, stop2))
            continue
        if (start2 - stop1) <= 2:
            modified_jump_spans.append((start1, stop2))
        else:
            modified_jump_spans.append((start1, stop1))
            modified_jump_spans.append((start2, stop2))
    all_inds_that_jump = [
        item for entry in inds_that_jump for item in list(range(entry[0], entry[1] + 1))
    ]
    all_modified_jump_spans = [
        item
        for entry in modified_jump_spans
        for item in list(range(entry[0], entry[1] + 1))
    ]
    assert all(
        item in all_modified_jump_spans for item in all_inds_that_jump
    ), "not all entries to NaN are represented by modified_jump_spans"
    # To further determine which indices are the original point and which are jump points
    # There could be a more efficient method of doing this
    bad_inds = []
    for span in modified_jump_spans:
        if span[0] <= 5:
            x_mean, y_mean = np.nanmean(
                dlc_df.loc[:, idx[("x", "y")]]
                .iloc[span[1] + 1 : span[1] + 5]
                .to_numpy(),
                axis=0,
            )
        if span[1] >= (len(dlc_df) - 5):
            x_mean, y_mean = np.nanmean(
                dlc_df.loc[:, idx[("x", "y")]]
                .iloc[span[0] - 5 : span[0] - 1]
                .to_numpy(),
                axis=0,
            )
        else:
            start_x, start_y = np.nanmean(
                dlc_df.loc[:, idx[("x", "y")]]
                .iloc[span[0] - 5 : span[0] - 1]
                .to_numpy(),
                axis=0,
            )
            stop_x, stop_y = np.nanmean(
                dlc_df.loc[:, idx[("x", "y")]]
                .iloc[span[1] + 1 : span[1] + 5]
                .to_numpy(),
                axis=0,
            )
            x_mean = np.nanmean([start_x, stop_x])
            y_mean = np.nanmean([start_y, stop_y])
        bad_x = np.where(
            (dlc_df.x.iloc[span[0] : span[1] + 1] < int(x_mean - max_dist_between))
            | (dlc_df.x.iloc[span[0] : span[1] + 1] > int(x_mean + max_dist_between))
        )[0]
        bad_y = np.where(
            (dlc_df.y.iloc[span[0] : span[1] + 1] < int(y_mean - max_dist_between))
            | (dlc_df.y.iloc[span[0] : span[1] + 1] > int(y_mean + max_dist_between))
        )[0]
        bad_inds.extend(
            np.arange(span[0], span[1] + 1)[list(set(bad_x).union(set(bad_y)))]
        )
    return bad_inds


def get_subthresh_inds(dlc_df: pd.DataFrame, likelihood_thresh: float):
    df_filter = dlc_df["likelihood"] < likelihood_thresh
    sub_thresh_inds = np.where(~np.isnan(dlc_df["likelihood"].where(df_filter)))[0]
    nan_inds = np.where(np.isnan(dlc_df["x"]))[0]
    all_nan_inds = list(set(sub_thresh_inds).union(set(nan_inds)))
    all_nan_inds.sort()
    sub_thresh_percent = (len(sub_thresh_inds) / len(dlc_df)) * 100
    # TODO: add option to return sub_thresh_percent
    return all_nan_inds


def get_span_start_stop(sub_thresh_inds):

    sub_thresh_spans = []
    for k, g in groupby(enumerate(sub_thresh_inds), lambda x: x[1] - x[0]):
        group = list(map(itemgetter(1), g))
        sub_thresh_spans.append((group[0], group[-1]))
    return sub_thresh_spans


def smooth_moving_avg(
    interp_df, smoothing_duration: float, sampling_rate: int, **kwargs
):
    idx = pd.IndexSlice
    moving_avg_window = int(smoothing_duration * sampling_rate)
    xy_arr = interp_df.loc[:, idx[("x", "y")]].values
    smoothed_xy_arr = bn.move_mean(
        xy_arr, window=moving_avg_window, axis=0, min_count=2
    )
    interp_df.loc[:, idx["x"]], interp_df.loc[:, idx["y"]] = [
        *zip(*smoothed_xy_arr.tolist())
    ]
    return interp_df


_key_to_smooth_func_dict = {
    "moving_avg": smooth_moving_avg,
}

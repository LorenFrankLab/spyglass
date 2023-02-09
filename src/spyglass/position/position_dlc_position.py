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
            # This is for use when finding "good spans" and is how many indices to bridge in between good spans
            # see inds_to_span in get_good_spans
            "num_inds_to_span": 20,
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
            if not "max_cm_between_pts" in key["params"]:
                raise KeyError("max_cm_between_pts is a required parameter")
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
        from .dlc_utils import OutputLogger, infer_output_dir

        output_dir = infer_output_dir(key=key, makedir=False)
        with OutputLogger(
            name=f"{key['nwb_file_name']}_{key['epoch']}_{key['dlc_model_name']}_log",
            path=f"{output_dir.as_posix()}/log.log",
            print_console=False,
        ) as logger:
            logger.logger.info("-----------------------")
            logger.logger.info("Interpolation and Smoothing")
            # Get labels to smooth from Parameters table
            params = (DLCSmoothInterpParams() & key).fetch1("params")
            # Get DLC output dataframe
            logger.logger.info("fetching Pose Estimation Dataframe")
            dlc_df = (DLCPoseEstimation.BodyPart() & key).fetch1_dataframe()
            dt = np.median(np.diff(dlc_df.index.to_numpy()))
            sampling_rate = 1 / dt
            df_w_nans, bad_inds = nan_inds(
                dlc_df.copy(),
                params["max_cm_between_pts"],
                likelihood_thresh=params["interp_params"].pop("likelihood_thresh"),
                inds_to_span=params["num_inds_to_span"],
            )
            nan_spans = get_span_start_stop(np.where(bad_inds)[0])
            logger.logger.info("interpolating across low likelihood times")
            interp_df = interp_pos(
                df_w_nans.copy(), nan_spans, **params["interp_params"]
            )
            if "smoothing_duration" in params["smoothing_params"]:
                smoothing_duration = params["smoothing_params"].pop(
                    "smoothing_duration"
                )
            dt = np.median(np.diff(dlc_df.index.to_numpy()))
            sampling_rate = 1 / dt
            logger.logger.info("smoothing position")
            smooth_func = _key_to_smooth_func_dict[
                params["smoothing_params"]["smooth_method"]
            ]
            logger.logger.info(
                "Smoothing using method: %s",
                str(params["smoothing_params"]["smooth_method"]),
            )
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
            logger.logger.info("inserted entry into DLCSmoothInterp")

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
        if "max_pts_to_interp" in kwargs:
            if span_len > kwargs["max_pts_to_interp"]:
                dlc_df.loc[idx[span_start:span_stop], idx["x"]] = np.nan
                dlc_df.loc[idx[span_start:span_stop], idx["y"]] = np.nan
                print(
                    f"inds {span_start} to {span_stop} "
                    f"length: {span_len} not interpolated"
                )
        if "max_cm_to_interp" in kwargs:
            if (
                np.linalg.norm(np.array([x[0], y[0]]) - np.array([x[1], y[1]]))
                > kwargs["max_cm_to_interp"]
            ):
                dlc_df.loc[idx[start_time:stop_time], idx["x"]] = np.nan
                dlc_df.loc[idx[start_time:stop_time], idx["y"]] = np.nan
                change = np.linalg.norm(np.array([x[0], y[0]]) - np.array([x[1], y[1]]))
                print(
                    f"inds {span_start} to {span_stop + 1} "
                    f"with change in position: {change:.2f} not interpolated"
                )
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
    print("Getting indices that jump more than {max_dist_between}cm")
    idx = pd.IndexSlice
    total_diff = np.zeros(shape=(len(dlc_df)))
    total_diff[1:] = np.sqrt(
        np.sum(
            np.abs(np.diff(dlc_df.loc[:, idx[("x", "y")]].to_numpy(), axis=0)) ** 2,
            axis=1,
        )
    )
    too_much_jump_inds = np.where(total_diff > max_dist_between)[0]
    return too_much_jump_inds


def nan_inds(
    dlc_df: pd.DataFrame, max_dist_between, likelihood_thresh: float, inds_to_span: int
):
    idx = pd.IndexSlice
    # Could either NaN sub-likelihood threshold inds here and then not consider in jumping...
    # OR just keep in back pocket when checking jumps against last good point
    subthresh_inds = get_subthresh_inds(dlc_df, likelihood_thresh=likelihood_thresh)
    df_subthresh_indices = dlc_df.index[subthresh_inds]
    dlc_df.loc[idx[df_subthresh_indices], idx[("x", "y")]] = np.nan
    # To further determine which indices are the original point and which are jump points
    # There could be a more efficient method of doing this
    # screen inds for jumps to baseline
    subthresh_inds_mask = np.zeros(len(dlc_df), dtype=bool)
    subthresh_inds_mask[subthresh_inds] = True
    jump_inds_mask = np.zeros(len(dlc_df), dtype=bool)
    orig_spans, good_spans = get_good_spans(
        subthresh_inds_mask, inds_to_span=inds_to_span
    )

    for span in good_spans[::-1]:
        if np.sum(np.isnan(dlc_df.iloc[span[0] : span[-1]].x)) > 0:
            nan_mask = np.isnan(dlc_df.iloc[span[0] : span[-1]].x)
            good_start = np.arange(span[0], span[1])[~nan_mask]
            start_point = good_start[int(len(good_start) // 2)]
        else:
            start_point = span[0] + int(span_length(span) // 2)
        for ind in range(start_point, span[0], -1):
            if subthresh_inds_mask[ind]:
                continue
            previous_good_inds = np.where(
                np.logical_and(
                    ~np.isnan(dlc_df.iloc[ind + 1 : start_point].x),
                    ~jump_inds_mask[ind + 1 : start_point],
                    ~subthresh_inds_mask[ind + 1 : start_point],
                )
            )[0]
            if len(previous_good_inds) >= 1:
                last_good_ind = ind + 1 + np.min(previous_good_inds)
            else:
                last_good_ind = start_point
            good_x, good_y = dlc_df.loc[idx[dlc_df.index[last_good_ind]], ["x", "y"]]
            if (
                (dlc_df.y.iloc[ind] < int(good_y - max_dist_between))
                | (dlc_df.y.iloc[ind] > int(good_y + max_dist_between))
            ) | (
                (dlc_df.x.iloc[ind] < int(good_x - max_dist_between))
                | (dlc_df.x.iloc[ind] > int(good_x + max_dist_between))
            ):
                jump_inds_mask[ind] = True
        for ind in range(start_point, span[-1], 1):
            if subthresh_inds_mask[ind]:
                continue
            previous_good_inds = np.where(
                np.logical_and(
                    ~np.isnan(dlc_df.iloc[start_point:ind].x),
                    ~jump_inds_mask[start_point:ind],
                    ~subthresh_inds_mask[start_point:ind],
                )
            )[0]
            if len(previous_good_inds) >= 1:
                last_good_ind = start_point + np.max(previous_good_inds)
            else:
                last_good_ind = start_point
            good_x, good_y = dlc_df.loc[idx[dlc_df.index[last_good_ind]], ["x", "y"]]
            if (
                (dlc_df.y.iloc[ind] < int(good_y - max_dist_between))
                | (dlc_df.y.iloc[ind] > int(good_y + max_dist_between))
            ) | (
                (dlc_df.x.iloc[ind] < int(good_x - max_dist_between))
                | (dlc_df.x.iloc[ind] > int(good_x + max_dist_between))
            ):
                jump_inds_mask[ind] = True
        bad_inds_mask = np.logical_or(jump_inds_mask, subthresh_inds_mask)
        dlc_df.loc[bad_inds_mask, idx[("x", "y")]] = np.nan
    return dlc_df, bad_inds_mask


def get_good_spans(bad_inds_mask, inds_to_span: int = 50):
    """
    This function takes in a boolean mask of good and bad indices and
    determines spans of consecutive good indices. It combines two neighboring spans
    with a separation of less than inds_to_span and treats them as a single good span.

    Parameters
    ----------
    bad_inds_mask : boolean mask
        A boolean mask where True is a bad index and False is a good index.
    inds_to_span : int, default 50
        This indicates how many indices between two good spans should
        be bridged to form a single good span.
        For instance if span A is (1500, 2350) and span B is (2370, 3700),
        then span A and span B would be combined into span A (1500, 3700)
        since one would want to identify potential jumps in the space in between the original A and B.

    Returns
    -------
    good_spans : list
        List of spans of good indices, unmodified.
    modified_spans : list
        spans that are amended to bridge up to inds_to_span consecutive bad indices
    """
    good_spans = get_consecutive_inds(np.arange(len(bad_inds_mask))[~bad_inds_mask])
    if len(good_spans) > 1:
        modified_spans = []
        for (start1, stop1), (start2, stop2) in zip(good_spans[:-1], good_spans[1:]):
            check_existing = [
                entry
                for entry in modified_spans
                if start1 in range(entry[0] - inds_to_span, entry[1] + inds_to_span)
            ]
            if len(check_existing) > 0:
                modify_ind = modified_spans.index(check_existing[0])
                if (start2 - stop1) <= inds_to_span:
                    modified_spans[modify_ind] = (check_existing[0][0], stop2)
                else:
                    modified_spans[modify_ind] = (check_existing[0][0], stop1)
                    modified_spans.append((start2, stop2))
                continue
            if (start2 - stop1) <= inds_to_span:
                modified_spans.append((start1, stop2))
            else:
                modified_spans.append((start1, stop1))
                modified_spans.append((start2, stop2))

    return good_spans, modified_spans


def get_consecutive_inds(indices):
    """_summary_

    Parameters
    ----------
    indices : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    span_inds = []
    # Get spans of consecutive indices that jump
    for k, g in groupby(enumerate(indices), lambda x: x[1] - x[0]):
        group = list(map(itemgetter(1), g))
        span_inds.append((group[0], group[-1]))
    return span_inds


def span_length(x):
    return x[-1] - x[0]


def get_subthresh_inds(dlc_df: pd.DataFrame, likelihood_thresh: float):
    df_filter = dlc_df["likelihood"] < likelihood_thresh
    sub_thresh_inds = np.where(~np.isnan(dlc_df["likelihood"].where(df_filter)))[0]
    nand_inds = np.where(np.isnan(dlc_df["x"]))[0]
    all_nan_inds = list(set(sub_thresh_inds).union(set(nand_inds)))
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

import datajoint as dj
import numpy as np
import pandas as pd
import pynwb

from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.position.v1.dlc_utils import (
    _key_to_smooth_func_dict,
    get_span_start_stop,
    interp_pos,
    validate_option,
    validate_smooth_params,
)
from spyglass.utils.dj_mixin import SpyglassMixin

from .position_dlc_pose_estimation import DLCPoseEstimation

schema = dj.schema("position_v1_dlc_position")


@schema
class DLCSmoothInterpParams(SpyglassMixin, dj.Manual):
    """
    Parameters for extracting the smoothed head position.

    Attributes
    ----------
    interpolate : bool, default True
        whether to interpolate over NaN spans
    smooth : bool, default True
        whether to smooth the dataset
    smoothing_params : dict
        smoothing_duration : float, default 0.05
            number of frames to smooth over: sampling_rate*smoothing_duration = num_frames
    interp_params : dict
        max_cm_to_interp : int, default 20
            maximum distance between high likelihood points on either side of a NaN span
            to interpolate over
    likelihood_thresh : float, default 0.95
        likelihood below which to NaN and interpolate over
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
            "smooth": True,
            "smoothing_params": {
                "smoothing_duration": 0.05,
                "smooth_method": "moving_avg",
            },
            "interpolate": True,
            "likelihood_thresh": 0.95,
            "interp_params": {"max_cm_to_interp": 15},
            "max_cm_between_pts": 20,
            # This is for use when finding "good spans" and is how many indices
            # to bridge in between good spans see inds_to_span in get_good_spans
            "num_inds_to_span": 20,
        }
        cls.insert1(
            {"dlc_si_params_name": "default", "params": default_params},
            **kwargs,
        )

    @classmethod
    def insert_nan_params(cls, **kwargs):
        nan_params = {
            "smooth": False,
            "interpolate": False,
            "likelihood_thresh": 0.95,
            "max_cm_between_pts": 20,
            "num_inds_to_span": 20,
        }
        cls.insert1(
            {"dlc_si_params_name": "just_nan", "params": nan_params}, **kwargs
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

    @classmethod
    def get_nan_params(cls):
        query = cls & {"dlc_si_params_name": "just_nan"}
        if not len(query) > 0:
            cls().insert_nan_params(skip_duplicates=True)
            nan_params = (cls & {"dlc_si_params_name": "just_nan"}).fetch1()
        else:
            nan_params = query.fetch1()
        return nan_params

    @staticmethod
    def get_available_methods():
        return _key_to_smooth_func_dict.keys()

    def insert1(self, key, **kwargs):
        params = key.get("params")
        if not isinstance(params, dict):
            raise KeyError("'params' must be a dict in key")

        validate_option(
            option=params.get("max_cm_between_pts"), name="max_cm_between_pts"
        )
        validate_smooth_params(params)

        validate_option(
            params.get("likelihood_thresh"),
            name="likelihood_thresh",
            types=(float),
            val_range=(0, 1),
        )

        super().insert1(key, **kwargs)


@schema
class DLCSmoothInterpSelection(SpyglassMixin, dj.Manual):
    definition = """
    -> DLCPoseEstimation.BodyPart
    -> DLCSmoothInterpParams
    ---

    """


@schema
class DLCSmoothInterp(SpyglassMixin, dj.Computed):
    """
    Interpolates across low likelihood periods and smooths the position
    Can take a few minutes.
    """

    definition = """
    -> DLCSmoothInterpSelection
    ---
    -> AnalysisNwbfile
    dlc_smooth_interp_position_object_id : varchar(80)
    dlc_smooth_interp_info_object_id : varchar(80)
    """

    def make(self, key):
        from .dlc_utils import OutputLogger, infer_output_dir

        METERS_PER_CM = 0.01

        output_dir = infer_output_dir(key=key, makedir=False)
        with OutputLogger(
            name=f"{key['nwb_file_name']}_{key['epoch']}_{key['dlc_model_name']}_log",
            path=f"{output_dir.as_posix()}/log.log",
            print_console=False,
        ) as logger:
            logger.logger.info("-----------------------")
            idx = pd.IndexSlice
            # Get labels to smooth from Parameters table
            params = (DLCSmoothInterpParams() & key).fetch1("params")
            # Get DLC output dataframe
            logger.logger.info("fetching Pose Estimation Dataframe")
            dlc_df = (DLCPoseEstimation.BodyPart() & key).fetch1_dataframe()
            dt = np.median(np.diff(dlc_df.index.to_numpy()))
            sampling_rate = 1 / dt
            logger.logger.info("Identifying indices to NaN")
            df_w_nans, bad_inds = nan_inds(
                dlc_df.copy(),
                params["max_cm_between_pts"],
                likelihood_thresh=params.pop("likelihood_thresh"),
                inds_to_span=params["num_inds_to_span"],
            )

            nan_spans = get_span_start_stop(np.where(bad_inds)[0])
            if params["interpolate"]:
                logger.logger.info("interpolating across low likelihood times")
                interp_df = interp_pos(
                    df_w_nans.copy(), nan_spans, **params["interp_params"]
                )
            else:
                interp_df = df_w_nans.copy()
                logger.logger.info("skipping interpolation")
            if params["smooth"]:
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
            else:
                smooth_df = interp_df.copy()
                logger.logger.info("skipping smoothing")
            final_df = smooth_df.drop(["likelihood"], axis=1)
            final_df = final_df.rename_axis("time").reset_index()
            position_nwb_data = (
                (DLCPoseEstimation.BodyPart() & key)
                .fetch_nwb()[0]["dlc_pose_estimation_position"]
                .get_spatial_series()
            )
            key["analysis_file_name"] = AnalysisNwbfile().create(
                key["nwb_file_name"]
            )
            # Add dataframe to AnalysisNwbfile
            nwb_analysis_file = AnalysisNwbfile()
            position = pynwb.behavior.Position()
            video_frame_ind = pynwb.behavior.BehavioralTimeSeries()
            logger.logger.info("Creating NWB objects")
            position.create_spatial_series(
                name="position",
                timestamps=final_df.time.to_numpy(),
                conversion=METERS_PER_CM,
                data=final_df.loc[:, idx[("x", "y")]].to_numpy(),
                reference_frame=position_nwb_data.reference_frame,
                comments=position_nwb_data.comments,
                description="x_position, y_position",
            )
            video_frame_ind.create_timeseries(
                name="video_frame_ind",
                timestamps=final_df.time.to_numpy(),
                data=final_df.loc[:, idx["video_frame_ind"]].to_numpy(),
                unit="index",
                comments="no comments",
                description="video_frame_ind",
            )
            key["dlc_smooth_interp_position_object_id"] = (
                nwb_analysis_file.add_nwb_object(
                    analysis_file_name=key["analysis_file_name"],
                    nwb_object=position,
                )
            )
            key["dlc_smooth_interp_info_object_id"] = (
                nwb_analysis_file.add_nwb_object(
                    analysis_file_name=key["analysis_file_name"],
                    nwb_object=video_frame_ind,
                )
            )
            nwb_analysis_file.add(
                nwb_file_name=key["nwb_file_name"],
                analysis_file_name=key["analysis_file_name"],
            )
            self.insert1(key)
            logger.logger.info("inserted entry into DLCSmoothInterp")

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


def nan_inds(
    dlc_df: pd.DataFrame,
    max_dist_between,
    likelihood_thresh: float,
    inds_to_span: int,
):
    idx = pd.IndexSlice

    # Could either NaN sub-likelihood threshold inds here and then not consider
    # in jumping... OR just keep in back pocket when checking jumps against
    # last good point

    subthresh_inds = get_subthresh_inds(
        dlc_df, likelihood_thresh=likelihood_thresh
    )
    df_subthresh_indices = dlc_df.index[subthresh_inds]
    dlc_df.loc[idx[df_subthresh_indices], idx[("x", "y")]] = np.nan

    # To further determine which indices are the original point and which are
    # jump points. There could be a more efficient method of doing this screen
    # inds for jumps to baseline

    subthresh_inds_mask = np.zeros(len(dlc_df), dtype=bool)
    subthresh_inds_mask[subthresh_inds] = True
    jump_inds_mask = np.zeros(len(dlc_df), dtype=bool)
    _, good_spans = get_good_spans(
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
            good_x, good_y = dlc_df.loc[
                idx[dlc_df.index[last_good_ind]], ["x", "y"]
            ]
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
            good_x, good_y = dlc_df.loc[
                idx[dlc_df.index[last_good_ind]], ["x", "y"]
            ]
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
    determines spans of consecutive good indices. It combines two neighboring
    spans with a separation of less than inds_to_span and treats them as a
    single good span.

    Parameters
    ----------
    bad_inds_mask : boolean mask
        A boolean mask where True is a bad index and False is a good index.
    inds_to_span : int, default 50
        This indicates how many indices between two good spans should
        be bridged to form a single good span.
        For instance if span A is (1500, 2350) and span B is (2370, 3700),
        then span A and span B would be combined into span A (1500, 3700)
        since one would want to identify potential jumps in the space in between
        the original A and B.

    Returns
    -------
    good_spans : list
        List of spans of good indices, unmodified.
    modified_spans : list
        spans that are amended to bridge up to inds_to_span consecutive bad indices
    """
    good_spans = get_span_start_stop(
        np.arange(len(bad_inds_mask))[~bad_inds_mask]
    )
    if len(good_spans) > 1:
        modified_spans = []
        for (start1, stop1), (start2, stop2) in zip(
            good_spans[:-1], good_spans[1:]
        ):
            check_existing = [
                entry
                for entry in modified_spans
                if start1
                in range(entry[0] - inds_to_span, entry[1] + inds_to_span)
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
    else:
        return None, good_spans


def span_length(x):
    return x[-1] - x[0]


def get_subthresh_inds(dlc_df: pd.DataFrame, likelihood_thresh: float):
    df_filter = dlc_df["likelihood"] < likelihood_thresh
    sub_thresh_inds = np.where(
        ~np.isnan(dlc_df["likelihood"].where(df_filter))
    )[0]
    nand_inds = np.where(np.isnan(dlc_df["x"]))[0]
    all_nan_inds = list(set(sub_thresh_inds).union(set(nand_inds)))
    all_nan_inds.sort()
    # TODO: add option to return sub_thresh_percent
    # sub_thresh_percent = (len(sub_thresh_inds) / len(dlc_df)) * 100
    return all_nan_inds

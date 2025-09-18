from pathlib import Path
from typing import Tuple, Union

import datajoint as dj
import numpy as np
import pandas as pd
import pynwb

from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.position.v1.dlc_utils import (
    _key_to_smooth_func_dict,
    file_log,
    get_span_start_stop,
    infer_output_dir,
    interp_pos,
    validate_option,
    validate_smooth_params,
)
from spyglass.position.v1.position_dlc_pose_estimation import DLCPoseEstimation
from spyglass.settings import test_mode
from spyglass.utils import SpyglassMixin, logger

schema = dj.schema("position_v1_dlc_position")


@schema
class DLCSmoothInterpParams(SpyglassMixin, dj.Manual):
    """
    Parameters for extracting the smoothed head position.

    Attributes
    ----------
    dlc_si_params_name : str
        Name for this set of parameters
    params : dict
        Dictionary of parameters, including...
        interpolate : bool, default True
            whether to interpolate over NaN spans
        smooth : bool, default True
            whether to smooth the dataset
        smoothing_params : dict
            smoothing_duration : float, default 0.05
                number of frames to smooth over:
                sampling_rate*smoothing_duration = num_frames
        interp_params : dict
            max_cm_to_interp : int, default 20
                maximum distance between high likelihood points on either side
                of a NaN span to interpolate over
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
        """Insert parameters for smoothing and interpolation."""
        cls.insert1(
            {"dlc_si_params_name": params_name, "params": params},
            **kwargs,
        )

    @classmethod
    def insert_default(cls, **kwargs):
        """Insert the default set of parameters."""

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
    def insert_nan_params(cls, **kwargs) -> None:
        """Insert parameters that only NaN the data."""
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
    def get_default(cls) -> dict:
        """Return the default set of parameters for smoothing calculation."""
        query = cls & {"dlc_si_params_name": "default"}
        if not len(query) > 0:
            cls().insert_default(skip_duplicates=True)
            default = (cls & {"dlc_si_params_name": "default"}).fetch1()
        else:
            default = query.fetch1()
        return default

    @classmethod
    def get_nan_params(cls) -> dict:
        """Return the parameters that NaN the data."""
        query = cls & {"dlc_si_params_name": "just_nan"}
        if not len(query) > 0:
            cls().insert_nan_params(skip_duplicates=True)
            nan_params = (cls & {"dlc_si_params_name": "just_nan"}).fetch1()
        else:
            nan_params = query.fetch1()
        return nan_params

    @staticmethod
    def get_available_methods():
        """Return the available smoothing methods."""
        return _key_to_smooth_func_dict.keys()

    def insert1(self, key, **kwargs):
        """Override insert1 to validate params."""
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
            types=float,
            val_range=(0, 1),
        )

        super().insert1(key, **kwargs)


@schema
class DLCSmoothInterpSelection(SpyglassMixin, dj.Manual):
    definition = """
    -> DLCPoseEstimation.BodyPart
    -> DLCSmoothInterpParams
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
    log_path = None

    def make(self, key):
        """Populate the DLCSmoothInterp table.

        Uses a decorator to log the output to a file.

        1. Fetches the DLC output dataframe from DLCPoseEstimation
        2. NaNs low likelihood points and interpolates across them
        3. Optionally smooths and interpolates the data
        4. Create position and video frame index NWB objects
        5. Add NWB objects to AnalysisNwbfile
        6. Insert the key into DLCSmoothInterp.
        """
        self.log_path = (
            Path(infer_output_dir(key=key, makedir=False)) / "log.log"
        )
        self._logged_make(key)
        logger.info("inserted entry into DLCSmoothInterp")

    @file_log(logger, console=False)
    def _logged_make(self, key):
        METERS_PER_CM = 0.01

        logger.info("-----------------------")
        idx = pd.IndexSlice
        # Get labels to smooth from Parameters table
        params = (DLCSmoothInterpParams() & key).fetch1("params")
        # Get DLC output dataframe
        logger.info("fetching Pose Estimation Dataframe")

        bp_key = key.copy()
        if test_mode:  # during testing, analysis_file not in BodyPart table
            bp_key.pop("analysis_file_name", None)

        dlc_df = (DLCPoseEstimation.BodyPart() & bp_key).fetch1_dataframe()
        dt = np.median(np.diff(dlc_df.index.to_numpy()))
        logger.info("Identifying indices to NaN")
        likelihood_thresh = params.pop("likelihood_thresh")
        df_w_nans, bad_inds = nan_inds(
            dlc_df.copy(),
            max_dist_between=params["max_cm_between_pts"],
            likelihood_thresh=likelihood_thresh,
            inds_to_span=params["num_inds_to_span"],
        )

        nan_spans = get_span_start_stop(np.where(bad_inds)[0])

        if params.get("interpolate"):
            interp_params = params.get("interp_params", dict())
            logger.info("interpolating across low likelihood times")
            interp_df = interp_pos(df_w_nans.copy(), nan_spans, **interp_params)
        else:
            interp_df = df_w_nans.copy()
            logger.info("skipping interpolation")

        if params.get("smooth"):
            smooth_params = params.get("smoothing_params")
            smooth_method = smooth_params.get("smooth_method")
            smooth_func = _key_to_smooth_func_dict[smooth_method]

            # Handle duplicate smoothing_duration key
            smooth_dur = smooth_params.get("smoothing_duration") or params[
                "smoothing_params"
            ].pop("smoothing_duration", None)

            dt = np.median(np.diff(dlc_df.index.to_numpy()))
            logger.info(f"Smoothing using method: {smooth_method}")
            smooth_df = smooth_func(
                interp_df, smoothing_duration=smooth_dur, sampling_rate=1 / dt
            )
        else:
            smooth_df = interp_df.copy()
            logger.info("skipping smoothing")

        final_df = smooth_df.drop(["likelihood"], axis=1)
        final_df = final_df.rename_axis("time").reset_index()
        position_nwb_data = (
            (DLCPoseEstimation.BodyPart() & bp_key)
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
        logger.info("Creating NWB objects")
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


def nan_inds(
    dlc_df: pd.DataFrame,
    max_dist_between,
    likelihood_thresh: float,
    inds_to_span: int,
):
    """Replace low likelihood points with NaNs and interpolate over them."""
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

    if len(good_spans) == 0:
        # Prevents ref before assignment error of mask on return
        # TODO: instead of raise, insert empty dataframe
        raise ValueError("No good spans found in the data")

    for span in good_spans[::-1]:
        if np.sum(np.isnan(dlc_df.iloc[span[0] : span[-1]].x)) > 0:
            nan_mask = np.isnan(dlc_df.iloc[span[0] : span[-1]].x)
            good_start = np.arange(span[0], span[1])[~nan_mask]
            start_point = good_start[int(len(good_start) // 2)]
        else:
            start_point = span[0] + int(span_length(span) // 2)

        for ind in range(start_point, span[0], -1):
            if subthresh_inds_mask[ind]:
                continue  # pragma: no cover
            previous_good_inds = np.where(
                np.logical_and(
                    ~np.isnan(dlc_df.iloc[ind + 1 : start_point].x),
                    ~jump_inds_mask[ind + 1 : start_point],
                    ~subthresh_inds_mask[ind + 1 : start_point],
                )
            )[0]
            last_good_ind = (
                ind + 1 + np.min(previous_good_inds)
                if len(previous_good_inds) > 0
                else start_point
            )
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
                continue  # pragma: no cover
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
    good = get_span_start_stop(np.arange(len(bad_inds_mask))[~bad_inds_mask])

    if len(good) < 1:
        return None, good
    elif len(good) == 1:  # if all good, no need to modify
        return good, good

    modified_spans = []
    for (start1, stop1), (start2, stop2) in zip(good[:-1], good[1:]):
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
    return good, modified_spans


def span_length(x):
    """Return the length of a span."""
    return x[-1] - x[0]


def get_subthresh_inds(
    dlc_df: pd.DataFrame, likelihood_thresh: float, ret_sub_thresh: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
    """Return indices of subthresh points.

    Parameters
    ----------
    dlc_df : pd.DataFrame
        Dataframe containing the DLC data
    likelihood_thresh : float
        Likelihood threshold for subthresh points
    ret_sub_thresh : bool, default False
        Whether to return the percentage of subthresh points

    Returns
    -------
    all_nan_inds : np.ndarray
        Indices of subthresh points
    sub_thresh_percent: float, optional
        Percentage of subthresh points
    """
    df_filter = dlc_df["likelihood"] < likelihood_thresh
    sub_thresh_inds = np.where(
        ~np.isnan(dlc_df["likelihood"].where(df_filter))
    )[0]
    nand_inds = np.where(np.isnan(dlc_df["x"]))[0]
    all_nan_inds = list(set(sub_thresh_inds).union(set(nand_inds)))
    all_nan_inds.sort()
    sub_thresh_percent = (len(sub_thresh_inds) / len(dlc_df)) * 100

    if ret_sub_thresh:
        return all_nan_inds, sub_thresh_percent
    return all_nan_inds

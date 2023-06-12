from functools import reduce

import datajoint as dj
import numpy as np
import pandas as pd
import pynwb
from position_tools import get_distance, get_velocity

from ...common.common_behav import RawPosition
from ...common.common_nwbfile import AnalysisNwbfile
from ...utils.dj_helper_fn import fetch_nwb
from .dlc_utils import _key_to_smooth_func_dict, get_span_start_stop, interp_pos
from .position_dlc_cohort import DLCSmoothInterpCohort
from .position_dlc_position import DLCSmoothInterpParams

schema = dj.schema("position_v1_dlc_centroid")


@schema
class DLCCentroidParams(dj.Manual):
    """
    Parameters for calculating the centroid
    """

    # TODO: whether to keep all params in a params dict
    # or break out into individual secondary keys
    definition = """
    dlc_centroid_params_name: varchar(80) # name for this set of parameters
    ---
    params: longblob
    """

    _available_centroid_methods = [
        "four_led_centroid",
        "two_pt_centroid",
        "one_pt_centroid",
    ]
    _four_led_labels = ["greenLED", "redLED_L", "redLED_C", "redLED_R"]
    _two_pt_labels = ["point1", "point2"]

    @classmethod
    def insert_default(cls, **kwargs):
        """
        Inserts default centroid parameters. Assumes 2 LEDs tracked
        """
        params = {
            "centroid_method": "two_pt_centroid",
            "points": {
                "point1": "greenLED",
                "point2": "redLED_C",
            },
            "interpolate": True,
            "interp_params": {"max_cm_to_interp": 15},
            "smooth": True,
            "smoothing_params": {
                "smoothing_duration": 0.05,
                "smooth_method": "moving_avg",
            },
            "max_LED_separation": 12,
            "speed_smoothing_std_dev": 0.100,
        }
        cls.insert1(
            {"dlc_centroid_params_name": "default", "params": params}, **kwargs
        )

    @classmethod
    def get_default(cls):
        query = cls & {"dlc_centroid_params_name": "default"}
        if not len(query) > 0:
            cls().insert_default(skip_duplicates=True)
            default = (cls & {"dlc_centroid_params_name": "default"}).fetch1()
        else:
            default = query.fetch1()
        return default

    def insert1(self, key, **kwargs):
        """
        Check provided parameter dictionary to make sure
        it contains all necessary items
        """
        params = key["params"]
        if "centroid_method" in params:
            if params["centroid_method"] in self._available_centroid_methods:
                if params["centroid_method"] == "four_led_centroid":
                    if any(
                        x not in self._four_led_labels for x in params["points"]
                    ):
                        raise KeyError(
                            f"Please make sure to specify all necessary labels: "
                            f"{self._four_led_labels} "
                            f"if using the 'four_led_centroid' method"
                        )
                elif params["centroid_method"] == "two_pt_centroid":
                    if any(
                        x not in self._two_pt_labels for x in params["points"]
                    ):
                        raise KeyError(
                            f"Please make sure to specify all necessary labels: "
                            f"{self._two_pt_labels} "
                            f"if using the 'two_pt_centroid' method"
                        )
                elif params["centroid_method"] == "one_pt_centroid":
                    if "point1" not in params["points"]:
                        raise KeyError(
                            "Please make sure to specify the necessary label: "
                            "'point1' "
                            "if using the 'one_pt_centroid' method"
                        )
                else:
                    raise Exception("This shouldn't happen lol oops")
            else:
                raise ValueError(
                    f"The given 'centroid_method': {params['centroid_method']} "
                    f"is not in the available methods: "
                    f"{self._available_centroid_methods}"
                )
        else:
            raise KeyError(
                "'centroid_method' needs to be provided as a parameter"
            )

        if "max_LED_separation" in params:
            if not isinstance(params["max_LED_separation"], (int, float)):
                raise TypeError(
                    f"parameter 'max_LED_separation' is type: "
                    f"{type(params['max_LED_separation'])}, "
                    f"it should be one of type (float, int)"
                )
        if "smooth" in params:
            if params["smooth"]:
                if "smoothing_params" in params:
                    if "smooth_method" in params["smoothing_params"]:
                        smooth_method = params["smoothing_params"][
                            "smooth_method"
                        ]
                        if smooth_method not in _key_to_smooth_func_dict:
                            raise KeyError(
                                f"smooth_method: {smooth_method} not an available method."
                            )
                    if not "smoothing_duration" in params["smoothing_params"]:
                        raise KeyError(
                            "smoothing_duration must be passed as a smoothing_params within key['params']"
                        )
                    else:
                        assert isinstance(
                            params["smoothing_params"]["smoothing_duration"],
                            (float, int),
                        ), "smoothing_duration must be a float or int"
                else:
                    raise ValueError("smoothing_params not in key['params']")

        super().insert1(key, **kwargs)


@schema
class DLCCentroidSelection(dj.Manual):
    """
    Table to pair a cohort of bodypart entries with
    the parameters for calculating their centroid
    """

    definition = """
    -> DLCSmoothInterpCohort
    -> DLCCentroidParams
    ---
    """


@schema
class DLCCentroid(dj.Computed):
    """
    Table to calculate the centroid of a group of bodyparts
    """

    definition = """
    -> DLCCentroidSelection
    ---
    -> AnalysisNwbfile
    dlc_position_object_id : varchar(80)
    dlc_velocity_object_id : varchar(80)
    """

    def make(self, key):
        from .dlc_utils import OutputLogger, infer_output_dir

        idx = pd.IndexSlice
        output_dir = infer_output_dir(key=key, makedir=False)
        with OutputLogger(
            name=f"{key['nwb_file_name']}_{key['epoch']}_{key['dlc_model_name']}_log",
            path=f"{output_dir.as_posix()}/log.log",
            print_console=False,
        ) as logger:
            logger.logger.info("-----------------------")
            logger.logger.info("Centroid Calculation")
            # Get labels to smooth from Parameters table
            cohort_entries = DLCSmoothInterpCohort.BodyPart & key
            params = (DLCCentroidParams() & key).fetch1("params")
            centroid_method = params.pop("centroid_method")
            bodyparts_avail = cohort_entries.fetch("bodypart")
            speed_smoothing_std_dev = params.pop("speed_smoothing_std_dev")
            # TODO, generalize key naming
            if centroid_method == "four_led_centroid":
                centroid_func = _key_to_func_dict[centroid_method]
                if "greenLED" in params["points"]:
                    assert (
                        params["points"]["greenLED"] in bodyparts_avail
                    ), f'{params["points"]["greenLED"]} not a bodypart used in this model'
                else:
                    raise ValueError(
                        "A green led needs to be specified for the 4 led centroid method"
                    )
                if "redLED_L" in params["points"]:
                    assert (
                        params["points"]["redLED_L"] in bodyparts_avail
                    ), f'{params["points"]["redLED_L"]} not a bodypart used in this model'
                else:
                    raise ValueError(
                        "A left red led needs to be specified for the 4 led centroid method"
                    )
                if "redLED_C" in params["points"]:
                    assert (
                        params["points"]["redLED_C"] in bodyparts_avail
                    ), f'{params["points"]["redLED_C"]} not a bodypart used in this model'
                else:
                    raise ValueError(
                        "A center red led needs to be specified for the 4 led centroid method"
                    )
                if "redLED_R" in params["points"]:
                    assert (
                        params["points"]["redLED_R"] in bodyparts_avail
                    ), f'{params["points"]["redLED_R"]} not a bodypart used in this model'
                else:
                    raise ValueError(
                        "A right red led needs to be specified for the 4 led centroid method"
                    )
                bodyparts_to_use = [
                    params["points"]["greenLED"],
                    params["points"]["redLED_L"],
                    params["points"]["redLED_C"],
                    params["points"]["redLED_R"],
                ]

            elif centroid_method == "two_pt_centroid":
                centroid_func = _key_to_func_dict[centroid_method]
                if "point1" in params["points"]:
                    assert (
                        params["points"]["point1"] in bodyparts_avail
                    ), f'{params["points"]["point1"]} not a bodypart used in this model'
                else:
                    raise ValueError(
                        "point1 needs to be specified for the 2 pt centroid method"
                    )
                if "point2" in params["points"]:
                    assert (
                        params["points"]["point2"] in bodyparts_avail
                    ), f'{params["points"]["point2"]} not a bodypart used in this model'
                else:
                    raise ValueError(
                        "point2 needs to be specified for the 2 pt centroid method"
                    )
                bodyparts_to_use = [
                    params["points"]["point1"],
                    params["points"]["point2"],
                ]

            elif centroid_method == "one_pt_centroid":
                centroid_func = _key_to_func_dict[centroid_method]
                if "point1" in params["points"]:
                    assert (
                        params["points"]["point1"] in bodyparts_avail
                    ), f'{params["points"]["point1"]} not a bodypart used in this model'
                else:
                    raise ValueError(
                        "point1 needs to be specified for the 1 pt centroid method"
                    )
                bodyparts_to_use = [params["points"]["point1"]]

            else:
                raise ValueError("Please specify a centroid method to use.")
            pos_df = pd.concat(
                {
                    bodypart: (
                        DLCSmoothInterpCohort.BodyPart
                        & {**key, **{"bodypart": bodypart}}
                    ).fetch1_dataframe()
                    for bodypart in bodyparts_to_use
                },
                axis=1,
            )
            dt = np.median(np.diff(pos_df.index.to_numpy()))
            sampling_rate = 1 / dt
            logger.logger.info(
                "Calculating centroid with %s", str(centroid_method)
            )
            centroid = centroid_func(pos_df, **params)
            centroid_df = pd.DataFrame(
                centroid,
                columns=["x", "y"],
                index=pos_df.index.to_numpy(),
            )
            if params["interpolate"]:
                if np.any(np.isnan(centroid)):
                    logger.logger.info("interpolating over NaNs")
                    nan_inds = (
                        pd.isnull(centroid_df.loc[:, idx[("x", "y")]])
                        .any(axis=1)
                        .to_numpy()
                        .nonzero()[0]
                    )
                    nan_spans = get_span_start_stop(nan_inds)
                    interp_df = interp_pos(
                        centroid_df.copy(), nan_spans, **params["interp_params"]
                    )
                else:
                    logger.logger.info("no NaNs to interpolate over")
                    interp_df = centroid_df.copy()
            else:
                interp_df = centroid_df.copy()
            if params["smooth"]:
                if "smoothing_duration" in params["smoothing_params"]:
                    smoothing_duration = params["smoothing_params"].pop(
                        "smoothing_duration"
                    )
                    dt = np.median(np.diff(pos_df.index.to_numpy()))
                    sampling_rate = 1 / dt
                    logger.logger.info("smoothing position")
                    smooth_func = _key_to_smooth_func_dict[
                        params["smoothing_params"]["smooth_method"]
                    ]
                    logger.logger.info(
                        "Smoothing using method: %s",
                        str(params["smoothing_params"]["smooth_method"]),
                    )
                    final_df = smooth_func(
                        interp_df,
                        smoothing_duration=smoothing_duration,
                        sampling_rate=sampling_rate,
                        **params["smoothing_params"],
                    )
                else:
                    raise KeyError(
                        "smoothing_duration needs to be passed within smoothing_params"
                    )
            else:
                final_df = interp_df.copy()
            logger.logger.info("getting velocity")
            velocity = get_velocity(
                final_df.loc[:, idx[("x", "y")]].to_numpy(),
                time=pos_df.index.to_numpy(),
                sigma=speed_smoothing_std_dev,
                sampling_frequency=sampling_rate,
            )  # cm/s
            speed = np.sqrt(np.sum(velocity**2, axis=1))  # cm/s
            # Create dataframe
            velocity_df = pd.DataFrame(
                np.concatenate((velocity, speed[:, np.newaxis]), axis=1),
                columns=["velocity_x", "velocity_y", "speed"],
                index=pos_df.index.to_numpy(),
            )
            total_nan = np.sum(
                final_df.loc[:, idx[("x", "y")]].isna().any(axis=1)
            )
            pretrack_nan = np.sum(
                final_df.iloc[:1000].loc[:, idx[("x", "y")]].isna().any(axis=1)
            )
            logger.logger.info("total NaNs in centroid dataset: %d", total_nan)
            logger.logger.info(
                "NaNs in centroid dataset before ind 1000: %d", pretrack_nan
            )
            position = pynwb.behavior.Position()
            velocity = pynwb.behavior.BehavioralTimeSeries()
            spatial_series = (RawPosition() & key).fetch_nwb()[0][
                "raw_position"
            ]
            METERS_PER_CM = 0.01
            position.create_spatial_series(
                name="position",
                timestamps=final_df.index.to_numpy(),
                conversion=METERS_PER_CM,
                data=final_df.loc[:, idx[("x", "y")]].to_numpy(),
                reference_frame=spatial_series.reference_frame,
                comments=spatial_series.comments,
                description="x_position, y_position",
            )
            velocity.create_timeseries(
                name="velocity",
                timestamps=velocity_df.index.to_numpy(),
                conversion=METERS_PER_CM,
                unit="m/s",
                data=velocity_df.loc[
                    :, idx[("velocity_x", "velocity_y", "speed")]
                ].to_numpy(),
                comments=spatial_series.comments,
                description="x_velocity, y_velocity, speed",
            )
            velocity.create_timeseries(
                name="video_frame_ind",
                unit="index",
                timestamps=final_df.index.to_numpy(),
                data=pos_df[
                    pos_df.columns.levels[0][0]
                ].video_frame_ind.to_numpy(),
                description="video_frame_ind",
                comments="no comments",
            )
            # Add to Analysis NWB file
            key["analysis_file_name"] = AnalysisNwbfile().create(
                key["nwb_file_name"]
            )
            nwb_analysis_file = AnalysisNwbfile()
            key["dlc_position_object_id"] = nwb_analysis_file.add_nwb_object(
                key["analysis_file_name"], position
            )
            key["dlc_velocity_object_id"] = nwb_analysis_file.add_nwb_object(
                key["analysis_file_name"], velocity
            )

            nwb_analysis_file.add(
                nwb_file_name=key["nwb_file_name"],
                analysis_file_name=key["analysis_file_name"],
            )
            self.insert1(key)
            logger.logger.info("inserted entry into DLCCentroid")

    def fetch_nwb(self, *attrs, **kwargs):
        return fetch_nwb(
            self, (AnalysisNwbfile, "analysis_file_abs_path"), *attrs, **kwargs
        )

    def fetch1_dataframe(self):
        nwb_data = self.fetch_nwb()[0]
        index = pd.Index(
            np.asarray(
                nwb_data["dlc_position"].get_spatial_series().timestamps
            ),
            name="time",
        )
        COLUMNS = [
            "video_frame_ind",
            "position_x",
            "position_y",
            "velocity_x",
            "velocity_y",
            "speed",
        ]
        return pd.DataFrame(
            np.concatenate(
                (
                    np.asarray(
                        nwb_data["dlc_velocity"]
                        .time_series["video_frame_ind"]
                        .data,
                        dtype=int,
                    )[:, np.newaxis],
                    np.asarray(
                        nwb_data["dlc_position"].get_spatial_series().data
                    ),
                    np.asarray(
                        nwb_data["dlc_velocity"].time_series["velocity"].data
                    ),
                ),
                axis=1,
            ),
            columns=COLUMNS,
            index=index,
        )


def four_led_centroid(pos_df: pd.DataFrame, **params):
    """
    Determines the centroid of 4 LEDS on an implant LED ring.
    Assumed to be the Green LED, and 3 red LEDs called: redLED_C, redLED_L, redLED_R
    By default, uses (greenled + redLED_C) / 2 to calculate centroid
    If Green LED is NaN, but red center LED is not,
        then the red center LED is called the centroid
    If green and red center LEDs are NaN, but red left and red right LEDs are not,
        then the centroid is (redLED_L + redLED_R) / 2
    If red center LED is NaN, but the other 3 LEDS are not,
        then the centroid is (greenled + (redLED_L + redLED_R) / 2) / 2
    If red center and left LEDs are NaN, but green and red right LEDs are not,
        then the centroid is (greenled + redLED_R) / 2
    If red center and right LEDs are NaN, but green and red left LEDs are not,
        then the centroid is (greenled + redLED_L) / 2
    If all red LEDs are NaN, but green LED is not,
        then the green LED is called the centroid
    If all LEDs are NaN, then the centroid is NaN

    Parameters
    ----------
    pos_df : pd.DataFrame
        dataframe containing x and y position for each LED of interest,
        index is timestamps. Column names specified by params
    **params : dict
        contains 'greenLED' and 'redLED_C', 'redLED_R', 'redLED_L' keys,
        whose values specify the column names in `pos_df`

    Returns
    -------
    centroid : np.ndarray
        numpy array with shape (n_time, 2)
        centroid[0] is the x coord and centroid[1] is the y coord
    """
    centroid = np.zeros(shape=(len(pos_df), 2))
    idx = pd.IndexSlice
    # TODO: this feels messy, clean-up
    green_led = params["points"].pop("greenLED", None)
    red_led_C = params["points"].pop("redLED_C", None)
    red_led_L = params["points"].pop("redLED_L", None)
    red_led_R = params["points"].pop("redLED_R", None)
    green_nans = pos_df.loc[:, idx[green_led, ("x", "y")]].isna().any(axis=1)
    red_C_nans = pos_df.loc[:, idx[red_led_C, ("x", "y")]].isna().any(axis=1)
    red_L_nans = pos_df.loc[:, idx[red_led_L, ("x", "y")]].isna().any(axis=1)
    red_R_nans = pos_df.loc[:, idx[red_led_R, ("x", "y")]].isna().any(axis=1)
    # TODO: implement checks to make sure not rewriting previously set index in centroid
    # If all given LEDs are not NaN
    dist_between_green_red = get_distance(
        pos_df.loc[:, idx[red_led_C, ("x", "y")]].to_numpy(),
        pos_df.loc[:, idx[green_led, ("x", "y")]].to_numpy(),
    )
    g_c_is_too_separated = (
        dist_between_green_red >= params["max_LED_separation"]
    )
    all_good_mask = reduce(
        np.logical_and,
        (
            ~green_nans,
            ~red_C_nans,
            ~red_L_nans,
            ~red_R_nans,
            ~g_c_is_too_separated,
        ),
    )
    centroid[all_good_mask] = [
        *zip(
            (
                pos_df.loc[idx[all_good_mask], idx[red_led_C, "x"]]
                + pos_df.loc[idx[all_good_mask], idx[green_led, "x"]]
            )
            / 2,
            (
                pos_df.loc[idx[all_good_mask], idx[red_led_C, "y"]]
                + pos_df.loc[idx[all_good_mask], idx[green_led, "y"]]
            )
            / 2,
        )
    ]
    # If green LED and red center LED are both not NaN
    green_red_C = np.logical_and(
        ~green_nans, ~red_C_nans, ~g_c_is_too_separated
    )
    if np.sum(green_red_C) > 0:
        centroid[green_red_C] = [
            *zip(
                (
                    pos_df.loc[idx[green_red_C], idx[red_led_C, "x"]]
                    + pos_df.loc[idx[green_red_C], idx[green_led, "x"]]
                )
                / 2,
                (
                    pos_df.loc[idx[green_red_C], idx[red_led_C, "y"]]
                    + pos_df.loc[idx[green_red_C], idx[green_led, "y"]]
                )
                / 2,
            )
        ]
    # If all given LEDs are NaN
    all_bad_mask = reduce(
        np.logical_and, (green_nans, red_C_nans, red_L_nans, red_R_nans)
    )
    centroid[all_bad_mask, :] = np.nan
    # If green LED is NaN, but red center LED is not
    no_green_red_C = np.logical_and(green_nans, ~red_C_nans)
    if np.sum(no_green_red_C) > 0:
        centroid[no_green_red_C] = [
            *zip(
                pos_df.loc[idx[no_green_red_C], idx[red_led_C, "x"]],
                pos_df.loc[idx[no_green_red_C], idx[red_led_C, "y"]],
            )
        ]
    # If green and red center LEDs are NaN, but red left and red right LEDs are not
    dist_between_left_right = get_distance(
        pos_df.loc[:, idx[red_led_L, ("x", "y")]].to_numpy(),
        pos_df.loc[:, idx[red_led_R, ("x", "y")]].to_numpy(),
    )
    l_r_is_too_separated = (
        dist_between_left_right >= params["max_LED_separation"]
    )
    no_green_no_red_C_red_L_red_R = reduce(
        np.logical_and,
        (
            green_nans,
            red_C_nans,
            ~red_L_nans,
            ~red_R_nans,
            ~l_r_is_too_separated,
        ),
    )
    if np.sum(no_green_no_red_C_red_L_red_R) > 0:
        centroid[no_green_no_red_C_red_L_red_R] = [
            *zip(
                (
                    pos_df.loc[
                        idx[no_green_no_red_C_red_L_red_R], idx[red_led_L, "x"]
                    ]
                    + pos_df.loc[
                        idx[no_green_no_red_C_red_L_red_R], idx[red_led_R, "x"]
                    ]
                )
                / 2,
                (
                    pos_df.loc[
                        idx[no_green_no_red_C_red_L_red_R], idx[red_led_L, "y"]
                    ]
                    + pos_df.loc[
                        idx[no_green_no_red_C_red_L_red_R], idx[red_led_R, "y"]
                    ]
                )
                / 2,
            )
        ]
    # If red center LED is NaN, but green, red left, and right LEDs are not
    dist_between_left_green = get_distance(
        pos_df.loc[:, idx[red_led_L, ("x", "y")]].to_numpy(),
        pos_df.loc[:, idx[green_led, ("x", "y")]].to_numpy(),
    )
    dist_between_right_green = get_distance(
        pos_df.loc[:, idx[red_led_R, ("x", "y")]].to_numpy(),
        pos_df.loc[:, idx[green_led, ("x", "y")]].to_numpy(),
    )
    l_g_is_too_separated = (
        dist_between_left_green >= params["max_LED_separation"]
    )
    r_g_is_too_separated = (
        dist_between_right_green >= params["max_LED_separation"]
    )
    green_red_L_red_R_no_red_C = reduce(
        np.logical_and,
        (
            ~green_nans,
            red_C_nans,
            ~red_L_nans,
            ~red_R_nans,
            ~l_r_is_too_separated,
            ~l_g_is_too_separated,
            ~r_g_is_too_separated,
        ),
    )
    if np.sum(green_red_L_red_R_no_red_C) > 0:
        midpoint = (
            (
                pos_df.loc[idx[green_red_L_red_R_no_red_C], idx[red_led_L, "x"]]
                + pos_df.loc[
                    idx[green_red_L_red_R_no_red_C], idx[red_led_R, "x"]
                ]
            )
            / 2,
            (
                pos_df.loc[idx[green_red_L_red_R_no_red_C], idx[red_led_L, "y"]]
                + pos_df.loc[
                    idx[green_red_L_red_R_no_red_C], idx[red_led_R, "y"]
                ]
            )
            / 2,
        )
        centroid[green_red_L_red_R_no_red_C] = [
            *zip(
                (
                    midpoint[0]
                    + pos_df.loc[
                        idx[green_red_L_red_R_no_red_C], idx[green_led, "x"]
                    ]
                )
                / 2,
                (
                    midpoint[1]
                    + pos_df.loc[
                        idx[green_red_L_red_R_no_red_C], idx[green_led, "y"]
                    ]
                )
                / 2,
            )
        ]
    # If red center and left LED is NaN, but green and red right LED are not
    green_red_R_no_red_C_no_red_L = reduce(
        np.logical_and,
        (
            ~green_nans,
            red_C_nans,
            red_L_nans,
            ~red_R_nans,
            ~r_g_is_too_separated,
        ),
    )
    if np.sum(green_red_R_no_red_C_no_red_L) > 0:
        centroid[green_red_R_no_red_C_no_red_L] = [
            *zip(
                (
                    pos_df.loc[
                        idx[green_red_R_no_red_C_no_red_L], idx[red_led_R, "x"]
                    ]
                    + pos_df.loc[
                        idx[green_red_R_no_red_C_no_red_L], idx[green_led, "x"]
                    ]
                )
                / 2,
                (
                    pos_df.loc[
                        idx[green_red_R_no_red_C_no_red_L], idx[red_led_R, "y"]
                    ]
                    + pos_df.loc[
                        idx[green_red_R_no_red_C_no_red_L], idx[green_led, "y"]
                    ]
                )
                / 2,
            )
        ]
    # If red center and right LED is NaN, but green and red left LED are not
    green_red_L_no_red_C_no_red_R = reduce(
        np.logical_and,
        (
            ~green_nans,
            red_C_nans,
            ~red_L_nans,
            red_R_nans,
            ~l_g_is_too_separated,
        ),
    )
    if np.sum(green_red_L_no_red_C_no_red_R) > 0:
        centroid[green_red_L_no_red_C_no_red_R] = [
            *zip(
                (
                    pos_df.loc[
                        idx[green_red_L_no_red_C_no_red_R], idx[red_led_L, "x"]
                    ]
                    + pos_df.loc[
                        idx[green_red_L_no_red_C_no_red_R], idx[green_led, "x"]
                    ]
                )
                / 2,
                (
                    pos_df.loc[
                        idx[green_red_L_no_red_C_no_red_R], idx[red_led_L, "y"]
                    ]
                    + pos_df.loc[
                        idx[green_red_L_no_red_C_no_red_R], idx[green_led, "y"]
                    ]
                )
                / 2,
            )
        ]
    # If all LEDS are NaN except red left LED
    red_L_no_green_no_red_C_no_red_R = reduce(
        np.logical_and, (green_nans, red_C_nans, ~red_L_nans, red_R_nans)
    )
    if np.sum(red_L_no_green_no_red_C_no_red_R) > 0:
        centroid[red_L_no_green_no_red_C_no_red_R] = [
            *zip(
                pos_df.loc[
                    idx[red_L_no_green_no_red_C_no_red_R], idx[red_led_L, "x"]
                ],
                pos_df.loc[
                    idx[red_L_no_green_no_red_C_no_red_R], idx[red_led_L, "y"]
                ],
            )
        ]
    # If all LEDS are NaN except red right LED
    red_R_no_green_no_red_C_no_red_L = reduce(
        np.logical_and, (green_nans, red_C_nans, red_L_nans, ~red_R_nans)
    )
    if np.sum(red_R_no_green_no_red_C_no_red_L) > 0:
        centroid[red_R_no_green_no_red_C_no_red_L] = [
            *zip(
                pos_df.loc[
                    idx[red_R_no_green_no_red_C_no_red_L], idx[red_led_R, "x"]
                ],
                pos_df.loc[
                    idx[red_R_no_green_no_red_C_no_red_L], idx[red_led_R, "y"]
                ],
            )
        ]
    # If all red LEDs are NaN, but green LED is not
    green_no_red = reduce(
        np.logical_and, (~green_nans, red_C_nans, red_L_nans, red_R_nans)
    )
    if np.sum(green_no_red) > 0:
        centroid[green_no_red] = [
            *zip(
                pos_df.loc[idx[green_no_red], idx[green_led, "x"]],
                pos_df.loc[idx[green_no_red], idx[green_led, "y"]],
            )
        ]
    too_separated_inds = reduce(
        np.logical_or,
        (
            g_c_is_too_separated,
            l_r_is_too_separated,
            l_g_is_too_separated,
            r_g_is_too_separated,
        ),
    )
    if np.sum(too_separated_inds) > 0:
        centroid[too_separated_inds, :] = np.nan
    return centroid


def two_pt_centroid(pos_df: pd.DataFrame, **params):
    """
    Determines the centroid of 2 points using (point1 + point2) / 2
    For a given timestamp, if one point is NaN,
    then the other point is assigned as the centroid.
    If both are NaN, the centroid is NaN

    Parameters
    ----------
    pos_df : pd.DataFrame
        dataframe containing x and y position for each point of interest,
        index is timestamps. Column names specified by params
    **params : dict
        contains 'point1' and 'point2' keys,
        whose values specify the column names in `pos_df`

    Returns
    -------
    centroid : np.ndarray
        numpy array with shape (n_time, 2)
        centroid[0] is the x coord and centroid[1] is the y coord
    """

    idx = pd.IndexSlice
    centroid = np.zeros(shape=(len(pos_df), 2))
    PT1 = params["points"].pop("point1", None)
    PT2 = params["points"].pop("point2", None)
    pt1_nans = pos_df.loc[:, idx[PT1, ("x", "y")]].isna().any(axis=1)
    pt2_nans = pos_df.loc[:, idx[PT2, ("x", "y")]].isna().any(axis=1)
    dist_between_points = get_distance(
        pos_df.loc[:, idx[PT1, ("x", "y")]].to_numpy(),
        pos_df.loc[:, idx[PT2, ("x", "y")]].to_numpy(),
    )
    is_too_separated = dist_between_points >= params["max_LED_separation"]
    all_good_mask = np.logical_and(~pt1_nans, ~pt2_nans, ~is_too_separated)
    centroid[all_good_mask] = [
        *zip(
            (
                pos_df.loc[idx[all_good_mask], idx[PT1, "x"]]
                + pos_df.loc[idx[all_good_mask], idx[PT2, "x"]]
            )
            / 2,
            (
                pos_df.loc[idx[all_good_mask], idx[PT1, "y"]]
                + pos_df.loc[idx[all_good_mask], idx[PT2, "y"]]
            )
            / 2,
        )
    ]
    # If only point1 is good
    pt1_mask = np.logical_and(~pt1_nans, pt2_nans)
    if np.sum(pt1_mask) > 0:
        centroid[pt1_mask] = [
            *zip(
                pos_df.loc[idx[pt1_mask], idx[PT1, "x"]],
                pos_df.loc[idx[pt1_mask], idx[PT1, "y"]],
            )
        ]
    # If only point2 is good
    pt2_mask = np.logical_and(pt1_nans, ~pt2_nans)
    if np.sum(pt2_mask) > 0:
        centroid[pt2_mask] = [
            *zip(
                pos_df.loc[idx[pt2_mask], idx[PT2, "x"]],
                pos_df.loc[idx[pt2_mask], idx[PT2, "y"]],
            )
        ]
    # If neither point is not NaN
    all_bad_mask = np.logical_and(pt1_nans, pt2_nans)
    centroid[all_bad_mask, :] = np.nan
    # If LEDs are too far apart
    centroid[is_too_separated, :] = np.nan

    return centroid


def one_pt_centroid(pos_df: pd.DataFrame, **params):
    """
    Passes through the provided point as the centroid
    For a given timestamp, if the point is NaN,
    then the centroid is NaN

    Parameters
    ----------
    pos_df : pd.DataFrame
        dataframe containing x and y position for the point of interest,
        index is timestamps. Column name specified by params
    **kwargs : dict
        contains a 'point1' key,
        whose value specifies the column name in `pos_df`

    Returns
    -------
    centroid : np.ndarray
        numpy array with shape (n_time, 2)
        centroid[0] is the x coord and centroid[1] is the y coord
    """
    idx = pd.IndexSlice
    PT1 = params["points"].pop("point1", None)
    centroid = pos_df.loc[:, idx[PT1, ("x", "y")]].to_numpy()
    return centroid


_key_to_func_dict = {
    "four_led_centroid": four_led_centroid,
    "two_pt_centroid": two_pt_centroid,
    "one_pt_centroid": one_pt_centroid,
}

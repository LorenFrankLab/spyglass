from pathlib import Path

import datajoint as dj
import numpy as np
import pandas as pd
import pynwb
from position_tools import get_distance, get_velocity

from spyglass.common.common_behav import RawPosition
from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.position.v1.dlc_utils import (
    Centroid,
    _key_to_smooth_func_dict,
    file_log,
    get_span_start_stop,
    infer_output_dir,
    interp_pos,
    validate_list,
    validate_option,
    validate_smooth_params,
)
from spyglass.position.v1.position_dlc_cohort import DLCSmoothInterpCohort
from spyglass.position.v1.position_dlc_position import DLCSmoothInterpParams
from spyglass.utils import SpyglassMixin, logger

schema = dj.schema("position_v1_dlc_centroid")


@schema
class DLCCentroidParams(SpyglassMixin, dj.Manual):
    """Parameters for calculating the centroid

    Attributes
    ----------
    dlc_centroid_params_name : str
        Name for this set of parameters
    params : dict
        Dictionary of parameters, including...
        centroid_method : str
            Method for determining centroid. Options are:
            'two_pt_centroid': Two-point centroid calculation
            'four_led_centroid': Four-LED centroid calculation
            'null': No centroid calculation
        points : dict
            Dictionary of bodyparts to use for centroid calculation
        interpolate : bool
            Whether to interpolate over NaNs in the data
        interp_params : dict
            Parameters for interpolation like max_cm_to_interp: float
        smooth : bool
            Whether to smooth the data
        smoothing_params : dict
            Parameters for smoothing the data
            smoothing_duration : float
            smooth_method : str (e.g. 'moving_avg')
        max_LED_separation : float
            Maximum separation between LEDs for centroid calculation
        speed_smoothing_std_dev : float
            Standard deviation for Gaussian smoothing of speed data
    """

    definition = """
    dlc_centroid_params_name: varchar(80) # name for this set of parameters
    ---
    params: longblob
    """

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
    def get_default(cls) -> dict:
        """Get the default centroid parameters"""
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
        centroid_method = params.get("centroid_method")
        validate_option(  # Ensure centroid method is valid
            option=centroid_method,
            options=_key_to_points.keys(),
            name="centroid_method",
        )
        points = params.get("points")
        validate_list(  # Ensure points are valid for centroid method
            required_items=set(points),
            option_list=points,
            name="points",
            condition=centroid_method,
        )

        validate_option(
            option=params.get("max_LED_separation"),
            name="max_LED_separation",
            types=(int, float),
            permit_none=True,
        )

        validate_smooth_params(params)

        super().insert1(key, **kwargs)


@schema
class DLCCentroidSelection(SpyglassMixin, dj.Manual):
    """
    Table to pair a cohort of bodypart entries with
    the parameters for calculating their centroid
    """

    definition = """
    -> DLCSmoothInterpCohort
    -> DLCCentroidParams
    """


@schema
class DLCCentroid(SpyglassMixin, dj.Computed):
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
    log_path = None

    def make(self, key):
        """Populate the DLCCentroid table with the centroid of the bodyparts

        Uses a decorator around the _logged_make method to log the process
        to a file.

        1. Fetch parameters and centroid method.
        2. Fetch a concatenated dataframe of all bodyparts from
            DLCSmoothInterpCohort.
        3. Use the Centroid class to calculate the centroid.
        4. Optionally, interpolate over NaNs and smooth the centroid.
        5. Create a Position and Velocity objects for the centroid and video
            frame indices.
        5. Add these objects to the Analysis NWB file.
        """
        output_dir = infer_output_dir(key=key, makedir=False)
        self.log_path = Path(output_dir, "log.log")
        self._logged_make(key)
        logger.info("inserted entry into DLCCentroid")

    def _fetch_pos_df(self, key, bodyparts_to_use):
        return pd.concat(
            {
                bodypart: (
                    DLCSmoothInterpCohort.BodyPart
                    & {**key, **{"bodypart": bodypart}}
                ).fetch1_dataframe()
                for bodypart in bodyparts_to_use
            },
            axis=1,
        )

    def _available_bodyparts(self, key):
        return (DLCSmoothInterpCohort.BodyPart & key).fetch("bodypart")

    @file_log(logger)
    def _logged_make(self, key):
        METERS_PER_CM = 0.01
        idx = pd.IndexSlice
        logger.info("Centroid Calculation")

        # Get labels to smooth from Parameters table
        params = (DLCCentroidParams() & key).fetch1("params")

        points = params.get("points")
        centroid_method = params.get("centroid_method")

        # Handle the null centroid case
        if centroid_method == "null":
            logger.warning("Null centroid method selected")
            analysis_file_name = AnalysisNwbfile().create(key["nwb_file_name"])
            null_obj_id = AnalysisNwbfile().add_nwb_object(
                analysis_file_name, pd.DataFrame()
            )
            key.update(
                {
                    "analysis_file_name": analysis_file_name,
                    "dlc_position_object_id": null_obj_id,
                    "dlc_velocity_object_id": null_obj_id,
                }
            )
            AnalysisNwbfile().add(
                nwb_file_name=key["nwb_file_name"],
                analysis_file_name=key["analysis_file_name"],
            )
            self.insert1(key)
            return

        required_points = _key_to_points.get(centroid_method)
        for point in required_points:
            if points[point] not in self._available_bodyparts(key):
                raise ValueError(
                    "Bodypart in points not in model."
                    f"\tBodypart {points[point]}"
                    f"\tIn Model {self._available_bodyparts(key)}"
                )
        bodyparts_to_use = [points[point] for point in required_points]

        pos_df = self._fetch_pos_df(key=key, bodyparts_to_use=bodyparts_to_use)

        logger.info("Calculating centroid")  # now done using number of points
        centroid = Centroid(
            pos_df=pos_df,
            points=params.get("points"),
            max_LED_separation=params.get("max_LED_separation"),
        ).centroid
        centroid_df = pd.DataFrame(
            centroid,
            columns=["x", "y"],
            index=pos_df.index.to_numpy(),
        )

        if params.get("interpolate"):
            if np.any(np.isnan(centroid)):
                logger.info("interpolating over NaNs")
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
                interp_df = centroid_df.copy()
        else:
            interp_df = centroid_df.copy()

        sampling_rate = 1 / np.median(np.diff(pos_df.index.to_numpy()))
        if params.get("smooth"):
            smooth_params = params["smoothing_params"]
            dt = np.median(np.diff(pos_df.index.to_numpy()))
            sampling_rate = 1 / dt
            smooth_func = _key_to_smooth_func_dict[
                smooth_params["smooth_method"]
            ]
            logger.info(
                f"Smoothing using method: {smooth_func.__name__}",
            )
            final_df = smooth_func(
                interp_df, sampling_rate=sampling_rate, **smooth_params
            )
        else:
            final_df = interp_df.copy()

        logger.info("getting velocity")
        velocity = get_velocity(
            final_df.loc[:, idx[("x", "y")]].to_numpy(),
            time=pos_df.index.to_numpy(),
            sigma=params.pop("speed_smoothing_std_dev"),
            sampling_frequency=sampling_rate,
        )
        speed = np.sqrt(np.sum(velocity**2, axis=1))  # cm/s
        velocity_df = pd.DataFrame(
            np.concatenate((velocity, speed[:, np.newaxis]), axis=1),
            columns=["velocity_x", "velocity_y", "speed"],
            index=pos_df.index.to_numpy(),
        )
        total_nan = np.sum(final_df.loc[:, idx[("x", "y")]].isna().any(axis=1))

        logger.info(f"total NaNs in centroid dataset: {total_nan}")
        position = pynwb.behavior.Position()
        velocity = pynwb.behavior.BehavioralTimeSeries()
        if query := (RawPosition() & key):
            spatial_series = query.fetch_nwb()[0]["raw_position"]
        else:
            spatial_series = None  # pragma: no cover

        common_attrs = {
            "conversion": METERS_PER_CM,
            "comments": getattr(spatial_series, "comments", ""),
        }
        position.create_spatial_series(
            name="position",
            timestamps=final_df.index.to_numpy(),
            data=final_df.loc[:, idx[("x", "y")]].to_numpy(),
            reference_frame=getattr(spatial_series, "reference_frame", ""),
            description="x_position, y_position",
            **common_attrs,
        )
        velocity.create_timeseries(
            name="velocity",
            timestamps=velocity_df.index.to_numpy(),
            unit="m/s",
            data=velocity_df.loc[
                :, idx[("velocity_x", "velocity_y", "speed")]
            ].to_numpy(),
            description="x_velocity, y_velocity, speed",
            **common_attrs,
        )
        velocity.create_timeseries(
            name="video_frame_ind",
            unit="index",
            timestamps=final_df.index.to_numpy(),
            data=pos_df[pos_df.columns.levels[0][0]].video_frame_ind.to_numpy(),
            description="video_frame_ind",
            comments="no comments",
        )

        # Add to Analysis NWB file
        analysis_file_name = AnalysisNwbfile().create(key["nwb_file_name"])
        nwb_analysis_file = AnalysisNwbfile()
        key["dlc_position_object_id"] = nwb_analysis_file.add_nwb_object(
            analysis_file_name, position
        )
        key["dlc_velocity_object_id"] = nwb_analysis_file.add_nwb_object(
            analysis_file_name, velocity
        )

        nwb_analysis_file.add(
            nwb_file_name=key["nwb_file_name"],
            analysis_file_name=analysis_file_name,
        )

        self.insert1(
            {
                **key,
                "analysis_file_name": analysis_file_name,
            }
        )

    def fetch1_dataframe(self) -> pd.DataFrame:
        """Fetch a single dataframe."""
        _ = self.ensure_single_entry()
        nwb_data = self.fetch_nwb()[0]
        position = nwb_data["dlc_position"]
        if getattr(position, "empty", False):
            return position
        index = pd.Index(
            np.asarray(position.get_spatial_series().timestamps), name="time"
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


_key_to_points = {
    "four_led_centroid": ["greenLED", "redLED_L", "redLED_C", "redLED_R"],
    "two_pt_centroid": ["point1", "point2"],
    "one_pt_centroid": ["point1"],
    "null": [],
}

import copy
import os
from pathlib import Path

import bottleneck
import cv2
import datajoint as dj
import numpy as np
import pandas as pd
import pynwb
import pynwb.behavior
from datajoint.utils import to_camel_case
from position_tools import (
    get_angle,
    get_centriod,
    get_distance,
    get_speed,
    get_velocity,
    interpolate_nan,
)
from position_tools.core import gaussian_smooth
from tqdm import tqdm as tqdm

from ...common.common_behav import RawPosition
from ...common.common_nwbfile import AnalysisNwbfile
from ...utils.dj_helper_fn import fetch_nwb
from .dlc_utils import check_videofile, get_video_path

schema = dj.schema("position_v1_trodes_position")


@schema
class TrodesPosParams(dj.Manual):
    """
    Parameters for calculating the position (centroid, velocity, orientation)
    """

    definition = """
    trodes_pos_params_name: varchar(80) # name for this set of parameters
    ---
    params: longblob
    """

    @property
    def default_pk(self):
        return {"trodes_pos_params_name": "default"}

    @property
    def default_params(self):
        return {
            "max_LED_separation": 9.0,
            "max_plausible_speed": 300.0,
            "position_smoothing_duration": 0.125,
            "speed_smoothing_std_dev": 0.100,
            "orient_smoothing_std_dev": 0.001,
            "led1_is_front": 1,
            "is_upsampled": 0,
            "upsampling_sampling_rate": None,
            "upsampling_interpolation_method": "linear",
        }

    @classmethod
    def insert_default(cls, **kwargs):
        """
        Insert default parameter set for position determination
        """
        cls.insert1(
            {**cls().default_pk, "params": cls().default_params},
            skip_duplicates=True,
        )

    @classmethod
    def get_default(cls):
        query = cls & cls().default_pk
        if not len(query) > 0:
            cls().insert_default(skip_duplicates=True)
            return (cls & cls().default_pk).fetch1()

        return query.fetch1()

    @classmethod
    def get_accepted_params(cls):
        return [k for k in cls().default_params.keys()]


@schema
class TrodesPosSelection(dj.Manual):
    """
    Table to pair an interval with position data
    and position determination parameters
    """

    definition = """
    -> RawPosition
    -> TrodesPosParams
    """

    @classmethod
    def insert_with_default(
        cls,
        key: dict,
        skip_duplicates: bool = False,
        edit_defaults: dict = {},
        edit_name: str = None,
    ) -> None:
        """Insert key with default parameters.

        To change defaults, supply a dict as edit_defaults with a name for
        the new paramset as edit_name.

        Parameters
        ----------
        key: Union[dict, str]
            Restriction uniquely identifying entr(y/ies) in RawPosition.
        skip_duplicates: bool, optional
            Skip duplicate entries.
        edit_defauts: dict, optional
            Dictionary of overrides to default parameters.
        edit_name: str, optional
            If edit_defauts is passed, the name of the new entry

        Raises
        ------
        ValueError
            Key does not identify any entries in RawPosition.
        """
        query = RawPosition & key
        if not query:
            raise ValueError(f"Found no entries found for {key}")

        param_pk, param_name = list(TrodesPosParams().default_pk.items())[0]

        if bool(edit_defaults) ^ bool(edit_name):  # XOR: only one of them
            raise ValueError("Must specify both edit_defauts and edit_name")

        elif edit_defaults and edit_name:
            TrodesPosParams.insert1(
                {
                    param_pk: edit_name,
                    "params": {
                        **TrodesPosParams().default_params,
                        **edit_defaults,
                    },
                },
                skip_duplicates=skip_duplicates,
            )

        cls.insert(
            [
                {**k, param_pk: edit_name or param_name}
                for k in query.fetch("KEY", as_dict=True)
            ],
            skip_duplicates=skip_duplicates,
        )


@schema
class TrodesPosV1(dj.Computed):
    """
    Table to calculate the position based on Trodes tracking
    """

    definition = """
    -> TrodesPosSelection
    ---
    -> AnalysisNwbfile
    position_object_id : varchar(80)
    orientation_object_id : varchar(80)
    velocity_object_id : varchar(80)
    """

    def make(self, key):
        METERS_PER_CM = 0.01

        print(f"Computing position for: {key}")

        orig_key = copy.deepcopy(key)
        key["analysis_file_name"] = AnalysisNwbfile().create(
            key["nwb_file_name"]
        )

        position_info_parameters = (TrodesPosParams() & key).fetch1("params")

        # Rename params to match specificity discussed in #628
        position_info_parameters[
            "max_LED_separation"
        ] = position_info_parameters["max_separation"]
        position_info_parameters[
            "max_plausible_speed"
        ] = position_info_parameters["max_speed"]

        spatial_series = (RawPosition.PosObject & key).fetch_nwb()[0][
            "raw_position"
        ]

        spatial_df = (RawPosition.PosObject & key).fetch1_dataframe()
        video_frame_ind = getattr(spatial_df, "video_frame_ind", None)

        position = pynwb.behavior.Position()
        orientation = pynwb.behavior.CompassDirection()
        velocity = pynwb.behavior.BehavioralTimeSeries()

        # NOTE: CBroz1 removed a try/except ValueError that surrounded all
        #       .create_X_series methods. dpeg22 could not recall purpose

        position_info = self.calculate_position_info_from_spatial_series(
            spatial_df=spatial_df,
            meters_to_pixels=spatial_series.conversion,
            **position_info_parameters,
        )

        time_comments = dict(
            comments=spatial_series.comments,
            timestamps=position_info["time"],
        )
        time_comments_ref = dict(
            **time_comments,
            reference_frame=spatial_series.reference_frame,
        )

        # create nwb objects for insertion into analysis nwb file
        position.create_spatial_series(
            name="position",
            conversion=METERS_PER_CM,
            data=position_info["position"],
            description="x_position, y_position",
            **time_comments_ref,
        )

        orientation.create_spatial_series(
            name="orientation",
            conversion=1.0,
            data=position_info["orientation"],
            description="orientation",
            **time_comments_ref,
        )

        velocity.create_timeseries(
            name="velocity",
            conversion=METERS_PER_CM,
            unit="m/s",
            data=np.concatenate(
                (
                    position_info["velocity"],
                    position_info["speed"][:, np.newaxis],
                ),
                axis=1,
            ),
            description="x_velocity, y_velocity, speed",
            **time_comments,
        )

        if video_frame_ind:
            velocity.create_timeseries(
                name="video_frame_ind",
                unit="index",
                data=spatial_df.video_frame_ind.to_numpy(),
                description="video_frame_ind",
                **time_comments,
            )
        else:
            print(
                "No video frame index found. Assuming all camera frames "
                + "are present."
            )
            velocity.create_timeseries(
                name="video_frame_ind",
                unit="index",
                data=np.arange(len(position_info["time"])),
                description="video_frame_ind",
                **time_comments,
            )

        # Insert into analysis nwb file
        nwb_analysis_file = AnalysisNwbfile()

        key.update(
            dict(
                position_object_id=nwb_analysis_file.add_nwb_object(
                    key["analysis_file_name"], position
                ),
                orientation_object_id=nwb_analysis_file.add_nwb_object(
                    key["analysis_file_name"], orientation
                ),
                velocity_object_id=nwb_analysis_file.add_nwb_object(
                    key["analysis_file_name"], velocity
                ),
            )
        )

        AnalysisNwbfile().add(key["nwb_file_name"], key["analysis_file_name"])

        self.insert1(key)

        from ..position_merge import PositionOutput

        part_name = to_camel_case(self.table_name.split("__")[-1])

        # TODO: The next line belongs in a merge table function
        PositionOutput._merge_insert(
            [orig_key], part_name=part_name, skip_duplicates=True
        )

    @staticmethod
    def calculate_position_info_from_spatial_series(
        spatial_df: pd.DataFrame,
        meters_to_pixels: float,
        max_LED_separation,
        max_plausible_speed,
        speed_smoothing_std_dev,
        position_smoothing_duration,
        orient_smoothing_std_dev,
        led1_is_front,
        is_upsampled,
        upsampling_sampling_rate,
        upsampling_interpolation_method,
        **kwargs,
    ):
        """Calculate position info from 2D spatial series."""
        # TODO: remove in favor of fetching the same from common?
        CM_TO_METERS = 100
        # Accepts x/y 'loc' or 'loc1' format for first pos. Renames to 'loc'
        DEFAULT_COLS = ["xloc", "yloc", "xloc2", "yloc2", "xloc1", "yloc1"]

        cols = list(spatial_df.columns)
        if len(cols) != 4 or not all([c in DEFAULT_COLS for c in cols]):
            choice = dj.utils.user_choice(
                f"Unexpected columns in raw position. Assume "
                + f"{DEFAULT_COLS[:4]}?\n{spatial_df}\n"
            )
            if choice.lower() not in ["yes", "y"]:
                raise ValueError(f"Unexpected columns in raw position: {cols}")
        # rename first 4 columns, keep rest. Rest dropped below
        spatial_df.columns = DEFAULT_COLS[:4] + cols[4:]

        # Get spatial series properties
        time = np.asarray(spatial_df.index)  # seconds
        position = np.asarray(spatial_df.iloc[:, :4])  # meters

        # remove NaN times
        is_nan_time = np.isnan(time)
        position = position[~is_nan_time]
        time = time[~is_nan_time]

        dt = np.median(np.diff(time))
        sampling_rate = 1 / dt

        # Define LEDs
        if led1_is_front:
            front_LED = position[:, [0, 1]].astype(float)
            back_LED = position[:, [2, 3]].astype(float)
        else:
            back_LED = position[:, [0, 1]].astype(float)
            front_LED = position[:, [2, 3]].astype(float)

        # Convert to cm
        back_LED *= meters_to_pixels * CM_TO_METERS
        front_LED *= meters_to_pixels * CM_TO_METERS

        # Set points to NaN where the front and back LEDs are too separated
        dist_between_LEDs = get_distance(back_LED, front_LED)
        is_too_separated = dist_between_LEDs >= max_LED_separation

        back_LED[is_too_separated] = np.nan
        front_LED[is_too_separated] = np.nan

        # Calculate speed
        front_LED_speed = get_speed(
            front_LED,
            time,
            sigma=speed_smoothing_std_dev,
            sampling_frequency=sampling_rate,
        )
        back_LED_speed = get_speed(
            back_LED,
            time,
            sigma=speed_smoothing_std_dev,
            sampling_frequency=sampling_rate,
        )

        # Set to points to NaN where the speed is too fast
        is_too_fast = (front_LED_speed > max_plausible_speed) | (
            back_LED_speed > max_plausible_speed
        )
        back_LED[is_too_fast] = np.nan
        front_LED[is_too_fast] = np.nan

        # Interpolate the NaN points
        back_LED = interpolate_nan(back_LED)
        front_LED = interpolate_nan(front_LED)

        # Smooth
        moving_average_window = int(position_smoothing_duration * sampling_rate)
        back_LED = bottleneck.move_mean(
            back_LED, window=moving_average_window, axis=0, min_count=1
        )
        front_LED = bottleneck.move_mean(
            front_LED, window=moving_average_window, axis=0, min_count=1
        )

        if is_upsampled:
            position_df = pd.DataFrame(
                {
                    "time": time,
                    "back_LED_x": back_LED[:, 0],
                    "back_LED_y": back_LED[:, 1],
                    "front_LED_x": front_LED[:, 0],
                    "front_LED_y": front_LED[:, 1],
                }
            ).set_index("time")

            upsampling_start_time = time[0]
            upsampling_end_time = time[-1]

            n_samples = (
                int(
                    np.ceil(
                        (upsampling_end_time - upsampling_start_time)
                        * upsampling_sampling_rate
                    )
                )
                + 1
            )
            new_time = np.linspace(
                upsampling_start_time, upsampling_end_time, n_samples
            )
            new_index = pd.Index(
                np.unique(np.concatenate((position_df.index, new_time))),
                name="time",
            )
            position_df = (
                position_df.reindex(index=new_index)
                .interpolate(method=upsampling_interpolation_method)
                .reindex(index=new_time)
            )

            time = np.asarray(position_df.index)
            back_LED = np.asarray(
                position_df.loc[:, ["back_LED_x", "back_LED_y"]]
            )
            front_LED = np.asarray(
                position_df.loc[:, ["front_LED_x", "front_LED_y"]]
            )

            sampling_rate = upsampling_sampling_rate

        # Calculate position, orientation, velocity, speed
        position = get_centriod(back_LED, front_LED)  # cm

        orientation = get_angle(back_LED, front_LED)  # radians
        is_nan = np.isnan(orientation)

        # Unwrap orientation before smoothing
        orientation[~is_nan] = np.unwrap(orientation[~is_nan])
        orientation[~is_nan] = gaussian_smooth(
            orientation[~is_nan],
            orient_smoothing_std_dev,
            sampling_rate,
            axis=0,
            truncate=8,
        )
        # convert back to between -pi and pi
        orientation[~is_nan] = np.angle(np.exp(1j * orientation[~is_nan]))

        velocity = get_velocity(
            position,
            time=time,
            sigma=speed_smoothing_std_dev,
            sampling_frequency=sampling_rate,
        )  # cm/s
        speed = np.sqrt(np.sum(velocity**2, axis=1))  # cm/s

        return {
            "time": time,
            "position": position,
            "orientation": orientation,
            "velocity": velocity,
            "speed": speed,
        }

    def fetch_nwb(self, *attrs, **kwargs):
        return fetch_nwb(
            self, (AnalysisNwbfile, "analysis_file_abs_path"), *attrs, **kwargs
        )

    def fetch1_dataframe(self):
        nwb_data = self.fetch_nwb()[0]
        index = pd.Index(
            np.asarray(nwb_data["position"].get_spatial_series().timestamps),
            name="time",
        )
        COLUMNS = [
            "video_frame_ind",
            "position_x",
            "position_y",
            "orientation",
            "velocity_x",
            "velocity_y",
            "speed",
        ]
        return pd.DataFrame(
            np.concatenate(
                (
                    np.asarray(
                        nwb_data["velocity"]
                        .time_series["video_frame_ind"]
                        .data,
                        dtype=int,
                    )[:, np.newaxis],
                    np.asarray(nwb_data["position"].get_spatial_series().data),
                    np.asarray(
                        nwb_data["orientation"].get_spatial_series().data
                    )[:, np.newaxis],
                    np.asarray(
                        nwb_data["velocity"].time_series["velocity"].data
                    ),
                ),
                axis=1,
            ),
            columns=COLUMNS,
            index=index,
        )


@schema
class TrodesPosVideo(dj.Computed):
    """Creates a video of the computed head position and orientation as well as
    the original LED positions overlaid on the video of the animal.

    Use for debugging the effect of position extraction parameters."""

    definition = """
    -> TrodesPosV1
    --- 
    has_video : bool
    """

    def make(self, key):
        M_TO_CM = 100

        print("Loading position data...")
        raw_position_df = (
            RawPosition.PosObject
            & {
                "nwb_file_name": key["nwb_file_name"],
                "interval_list_name": key["interval_list_name"],
            }
        ).fetch1_dataframe()
        position_info_df = (TrodesPosV1() & key).fetch1_dataframe()

        print("Loading video data...")
        epoch = (
            int(
                key["interval_list_name"]
                .replace("pos ", "")
                .replace(" valid times", "")
            )
            + 1
        )

        (
            video_path,
            video_filename,
            meters_per_pixel,
            video_time,
        ) = get_video_path(
            {"nwb_file_name": key["nwb_file_name"], "epoch": epoch}
        )

        if not video_path:
            self.insert1(dict(**key, has_video=False))
            return

        video_dir = os.path.dirname(video_path) + "/"
        video_path = check_videofile(
            video_path=video_dir, video_filename=video_filename
        )[0].as_posix()
        nwb_base_filename = key["nwb_file_name"].replace(".nwb", "")
        current_dir = Path(os.getcwd())
        output_video_filename = (
            f"{current_dir.as_posix()}/{nwb_base_filename}_"
            f"{epoch:02d}_{key['trodes_pos_params_name']}.mp4"
        )
        centroids = {
            "red": np.asarray(raw_position_df[["xloc", "yloc"]]),
            "green": np.asarray(raw_position_df[["xloc2", "yloc2"]]),
        }
        position_mean = np.asarray(
            position_info_df[["position_x", "position_y"]]
        )
        orientation_mean = np.asarray(position_info_df[["orientation"]])
        position_time = np.asarray(position_info_df.index)
        cm_per_pixel = meters_per_pixel * M_TO_CM

        print("Making video...")
        self.make_video(
            video_path,
            centroids,
            position_mean,
            orientation_mean,
            video_time,
            position_time,
            output_video_filename=output_video_filename,
            cm_to_pixels=cm_per_pixel,
            disable_progressbar=False,
        )
        self.insert1(dict(**key, has_video=True))

    @staticmethod
    def convert_to_pixels(data, frame_size, cm_to_pixels=1.0):
        """Converts from cm to pixels and flips the y-axis.
        Parameters
        ----------
        data : ndarray, shape (n_time, 2)
        frame_size : array_like, shape (2,)
        cm_to_pixels : float

        Returns
        -------
        converted_data : ndarray, shape (n_time, 2)
        """
        return data / cm_to_pixels

    @staticmethod
    def fill_nan(variable, video_time, variable_time):
        video_ind = np.digitize(variable_time, video_time[1:])

        n_video_time = len(video_time)
        try:
            n_variable_dims = variable.shape[1]
            filled_variable = np.full((n_video_time, n_variable_dims), np.nan)
        except IndexError:
            filled_variable = np.full((n_video_time,), np.nan)
        filled_variable[video_ind] = variable

        return filled_variable

    def make_video(
        self,
        video_filename,
        centroids,
        position_mean,
        orientation_mean,
        video_time,
        position_time,
        output_video_filename="output.mp4",
        cm_to_pixels=1.0,
        disable_progressbar=False,
        arrow_radius=15,
        circle_radius=8,
    ):
        RGB_PINK = (234, 82, 111)
        RGB_YELLOW = (253, 231, 76)
        RGB_WHITE = (255, 255, 255)

        video = cv2.VideoCapture(video_filename)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        frame_size = (int(video.get(3)), int(video.get(4)))
        frame_rate = video.get(5)
        n_frames = int(orientation_mean.shape[0])
        print(f"video filepath: {output_video_filename}")
        out = cv2.VideoWriter(
            output_video_filename, fourcc, frame_rate, frame_size, True
        )

        centroids = {
            color: self.fill_nan(data, video_time, position_time)
            for color, data in centroids.items()
        }
        position_mean = self.fill_nan(position_mean, video_time, position_time)
        orientation_mean = self.fill_nan(
            orientation_mean, video_time, position_time
        )

        for time_ind in tqdm(
            range(n_frames - 1), desc="frames", disable=disable_progressbar
        ):
            is_grabbed, frame = video.read()
            if is_grabbed:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                red_centroid = centroids["red"][time_ind]
                green_centroid = centroids["green"][time_ind]

                position = position_mean[time_ind]
                position = self.convert_to_pixels(
                    position, frame_size, cm_to_pixels
                )
                orientation = orientation_mean[time_ind]

                if np.all(~np.isnan(red_centroid)):
                    cv2.circle(
                        img=frame,
                        center=tuple(red_centroid.astype(int)),
                        radius=circle_radius,
                        color=RGB_YELLOW,
                        thickness=-1,
                        shift=cv2.CV_8U,
                    )

                if np.all(~np.isnan(green_centroid)):
                    cv2.circle(
                        img=frame,
                        center=tuple(green_centroid.astype(int)),
                        radius=circle_radius,
                        color=RGB_PINK,
                        thickness=-1,
                        shift=cv2.CV_8U,
                    )

                if np.all(~np.isnan(position)) & np.all(~np.isnan(orientation)):
                    arrow_tip = (
                        int(position[0] + arrow_radius * np.cos(orientation)),
                        int(position[1] + arrow_radius * np.sin(orientation)),
                    )
                    cv2.arrowedLine(
                        img=frame,
                        pt1=tuple(position.astype(int)),
                        pt2=arrow_tip,
                        color=RGB_WHITE,
                        thickness=4,
                        line_type=8,
                        shift=cv2.CV_8U,
                        tipLength=0.25,
                    )

                if np.all(~np.isnan(position)):
                    cv2.circle(
                        img=frame,
                        center=tuple(position.astype(int)),
                        radius=circle_radius,
                        color=RGB_WHITE,
                        thickness=-1,
                        shift=cv2.CV_8U,
                    )

                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame)
            else:
                break

        video.release()
        out.release()
        cv2.destroyAllWindows()

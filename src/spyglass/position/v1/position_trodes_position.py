import os
from pathlib import Path

import bottleneck
import cv2
import datajoint as dj
import numpy as np
import pandas as pd
import pynwb
import pynwb.behavior
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

    @classmethod
    def insert_default(cls, **kwargs):
        """
        Insert default parameter set for position determination
        """
        params = {
            "max_separation": 9.0,
            "max_speed": 300.0,
            "position_smoothing_duration": 0.125,
            "speed_smoothing_std_dev": 0.100,
            "orient_smoothing_std_dev": 0.001,
            "led1_is_front": 1,
            "is_upsampled": 0,
            "upsampling_sampling_rate": None,
            "upsampling_interpolation_method": "linear",
        }
        cls.insert1(
            {"trodes_pos_params_name": "default", "params": params},
            skip_duplicates=True,
        )

    @classmethod
    def get_default(cls):
        query = cls & {"trodes_pos_params_name": "default"}
        if not len(query) > 0:
            cls().insert_default(skip_duplicates=True)
            default = (cls & {"trodes_pos_params_name": "default"}).fetch1()
        else:
            default = query.fetch1()
        return default

    @classmethod
    def get_accepted_params(cls):
        default = cls.get_default()
        return list(default["params"].keys())


@schema
class TrodesPosSelection(dj.Manual):
    """
    Table to pair an interval with position data
    and position determination parameters
    """

    definition = """
    -> RawPosition
    -> TrodesPosParams
    ---
    """


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
        print(f"Computing position for: {key}")
        key["analysis_file_name"] = AnalysisNwbfile().create(key["nwb_file_name"])
        raw_position = (RawPosition() & key).fetch_nwb()[0]
        position_info_parameters = (TrodesPosParams() & key).fetch1("params")
        position = pynwb.behavior.Position()
        orientation = pynwb.behavior.CompassDirection()
        velocity = pynwb.behavior.BehavioralTimeSeries()

        METERS_PER_CM = 0.01
        raw_pos_df = pd.DataFrame(
            data=raw_position["raw_position"].data,
            index=pd.Index(raw_position["raw_position"].timestamps, name="time"),
            columns=raw_position["raw_position"].description.split(", "),
        )
        try:
            # calculate the processed position
            spatial_series = raw_position["raw_position"]
            position_info = self.calculate_position_info_from_spatial_series(
                spatial_series,
                position_info_parameters["max_separation"],
                position_info_parameters["max_speed"],
                position_info_parameters["speed_smoothing_std_dev"],
                position_info_parameters["position_smoothing_duration"],
                position_info_parameters["orient_smoothing_std_dev"],
                position_info_parameters["led1_is_front"],
                position_info_parameters["is_upsampled"],
                position_info_parameters["upsampling_sampling_rate"],
                position_info_parameters["upsampling_interpolation_method"],
            )
            # create nwb objects for insertion into analysis nwb file
            position.create_spatial_series(
                name="position",
                timestamps=position_info["time"],
                conversion=METERS_PER_CM,
                data=position_info["position"],
                reference_frame=spatial_series.reference_frame,
                comments=spatial_series.comments,
                description="x_position, y_position",
            )

            orientation.create_spatial_series(
                name="orientation",
                timestamps=position_info["time"],
                conversion=1.0,
                data=position_info["orientation"],
                reference_frame=spatial_series.reference_frame,
                comments=spatial_series.comments,
                description="orientation",
            )

            velocity.create_timeseries(
                name="velocity",
                timestamps=position_info["time"],
                conversion=METERS_PER_CM,
                unit="m/s",
                data=np.concatenate(
                    (position_info["velocity"], position_info["speed"][:, np.newaxis]),
                    axis=1,
                ),
                comments=spatial_series.comments,
                description="x_velocity, y_velocity, speed",
            )
            try:
                velocity.create_timeseries(
                    name="video_frame_ind",
                    unit="index",
                    timestamps=position_info["time"],
                    data=raw_pos_df.video_frame_ind.to_numpy(),
                    description="video_frame_ind",
                    comments=spatial_series.comments,
                )
            except AttributeError:
                print(
                    "No video frame index found. Assuming all camera frames are present."
                )
                velocity.create_timeseries(
                    name="video_frame_ind",
                    unit="index",
                    timestamps=position_info["time"],
                    data=np.arange(len(position_info["time"])),
                    description="video_frame_ind",
                    comments=spatial_series.comments,
                )
        except ValueError:
            pass

        # Insert into analysis nwb file
        nwb_analysis_file = AnalysisNwbfile()

        key["position_object_id"] = nwb_analysis_file.add_nwb_object(
            key["analysis_file_name"], position
        )
        key["orientation_object_id"] = nwb_analysis_file.add_nwb_object(
            key["analysis_file_name"], orientation
        )
        key["velocity_object_id"] = nwb_analysis_file.add_nwb_object(
            key["analysis_file_name"], velocity
        )

        AnalysisNwbfile().add(key["nwb_file_name"], key["analysis_file_name"])

        self.insert1(key)
        from ..position_merge import PositionOutput

        key["source"] = "Trodes"
        key["version"] = 1
        trodes_key = key.copy()
        valid_fields = PositionOutput().fetch().dtype.fields.keys()
        entries_to_delete = [entry for entry in key.keys() if entry not in valid_fields]
        for entry in entries_to_delete:
            del key[entry]
        PositionOutput().insert1(key=key, params=trodes_key, skip_duplicates=True)

    @staticmethod
    def calculate_position_info_from_spatial_series(
        spatial_series,
        max_LED_separation,
        max_plausible_speed,
        speed_smoothing_std_dev,
        position_smoothing_duration,
        orient_smoothing_std_dev,
        led1_is_front,
        is_upsampled,
        upsampling_sampling_rate,
        upsampling_interpolation_method,
    ):
        CM_TO_METERS = 100

        # Get spatial series properties
        time = np.asarray(spatial_series.timestamps)  # seconds
        position = np.asarray(
            pd.DataFrame(
                spatial_series.data, columns=spatial_series.description.split(", ")
            ).loc[:, ["xloc", "yloc", "xloc2", "yloc2"]]
        )  # meters

        # remove NaN times
        is_nan_time = np.isnan(time)
        position = position[~is_nan_time]
        time = time[~is_nan_time]

        dt = np.median(np.diff(time))
        sampling_rate = 1 / dt
        meters_to_pixels = spatial_series.conversion

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
                np.unique(np.concatenate((position_df.index, new_time))), name="time"
            )
            position_df = (
                position_df.reindex(index=new_index)
                .interpolate(method=upsampling_interpolation_method)
                .reindex(index=new_time)
            )

            time = np.asarray(position_df.index)
            back_LED = np.asarray(position_df.loc[:, ["back_LED_x", "back_LED_y"]])
            front_LED = np.asarray(position_df.loc[:, ["front_LED_x", "front_LED_y"]])

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
                        nwb_data["velocity"].time_series["video_frame_ind"].data,
                        dtype=int,
                    )[:, np.newaxis],
                    np.asarray(nwb_data["position"].get_spatial_series().data),
                    np.asarray(nwb_data["orientation"].get_spatial_series().data)[
                        :, np.newaxis
                    ],
                    np.asarray(nwb_data["velocity"].time_series["velocity"].data),
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
    """

    def make(self, key):
        M_TO_CM = 100

        print("Loading position data...")
        raw_position_df = (
            RawPosition()
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

        video_path, video_filename, meters_per_pixel, video_time = get_video_path(
            {"nwb_file_name": key["nwb_file_name"], "epoch": epoch}
        )
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
        position_mean = np.asarray(position_info_df[["position_x", "position_y"]])
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
        orientation_mean = self.fill_nan(orientation_mean, video_time, position_time)

        for time_ind in tqdm(
            range(n_frames - 1), desc="frames", disable=disable_progressbar
        ):
            is_grabbed, frame = video.read()
            if is_grabbed:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                red_centroid = centroids["red"][time_ind]
                green_centroid = centroids["green"][time_ind]

                position = position_mean[time_ind]
                position = self.convert_to_pixels(position, frame_size, cm_to_pixels)
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

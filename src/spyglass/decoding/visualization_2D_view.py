from typing import Callable, Dict, Tuple

import numpy as np
import sortingview.views.franklab as vvf
import xarray as xr
from replay_trajectory_classification.environments import get_grid, get_track_interior


def create_static_track_animation(
    *,
    track_rect_width: float,
    track_rect_height: float,
    ul_corners: np.ndarray,
    timestamps: np.ndarray,
    positions: np.ndarray,
    compute_real_time_rate: bool = False,
    head_dir=None,
):
    # float32 gives about 7 digits of decimal precision; we want 3 digits right of the decimal.
    # So need to compress-store the timestamp if the start is greater than say 5000.
    first_timestamp = 0
    if timestamps[0] > 5000:
        first_timestamp = timestamps[0]
        timestamps -= first_timestamp
    data = {
        "type": "TrackAnimation",
        "trackBinWidth": track_rect_width,
        "trackBinHeight": track_rect_height,
        "trackBinULCorners": ul_corners.astype(np.float32),
        "totalRecordingFrameLength": len(timestamps),
        "timestamps": timestamps.astype(np.float32),
        "positions": positions.astype(np.float32),
        "xmin": np.min(ul_corners[0]),
        "xmax": np.max(ul_corners[0]) + track_rect_width,
        "ymin": np.min(ul_corners[1]),
        "ymax": np.max(ul_corners[1]) + track_rect_height
        # Speed: should this be displayed?
        # TODO: Better approach for accommodating further data streams
    }
    if head_dir is not None:
        # print(f'Loading head direction: {head_dir}')
        data["headDirection"] = head_dir.astype(np.float32)
    if compute_real_time_rate:
        median_delta_t = np.median(np.diff(timestamps))
        sampling_frequency_Hz = 1 / median_delta_t
        data["samplingFrequencyHz"] = sampling_frequency_Hz
    if first_timestamp > 0:
        data["timestampStart"] = first_timestamp

    return data


def get_base_track_information(base_probabilities: xr.Dataset):
    x_count = len(base_probabilities.x_position)
    y_count = len(base_probabilities.y_position)
    x_min = np.min(base_probabilities.x_position).item()
    y_min = np.min(base_probabilities.y_position).item()
    x_width = round(
        (np.max(base_probabilities.x_position).item() - x_min) / (x_count - 1), 6
    )
    y_width = round(
        (np.max(base_probabilities.y_position).item() - y_min) / (y_count - 1), 6
    )
    return (x_count, x_min, x_width, y_count, y_min, y_width)


def memo_linearize(
    t: Tuple[float, float, float],
    /,
    location_lookup: Dict[Tuple[float, float], int],
    x_count: int,
    x_min: float,
    x_width: float,
    y_min: float,
    y_width: float,
):
    (_, y, x) = t
    my_tuple = (x, y)
    if my_tuple not in location_lookup:
        lin = x_count * round((y - y_min) / y_width) + round((x - x_min) / x_width)
        location_lookup[my_tuple] = lin
    return location_lookup[my_tuple]


def generate_linearization_function(
    location_lookup: Dict[Tuple[float, float], int],
    x_count: int,
    x_min: float,
    x_width: float,
    y_min: float,
    y_width: float,
):
    args = {
        "location_lookup": location_lookup,
        "x_count": x_count,
        "x_min": x_min,
        "x_width": x_width,
        "y_min": y_min,
        "y_width": y_width,
    }

    def inner(t: Tuple[float, float, float]):
        return memo_linearize(t, **args)

    return inner


def discretize_and_trim(base_slice: xr.DataArray):
    i = np.multiply(base_slice, 255).astype(np.uint8)
    i_stack = i.stack(unified_index=["time", "y_position", "x_position"])

    return i_stack.where(i_stack > 0, drop=True).astype(np.uint8)


def get_positions(
    i_trim: xr.Dataset, linearization_fn: Callable[[Tuple[float, float]], int]
):
    linearizer_map = map(linearization_fn, i_trim.unified_index.values)
    return np.array(list(linearizer_map), dtype=np.uint16)


def get_observations_per_frame(i_trim: xr.DataArray, base_slice: xr.DataArray):
    (times, time_counts_np) = np.unique(i_trim.time.values, return_counts=True)
    time_counts = xr.DataArray(time_counts_np, coords={"time": times})
    raw_times = base_slice.time
    (_, good_counts) = xr.align(raw_times, time_counts, join="left", fill_value=0)
    observations_per_frame = good_counts.values.astype(np.uint8)
    return observations_per_frame


def extract_slice_data(
    base_slice: xr.DataArray, location_fn: Callable[[Tuple[float, float]], int]
):
    i_trim = discretize_and_trim(base_slice)

    positions = get_positions(i_trim, location_fn)
    observations_per_frame = get_observations_per_frame(i_trim, base_slice)
    return i_trim.values, positions, observations_per_frame


def process_decoded_data(posterior: xr.DataArray):
    frame_step_size = 100_000
    location_lookup = {}

    (x_count, x_min, x_width, y_count, y_min, y_width) = get_base_track_information(
        posterior
    )
    location_fn = generate_linearization_function(
        location_lookup, x_count, x_min, x_width, y_min, y_width
    )

    total_frame_count = len(posterior.time)
    final_frame_bounds = np.zeros(total_frame_count, dtype=np.uint8)
    # intentionally oversized preallocation--will trim later
    # Note: By definition there can't be more than 255 observations per frame (since we drop any observation
    # lower than 1/255 and the probabilities for any frame sum to 1). However, this preallocation may be way
    # too big for memory for long recordings. We could use a smaller one, but would need to include logic
    # to expand the length of the array if its actual allocated bounds are exceeded.
    final_values = np.zeros(total_frame_count * 255, dtype=np.uint8)
    final_locations = np.zeros(total_frame_count * 255, dtype=np.uint16)

    frames_done = 0
    total_observations = 0
    while frames_done <= total_frame_count:
        base_slice = posterior.isel(
            time=slice(frames_done, frames_done + frame_step_size)
        )
        observations, positions, observations_per_frame = extract_slice_data(
            base_slice, location_fn
        )
        final_frame_bounds[
            frames_done : frames_done + len(observations_per_frame)
        ] = observations_per_frame
        final_values[
            total_observations : total_observations + len(observations)
        ] = observations
        final_locations[
            total_observations : total_observations + len(observations)
        ] = positions
        total_observations += len(observations)
        frames_done += frame_step_size
    # These were intentionally oversized in preallocation; trim to the number of actual values.
    final_values.resize(total_observations)
    final_locations.resize(total_observations)

    return {
        "type": "DecodedPositionData",
        "xmin": x_min,
        "binWidth": x_width,
        "xcount": x_count,
        "ymin": y_min,
        "binHeight": y_width,
        "ycount": y_count,
        "uniqueLocations": np.unique(final_locations),
        "values": final_values,
        "locations": final_locations,
        "frameBounds": final_frame_bounds,
    }


def create_track_animation_object(*, static_track_animation: any):
    if "decodedData" in static_track_animation:
        decoded_data = static_track_animation["decodedData"]
        decoded_data_obj = vvf.DecodedPositionData(
            x_min=decoded_data["xmin"],
            x_count=decoded_data["xcount"],
            y_min=decoded_data["ymin"],
            y_count=decoded_data["ycount"],
            bin_width=decoded_data["binWidth"],
            bin_height=decoded_data["binHeight"],
            values=decoded_data["values"].astype(np.int16),
            locations=decoded_data["locations"],
            frame_bounds=decoded_data["frameBounds"].astype(np.int16),
        )
    else:
        decoded_data_obj = None

    timestamp_start = (
        static_track_animation["timestampStart"]
        if "timestampStart" in static_track_animation
        else None
    )
    head_direction = (
        static_track_animation["headDirection"]
        if "headDirection" in static_track_animation
        else None
    )
    return vvf.TrackPositionAnimationV1(
        track_bin_width=static_track_animation["trackBinWidth"],
        track_bin_height=static_track_animation["trackBinHeight"],
        track_bin_ul_corners=static_track_animation["trackBinULCorners"],
        total_recording_frame_length=static_track_animation[
            "totalRecordingFrameLength"
        ],
        timestamp_start=timestamp_start,
        timestamps=static_track_animation["timestamps"],
        positions=static_track_animation["positions"],
        x_min=static_track_animation["xmin"],
        x_max=static_track_animation["xmax"],
        y_min=static_track_animation["ymin"],
        y_max=static_track_animation["ymax"],
        sampling_frequency_hz=static_track_animation["samplingFrequencyHz"],
        head_direction=head_direction,
        decoded_data=decoded_data_obj,
    )


def get_ul_corners(width: float, height: float, centers):
    ul = np.subtract(centers, (width / 2, -height / 2))

    # Reshape so we have an x array and a y array
    return ul.T


def make_track(position, bin_size: float = 1.0):
    (edges, _, place_bin_centers, _) = get_grid(position, bin_size)
    is_track_interior = get_track_interior(position, edges)

    # bin dimensions are the difference between bin centers in the x and y directions.
    bin_width = np.max(np.diff(place_bin_centers, axis=0)[:, 0])
    bin_height = np.max(np.diff(place_bin_centers, axis=0)[:, 1])

    # so we can represent the track as a collection of rectangles of width bin_width and height bin_height,
    # centered on the values of place_bin_centers where track_interior = true.
    # Note, the original code uses Fortran ordering.
    true_ctrs = place_bin_centers[is_track_interior.ravel(order="F")]
    upper_left_points = get_ul_corners(bin_width, bin_height, true_ctrs)

    return bin_width, bin_height, upper_left_points


def create_2D_decode_view(
    position_time: np.ndarray,
    position: np.ndarray,
    posterior: xr.DataArray,
    bin_size: float,
    head_dir: np.ndarray = None,
) -> vvf.TrackPositionAnimationV1:
    """Creates a 2D decoding movie view

    Parameters
    ----------
    position_time : np.ndarray, shape (n_time,)
    position : np.ndarray, shape (n_time, 2)
    posterior : xr.DataArray, shape (n_time, n_position_bins)
    bin_size : float
    head_dir : np.ndarray, optional

    Returns
    -------
    view : vvf.TrackPositionAnimationV1

    """
    position_time = np.squeeze(np.asarray(position_time)).copy()
    position = np.asarray(position)
    if head_dir is not None:
        head_dir = np.squeeze(np.asarray(head_dir))

    track_width, track_height, upper_left_points = make_track(
        position, bin_size=bin_size
    )

    data = create_static_track_animation(
        ul_corners=upper_left_points,
        track_rect_height=track_height,
        track_rect_width=track_width,
        timestamps=position_time,
        positions=position.T,
        head_dir=head_dir,
        compute_real_time_rate=True,
    )
    data["decodedData"] = process_decoded_data(posterior)

    return create_track_animation_object(static_track_animation=data)

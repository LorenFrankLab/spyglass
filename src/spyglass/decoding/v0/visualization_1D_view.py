from typing import Union

import numpy as np
import sortingview.views.franklab as vvf
import xarray as xr

from spyglass.decoding.v0.utils import discretize_and_trim


def get_observations_per_time(
    trimmed_posterior: xr.DataArray, base_data: xr.Dataset
) -> np.ndarray:
    """Get the number of observations per time bin."""
    times, counts = np.unique(trimmed_posterior.time.values, return_counts=True)
    indexed_counts = xr.DataArray(counts, coords={"time": times})
    _, good_counts = xr.align(
        base_data.time, indexed_counts, join="left", fill_value=0
    )  # type: ignore

    return good_counts.values.astype(np.uint8)


def get_sampling_freq(times: np.ndarray) -> float:
    """Get the sampling frequency of the data."""
    round_times = np.floor(1000 * times)
    median_delta_t_ms = np.median(np.diff(round_times)).item()
    return 1000 / median_delta_t_ms  # from time-delta to Hz


def get_trimmed_bin_center_index(
    place_bin_centers: np.ndarray, trimmed_place_bin_centers: np.ndarray
) -> np.ndarray:
    """Get the index of the trimmed bin centers in the full array."""
    return np.searchsorted(
        place_bin_centers, trimmed_place_bin_centers, side="left"
    ).astype(np.uint16)


def create_1D_decode_view(
    posterior: xr.DataArray,
    linear_position: np.ndarray = None,
    ref_time_sec: Union[np.float64, None] = None,
) -> vvf.DecodedLinearPositionData:
    """Creates a view of an interactive heatmap of position vs. time.

    Parameters
    ----------
    posterior : xr.DataArray, shape (n_time, n_position_bins)
    linear_position : np.ndarray, shape (n_time, ), optional
    ref_time_sec : np.float64, optional
        Reference time for the purpose of offsetting the start time

    Returns
    -------
    view : vvf.DecodedLinearPositionData

    """
    if linear_position is not None:
        linear_position = np.asarray(linear_position).squeeze()

    trimmed_posterior = discretize_and_trim(posterior, ndims=1)
    observations_per_time = get_observations_per_time(
        trimmed_posterior, posterior
    )
    sampling_freq = get_sampling_freq(posterior.time)
    start_time_sec = posterior.time.values[0]
    if ref_time_sec is not None:
        start_time_sec = start_time_sec - ref_time_sec

    trimmed_bin_center_index = get_trimmed_bin_center_index(
        posterior.position.values, trimmed_posterior.position.values
    )

    return vvf.DecodedLinearPositionData(
        values=trimmed_posterior.values,
        positions=trimmed_bin_center_index,
        frame_bounds=observations_per_time,
        positions_key=posterior.position.values.astype(np.float32),
        observed_positions=linear_position,
        start_time_sec=start_time_sec,
        sampling_frequency=sampling_freq,
    )

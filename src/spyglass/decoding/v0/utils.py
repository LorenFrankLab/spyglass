import numpy as np
import xarray as xr


def discretize_and_trim(series: xr.DataArray, ndims=2) -> xr.DataArray:
    """Discretizes a continuous series and trims the zeros.

    Parameters
    ----------
    series : xr.DataArray, shape (n_time, n_position_bins)
        Continuous series to be discretized
    ndims : int, optional
        Number of dimensions of the series. Default is 2 for 1D series (time,
        position), 3 for 2D series (time, y_position, x_position)

    Returns
    -------
    discretized : xr.DataArray, shape (n_time, n_position_bins)
        Discretized and trimmed series
    """
    index = (
        ["time", "position"]
        if ndims == 2
        else ["time", "y_position", "x_position"]
    )
    discretized = np.multiply(series, 255).astype(np.uint8)  # type: ignore
    stacked = discretized.stack(unified_index=index)
    return stacked.where(stacked > 0, drop=True).astype(np.uint8)

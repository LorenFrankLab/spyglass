"""Pure-function pose processing utilities.

Extracted from PoseV2 so they can be tested and reused without a live
DataJoint connection.
"""

import numpy as np
import pandas as pd


def convert_to_cm(
    pose_df: pd.DataFrame,
    meters_per_pixel: float,
) -> pd.DataFrame:
    """Convert x, y pose coordinates from pixels to centimetres.

    Parameters
    ----------
    pose_df : pd.DataFrame
        Pose DataFrame with MultiIndex columns. The innermost level must
        contain coordinate names ``'x'`` and ``'y'``.  Likelihood columns
        are not modified.
    meters_per_pixel : float
        Camera calibration factor: metres represented by one pixel.

    Returns
    -------
    pd.DataFrame
        Copy of pose_df with x and y columns scaled to centimetres.
    """
    pose_df = pose_df.copy()
    xy_mask = pose_df.columns.get_level_values(-1).isin(["x", "y"])
    pose_df.loc[:, xy_mask] = pose_df.loc[:, xy_mask] * (meters_per_pixel * 100)
    return pose_df


def apply_likelihood_threshold(
    pose_df: pd.DataFrame,
    likelihood_thresh: float,
) -> pd.DataFrame:
    """Set x/y to NaN where likelihood is below threshold.

    Parameters
    ----------
    pose_df : pd.DataFrame
        DataFrame with MultiIndex columns (scorer, bodypart, coord) where
        coord must include 'x', 'y', and 'likelihood'.
    likelihood_thresh : float
        Threshold in [0, 1].  Frames where ``likelihood < threshold`` or
        likelihood is NaN are masked (treated as low-confidence).

    Returns
    -------
    pd.DataFrame
        Copy of pose_df with low-likelihood positions set to NaN.

    Raises
    ------
    KeyError
        If any bodypart is missing a 'likelihood' column.  Callers must
        ensure pose data contains likelihood before applying this threshold.
    """
    pose_df = pose_df.copy()
    idx = pd.IndexSlice

    if (
        isinstance(pose_df.columns, pd.MultiIndex)
        and pose_df.columns.nlevels >= 2
    ):
        coord_level = pose_df.columns.nlevels - 1
        bodypart_level = coord_level - 1
        bodyparts = pose_df.columns.get_level_values(bodypart_level).unique()

        for bodypart in bodyparts:
            if pose_df.columns.nlevels == 3:
                likelihood = pose_df.loc[:, idx[:, bodypart, "likelihood"]]
                if isinstance(likelihood, pd.DataFrame):
                    likelihood = likelihood.min(axis=1)
            else:
                likelihood = pose_df.loc[:, idx[bodypart, "likelihood"]]
            # ~(>= thresh) treats NaN as low-confidence (NaN >= thresh is False)
            low = ~(likelihood >= likelihood_thresh)
            if pose_df.columns.nlevels == 3:
                pose_df.loc[low, idx[:, bodypart, ["x", "y"]]] = np.nan
            else:
                pose_df.loc[low, idx[bodypart, ["x", "y"]]] = np.nan

    return pose_df


def _smooth_bodypart_positions(
    pose_flat: pd.DataFrame,
    smooth_params: dict,
    sampling_rate: float,
) -> pd.DataFrame:
    """Apply interpolation + moving-average to each bodypart independently.

    Replicates V1's ``DLCSmoothInterp`` step: per-bodypart smoothing applied
    before orientation and centroid are computed, so that orientation benefits
    from pre-cleaned trajectories rather than raw (noisy) bodypart positions.

    Parameters
    ----------
    pose_flat : pd.DataFrame
        2-level MultiIndex ``(bodypart, coord)`` DataFrame.  Must have ``x``
        and ``y`` columns per bodypart; ``likelihood`` columns are unchanged.
        Index must be timestamps in seconds.
    smooth_params : dict
        Smoothing configuration dict.  Relevant keys:

        - ``interpolate`` – bool, whether to interpolate NaN spans.
        - ``interp_params`` – dict with ``max_pts_to_interp`` and
          ``max_cm_to_interp`` (passed straight to ``interp_position``).
        - ``smooth`` – bool, whether to apply moving-average.
        - ``smoothing_params`` – dict with at least ``smoothing_duration``
          in seconds.
    sampling_rate : float
        Frames per second; used to convert ``smoothing_duration`` to a window
        size.

    Returns
    -------
    pd.DataFrame
        Copy of ``pose_flat`` with ``x``/``y`` smoothed per bodypart.
        ``likelihood`` columns and all other columns are left untouched.
    """
    from spyglass.position.utils.interpolation import (
        interp_position,
        smooth_moving_avg,
    )
    from spyglass.position.utils.orientation import get_span_start_stop

    do_interp = smooth_params.get("interpolate", False)
    do_smooth = smooth_params.get("smooth", False)
    if not do_interp and not do_smooth:
        return pose_flat.copy()

    interp_p = smooth_params.get("interp_params", {})
    sp = smooth_params.get("smoothing_params", {})
    smoothing_duration = sp.get("smoothing_duration", 0.05)

    result = pose_flat.copy()
    bodyparts = pose_flat.columns.get_level_values(0).unique()

    for bp in bodyparts:
        bp_df = pd.DataFrame(
            {
                "x": pose_flat[(bp, "x")].values,
                "y": pose_flat[(bp, "y")].values,
            },
            index=pose_flat.index,
        )

        if do_interp:
            is_nan = np.isnan(bp_df["x"]) | np.isnan(bp_df["y"])
            if np.any(is_nan):
                nan_spans = get_span_start_stop(np.where(is_nan)[0])
                bp_df = interp_position(
                    bp_df,
                    nan_spans,
                    max_pts_to_interp=interp_p.get("max_pts_to_interp"),
                    max_cm_to_interp=interp_p.get("max_cm_to_interp"),
                )

        if do_smooth:
            bp_df = smooth_moving_avg(
                bp_df,
                smoothing_duration=smoothing_duration,
                sampling_rate=sampling_rate,
            )

        result.loc[:, (bp, "x")] = bp_df["x"].values
        result.loc[:, (bp, "y")] = bp_df["y"].values

    return result


def compute_pose_outputs(
    pose_df: pd.DataFrame,
    orient_params: dict,
    centroid_params: dict,
    smooth_params: dict,
) -> dict:
    """Run the full pose-processing pipeline as pure computation.

    No DataJoint access.  All inputs are plain Python / NumPy / pandas objects.

    Parameters
    ----------
    pose_df : pd.DataFrame
        Raw pose DataFrame.  Columns may be a 2- or 3-level MultiIndex
        (scorer, bodypart, coord) or (bodypart, coord).  Index must be
        timestamps in seconds.
    orient_params : dict
        Orientation parameters — ``method`` key required.
    centroid_params : dict
        Centroid parameters — ``method`` and ``points`` keys required.
    smooth_params : dict
        Smoothing parameters — ``likelihood_thresh``, ``interpolate``,
        ``smooth`` keys supported.

    Returns
    -------
    dict
        Keys: ``orientation`` (ndarray, shape n), ``centroid`` (ndarray,
        shape n×2), ``velocity_2d`` (ndarray, shape n×2), ``speed``
        (ndarray, shape n), ``timestamps`` (ndarray, shape n),
        ``sampling_rate`` (float).
    """
    from spyglass.position.utils.centroid import calculate_centroid
    from spyglass.position.utils.general import flatten_multiindex
    from spyglass.position.utils.interpolation import (
        get_smoothing_function,
        interp_position,
    )
    from spyglass.position.utils.orientation import (
        bisector_orientation,
        get_span_start_stop,
        no_orientation,
        smooth_orientation,
        two_pt_orientation,
    )

    timestamps = pose_df.index.values
    sampling_rate = float(1 / np.median(np.diff(timestamps)))

    # --- likelihood threshold -------------------------------------------------
    # Default 0.95 matches PoseParams default; pass explicitly to override.
    likelihood_thresh = smooth_params.get("likelihood_thresh", 0.95)
    pose_df = apply_likelihood_threshold(pose_df, likelihood_thresh)

    # flatten to (bodypart, coord) for utility functions
    pose_flat = flatten_multiindex(pose_df)

    # --- bodypart pre-smoothing (matches V1 DLCSmoothInterp) -----------------
    # Smooth each bodypart independently before orientation / centroid so that
    # orientation benefits from cleaned trajectories, not raw detections.
    if smooth_params.get("interpolate", False) or smooth_params.get(
        "smooth", False
    ):
        pose_flat = _smooth_bodypart_positions(
            pose_flat, smooth_params, sampling_rate
        )

    # --- orientation ----------------------------------------------------------
    method = orient_params["method"]
    if method == "two_pt":
        orientation = two_pt_orientation(
            pose_flat,
            point1=orient_params["bodypart1"],
            point2=orient_params["bodypart2"],
        )
    elif method == "bisector":
        orientation = bisector_orientation(
            pose_flat,
            led1=orient_params["led1"],
            led2=orient_params["led2"],
            led3=orient_params["led3"],
        )
    elif method == "none":
        orientation = no_orientation(pose_flat)
    else:
        raise ValueError(f"Unknown orientation method: {method!r}")

    if orient_params.get("smooth", False):
        sp = orient_params.get("smoothing_params", {})
        orientation = smooth_orientation(
            orientation,
            timestamps,
            sp.get("std_dev", 0.001),
            orient_params.get("interpolate", True),
        )

    # --- centroid -------------------------------------------------------------
    max_sep = centroid_params.get("max_LED_separation", None)
    centroid = calculate_centroid(pose_flat, centroid_params["points"], max_sep)

    # --- smoothing / interpolation of centroid --------------------------------
    pos_df = pd.DataFrame(centroid, columns=["x", "y"], index=timestamps)

    if smooth_params.get("interpolate", False):
        interp_p = smooth_params.get("interp_params", {})
        is_nan = np.isnan(pos_df["x"]) | np.isnan(pos_df["y"])
        if np.any(is_nan):
            nan_spans = get_span_start_stop(np.where(is_nan)[0])
            pos_df = interp_position(
                pos_df,
                nan_spans,
                max_pts_to_interp=interp_p.get("max_pts_to_interp"),
                max_cm_to_interp=interp_p.get("max_cm_to_interp"),
            )

    if smooth_params.get("smooth", False):
        sp = smooth_params["smoothing_params"]
        sm = sp["method"]
        smooth_func = get_smoothing_function(sm)
        if sm == "moving_avg":
            pos_df = smooth_func(
                pos_df,
                smoothing_duration=sp["smoothing_duration"],
                sampling_rate=sampling_rate,
            )
        elif sm == "savgol":
            pos_df = smooth_func(
                pos_df,
                window_length=sp["window_length"],
                polyorder=sp.get("polyorder", 3),
            )
        elif sm == "gaussian":
            pos_df = smooth_func(
                pos_df,
                std_dev=sp["std_dev"],
                sampling_rate=sampling_rate,
            )

    centroid_smooth = pos_df[["x", "y"]].values

    # --- velocity -------------------------------------------------------------
    from spyglass.position.utils.velocity import compute_velocity

    vel_std = smooth_params.get("velocity_smoothing_std_dev") or None
    velocity_2d, speed = compute_velocity(
        centroid_smooth, timestamps, smooth_std_dev=vel_std
    )

    return {
        "orientation": orientation,
        "centroid": centroid_smooth,
        "velocity_2d": velocity_2d,
        "speed": speed,
        "timestamps": timestamps,
        "sampling_rate": sampling_rate,
    }


def calculate_velocity(
    position: np.ndarray,
    timestamps: np.ndarray,
    sampling_rate: float,  # noqa: ARG001 — kept for API consistency
) -> np.ndarray:
    """Calculate scalar speed (cm/s) from (x, y) position.

    Parameters
    ----------
    position : np.ndarray
        Shape (n_frames, 2).  x in column 0, y in column 1.
    timestamps : np.ndarray
        Shape (n_frames,).  Time in seconds.
    sampling_rate : float
        Sampling rate in Hz (unused in computation; kept for interface parity).

    Returns
    -------
    np.ndarray
        Shape (n_frames,).  First element is NaN; remainder are speeds in
        units of position / time (typically cm/s).
    """
    dx = np.diff(position[:, 0])
    dy = np.diff(position[:, 1])
    displacement = np.sqrt(dx**2 + dy**2)
    dt = np.diff(timestamps)
    velocity = displacement / dt
    return np.concatenate([[np.nan], velocity])

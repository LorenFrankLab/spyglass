import inspect
import logging
from typing import List

import numpy as np
import pandas as pd
import numpy.typing as npt
import xarray as xr
from scipy.ndimage import label


def create_interval_labels(
    is_missing: npt.NDArray[np.bool_],
) -> npt.NDArray[np.intp]:
    """Create interval labels from a missing data mask.

    Uses scipy.ndimage.label to identify contiguous regions of valid data
    (where is_missing=False) and assigns sequential integer labels.

    Parameters
    ----------
    is_missing : npt.NDArray[np.bool_], shape (n_time,)
        Boolean mask where True indicates time points outside intervals

    Returns
    -------
    interval_labels : npt.NDArray[np.intp], shape (n_time,)
        Integer labels where:
        - -1 indicates time points outside any interval
        - 0, 1, 2, ... indicate the 1st, 2nd, 3rd interval index
    """
    # label() returns 1-indexed labels (0=outside, 1=first region, 2=second, ...)
    # We want: -1=outside, 0=first interval, 1=second interval, ...
    raw_labels, _ = label(~is_missing)
    return raw_labels - 1


def concatenate_interval_results(
    interval_results: List[xr.Dataset],
) -> xr.Dataset:
    """Concatenate results from multiple intervals along time dimension.

    All datasets must have compatible structure (same variables, compatible
    coordinates except time). Time coordinates will be concatenated and an
    interval_labels coordinate will be added to track which interval each
    time point belongs to.

    Parameters
    ----------
    interval_results : List[xr.Dataset], length n_intervals
        Results from each decoding interval. Each dataset must have a 'time'
        dimension and coordinate. Empty datasets should be filtered out before
        calling this function.

    Returns
    -------
    xr.Dataset
        Concatenated results with interval_labels coordinate.
        The interval_labels coordinate contains integer values where each
        value indicates which interval the corresponding time point belongs to.

    Raises
    ------
    ValueError
        If interval_results is empty or contains empty datasets

    Examples
    --------
    >>> import xarray as xr
    >>> import numpy as np
    >>> ds1 = xr.Dataset({"x": ("time", [1, 2, 3])}, coords={"time": [0.0, 0.1, 0.2]})
    >>> ds2 = xr.Dataset({"x": ("time", [4, 5])}, coords={"time": [1.0, 1.1]})
    >>> result = concatenate_interval_results([ds1, ds2])
    >>> result.interval_labels.values
    array([0, 0, 0, 1, 1])
    >>> result.time.values
    array([0. , 0.1, 0.2, 1. , 1.1])
    """
    if not interval_results:
        raise ValueError("All decoding intervals are empty")

    # Validate each result has time points
    for i, result in enumerate(interval_results):
        if len(result.time) == 0:
            raise ValueError(
                f"Interval {i} has empty time dimension - "
                f"should not be included in interval_results list"
            )

    # Pre-allocate with known size for efficiency
    total_length = sum(len(result.time) for result in interval_results)
    interval_labels = np.empty(total_length, dtype=np.intp)

    offset = 0
    for interval_idx, result in enumerate(interval_results):
        n_times = len(result.time)
        interval_labels[offset : offset + n_times] = interval_idx
        offset += n_times

    concatenated = xr.concat(interval_results, dim="time")
    return concatenated.assign_coords(interval_labels=("time", interval_labels))


def _get_interval_range(key: dict) -> tuple[float, float]:
    """Return maximum range of model times in encoding/decoding intervals.

    Note: This function accesses the database to fetch interval times.

    Parameters
    ----------
    key : dict
        The decoding selection key

    Returns
    -------
    tuple[float, float]
        The minimum and maximum times for the model
    """
    # Lazy import to avoid database connection at module load time
    from spyglass.common.common_interval import IntervalList

    encoding_interval = (
        IntervalList
        & {
            "nwb_file_name": key["nwb_file_name"],
            "interval_list_name": key["encoding_interval"],
        }
    ).fetch1("valid_times")

    decoding_interval = (
        IntervalList
        & {
            "nwb_file_name": key["nwb_file_name"],
            "interval_list_name": key["decoding_interval"],
        }
    ).fetch1("valid_times")

    return (
        float(
            min(
                np.asarray(encoding_interval).min(),
                np.asarray(decoding_interval).min(),
            )
        ),
        float(
            max(
                np.asarray(encoding_interval).max(),
                np.asarray(decoding_interval).max(),
            )
        ),
    )


def get_valid_kwargs(
    classifier,
    decoding_kwargs: dict,
    logger: logging.Logger,
) -> tuple[dict, dict]:
    """Get valid fit and predict kwargs, warning about any ignored kwargs.

    Inspects the classifier's fit and predict method signatures to determine
    which kwargs are valid. Logs a warning if any provided kwargs are not
    valid for either method.

    Parameters
    ----------
    classifier : object
        Classifier instance with fit and predict methods
    decoding_kwargs : dict
        User-provided kwargs for fit/predict
    logger : logging.Logger
        Logger for warnings

    Returns
    -------
    fit_kwargs : dict
        Kwargs valid for classifier.fit
    predict_kwargs : dict
        Kwargs valid for classifier.predict
    """
    fit_sig = inspect.signature(classifier.fit)
    valid_fit_kwargs: set[str] = set(fit_sig.parameters.keys())
    predict_sig = inspect.signature(classifier.predict)
    valid_predict_kwargs: set[str] = set(predict_sig.parameters.keys())

    # Warn about kwargs that are not valid for either fit or predict
    if decoding_kwargs:
        all_valid_kwargs = valid_fit_kwargs | valid_predict_kwargs
        ignored_kwargs = set(decoding_kwargs.keys()) - all_valid_kwargs
        if ignored_kwargs:
            logger.warning(
                f"The following decoding_kwargs are not valid for "
                f"classifier.fit or classifier.predict and will be ignored: "
                f"{sorted(ignored_kwargs)}. "
                f"Valid fit kwargs: {sorted(valid_fit_kwargs)}. "
                f"Valid predict kwargs: {sorted(valid_predict_kwargs)}."
            )

    fit_kwargs = {
        k: value
        for k, value in decoding_kwargs.items()
        if k in valid_fit_kwargs
    }
    predict_kwargs = {
        k: value
        for k, value in decoding_kwargs.items()
        if k in valid_predict_kwargs
    }

    return fit_kwargs, predict_kwargs


def declared_tracking_intervals(position_info, decoding_kwargs=None):
    """Keep declared epoch continuity instead of inferring it from camera gaps."""
    kwargs = decoding_kwargs or {}
    intervals = kwargs.get(
        "valid_position_intervals",
        position_info.attrs.get("valid_position_intervals"),
    )
    if intervals is None:
        raise ValueError(
            "Tracking continuity is required: retain PositionGroup epoch support "
            "or pass decoding_kwargs['valid_position_intervals']."
        )
    intervals = np.asarray(intervals, dtype=float)
    if (
        intervals.ndim != 2
        or intervals.shape[1] != 2
        or not len(intervals)
        or not np.isfinite(intervals).all()
        or np.any(intervals[:, 0] >= intervals[:, 1])
        or np.any(intervals[1:, 0] < intervals[:-1, 1])
    ):
        raise ValueError(
            "Tracking intervals must be ordered and non-overlapping; resolve overlapping epochs upstream"
        )
    if np.any(np.diff(position_info.index.to_numpy(dtype=float)) <= 0):
        raise ValueError(
            "Resolve duplicate or unordered position timestamps upstream"
        )
    return intervals


def prepare_decoder_grid(
    classifier, position_info, position_columns, time_range, decoding_kwargs
):
    """Build uniform bins, measured support, and explicitly aligned covariates.

    Training masks and environment/group labels retain their original rows.
    A timestamped missing mask uses preceding-sample ownership. Transition
    covariates may use original timestamps with declared interpolation kinds,
    or already match the actual decode centers. Values during masked HMM gaps
    must be supplied explicitly; tracking continuity does not fill those gaps.
    """
    from non_local_detector.analysis import align_tracking_to_results

    edges = classifier.calculate_time_edges(time_range, trim=True)
    centers = (edges[:-1] + edges[1:]) / 2
    grid = xr.Dataset(
        coords={
            "time": centers,
            "time_bin_start": ("time", edges[:-1]),
            "time_bin_end": ("time", edges[1:]),
        }
    )
    intervals = declared_tracking_intervals(position_info, decoding_kwargs)
    _, supported = align_tracking_to_results(
        position_info[position_columns],
        grid,
        position_columns=position_columns,
        valid_position_intervals=intervals,
    )
    supplied_missing = decoding_kwargs.get("is_missing")
    if isinstance(supplied_missing, pd.Series):
        if np.array_equal(supplied_missing.index.to_numpy(), centers):
            supplied_missing = supplied_missing.to_numpy()
        elif np.array_equal(
            supplied_missing.index.to_numpy(), position_info.index.to_numpy()
        ):
            tracking = position_info[position_columns].copy()
            tracking["__missing__"] = supplied_missing.to_numpy()
            aligned, _ = align_tracking_to_results(
                tracking,
                grid,
                position_columns=position_columns,
                valid_position_intervals=intervals,
                categorical_columns=["__missing__"],
            )
            supplied_missing = (
                aligned["__missing__"]
                .where(aligned["__missing__"].notna(), True)
                .to_numpy(dtype=bool)
            )
        else:
            raise ValueError(
                "Timestamped is_missing must match original tracking or actual decode centers"
            )
    if supplied_missing is None:
        supplied_missing = np.zeros(len(centers), dtype=bool)
    supplied_missing = np.asarray(supplied_missing, dtype=bool)
    if supplied_missing.shape != centers.shape:
        raise ValueError(
            "is_missing needs one value per decode bin; use a timestamped Series to align an original tracking mask"
        )
    missing = ~supported | supplied_missing

    covariates = decoding_kwargs.get("discrete_transition_covariate_data")
    kinds = decoding_kwargs.get("discrete_transition_covariate_kinds", {})
    if covariates is not None:
        if isinstance(covariates, pd.DataFrame) and not np.array_equal(
            covariates.index.to_numpy(), centers
        ):
            if not np.array_equal(
                covariates.index.to_numpy(), position_info.index.to_numpy()
            ):
                raise ValueError(
                    "Covariate DataFrame timestamps must match original tracking or actual decode centers"
                )
            if set(kinds) - set(covariates) or any(
                kind not in {"continuous", "categorical", "circular"}
                for kind in kinds.values()
            ):
                raise ValueError(
                    "Declare existing covariate kinds as continuous, categorical, or circular"
                )
            tracking = position_info[position_columns].copy()
            names = {column: f"__covariate__{column}" for column in covariates}
            for column, renamed in names.items():
                tracking[renamed] = covariates[column].to_numpy()
            categorical = [
                names[column]
                for column in covariates
                if kinds.get(column) == "categorical"
                or (
                    column not in kinds
                    and (
                        pd.api.types.is_bool_dtype(covariates[column])
                        or not pd.api.types.is_numeric_dtype(covariates[column])
                    )
                )
            ]
            circular = [
                names[column]
                for column in covariates
                if kinds.get(column) == "circular"
            ]
            aligned, _ = align_tracking_to_results(
                tracking,
                grid,
                position_columns=position_columns,
                valid_position_intervals=intervals,
                categorical_columns=categorical,
                circular_columns=circular,
            )
            covariates = aligned[list(names.values())].rename(
                columns={value: key for key, value in names.items()}
            )
        values = (
            covariates.values()
            if isinstance(covariates, dict)
            else (covariates[column].to_numpy() for column in covariates)
        )
        for value in values:
            value = np.asarray(value)
            if value.shape != centers.shape:
                raise ValueError(
                    "Transition covariates need one row per actual decode bin"
                )
            if pd.isna(value).any() or (
                np.issubdtype(value.dtype, np.number)
                and not np.isfinite(value).all()
            ):
                raise ValueError(
                    "Transition covariates must be finite even in masked HMM gaps; supply explicitly grid-aligned covariates rather than interpolate across missing tracking"
                )
    return edges, missing, covariates


def spikes_for_sequence(
    spike_times,
    edges,
    *,
    shared_stop=False,
    shared_boundary=None,
    spike_waveform_features=None,
):
    """Assign a shared sequence boundary once, preserving event/mark alignment."""
    masks = [
        (times >= edges[0])
        & (times <= edges[-1])
        & (
            (
                times
                < (edges[-1] if shared_boundary is None else shared_boundary)
            )
            if shared_stop
            else True
        )
        for times in spike_times
    ]
    times = [
        np.asarray(events)[mask]
        for events, mask in zip(spike_times, masks, strict=True)
    ]
    marks = (
        None
        if spike_waveform_features is None
        else [
            np.asarray(features)[mask]
            for features, mask in zip(
                spike_waveform_features, masks, strict=True
            )
        ]
    )
    return times, marks


def _interval_bin_masks(edges, intervals):
    """Yield whole-bin masks with O(n_bins) workspace and timestamp roundoff."""
    intervals = np.asarray(intervals, dtype=float)
    if (
        intervals.ndim != 2
        or intervals.shape[1] != 2
        or not np.isfinite(intervals).all()
    ):
        raise ValueError("Intervals must contain finite start/stop pairs")
    lower, upper = edges[:-1], edges[1:]
    width_bound = np.diff(edges) * 0.01
    for start, stop in intervals:
        lower_tol = np.minimum(
            4 * np.maximum(np.spacing(np.abs(lower)), np.spacing(abs(start))),
            width_bound,
        )
        upper_tol = np.minimum(
            4 * np.maximum(np.spacing(np.abs(upper)), np.spacing(abs(stop))),
            width_bound,
        )
        yield (lower >= start - lower_tol) & (upper <= stop + upper_tol)


def decoder_interval_labels(edges, intervals):
    """Preserve requested interval identity independently of missing tracking."""
    labels = np.full(len(edges) - 1, -1, dtype=np.intp)
    for number, selected in enumerate(_interval_bin_masks(edges, intervals)):
        labels[selected] = number
    return labels

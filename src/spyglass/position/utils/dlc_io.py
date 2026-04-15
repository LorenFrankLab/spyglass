"""Shared DLC I/O utilities for Position V1/V2 pipelines.

Consolidates duplicate DLC file reading logic with consistent API.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd


def parse_dlc_h5_output(
    h5_path: Union[Path, str],
    bodyparts: Optional[list[str]] = None,
    likelihood_threshold: Optional[float] = None,
    return_metadata: bool = True,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, str, list]]:
    """Standardized DLC H5/CSV parsing for V1/V2 pipelines.

    Handles both .h5 and .csv DLC output files with consistent formatting.
    Supports filtering by bodyparts and likelihood threshold.

    Parameters
    ----------
    h5_path : Union[Path, str]
        Path to DLC output file (.h5 or .csv)
    bodyparts : Optional[list[str]], optional
        If provided, filter to only these bodyparts
    likelihood_threshold : Optional[float], optional
        If provided, set low-confidence points to NaN
    return_metadata : bool, optional
        If True, return (df, scorer, bodyparts) tuple.
        If False, return just DataFrame. Default True.

    Returns
    -------
    Union[pd.DataFrame, Tuple[pd.DataFrame, str, list]]
        If return_metadata=True: (dataframe, scorer, bodyparts_list)
        If return_metadata=False: dataframe only

    Raises
    ------
    FileNotFoundError
        If h5_path doesn't exist
    ValueError
        If file format is unsupported
    ValueError
        If bodyparts filter contains invalid bodyparts

    Examples
    --------
    >>> # Basic usage with metadata
    >>> df, scorer, bodyparts = parse_dlc_h5_output("video1DLC_resnet50.h5")
    >>> print(f"Found {len(bodyparts)} bodyparts from {scorer}")

    >>> # Just get DataFrame
    >>> df = parse_dlc_h5_output("video1DLC_resnet50.h5", return_metadata=False)

    >>> # Filter specific bodyparts with likelihood threshold
    >>> df, _, _ = parse_dlc_h5_output(
    ...     "output.h5",
    ...     bodyparts=["nose", "tail"],
    ...     likelihood_threshold=0.9
    ... )
    """
    h5_path = Path(h5_path)

    if not h5_path.exists():
        raise FileNotFoundError(f"DLC output file not found: {h5_path}")

    # Load DLC data based on file extension
    if h5_path.suffix == ".h5":
        df = pd.read_hdf(h5_path)
    elif h5_path.suffix == ".csv":
        df = pd.read_csv(h5_path, header=[0, 1, 2], index_col=0)
    else:
        raise ValueError(
            f"Unsupported file format: {h5_path.suffix}. "
            "Expected .h5 or .csv DLC output file."
        )

    # Validate MultiIndex structure (DLC standard: [scorer, bodypart, coords])
    if not isinstance(df.columns, pd.MultiIndex) or df.columns.nlevels != 3:
        raise ValueError(
            f"Invalid DLC file structure. Expected 3-level MultiIndex columns "
            f"[scorer, bodypart, coords], got {df.columns.nlevels} levels."
        )

    # Extract metadata
    scorer = df.columns.get_level_values(0)[0]
    available_bodyparts = df.columns.get_level_values(1).unique().tolist()

    # Filter bodyparts if requested
    if bodyparts is not None:
        # Validate bodyparts exist
        invalid_parts = [
            bp for bp in bodyparts if bp not in available_bodyparts
        ]
        if invalid_parts:
            raise ValueError(
                f"Bodyparts not found in DLC output: {invalid_parts}. "
                f"Available: {available_bodyparts}"
            )

        # Filter DataFrame columns
        filtered_cols = [
            col
            for col in df.columns
            if col[1] in bodyparts  # col[1] is bodypart level
        ]
        df = df[filtered_cols]
        available_bodyparts = bodyparts

    # Apply likelihood threshold if provided
    if likelihood_threshold is not None:
        for bodypart in available_bodyparts:
            # Get likelihood column for this bodypart
            likelihood_col = (scorer, bodypart, "likelihood")
            if likelihood_col in df.columns:
                # Set x,y to NaN where likelihood < threshold
                low_conf_mask = df[likelihood_col] < likelihood_threshold
                for coord in ["x", "y"]:
                    coord_col = (scorer, bodypart, coord)
                    if coord_col in df.columns:
                        df.loc[low_conf_mask, coord_col] = np.nan

    if return_metadata:
        return df, scorer, available_bodyparts
    else:
        return df


def get_dlc_bodyparts(h5_path: Union[Path, str]) -> list[str]:
    """Extract bodypart names from DLC output file without loading full data.

    Parameters
    ----------
    h5_path : Union[Path, str]
        Path to DLC output file

    Returns
    -------
    list[str]
        List of bodypart names
    """
    h5_path = Path(h5_path)

    if h5_path.suffix == ".h5":
        # For h5, we can get columns without loading full data
        with pd.HDFStore(h5_path, mode="r") as store:
            # Get first key (typically 'df_with_missing')
            key = list(store.keys())[0]
            try:
                columns = store.get_storer(key).attrs.non_index_axes[0][1]
                return pd.Index(columns).get_level_values(1).unique().tolist()
            except AttributeError:
                # Fallback: read the data directly if metadata is not available
                df = pd.read_hdf(h5_path, key=key)
                return df.columns.get_level_values(1).unique().tolist()
    else:
        # For CSV, need to read header only
        df_header = pd.read_csv(h5_path, header=[0, 1, 2], index_col=0, nrows=0)
        return df_header.columns.get_level_values(1).unique().tolist()


def get_dlc_scorer(h5_path: Union[Path, str]) -> str:
    """Extract scorer name from DLC output file without loading full data.

    Parameters
    ----------
    h5_path : Union[Path, str]
        Path to DLC output file

    Returns
    -------
    str
        Scorer/model name
    """
    h5_path = Path(h5_path)

    if h5_path.suffix == ".h5":
        with pd.HDFStore(h5_path, mode="r") as store:
            key = list(store.keys())[0]
            try:
                columns = store.get_storer(key).attrs.non_index_axes[0][1]
                return pd.Index(columns).get_level_values(0)[0]
            except AttributeError:
                # Fallback: read the data directly if metadata is not available
                df = pd.read_hdf(h5_path, key=key)
                return df.columns.get_level_values(0)[0]
    else:
        df_header = pd.read_csv(h5_path, header=[0, 1, 2], index_col=0, nrows=0)
        return df_header.columns.get_level_values(0)[0]


def reformat_dlc_data(df: pd.DataFrame, bodyparts: list[str] = None) -> dict:
    """Reformat DLC DataFrame to bodypart-keyed dictionary (V1 compatibility).

    Converts DLC MultiIndex DataFrame to nested dict format used in V1.

    Parameters
    ----------
    df : pd.DataFrame
        DLC DataFrame with MultiIndex [scorer, bodypart, coords]
    bodyparts : list[str], optional
        If provided, filter to these bodyparts only

    Returns
    -------
    dict
        Nested dict: {bodypart: {coord: np.array}}
    """
    scorer = df.columns.get_level_values(0)[0]
    available_bodyparts = (
        bodyparts or df.columns.get_level_values(1).unique().tolist()
    )

    bodypart_data = {}
    for bodypart in available_bodyparts:
        bodypart_data[bodypart] = {}
        for coord in ["x", "y", "likelihood"]:
            col = (scorer, bodypart, coord)
            if col in df.columns:
                bodypart_data[bodypart][coord] = df[col].values

    return bodypart_data


def validate_dlc_file(h5_path: Union[Path, str]) -> bool:
    """Validate that file is a proper DLC output file.

    Parameters
    ----------
    h5_path : Union[Path, str]
        Path to potential DLC file

    Returns
    -------
    bool
        True if valid DLC file structure, False otherwise
    """
    try:
        h5_path = Path(h5_path)
        if not h5_path.exists() or h5_path.suffix not in [".h5", ".csv"]:
            return False

        # Try to parse without loading full data
        get_dlc_scorer(h5_path)
        get_dlc_bodyparts(h5_path)
        return True

    except (Exception,):
        return False

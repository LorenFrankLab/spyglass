"""SLEAP analysis file I/O utilities.

Provides parsers for SLEAP output formats compatible with the Spyglass
position pipeline's DataFrame contract.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def parse_sleap_analysis_h5(
    file_path: Union[str, Path],
    return_metadata: bool = False,
    track_index: Optional[int] = None,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, str, List[str]]]:
    """Parse a SLEAP .analysis.h5 file into a pose DataFrame.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the SLEAP .analysis.h5 file.
    return_metadata : bool, optional
        If True, return ``(df, scorer, bodyparts)`` matching the
        ``parse_dlc_h5_output`` contract. Default False.
    track_index : int, optional
        For multi-animal recordings, index of the track to select. If None,
        selects the track with the highest frame occupancy.

    Returns
    -------
    pd.DataFrame or tuple
        DataFrame with 2-level MultiIndex columns ``(bodypart, coord)``
        where ``coord ∈ {"x", "y", "likelihood"}``. Index is integer frame
        number; NaN values preserve missing frames.
        If ``return_metadata=True``, returns ``(df, scorer, bodyparts)``.

    Raises
    ------
    FileNotFoundError
        If ``file_path`` does not exist.
    ImportError
        If ``h5py`` is not installed.
    ValueError
        If ``track_index`` is out of range.

    Notes
    -----
    SLEAP .analysis.h5 dataset layout (the "matlab" preset written by
    ``Labels.export_analysis`` / ``sleap_io.save_analysis_h5``, which is
    SLEAP's default and the format documented in SLEAP's own export
    tutorial — column-major order for MATLAB compatibility):

    - ``node_names``    — shape (n_nodes,), bodypart name strings
    - ``track_names``   — shape (n_tracks,), track/animal name strings
    - ``tracks``        — shape (n_tracks, 2, n_nodes, n_frames); NaN = missing
    - ``track_occupancy`` — shape (n_frames, n_tracks), bool (frame-first;
                            this is a documented quirk — every other array
                            here is track-first)
    - ``instance_scores`` — shape (n_tracks, n_frames), overall confidence
    - ``point_scores``  — shape (n_tracks, n_nodes, n_frames), per-node
                          confidence; may be absent in older exports

    When ``point_scores`` is absent, ``likelihood`` is filled from
    ``instance_scores`` broadcast per node (conservative fallback).
    """
    try:
        import h5py
    except ImportError:
        raise ImportError(
            "h5py is required to parse SLEAP analysis files. "
            "Install with: pip install h5py"
        )

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"SLEAP analysis file not found: {file_path}")

    with h5py.File(file_path, "r") as f:
        node_names = [
            n.decode() if isinstance(n, bytes) else n
            for n in f["node_names"][:]
        ]
        track_names = [
            t.decode() if isinstance(t, bytes) else t
            for t in f["track_names"][:]
        ]
        # (n_tracks, 2, n_nodes, n_frames)
        tracks = f["tracks"][:]

        if "point_scores" in f:
            # (n_tracks, n_nodes, n_frames)
            point_scores = f["point_scores"][:]
        else:
            # Fallback: broadcast instance_scores across nodes
            instance_scores = f["instance_scores"][:]  # (n_tracks, n_frames)
            n_nodes = len(node_names)
            point_scores = np.broadcast_to(
                instance_scores[:, np.newaxis, :],
                (instance_scores.shape[0], n_nodes, instance_scores.shape[1]),
            ).copy()

        if "track_occupancy" in f:
            track_occupancy = f["track_occupancy"][:]  # (n_frames, n_tracks)
        else:
            track_occupancy = np.ones(
                (tracks.shape[3], tracks.shape[0]), dtype=bool
            )

    n_tracks = tracks.shape[0]

    if n_tracks == 1:
        selected_track = 0
    elif track_index is not None:
        if track_index >= n_tracks or track_index < 0:
            raise ValueError(
                f"track_index {track_index} out of range "
                f"(n_tracks={n_tracks})"
            )
        selected_track = track_index
    else:
        # Highest-occupancy track
        occupancy_counts = track_occupancy.sum(axis=0)
        selected_track = int(np.argmax(occupancy_counts))

    scorer = track_names[selected_track] if track_names else "SLEAP"

    # tracks[selected_track]: (2, n_nodes, n_frames) -> (n_frames, n_nodes, 2)
    xy = np.transpose(tracks[selected_track], (2, 1, 0))
    # point_scores[selected_track]: (n_nodes, n_frames) -> (n_frames, n_nodes)
    likelihood = point_scores[selected_track].T

    # Build 2-level MultiIndex: (bodypart, coord)
    tuples = []
    for bp in node_names:
        tuples.extend([(bp, "x"), (bp, "y"), (bp, "likelihood")])
    col_index = pd.MultiIndex.from_tuples(tuples, names=["bodyparts", "coords"])

    n_frames = xy.shape[0]
    data = np.empty((n_frames, len(node_names) * 3), dtype=float)
    for i in range(len(node_names)):
        data[:, i * 3] = xy[:, i, 0]
        data[:, i * 3 + 1] = xy[:, i, 1]
        data[:, i * 3 + 2] = likelihood[:, i]

    df = pd.DataFrame(data, index=range(n_frames), columns=col_index)

    if return_metadata:
        return df, scorer, node_names
    return df

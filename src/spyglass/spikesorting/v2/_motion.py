"""SpikeInterface ``Motion`` serialization and summaries.

``motion_to_storage_dict`` / ``motion_from_storage_dict`` round-trip a
``Motion`` through a DataJoint-blob-safe dict; ``motion_max_abs_displacement_um``
and ``motion_n_temporal_bins`` summarize one for a stored row.

DB-FREE AT IMPORT. This module activates no ``dj.schema`` and opens no DB
connection; SpikeInterface is imported lazily inside the function that needs
it.
"""

from __future__ import annotations

import numpy as np


def motion_to_storage_dict(motion) -> dict:
    """Flatten a SpikeInterface ``Motion`` to a DataJoint-blob-safe dict.

    The stored blob keeps every field the ``Motion`` constructor needs so
    :func:`motion_from_storage_dict` reconstructs the object exactly:

    - ``displacement``: list (one entry per recording segment) of 2-D
      ``float`` arrays, each shape ``(n_temporal_bins, n_spatial_bins)`` (um).
    - ``temporal_bins_s``: list (per segment) of 1-D bin-center arrays (s).
    - ``spatial_bins_um``: a single 1-D array of window centers (um),
      ``shape == (n_spatial_bins,)`` -- shared across segments.
    - ``direction``: the motion axis (``"x"`` / ``"y"`` / ``"z"``).
    - ``interpolation_method``: how displacement interpolates between bins.

    Arrays stay as NumPy arrays (DataJoint's blob codec round-trips them);
    only the container shape is normalized so the rehydrate side is
    deterministic regardless of how the codec returns nested lists.
    """
    return {
        "displacement": [np.asarray(d) for d in motion.displacement],
        "temporal_bins_s": [np.asarray(t) for t in motion.temporal_bins_s],
        "spatial_bins_um": np.asarray(motion.spatial_bins_um),
        "direction": str(motion.direction),
        "interpolation_method": str(motion.interpolation_method),
    }


def motion_from_storage_dict(blob: dict):
    """Rebuild a SpikeInterface ``Motion`` from a stored blob.

    Inverse of :func:`motion_to_storage_dict`. Coerces each entry back to a
    clean NumPy array before constructing ``Motion`` so a blob codec that
    returns the per-segment lists as object arrays (rather than Python lists)
    still rehydrates to the 2-D / 1-D shapes ``Motion`` asserts on. Every key
    is read directly (no defaulting): :func:`motion_to_storage_dict` always
    writes all five, so a missing key means a corrupt blob and should raise
    rather than silently rehydrate a wrong motion axis.
    """
    from spikeinterface.core.motion import Motion

    return Motion(
        [np.asarray(d) for d in blob["displacement"]],
        [np.asarray(t) for t in blob["temporal_bins_s"]],
        np.asarray(blob["spatial_bins_um"]),
        direction=str(blob["direction"]),
        interpolation_method=str(blob["interpolation_method"]),
    )


def motion_max_abs_displacement_um(motion) -> float:
    """Largest absolute displacement (um) over all segments / bins / windows.

    The single-number drift-severity summary used to flag high-drift
    sessions. NaNs are NOT masked -- a non-finite estimate surfaces in the
    stored metric rather than being silently hidden by a ``nan``-aware max.
    """
    flat = np.concatenate([np.asarray(d).ravel() for d in motion.displacement])
    return float(np.max(np.abs(flat)))


def motion_n_temporal_bins(motion) -> int:
    """Total number of temporal bins across all recording segments."""
    return int(sum(np.asarray(t).shape[0] for t in motion.temporal_bins_s))

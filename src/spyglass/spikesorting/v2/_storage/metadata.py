"""NWB / recording-metadata helpers for the v2 spike-sorting tables.

Heavy dependencies are imported inside the functions.
"""

from __future__ import annotations

import numpy as np


def resolve_conversion_and_offset(recording) -> tuple[float, float]:
    """Resolve the ElectricalSeries ``(conversion, offset)`` for a recording.

    Traces are written unscaled, so readback uses
    ``volts = raw * conversion + offset``. NWB holds one scalar of each, so
    per-channel differences are rejected rather than mis-scaled.

    Returns
    -------
    (conversion, offset) : tuple of float
        Volts-per-count and volts, for the ElectricalSeries.

    Raises
    ------
    ValueError
        If the recording has heterogeneous channel gains, a non-positive
        channel gain, or heterogeneous channel offsets.
    """
    gains = np.unique(recording.get_channel_gains())
    if len(gains) != 1:
        raise ValueError(
            "resolve_conversion_and_offset: recording has heterogeneous "
            f"channel gains {gains.tolist()}; v2 ElectricalSeries write "
            "requires a single conversion factor. Verify probe metadata for "
            "the sort group."
        )
    if gains[0] <= 0:
        raise ValueError(
            "resolve_conversion_and_offset: recording has a non-positive "
            f"channel gain {float(gains[0])}; cannot scale to volts. Verify "
            "probe metadata for the sort group."
        )
    offsets = np.unique(recording.get_channel_offsets())
    if len(offsets) != 1:
        raise ValueError(
            "resolve_conversion_and_offset: recording has heterogeneous "
            f"channel offsets {offsets.tolist()}; a single ElectricalSeries "
            "offset cannot represent them."
        )
    return float(gains[0]) * 1e-6, float(offsets[0]) * 1e-6


def electrode_table_region(nwbf, electrode_ids, description: str):
    """Build an ElectricalSeries electrode-table region from electrode ids.

    pynwb's ``region`` takes row indices, not electrode ids, so ids are mapped
    to rows; passing ids would silently mislabel channels when they differ.

    Parameters
    ----------
    nwbf : pynwb.NWBFile
        File whose ``electrodes`` table the region indexes into.
    electrode_ids : iterable of int
        Spyglass electrode ids (e.g. the recording's channel ids).
    description : str
        Region description.

    Returns
    -------
    hdmf.common.table.DynamicTableRegion
        Region over the NWB ``electrodes`` table rows that correspond to
        ``electrode_ids``, for attaching to the ElectricalSeries.

    Raises
    ------
    ValueError
        If an id is not in the electrodes table.
    """
    from spyglass.utils.nwb_helper_fn import (
        get_electrode_indices,
        invalid_electrode_index,
    )

    ids = [int(e) for e in electrode_ids]
    indices = get_electrode_indices(nwbf, ids)
    missing = [
        eid for eid, idx in zip(ids, indices) if idx == invalid_electrode_index
    ]
    if missing:
        raise ValueError(
            "electrode_table_region: electrode ids "
            f"{missing} are not in the NWB electrodes table"
        )
    return nwbf.create_electrode_table_region(
        region=[int(i) for i in indices], description=description
    )


def write_buffer_gb(
    n_channels: int,
    sampling_frequency: float,
    itemsize: int = 8,
) -> float:
    """Streaming-write buffer size (GB) bounded to ~30 s of data.

    A fixed 5 GB buffer would hold a whole tetrode recording and spike RAM
    under parallel workers. Capped at 5 GB.

    Parameters
    ----------
    n_channels : int
        Number of channels in the recording being written.
    sampling_frequency : float
        Sampling frequency of the recording, in Hz.
    itemsize : int, optional
        Bytes per sample of the trace dtype. Default ``8`` (float64).

    Returns
    -------
    float
        Buffer size in GB, in ``[0, 5]``.
    """
    seconds_gb = (
        float(n_channels) * float(sampling_frequency) * 30.0 * itemsize / 1e9
    )
    return min(5.0, seconds_gb)

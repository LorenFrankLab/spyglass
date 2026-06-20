"""NWB / recording-metadata helpers for the v2 spike-sorting tables.

ElectricalSeries conversion/offset resolution, the electrode-table-region
builder, and the AnalysisNwbfile content hash. The heavy dependencies
(numpy, pynwb/HDMF, spyglass.common, spyglass.utils.nwb_helper_fn) are
imported lazily inside the functions, so the module itself imports light --
no DataJoint / SpikeInterface / spyglass.common at import. ``utils``
re-exports these names so existing ``from .utils import
electrode_table_region`` (etc.) call sites are unchanged.
"""

from __future__ import annotations


def resolve_conversion_and_offset(recording) -> tuple[float, float]:
    """Resolve the ElectricalSeries ``(conversion, offset)`` for a recording.

    v2 writes traces UNSCALED (``return_in_uV=False``), so the persisted
    ElectricalSeries must carry BOTH the gain (as ``conversion``) and the
    per-channel offset (as ``offset``) to recover physical volts on readback:
    ``volts = raw * conversion + offset``. SpikeInterface stores per-channel
    gain/offset in microvolts (``uV = raw*gain + offset``); a single NWB
    ``conversion``/``offset`` scalar can only represent a UNIFORM gain/offset,
    so heterogeneous values are rejected rather than silently mis-scaled.

    Dropping the offset (the prior behavior, inherited from v1) silently biased
    every channel by ``offset`` uV on readback for recordings with a non-zero
    DC offset (e.g. Intan / Open Ephys). A non-positive gain (``0`` ->
    all-zero recording, negative -> sign flip) is also rejected.

    Returns
    -------
    (conversion, offset) : tuple of float
        Volts-per-count and volts, for the ElectricalSeries.

    Raises
    ------
    ValueError
        If the recording has heterogeneous channel gains, a non-positive
        channel gain, or heterogeneous channel offsets -- none of which a
        single scalar ``conversion``/``offset`` can represent.
    """
    import numpy as np

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

    ``pynwb.NWBFile.create_electrode_table_region(region=...)`` interprets
    ``region`` as ROW INDICES into the electrodes table, NOT electrode ids.
    Passing spyglass ``electrode_id`` values directly is correct only when
    ``electrode.id == row position``; for an electrodes table whose ids are
    non-contiguous or reordered it silently points the ElectricalSeries at the
    wrong electrode rows -- wrong channel locations and brain-region
    attribution on readback. Map ids -> row indices via
    ``get_electrode_indices`` so the region is correct for any electrodes
    table, and fail loud on an unknown id rather than aliasing it.

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


def _hash_nwb_recording(analysis_file_name: str) -> str:
    """Return a content hash of a recording's AnalysisNwbfile.

    Delegates to ``AnalysisNwbfile().get_hash`` (the project's blessed
    wrapper over ``NwbfileHasher``) so v2 verification uses the same
    hashing path as the v1 recompute machinery.

    Parameters
    ----------
    analysis_file_name : str
        Name of the AnalysisNwbfile holding the preprocessed recording.

    Returns
    -------
    str
        The ``NwbfileHasher`` digest of the file.
    """
    from spyglass.common.common_nwbfile import AnalysisNwbfile

    return AnalysisNwbfile().get_hash(analysis_file_name)

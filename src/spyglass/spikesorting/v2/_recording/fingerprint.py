"""Content fingerprint for the v2 preprocessed recording cache.

A representation-blind hash of the persisted ``ElectricalSeries`` (traces,
timestamps, geometry, scaling metadata), so a content-identical rebuild hashes
equal. Design notes are in the developer docs. Imports are lazy, so
``recording.py`` and ``recompute.py`` both import it without a cycle.
"""

from __future__ import annotations

TRACE_ROUNDING = 4  # µV; well below the ephys noise floor
TIMESTAMP_ROUNDING = 9  # seconds; sub-sample, absorbs float64 ULP noise

#: Fixed component keys; ``combined_hash`` of them is the row's ``content_hash``.
FINGERPRINT_COMPONENTS = ("traces", "timestamps", "geometry", "metadata")


def recording_content_fingerprint(
    analysis_abs_path,
    *,
    electrical_series_path,
) -> dict[str, str]:
    """Return the component-hash dict identifying a persisted recording.

    Opens the path directly, not via ``AnalysisNwbfile.get_abs_path``, whose
    checksum check fails on a byte-different rebuild.

    Parameters
    ----------
    analysis_abs_path : str or pathlib.Path
        Absolute path to the persisted analysis NWB (the just-written or
        being-rebuilt artifact).
    electrical_series_path : str
        In-file path of the preprocessed ElectricalSeries (e.g.
        ``"acquisition/ProcessedElectricalSeries"``).

    Returns
    -------
    dict[str, str]
        ``{"traces", "timestamps", "geometry", "metadata"}`` component hashes.
    """
    import hashlib

    import numpy as np
    import pynwb

    from spyglass.spikesorting.v2._storage.recompute import (
        combined_hash,
        hash_recording_traces,
    )
    from spyglass.spikesorting.v2._storage.nwb import read_recording_nwb
    from spyglass.spikesorting.v2._core.signal_math import (
        _segment_times_slice,
    )
    from spyglass.utils.nwb_hash import (
        electrode_coordinates,
        electrode_geometry_hash,
    )

    analysis_abs_path = str(analysis_abs_path)
    series_name = electrical_series_path.rsplit("/", 1)[-1]

    recording = read_recording_nwb(
        analysis_abs_path,
        electrical_series_path=electrical_series_path,
    )
    # An empty readback would hash to a content-free constant.
    n_segments = recording.get_num_segments()
    if n_segments == 0 or any(
        recording.get_num_frames(segment_index=s) == 0
        for s in range(n_segments)
    ):
        raise ValueError(
            f"recording_content_fingerprint: {analysis_abs_path!r} read back "
            f"with no trace samples (segments={n_segments}) via "
            f"electrical_series_path={electrical_series_path!r}; refusing to "
            "fingerprint empty/partial data."
        )
    traces_hash = combined_hash(
        hash_recording_traces(recording, rounding=TRACE_ROUNDING)
    )

    timestamp_hashes: dict[str, str] = {}
    for segment in range(recording.get_num_segments()):
        digest = hashlib.md5()
        n_frames = recording.get_num_frames(segment_index=segment)
        # Chunked; the digest equals hashing the whole vector at once.
        for start in range(0, n_frames, 300_000):
            times = np.round(
                _segment_times_slice(
                    recording,
                    start,
                    min(start + 300_000, n_frames),
                    segment_index=segment,
                ),
                TIMESTAMP_ROUNDING,
            )
            digest.update(np.ascontiguousarray(times, dtype="<f8").tobytes())
        timestamp_hashes[f"segment_{segment}"] = digest.hexdigest()
    timestamps_hash = combined_hash(timestamp_hashes)

    n_frames = [
        str(int(recording.get_num_frames(segment_index=s)))
        for s in range(recording.get_num_segments())
    ]
    metadata = {
        "sampling_frequency": repr(float(recording.get_sampling_frequency())),
        "channel_ids": ",".join(str(c) for c in recording.get_channel_ids()),
        "dtype": str(recording.get_dtype()),
        "n_channels": str(int(recording.get_num_channels())),
        "n_frames": ",".join(n_frames),
        "electrical_series_path": electrical_series_path,
    }

    with pynwb.NWBHDF5IO(
        analysis_abs_path, mode="r", load_namespaces=True
    ) as io:
        nwbfile = io.read()
        series = nwbfile.acquisition[series_name]
        geometry_hash = electrode_geometry_hash(electrode_coordinates(series))
        metadata["conversion"] = repr(float(series.conversion))
        metadata["offset"] = repr(
            float(series.offset) if series.offset is not None else 0.0
        )
        metadata["filtering"] = series.filtering or ""

    components = {
        "traces": traces_hash,
        "timestamps": timestamps_hash,
        "geometry": geometry_hash,
        "metadata": combined_hash(metadata),
    }
    assert set(components) == set(FINGERPRINT_COMPONENTS), (
        "recording_content_fingerprint produced an unexpected component set: "
        f"{sorted(components)} != {sorted(FINGERPRINT_COMPONENTS)}"
    )
    return components


def recording_artifact_lock(recording_id):
    """Return the cross-process lock for one recording's artifact slot.

    See :func:`._storage.analyzer_cache.artifact_slot_lock`.

    Parameters
    ----------
    recording_id
        The recording whose canonical artifact the caller will mutate.

    Returns
    -------
    filelock.FileLock
        An unacquired lock.
    """
    from spyglass.spikesorting.v2._storage.analyzer_cache import (
        artifact_slot_lock,
    )

    return artifact_slot_lock("recording", recording_id)

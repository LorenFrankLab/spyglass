"""Content fingerprint for the v2 preprocessed recording cache.

The *content fingerprint* is the recording's scientific identity: a
representation-blind hash of what the persisted ``ElectricalSeries`` actually
contains -- traces, timestamps, probe geometry, and the scaling metadata needed
to interpret them. It is reproducible by construction (a content-identical but
byte-different rebuild hashes equal), so one value can drive recompute
``matched``, delete authority, and rebuild reconciliation alike -- unlike the
whole-file ``NwbfileHasher`` digest, which folds volatile per-object ids and
file-creation timestamps into the hash and so can never confirm a rebuild.

DESIGN NOTES.

- **Readback from disk, never caller-supplied.** Every component is read from
  the persisted file at the *known* absolute path -- including
  ``conversion``/``offset``/``filtering``/dtype/shape. Nothing is passed in but
  the two rounding constants, so a write/scaling bug surfaces in the hash rather
  than being masked by a stale caller argument.

- **Bypass the DataJoint ``~external`` checksum on readback.** The helper opens
  the absolute path directly via SpikeInterface / pynwb. It is NEVER resolved
  through ``AnalysisNwbfile.get_abs_path``, whose checksum-validating fallback
  would re-raise -- against the *stale* checksum -- the exact failure the
  content-fingerprint design fixes, before a single byte could be read during a
  rebuild.

- **Geometry from the persisted electrodes region, not ``get_channel_locations``.**
  The persisted ``ElectricalSeries.electrodes`` region is the stable on-disk
  contract; ``get_channel_locations`` is an SI readback surface whose field set
  can shift across versions. The SI surface is used only as a parity check in
  the tests, never as the canonical source.

- **Lineage is deliberately NOT hashed.** The raw ``source_object_id`` the
  recording was built from is provenance, not content -- two byte-identical
  recordings are the same recording regardless of where they came from. The
  ``traces`` component already differs whenever the raw source series differs
  (different data), and for the fingerprint's jobs (rebuild reconcile, recompute
  match, delete authority -- all same-``recording_id``, same-lineage) the source
  id is invariant. ``content_hash`` is never a key (only ever compared within a
  single ``recording_id``), so a cross-recording collision would be harmless
  anyway. Self-describing source provenance, if wanted, belongs in a separate
  provenance workstream -- not folded into a content hash.

- **Rounding is contract.** ``TRACE_ROUNDING`` (µV, below the ephys noise floor)
  and ``TIMESTAMP_ROUNDING`` (seconds, sub-sample) are kept as separate named
  constants. Round-then-hash is deliberate: a false *match* (different
  recordings hash equal) is the dangerous direction and is vanishingly unlikely
  for continuous ephys at this precision; a false *mismatch* (a ULP on a
  rounding boundary) merely skips reclamation and is never unsafe.

DB-LIGHT AT IMPORT. This module activates no ``dj.schema`` and opens no DB
connection at import: numpy / pynwb / SpikeInterface / filelock and the
``_recompute`` helpers are imported lazily inside the functions, so both
``recording.py`` and ``recompute.py`` import it without a cycle.
"""

from __future__ import annotations

TRACE_ROUNDING = 4  # µV; well below the ephys noise floor
TIMESTAMP_ROUNDING = 9  # seconds; sub-sample, absorbs float64 ULP noise

#: Canonical component keys of a recording content fingerprint (a fixed set).
#: ``combined_hash`` of the component dict is the scalar ``content_hash`` stored
#: on the row, so a producer that silently dropped or renamed a component would
#: yield a stable-but-wrong scalar that never matches the stored hash (a silent
#: reclamation stall). The producer asserts its output matches this set so such
#: a key-set drift fails loudly at construction instead.
FINGERPRINT_COMPONENTS = ("traces", "timestamps", "geometry", "metadata")


def geometry_component_hash(coords) -> str:
    """Hash a ``(n_channels, n_coords)`` probe-geometry array.

    Little-endian float64 contiguous bytes, prefixed with the shape so a
    transpose / channel-count change cannot collide. Shared by the fingerprint
    (persisted-region source) and the geometry-parity test
    (``get_channel_locations`` source), so equal geometry from either source
    hashes equal.
    """
    import hashlib

    import numpy as np

    arr = np.ascontiguousarray(np.asarray(coords, dtype=np.float64).astype("<f8"))
    if arr.ndim != 2 or arr.shape[1] == 0:
        raise ValueError(
            "geometry_component_hash: expected a (n_channels, n_coords>=1) "
            f"array but got shape {arr.shape} -- the persisted electrodes "
            "region carries no rel_x/rel_y[/rel_z] coordinates (no probe "
            "geometry). Refusing to hash an empty geometry component."
        )
    digest = hashlib.md5()
    digest.update(str(arr.shape).encode())
    digest.update(arr.tobytes())
    return digest.hexdigest()


def recording_content_fingerprint(
    analysis_abs_path,
    *,
    electrical_series_path,
    trace_rounding: int = TRACE_ROUNDING,
    timestamp_rounding: int = TIMESTAMP_ROUNDING,
) -> dict[str, str]:
    """Return the component-hash dict identifying a persisted recording.

    Opens ``analysis_abs_path`` DIRECTLY (never ``AnalysisNwbfile.get_abs_path``,
    which validates the stale ``~external`` checksum). Every field is read from
    the persisted ``ElectricalSeries``; nothing but the rounding constants is
    caller-supplied.

    Parameters
    ----------
    analysis_abs_path : str or pathlib.Path
        Absolute path to the persisted analysis NWB (the just-written or
        being-rebuilt artifact).
    electrical_series_path : str
        In-file path of the preprocessed ElectricalSeries (e.g.
        ``"acquisition/ProcessedElectricalSeries"``).
    trace_rounding, timestamp_rounding : int
        Decimal precision for the trace (µV) and timestamp (s) components.

    Returns
    -------
    dict[str, str]
        ``{"traces", "timestamps", "geometry", "metadata"}`` component hashes.
        The scalar row identity is ``combined_hash`` of this dict; the named
        components let a recompute diff report which axis drifted.
    """
    import hashlib

    import numpy as np
    import pynwb
    import spikeinterface.extractors as se

    from spyglass.spikesorting.v2._recompute import (
        combined_hash,
        hash_recording_traces,
    )

    analysis_abs_path = str(analysis_abs_path)
    series_name = electrical_series_path.rsplit("/", 1)[-1]

    # --- traces / timestamps / scaling+shape metadata via SpikeInterface ---
    recording = se.read_nwb_recording(
        analysis_abs_path,
        electrical_series_path=electrical_series_path,
        load_time_vector=True,
    )
    # Fail closed on a degenerate readback (zero segments / zero frames): an
    # empty trace loop would hash to a content-free constant, so two distinct
    # broken/truncated readbacks would report "no drift" on a component that was
    # never actually read. Refuse rather than fingerprint partial data.
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
        hash_recording_traces(recording, rounding=trace_rounding)
    )

    timestamp_hashes: dict[str, str] = {}
    for segment in range(recording.get_num_segments()):
        times = np.round(
            np.asarray(
                recording.get_times(segment_index=segment), dtype=np.float64
            ),
            timestamp_rounding,
        )
        digest = hashlib.md5()
        digest.update(np.ascontiguousarray(times.astype("<f8")).tobytes())
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

    # --- geometry + conversion/offset/filtering via the persisted series ---
    with pynwb.NWBHDF5IO(
        analysis_abs_path, mode="r", load_namespaces=True
    ) as io:
        nwbfile = io.read()
        series = nwbfile.acquisition[series_name]
        region = series.electrodes
        table = region.table
        coord_cols = [
            c for c in ("rel_x", "rel_y", "rel_z") if c in table.colnames
        ]
        coords = np.asarray(
            [
                [float(table[col][int(row)]) for col in coord_cols]
                for row in region.data[:]
            ],
            dtype=np.float64,
        )
        geometry_hash = geometry_component_hash(coords)
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
    # Guard the identity contract: the scalar content_hash is combined_hash of
    # this dict, so a dropped/renamed component would silently change it. Fail
    # loudly here rather than store a stable-but-wrong hash.
    assert set(components) == set(FINGERPRINT_COMPONENTS), (
        "recording_content_fingerprint produced an unexpected component set: "
        f"{sorted(components)} != {sorted(FINGERPRINT_COMPONENTS)}"
    )
    return components


def recording_artifact_lock(recording_id, *, timeout: float = -1):
    """Return a cross-process lock serializing one recording's artifact slot.

    Mirrors :func:`._analyzer_cache.analyzer_curation_lock`: a per-
    ``recording_id`` ``filelock.FileLock`` so a rebuild (``get_recording`` read-
    repair / ``_rebuild_nwb_artifact``) and a reclamation
    (``RecordingArtifactRecompute.delete_files``) of the *same* recording can
    never interleave -- no unlink racing a write, no reader seeing a
    half-written HDF5. Different recordings stay free to run in parallel.

    The lock file lives under the shared analyzer/lock root
    (:func:`._analyzer_cache.analyzer_cache_root`), a stable per-install path --
    NOT a per-worker temp -- so all workers on a machine contend on the same
    file. As with ``analyzer_curation_lock`` this serializes processes on ONE
    machine; it does not coordinate across hosts sharing an NFS mount.

    Parameters
    ----------
    recording_id
        The recording whose canonical artifact the caller will mutate.
    timeout : float, optional
        Seconds to wait before raising ``filelock.Timeout``. Default ``-1``
        blocks indefinitely (serialize-don't-fail); the lock releases when the
        holding process exits, so a crashed job cannot wedge the next one.

    Returns
    -------
    filelock.FileLock
        An unacquired lock; use it as a context manager or call ``.acquire()``.
    """
    from filelock import FileLock

    from spyglass.spikesorting.v2._analyzer_cache import analyzer_cache_root

    root = analyzer_cache_root()
    root.mkdir(parents=True, exist_ok=True)
    return FileLock(
        str(root / f"recording_{recording_id}.artifact.lock"), timeout=timeout
    )

"""DB-free content-hashing helpers for v2 recompute verification.

A whole-file ``NwbfileHasher`` digest is NOT reproducible across regenerations
-- it folds in volatile metadata (per-object ``object_id`` attrs, file-creation
timestamps) -- so it can never confirm a content-identical rebuild. These
helpers therefore hash reproducible CONTENT: the preprocessed
``ElectricalSeries`` traces (rounded, little-endian), and the deterministic
SortingAnalyzer extension data. ``hash_recording_traces`` is the ``traces``
building block of the recording ``content_hash`` (see
:mod:`._recording_fingerprint`); ``hash_extension_data`` backs the analyzer
recompute comparison. ``noise_levels`` is excluded from the analyzer comparison
because it estimates noise from unseeded random chunks and is genuinely
stochastic run-to-run.
"""

from __future__ import annotations

import hashlib

import numpy as np

# Deterministic, sort-time analyzer extensions used for recompute comparison.
# ``noise_levels`` is intentionally excluded -- it is an unseeded random-chunk
# noise estimate, not reproducible run-to-run, so including it would make every
# recompute report a spurious mismatch.
ANALYZER_RECOMPUTE_EXTENSIONS = ("random_spikes", "templates", "waveforms")


def hash_extension_data(
    analyzer, extensions=ANALYZER_RECOMPUTE_EXTENSIONS, *, rounding: int = 4
) -> dict[str, str]:
    """Return ``{extension_name: content_hash}`` for an analyzer's extensions.

    Hashes the extension DATA (``get_extension(name).get_data()``), rounding
    float arrays to ``rounding`` decimals, NOT the on-disk files (which carry
    volatile provenance metadata). Only ``extensions`` that are present are
    hashed.
    """
    hashes: dict[str, str] = {}
    for name in extensions:
        if not analyzer.has_extension(name):
            continue
        data = analyzer.get_extension(name).get_data()
        arrays = data if isinstance(data, (tuple, list)) else [data]
        digest = hashlib.md5()
        for array in arrays:
            array = np.asarray(array)
            if array.dtype.kind == "f":
                array = np.round(array, rounding)
            digest.update(np.ascontiguousarray(array).tobytes())
        hashes[name] = digest.hexdigest()
    return hashes


def hash_recording_traces(
    recording, *, rounding: int = 4, chunk_frames: int = 300_000
) -> dict[str, str]:
    """Return ``{segment_i: content_hash}`` of a recording's rounded traces.

    Chunked over frames to bound memory; deterministic given the same
    preprocessing pipeline, raw data, and SpikeInterface version.
    """
    hashes: dict[str, str] = {}
    for segment in range(recording.get_num_segments()):
        digest = hashlib.md5()
        n_frames = recording.get_num_frames(segment_index=segment)
        for start in range(0, n_frames, chunk_frames):
            end = min(start + chunk_frames, n_frames)
            traces = np.asarray(
                recording.get_traces(
                    segment_index=segment, start_frame=start, end_frame=end
                ),
                dtype=np.float64,
            )
            # Serialize explicit little-endian so the digest is byte-stable
            # across architectures (the content-fingerprint contract). traces
            # is float64 above; ``<f8`` is a no-op on x86/ARM but defensive.
            digest.update(
                np.ascontiguousarray(
                    np.round(traces, rounding).astype("<f8")
                ).tobytes()
            )
        hashes[f"segment_{segment}"] = digest.hexdigest()
    return hashes


def combined_hash(hash_dict: dict[str, str]) -> str:
    """Fold a per-object hash dict into one stable 64-char digest."""
    digest = hashlib.sha256()
    for key in sorted(hash_dict):
        digest.update(key.encode())
        digest.update(hash_dict[key].encode())
    return digest.hexdigest()


def compare_hash_dicts(
    old: dict[str, str], new: dict[str, str]
) -> tuple[bool, list[str], list[str], list[str]]:
    """Compare two per-object hash dicts.

    Returns ``(matched, missing_from_old, missing_from_new, differing)`` where
    ``missing_from_old`` are object names present only in ``new`` and vice
    versa, and ``differing`` are names present in both with unequal hashes.
    ``matched`` is True only when all three lists are empty.
    """
    old_keys, new_keys = set(old), set(new)
    missing_from_new = sorted(old_keys - new_keys)
    missing_from_old = sorted(new_keys - old_keys)
    differing = sorted(k for k in old_keys & new_keys if old[k] != new[k])
    matched = not (missing_from_new or missing_from_old or differing)
    return matched, missing_from_old, missing_from_new, differing


def current_nwb_namespaces(abs_path: str) -> dict:
    """Return the pynwb namespace versions embedded in an NWB file."""
    from spyglass.utils.nwb_hash import get_file_namespaces

    deps = dict(get_file_namespaces(abs_path))
    deps.pop("version", None)
    return deps

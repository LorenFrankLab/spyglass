"""Content-hashing helpers for v2 recompute verification.

A whole-file ``NwbfileHasher`` digest includes object ids and creation
timestamps, so it cannot confirm a rebuild. These helpers hash content: rounded
little-endian traces and the seed-pinned analyzer extension data.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pynwb

from spyglass.utils.nwb_hash import get_file_namespaces, get_namespace_versions

# Each digest carries the format version it was computed under.
ANALYZER_CONTENT_HASH_VERSION = 2
ANALYZER_CONTENT_HASH_PREFIX = f"array-v{ANALYZER_CONTENT_HASH_VERSION}:"
_HASH_CHUNK_BYTES = 1024 * 1024


def _copy_named_fields(destination, source) -> None:
    """Copy structured values without copying unspecified alignment bytes."""
    if source.dtype.names is None:
        np.copyto(destination, source, casting="equiv")
        return
    for name in source.dtype.names:
        # Also handles nested records inside subarray fields.
        _copy_named_fields(destination[name], source[name])


def _hash_array(digest, array, *, rounding: int) -> None:
    """Hash array interpretation and C-order values in bounded buffers."""
    array = np.asarray(array)
    if array.dtype.hasobject:
        raise TypeError("Analyzer content hashes require non-object arrays.")
    dtype = array.dtype.newbyteorder("<")
    header = json.dumps(
        {"shape": array.shape, "dtype": np.lib.format.dtype_to_descr(dtype)},
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    # Buffered C-order slices avoid converting a whole memmapped array.
    chunks = np.nditer(
        array,
        flags=["external_loop", "buffered", "zerosize_ok"],
        op_flags=["readonly"],
        op_dtypes=[dtype],
        casting="equiv",
        order="C",
        buffersize=max(1, _HASH_CHUNK_BYTES // max(1, dtype.itemsize)),
    )
    for chunk in chunks:
        if dtype.names is not None:
            # Zero the padding bytes, which may be uninitialized.
            normalized = np.zeros(chunk.shape, dtype=dtype)
            _copy_named_fields(normalized, chunk)
            chunk = normalized
        elif dtype.kind == "f":
            chunk = np.round(chunk, rounding)
        digest.update(memoryview(np.ascontiguousarray(chunk)).cast("B"))


# Sort-time extensions hashed for recompute; all are seed-pinned or
# deterministic, so a rebuild is content-identical.
ANALYZER_RECOMPUTE_EXTENSIONS = (
    "random_spikes",
    "noise_levels",
    "templates",
    "waveforms",
)


def hash_extension_data(
    analyzer, extensions=ANALYZER_RECOMPUTE_EXTENSIONS, *, rounding: int = 4
) -> dict[str, str]:
    """Return ``{extension_name: content_hash}`` for an analyzer's extensions.

    Hashes shapes, little-endian dtypes and values, with floats rounded to
    ``rounding`` decimals. Absent extensions are skipped.
    """
    hashes: dict[str, str] = {}
    for name in extensions:
        if not analyzer.has_extension(name):
            continue
        data = analyzer.get_extension(name).get_data()
        arrays = data if isinstance(data, (tuple, list)) else [data]
        digest = hashlib.sha256()
        digest.update(ANALYZER_CONTENT_HASH_PREFIX.encode())
        digest.update(int(rounding).to_bytes(8, "big", signed=True))
        digest.update(len(arrays).to_bytes(8, "big"))
        for array in arrays:
            _hash_array(digest, array, rounding=rounding)
        hashes[name] = ANALYZER_CONTENT_HASH_PREFIX + digest.hexdigest()
    return hashes


#: Base analyzer extensions ``Sorting.make`` computes, in build order.
BASE_ANALYZER_EXTENSIONS = ANALYZER_RECOMPUTE_EXTENSIONS


def analyzer_seed_modes(analyzer) -> dict[str, object]:
    """Map each present base extension to its effective seed provenance.

    Returns ``{extension: seed}``, or ``"unseeded"`` when the stored params
    hold no seed. Absent extensions are omitted.
    """
    modes: dict[str, object] = {}
    for name in BASE_ANALYZER_EXTENSIONS:
        if not analyzer.has_extension(name):
            continue
        params = analyzer.get_extension(name).params or {}
        seed = params.get("seed")
        if seed is None:
            # ``noise_levels`` nests its seed in ``random_slices_kwargs``.
            seed = (params.get("random_slices_kwargs") or {}).get("seed")
        modes[name] = "unseeded" if seed is None else seed
    return modes


def analyzer_inventory_storage_changed(
    manifest, current_fingerprint: str | None
) -> bool:
    """Whether a versions row describes a different folder generation.

    An empty manifest means an absent or zero-unit analyzer. Path, size and
    mtime drift identify a rebuild without reading extension arrays.
    """
    manifest = manifest or {}
    if current_fingerprint is None and not manifest:
        return False
    return manifest.get("storage_fingerprint") != current_fingerprint


def analyzer_inventory_refresh_needed(
    manifest, current_fingerprint: str | None, *, reclaimed: bool
) -> bool:
    """Whether a versions row must be re-inventoried from the current folder.

    A folder deliberately reclaimed by ``delete_files`` is not refreshed:
    that would cascade away the audit rows that hang off the inventory.
    """
    if reclaimed and current_fingerprint is None:
        return False
    return analyzer_inventory_storage_changed(manifest, current_fingerprint)


def hash_recording_traces(recording, *, rounding: int = 4) -> dict[str, str]:
    """Return ``{segment_i: content_hash}`` of a recording's rounded traces.

    Chunked over 300,000 frames to bound memory; deterministic given the same
    preprocessing pipeline, raw data, and SpikeInterface version.
    """
    hashes: dict[str, str] = {}
    for segment in range(recording.get_num_segments()):
        digest = hashlib.md5()
        n_frames = recording.get_num_frames(segment_index=segment)
        for start in range(0, n_frames, 300_000):
            end = min(start + 300_000, n_frames)
            rounded = np.round(
                np.asarray(
                    recording.get_traces(
                        segment_index=segment, start_frame=start, end_frame=end
                    ),
                    dtype=np.float64,
                ),
                rounding,
            )
            # Little-endian so the digest is stable across architectures.
            digest.update(np.ascontiguousarray(rounded, dtype="<f8"))
        hashes[f"segment_{segment}"] = digest.hexdigest()
    return hashes


def combined_hash(hash_dict: dict[str, str]) -> str:
    """Fold a per-object hash dict into one stable 64-char digest."""
    digest = hashlib.sha256()
    for key in sorted(hash_dict):
        digest.update(key.encode())
        digest.update(hash_dict[key].encode())
    return digest.hexdigest()


#: Metric-analyzer extensions: the base set plus ``principal_components``.
METRIC_ANALYZER_HASH_EXTENSIONS = ANALYZER_RECOMPUTE_EXTENSIONS + (
    "principal_components",
)

#: Extensions hashed per evaluation analyzer role.
_ROLE_HASH_EXTENSIONS = {
    "display": ANALYZER_RECOMPUTE_EXTENSIONS,
    "metric": METRIC_ANALYZER_HASH_EXTENSIONS,
}


def analyzer_hash_for_role(analyzer, role: str) -> str:
    """Content hash of ``analyzer`` over the extensions consumed for ``role``."""
    return ANALYZER_CONTENT_HASH_PREFIX + combined_hash(
        hash_extension_data(analyzer, extensions=_ROLE_HASH_EXTENSIONS[role])
    )


def analyzer_role_hashes(display_analyzer, metric_analyzer=None) -> dict:
    """Return ``{role: content_hash}`` for the canonical analyzers consumed.

    Hashes the Spyglass-parameterized extensions. Derived extensions use SI
    defaults, so the hashed base plus the separately checked SI version cover
    them. ``metric_analyzer=None`` yields only the display entry.
    """
    hashes = {"display": analyzer_hash_for_role(display_analyzer, "display")}
    if metric_analyzer is not None:
        hashes["metric"] = analyzer_hash_for_role(metric_analyzer, "metric")
    return hashes


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
    deps = dict(get_file_namespaces(abs_path))
    deps.pop("version", None)
    return deps


def current_env_namespaces() -> dict:
    """Return the pynwb namespace versions registered in the current process.

    Includes extensions loaded so far this session. Compare with
    :func:`current_nwb_namespaces` via :func:`env_matches`.
    """
    deps = get_namespace_versions(pynwb.get_manager().type_map)
    deps.pop("version", None)
    return deps


def env_matches(file_deps: dict | None, env_deps: dict) -> bool:
    """Whether a file's inventoried namespaces are reproducible in ``env_deps``.

    Compatible iff every shared namespace agrees on version; file-only
    namespaces (extensions load lazily) are ignored. No deps or no shared
    namespace is incompatible. This is a pre-filter: the ``content_hash``
    match decides deletion, and ``force_attempt`` overrides a false negative.
    """
    if not file_deps:
        return False
    shared = set(file_deps) & set(env_deps)
    if not shared:
        return False
    return all(file_deps[ns] == env_deps[ns] for ns in shared)

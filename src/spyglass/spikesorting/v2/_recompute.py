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


#: Base analyzer extensions ``Sorting.make`` computes, in build order. Used to
#: report each extension's seed mode in the recompute manifest (a superset of
#: ``ANALYZER_RECOMPUTE_EXTENSIONS``, which is only the seed-pinned content the
#: recompute hash covers).
BASE_ANALYZER_EXTENSIONS = (
    "random_spikes",
    "noise_levels",
    "templates",
    "waveforms",
)


def analyzer_seed_modes(analyzer) -> dict[str, object]:
    """Map each present base extension to its effective seed provenance.

    Returns ``{extension: seed}`` when the extension's stored params carry an
    explicit, non-``None`` ``seed`` (e.g. the seed-pinned ``random_spikes``
    subsample), and ``{extension: "unseeded"}`` otherwise -- e.g. ``noise_levels``
    (computed without an explicit seed) or any extension whose params hold no
    seed. Surfacing this in the recompute manifest stops it from silently
    implying a pinned seed for an extension that has none. Absent extensions are
    omitted.
    """
    modes: dict[str, object] = {}
    for name in BASE_ANALYZER_EXTENSIONS:
        if not analyzer.has_extension(name):
            continue
        params = analyzer.get_extension(name).params or {}
        seed = params.get("seed")
        modes[name] = "unseeded" if seed is None else seed
    return modes


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


#: Extensions hashed for the whitened METRIC analyzer: the base content set plus
#: ``principal_components``, which PC/NN metrics compute and consume on it. Hashing
#: only the base set would let a principal_components change on a PC/NN evaluation
#: drift undetected. ``hash_extension_data`` skips any extension not present.
METRIC_ANALYZER_HASH_EXTENSIONS = ANALYZER_RECOMPUTE_EXTENSIONS + (
    "principal_components",
)


def analyzer_role_hashes(display_analyzer, metric_analyzer=None) -> dict:
    """Return ``{role: content_hash}`` for the canonical analyzers consumed.

    A ``CurationEvaluation`` always reads the ``"display"`` analyzer and, when it
    requests PC/NN metrics, the whitened ``"metric"`` analyzer. The hash for each
    role covers the SEED/PARAM-DRIVEN content surfaces: the base content
    extensions (``random_spikes``/``templates``/``waveforms`` -- their seed and
    region waveform window are Spyglass-pinned), plus ``principal_components`` for
    the metric role (Spyglass-pinned PCA params that recompute on a param
    mismatch, so it can drift without an SI-version bump).

    The DERIVED display extensions an evaluation also computes
    (``spike_amplitudes``/``template_metrics``/``template_similarity``/...) are
    NOT hashed directly: they are computed once with SpikeInterface DEFAULT params
    (no Spyglass pinning, no param-driven recompute), so they are deterministic
    given the base extensions (hashed) and the SI version (checked separately by
    detect_stale_source) -- covered transitively. This keeps the manifest stable
    (``template_metrics`` is a DataFrame, not cleanly content-hashable) while
    still flagging every realistic drift path.

    The single source of the role -> hashed-extensions mapping, shared by the
    store (make_compute) and compare (detect_stale_source) sides so they cannot
    drift. ``metric_analyzer=None`` (no PC metrics) yields just the display entry.
    """
    hashes = {"display": combined_hash(hash_extension_data(display_analyzer))}
    if metric_analyzer is not None:
        hashes["metric"] = combined_hash(
            hash_extension_data(
                metric_analyzer, extensions=METRIC_ANALYZER_HASH_EXTENSIONS
            )
        )
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
    from spyglass.utils.nwb_hash import get_file_namespaces

    deps = dict(get_file_namespaces(abs_path))
    deps.pop("version", None)
    return deps


def current_env_namespaces() -> dict:
    """Return the pynwb namespace versions registered in the current process.

    Reads the live ``pynwb`` type-map catalog (the base NWB/HDMF stack -- and
    any extension already loaded this session). The counterpart to
    :func:`current_nwb_namespaces`, which reads versions embedded in a file:
    comparing the two (see :func:`env_matches`) decides whether a re-preprocess
    in this environment would write namespace-comparable output.
    """
    import pynwb

    name_cat = pynwb.get_manager().type_map.namespace_catalog
    deps = {
        ns: name_cat.get_namespace(ns).get("version", None)
        for ns in name_cat.namespaces
    }
    deps.pop("version", None)
    return deps


def env_matches(file_deps: dict | None, env_deps: dict) -> bool:
    """Whether a file's inventoried namespaces are reproducible in ``env_deps``.

    Compatible iff every namespace the file and the environment have in COMMON
    agrees on version. Namespaces present only in the file -- e.g. an extension
    not yet registered in the live catalog (extensions load lazily) -- are not
    compared, so an unregistered extension never spuriously fails the gate (the
    lenient half of v1's ``_dicts_match``). Unlike v1, which compared a fixed
    namespace allowlist, this compares ALL shared namespaces and so is slightly
    stricter -- acceptable because the env gate is only a pre-filter, not the
    delete authority (the real authority is the ``content_hash`` recompute
    match): a false *compatible* here merely costs one failed attempt, and a
    false *incompatible* only skips a recomputable artifact (recover with
    ``force_attempt``), never anything unsafe.

    A file with no inventoried deps (absent at inventory time) or no namespace
    in common with the environment is treated as incompatible -- nothing could
    be verified.
    """
    if not file_deps:
        return False
    shared = set(file_deps) & set(env_deps)
    if not shared:
        return False
    return all(file_deps[ns] == env_deps[ns] for ns in shared)

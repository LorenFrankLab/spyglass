"""Path policy for the regeneratable SortingAnalyzer cache.

A SpikeInterface ``SortingAnalyzer`` folder (waveforms, templates, metric
extensions) is large (5-50 GB) regeneratable SCRATCH, not a canonical
database artifact: a valid ``Sorting`` row keeps its FK-guaranteed upstream
``Recording`` / NWB, so the analyzer can always be rebuilt. This module is
the single place that decides WHERE that cache lives, so the location is a
pure function of ``sorting_id`` + one configured root rather than an
absolute path scattered through (or persisted by) the ``Sorting`` table --
which is the path-drift class of bug this replaces.

Resolution (``analyzer_cache_root``):

1. ``dj.config["custom"]["spikesorting_v2_analyzer_dir"]`` when set -- point
   it at shared storage for a persistent cache;
2. otherwise ``Path(temp_dir) / "spikesorting_v2" / "analyzers"`` -- scratch
   semantics under Spyglass's configured temp directory (the default, and
   identical to the path used before this module existed).

Changing the root is an explicit cache-relocation choice: old folders simply
become cache misses (``get_analyzer`` rebuilds into the new root) and can be
cleaned by the operator -- never a stale-row inconsistency.

Storage format. Every canonical cache folder is a SpikeInterface
``binary_folder`` analyzer named ``{sorting_id}__{payload}.analyzer``
(:data:`ANALYZER_FOLDER_SUFFIX`). ``binary_folder`` is the ONLY SI 0.104.3
format whose waveform extraction writes straight into a memmapped
``waveforms.npy`` (``zarr`` and ``memory`` extract into a shared-memory buffer
sized for the whole waveform volume and then copy it), so the extraction peak
is bounded by the worker chunk buffers rather than
``n_spikes * n_samples * n_channels``. :func:`load_analyzer_folder` is the one
loader: it maps ``waveforms.npy`` lazily (``mmap_mode="r"``) instead of SI's
eager ``np.load``, so opening a cache for review or metrics does not read the
whole waveform volume either. Pre-launch ``.zarr`` caches are not read; they
are disposable and simply rebuild under this convention (delete the old
``*.zarr`` folders under the analyzer root by hand).

This module reads ``dj.config`` and ``temp_dir`` but opens no DB connection
and activates no ``dj.schema``; the reads happen at call time so import stays
side-effect free.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

#: Suffix of every canonical analyzer-cache folder (an SI ``binary_folder``
#: store). Not ``.zarr``: SI's loader treats a ``.zarr`` suffix as the zarr
#: format, and the cache is deliberately binary_folder (see the module doc).
ANALYZER_FOLDER_SUFFIX = ".analyzer"

# The recipe name is embedded in the analyzer cache folder
# ``{sorting_id}__{waveform_params_name}.analyzer`` (see ``analyzer_path``), so it
# must be path-safe -- no separators, dots, or traversal. Validated both at
# insert (``AnalyzerWaveformParameters``) and at load (``get_analyzer`` accepts
# an un-FK'd free-string recipe name). Lives here -- the DB-free owner of the
# analyzer-path policy -- so the load path can validate without importing the
# ``sorting`` schema module.
_WAVEFORM_PARAMS_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_CURATION_CACHE_RE = re.compile(
    r"^curation_"
    r"(?P<curation_uuid>[0-9a-f]{32})_"
    r"(?P<role>display|metric)_"
    r"(?P<waveform_recipe_hash>[0-9a-f]{64})_"
    r"si_(?P<spikeinterface_version_hash>[0-9a-f]{16})"
    r"(?:_ext_(?P<extension_request_hash>[0-9a-f]{16}))?$"
)


@dataclass(frozen=True)
class AnalyzerCacheFolderIdentity:
    """Parsed identity of one canonical analyzer-cache folder."""

    kind: str
    sorting_id: uuid.UUID
    waveform_params_name: str | None = None
    curation_uuid: uuid.UUID | None = None
    role: str | None = None
    waveform_recipe_hash: str | None = None
    spikeinterface_version_hash: str | None = None
    extension_request_hash: str | None = None


def analyzer_folder_storage_fingerprint(folder, *, exclude_names=()) -> str:
    """Hash an analyzer folder's file paths, sizes, and mtimes without reads."""
    digest = hashlib.sha256()
    root = Path(folder)
    excluded = set(exclude_names)
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name not in excluded
    )
    for path in files:
        stat = path.stat()
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(stat.st_size.to_bytes(8, "big"))
        digest.update(stat.st_mtime_ns.to_bytes(8, "big"))
    return digest.hexdigest()


def assert_path_safe_waveform_params_name(name) -> None:
    """Reject a ``waveform_params_name`` that is not path-safe.

    It must match ``^[A-Za-z0-9_]+$`` (letters, digits, underscore) because it
    is embedded in the analyzer cache folder name.
    """
    if not _WAVEFORM_PARAMS_NAME_RE.match(str(name)):
        raise ValueError(
            "AnalyzerWaveformParameters: waveform_params_name "
            f"{name!r} is not path-safe; it is embedded in the analyzer "
            "cache folder name, so it must match ^[A-Za-z0-9_]+$ (letters, "
            "digits, underscore)."
        )


def is_canonical_analyzer_folder_name(name: str) -> bool:
    """Whether ``name`` is a canonical analyzer-cache folder name.

    Canonical analyzer folders are
    ``{sorting_id}__{waveform_params_name}.analyzer`` (see :func:`analyzer_path`):
    a UUID sorting id, a ``__`` separator, a path-safe recipe name, and the
    ``.analyzer`` suffix. The disk-side orphan sweep
    uses this to refuse deleting any directory under a (possibly misconfigured)
    analyzer root that is not a canonical analyzer cache.
    """
    return analyzer_cache_folder_identity(name) is not None


def analyzer_cache_folder_identity(
    name: str,
) -> AnalyzerCacheFolderIdentity | None:
    """Parse a raw- or curation-kind canonical cache folder name.

    Raw cache names have the ``{sorting_id}__{waveform_params_name}.analyzer``
    shape. Curation cache names use the same outer shape with a structured,
    path-safe payload carrying the immutable curation generation, analyzer
    role, recipe-content hash, SpikeInterface-version hash and -- for a
    derivative carrying extra extensions -- the extension-request hash.
    Returning a typed identity lets cleanup share one conservative
    canonical-name gate across both cache kinds.
    """
    if not name.endswith(ANALYZER_FOLDER_SUFFIX) or "__" not in name:
        return None
    sorting_id_text, _, payload = name[
        : -len(ANALYZER_FOLDER_SUFFIX)
    ].partition("__")
    try:
        sorting_id = uuid.UUID(sorting_id_text)
    except (ValueError, TypeError):
        return None
    match = _CURATION_CACHE_RE.fullmatch(payload)
    if match is not None:
        return AnalyzerCacheFolderIdentity(
            kind="curation",
            sorting_id=sorting_id,
            curation_uuid=uuid.UUID(hex=match.group("curation_uuid")),
            role=match.group("role"),
            waveform_recipe_hash=match.group("waveform_recipe_hash"),
            spikeinterface_version_hash=match.group(
                "spikeinterface_version_hash"
            ),
            extension_request_hash=match.group("extension_request_hash"),
        )
    if not _WAVEFORM_PARAMS_NAME_RE.fullmatch(payload):
        return None
    return AnalyzerCacheFolderIdentity(
        kind="raw",
        sorting_id=sorting_id,
        waveform_params_name=payload,
    )


# Memoize one ``FileLock`` instance per lock-file path so a same-thread nested
# acquisition (the read path taking the lock while the compute path already
# holds it) is reentrant rather than a self-deadlock -- see
# ``analyzer_cache_lock``. The registry persists for the process lifetime (one
# small entry per sort touched); ``_ANALYZER_LOCK_REGISTRY_GUARD`` serializes
# the get-or-create so two threads cannot mint two instances for one path.
_ANALYZER_LOCK_REGISTRY: dict = {}
_ANALYZER_LOCK_REGISTRY_GUARD = threading.Lock()


def analyzer_cache_root() -> Path:
    """Return the configured root directory for SortingAnalyzer caches.

    ``dj.config["custom"]["spikesorting_v2_analyzer_dir"]`` when truthy,
    else ``Path(temp_dir) / "spikesorting_v2" / "analyzers"``.

    Returns
    -------
    pathlib.Path
        The root directory under which analyzer caches are stored.
    """
    import datajoint as dj

    from spyglass.settings import temp_dir

    custom = dj.config.get("custom") or {}
    configured = custom.get("spikesorting_v2_analyzer_dir")
    if configured:
        return Path(configured)
    return Path(temp_dir) / "spikesorting_v2" / "analyzers"


def analyzer_path(sorting_id, waveform_params_name: str) -> Path:
    """Return the analyzer-cache folder for a ``(sorting_id, recipe)`` pair.

    ``analyzer_cache_root() / f"{sorting_id}__{waveform_params_name}.analyzer"``.
    A sort may have more than one analyzer recipe (an unwhitened display recipe
    and a whitened metric recipe built on demand for PC/NN metrics), so the
    folder is keyed by both the ``sorting_id`` and the ``waveform_params_name``
    that produced it -- the two never collide. Deterministic in ``(sorting_id, waveform_params_name)`` +
    the configured root, so every code path resolves the same folder without
    the ``Sorting`` row needing to store the path.

    ``waveform_params_name`` is embedded in the folder name, so callers must
    pass a path-safe name; the ``AnalyzerWaveformParameters`` insert guard
    validates it (``^[A-Za-z0-9_]+$``) before any row -- and therefore any
    folder -- can use it. The folder is an SI ``binary_folder`` store (see the
    module doc); load it with :func:`load_analyzer_folder`.

    Parameters
    ----------
    sorting_id
        The sorting whose analyzer is cached.
    waveform_params_name : str
        The ``AnalyzerWaveformParameters`` row name that produced the analyzer.

    Returns
    -------
    pathlib.Path
        The analyzer-cache folder path for this ``(sorting_id, recipe)`` pair.
    """
    return analyzer_cache_root() / (
        f"{sorting_id}__{waveform_params_name}{ANALYZER_FOLDER_SUFFIX}"
    )


def load_analyzer_folder(folder, *, recording=None):
    """Load a cached ``binary_folder`` analyzer with memmapped waveforms.

    The single loader for every analyzer folder this package writes. SI's own
    ``load_sorting_analyzer`` eagerly ``np.load``s every saved extension, which
    for ``waveforms`` reads the whole ``n_spikes x n_samples x n_channels``
    buffer into RAM on every open -- tens of GB for a long, unit-rich sort.
    This loader opens the analyzer with ``load_extensions=False``, leaves other
    extensions for SI's on-demand ``get_extension`` loader, and attaches
    ``waveforms`` as a read-only ``np.memmap`` of ``extensions/waveforms/waveforms.npy``. SI's
    consumers index the waveform buffer per unit (``get_waveforms_one_unit``,
    template and PCA fitting), so each of them touches one unit's slice at a
    time, and SI's binary ``_save_data`` already special-cases a memmapped
    ``waveforms`` array (it is never re-written). Numerically nothing changes:
    the same bytes are read, lazily.

    Parameters
    ----------
    folder : path-like
        A canonical or derivative analyzer-cache folder (``binary_folder``).
    recording : spikeinterface.BaseRecording, optional
        Override the recording reference stored in the folder.

    Returns
    -------
    spikeinterface.SortingAnalyzer
    """
    import numpy as np
    import spikeinterface as si
    from spikeinterface.core.sortinganalyzer import get_extension_class

    folder = Path(folder)
    analyzer = si.SortingAnalyzer.load(
        folder,
        recording=recording,
        load_extensions=False,
        format="binary_folder",
    )
    if not analyzer.has_recording():
        # SI can silently load a recordingless analyzer when recording.json is
        # invalid. Treat that as an invalid cache so the normal rebuild path
        # reconstructs the exact recording and artifact exclusions.
        raise ValueError(f"Analyzer recording could not be loaded: {folder}")
    # Keep derivative save/select/merge operations on the same pickle contract
    # as build_analyzer. SI does not persist this flag when loading an extractor.
    analyzer.recording._serializability["json"] = False
    if "waveforms" not in analyzer.get_saved_extension_names():
        return analyzer
    extension = get_extension_class("waveforms")(analyzer)
    extension.load_params()
    extension.load_run_info()
    run_info = extension.run_info
    data_file = extension._get_binary_extension_folder() / "waveforms.npy"
    if (
        run_info is not None and not run_info.get("run_completed", False)
    ) or not data_file.is_file():
        # Mirror SI: an incomplete / dataless extension is "not computed".
        return analyzer
    extension.data["waveforms"] = np.load(data_file, mmap_mode="r")
    analyzer.extensions["waveforms"] = extension
    return analyzer


def load_analyzer_extensions(analyzer):
    """Load all extensions for expert SI editing/export, retaining waveform mmap.

    SI's save/select/merge methods copy only loaded extensions. Internal
    read-only views can stay lazy; public mutable analyzers must carry all data.
    """
    for name in analyzer.get_saved_extension_names():
        analyzer.get_extension(name)
    return analyzer


def copy_analyzer_folder(analyzer, folder):
    """Copy a published cache, including extensions SI has not loaded.

    SI's save_as copies only loaded extensions. Let SI recreate the small
    metadata files (rebasing relative extractor paths), then copy extension
    files directly so waveform/amplitude buffers never need to enter RAM.
    """
    import spikeinterface as si

    recording = (
        analyzer.recording
        if analyzer.has_recording() or analyzer.has_temporary_recording()
        else None
    )
    si.SortingAnalyzer.create_binary_folder(
        folder=folder,
        sorting=analyzer.get_sorting_provenance() or analyzer.sorting,
        recording=recording,
        sparsity=analyzer.sparsity,
        return_in_uV=analyzer.return_in_uV,
        rec_attributes=analyzer.rec_attributes,
        backend_options={},
    )
    shutil.copytree(
        Path(analyzer.folder) / "extensions",
        Path(folder) / "extensions",
        dirs_exist_ok=True,
    )
    return load_analyzer_folder(folder, recording=recording)


def analyzer_extension_array(analyzer, extension_name, data_name):
    """Read a numeric array without loading other data or registering a copy.

    Callers must treat the result as read-only. Keeping this mmap outside SI's
    extension state also prevents its save methods from overwriting the source
    of a mapped array while trying to copy that array.
    """
    import numpy as np

    if (
        analyzer.format != "binary_folder"
        or extension_name in analyzer.extensions
    ):
        return analyzer.get_extension(extension_name).data[data_name]
    return np.load(
        Path(analyzer.folder)
        / "extensions"
        / extension_name
        / f"{data_name}.npy",
        mmap_mode="r",
    )


def analyzer_extension_params(analyzer, name):
    """Read extension parameters without loading a disk-backed payload."""
    from spikeinterface.core.sortinganalyzer import get_extension_class

    if analyzer.format != "binary_folder" or name in analyzer.extensions:
        return analyzer.get_extension(name).params or {}
    extension = get_extension_class(name)(analyzer)
    extension.load_params()
    extension.load_run_info()
    if extension.run_info is not None and not extension.run_info.get(
        "run_completed", False
    ):
        raise ValueError(f"Analyzer extension {name!r} is incomplete.")
    return extension.params or {}


def waveform_recipe_hash(recipe_row) -> str:
    """Return the content fingerprint of one analyzer-waveform recipe row."""
    from spyglass.spikesorting.v2._lookup_validation import _jsonable_blob
    from spyglass.spikesorting.v2._parameter_identity import (
        parameter_fingerprint,
    )

    return parameter_fingerprint(
        "AnalyzerWaveformParameters",
        params=_jsonable_blob(recipe_row["params"]),
        params_schema_version=int(recipe_row["params_schema_version"]),
    )


def _spikeinterface_version_hash(version: str) -> str:
    """Return the path-safe digest component for one SI version string."""
    return hashlib.sha256(str(version).encode("utf-8")).hexdigest()[:16]


def curation_analyzer_path(
    sorting_id,
    curation_uuid,
    role: str,
    waveform_recipe_hash_value: str,
    spikeinterface_version: str,
    extension_request_hash: str | None = None,
) -> Path:
    """Return a per-curation analyzer path keyed by immutable generation.

    The numeric ``curation_id`` is intentionally absent because it may be
    reused after deletion. Distinct curation generations, recipe content,
    analyzer roles, and SpikeInterface versions therefore cannot share a cache
    slot even when their human-readable recipe name is the same.
    ``extension_request_hash`` (16 hex chars) names a DERIVATIVE of the base
    cache that carries extra extensions with specific parameters (see
    ``_curation_analyzer.derived_extension_request_hash``); ``None`` is the
    base cache itself.
    """
    sorting_uuid = uuid.UUID(str(sorting_id))
    generation_uuid = uuid.UUID(str(curation_uuid))
    if role not in {"display", "metric"}:
        raise ValueError(
            "curation analyzer role must be 'display' or 'metric'; "
            f"got {role!r}."
        )
    recipe_hash = str(waveform_recipe_hash_value)
    if not _HASH_RE.fullmatch(recipe_hash):
        raise ValueError(
            "waveform_recipe_hash must be a 64-character lowercase SHA-256 "
            f"digest; got {waveform_recipe_hash_value!r}."
        )
    payload = (
        f"curation_{generation_uuid.hex}_{role}_{recipe_hash}_si_"
        f"{_spikeinterface_version_hash(spikeinterface_version)}"
    )
    if extension_request_hash is not None:
        if not re.fullmatch(r"[0-9a-f]{16}", str(extension_request_hash)):
            raise ValueError(
                "extension_request_hash must be 16 lowercase hex characters; "
                f"got {extension_request_hash!r}."
            )
        payload += f"_ext_{extension_request_hash}"
    return (
        analyzer_cache_root()
        / f"{sorting_uuid}__{payload}{ANALYZER_FOLDER_SUFFIX}"
    )


def analyzer_cache_lock(sorting_id):
    """Return a cross-process lock serializing analyzer-cache access per sort.

    Every canonical analyzer-cache folder reader/writer/deleter holds this lock:
    the populate build, the self-healing read/rebuild path
    (``Sorting.get_analyzer`` / ``load_or_rebuild_analyzer``), the atomic
    publish, ``remove_analyzer_cache``, the recompute delete gate, and the
    ``CurationEvaluation`` raw-sort fast path (which loads the shared analyzer
    and persists extensions + ``quality_metrics`` into it). Holding the lock
    around the load/mutate/publish region lets ONE job touch a sort's analyzer
    at a time, so a concurrent build never corrupts the shared store and a
    reader never observes the brief move-aside window of an atomic publish.

    The lock is keyed on ``sorting_id`` (which every recipe folder of that sort
    shares -- both the display and the whitened-metric analyzer), so it
    serializes one sort's jobs against each other while leaving jobs for
    DIFFERENT sorts free to run in parallel. The lock file lives under
    ``analyzer_cache_root()`` next to the folders it guards.

    **Reentrant per process.** The read path acquires this lock while the
    compute path already holds it for the same sort (the fast path loads inside
    its own ``with`` block). A fresh ``FileLock`` per call would self-deadlock
    there, so the instance is memoized per lock-file path: filelock's instance
    counter makes a same-thread nested acquisition reentrant (instant), while a
    DIFFERENT thread or a DIFFERENT process still contends on the OS-level lock.
    ``populate(processes=N)`` (the standard parallel case) uses separate
    processes, so cross-job serialization is preserved.

    Shared-storage deployments must provide cross-host POSIX file locking on
    this directory (including the lock files). Validate the server/client mount
    configuration on the deployment hosts; local-filesystem tests do not prove
    that contract. Lock-acquisition errors propagate rather than permitting an
    unprotected write. Scientific result ownership is established separately by
    the Sorting insert: a duplicate compute cannot publish over its winner.

    The lock blocks indefinitely by default (the intended serialize-don't-fail
    behavior); it releases when the holding process exits, so a crashed job
    cannot wedge the next one. A per-call timeout is available via
    ``analyzer_cache_lock(sorting_id).acquire(timeout=...)`` -- the memoized
    instance has no constructor-level timeout knob because a second caller would
    silently inherit the first's value.

    Parameters
    ----------
    sorting_id
        The sorting whose analyzer-cache folders the job will access.

    Returns
    -------
    filelock.FileLock
        A memoized, reentrant lock; use it as a context manager or call
        ``.acquire()``.
    """
    from filelock import FileLock

    root = analyzer_cache_root()
    root.mkdir(parents=True, exist_ok=True)
    lock_path = str(root / f"{sorting_id}.analyzer.lock")
    with _ANALYZER_LOCK_REGISTRY_GUARD:
        lock = _ANALYZER_LOCK_REGISTRY.get(lock_path)
        if lock is None:
            lock = FileLock(lock_path)
            _ANALYZER_LOCK_REGISTRY[lock_path] = lock
        return lock


def _publish_sibling(canonical_folder, kind: str) -> Path:
    """Return a hidden sibling of ``canonical_folder`` for staging.

    The build/move-aside folders MUST sit in the SAME directory as the canonical
    slot (not a sub-directory): a SpikeInterface ``SortingAnalyzer`` folder
    stores its recording reference as a path RELATIVE to the analyzer folder
    (``recording.json`` is dumped ``relative_to=folder``), so the publish
    ``os.replace`` only preserves that reference -- and therefore the ability
    to (re)compute recording-dependent extensions like ``spike_amplitudes``
    after a load -- when the temp and the canonical slot are at the same depth
    relative to the recording. A leading ``.`` distinguishes staging from
    canonical caches. A UUID avoids collisions between workers on different
    hosts, where process IDs can coincide.
    """
    parent = canonical_folder.parent
    return parent / (
        f".{canonical_folder.stem}.{kind}-{uuid.uuid4().hex}"
        f"{canonical_folder.suffix}"
    )


def _sorting_id_from_analyzer_folder(canonical_folder) -> str:
    """Return the sorting id embedded in a canonical analyzer folder path."""
    folder = Path(canonical_folder)
    if (
        not folder.name.endswith(ANALYZER_FOLDER_SUFFIX)
        or "__" not in folder.name
    ):
        raise ValueError(
            "publish_analyzer_atomically requires a canonical analyzer-cache "
            "folder named '{sorting_id}__{payload}"
            f"{ANALYZER_FOLDER_SUFFIX}'; got {folder.name!r}."
        )
    sorting_id, _, payload = folder.name[
        : -len(ANALYZER_FOLDER_SUFFIX)
    ].partition("__")
    if not sorting_id or not (
        _WAVEFORM_PARAMS_NAME_RE.match(payload)
        or _CURATION_CACHE_RE.fullmatch(payload)
    ):
        raise ValueError(
            "publish_analyzer_atomically requires a canonical analyzer-cache "
            "folder named '{sorting_id}__{payload}"
            f"{ANALYZER_FOLDER_SUFFIX}' with a path-safe waveform_params_name "
            f"or a curation-cache payload; got {folder.name!r}."
        )
    return sorting_id


class StagedAnalyzer:
    """Attempt-owned analyzer, kept private until its Sorting insert succeeds.

    The ownership lock spans compute and insert. Cleanup checks it without
    blocking, so even a long wait for a database transaction cannot make an
    active attempt look abandoned. Process death releases the OS lock; the
    explicit cache audit can then reclaim the directory. ``close`` releases
    ownership on normal completion or failure, including failed publication.
    """

    def __init__(self, canonical_folder):
        from filelock import FileLock

        self.canonical_folder = Path(canonical_folder)
        self.sorting_id = _sorting_id_from_analyzer_folder(canonical_folder)
        self.canonical_folder.parent.mkdir(parents=True, exist_ok=True)
        self.folder = _publish_sibling(self.canonical_folder, "build")
        self._lock = FileLock(str(self.folder) + ".lock")
        self._lock.acquire()

    def publish(self):
        """Install the completed attempt; the caller establishes DB ownership."""
        with analyzer_cache_lock(self.sorting_id):
            _install_staged_analyzer(self.canonical_folder, self.folder)
        return self.canonical_folder

    def close(self):
        """Discard this attempt's staging only; never remove a published cache."""
        try:
            try:
                if self.folder.exists():
                    shutil.rmtree(self.folder)
            finally:
                self._lock.release()
            Path(self._lock.lock_file).unlink(missing_ok=True)
        except OSError as exc:
            from spyglass.utils import logger

            logger.warning(
                "Analyzer staging cleanup failed for %s: %s", self.folder, exc
            )

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()


def publish_analyzer_atomically(canonical_folder, build_into):
    """Build an analyzer into a private temp folder, then move it into the slot.

    The low-level build (``build_into``) writes into a FRESH private folder that
    is a HIDDEN SIBLING of ``canonical_folder`` (same parent directory, so the
    move is a rename AND the analyzer's relative recording path survives it --
    see :func:`_publish_sibling`); only on success is that temp moved into the
    canonical slot. This is the canonical-cache safety layer that keeps a build
    from writing straight into (and corrupting) the live folder a reader might be
    loading -- distinct from ``build_into`` itself, which simply builds wherever
    it is told (recompute / merged-curation temp analyzers reuse the low-level
    build directly, never this publisher).

    A directory rename is NOT a clean atomic swap: POSIX ``rename(2)`` requires
    the destination to be empty (else ``ENOTEMPTY``), so a rebuild over an
    existing folder cannot ``os.replace`` straight onto it. The publish
    sequence is therefore:

    - canonical **absent**  -> ``os.replace(temp, canonical)``;
    - canonical **present** -> ``os.replace(canonical, trash)``,
      ``os.replace(temp, canonical)``, ``rmtree(trash)``. If the install move
      fails, the original is moved back from trash so the slot is restored; if
      THAT restore also fails (a rare double failure), the trash is LEFT on disk
      (the only surviving copy) for manual recovery -- the slot may be absent
      but is recoverable.

    The brief window where the canonical slot is moved aside is reader-safe
    because this publisher acquires ``analyzer_cache_lock(sorting_id)`` (the read
    path takes the same lock); the directory replace is not itself atomic.
    Callers that already hold the lock remain safe because the lock is
    process-reentrant.
    ``build_into`` that writes no folder (a zero-unit sort short-circuits before
    building) leaves the slot untouched. The temp is always removed; the trash
    is removed only after a successful install (never on the failed-rollback
    path, where it is the surviving copy).

    Parameters
    ----------
    canonical_folder : pathlib.Path
        The cache slot to publish into (``analyzer_path(sorting_id, recipe)``).
    build_into : callable
        ``build_into(temp_folder: pathlib.Path) -> None`` builds the analyzer
        into the given private temp folder (it may legitimately write nothing
        for a zero-unit sort).

    Returns
    -------
    pathlib.Path
        ``canonical_folder``.
    """
    canonical_folder = Path(canonical_folder)
    sorting_id = _sorting_id_from_analyzer_folder(canonical_folder)
    with (
        analyzer_cache_lock(sorting_id),
        StagedAnalyzer(canonical_folder) as staged,
    ):
        build_into(staged.folder)
        _install_staged_analyzer(canonical_folder, staged.folder)
        return canonical_folder


def _install_staged_analyzer(canonical_folder, temp_folder):
    """Rename a completed build into place under ``analyzer_cache_lock``."""
    if not temp_folder.exists():
        return  # Zero-unit sorting: no analyzer was built.
    if not canonical_folder.exists():
        os.replace(temp_folder, canonical_folder)
        return
    trash_folder = _publish_sibling(canonical_folder, "trash")
    os.replace(canonical_folder, trash_folder)
    try:
        os.replace(temp_folder, canonical_folder)
    except BaseException:
        # Preserve trash if rollback itself fails: it is the surviving copy.
        os.replace(trash_folder, canonical_folder)
        raise
    shutil.rmtree(trash_folder, ignore_errors=True)


def cleanup_analyzer_staging(
    sorting_id=None, *, dry_run=True, candidates=None
) -> list[str]:
    """Report/remove abandoned build and trash folders, skipping active owners.

    Both locks are nonblocking: a cache publisher holds the sort lock, and a
    Sorting compute holds its build lock until insertion finishes. No PID or
    age heuristic is needed, including on supported shared filesystems. Legacy
    PID-named folders are recognized as well. Unrelated hidden directories are
    never candidates. ``candidates`` limits a confirmed deletion to the paths
    previously presented to the operator; ownership is still rechecked.
    """
    from filelock import FileLock, Timeout

    root = analyzer_cache_root()
    abandoned = []
    allowed = None if candidates is None else set(map(str, candidates))
    pattern = (
        ".*.analyzer" if sorting_id is None else f".{sorting_id}__*.analyzer"
    )
    for folder in sorted(root.glob(pattern)):
        if allowed is not None and str(folder) not in allowed:
            continue
        match = re.fullmatch(
            r"\.(.+)\.(?:build|trash)-([0-9a-f]+)\.analyzer", folder.name
        )
        if not match or not folder.is_dir():
            continue
        identity = analyzer_cache_folder_identity(
            match[1] + ANALYZER_FOLDER_SUFFIX
        )
        if identity is None or (
            sorting_id is not None
            and str(identity.sorting_id) != str(sorting_id)
        ):
            continue
        owner = FileLock(str(folder) + ".lock")
        try:
            with analyzer_cache_lock(identity.sorting_id).acquire(timeout=0):
                with owner.acquire(timeout=0):
                    if folder.exists():
                        abandoned.append(str(folder))
                        if not dry_run:
                            shutil.rmtree(folder)
                Path(owner.lock_file).unlink(missing_ok=True)
        except Timeout:
            continue
    return abandoned


def remove_analyzer_cache(sorting_id, *, missing_ok: bool = True) -> bool:
    """Remove ALL analyzer-cache folders for a ``sorting_id``.

    A sort can have several analyzer folders on disk (the display and metric
    recipes, per-curation caches and their derivatives), so this removes every
    ``{sorting_id}__*.analyzer`` folder under the cache root -- deleting the
    sort orphans every one. The glob is anchored on the full
    ``sorting_id`` (a fixed-length UUID) followed by the ``__`` separator, so
    one sort's folders never match another's.

    ``missing_ok=True`` (default) makes the no-folders case a no-op returning
    ``False`` (the common case: zero-unit sorts never wrote one, and the cache
    is regeneratable). ``missing_ok=False`` raises ``FileNotFoundError`` when no
    folder exists. A removal failure (e.g. a permission error) propagates
    rather than being swallowed.

    Parameters
    ----------
    sorting_id
        The sorting whose analyzer-cache folders should be removed.
    missing_ok : bool, optional
        If ``True`` (the default), no matching folder is a no-op. If
        ``False``, it raises ``FileNotFoundError``.

    Returns
    -------
    bool
        ``True`` if at least one folder was removed, ``False`` if none existed
        and ``missing_ok=True``.

    Raises
    ------
    FileNotFoundError
        If no matching folder exists and ``missing_ok=False``.
    """
    root = analyzer_cache_root()
    pattern = f"{sorting_id}__*{ANALYZER_FOLDER_SUFFIX}"
    with analyzer_cache_lock(sorting_id):
        folders = sorted(root.glob(pattern)) if root.exists() else []
        staging = cleanup_analyzer_staging(sorting_id, dry_run=False)
        if not folders and not staging:
            if missing_ok:
                return False
            raise FileNotFoundError(root / pattern)
        for folder in folders:
            shutil.rmtree(folder, ignore_errors=False)
    return True


def collect_analyzer_cache_references(sorting_table) -> dict:
    """Collect live analyzer references for the supplied sorting relation.

    This is the single DB-backed reference collector for the analyzer cache.
    Raw references cover each sort's display recipe and PC-requesting
    evaluation metric recipes. Curation references cover the per-generation
    display cache for every live committed curation outside the raw namespace,
    plus its PC-requesting metric recipes. The caller supplies ``sorting_table``
    to keep this module free of schema activation at import time. Its
    restriction applies to all curation, evaluation, and reclamation queries.

    Missing curation folders are not DB-side orphans: they are lazy,
    regeneratable caches built on first interactive use. ``units_bearing``
    therefore continues to describe only raw sort display analyzers, whose
    absence is operationally useful to report.
    """
    import spikeinterface as si

    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.metric_curation import (
        CurationEvaluationSelection,
    )
    from spyglass.spikesorting.v2.recompute import SortingAnalyzerRecompute
    from spyglass.spikesorting.v2.sorting import AnalyzerWaveformParameters

    sorting_keys = sorting_table.proj()
    curations = CurationV2 & sorting_keys
    units_bearing = []
    sorting_rows = sorting_table.fetch(
        "sorting_id", "display_waveform_params_name", "n_units", as_dict=True
    )
    referenced_paths: set[str] = set()
    for row in sorting_rows:
        path = analyzer_path(
            row["sorting_id"], row["display_waveform_params_name"]
        )
        referenced_paths.add(str(path))
        if int(row["n_units"]) > 0:
            units_bearing.append((row["sorting_id"], str(path), path.exists()))

    # Join the selection to its CurationV2 row so each PC-requesting
    # evaluation carries the generation UUID it belongs to: the FK guarantees
    # a live selection has a curation, and one relational fetch cannot see a
    # selection and its curation in different states.
    pc_rows = (
        CurationEvaluationSelection.pc_requesting()
        * curations.proj("curation_uuid")
    ).fetch(
        "sorting_id",
        "curation_id",
        "curation_uuid",
        "metric_waveform_params_name",
        as_dict=True,
    )
    recipe_hashes: dict[str, str] = {}

    def _recipe_hash(name: str) -> str:
        if name not in recipe_hashes:
            recipe_row = (
                AnalyzerWaveformParameters & {"waveform_params_name": name}
            ).fetch1()
            recipe_hashes[name] = waveform_recipe_hash(recipe_row)
        return recipe_hashes[name]

    display_recipe_by_sorting = {
        str(row["sorting_id"]): row["display_waveform_params_name"]
        for row in sorting_rows
    }
    curation_rows = curations.fetch(
        "sorting_id", "curation_id", "curation_uuid", as_dict=True
    )
    for row in curation_rows:
        sorting_id = str(row["sorting_id"])
        curation_id = int(row["curation_id"])
        key = {"sorting_id": row["sorting_id"], "curation_id": curation_id}
        if not (CurationV2.Unit & key) or not CurationV2.is_committed_curation(
            key
        ):
            continue
        # A raw-namespace curation has no base cache of its own (it shares the
        # sort analyzer), but a parameter-specific DERIVATIVE of it is keyed by
        # its generation; referencing the would-be base path lets
        # ``derivative_base_path`` recognize such derivatives as live.
        recipe_name = display_recipe_by_sorting[sorting_id]
        referenced_paths.add(
            str(
                curation_analyzer_path(
                    row["sorting_id"],
                    row["curation_uuid"],
                    "display",
                    _recipe_hash(recipe_name),
                    si.__version__,
                )
            )
        )

    for row in pc_rows:
        recipe_name = row["metric_waveform_params_name"]
        key = {
            "sorting_id": row["sorting_id"],
            "curation_id": int(row["curation_id"]),
        }
        if not (CurationV2.Unit & key) or not CurationV2.is_committed_curation(
            key
        ):
            continue
        if CurationV2.matches_raw_namespace(key):
            referenced_paths.add(
                str(analyzer_path(row["sorting_id"], recipe_name))
            )
            continue
        referenced_paths.add(
            str(
                curation_analyzer_path(
                    row["sorting_id"],
                    row["curation_uuid"],
                    "metric",
                    _recipe_hash(recipe_name),
                    si.__version__,
                )
            )
        )

    reclaimed_paths = {
        str(analyzer_path(row["sorting_id"], row["waveform_params_name"]))
        for row in (
            SortingAnalyzerRecompute & sorting_keys & "deleted=1"
        ).fetch("sorting_id", "waveform_params_name", as_dict=True)
    }
    return {
        "units_bearing": units_bearing,
        "referenced_paths": referenced_paths,
        "reclaimed_paths": reclaimed_paths,
    }


def derivative_base_path(path) -> str:
    """Return the base-cache path of a derivative folder path (or itself)."""
    folder = Path(path)
    identity = analyzer_cache_folder_identity(folder.name)
    if identity is None or identity.extension_request_hash is None:
        return str(folder)
    base_name = folder.name.replace(
        f"_ext_{identity.extension_request_hash}", "", 1
    )
    return str(folder.with_name(base_name))


def classify_orphaned_analyzer_folders(
    units_bearing,
    referenced_paths,
    disk_dir_paths,
    reclaimed_paths=(),
) -> dict:
    """Classify analyzer-folder leaks into DB-side, disk-side, and reclaimed.

    Pure (DB-free) set logic for ``Sorting.find_orphaned_analyzer_folders``; the
    caller gathers the DB / filesystem facts (which sorting_ids bear units,
    whether each computed folder exists, which folders sit under the analyzer
    root, and which missing folders were intentionally reclaimed) and this
    function classifies them.

    A **DB-side orphan** is a units-bearing ``Sorting`` row whose computed
    analyzer folder is gone on disk (regeneratable scratch removed out of band);
    reported only, never auto-deleted. A **reclaimed** folder is the same missing
    row-side path, but with a recompute ``deleted=1`` audit trail showing the
    absence was intentional. A **disk-side orphan** is an on-disk folder under
    the analyzer root that no row references (the row was deleted via a path that
    bypassed the ``Sorting.delete`` override). Input order of ``units_bearing``
    and ``disk_dir_paths`` is preserved.

    Parameters
    ----------
    units_bearing : iterable of (sorting_id, computed_path, exists)
        The ``n_units > 0`` rows: each ``sorting_id``, its computed analyzer
        path (as a string), and whether that path currently exists on disk.
    referenced_paths : iterable of str
        Computed analyzer paths of EVERY ``Sorting`` row (any ``n_units``); a
        disk folder in this set is referenced and not an orphan.
    disk_dir_paths : iterable of str
        Directory paths found directly under the analyzer root.
    reclaimed_paths : iterable of str, optional
        Computed analyzer paths intentionally removed by the recompute deletion
        workflow (``deleted=1``). Missing units-bearing rows in this set are
        reported separately from unexpected DB-side orphans.

    Returns
    -------
    dict
        ``{"db_side": [{"sorting_id", "computed_analyzer_path"}, ...],
        "disk_side": [folder_path_str, ...],
        "reclaimed": [{"sorting_id", "computed_analyzer_path"}, ...]}``.
    """
    reclaimed_set = set(reclaimed_paths)
    db_side = []
    reclaimed = []
    for sorting_id, computed_path, exists in units_bearing:
        if exists:
            continue
        row = {
            "sorting_id": sorting_id,
            "computed_analyzer_path": computed_path,
        }
        if computed_path in reclaimed_set:
            reclaimed.append(row)
        else:
            db_side.append(row)
    referenced = set(referenced_paths)
    # A derivative folder (``..._ext_{hash}.analyzer``) is referenced exactly
    # when its base cache is: derivatives are keyed by the same live curation
    # generation, so they are never enumerated separately by the collector.
    disk_side = [
        path
        for path in disk_dir_paths
        if path not in referenced
        and derivative_base_path(path) not in referenced
    ]
    return {"db_side": db_side, "disk_side": disk_side, "reclaimed": reclaimed}

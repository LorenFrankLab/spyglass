"""Path policy for the regeneratable SortingAnalyzer cache.

Analyzer folders (5-50 GB) are rebuildable scratch, so their path is derived
from ``sorting_id`` and one configured root rather than stored. Changing the
root turns old folders into cache misses.

Folders are SI ``binary_folder`` analyzers, the only SI 0.104.3 format that
extracts waveforms straight into a memmap; :func:`load_analyzer_folder` maps
them lazily.
"""

from __future__ import annotations

from spyglass.spikesorting.v2.exceptions import SpikeSortingV2Error
from spyglass.spikesorting.v2._core.parameter_identity import (
    parameter_fingerprint,
)
from spyglass.spikesorting.v2._storage.spikeinterface import (
    attach_memmapped_waveforms,
    saved_extension_data_file,
    saved_extension_metadata,
    use_pickle_recording_serialization,
)

import hashlib
import json
import os
import re
import shutil
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
import errno

import datajoint as dj
import numpy as np

#: Not ``.zarr``: SI would load it as zarr.
ANALYZER_FOLDER_SUFFIX = ".analyzer"
ANALYZER_SPARSITY_RECEIPT = "spyglass_sparsity.json"


def write_analyzer_sparsity_receipt(folder, *, peak_sign: str) -> None:
    """Record the polarity that actually selected a sparse analyzer's channels."""
    (Path(folder) / ANALYZER_SPARSITY_RECEIPT).write_text(
        json.dumps({"version": 1, "peak_sign": peak_sign}, sort_keys=True)
        + "\n"
    )


def assert_analyzer_sparsity_polarity(
    analyzer, *, expected_peak_sign: str
) -> None:
    """Compare cached polarity with the sorter's already-resolved peak sign.

    Dense analyzers dropped no channels and are not checked.

    Raises
    ------
    ValueError
        If the sparse analyzer has no receipt, an unknown receipt, or a
        recorded polarity other than ``expected_peak_sign``.
    """
    if not analyzer.is_sparse():
        return
    receipt = Path(analyzer.folder) / ANALYZER_SPARSITY_RECEIPT
    if not receipt.exists():
        raise ValueError(
            f"Sparse analyzer has no sparsity receipt ({receipt}); its "
            "channel polarity is unknown. Rebuild the analyzer."
        )
    recorded = json.loads(receipt.read_text())
    if not isinstance(recorded, dict) or recorded.get("version") != 1:
        raise ValueError(f"Unknown analyzer sparsity receipt: {receipt}")
    actual = recorded.get("peak_sign")
    if actual != expected_peak_sign:
        raise ValueError(
            "Analyzer sparsity polarity is incompatible with its sorter: "
            f"cached={actual!r}, expected={expected_peak_sign!r}. "
            "Rebuild the analyzer to retain the sorter's peak channels."
        )


# The recipe name is part of the folder name, so it must be path-safe.
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

    The orphan sweep deletes only canonical folders, in case the root is
    misconfigured.
    """
    return analyzer_cache_folder_identity(name) is not None


def analyzer_cache_folder_identity(
    name: str,
) -> AnalyzerCacheFolderIdentity | None:
    """Parse a raw- or curation-kind canonical cache folder name.

    Raw: ``{sorting_id}__{waveform_params_name}.analyzer``. Curation: the same
    shape with a payload of curation uuid, role, recipe hash, SI-version hash
    and optional extension-request hash. Returns ``None`` if not canonical.
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


def v2_storage_root(custom_key: str, name: str) -> Path:
    """Return a durable v2 storage folder.

    ``dj.config["custom"][custom_key]`` when set, else
    ``Path(base_dir) / "spikesorting_v2" / name``. Not under ``temp_dir``,
    whose week-old files the lab cleanup script deletes.
    """
    from spyglass.settings import base_dir

    configured = (dj.config.get("custom") or {}).get(custom_key)
    if configured:
        return Path(configured)
    return Path(base_dir) / "spikesorting_v2" / name


def analyzer_cache_root() -> Path:
    """Return the root directory for SortingAnalyzer caches.

    ``dj.config["custom"]["spikesorting_v2_analyzer_dir"]`` when set, else
    ``Path(base_dir) / "spikesorting_v2" / "analyzers"``.
    """
    return v2_storage_root("spikesorting_v2_analyzer_dir", "analyzers")


def analyzer_path(sorting_id, waveform_params_name: str) -> Path:
    """Return the analyzer-cache folder for a ``(sorting_id, recipe)`` pair.

    ``analyzer_cache_root() / f"{sorting_id}__{waveform_params_name}.analyzer"``.
    A sort can have display and metric analyzers, hence the recipe name.

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


class AnalyzerRecordingUnavailableError(SpikeSortingV2Error, ValueError):
    """A cached analyzer folder loaded without its recording."""


def load_analyzer_folder(folder, *, recording=None):
    """Load a cached ``binary_folder`` analyzer with memmapped waveforms.

    SI's loader reads all waveforms into RAM (tens of GB for a long sort).
    Here ``waveforms`` is a read-only memmap; other extensions load on demand.

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
    import spikeinterface as si

    folder = Path(folder)
    analyzer = si.SortingAnalyzer.load(
        folder,
        recording=recording,
        load_extensions=False,
        format="binary_folder",
    )
    if not analyzer.has_recording():
        # SI silently loads without a recording when it cannot open one;
        # treat that as invalid so the cache is rebuilt.
        raise AnalyzerRecordingUnavailableError(
            f"Analyzer recording could not be loaded: {folder}"
        )
    # SI does not persist this flag on load.
    use_pickle_recording_serialization(analyzer.recording)
    attach_memmapped_waveforms(analyzer)
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

    SI's ``save_as`` copies only loaded extensions, so extension files are
    copied directly without entering RAM. Uses the analyzer's own sorting,
    whose estimated rate (e.g. 30000.000000000182 Hz) differs in the last bits
    from the provenance units' 30000.0.
    """
    import spikeinterface as si

    recording = (
        analyzer.recording
        if analyzer.has_recording() or analyzer.has_temporary_recording()
        else None
    )
    si.SortingAnalyzer.create_binary_folder(
        folder=folder,
        sorting=analyzer.sorting,
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
    receipt = Path(analyzer.folder) / ANALYZER_SPARSITY_RECEIPT
    if receipt.is_file():
        shutil.copyfile(receipt, Path(folder) / ANALYZER_SPARSITY_RECEIPT)
    return load_analyzer_folder(folder, recording=recording)


def analyzer_extension_array(analyzer, extension_name, data_name):
    """Read a numeric array without loading other data or registering a copy.

    Read-only. Kept out of SI's extension state so SI's save cannot overwrite
    the mapped source.
    """
    if (
        analyzer.format != "binary_folder"
        or extension_name in analyzer.extensions
    ):
        return analyzer.get_extension(extension_name).data[data_name]
    return np.load(
        saved_extension_data_file(analyzer, extension_name, data_name),
        mmap_mode="r",
    )


def analyzer_extension_params(analyzer, name):
    """Read extension parameters without loading a disk-backed payload."""
    if analyzer.format != "binary_folder" or name in analyzer.extensions:
        return analyzer.get_extension(name).params or {}
    extension, completed = saved_extension_metadata(analyzer, name)
    if not completed:
        raise ValueError(f"Analyzer extension {name!r} is incomplete.")
    return extension.params or {}


def waveform_recipe_hash(recipe_row) -> str:
    """Return the content fingerprint of one analyzer-waveform recipe row."""
    from spyglass.spikesorting.v2._core.lookup_validation import _jsonable_blob

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

    Omits ``curation_id``, which may be reused after deletion.
    ``extension_request_hash`` (16 hex chars) names a derivative with extra
    extensions; ``None`` is the base cache.
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

    Every reader, writer and deleter of a sort's analyzer folders holds it, so
    no one sees a half-built folder or a publish's move-aside window.
    ``is_singleton=True`` makes nested acquisition in one thread reentrant;
    other threads and processes still contend. Blocks indefinitely; pass a
    timeout to ``.acquire`` instead (filelock forbids a second constructor
    value). Shared storage must support cross-host POSIX locks.

    Parameters
    ----------
    sorting_id
        The sorting whose analyzer-cache folders the job will access.

    Returns
    -------
    filelock.FileLock
        The per-path singleton, reentrant lock; use it as a context manager
        or call ``.acquire()``.
    """
    from filelock import FileLock

    root = analyzer_cache_root()
    root.mkdir(parents=True, exist_ok=True)
    return FileLock(
        str(root / f"{sorting_id}.analyzer.lock"), is_singleton=True
    )


def artifact_slot_lock(prefix: str, slot_id):
    """Return a cross-process lock serializing one trace artifact's slot.

    Keeps a rebuild and a reclamation of the same artifact from interleaving.
    ``prefix`` keeps artifact kinds from sharing a lock file. Lives under
    :func:`analyzer_cache_root`, with the same cross-host locking requirement
    as :func:`analyzer_cache_lock`.

    Parameters
    ----------
    prefix : str
        Artifact kind, e.g. ``"recording"``, ``"concat_recording"`` or
        ``"motion_corrected"``.
    slot_id
        The id of the artifact whose canonical file the caller will mutate.

    Returns
    -------
    filelock.FileLock
        An unacquired lock; acquiring blocks indefinitely and the lock is
        released when the holding process exits.
    """
    from filelock import FileLock

    root = analyzer_cache_root()
    root.mkdir(parents=True, exist_ok=True)
    return FileLock(str(root / f"{prefix}_{slot_id}.artifact.lock"))


_POPULATING_SORTING_IDS: ContextVar[tuple[str, ...]] = ContextVar(
    "analyzer_populating_sorting_ids", default=()
)


@contextmanager
def analyzer_population(sorting_id):
    """Keep a sort's publication lock until the framework commits or rolls back.

    Folders are published before the Sorting row commits, so an orphan sweep
    must wait. Context-local ownership tells the framework's transaction from
    a caller-owned one, whose commit cannot be observed.
    """
    sid = str(sorting_id)
    with analyzer_cache_lock(sid):
        token = _POPULATING_SORTING_IDS.set(
            (*_POPULATING_SORTING_IDS.get(), sid)
        )
        try:
            yield
        finally:
            _POPULATING_SORTING_IDS.reset(token)


def analyzer_population_active(sorting_id) -> bool:
    """Whether this execution context owns the sort's framework transaction."""
    return str(sorting_id) in _POPULATING_SORTING_IDS.get()


class AnalyzerPublicationMixin:
    """Keep Sorting's analyzer lock around the framework's full populate call."""

    def _populate1(self, key, jobs, *args, **kwargs):
        with analyzer_population(key["sorting_id"]):
            return super()._populate1(key, jobs, *args, **kwargs)


@contextmanager
def analyzer_publication_transaction(table, sorting_id):
    """Protect an insert's own commit, or join its protected populate scope."""
    if table.connection.in_transaction and not analyzer_population_active(
        sorting_id
    ):
        raise dj.errors.DataJointError(
            "Sorting.make_insert cannot publish inside a caller-owned "
            "database transaction. Use populate(), or call make_insert "
            "outside a transaction so its analyzer lock covers COMMIT."
        )
    with analyzer_cache_lock(sorting_id), table._safe_context():
        yield


def _publish_sibling(canonical_folder, kind: str) -> Path:
    """Return a hidden sibling of ``canonical_folder`` for staging.

    Same directory as the slot because SI stores the recording path relative
    to the analyzer folder. The UUID avoids collisions across hosts.
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

    The ownership lock spans compute and insert; cleanup checks it without
    blocking, so a slow transaction never looks abandoned. Process death
    releases it.
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

            # A logged exception would pin its traceback's frames and analyzers.
            logger.warning(
                "Analyzer staging cleanup failed for %s: %s",
                self.folder,
                str(exc),
            )

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()


def publish_analyzer_atomically(canonical_folder, build_into):
    """Build an analyzer into a private temp folder, then move it into the slot.

    The build goes to a hidden sibling (:func:`_publish_sibling`) and is
    installed with :func:`install_staged_folder` under
    ``analyzer_cache_lock``, so readers never see a partial folder. A build
    that writes nothing (zero units) leaves the slot untouched.

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
    install_staged_folder(temp_folder, canonical_folder)


def install_staged_folder(staged, target) -> None:
    """Move a staged folder into ``target``, replacing what is there.

    A directory rename cannot replace a non-empty folder, so an existing
    ``target`` is first moved aside to a hidden ``trash`` sibling
    (:func:`_publish_sibling`), restored if the install move fails, and
    removed only once the new folder is in place. If the restore also fails,
    the trash sibling is left on disk: it is the surviving copy.

    Parameters
    ----------
    staged : path-like
        The completed folder to install; it must exist.
    target : path-like
        The durable folder to install into.
    """
    target = Path(target)
    if not target.exists():
        os.replace(staged, target)
        return
    trash = _publish_sibling(target, "trash")
    os.replace(target, trash)
    try:
        os.replace(staged, target)
    except BaseException:
        os.replace(trash, target)
        raise
    shutil.rmtree(trash, ignore_errors=True)


def cleanup_analyzer_staging(
    sorting_id=None, *, dry_run=True, candidates=None
) -> list[str]:
    """Report/remove abandoned build and trash folders, skipping active owners.

    A folder is active if its sort lock or build lock is held (checked
    without blocking). Only ``_publish_sibling`` names are candidates.
    ``candidates`` limits deletion to paths already shown to the operator.
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
            r"\.(.+)\.(?:build|trash)-([0-9a-f]{32})\.analyzer", folder.name
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
                    if folder.exists() and (
                        dry_run or _remove_unless_in_use(folder)
                    ):
                        abandoned.append(str(folder))
                Path(owner.lock_file).unlink(missing_ok=True)
        except Timeout:
            continue
    return abandoned


def _remove_unless_in_use(folder: Path) -> bool:
    """Remove ``folder``; return False if a still-open file blocks it.

    On NFS a still-mapped file leaves a ``.nfs`` placeholder (``EBUSY`` /
    ``ENOTEMPTY``); that folder is left for later. Other errors raise.
    """
    from spyglass.utils import logger

    try:
        shutil.rmtree(folder)
    except OSError as exc:
        if exc.errno not in (errno.EBUSY, errno.ENOTEMPTY):
            raise
        logger.warning(
            "Analyzer folder %s is still in use (%s); left for a "
            "later cleanup.",
            folder,
            str(exc),
        )
        return False
    return True


def remove_analyzer_cache(sorting_id, *, missing_ok: bool = True) -> bool:
    """Remove ALL analyzer-cache folders for a ``sorting_id``.

    Removes every ``{sorting_id}__*.analyzer`` folder (display, metric and
    curation caches) and abandoned staging. Removal errors propagate.

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


def remove_previous_sort_analyzer_caches(sorting_id, *, keep) -> None:
    """Remove analyzer folders an earlier sort left under a reused ``sorting_id``.

    Call only once a new ``Sorting`` insert owns ``sorting_id``. A cascade
    delete skips ``Sorting.delete``, and a re-sort reuses the deterministic id,
    so stale folders with old spike trains can remain. Each is renamed to
    trash first in case another process still maps it.

    Parameters
    ----------
    sorting_id
        The sort whose new row was just inserted.
    keep : path-like
        The canonical folder the caller is about to replace atomically.
    """
    root = analyzer_cache_root()
    if not root.exists():
        return
    keep = Path(keep)
    with analyzer_cache_lock(sorting_id):
        for folder in list(
            root.glob(f"{sorting_id}__*{ANALYZER_FOLDER_SUFFIX}")
        ):
            if folder != keep:
                trash = _publish_sibling(folder, "trash")
                os.replace(folder, trash)
                _remove_unless_in_use(trash)


def collect_analyzer_cache_references(sorting_table) -> dict:
    """Collect live analyzer references for the supplied sorting relation.

    References: each sort's display analyzer, every live committed curation's
    display cache, and PC-requesting metric analyzers. ``sorting_table``'s
    restriction scopes every query. ``units_bearing`` reports only raw display
    analyzers; curation caches are built lazily, so their absence is normal.
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

    # One join, so a selection and its curation are read consistently.
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
        "sorting_id",
        "curation_id",
        "curation_uuid",
        "merges_applied",
        as_dict=True,
    )
    unit_bearing_curations = {
        (str(row["sorting_id"]), int(row["curation_id"]))
        for row in (curations & CurationV2.Unit.proj()).fetch(
            "sorting_id", "curation_id", as_dict=True
        )
    }
    # Unit-bearing committed curations; the PC loop below also reads it.
    live_committed: set[tuple[str, int]] = set()
    for row in curation_rows:
        sorting_id = str(row["sorting_id"])
        curation_id = int(row["curation_id"])
        key = {"sorting_id": row["sorting_id"], "curation_id": curation_id}
        if (sorting_id, curation_id) not in unit_bearing_curations:
            continue
        if not CurationV2.is_committed_curation(
            key, merges_applied=row["merges_applied"]
        ):
            continue
        live_committed.add((sorting_id, curation_id))
        # A raw-namespace curation shares the sort analyzer, but its
        # derivatives are keyed by this would-be base path.
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
        if (str(row["sorting_id"]), key["curation_id"]) not in live_committed:
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

    - DB-side: a units-bearing row whose folder is missing (reported only).
    - Reclaimed: the same, with a recompute ``deleted=1`` record.
    - Disk-side: a folder no row references.

    Input order is preserved.

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
    # A derivative is referenced exactly when its base cache is.
    disk_side = [
        path
        for path in disk_dir_paths
        if path not in referenced
        and derivative_base_path(path) not in referenced
    ]
    return {"db_side": db_side, "disk_side": disk_side, "reclaimed": reclaimed}


def find_orphaned_analyzer_folders(
    table_cls, *, sorting_id=None, dry_run: bool = True
) -> dict:
    """Audit analyzer-folder disk leaks; never delete database rows.

    The body of ``Sorting.find_orphaned_analyzer_folders``. ``table_cls`` is
    the ``Sorting`` class.
    """
    from spyglass.utils import logger

    if not dry_run and table_cls.connection.in_transaction:
        raise dj.errors.DataJointError(
            "Analyzer orphan cleanup must run outside a database transaction "
            "so its locked ownership recheck sees newly committed Sorting rows."
        )

    if sorting_id is not None:
        sorting_id = uuid.UUID(str(sorting_id))
    references = collect_analyzer_cache_references(
        table_cls & ({} if sorting_id is None else {"sorting_id": sorting_id})
    )
    analyzer_root = analyzer_cache_root()
    # Only canonical folders, even under a misconfigured root.
    disk_dir_paths = (
        [
            str(c)
            for c in sorted(
                analyzer_root.iterdir()
                if sorting_id is None
                else analyzer_root.glob(f"{sorting_id}*")
            )
            if c.is_dir()
            and not c.name.startswith(".")
            and is_canonical_analyzer_folder_name(c.name)
        ]
        if analyzer_root.exists()
        else []
    )
    classification = classify_orphaned_analyzer_folders(
        references["units_bearing"],
        references["referenced_paths"],
        disk_dir_paths,
        reclaimed_paths=references["reclaimed_paths"],
    )
    db_side = classification["db_side"]
    disk_side = classification["disk_side"]
    reclaimed = classification["reclaimed"]
    staging = cleanup_analyzer_staging(sorting_id)

    logger.info(
        "Sorting.find_orphaned_analyzer_folders: "
        f"{len(db_side)} DB-side orphan(s) (row present, folder missing), "
        f"{len(disk_side)} disk-side orphan(s) (folder present, no row), "
        f"{len(reclaimed)} reclaimed folder(s) (missing with deleted=1), "
        f"{len(staging)} abandoned staging folder(s)."
    )
    for row in db_side:
        logger.info(
            "  DB-side orphan: sorting_id=%s computed_analyzer_path=%s",
            row["sorting_id"],
            row["computed_analyzer_path"],
        )
    for row in reclaimed:
        logger.info(
            "  reclaimed analyzer: sorting_id=%s computed_analyzer_path=%s",
            row["sorting_id"],
            row["computed_analyzer_path"],
        )
    for folder in disk_side:
        logger.info("  disk-side orphan: %s", folder)
    for folder in staging:
        logger.info("  abandoned staging: %s", folder)

    if dry_run or not (disk_side or staging):
        return {
            "db_side": db_side,
            "disk_side": disk_side,
            "reclaimed": reclaimed,
            "staging": staging,
        }

    msg = (
        f"Delete {len(disk_side)} orphaned analyzer folder(s) and "
        f"{len(staging)} abandoned staging folder(s)? This cannot "
        "be undone [yes/no]: "
    )
    if dj.utils.user_choice(msg).lower() in ("y", "yes"):
        # Recheck ownership after confirmation; never reclaim a live build.
        deleted_staging = cleanup_analyzer_staging(
            sorting_id, dry_run=False, candidates=staging
        )
        deleted_folders = []
        for folder in disk_side:
            identity = analyzer_cache_folder_identity(Path(folder).name)
            if identity is None:  # pragma: no cover - classified above
                continue
            with analyzer_cache_lock(identity.sorting_id):
                # Recheck under the lock, which Sorting holds through COMMIT,
                # in case an owner committed since the scan.
                live = collect_analyzer_cache_references(
                    table_cls & {"sorting_id": identity.sorting_id}
                )["referenced_paths"]
                if folder in live or derivative_base_path(folder) in live:
                    continue
                if Path(folder).exists() and _remove_unless_in_use(
                    Path(folder)
                ):
                    deleted_folders.append(folder)
        logger.info(
            "Sorting.find_orphaned_analyzer_folders: deleted "
            f"{len(deleted_folders)} disk-side orphan folder(s) and "
            f"{len(deleted_staging)} abandoned staging folder(s)."
        )
    else:
        logger.info(
            "Sorting.find_orphaned_analyzer_folders: aborted; nothing "
            "deleted."
        )
    return {
        "db_side": db_side,
        "disk_side": disk_side,
        "reclaimed": reclaimed,
        "staging": staging,
    }

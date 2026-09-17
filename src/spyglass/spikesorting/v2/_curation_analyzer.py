"""Curation-scoped SortingAnalyzer resolution and immutable cache ownership.

Raw-namespace curations reuse the existing sort analyzer. Committed curations
whose unit set differs from the raw sort build a per-generation analyzer over
their exact stored spike trains. The cache key includes ``curation_uuid`` plus
the recipe content and SpikeInterface version, and a manifest validates the
scientific inputs and every published extension before a folder is reused.

Published merged-curation analyzers are immutable by ownership. Supported
callers use the private resolver for reads. Extra extensions with specific
parameters are computed ONCE into a disk-backed DERIVATIVE cache keyed by the
base identity plus the extension request (``derived_extension_request_hash``)
and reused on every later request with the same parameters -- never a
whole-analyzer memory copy. Expert callers that need a mutable SI object get a
context-managed ``binary_folder`` working copy in a temp dir
(``open_curation_analyzer``), so no path can mutate a published cache.
Raw-namespace curations instead share the sort analyzer and persist missing
extensions through ``Sorting.add_extensions`` under its cache lock.

Every folder is a ``binary_folder`` analyzer loaded through
``_analyzer_cache.load_analyzer_folder`` (memmapped waveforms).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

from spyglass.spikesorting.v2._analyzer_cache import (
    analyzer_cache_lock,
    analyzer_folder_storage_fingerprint,
    copy_analyzer_folder,
    curation_analyzer_path,
    load_analyzer_extensions,
    load_analyzer_folder,
    publish_analyzer_atomically,
    waveform_recipe_hash,
)
from spyglass.spikesorting.v2._sorting_analyzer import (
    BASE_ANALYZER_EXTENSIONS,
    STANDARD_DISPLAY_ANALYZER_EXTENSIONS,
)
from spyglass.spikesorting.v2.exceptions import ZeroUnitAnalyzerError

CURATION_ANALYZER_MANIFEST = "spyglass_curation_analyzer_manifest.json"
MERGE_POLICY_VERSION = "absolute_time_cross_unit_dedup_0_4ms_v1"


@dataclass(frozen=True)
class CurationAnalyzerManifest:
    """Validated identity and completeness contract for a cached analyzer."""

    sorting_id: str
    curation_uuid: str
    curation_id: int
    role: str
    curated_unit_ids: tuple[int, ...]
    contributor_map: dict[str, list[int]]
    merged_spike_content_hash: str
    merge_policy_version: str
    source_artifact_hashes: dict[str, str]
    waveform_recipe_hash: str
    spikeinterface_version: str
    extension_request: dict[str, dict]
    extension_inventory: dict[str, str]
    storage_fingerprint: str

    def as_json_dict(self) -> dict:
        """Return the canonical JSON-native representation."""
        row = asdict(self)
        row["curated_unit_ids"] = list(self.curated_unit_ids)
        return row


def _canonical_json(value) -> str:
    """Serialize a JSON-like value deterministically, including numpy blobs."""
    from spyglass.spikesorting.v2._lookup_validation import _jsonable_blob

    return json.dumps(
        _jsonable_blob(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _content_hash(value) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _folder_storage_fingerprint(folder: Path) -> str:
    """Fingerprint stored files by relative path, size, and mtime.

    Cache validation deliberately avoids decoding multi-GB extension arrays.
    The manifest itself is excluded because it records this fingerprint and is
    written only after all analyzer files have been published to the staging
    directory.
    """
    return analyzer_folder_storage_fingerprint(
        folder, exclude_names=(CURATION_ANALYZER_MANIFEST,)
    )


def _normalize_curation_key(curation_ref) -> dict:
    """Return a concrete curation restriction from a key or facade object."""
    if hasattr(curation_ref, "as_key"):
        curation_ref = curation_ref.as_key()
    if not isinstance(curation_ref, Mapping):
        raise TypeError(
            "curation analyzer resolution requires a curation key mapping or "
            f"CurationRef-like object; got {type(curation_ref).__name__}."
        )
    restriction = {
        field: curation_ref[field]
        for field in ("sorting_id", "curation_id", "curation_uuid")
        if field in curation_ref
    }
    if not restriction:
        raise ValueError(
            "curation analyzer resolution requires sorting_id/curation_id or "
            "curation_uuid."
        )
    return restriction


def _resolve_curation_row(curation_ref) -> dict:
    from spyglass.spikesorting.v2.curation import CurationV2

    restriction = _normalize_curation_key(curation_ref)
    relation = CurationV2 & restriction
    if len(relation) != 1:
        raise ValueError(
            "curation analyzer resolution requires exactly one existing "
            f"CurationV2 row; restriction {restriction!r} matched "
            f"{len(relation)}."
        )
    return relation.fetch1()


def _classify_curation_row(row: Mapping) -> str:
    """Classify an already-resolved CurationV2 row: raw / merged / preview /
    zero-unit. The caller resolved the row; this does not fetch it again."""
    from spyglass.spikesorting.v2.curation import CurationV2

    key = {
        "sorting_id": row["sorting_id"],
        "curation_id": int(row["curation_id"]),
    }
    if CurationV2.has_unapplied_proposed_merges(
        key, merges_applied=row["merges_applied"]
    ):
        return "preview"
    if not (CurationV2.Unit & key):
        return "zero-unit"
    if CurationV2.matches_raw_namespace(key):
        return "raw"
    return "merged"


def _resolve_recipe(waveform_recipe: str, role: str) -> dict:
    from spyglass.spikesorting.v2.sorting import AnalyzerWaveformParameters

    if role not in {"display", "metric"}:
        raise ValueError(
            "curation analyzer role must be 'display' or 'metric'; "
            f"got {role!r}."
        )
    relation = AnalyzerWaveformParameters & {
        "waveform_params_name": waveform_recipe
    }
    if len(relation) != 1:
        raise ValueError(
            "curation analyzer waveform_recipe must name exactly one "
            f"AnalyzerWaveformParameters row; got {waveform_recipe!r}."
        )
    row = relation.fetch1()
    params = dict(row["params"])
    expected_purpose = role
    if params.get("purpose") != expected_purpose or bool(
        params.get("whiten")
    ) != (role == "metric"):
        raise ValueError(
            f"waveform recipe {waveform_recipe!r} is not a {role} recipe: "
            f"expected purpose={expected_purpose!r}, whiten={role == 'metric'}; "
            f"got purpose={params.get('purpose')!r}, "
            f"whiten={bool(params.get('whiten'))}."
        )
    return row


def curation_analyzer_cache_path(
    curation_ref, waveform_recipe: str, role: str = "display"
) -> Path:
    """Resolve the canonical cache path for one curation-generation recipe."""
    row = _resolve_curation_row(curation_ref)
    recipe_row = _resolve_recipe(waveform_recipe, role)
    return curation_analyzer_path(
        row["sorting_id"],
        row["curation_uuid"],
        role,
        waveform_recipe_hash(recipe_row),
        _spikeinterface_version(),
    )


def _source_artifact_hashes(sorting_id) -> dict[str, str]:
    """Resolve the current recording artifact hash behind one sort."""
    from spyglass.spikesorting.v2.recording import Recording
    from spyglass.spikesorting.v2.session_group import ConcatenatedRecording
    from spyglass.spikesorting.v2.sorting import SortingSelection

    source = SortingSelection.resolve_source({"sorting_id": sorting_id})
    if source.kind == "recording":
        content_hash = (Recording & source.key).fetch1("content_hash")
    else:
        content_hash = (ConcatenatedRecording & source.key).fetch1(
            "content_hash"
        )
    return {source.kind: str(content_hash)}


def _raw_contributor_map(key) -> dict[str, list[int]]:
    """Return each curated unit's ordered original-sort contributors."""
    from spyglass.spikesorting.v2.curation import CurationV2

    rows = (CurationV2.MergeGroup & key).fetch(
        "unit_id",
        "contributor_unit_id",
        as_dict=True,
        order_by=("unit_id", "contributor_unit_id"),
    )
    groups: dict[str, list[int]] = {}
    for row in rows:
        groups.setdefault(str(int(row["unit_id"])), []).append(
            int(row["contributor_unit_id"])
        )
    return groups


def hash_sorting_spike_content(sorting) -> str:
    """Hash exact unit ids and spike-sample frames for every segment."""
    import numpy as np

    digest = hashlib.sha256()
    # Sampling frequency is recording metadata, not spike content. SI may
    # replace a sorting's frequency with the recording's nearly-equal float
    # while creating an analyzer (for example 100.0 vs 99.99999999999999).
    # Including it would therefore reject a scientifically identical cache
    # after its first zarr round-trip. Source recording identity is validated
    # independently by ``source_artifact_hashes``.
    digest.update(b"spyglass-curation-spikes-v1\0")
    digest.update(
        int(sorting.get_num_segments()).to_bytes(8, "little", signed=False)
    )
    unit_ids = sorted(int(unit_id) for unit_id in sorting.get_unit_ids())
    digest.update(len(unit_ids).to_bytes(8, "little", signed=False))
    for unit_id in unit_ids:
        digest.update(int(unit_id).to_bytes(8, "little", signed=True))
        for segment_index in range(sorting.get_num_segments()):
            frames = np.asarray(
                sorting.get_unit_spike_train(
                    unit_id=unit_id, segment_index=segment_index
                ),
                dtype="<i8",
            )
            digest.update(int(segment_index).to_bytes(4, "little"))
            digest.update(int(frames.size).to_bytes(8, "little"))
            digest.update(frames.tobytes(order="C"))
    return digest.hexdigest()


def _manifest_prefix(row, sorting, recipe_row, role: str) -> dict:
    key = {
        "sorting_id": row["sorting_id"],
        "curation_id": int(row["curation_id"]),
    }
    return {
        "sorting_id": str(row["sorting_id"]),
        "curation_uuid": str(row["curation_uuid"]),
        "curation_id": int(row["curation_id"]),
        "role": role,
        "curated_unit_ids": tuple(
            sorted(int(unit_id) for unit_id in sorting.get_unit_ids())
        ),
        "contributor_map": _raw_contributor_map(key),
        "merged_spike_content_hash": hash_sorting_spike_content(sorting),
        "merge_policy_version": MERGE_POLICY_VERSION,
        "source_artifact_hashes": _source_artifact_hashes(row["sorting_id"]),
        "waveform_recipe_hash": waveform_recipe_hash(recipe_row),
        "spikeinterface_version": _spikeinterface_version(),
        # The base cache carries no extra extensions; a derivative records the
        # exact request (names + normalized params) it was built for.
        "extension_request": {},
    }


def normalize_extension_request(extra_extensions) -> dict[str, dict]:
    """Return ``{name: params}`` with JSON-native params, sorted by name."""
    from spyglass.spikesorting.v2._lookup_validation import _jsonable_blob

    request = {}
    for name in sorted(extra_extensions or {}):
        request[str(name)] = _jsonable_blob(dict(extra_extensions[name] or {}))
    return request


def derived_extension_request_hash(extension_request: Mapping) -> str:
    """16-hex identity of an extension request (names + parameters)."""
    return _content_hash(normalize_extension_request(extension_request))[:16]


def _spikeinterface_version() -> str:
    import spikeinterface as si

    return str(si.__version__)


def _expected_extensions(
    role: str, extra: tuple[str, ...] = ()
) -> tuple[str, ...]:
    if role == "display":
        base = (
            *BASE_ANALYZER_EXTENSIONS,
            *STANDARD_DISPLAY_ANALYZER_EXTENSIONS,
        )
    else:
        base = BASE_ANALYZER_EXTENSIONS
    return base + tuple(name for name in extra if name not in base)


def _extension_inventory(
    analyzer, role: str, extra: tuple[str, ...] = (), *, exact: bool = True
) -> dict[str, str]:
    """Validate the extension set and fingerprint its small parameters.

    ``exact=True`` (a base cache) requires the saved set to equal the role's
    expected set. ``exact=False`` (a derivative) requires the base extensions
    plus ``extra`` (the request) to be PRESENT and inventories every saved
    extension: a derivative copies its source as-is -- the shared raw-sort
    analyzer carries only the base set plus whatever was persisted on
    demand, a merged cache carries the full display set -- so the request is
    the only addition it guarantees.
    """
    from spyglass.spikesorting.v2._analyzer_cache import (
        analyzer_extension_params,
    )

    saved = set(analyzer.get_saved_extension_names())
    if exact:
        expected = _expected_extensions(role, extra)
        if saved != set(expected):
            raise ValueError(
                "curation analyzer extension set is incomplete or unexpected: "
                f"expected {sorted(expected)}, found {sorted(saved)}."
            )
    else:
        required = set(BASE_ANALYZER_EXTENSIONS) | set(extra)
        if not required <= saved:
            raise ValueError(
                "curation analyzer derivative is missing extension(s): "
                f"{sorted(required - saved)}; found {sorted(saved)}."
            )
    inventory: dict[str, str] = {}
    for name in expected if exact else sorted(saved):
        inventory[name] = _content_hash(
            analyzer_extension_params(analyzer, name)
        )
    return inventory


def extension_params_match(analyzer, name: str, requested: Mapping) -> bool:
    """Whether ``analyzer`` carries ``name`` with every requested parameter.

    ``requested`` is the caller's (possibly partial) parameter dict; a key it
    does not name is unconstrained (SI's default is accepted), so an empty
    request matches any present extension. Values compare JSON-natively
    (``1`` == ``1.0``). A present extension whose stored ``params`` differ on
    a requested key does NOT satisfy the request and must be recomputed into
    a derivative rather than silently reused.
    """
    from spyglass.spikesorting.v2._lookup_validation import _jsonable_blob

    if not analyzer.has_extension(name):
        return False
    if not requested:
        return True
    from spyglass.spikesorting.v2._analyzer_cache import (
        analyzer_extension_params,
    )

    stored = _jsonable_blob(dict(analyzer_extension_params(analyzer, name)))
    wanted = _jsonable_blob(dict(requested))
    return all(
        key in stored and stored[key] == value for key, value in wanted.items()
    )


def _read_manifest(folder: Path) -> dict:
    return json.loads((folder / CURATION_ANALYZER_MANIFEST).read_text())


def _write_manifest(folder: Path, manifest: CurationAnalyzerManifest) -> None:
    (folder / CURATION_ANALYZER_MANIFEST).write_text(
        json.dumps(
            manifest.as_json_dict(),
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


def _load_valid_cached_analyzer(folder: Path, expected_prefix: dict, role: str):
    """Return a validated cached analyzer, or ``None`` on any corruption."""
    from spyglass.utils import logger

    extra = tuple(expected_prefix.get("extension_request", {}))
    is_derivative = bool(expected_prefix.get("extension_request"))
    try:
        stored = _read_manifest(folder)
        comparable = dict(stored)
        stored_inventory = comparable.pop("extension_inventory")
        stored_storage_fingerprint = comparable.pop("storage_fingerprint")
        expected = dict(expected_prefix)
        expected["curated_unit_ids"] = list(expected["curated_unit_ids"])
        if comparable != expected:
            mismatched = sorted(
                field
                for field in set(comparable) | set(expected)
                if comparable.get(field) != expected.get(field)
            )
            logger.warning(
                "Curation analyzer manifest identity mismatch at %s: %s.",
                folder,
                mismatched,
            )
            return None
        current_storage_fingerprint = _folder_storage_fingerprint(folder)
        if current_storage_fingerprint != stored_storage_fingerprint:
            logger.warning(
                "Curation analyzer storage fingerprint mismatch at %s: "
                "stored=%s current=%s.",
                folder,
                stored_storage_fingerprint,
                current_storage_fingerprint,
            )
            return None
        analyzer = load_analyzer_folder(folder)
        current_inventory = _extension_inventory(
            analyzer, role, extra, exact=not is_derivative
        )
        if current_inventory != stored_inventory:
            logger.warning(
                "Curation analyzer extension inventory mismatch at %s: "
                "stored=%s current=%s.",
                folder,
                stored_inventory,
                current_inventory,
            )
            return None
        current_spike_hash = hash_sorting_spike_content(analyzer.sorting)
        if current_spike_hash != expected_prefix["merged_spike_content_hash"]:
            logger.warning(
                "Curation analyzer spike-content mismatch at %s: expected=%s "
                "current=%s.",
                folder,
                expected_prefix["merged_spike_content_hash"],
                current_spike_hash,
            )
            return None
        return analyzer
    except Exception as exc:  # noqa: BLE001 - cache is regeneratable
        logger.warning(
            "Curation analyzer validation failed at %s: %r.", folder, exc
        )
        return None


def _resolve_published_analyzer(
    folder: Path,
    sorting_id,
    expected_prefix: dict,
    role: str,
    build_into,
):
    """Build or reuse one validated cache slot under the per-sort lock."""
    from spyglass.utils import logger

    with analyzer_cache_lock(sorting_id):
        if folder.exists():
            analyzer = _load_valid_cached_analyzer(
                folder, expected_prefix, role
            )
            if analyzer is not None:
                return analyzer
            logger.warning(
                "Curation analyzer cache %s is stale, incomplete, or corrupt; "
                "rebuilding it from the committed curation.",
                folder,
            )

        publish_analyzer_atomically(folder, build_into)
        analyzer = _load_valid_cached_analyzer(folder, expected_prefix, role)
        if analyzer is None:  # pragma: no cover - publish validation defense
            raise RuntimeError(
                "curation analyzer cache failed validation immediately after "
                f"publish: {folder}."
            )
        return analyzer


def build_merged_analyzer(
    curation_ref,
    waveform_recipe: str,
    role: str = "display",
    *,
    analyzer_folder,
):
    """Build one merged-curation analyzer through the shared low-level builder."""
    from spyglass.spikesorting.v2._sorting_analyzer import (
        build_analyzer,
        ensure_extensions,
        reconstruct_recording_and_sorting,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        Sorting,
        SortingSelection,
    )
    from spyglass.spikesorting.v2.utils import _resolved_job_kwargs

    row = _resolve_curation_row(curation_ref)
    key = {
        "sorting_id": row["sorting_id"],
        "curation_id": int(row["curation_id"]),
    }
    CurationV2.assert_committed_curation(
        key,
        context="curation analyzer",
        merges_applied=row["merges_applied"],
    )
    if not (CurationV2.Unit & key):
        raise ZeroUnitAnalyzerError(
            "curation analyzer: curation "
            f"(sorting_id={row['sorting_id']}, "
            f"curation_id={row['curation_id']}) has zero units; no "
            "SortingAnalyzer exists."
        )
    recipe_row = _resolve_recipe(waveform_recipe, role)
    recording, _raw_sorting = reconstruct_recording_and_sorting(
        Sorting(), {"sorting_id": row["sorting_id"]}
    )
    curated_sorting = CurationV2.get_merged_sorting(key)
    sorter_row = (
        SorterParameters
        & (
            (SortingSelection & {"sorting_id": row["sorting_id"]}).proj(
                "sorter", "sorter_params_name"
            )
        )
    ).fetch1()
    job_kwargs = _resolved_job_kwargs(sorter_row["job_kwargs"])
    build_analyzer(
        curated_sorting,
        recording,
        {"sorting_id": row["sorting_id"]},
        sorter_row=sorter_row,
        job_kwargs=job_kwargs,
        analyzer_folder=Path(analyzer_folder),
        waveform_params=dict(recipe_row["params"]),
    )
    analyzer = load_analyzer_folder(analyzer_folder)
    if not analyzer.has_recording():
        analyzer.set_temporary_recording(recording)
    if role == "display":
        ensure_extensions(
            analyzer,
            STANDARD_DISPLAY_ANALYZER_EXTENSIONS,
            job_kwargs=job_kwargs,
        )
    return analyzer


def _resolve_curation_analyzer(
    curation_ref,
    waveform_recipe: str,
    role: str = "display",
    *,
    extra_extensions: Mapping[str, dict] | None = None,
):
    """Resolve the cache-backed analyzer for controlled internal reads.

    Returns the PUBLISHED analyzer (raw namespace: the shared sort analyzer;
    merged namespace: the immutable per-generation cache, or -- when
    ``extra_extensions`` names extensions the base does not carry -- the
    disk-backed derivative built for exactly that request). Callers must treat
    the result as read-only; use :func:`open_curation_analyzer` for a mutable
    working copy.
    """
    import spikeinterface as si

    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting

    row = _resolve_curation_row(curation_ref)
    key = {
        "sorting_id": row["sorting_id"],
        "curation_id": int(row["curation_id"]),
    }
    namespace = _classify_curation_row(row)
    if namespace == "preview":
        CurationV2.assert_committed_curation(
            key,
            context="curation analyzer",
            merges_applied=row["merges_applied"],
        )
    if namespace == "zero-unit":
        raise ZeroUnitAnalyzerError(
            "curation analyzer: curation "
            f"(sorting_id={row['sorting_id']}, "
            f"curation_id={row['curation_id']}) has zero units; no "
            "SortingAnalyzer exists."
        )
    recipe_row = _resolve_recipe(waveform_recipe, role)
    # The COMPLETE normalized request is the derivative's identity and its
    # required result; the subset that actually needs computing is a separate
    # local (``needs_compute``) so a later, larger request never reuses a
    # smaller derivative.
    request = normalize_extension_request(extra_extensions)
    if namespace == "raw":
        # Raw curations share the canonical sort analyzer. An ABSENT requested
        # extension is persisted there under the per-sort cache lock so later
        # views reuse it. A PRESENT extension whose stored parameters differ
        # from the request is never overwritten in the shared analyzer
        # (recomputing would rewrite what every other reader sees); it is
        # served from a derivative keyed by the request, exactly as for a
        # merged curation.
        base = Sorting().get_analyzer(
            {"sorting_id": row["sorting_id"]},
            waveform_params_name=waveform_recipe,
            load_extensions=False,
        )
        absent = [name for name in request if not base.has_extension(name)]
        if absent:
            Sorting().add_extensions(
                {"sorting_id": row["sorting_id"]},
                absent,
                waveform_params_name=waveform_recipe,
                extension_params={name: request[name] for name in absent},
            )
            base = Sorting().get_analyzer(
                {"sorting_id": row["sorting_id"]},
                waveform_params_name=waveform_recipe,
                load_extensions=False,
            )
        # The manifest (spike-content hash + provenance) is only needed if a
        # derivative must be built; an ordinary read must not scan spike
        # trains. Resolved lazily below.
        expected_prefix = None
    else:
        curated_sorting = CurationV2.get_merged_sorting(key)
        expected_prefix = _manifest_prefix(
            row, curated_sorting, recipe_row, role
        )
        base_folder = curation_analyzer_path(
            row["sorting_id"],
            row["curation_uuid"],
            role,
            expected_prefix["waveform_recipe_hash"],
            si.__version__,
        )

        def _build_base(staging_folder):
            analyzer = build_merged_analyzer(
                key,
                waveform_recipe,
                role,
                analyzer_folder=staging_folder,
            )
            inventory = _extension_inventory(analyzer, role)
            manifest = CurationAnalyzerManifest(
                **expected_prefix,
                extension_inventory=inventory,
                storage_fingerprint=_folder_storage_fingerprint(
                    Path(staging_folder)
                ),
            )
            _write_manifest(Path(staging_folder), manifest)

        base = _resolve_published_analyzer(
            base_folder,
            row["sorting_id"],
            expected_prefix,
            role,
            _build_base,
        )

    needs_compute = {
        name: params
        for name, params in request.items()
        if not extension_params_match(base, name, params)
    }
    if not needs_compute:
        return base
    if expected_prefix is None:
        expected_prefix = _manifest_prefix(row, base.sorting, recipe_row, role)

    # Disk-backed derivative: the base (shared raw analyzer or immutable
    # per-generation cache) stays untouched; the derivative is keyed by the
    # COMPLETE request, carries every requested extension at the requested
    # parameters, and is reused thereafter. Built by copying the base on disk
    # (memmapped waveforms stream through np.save) and computing only what the
    # copy lacks -- plus whatever SpikeInterface invalidates downstream of it.
    derived_prefix = dict(expected_prefix)
    derived_prefix["extension_request"] = request
    derived_folder = curation_analyzer_path(
        row["sorting_id"],
        row["curation_uuid"],
        role,
        expected_prefix["waveform_recipe_hash"],
        si.__version__,
        extension_request_hash=derived_extension_request_hash(request),
    )

    def _build_derivative(staging_folder):
        from spyglass.spikesorting.v2._sorting_analyzer import (
            reconstruct_recording_and_sorting,
        )
        from spyglass.spikesorting.v2.sorting import (
            SorterParameters,
            SortingSelection,
        )
        from spyglass.spikesorting.v2.utils import _resolved_job_kwargs

        derivative = copy_analyzer_folder(base, Path(staging_folder))
        if not derivative.has_recording():
            recording, _sorting = reconstruct_recording_and_sorting(
                Sorting(), {"sorting_id": row["sorting_id"]}
            )
            derivative.set_temporary_recording(recording)
        sorter_job_kwargs = (
            SorterParameters
            & (SortingSelection & {"sorting_id": row["sorting_id"]})
        ).fetch1("job_kwargs")
        _compute_request_on_copy(
            derivative,
            request,
            needs_compute,
            job_kwargs=_resolved_job_kwargs(sorter_job_kwargs),
        )
        inventory = _extension_inventory(
            derivative, role, tuple(request), exact=False
        )
        manifest = CurationAnalyzerManifest(
            **derived_prefix,
            extension_inventory=inventory,
            storage_fingerprint=_folder_storage_fingerprint(
                Path(staging_folder)
            ),
        )
        _write_manifest(Path(staging_folder), manifest)

    return _resolve_published_analyzer(
        derived_folder,
        row["sorting_id"],
        derived_prefix,
        role,
        _build_derivative,
    )


def _compute_request_on_copy(analyzer, request, needs_compute, *, job_kwargs):
    """Compute ``needs_compute`` on a working copy, restoring what SI removes.

    SpikeInterface deletes every (transitive) dependent of a recomputed
    extension (``compute_one_extension`` -> ``_get_children_dependencies``).
    The required result is the base extension set plus every requested
    extension, so each required dependent of the computed set is recomputed
    too -- at the caller's parameters when requested, otherwise at its stored
    parameters (SI's ``templates`` re-derives its window from the waveforms
    extension regardless of stored values). ``analyzer.compute`` orders the
    plan by dependency. Only descendants of the recomputed set are touched, so
    a correlogram-only change never re-extracts waveforms. The result is
    checked against the whole request before returning.
    """
    from spikeinterface.core.sortinganalyzer import _get_children_dependencies

    required = set(BASE_ANALYZER_EXTENSIONS) | set(request)
    plan = {name: dict(params) for name, params in needs_compute.items()}
    for name in list(plan):
        for child in _get_children_dependencies(name):
            if child in required and child not in plan:
                plan[child] = dict(
                    request.get(child)
                    or (
                        analyzer.get_extension(child).params
                        if analyzer.has_extension(child)
                        else {}
                    )
                    or {}
                )
    compute_kwargs = {
        k: v for k, v in (job_kwargs or {}).items() if k != "random_seed"
    }
    for name in plan:
        if analyzer.has_extension(name):
            analyzer.delete_extension(name)
    analyzer.compute(list(plan), extension_params=plan, **compute_kwargs)
    missing = sorted(
        name for name in required if not analyzer.has_extension(name)
    )
    unmet = sorted(
        name
        for name, params in request.items()
        if not extension_params_match(analyzer, name, params)
    )
    if missing or unmet:
        raise ValueError(
            "curation analyzer derivative does not satisfy its request: "
            f"missing {missing}, parameter mismatch {unmet} (request "
            f"{request}, computed {sorted(plan)})."
        )


@contextmanager
def open_curation_analyzer(
    curation_ref,
    waveform_recipe: str,
    role: str = "display",
    *,
    extra_extensions: Mapping[str, dict] | None = None,
):
    """Yield a mutable, disk-backed WORKING COPY of a curation's analyzer.

    For expert SpikeInterface access (widgets that compute, exporters, ad-hoc
    extensions). The copy is a ``binary_folder`` analyzer in a private temp
    directory under Spyglass's temp dir; mutating it cannot alter the published
    cache, and the directory is removed when the context exits. Waveforms are
    copied on disk (streamed from the memmapped source), not held in memory.
    Raw-namespace and merged-namespace curations resolve through the same
    published caches (see :func:`_resolve_curation_analyzer`), so the copy
    carries the exact curated unit ids and spike trains.
    """
    import shutil
    import tempfile

    from spyglass.settings import temp_dir as spyglass_temp_dir
    from spyglass.spikesorting.v2._analyzer_cache import (
        ANALYZER_FOLDER_SUFFIX,
    )

    published = _resolve_curation_analyzer(
        curation_ref, waveform_recipe, role, extra_extensions=extra_extensions
    )
    tmp = tempfile.mkdtemp(
        prefix="v2_curation_analyzer_", dir=spyglass_temp_dir
    )
    try:
        working = copy_analyzer_folder(
            published, Path(tmp) / f"working{ANALYZER_FOLDER_SUFFIX}"
        )
        yield load_analyzer_extensions(working)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@contextmanager
def curation_analyzer_with_extensions(
    curation_ref,
    waveform_recipe: str,
    role: str = "display",
    *,
    extra_extensions: Mapping[str, dict] | None = None,
):
    """Yield the published (read-only) analyzer carrying ``extra_extensions``.

    Raw namespaces persist missing extensions to their shared sort analyzer;
    merged namespaces resolve (building once if needed) the disk-backed
    derivative keyed by the exact extension request. Nothing is copied into
    memory. The yielded analyzer must not be mutated; use
    :func:`open_curation_analyzer` for a working copy.
    """
    yield _resolve_curation_analyzer(
        curation_ref, waveform_recipe, role, extra_extensions=extra_extensions
    )

"""Curation-scoped SortingAnalyzer resolution and immutable cache ownership.

- Raw-namespace curations share the sort analyzer.
- Merged curations get a per-generation ``binary_folder`` cache keyed by
  ``curation_uuid``, recipe, and SpikeInterface version, validated by a
  manifest before reuse.
- Extra extensions go to a disk-backed derivative keyed by the request.
- ``open_curation_analyzer`` yields a mutable temp working copy.
"""

from __future__ import annotations

import functools
import gc
import hashlib
import json
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

from spyglass.spikesorting.v2._storage.analyzer_cache import (
    analyzer_cache_lock,
    analyzer_folder_storage_fingerprint,
    copy_analyzer_folder,
    curation_analyzer_path,
    load_analyzer_extensions,
    load_analyzer_folder,
    publish_analyzer_atomically,
    waveform_recipe_hash,
)
from spyglass.spikesorting.v2._core.selection_identity import sha256_json
from spyglass.spikesorting.v2._core.signal_math import resolve_peak_sign
from spyglass.spikesorting.v2._sorting.analyzer import (
    BASE_ANALYZER_EXTENSIONS,
    POLARITY_EXTENSIONS,
    STANDARD_DISPLAY_ANALYZER_EXTENSIONS,
    _AnalyzerSparsityInputsError,
    analyzer_sparsity_validator,
    polarity_extension_params,
    resolve_display_waveform_params_name,
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


def _content_hash(value) -> str:
    """Hash a JSON-like value deterministically, including numpy blobs."""
    from spyglass.spikesorting.v2._core.lookup_validation import _jsonable_blob

    return sha256_json(_jsonable_blob(value), allow_nan=False)


def _folder_storage_fingerprint(folder: Path) -> str:
    """Fingerprint stored files by relative path, size, and mtime.

    Avoids decoding extension arrays. Excludes the manifest, which records
    this fingerprint.
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
    """Classify a CurationV2 row as raw, merged, preview, or zero-unit."""
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


def _source_artifact_hashes(sorting_id) -> dict[str, str]:
    """Return the trace artifact hashes behind one sort, keyed by kind.

    Includes the ``MotionCorrectedRecording`` hash for a sort of corrected
    traces, so a cache built from other traces never validates.
    """
    from spyglass.spikesorting.v2.sorting import _TRACE_TABLES, SortingSelection

    lineage, traces = SortingSelection.resolve_effective_source(
        {"sorting_id": sorting_id}
    )
    if traces.kind == lineage.kind:
        # The traces are the lineage source's own row, fetched just now.
        return {lineage.kind: str(traces.row["content_hash"])}
    lineage_hash = (_TRACE_TABLES[lineage.kind] & lineage.key).fetch1(
        "content_hash"
    )
    return {
        lineage.kind: str(lineage_hash),
        traces.kind: str(traces.row["content_hash"]),
    }


def _raw_contributor_map(key) -> dict[str, list[int]]:
    """Return each curated unit's ordered original-sort contributors."""
    from spyglass.spikesorting.v2.curation import CurationV2

    return {
        str(unit_id): contributors
        for unit_id, contributors in CurationV2._raw_contributor_groups(
            key
        ).items()
    }


def hash_sorting_spike_content(sorting) -> str:
    """Hash exact unit ids and spike-sample frames for every segment."""
    import numpy as np

    digest = hashlib.sha256()
    # Omit sampling frequency: SI may swap in the recording's nearly-equal
    # float (100.0 vs 99.99999999999999), breaking reuse after a round-trip.
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


def _manifest_identity(row, recipe_row, role: str) -> dict:
    """The manifest fields resolved from the database (no spike trains)."""
    key = {
        "sorting_id": row["sorting_id"],
        "curation_id": int(row["curation_id"]),
    }
    return {
        "sorting_id": str(row["sorting_id"]),
        "curation_uuid": str(row["curation_uuid"]),
        "curation_id": int(row["curation_id"]),
        "role": role,
        "contributor_map": _raw_contributor_map(key),
        "merge_policy_version": MERGE_POLICY_VERSION,
        "source_artifact_hashes": _source_artifact_hashes(row["sorting_id"]),
        "waveform_recipe_hash": waveform_recipe_hash(recipe_row),
        "spikeinterface_version": _spikeinterface_version(),
        # Empty for a base cache; a derivative records its request here.
        "extension_request": {},
    }


def _manifest_with_sorting(identity: Mapping, sorting) -> dict:
    """Complete a :func:`_manifest_identity` with the curated sorting's part."""
    return {
        **identity,
        "curated_unit_ids": tuple(
            sorted(int(unit_id) for unit_id in sorting.get_unit_ids())
        ),
        "merged_spike_content_hash": hash_sorting_spike_content(sorting),
    }


def _manifest_prefix(row, sorting, recipe_row, role: str) -> dict:
    return _manifest_with_sorting(
        _manifest_identity(row, recipe_row, role), sorting
    )


def merged_analyzer_cache_inputs(curation_key, waveform_recipe, role):
    """Return JSON-native DB inputs for :func:`publish_resolved_merged_analyzer`.

    Used by ``CurationEvaluation.make_fetch``.
    """
    row = _resolve_curation_row(curation_key)
    recipe_row = _resolve_recipe(waveform_recipe, role)
    identity = _manifest_identity(row, recipe_row, role)
    return {
        "folder": str(
            curation_analyzer_path(
                row["sorting_id"],
                row["curation_uuid"],
                role,
                identity["waveform_recipe_hash"],
                identity["spikeinterface_version"],
            )
        ),
        "identity": identity,
    }


def normalize_extension_request(extra_extensions) -> dict[str, dict]:
    """Return ``{name: params}`` with JSON-native params, sorted by name."""
    from spyglass.spikesorting.v2._core.lookup_validation import _jsonable_blob

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

    ``exact=True`` (base cache) requires the saved set to equal the role's
    expected set. ``exact=False`` (derivative) only requires the base
    extensions plus ``extra``, since a derivative copies whatever its source
    carries.
    """
    from spyglass.spikesorting.v2._storage.analyzer_cache import (
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

    Keys absent from ``requested`` are unconstrained. Values compare
    JSON-natively (``1 == 1.0``).
    """
    from spyglass.spikesorting.v2._core.lookup_validation import _jsonable_blob

    if not analyzer.has_extension(name):
        return False
    if not requested:
        return True
    from spyglass.spikesorting.v2._storage.analyzer_cache import (
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


def _publish_manifest(
    staging_folder, prefix: Mapping, extension_inventory: dict[str, str]
) -> None:
    """Write the manifest of a fully built staging folder.

    Parameters
    ----------
    staging_folder : str or Path
        The analyzer folder being published; every analyzer file must already
        be in it, because the manifest's storage fingerprint covers them.
    prefix : Mapping
        The manifest identity fields (see :func:`_manifest_with_sorting`).
    extension_inventory : dict[str, str]
        The validated extension inventory (:func:`_extension_inventory`).
    """
    folder = Path(staging_folder)
    _write_manifest(
        folder,
        CurationAnalyzerManifest(
            **prefix,
            extension_inventory=extension_inventory,
            storage_fingerprint=_folder_storage_fingerprint(folder),
        ),
    )


def _load_valid_cached_analyzer(
    folder: Path, expected_prefix: dict, role: str, *, validate_analyzer=None
):
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
        if validate_analyzer is not None:
            validate_analyzer(analyzer)
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
    except _AnalyzerSparsityInputsError:
        raise
    except Exception as exc:  # noqa: BLE001 - cache is regeneratable
        # Log text, not the exception: its traceback would pin the memmapped
        # analyzer.
        logger.warning(
            "Curation analyzer validation failed at %s: %s.", folder, repr(exc)
        )
        return None


def _resolve_published_analyzer(
    folder: Path,
    sorting_id,
    expected_prefix: dict,
    role: str,
    build_into,
    *,
    validate_analyzer=None,
):
    """Build or reuse one validated cache slot under the per-sort lock."""
    from spyglass.utils import logger

    with analyzer_cache_lock(sorting_id):
        if folder.exists():
            analyzer = _load_valid_cached_analyzer(
                folder,
                expected_prefix,
                role,
                validate_analyzer=validate_analyzer,
            )
            if analyzer is not None:
                return analyzer
            logger.warning(
                "Curation analyzer cache %s is stale, incomplete, or corrupt; "
                "rebuilding it from the committed curation.",
                folder,
            )

        from spyglass.spikesorting.v2._core.progress import report_progress

        report_progress(f"Building this curation's {role} analyzer…")
        publish_analyzer_atomically(folder, build_into)
        analyzer = _load_valid_cached_analyzer(
            folder, expected_prefix, role, validate_analyzer=validate_analyzer
        )
        if analyzer is None:  # pragma: no cover - publish validation defense
            raise RuntimeError(
                "curation analyzer cache failed validation immediately after "
                f"publish: {folder}."
            )
        return analyzer


def _build_resolved_merged_analyzer(
    curated_sorting,
    recording,
    *,
    sorting_id,
    sorter_row,
    job_kwargs,
    waveform_params,
    statistics_spans,
    role,
    analyzer_folder,
):
    """Build a merged-curation analyzer from resolved inputs; no DB access."""
    from spyglass.spikesorting.v2._sorting.analyzer import (
        build_analyzer,
        ensure_extensions,
    )

    build_analyzer(
        curated_sorting,
        recording,
        {"sorting_id": sorting_id},
        sorter_row=sorter_row,
        job_kwargs=job_kwargs,
        analyzer_folder=Path(analyzer_folder),
        waveform_params=dict(waveform_params),
        statistics_spans=statistics_spans,
    )
    analyzer = load_analyzer_folder(analyzer_folder)
    if not analyzer.has_recording():
        analyzer.set_temporary_recording(recording)
    if role == "display":
        ensure_extensions(
            analyzer,
            STANDARD_DISPLAY_ANALYZER_EXTENSIONS,
            peak_sign=resolve_peak_sign(sorter_row["params"]),
            job_kwargs=job_kwargs,
        )
    return analyzer


def publish_resolved_merged_analyzer(
    cache_inputs: Mapping,
    curated_sorting,
    recording,
    *,
    sorter_row,
    job_kwargs,
    waveform_params,
    statistics_spans,
):
    """Publish or reuse a merged curation's analyzer without the database.

    ``cache_inputs`` comes from :func:`merged_analyzer_cache_inputs`. Uses the
    cache slot :func:`_resolve_curation_analyzer` serves, so evaluation and
    review share one build. Returns the published (read-only) analyzer.
    """
    identity = dict(cache_inputs["identity"])
    role = identity["role"]
    expected_prefix = _manifest_with_sorting(identity, curated_sorting)

    def _build_base(staging_folder):
        analyzer = _build_resolved_merged_analyzer(
            curated_sorting,
            recording,
            sorting_id=identity["sorting_id"],
            sorter_row=sorter_row,
            job_kwargs=job_kwargs,
            waveform_params=waveform_params,
            statistics_spans=statistics_spans,
            role=role,
            analyzer_folder=staging_folder,
        )
        _publish_manifest(
            staging_folder,
            expected_prefix,
            _extension_inventory(analyzer, role),
        )

    return _resolve_published_analyzer(
        Path(cache_inputs["folder"]),
        identity["sorting_id"],
        expected_prefix,
        role,
        _build_base,
        validate_analyzer=analyzer_sparsity_validator(
            lambda: sorter_row.get("params")
        ),
    )


def display_analyzer_request(curation_ref) -> dict:
    """Return the resolver kwargs for a curation's DISPLAY analyzer.

    Uses the sort's stored ``Sorting.display_waveform_params_name`` so every
    display consumer shares one published cache.

    Parameters
    ----------
    curation_ref : CurationRef or Mapping
        A ``CurationRef`` or a curation key carrying ``sorting_id``.

    Returns
    -------
    dict
        ``{"curation_ref", "waveform_recipe", "role": "display"}`` for
        :func:`_resolve_curation_analyzer` and the context managers below.
    """
    from spyglass.spikesorting.v2.sorting import Sorting

    sorting_id = (
        curation_ref["sorting_id"]
        if isinstance(curation_ref, Mapping)
        else curation_ref.sorting_id
    )
    return {
        "curation_ref": curation_ref,
        "waveform_recipe": resolve_display_waveform_params_name(
            Sorting, sorting_id
        ),
        "role": "display",
    }


def _resolve_curation_analyzer(
    curation_ref,
    waveform_recipe: str,
    role: str = "display",
    *,
    extra_extensions: Mapping[str, dict] | None = None,
):
    """Resolve the published, read-only analyzer for internal reads.

    - raw namespace: the shared sort analyzer
    - merged namespace: the per-generation cache
    - unmet ``extra_extensions``: a disk-backed derivative for that request

    Use :func:`open_curation_analyzer` for a mutable working copy.
    """
    from spyglass.spikesorting.v2._sorting.analyzer import (
        reconstruct_recording_and_sorting,
    )
    from spyglass.spikesorting.v2._sorting.fetch import (
        fetch_sorter_analyzer_inputs,
    )
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

    # Lazy: only a cache check or build needs these.
    @functools.cache
    def sorter_inputs() -> tuple[dict, dict]:
        return fetch_sorter_analyzer_inputs({"sorting_id": row["sorting_id"]})

    def sorter_params():
        return sorter_inputs()[0].get("params")

    def source_recording():
        recording, _sorting = reconstruct_recording_and_sorting(
            Sorting(), {"sorting_id": row["sorting_id"]}
        )
        return recording

    validate_analyzer = analyzer_sparsity_validator(sorter_params)
    # The full request (not ``needs_compute``) keys the derivative, so a
    # larger request never reuses a smaller derivative.
    request = normalize_extension_request(extra_extensions)
    if any(name in request for name in POLARITY_EXTENSIONS):
        # Polarity joins the identity so requests with and without it differ.
        request = normalize_extension_request(
            polarity_extension_params(
                request, resolve_peak_sign(sorter_params()), request
            )
        )
    if namespace == "raw":
        # Absent extensions persist to the shared analyzer. Present ones with
        # other params go to a derivative so other readers are unaffected.
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
        # Built lazily: the manifest scans spike trains and only a derivative
        # needs it.
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
            _spikeinterface_version(),
        )

        def _build_base(staging_folder):
            sorter_row, job_kwargs = sorter_inputs()
            analyzer = _build_resolved_merged_analyzer(
                curated_sorting,
                source_recording(),
                sorting_id=row["sorting_id"],
                sorter_row=sorter_row,
                job_kwargs=job_kwargs,
                waveform_params=dict(recipe_row["params"]),
                statistics_spans=Sorting().get_statistics_spans(
                    {"sorting_id": row["sorting_id"]}
                ),
                role=role,
                analyzer_folder=staging_folder,
            )
            _publish_manifest(
                staging_folder,
                expected_prefix,
                _extension_inventory(analyzer, role),
            )

        base = _resolve_published_analyzer(
            base_folder,
            row["sorting_id"],
            expected_prefix,
            role,
            _build_base,
            validate_analyzer=validate_analyzer,
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

    # Derivative: an on-disk copy of the base plus the unmet extensions; the
    # base stays untouched.
    derived_prefix = dict(expected_prefix)
    derived_prefix["extension_request"] = request
    derived_folder = curation_analyzer_path(
        row["sorting_id"],
        row["curation_uuid"],
        role,
        expected_prefix["waveform_recipe_hash"],
        _spikeinterface_version(),
        extension_request_hash=derived_extension_request_hash(request),
    )

    def _build_derivative(staging_folder):
        derivative = copy_analyzer_folder(base, Path(staging_folder))
        if not derivative.has_recording():
            derivative.set_temporary_recording(source_recording())
        _compute_request_on_copy(
            derivative,
            request,
            needs_compute,
            job_kwargs=sorter_inputs()[1],
        )
        _publish_manifest(
            staging_folder,
            derived_prefix,
            _extension_inventory(derivative, role, tuple(request), exact=False),
        )

    return _resolve_published_analyzer(
        derived_folder,
        row["sorting_id"],
        derived_prefix,
        role,
        _build_derivative,
        validate_analyzer=validate_analyzer,
    )


def _compute_request_on_copy(analyzer, request, needs_compute, *, job_kwargs):
    """Compute ``needs_compute`` on a working copy, restoring what SI removes.

    SpikeInterface deletes every dependent of a recomputed extension, so each
    required dependent (base set plus request) is recomputed too, at the
    requested params or else its stored params. Runs inside
    ``isolated_si_metric_defaults`` because SI otherwise merges stored
    ``metric_params`` into its class-level defaults.

    Raises
    ------
    ValueError
        If the result does not satisfy ``request``.
    """
    from spikeinterface.core.sortinganalyzer import _get_children_dependencies

    from spyglass.spikesorting.v2._core.si_metric_patches import (
        isolated_si_metric_defaults,
    )
    from spyglass.spikesorting.v2._sorting.dispatch import without_random_seed

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
    compute_kwargs = without_random_seed(job_kwargs)
    for name in plan:
        if analyzer.has_extension(name):
            analyzer.delete_extension(name)
    with isolated_si_metric_defaults():
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

    The copy lives in a temp dir under Spyglass's temp dir, removed on exit,
    so mutating it cannot alter the published cache.
    """
    published = _resolve_curation_analyzer(
        curation_ref, waveform_recipe, role, extra_extensions=extra_extensions
    )
    with analyzer_working_copy(published) as working:
        yield working


@contextmanager
def analyzer_working_copy(published):
    """Yield a mutable working copy of an already-resolved published analyzer.

    The copy step of :func:`open_curation_analyzer`, for callers that resolved
    (and validated) the published analyzer themselves.
    """
    import shutil
    import tempfile

    from spyglass.settings import temp_dir as spyglass_temp_dir
    from spyglass.spikesorting.v2._storage.analyzer_cache import (
        ANALYZER_FOLDER_SUFFIX,
    )

    tmp = tempfile.mkdtemp(
        prefix="v2_curation_analyzer_", dir=spyglass_temp_dir
    )
    working = None
    try:
        working = copy_analyzer_folder(
            published, Path(tmp) / f"working{ANALYZER_FOLDER_SUFFIX}"
        )
        yield load_analyzer_extensions(working)
    finally:
        # The caller still holds the analyzer; drop its memmaps before rmtree
        # or NFS leaves .nfs placeholders behind.
        if working is not None:
            for extension in working.extensions.values():
                extension.data.clear()
            working.extensions.clear()
        working = None
        gc.collect()
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

    Do not mutate it; use :func:`open_curation_analyzer` for a working copy.
    """
    yield _resolve_curation_analyzer(
        curation_ref, waveform_recipe, role, extra_extensions=extra_extensions
    )

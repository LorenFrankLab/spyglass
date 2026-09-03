"""Curation-scoped SortingAnalyzer resolution and immutable cache ownership.

Raw-namespace curations reuse the existing sort analyzer. Committed curations
whose unit set differs from the raw sort build a per-generation analyzer over
their exact stored spike trains. The cache key includes ``curation_uuid`` plus
the recipe content and SpikeInterface version, and a manifest validates the
scientific inputs and every published extension before a folder is reused.

Published curation analyzers are immutable by ownership. Supported callers use
the private resolver for reads, request a detached memory copy for direct
access, or compute unusual extensions on a context-managed memory derivative.
No supported path mutates the published zarr folder.
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
    curation_analyzer_path,
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
    extension_inventory: dict[str, str]

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


def classify_curation_analyzer_namespace(curation_ref) -> str:
    """Classify one curation as raw, merged, preview, or zero-unit."""
    from spyglass.spikesorting.v2.curation import CurationV2

    row = _resolve_curation_row(curation_ref)
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
    }


def _spikeinterface_version() -> str:
    import spikeinterface as si

    return str(si.__version__)


def _expected_extensions(role: str) -> tuple[str, ...]:
    if role == "display":
        return (
            *BASE_ANALYZER_EXTENSIONS,
            *STANDARD_DISPLAY_ANALYZER_EXTENSIONS,
        )
    return BASE_ANALYZER_EXTENSIONS


def _extension_inventory(analyzer, role: str) -> dict[str, str]:
    """Validate and fingerprint the complete published extension set."""
    expected = _expected_extensions(role)
    saved = set(analyzer.get_saved_extension_names())
    if saved != set(expected):
        raise ValueError(
            "curation analyzer extension set is incomplete or unexpected: "
            f"expected {sorted(expected)}, found {sorted(saved)}."
        )
    inventory: dict[str, str] = {}
    for name in expected:
        extension = analyzer.get_extension(name)
        # Reading the payload is a completeness check: a killed zarr write can
        # leave extension metadata present while its data arrays are partial.
        extension.get_data()
        inventory[name] = _content_hash(extension.params or {})
    return inventory


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
    import spikeinterface as si

    from spyglass.utils import logger

    try:
        stored = _read_manifest(folder)
        comparable = dict(stored)
        stored_inventory = comparable.pop("extension_inventory")
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
        analyzer = si.load_sorting_analyzer(folder)
        current_inventory = _extension_inventory(analyzer, role)
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
    import spikeinterface as si

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
    analyzer = si.load_sorting_analyzer(analyzer_folder)
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
):
    """Resolve the cache-backed analyzer for controlled internal reads."""
    import spikeinterface as si

    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting

    row = _resolve_curation_row(curation_ref)
    key = {
        "sorting_id": row["sorting_id"],
        "curation_id": int(row["curation_id"]),
    }
    namespace = classify_curation_analyzer_namespace(key)
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
    if namespace == "raw":
        return Sorting().get_analyzer(
            {"sorting_id": row["sorting_id"]},
            waveform_params_name=waveform_recipe,
        )

    curated_sorting = CurationV2.get_merged_sorting(key)
    expected_prefix = _manifest_prefix(row, curated_sorting, recipe_row, role)
    folder = curation_analyzer_path(
        row["sorting_id"],
        row["curation_uuid"],
        role,
        expected_prefix["waveform_recipe_hash"],
        si.__version__,
    )

    def _build(staging_folder):
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
        )
        _write_manifest(Path(staging_folder), manifest)

    return _resolve_published_analyzer(
        folder,
        row["sorting_id"],
        expected_prefix,
        role,
        _build,
    )


def get_curation_analyzer(
    curation_ref,
    waveform_recipe: str,
    role: str = "display",
):
    """Return a detached memory analyzer for direct caller access.

    The persistent object stays internal because SpikeInterface analyzers are
    mutable. Mutating this returned copy cannot alter the published cache.
    """
    return _resolve_curation_analyzer(
        curation_ref, waveform_recipe, role
    ).save_as(format="memory")


@contextmanager
def curation_analyzer_with_extensions(
    curation_ref,
    waveform_recipe: str,
    role: str = "display",
    *,
    extra_extensions: Mapping[str, dict] | None = None,
):
    """Yield a read analyzer, deriving missing extensions in memory only."""
    from spyglass.spikesorting.v2._sorting_analyzer import ensure_extensions
    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        SortingSelection,
    )
    from spyglass.spikesorting.v2.utils import _resolved_job_kwargs

    base = _resolve_curation_analyzer(curation_ref, waveform_recipe, role)
    requested = dict(extra_extensions or {})
    missing = [name for name in requested if not base.has_extension(name)]
    if not missing:
        yield base
        return

    row = _resolve_curation_row(curation_ref)
    derivative = base.save_as(format="memory")
    if not derivative.has_recording():
        from spyglass.spikesorting.v2._sorting_analyzer import (
            reconstruct_recording_and_sorting,
        )
        from spyglass.spikesorting.v2.sorting import Sorting

        recording, _sorting = reconstruct_recording_and_sorting(
            Sorting(), {"sorting_id": row["sorting_id"]}
        )
        derivative.set_temporary_recording(recording)
    sorter_job_kwargs = (
        SorterParameters
        & (SortingSelection & {"sorting_id": row["sorting_id"]})
    ).fetch1("job_kwargs")
    ensure_extensions(
        derivative,
        missing,
        job_kwargs=_resolved_job_kwargs(sorter_job_kwargs),
        extension_params={name: requested[name] for name in missing},
    )
    try:
        yield derivative
    finally:
        del derivative

"""Web-based curation views for v2 sorts, built with FigPack.

Replaces v1's ``spyglass.spikesorting.v1.figurl_curation``.
``FigPackCurationSelection`` records a UI configuration (label palette,
unit-table properties, upload mode) for a committed ``CurationV2`` row.
``FigPackCuration`` builds and publishes the view and stores its URI;
``fetch_curation_from_uri`` reads edited labels and merge groups back in the
shape ``CurationV2.insert_curation`` takes.

SpikeInterface supplies the sorting summary; Spyglass adds raster and
autocorrelogram tabs, the evaluation table, and a label control. Commits stay
in the notebook; the browser saves only annotations.

``figpack`` and ``figpack_spike_sorting`` (the ``spikesorting-v2-curation``
extra) are imported lazily, so this module loads without them.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple
import urllib.error
import urllib.request

import datajoint as dj
import pandas as pd
import spikeinterface

from spyglass.spikesorting.v2._storage.analyzer_cache import (
    install_staged_folder,
)
from spyglass.spikesorting.v2._curation.transforms import ManualMergeAction
from spyglass.spikesorting.v2._review.annotations import (
    curation_annotations_to_labels_and_merges,
    default_label_options,
    figpack_config_hash,
    labels_and_merges_to_annotations,
    normalize_displayed_unit_properties,
    pack_display_config,
    unpack_display_config,
)
from spyglass.spikesorting.v2._storage.observation_io import (
    ReviewTimelineInputs,
)
from spyglass.spikesorting.v2._review.view import (
    build_curation_view,
    require_figpack,
)
from spyglass.spikesorting.v2._core.selection_identity import (
    assert_supplied_id_matches,
    deterministic_id,
    existing_selection_pk,
)
from spyglass.spikesorting.v2._storage.staged_outputs import (
    StagedOutputCleanupMixin,
)
from spyglass.spikesorting.v2.curation import CurationV2
from spyglass.spikesorting.v2.exceptions import (
    FigPackDisplayedUnitPropertyError,
    FigPackIdentityError,
    FigPackRetrievalError,
    FigPackUploadError,
    SchemaBypassError,
)
from spyglass.spikesorting.v2._core.table_integrity import (
    SelectionMasterInsertGuard,
)
from spyglass.utils import SpyglassMixin
from spyglass.utils.mixins.populate import StagedOutputs
from spyglass.settings import sg_config
from spyglass.spikesorting.v2._review import visualization as _viz
from spyglass.spikesorting.v2._review.profile import ReviewDisplayOptions
from spyglass.spikesorting.v2._storage.analyzer_cache import v2_storage_root
from spyglass.spikesorting.v2._storage.observation_io import (
    resolve_review_timeline_inputs,
    review_timeline_from_inputs,
)
from spyglass.spikesorting.v2.curation_api import (
    CurationRef,
    EvaluationResult,
    _read_unit_properties,
)
from spyglass.spikesorting.v2.metric_curation import QualityMetricParameters
from spyglass.utils import logger
from spyglass.utils.mixins.populate import populate_key_lock
from spyglass.utils.progress import report_progress

schema = dj.schema("spikesorting_v2_figpack_curation")


@dataclass(frozen=True)
class FigPackBuildResult:
    """Outcome of resolving a persisted FigPack curation view."""

    uri: str
    reused: bool
    figpack_curation_id: uuid.UUID


# ---- figpack access + storage helpers (lazy / DB-light) ------------------


def figpack_cache_root() -> Path:
    """Return the root directory for saved FigPack bundles.

    ``dj.config["custom"]["spikesorting_v2_figpack_dir"]`` when set, else
    ``Path(base_dir) / "spikesorting_v2" / "figpack"``.
    """
    return v2_storage_root("spikesorting_v2_figpack_dir", "figpack")


def figpack_bundle_path(figpack_curation_id) -> Path:
    """Return the durable bundle folder for one offline FigPack curation."""
    return figpack_cache_root() / f"{figpack_curation_id}"


def _figure_base(uri: str) -> str:
    """Return a figure URI without ``file://``, ``index.html`` or a final ``/``."""
    text = str(uri)
    if text.endswith("index.html"):
        text = text[: -len("index.html")]
    if text.startswith("file://"):
        text = text[len("file://") :]
    return text.rstrip("/")


def _load_figure_json(uri: str, filename: str, *, missing_ok: bool) -> dict:
    """Fetch one figure JSON sidecar (HTTP or local); fail closed.

    With ``missing_ok``, only a 404 or missing local file yields ``{}``.
    Any other failure raises ``FigPackRetrievalError`` so an unreachable
    figure is never read as "no edits".
    """
    base = _figure_base(uri)
    sidecar_url = base + f"/{filename}"

    if base.startswith(("http://", "https://")):
        from spyglass import __version__

        # figpack.org refuses Python's default User-Agent with HTTP 403.
        request = urllib.request.Request(
            sidecar_url, headers={"User-Agent": f"spyglass/{__version__}"}
        )
        try:
            # A read timeout raises TimeoutError, not URLError.
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            if exc.code == 404 and missing_ok:
                return {}
            hint = (
                " figpack.org removes unpinned figures after they expire "
                "(24 hours, or 6 hours if ephemeral)."
                if exc.code == 404
                else ""
            )
            raise FigPackRetrievalError(
                f"Failed to fetch {sidecar_url}: HTTP {exc.code}.{hint}"
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            reason = getattr(exc, "reason", exc)
            raise FigPackRetrievalError(
                f"Could not reach {sidecar_url}: {reason}. Refusing to "
                "treat an unreachable figure as having no edits."
            ) from exc
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise FigPackRetrievalError(
                f"Malformed JSON at {sidecar_url}: {exc}."
            ) from exc

    figure_dir = Path(base)
    if not figure_dir.exists():
        raise FigPackRetrievalError(
            f"FigPack figure path does not exist: {figure_dir}. Refusing to "
            "treat a missing/typoed figure as having no edits."
        )
    path = figure_dir / filename
    if not path.exists():
        if missing_ok:
            return {}
        raise FigPackRetrievalError(
            f"FigPack figure is missing required {filename}: {figure_dir}."
        )
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise FigPackRetrievalError(
            f"Malformed JSON at {path}: {exc}."
        ) from exc


def _is_hosted(uri: str) -> bool:
    """Return whether ``uri`` is a figpack.org figure, not a local bundle."""
    return _figure_base(uri).startswith(("http://", "https://"))


def _load_annotations_json(uri: str) -> dict:
    """Load a figure's annotations.

    A local bundle without the file is a pristine figure. Every hosted figure
    is uploaded with its annotations, so a missing file there raises.
    """
    return _load_figure_json(
        uri, "annotations.json", missing_ok=not _is_hosted(uri)
    )


def _hosted_figure_state(uri: str) -> str:
    """Return ``"live"``, ``"expired"`` or ``"removed"`` for a hosted figure.

    figpack.org expires unpinned figures, then removes them. ``figpack.json``
    gives the expiration in epoch milliseconds, ``0`` for never.
    """
    info = _load_figure_json(uri, "figpack.json", missing_ok=True)
    if not info:
        return "removed"
    expiration = int(info.get("expiration") or 0)
    if expiration and expiration <= time.time() * 1000:
        return "expired"
    return "live"


def _hosted_figure_has_edits(selection: dict, uri: str) -> bool:
    """Return whether a hosted figure's labels or merges differ from its seed."""
    row = (FigPackCurationSelection & selection).fetch1()
    curation_key = {
        "sorting_id": row["sorting_id"],
        "curation_id": int(row["curation_id"]),
    }
    seed = labels_and_merges_to_annotations(
        *_existing_curation_state(curation_key),
        label_options=list(row["label_options"]),
    )
    return curation_annotations_to_labels_and_merges(
        _load_annotations_json(uri)
    ) != curation_annotations_to_labels_and_merges(seed)


def _assert_figure_identity(
    uri: str,
    parent_curation_key: dict,
    *,
    expected_config_hash: str | None = None,
) -> dict:
    """Return the selection of the figure Spyglass built at ``uri`` for a parent.

    The figure must be a ``FigPackCuration`` row of this exact curation, so
    annotations are never attached to a different curation. FigPack hosts
    only its own file types, so the identity lives in the database.

    Parameters
    ----------
    uri : str
        The figure's local bundle path or hosted URL.
    parent_curation_key : dict
        ``sorting_id`` and ``curation_id`` of the curation the figure shows.
    expected_config_hash : str, optional
        The selection's ``figpack_config_hash`` when it is already known.

    Returns
    -------
    dict
        The ``FigPackCurationSelection`` row of the figure.

    Raises
    ------
    FigPackIdentityError
        If the parent is missing, or no single figure built from it has
        ``uri``.
    """
    try:
        parent_key = {
            "sorting_id": parent_curation_key["sorting_id"],
            "curation_id": int(parent_curation_key["curation_id"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise FigPackIdentityError(
            "Verified FigPack import requires a parent with sorting_id and "
            "curation_id."
        ) from exc
    if not (CurationV2 & parent_key):
        raise FigPackIdentityError(
            f"FigPack parent no longer exists: {parent_key}."
        )
    built = (FigPackCuration * FigPackCurationSelection & parent_key).fetch(
        "figpack_curation_id", "figpack_uri", as_dict=True
    )
    matches = [
        row["figpack_curation_id"]
        for row in built
        if _figure_base(row["figpack_uri"]) == _figure_base(uri)
    ]
    if len(matches) != 1:
        raise FigPackIdentityError(
            f"{uri} is not a FigPack figure built from curation "
            f"{parent_key}. Import only figures that FigPackCuration built "
            "for this curation."
        )
    selection_key = {"figpack_curation_id": matches[0]}
    selection = (FigPackCurationSelection & selection_key).fetch1()
    if expected_config_hash is not None and str(
        selection["figpack_config_hash"]
    ) != str(expected_config_hash):
        raise FigPackIdentityError(
            f"The FigPack figure at {uri} has a different configuration than "
            "the review."
        )
    try:
        _assert_selection_identity(selection, selection_key)
    except SchemaBypassError as exc:
        raise FigPackIdentityError(
            "FigPack figure references a selection whose content-addressed "
            "identity is invalid."
        ) from exc
    return selection


@contextmanager
def _curation_view_analyzer(curation_ref):
    """Yield a curation's published display analyzer for the curation view.

    Builds or extends the published cache with the sorting-summary
    extensions as needed. Must not be mutated.
    """
    from spyglass.spikesorting.v2._curation.analyzer import (
        curation_analyzer_with_extensions,
        display_analyzer_request,
    )

    required = _viz.DISPLAY_WIDGET_EXTENSIONS["plot_sorting_summary"]
    with curation_analyzer_with_extensions(
        **display_analyzer_request(curation_ref),
        extra_extensions={name: {} for name in required},
    ) as analyzer:
        yield analyzer


def _resolve_curation_view_inputs(
    curation_key: dict, display_options
) -> tuple[str, ReviewTimelineInputs]:
    """Resolve the DB inputs of :func:`._review.view.build_curation_view`.

    Returns ``(analyzer_folder, timeline_inputs)``. ``display_options`` is
    validated before any analyzer work.
    """
    require_figpack()
    ReviewDisplayOptions.from_mapping(display_options)

    with _curation_view_analyzer(curation_key) as analyzer:
        analyzer_folder = str(analyzer.folder)
    return analyzer_folder, resolve_review_timeline_inputs(curation_key)


def _assert_figpack_curatable(curation_key: dict) -> None:
    """Assert a curation is committed before building its exact analyzer.

    Called by ``insert_selection`` and again by ``make_fetch``, since
    ``allow_direct_insert`` can bypass the former.
    """
    CurationV2.assert_committed_curation(
        curation_key, context="FigPackCuration"
    )


def _assert_selection_identity(selection: dict, key: dict) -> None:
    """Recheck the content-addressed identity at consume time (bypass guard).

    Recomputes ``figpack_config_hash`` and ``figpack_curation_id`` from the
    row and raises ``SchemaBypassError`` on mismatch.
    """
    # insert_selection sets ephemeral=False when offline.
    if not bool(selection["upload"]) and bool(selection["ephemeral"]):
        raise SchemaBypassError(
            f"FigPackCurationSelection {key['figpack_curation_id']} has "
            "upload=False with ephemeral=True (inert offline; insert_selection "
            "normalizes it away). This is a raw-insert bypass -- drop the row "
            "and re-insert via insert_selection()."
        )
    try:
        displayed_unit_properties, review_config = unpack_display_config(
            selection["displayed_unit_properties"]
        )
    except (TypeError, ValueError) as exc:
        raise SchemaBypassError(
            f"FigPackCurationSelection {key['figpack_curation_id']} has "
            f"invalid displayed_unit_properties "
            f"{selection.get('displayed_unit_properties')!r} (a raw-insert "
            "bypass). Drop the row and re-insert via insert_selection()."
        ) from exc
    expected_hash = figpack_config_hash(
        sorting_id=selection["sorting_id"],
        curation_id=selection["curation_id"],
        curation_uuid=(
            CurationV2
            & {
                "sorting_id": selection["sorting_id"],
                "curation_id": selection["curation_id"],
            }
        ).fetch1("curation_uuid"),
        label_options=list(selection["label_options"]),
        displayed_unit_properties=displayed_unit_properties,
        upload=bool(selection["upload"]),
        ephemeral=bool(selection["ephemeral"]),
        review_config=review_config,
    )
    if selection["figpack_config_hash"] != expected_hash:
        raise SchemaBypassError(
            f"FigPackCurationSelection {key['figpack_curation_id']} has a "
            "figpack_config_hash that does not match its own label_options / "
            "displayed_unit_properties / upload / ephemeral (a raw-insert "
            "bypass). Drop the row and re-insert via insert_selection()."
        )
    expected_id = deterministic_id(
        "figpack_curation",
        {
            "sorting_id": selection["sorting_id"],
            "curation_id": selection["curation_id"],
            "figpack_config_hash": expected_hash,
        },
    )
    if uuid.UUID(str(key["figpack_curation_id"])) != expected_id:
        raise SchemaBypassError(
            f"FigPackCurationSelection id {key['figpack_curation_id']} is not "
            f"the content-addressed id {expected_id} for its fields (a raw-"
            "insert bypass). Drop the row and re-insert via insert_selection()."
        )


def _existing_curation_state(curation_key: dict) -> tuple[dict, list]:
    """Return editable committed state in the curation's own namespace.

    Existing labels seed the control. Already-applied merges are provenance,
    not pending edits, so the returned merge list is always empty.
    """
    return CurationV2._labels_by_unit(curation_key), []


def _review_context_table(curation_key: dict, review_config: dict | None):
    """Build the read-only evaluation/provenance table for a guided review."""
    if review_config is None:
        return None

    evaluation = EvaluationResult.from_key(
        {"curation_evaluation_id": review_config["curation_evaluation_id"]}
    )
    if evaluation.curation.as_key() != {
        "sorting_id": curation_key["sorting_id"],
        "curation_id": int(curation_key["curation_id"]),
    }:
        raise SchemaBypassError(
            "FigPack review evaluation does not target the selected curation."
        )

    unit_ids = [
        int(value)
        for value in (CurationV2.Unit & curation_key).fetch(
            "unit_id", order_by="unit_id"
        )
    ]
    if review_config.get("annotation_sets"):
        raise SchemaBypassError(
            "FigPack review configuration selects custom unit-annotation "
            "sets, which v2 does not support; start a new review."
        )
    properties = _read_unit_properties(
        CurationRef.from_key(curation_key), evaluation=evaluation
    )
    requested = list(review_config["displayed_unit_properties"])
    missing = [name for name in requested if name not in properties.columns]
    if missing:
        raise FigPackDisplayedUnitPropertyError(
            "Review requests properties absent from its selected evaluation: "
            f"{missing}. Available properties: "
            f"{list(map(str, properties.columns))}."
        )
    metrics = properties.reindex(unit_ids)[requested]

    proposed_labels = evaluation.proposed_labels
    suggested_groups = evaluation.suggested_merges
    raw_provenance = CurationV2._raw_contributor_groups(curation_key)

    def groups_for(unit_id: int) -> str:
        groups = [group for group in suggested_groups if unit_id in group]
        return "; ".join(",".join(map(str, group)) for group in groups)

    # Actionable columns first so they stay visible on narrow screens.
    actions = pd.DataFrame(
        {
            "proposed_labels": [
                ",".join(proposed_labels.get(unit_id, []))
                for unit_id in unit_ids
            ],
            "proposed_merge_groups": [
                groups_for(unit_id) for unit_id in unit_ids
            ],
            "merged_from": [
                (
                    ",".join(map(str, raw_provenance.get(unit_id, [])))
                    if len(raw_provenance.get(unit_id, [])) > 1
                    else ""
                )
                for unit_id in unit_ids
            ],
        },
        index=metrics.index,
    )
    table = pd.concat([actions, metrics], axis=1)
    coverage = evaluation.missing_qc_inputs().reindex(unit_ids)
    table.insert(0, "unavailable_qc", coverage)

    refractory_ms = QualityMetricParameters.get_isi_threshold_ms(
        evaluation.spec.metric_params_name
    )
    table.attrs["qc_context"] = (
        f"Evaluation `{evaluation.evaluation_id}`: "
        f"{int(coverage.ne('').sum())}/{len(coverage)} units have unavailable rule inputs. "
        "A missing-policy pass means not flagged, not verified quality. "
        f"`isi_violation` = violating intervals / (spikes − 1), "
        f"with a {refractory_ms:g} ms refractory window; "
        "it is not SI's `isi_violations_ratio` or a contamination percentage."
    )
    table.index.name = "unit_id"
    return table


def _write_annotations(bundle: Path, annotations: dict) -> None:
    """Write the seeded FigPack annotations into a saved bundle."""
    (bundle / "annotations.json").write_text(
        json.dumps(annotations, indent=2, sort_keys=True)
    )


def _publish_view(
    view,
    *,
    upload: bool,
    ephemeral: bool,
    title: str,
    figpack_curation_id,
    annotations: dict,
) -> tuple[str, str | None]:
    """Publish a built view; return its URI and any privately staged bundle.

    ``upload=True`` publishes to figpack.org (API key required unless
    ``ephemeral``) and returns ``(url, None)``. ``upload=False`` saves the
    bundle to a private sibling folder and returns ``(durable folder, staged
    folder)``; ``make_insert`` moves it into place.
    """
    if upload:
        api_key = sg_config.figpack_api_key
        if not ephemeral and not api_key:
            raise FigPackUploadError(
                "FigPack upload=True requires an API key: set "
                "dj.config['custom']['figpack_api_key'] (or the "
                "FIGPACK_API_KEY environment variable), use ephemeral=True "
                "for a temporary figure, or use upload=False to save a local "
                "bundle."
            )
        # Save locally first so hosted and local bundles carry the same
        # annotations; consolidated metadata keeps the upload file count down.
        from figpack.core._upload_bundle import _upload_bundle

        with tempfile.TemporaryDirectory(prefix="spyglass-figpack-") as tmp:
            view.save(tmp, title=title)
            _write_annotations(Path(tmp), annotations)
            url = _upload_bundle(
                tmp,
                api_key=api_key,
                title=title,
                ephemeral=ephemeral,
                use_consolidated_metadata_only=True,
            )
        return url, None

    bundle = figpack_bundle_path(figpack_curation_id)
    bundle.parent.mkdir(parents=True, exist_ok=True)
    # Attempt-unique folder; the durable one is untouched until insert.
    staged = bundle.parent / f".{bundle.name}.build-{uuid.uuid4().hex}"
    try:
        view.save(str(staged), title=title)
        _write_annotations(staged, annotations)
    except BaseException:
        shutil.rmtree(staged, ignore_errors=True)
        raise
    return str(bundle), str(staged)


class FigPackCurationFetched(NamedTuple):
    """DB inputs of :meth:`FigPackCuration.make_compute`.

    Attributes
    ----------
    curation_key : dict
        ``{"sorting_id": str, "curation_id": int}`` of the curation viewed.
    label_options : list of str
        Curation label palette, in display order.
    displayed_unit_properties : list of str or None
        Requested SpikeInterface unit-table columns (``None``: SI defaults).
    display_options : dict or None
        A profile-backed review's display budget (``None``: the defaults).
    upload, ephemeral : bool
        Publish mode of the selection.
    seed_labels : dict
        The curation's committed labels, ``{unit_id: [label, ...]}``.
    annotations : dict
        FigPack annotations seeded from ``seed_labels``.
    review_table : pandas.DataFrame or None
        A profile-backed review's evaluation table, indexed by ``unit_id``.
    analyzer_folder : str
        The resolved display analyzer folder, carrying the view extensions.
    timeline_inputs : ReviewTimelineInputs
        Resolved inputs of the curation's review timeline.
    """

    curation_key: dict
    label_options: list
    displayed_unit_properties: list | None
    display_options: dict | None
    upload: bool
    ephemeral: bool
    seed_labels: dict
    annotations: dict
    review_table: "pd.DataFrame | None"
    analyzer_folder: str
    timeline_inputs: ReviewTimelineInputs


class FigPackCurationComputed(NamedTuple):
    """The ``FigPackCuration`` secondary fields make_compute returns.

    ``staged_bundle`` is not a column: it is the folder an offline bundle
    was saved into (``None`` when hosted), moved to ``figpack_uri`` by
    ``make_insert``.
    """

    figpack_uri: str
    figpack_version: str
    figpack_spike_sorting_version: str
    spikeinterface_version: str
    staged_bundle: str | None

    def staged_outputs(self) -> StagedOutputs:
        """The staged offline bundle ``make_insert`` moves into place."""
        return StagedOutputs(
            paths=(self.staged_bundle,) if self.staged_bundle else ()
        )


# ---- tables --------------------------------------------------------------


@schema
class FigPackCurationSelection(
    SelectionMasterInsertGuard, SpyglassMixin, dj.Manual
):
    """A committed ``CurationV2`` row paired with a FigPack UI configuration.

    ``figpack_curation_id`` is content-addressed over the curation and UI
    config, so one curation can have several configurations. Use
    ``insert_selection``; raw inserts are blocked.
    """

    definition = """
    figpack_curation_id: uuid
    ---
    -> CurationV2
    figpack_config_hash: char(64)  # sha256 over FigPack UI config
    label_options: blob            # curation label palette, in display order
    displayed_unit_properties=null: blob # display columns + optional immutable review snapshot
    upload: bool                   # True publishes a hosted figpack.org URI
    ephemeral: bool                # temporary hosted figure (no API key needed)
    """

    @classmethod
    def insert_selection(
        cls,
        curation_key: dict,
        *,
        label_options: list[str] | None = None,
        displayed_unit_properties: list[str] | None = None,
        upload: bool = False,
        ephemeral: bool = False,
        review_config: dict | None = None,
        figpack_curation_id=None,
    ) -> dict:
        """Insert or find a FigPack curation selection; return PK-only dict.

        Parameters
        ----------
        curation_key : dict
            ``{sorting_id, curation_id}`` of a committed ``CurationV2`` row.
        label_options : list of str, optional
            Curation label palette, in display order. Defaults to
            ``["accept", "mua", "noise"]``.
        displayed_unit_properties : list of str, optional
            Unit-table properties to pass to SpikeInterface's FigPack sorting
            summary. ``None`` (default) keeps SpikeInterface's defaults; ``[]``
            requests no property columns. Names are validated when the view is
            built.
        upload : bool, optional
            Publish a hosted figpack.org figure (requires a FigPack API
            key unless ``ephemeral``). Default ``False`` (save a local bundle).
        ephemeral : bool, optional
            For ``upload=True``, publish a temporary figure (no API key needed).
            Default ``False``.
        review_config : dict, optional
            Exact immutable review/profile snapshot. The browser facade owns
            this field; expert callers normally leave it ``None``.
        figpack_curation_id : optional
            Caller-supplied PK; must equal the content-addressed id if given.

        Returns
        -------
        dict
            ``{"figpack_curation_id": <uuid>}``.

        Raises
        ------
        ValueError
            If ``curation_key`` is missing ``sorting_id`` / ``curation_id``.
        DuplicateSelectionError
            If an existing row for this identity carries a non-deterministic id.
        """
        missing = [
            field
            for field in ("sorting_id", "curation_id")
            if field not in curation_key
        ]
        if missing:
            raise ValueError(
                "FigPackCurationSelection.insert_selection requires "
                f"curation_key with field(s) {missing}; got {curation_key}."
            )
        parent_key = {
            "sorting_id": curation_key["sorting_id"],
            "curation_id": curation_key["curation_id"],
        }
        _assert_figpack_curatable(parent_key)
        # ephemeral is inert offline; keep it out of the identity.
        if not upload:
            ephemeral = False

        label_options = (
            list(label_options) if label_options else default_label_options()
        )
        displayed_unit_properties = normalize_displayed_unit_properties(
            displayed_unit_properties
        )
        curation_uuid = (CurationV2 & parent_key).fetch1("curation_uuid")
        config_hash = figpack_config_hash(
            sorting_id=parent_key["sorting_id"],
            curation_id=parent_key["curation_id"],
            curation_uuid=curation_uuid,
            label_options=label_options,
            displayed_unit_properties=displayed_unit_properties,
            upload=upload,
            ephemeral=ephemeral,
            review_config=review_config,
        )
        identity = {**parent_key, "figpack_config_hash": config_hash}
        deterministic_figpack_id = deterministic_id(
            "figpack_curation", identity
        )
        assert_supplied_id_matches(
            figpack_curation_id,
            deterministic_figpack_id,
            field="figpack_curation_id",
        )

        existing = cls._find_existing_pk(identity, deterministic_figpack_id)
        if existing is not None:
            return existing

        new_row = {
            **identity,
            "figpack_curation_id": deterministic_figpack_id,
            "label_options": label_options,
            "displayed_unit_properties": pack_display_config(
                displayed_unit_properties, review_config
            ),
            "upload": bool(upload),
            "ephemeral": bool(ephemeral),
        }
        try:
            cls.insert1(new_row, allow_direct_insert=True)
        except dj.errors.DuplicateError:
            existing = cls._find_existing_pk(identity, deterministic_figpack_id)
            if existing is None:
                raise
            return existing
        return {"figpack_curation_id": deterministic_figpack_id}

    @classmethod
    def _find_existing_pk(cls, identity, deterministic_figpack_id):
        """Return the PK-only dict for ``identity`` or None; guard bad ids."""
        existing_ids = (cls & identity).fetch("figpack_curation_id")
        return existing_selection_pk(
            [uuid.UUID(str(cid)) for cid in existing_ids],
            deterministic_figpack_id,
            pk_field="figpack_curation_id",
            bypass_message=lambda bypassed: (
                "FigPackCurationSelection has duplicate selection rows for "
                f"{identity} with non-deterministic id(s) "
                f"{sorted(map(str, bypassed))} (expected the content-addressed "
                f"{deterministic_figpack_id}). This is an integrity bug -- a "
                "row was inserted bypassing insert_selection."
            ),
        )


@schema
class FigPackCuration(StagedOutputCleanupMixin, SpyglassMixin, dj.Computed):
    """A built FigPack curation view (URI) for one ``FigPackCurationSelection``.

    Stores the URI (hosted figure or local bundle) and package versions. A
    zero-unit sort raises ``ZeroUnitAnalyzerError``.

    Steps
    -----
    1. Validate the selection and resolve the display analyzer, seed
       labels and review timeline.
    2. Load the analyzer under the cache lock.
    3. Build the curation view.
    4. Publish it: upload, or stage a local bundle.
    5. Insert the URI, then move a local bundle into place.
    """

    _populate_key_lock = True

    definition = """
    -> FigPackCurationSelection
    ---
    figpack_uri: varchar(1000)
    figpack_version: varchar(32)
    figpack_spike_sorting_version: varchar(32)
    spikeinterface_version: varchar(32)
    """

    # Tri-part make keeps analyzer work and upload outside the transaction.
    # The analyzer self-heal runs in the first, untransacted make_fetch call.
    _parallel_make = True

    def make_fetch(self, key) -> FigPackCurationFetched:
        """Validate the selection and resolve every input the view needs.

        Rechecks the curation and selection identity, since
        ``allow_direct_insert`` bypasses ``insert_selection``, then resolves
        the display analyzer and review timeline inputs.
        """
        # Fail before building an analyzer if FigPack is not installed.
        import figpack  # noqa: F401
        import figpack_spike_sorting  # noqa: F401

        selection = (FigPackCurationSelection & key).fetch1()
        curation_key = {
            "sorting_id": selection["sorting_id"],
            "curation_id": selection["curation_id"],
        }
        label_options = list(selection["label_options"])
        displayed_unit_properties, review_config = unpack_display_config(
            selection["displayed_unit_properties"]
        )

        _assert_figpack_curatable(curation_key)
        _assert_selection_identity(selection, key)

        seed_labels, seed_merges = _existing_curation_state(curation_key)
        annotations = labels_and_merges_to_annotations(
            seed_labels, seed_merges, label_options=label_options
        )
        # None (the defaults) for selections without a review profile.
        display_options = (review_config or {}).get("display")
        review_table = _review_context_table(curation_key, review_config)
        analyzer_folder, timeline_inputs = _resolve_curation_view_inputs(
            curation_key, display_options
        )
        return FigPackCurationFetched(
            curation_key={
                "sorting_id": str(curation_key["sorting_id"]),
                "curation_id": int(curation_key["curation_id"]),
            },
            label_options=label_options,
            displayed_unit_properties=displayed_unit_properties,
            display_options=display_options,
            upload=bool(selection["upload"]),
            ephemeral=bool(selection["ephemeral"]),
            seed_labels=seed_labels,
            annotations=annotations,
            review_table=review_table,
            analyzer_folder=analyzer_folder,
            timeline_inputs=timeline_inputs,
        )

    def make_compute(
        self,
        key,
        curation_key,
        label_options,
        displayed_unit_properties,
        display_options,
        upload,
        ephemeral,
        seed_labels,
        annotations,
        review_table,
        analyzer_folder,
        timeline_inputs,
    ) -> FigPackCurationComputed:
        """Build the view and publish it; no DB access.

        A local bundle is staged in a private folder that ``make_insert``
        moves to the durable path.
        """
        import figpack
        import figpack_spike_sorting

        from spyglass.spikesorting.v2._storage.analyzer_cache import (
            analyzer_cache_lock,
            load_analyzer_folder,
        )

        # 2. Load the analyzer; the lock prevents reading a concurrent
        # publish mid-move.
        with analyzer_cache_lock(curation_key["sorting_id"]):
            analyzer = load_analyzer_folder(analyzer_folder)
        # 3. Build the curation view.
        view = build_curation_view(
            analyzer,
            curation_key,
            timeline=review_timeline_from_inputs(timeline_inputs),
            label_options=label_options,
            displayed_unit_properties=displayed_unit_properties,
            seed_labels=seed_labels,
            review_table=review_table,
            display_options=display_options,
        )
        title = (
            f"Spyglass curation {curation_key['sorting_id']}"
            f" / {curation_key['curation_id']}"
        )
        # 4. Publish it: upload, or stage a local bundle.
        uri, staged_bundle = _publish_view(
            view,
            upload=upload,
            ephemeral=ephemeral,
            title=title,
            figpack_curation_id=key["figpack_curation_id"],
            annotations=annotations,
        )
        return FigPackCurationComputed(
            figpack_uri=uri,
            figpack_version=figpack.__version__,
            figpack_spike_sorting_version=figpack_spike_sorting.__version__,
            spikeinterface_version=spikeinterface.__version__,
            staged_bundle=staged_bundle,
        )

    def make_insert(self, key, *computed) -> None:
        """Record the published view's URI and package versions.

        An offline bundle is moved into place only after the insert
        succeeds, so a failed populate never replaces a committed bundle.
        """
        row = FigPackCurationComputed(*computed)._asdict()
        staged_bundle = row.pop("staged_bundle")
        self.insert1({**key, **row})
        if staged_bundle is not None:
            install_staged_folder(staged_bundle, row["figpack_uri"])

    @classmethod
    def build_curation_view(
        cls,
        curation_key: dict,
        *,
        label_options: list[str] | None = None,
        displayed_unit_properties: list[str] | None = None,
        upload: bool = False,
        ephemeral: bool = False,
        review_config: dict | None = None,
    ) -> str:
        """Insert the selection, populate the view, and return its URI.

        Counterpart of v1's ``FigURLCurationSelection.generate_curation_uri``.
        """
        return cls.build_curation_view_result(
            curation_key,
            label_options=label_options,
            displayed_unit_properties=displayed_unit_properties,
            upload=upload,
            ephemeral=ephemeral,
            review_config=review_config,
        ).uri

    @classmethod
    def build_curation_view_result(
        cls,
        curation_key: dict,
        *,
        label_options: list[str] | None = None,
        displayed_unit_properties: list[str] | None = None,
        upload: bool = False,
        ephemeral: bool = False,
        review_config: dict | None = None,
    ) -> FigPackBuildResult:
        """Resolve a view and report whether its existing artifact was reused."""
        selection = FigPackCurationSelection.insert_selection(
            curation_key,
            label_options=label_options,
            displayed_unit_properties=displayed_unit_properties,
            upload=upload,
            ephemeral=ephemeral,
            review_config=review_config,
        )
        cls.drop_unavailable_view(selection)
        reused = bool(cls & selection)
        if not reused:
            report_progress("Building the review page…")
        cls.populate(selection)
        return FigPackBuildResult(
            uri=(cls & selection).fetch1("figpack_uri"),
            reused=reused,
            figpack_curation_id=selection["figpack_curation_id"],
        )

    @classmethod
    def drop_unavailable_view(cls, selection: dict) -> "str | None":
        """Forget a built view that can no longer be opened; say so.

        The row is dropped so populate rebuilds it, with a warning that draft
        labels saved only in that page are lost:

        - a local bundle whose folder is gone. A bundle outside this machine's
          :func:`figpack_cache_root` may still exist elsewhere, so its row is
          kept and only the warning is returned.
        - a hosted figure that figpack.org removed, or that expired with no
          edits.

        Parameters
        ----------
        selection : dict
            A ``FigPackCurationSelection`` primary key.

        Returns
        -------
        str or None
            The warning, when the page cannot be opened; otherwise ``None``.

        Raises
        ------
        FigPackRetrievalError
            When a hosted figure expired with edits, so they can be imported
            before it is rebuilt.
        PermissionError
            When dropping the row and the user is not on a lab team with the
            session's experimenter (the usual Spyglass delete permission).
        """
        upload = (FigPackCurationSelection & selection).fetch1("upload")
        # Populate's lock, so a concurrent rebuild cannot race the delete.
        with populate_key_lock(cls, selection):
            built = cls & selection
            if not built:
                return None
            uri = built.fetch1("figpack_uri")
            state = _hosted_figure_state(uri) if upload else None
            if state == "live" or (not upload and Path(uri).exists()):
                return None
            sorting_id, curation_id = (
                FigPackCurationSelection & selection
            ).fetch1("sorting_id", "curation_id")
            page = (
                f"The saved review page for sorting {sorting_id} curation "
                f"{curation_id}"
            )
            root = figpack_cache_root()
            if upload:
                if state == "expired" and _hosted_figure_has_edits(
                    selection, uri
                ):
                    review_id = str(selection["figpack_curation_id"])
                    raise FigPackRetrievalError(
                        f"{page} expired on figpack.org ({uri}) and has edits "
                        "that are not imported. Import them with "
                        f"FigPackReview.resume({review_id!r}), or pin the "
                        "figure on figpack.org. To discard the edits and "
                        "rebuild, run (FigPackCuration & "
                        f"{{'figpack_curation_id': {review_id!r}}}).delete()."
                    )
                gone = (
                    "expired on" if state == "expired" else "was removed from"
                )
                message = (
                    f"{page} {gone} figpack.org ({uri}). Unpinned figures "
                    "expire after 24 hours, or 6 hours if ephemeral. "
                    "Rebuilding it. Draft labels saved only in that page are "
                    "lost; committed curations are unaffected."
                )
            elif Path(uri).parent != root:
                message = (
                    f"{page} was built at {uri}, outside this machine's FigPack "
                    f"folder ({root}), and is not readable here. Open it on the "
                    "machine that built it, or set "
                    "dj.config['custom']['spikesorting_v2_figpack_dir'] to a "
                    "folder every machine shares. The page was left in place."
                )
                logger.warning(message)
                return message
            else:
                message = (
                    f"{page} is missing (expected at {uri}); it was moved or "
                    "its folder was cleaned up. Rebuilding it. Draft labels "
                    "saved only in that page are lost; committed curations are "
                    "unaffected."
                )
            logger.warning(message)
            # No table depends on this row; delete_quick raises if one does.
            built._check_delete_permission()
            built.delete_quick()
        return message

    @staticmethod
    def fetch_curation_from_uri(uri: str) -> tuple[dict, list]:
        """Read edited labels and merge groups back from a FigPack figure.

        Returns ``({unit_id: [label, ...]}, [[unit_id, ...], ...])``, the
        shape ``CurationV2.insert_curation(labels=..., merge_groups=...)``
        takes. A never-edited figure yields ``({}, [])``.
        """
        return curation_annotations_to_labels_and_merges(
            _load_annotations_json(uri)
        )

    @classmethod
    def save_curation_from_uri(
        cls,
        uri: str,
        parent_curation_key: dict,
        *,
        merge_action: ManualMergeAction = "preview",
        description: str = "curated in FigPack",
        reuse_existing: bool = False,
        allow_empty: bool = False,
        allow_unknown_unit_ids: bool = False,
        allow_custom_labels: bool = False,
        label_policy: str = "inherit",
    ) -> dict:
        """Import edited FigPack annotations as a child ``CurationV2`` row.

        ``parent_curation_key`` is the curation the view was built from, so
        several reviewers' edits can be saved as sibling children. A figure
        with no edits raises unless ``allow_empty=True``.
        """
        try:
            sorting_id = parent_curation_key["sorting_id"]
            parent_curation_id = parent_curation_key["curation_id"]
        except KeyError as exc:
            raise KeyError(
                "FigPackCuration.save_curation_from_uri requires "
                "parent_curation_key with 'sorting_id' and 'curation_id'."
            ) from exc

        _assert_figure_identity(uri, parent_curation_key)
        labels, merge_groups = cls.fetch_curation_from_uri(uri)
        if not labels and not merge_groups and not allow_empty:
            raise ValueError(
                "FigPackCuration.save_curation_from_uri found no edited "
                "labels or merge groups. Pass allow_empty=True to record an "
                "explicit no-change review."
            )
        return CurationV2.save_manual_curation(
            {"sorting_id": sorting_id},
            parent_curation_id=parent_curation_id,
            labels=labels,
            merge_groups=merge_groups,
            merge_action=merge_action,
            curation_source="figpack",
            description=description,
            reuse_existing=reuse_existing,
            allow_unknown_unit_ids=allow_unknown_unit_ids,
            allow_custom_labels=allow_custom_labels,
            label_policy=label_policy,
        )

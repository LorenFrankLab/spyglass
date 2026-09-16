"""Web-based curation views for v2 sorts, built with FigPack.

The v2 successor to v1's FigURL chain
(``spyglass.spikesorting.v1.figurl_curation``). ``FigPackCurationSelection``
records an explicit UI configuration (label palette, unit-table properties, and
upload mode) for a committed ``CurationV2`` row -- not just a bare FK -- so
repeated calls are idempotent and several UI configurations of the same curation
are representable. ``FigPackCuration`` builds the interactive view, publishes
or saves it, and stores the resulting URI; ``fetch_curation_from_uri`` reads the
edited labels and merge groups back in the exact shape
``CurationV2.insert_curation`` consumes.

The view is built by letting SpikeInterface compose the whole sorting summary
(``plot_sorting_summary(curation=False, backend="figpack")``) and attaching only
the ``SortingCuration`` control as a sibling -- SpikeInterface owns the layout,
while a profile-backed review adds one read-only metrics/suggestions table.
The analyzer is resolved for the exact committed curation generation, so merged
units render their real waveforms and correlograms. (SI's
``plot_sorting_summary(curation=True)`` is not used: released SpikeInterface
passes ``label_choices=`` while ``figpack-spike-sorting`` expects
``default_label_options=``; ``_curation_control_accepts_label_choices`` probes
for the day that upstream mismatch is fixed.)

The ``figpack`` and ``figpack_spike_sorting`` packages are an optional
dependency (the ``spikesorting-v2-curation`` extra); they are imported lazily so
this module loads without them, raising an actionable install message only when a
curation view is actually built.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import datajoint as dj

from spyglass.spikesorting.v2._figpack_curation import (
    FIGPACK_INSTALL_HINT,  # noqa: F401 -- re-exported for callers
    FIGURE_CONFIG_FILENAME,
    curation_annotations_to_labels_and_merges,
    default_label_options,
    figpack_config_hash,
    labels_and_merges_to_annotations,
    normalize_displayed_unit_properties,
    pack_display_config,
    unpack_display_config,
)
from spyglass.spikesorting.v2._review_view import (
    coerce_units_table_ids as _coerce_units_table_ids,
    require_figpack,
)
from spyglass.spikesorting.v2._selection_identity import (
    assert_supplied_id_matches,
    deterministic_id,
)
from spyglass.spikesorting.v2.curation import CurationV2
from spyglass.spikesorting.v2.exceptions import (
    DuplicateSelectionError,
    FigPackDisplayedUnitPropertyError,
    FigPackIdentityError,
    FigPackRetrievalError,
    FigPackUploadError,
    SchemaBypassError,
)
from spyglass.spikesorting.v2.sorting import Sorting
from spyglass.spikesorting.v2.utils import (
    SelectionMasterInsertGuard,
)
from spyglass.utils import SpyglassMixin, logger

schema = dj.schema("spikesorting_v2_figpack_curation")


@dataclass(frozen=True)
class FigPackBuildResult:
    """Outcome of resolving a persisted FigPack curation view."""

    uri: str
    reused: bool


# ---- figpack access + storage helpers (lazy / DB-light) ------------------


def _require_figpack():
    """Return ``(figpack.views, figpack_spike_sorting.views)`` or raise."""
    return require_figpack()


def _curation_control_accepts_label_choices() -> bool:
    """Whether ``SortingCuration`` accepts SI's ``label_choices`` kwarg.

    The attach approach exists because released SpikeInterface calls
    ``SortingCuration(label_choices=...)`` while ``figpack-spike-sorting`` 0.1.x
    defines ``default_label_options=``. When upstream aligns, this returns
    ``True`` and the one-call ``plot_sorting_summary(curation=True)`` path can
    replace the attach shim. Pure introspection; no view is built.
    """
    import inspect

    _, figpack_ss_views = _require_figpack()
    params = inspect.signature(figpack_ss_views.SortingCuration).parameters
    return "label_choices" in params


def figpack_cache_root() -> Path:
    """Return the configured root directory for saved FigPack bundles.

    ``dj.config["custom"]["spikesorting_v2_figpack_dir"]`` when truthy, else
    ``Path(temp_dir) / "spikesorting_v2" / "figpack"`` (mirrors
    ``analyzer_cache_root``).
    """
    from spyglass.settings import temp_dir

    custom = dj.config.get("custom") or {}
    configured = custom.get("spikesorting_v2_figpack_dir")
    if configured:
        return Path(configured)
    return Path(temp_dir) / "spikesorting_v2" / "figpack"


def figpack_bundle_path(figpack_curation_id) -> Path:
    """Return the durable bundle folder for one offline FigPack curation."""
    return figpack_cache_root() / f"{figpack_curation_id}"


def _load_figure_json(uri: str, filename: str, *, missing_ok: bool) -> dict:
    """Fetch one figure JSON sidecar (HTTP or local); fail closed.

    Mirrors how the FigPack frontend loads annotations: a GET on
    ``<figure>/annotations.json`` for a hosted figure, or a file read for a
    saved bundle directory. ONLY a genuine 404 / missing local file yields
    ``{}`` (a pristine, never-edited figure). An unreachable host, refused
    connection, non-404 HTTP error, or malformed JSON raises
    ``FigPackRetrievalError`` rather than silently looking like "no edits" --
    which could otherwise commit an empty child curation over a real one.
    """
    import urllib.error
    import urllib.request

    text = str(uri)
    if text.endswith("index.html"):
        text = text[: -len("index.html")]
    if text.startswith("file://"):
        text = text[len("file://") :]
    base = text.rstrip("/")
    sidecar_url = base + f"/{filename}"

    if base.startswith(("http://", "https://")):
        try:
            # Bound the fetch so a stalled host fails closed instead of hanging
            # indefinitely (a read timeout raises a bare TimeoutError, not a
            # URLError, so both are caught below).
            with urllib.request.urlopen(sidecar_url, timeout=30) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            if exc.code == 404 and missing_ok:
                return {}
            raise FigPackRetrievalError(
                f"Failed to fetch {sidecar_url}: HTTP {exc.code}."
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


def _load_annotations_json(uri: str) -> dict:
    """Load optional annotations; absence means a pristine figure."""
    return _load_figure_json(uri, "annotations.json", missing_ok=True)


def _load_figure_config(uri: str) -> dict:
    """Load the required Spyglass identity/profile figure sidecar."""
    try:
        return _load_figure_json(uri, FIGURE_CONFIG_FILENAME, missing_ok=False)
    except FigPackRetrievalError as exc:
        raise FigPackIdentityError(
            "FigPack figure has no verifiable Spyglass curation identity. "
            "Use import_legacy_figpack_curation(..., "
            "confirm_unverified_identity=True) only after independently "
            "confirming its parent."
        ) from exc


def _assert_figure_identity(
    uri: str,
    parent_curation_key: dict,
    *,
    expected_config_hash: str | None = None,
) -> dict:
    """Verify a figure against the parent's immutable curation generation."""
    required = {
        "sorting_id",
        "curation_uuid",
        "curation_id",
        "figpack_config_hash",
    }
    config = _load_figure_config(uri)
    missing = sorted(required - set(config))
    if missing:
        raise FigPackIdentityError(
            "FigPack figure identity is incomplete; missing "
            f"{missing}. Refusing an unverified import."
        )
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
    rows = (CurationV2 & parent_key).fetch(as_dict=True)
    if len(rows) != 1:
        raise FigPackIdentityError(
            f"FigPack parent no longer exists: {parent_key}."
        )
    row = rows[0]
    try:
        embedded_curation_id = int(config["curation_id"])
        embedded_uuid = uuid.UUID(str(config["curation_uuid"]))
    except (TypeError, ValueError, AttributeError) as exc:
        raise FigPackIdentityError(
            "FigPack figure identity contains a malformed curation_id or "
            "curation_uuid."
        ) from exc
    mismatches = []
    if str(config["sorting_id"]) != str(row["sorting_id"]):
        mismatches.append("sorting_id")
    if embedded_curation_id != int(row["curation_id"]):
        mismatches.append("curation_id")
    if embedded_uuid != uuid.UUID(str(row["curation_uuid"])):
        mismatches.append("curation_uuid")
    if expected_config_hash is not None and str(
        config["figpack_config_hash"]
    ) != str(expected_config_hash):
        mismatches.append("figpack_config_hash")
    if mismatches:
        raise FigPackIdentityError(
            "FigPack figure does not match the pinned parent generation; "
            f"mismatched field(s): {mismatches}. Expected "
            f"curation_uuid={row['curation_uuid']}, found "
            f"{config['curation_uuid']}. Refusing to attach annotations to a "
            "different curation."
        )
    selection_relation = FigPackCurationSelection & {
        **parent_key,
        "figpack_config_hash": str(config["figpack_config_hash"]),
    }
    if len(selection_relation) != 1:
        raise FigPackIdentityError(
            "FigPack figure config hash does not resolve to exactly one "
            "persisted selection for its pinned parent."
        )
    selection = selection_relation.fetch1()
    selection_key = {"figpack_curation_id": selection["figpack_curation_id"]}
    try:
        _assert_selection_identity(selection, selection_key)
    except SchemaBypassError as exc:
        raise FigPackIdentityError(
            "FigPack figure references a selection whose content-addressed "
            "identity is invalid."
        ) from exc
    return config


def _available_displayed_unit_properties(analyzer) -> list[str]:
    """Return unit-table columns SpikeInterface can render for ``analyzer``."""
    from spikeinterface.widgets.utils import make_units_table_from_analyzer

    table = make_units_table_from_analyzer(analyzer)
    return [str(column) for column in table.columns]


def _assert_displayed_unit_properties_available(
    analyzer, displayed_unit_properties
) -> None:
    """Fail closed when requested FigPack unit-table columns are unavailable."""
    requested = normalize_displayed_unit_properties(displayed_unit_properties)
    if requested is None:
        return
    available = _available_displayed_unit_properties(analyzer)
    missing = [prop for prop in requested if prop not in available]
    if missing:
        raise FigPackDisplayedUnitPropertyError(
            "FigPackCuration cannot display requested unit properties "
            f"{missing}; available properties on this sort's display analyzer "
            f"are {available}. `displayed_unit_properties=None` keeps "
            "SpikeInterface's default display behavior; otherwise compute the "
            "needed analyzer metric/template/property columns before building "
            "the FigPack view."
        )


@contextmanager
def _seeded_numpy_random(seed: int):
    """Pin numpy's global RNG for SI's display subsampling, then restore it.

    ``AmplitudesWidget(max_spikes_per_unit=...)`` subsamples with the
    unseeded global ``np.random.choice``; seeding it here (and restoring the
    prior state) makes the displayed sample deterministic without touching
    any scientific computation.
    """
    import numpy as np

    state = np.random.get_state()
    np.random.seed(int(seed))
    try:
        yield
    finally:
        np.random.set_state(state)


def _build_curation_view(
    curation_key: dict,
    *,
    label_options,
    displayed_unit_properties,
    seed_labels=None,
    review_table=None,
    display_options=None,
):
    """Build the FigPack curation view for a curation (minimal-attach).

    Lets SpikeInterface compose the whole sorting summary over the sort's
    display analyzer (ensuring the curation-view extensions and any explicitly
    requested unit-table columns are available), then attaches only the
    ``SortingCuration`` control as a sibling. ``display_options``
    (:class:`ReviewDisplayOptions`) bounds the bundle payload -- the per-unit
    amplitude sample and the correlogram pair filter -- and is display-only.

    A profile-backed review passes ``review_table`` (the selected
    evaluation's metrics, annotation columns and proposals, indexed by unit
    id). Its columns are shown in SpikeInterface's selectable unit table via
    ``extra_unit_properties`` -- the table that drives unit selection for
    the curation control -- in the requested order and INSTEAD of SI's
    default columns, so the official evaluation stays authoritative even
    when the display analyzer carries a same-named property. Returns the
    composed ``figpack.views`` object.
    """
    import spikeinterface.widgets as sw

    from spyglass.spikesorting.v2 import _visualization as _viz
    from spyglass.spikesorting.v2._curation_analyzer import (
        curation_analyzer_with_extensions,
    )
    from spyglass.spikesorting.v2._review_profile import ReviewDisplayOptions
    from spyglass.spikesorting.v2._review_unit_properties import (
        review_unit_properties,
    )
    from spyglass.spikesorting.v2._review_view import (
        compose_review_layout,
        curation_control,
    )

    _require_figpack()
    display = ReviewDisplayOptions.from_mapping(display_options)

    sorting_key = {"sorting_id": curation_key["sorting_id"]}
    waveform_recipe = (Sorting & sorting_key).fetch1(
        "display_waveform_params_name"
    )
    required = _viz.DISPLAY_WIDGET_EXTENSIONS["plot_sorting_summary"]
    with curation_analyzer_with_extensions(
        curation_key,
        waveform_recipe,
        "display",
        extra_extensions={name: {} for name in required},
    ) as analyzer:
        if review_table is not None:
            # Profile-backed: exactly the review columns, no SI defaults.
            analyzer_properties: list[str] | None = []
            extra_properties = review_unit_properties(
                review_table, analyzer.unit_ids
            )
        else:
            # Expert path: analyzer-native SI unit properties (or SI's
            # defaults when None).
            analyzer_properties = displayed_unit_properties
            extra_properties = None
        _assert_displayed_unit_properties_available(
            analyzer, analyzer_properties
        )
        with _seeded_numpy_random(display.amplitude_sampling_seed):
            summary = sw.plot_sorting_summary(
                analyzer,
                backend="figpack",
                curation=False,
                displayed_unit_properties=analyzer_properties,
                extra_unit_properties=extra_properties,
                max_amplitudes_per_unit=display.max_amplitudes_per_unit,
                min_similarity_for_correlograms=(
                    display.min_similarity_for_correlograms
                ),
                generate_url=False,
                display=False,
            ).view

    control = curation_control(label_options, seed_labels)
    summary_title = "Sorting summary"
    context = (
        f"Sorting `{curation_key['sorting_id']}`, curation "
        f"`{curation_key['curation_id']}`. {display.describe()}. "
        "Omitted pairs are filtered from the display, not evidence of no correlation."
    )
    if review_table is not None:
        context += " " + review_table.attrs.get("qc_context", "")
    view = compose_review_layout(
        summary, control, summary_title=summary_title, context=context
    )
    _coerce_units_table_ids(view)
    return view


def _assert_figpack_curatable(curation_key: dict) -> None:
    """Assert a curation is committed before building its exact analyzer.

    Enforced at BOTH ``insert_selection`` (early, friendly) and ``make`` (the
    integrity boundary): ``SelectionMasterInsertGuard`` has an
    ``allow_direct_insert`` escape hatch, so a row bypassing ``insert_selection``
    must still be re-validated before the view is built -- otherwise a preview
    could be rendered as though it had a final unit namespace. Mirrors
    ``CurationEvaluation.make_fetch`` re-asserting its preview guard.
    """
    CurationV2.assert_committed_curation(
        curation_key, context="FigPackCuration"
    )


def _assert_selection_identity(selection: dict, key: dict) -> None:
    """Recheck the content-addressed identity at consume time (bypass guard).

    ``figpack_config_hash`` and ``figpack_curation_id`` are derived from the
    selection's own fields, but ``allow_direct_insert`` can store a row whose PK
    / hash disagree with its label options, displayed properties, and upload
    mode. Recompute both from the row and raise ``SchemaBypassError`` on
    mismatch, mirroring the consume-time hash recheck the other v2
    content-addressed selections do.
    """
    # insert_selection normalizes ephemeral to False when offline, so an
    # offline + ephemeral row can only come from a raw-insert bypass.
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

    from spyglass.spikesorting.v2.curation_api import (
        CurationRef,
        EvaluationResult,
    )
    from spyglass.spikesorting.v2.unit_annotation import (
        AnnotationSetRef,
        read_unit_properties,
    )

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
    annotation_sets = tuple(
        AnnotationSetRef.from_snapshot(snapshot)
        for snapshot in review_config.get("annotation_sets", [])
    )
    properties = read_unit_properties(
        CurationRef.from_key(curation_key),
        evaluation=evaluation,
        annotation_sets=annotation_sets,
    )
    requested = [
        *review_config["displayed_unit_properties"],
        *(ref.column_name for ref in annotation_sets),
    ]
    missing = [name for name in requested if name not in properties.columns]
    if missing:
        raise FigPackDisplayedUnitPropertyError(
            "Review requests properties absent from its selected evaluation "
            f"and annotation sets: {missing}. Available properties: "
            f"{list(map(str, properties.columns))}."
        )
    metrics = properties.reindex(unit_ids)[requested]

    proposed_labels = evaluation.proposed_labels
    suggested_groups = evaluation.suggested_merges
    raw_provenance: dict[int, list[int]] = {}
    for row in (CurationV2.MergeGroup & curation_key).fetch(
        "unit_id",
        "contributor_unit_id",
        as_dict=True,
        order_by=("unit_id", "contributor_unit_id"),
    ):
        raw_provenance.setdefault(int(row["unit_id"]), []).append(
            int(row["contributor_unit_id"])
        )

    def groups_for(unit_id: int) -> str:
        groups = [group for group in suggested_groups if unit_id in group]
        return "; ".join(",".join(map(str, group)) for group in groups)

    # Actionable columns first (what the rule set proposes and what an
    # earlier merge did), then the profile's metric / annotation columns in
    # their requested order: on a laptop-width table the first columns are
    # the ones a curator acts on.
    import pandas as pd

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
    from spyglass.spikesorting.v2.metric_curation import QualityMetricParameters

    metric_kwargs = (
        QualityMetricParameters
        & {"metric_params_name": evaluation.spec.metric_params_name}
    ).fetch1("metric_kwargs")
    from spikeinterface.metrics.quality import (
        get_default_quality_metrics_params,
    )

    isi_params = get_default_quality_metrics_params()["isi_violation"]
    isi_params.update((metric_kwargs or {}).get("isi_violation") or {})
    refractory_ms = isi_params["isi_threshold_ms"]
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


def _write_figure_sidecars(
    bundle: Path, annotations: dict, figure_config: dict
) -> None:
    """Write seeded annotations and immutable Spyglass figure identity."""
    (bundle / "annotations.json").write_text(
        json.dumps(annotations, indent=2, sort_keys=True)
    )
    (bundle / FIGURE_CONFIG_FILENAME).write_text(
        json.dumps(figure_config, indent=2, sort_keys=True)
    )


def _publish_view(
    view,
    *,
    upload: bool,
    ephemeral: bool,
    title: str,
    figpack_curation_id,
    annotations: dict,
    figure_config: dict,
) -> str:
    """Publish a built view and return its URI (cloud URL or saved bundle path).

    ``upload=True`` publishes to figpack.org and requires ``FIGPACK_API_KEY``
    (unless ``ephemeral``); ``upload=False`` saves a durable static bundle and
    returns its folder path.
    """
    if upload:
        api_key = os.environ.get("FIGPACK_API_KEY")
        if not ephemeral and not api_key:
            raise FigPackUploadError(
                "FigPack upload=True requires the FIGPACK_API_KEY environment "
                "variable (or ephemeral=True for a temporary figure). Set it "
                "to publish to figpack.org, or use upload=False to save a local "
                "bundle."
            )
        # Build the exact bundle first. FigPack's recursive uploader includes
        # both sidecars while consolidated-only mode omits redundant per-array
        # zarr metadata, giving hosted and local reviews identical seeded state
        # and identity semantics without multiplying the hosted file count.
        from figpack.core._upload_bundle import _upload_bundle

        with tempfile.TemporaryDirectory(prefix="spyglass-figpack-") as tmp:
            view.save(tmp, title=title)
            _write_figure_sidecars(Path(tmp), annotations, figure_config)
            return _upload_bundle(
                tmp,
                api_key=api_key,
                title=title,
                ephemeral=ephemeral,
                use_consolidated_metadata_only=True,
            )

    bundle = figpack_bundle_path(figpack_curation_id)
    bundle.parent.mkdir(parents=True, exist_ok=True)
    if bundle.exists():
        shutil.rmtree(bundle)
    view.save(str(bundle), title=title)
    _write_figure_sidecars(bundle, annotations, figure_config)
    return str(bundle)


# ---- tables --------------------------------------------------------------


@schema
class FigPackCurationSelection(
    SelectionMasterInsertGuard, SpyglassMixin, dj.Manual
):
    """A committed ``CurationV2`` row paired with a FigPack UI configuration.

    The ``figpack_curation_id`` PK is content-addressed over the curation plus
    the UI config (label palette, unit-table properties, and upload/ephemeral
    mode), so the same configuration always maps to one row and distinct
    configurations of one curation coexist. A raw ``insert`` / ``insert1`` is
    blocked; use ``insert_selection``.
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
            built so explicit requests cannot be silently dropped.
        upload : bool, optional
            Publish a hosted figpack.org figure (requires ``FIGPACK_API_KEY``
            unless ``ephemeral``). Default ``False`` (save a local bundle).
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
        # ``ephemeral`` only affects a hosted upload; offline it is inert, so
        # normalize it to False so it cannot fork the content-addressed identity.
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
        bypassed = [
            cid
            for cid in existing_ids
            if uuid.UUID(str(cid)) != deterministic_figpack_id
        ]
        if bypassed:
            raise DuplicateSelectionError(
                "FigPackCurationSelection has duplicate selection rows for "
                f"{identity} with non-deterministic id(s) "
                f"{sorted(map(str, bypassed))} (expected the content-addressed "
                f"{deterministic_figpack_id}). This is an integrity bug -- a "
                "row was inserted bypassing insert_selection."
            )
        if len(existing_ids):
            return {"figpack_curation_id": deterministic_figpack_id}
        return None


@schema
class FigPackCuration(SpyglassMixin, dj.Computed):
    """A built FigPack curation view (URI) for one ``FigPackCurationSelection``.

    ``make`` builds the view, publishes it (hosted figpack.org figure when
    ``upload``, else a durable local bundle), and stores the URI plus the
    package versions used. A zero-unit sort raises ``ZeroUnitAnalyzerError``
    (from ``Sorting.get_analyzer``): there is no analyzer to summarize.
    """

    definition = """
    -> FigPackCurationSelection
    ---
    figpack_uri: varchar(1000)
    figpack_version: varchar(32)
    figpack_spike_sorting_version: varchar(32)
    spikeinterface_version: varchar(32)
    """

    def make(self, key):
        """Build, publish, and record one FigPack curation view.

        Monolithic ``make`` BY DESIGN -- this is a deliberate exception to the
        v2 tri-part (``make_fetch`` / ``make_compute`` / ``make_insert``)
        convention, recorded in
        ``test_integrity.test_tripart_dispatch_active_on_all_v2_computed_tables``.
        Tri-part exists to keep heavy work OUTSIDE the DataJoint transaction so
        it can roll back cleanly, but ``_publish_view`` performs a NETWORK upload
        when ``upload=True`` that cannot be transactionally rolled back, so
        splitting the work buys nothing here. Keeping it monolithic is the honest
        shape; do not "fix" it into tri-part to match the other tables.
        """
        import figpack
        import figpack_spike_sorting
        import spikeinterface

        selection = (FigPackCurationSelection & key).fetch1()
        curation_key = {
            "sorting_id": selection["sorting_id"],
            "curation_id": selection["curation_id"],
        }
        label_options = list(selection["label_options"])
        displayed_unit_properties, review_config = unpack_display_config(
            selection["displayed_unit_properties"]
        )
        upload = bool(selection["upload"])

        # Re-validate at the integrity boundary: insert_selection's guard is
        # bypassable (allow_direct_insert), so re-check everything it enforced
        # before any view is built -- the curation namespace and the content-
        # addressed identity.
        _assert_figpack_curatable(curation_key)
        _assert_selection_identity(selection, key)

        seed_labels, seed_merges = _existing_curation_state(curation_key)
        annotations = labels_and_merges_to_annotations(
            seed_labels, seed_merges, label_options=label_options
        )
        curation_uuid = (CurationV2 & curation_key).fetch1("curation_uuid")
        figure_config = {
            "sorting_id": str(curation_key["sorting_id"]),
            "curation_uuid": str(curation_uuid),
            "curation_id": int(curation_key["curation_id"]),
            "figpack_config_hash": str(selection["figpack_config_hash"]),
        }
        if review_config is not None:
            figure_config["review"] = review_config

        view = _build_curation_view(
            curation_key,
            label_options=label_options,
            displayed_unit_properties=displayed_unit_properties,
            seed_labels=seed_labels,
            review_table=_review_context_table(curation_key, review_config),
            # Profile-backed reviews persist their display budget in the
            # review configuration (part of the selection identity); an expert
            # selection uses the defaults.
            display_options=(review_config or {}).get("display"),
        )
        title = (
            f"Spyglass curation {curation_key['sorting_id']}"
            f" / {curation_key['curation_id']}"
        )
        uri = _publish_view(
            view,
            upload=upload,
            ephemeral=bool(selection["ephemeral"]),
            title=title,
            figpack_curation_id=key["figpack_curation_id"],
            annotations=annotations,
            figure_config=figure_config,
        )

        self.insert1(
            {
                **key,
                "figpack_uri": uri,
                "figpack_version": figpack.__version__,
                "figpack_spike_sorting_version": (
                    figpack_spike_sorting.__version__
                ),
                "spikeinterface_version": spikeinterface.__version__,
            }
        )

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

        The v2 analog of v1's ``FigURLCurationSelection.generate_curation_uri``:
        a one-call convenience that creates/inserts the
        ``FigPackCurationSelection`` row, populates ``FigPackCuration``, and
        returns the stored ``figpack_uri``.
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
        # An offline bundle lives under a temp dir that can be purged; if the
        # row exists but its local bundle is gone, drop the stale row so populate
        # rebuilds it rather than returning a dead path (mirrors run_v2_pipeline's
        # figpack reuse guard). A hosted (upload=True) URI is remote, so the
        # on-disk check does not apply.
        built = cls & selection
        reused = bool(built)
        if (
            not upload
            and built
            and not Path(built.fetch1("figpack_uri")).exists()
        ):
            built.delete(safemode=False)
            reused = False
        cls.populate(selection)
        return FigPackBuildResult(
            uri=(cls & selection).fetch1("figpack_uri"),
            reused=reused,
        )

    @staticmethod
    def fetch_curation_from_uri(uri: str) -> tuple[dict, list]:
        """Read edited labels and merge groups back from a FigPack figure.

        Fetches ``<uri>/annotations.json`` (hosted figure or saved bundle) and
        returns ``({unit_id: [label, ...]}, [[unit_id, ...], ...])`` -- the
        exact shape ``CurationV2.insert_curation(labels=..., merge_groups=...)``
        consumes. A never-edited figure yields ``({}, [])``.
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
        merge_action: str = "preview",
        description: str = "curated in FigPack",
        reuse_existing: bool = False,
        allow_empty: bool = False,
        allow_unknown_unit_ids: bool = False,
        allow_custom_labels: bool = False,
        label_policy: str = "inherit",
    ) -> dict:
        """Import edited FigPack annotations as a child ``CurationV2`` row.

        ``parent_curation_key`` is the curation the browser view was built from
        (``{"sorting_id": ..., "curation_id": ...}``). Requiring that parent
        key keeps the import path out of the root-curation default, so a team can
        safely save several reviewers' edits as sibling child curations and
        choose which one to commit or merge.

        A figure with no edited labels or merge groups raises by default instead
        of creating an empty child curation; pass ``allow_empty=True`` to record
        an explicit "reviewed, no changes" child.
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

    @classmethod
    def import_legacy_figpack_curation(
        cls,
        uri: str,
        *,
        asserted_parent: dict,
        confirm_unverified_identity: bool = False,
        **save_kwargs,
    ) -> dict:
        """Import an identity-less legacy figure behind an explicit escape.

        This operation cannot prove which curation produced the figure. It is
        intentionally separate from :meth:`save_curation_from_uri`; callers
        must independently verify the asserted parent and opt in by name.
        """
        if confirm_unverified_identity is not True:
            raise FigPackIdentityError(
                "Legacy FigPack import cannot verify its parent. Set "
                "confirm_unverified_identity=True only after independently "
                "confirming asserted_parent."
            )
        try:
            sorting_id = asserted_parent["sorting_id"]
            parent_curation_id = int(asserted_parent["curation_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "asserted_parent must contain sorting_id and curation_id."
            ) from exc
        labels, merge_groups = cls.fetch_curation_from_uri(uri)
        allow_empty = bool(save_kwargs.pop("allow_empty", False))
        if not labels and not merge_groups and not allow_empty:
            raise ValueError(
                "Legacy FigPack figure contains no edits. Pass "
                "allow_empty=True to record an explicit no-change review."
            )
        return CurationV2.save_manual_curation(
            {"sorting_id": sorting_id},
            parent_curation_id=parent_curation_id,
            labels=labels,
            merge_groups=merge_groups,
            merge_action=save_kwargs.pop("merge_action", "preview"),
            curation_source="figpack",
            description=save_kwargs.pop(
                "description", "legacy curation imported from FigPack"
            ),
            **save_kwargs,
        )


def import_legacy_figpack_curation(
    uri: str,
    *,
    asserted_parent: dict,
    confirm_unverified_identity: bool = False,
    **save_kwargs,
) -> dict:
    """Module-level explicit escape hatch for identity-less figures."""
    return FigPackCuration.import_legacy_figpack_curation(
        uri,
        asserted_parent=asserted_parent,
        confirm_unverified_identity=confirm_unverified_identity,
        **save_kwargs,
    )

"""Web-based curation views for v2 sorts, built with FigPack.

The v2 successor to v1's FigURL chain
(``spyglass.spikesorting.v1.figurl_curation``). ``FigPackCurationSelection``
records an explicit UI configuration (label palette, metric columns, upload
mode) for a committed ``CurationV2`` row -- not just a bare FK -- so repeated
calls are idempotent and several UI configurations of the same curation are
representable. ``FigPackCuration`` builds the interactive view, publishes or
saves it, and stores the resulting URI; ``fetch_curation_from_uri`` reads the
edited labels and merge groups back in the exact shape
``CurationV2.insert_curation`` consumes.

The view is built by letting SpikeInterface compose the whole sorting summary
(``plot_sorting_summary(curation=False, backend="figpack")``) and attaching only
the ``SortingCuration`` control as a sibling -- SpikeInterface owns the layout,
and the hand-written surface is one widget plus a wrapper. (SI's
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
import uuid
from pathlib import Path

import datajoint as dj

from spyglass.spikesorting.v2._figpack_curation import (
    FIGPACK_INSTALL_HINT,
    curation_annotations_to_labels_and_merges,
    default_label_options,
    figpack_config_hash,
    labels_and_merges_to_annotations,
)
from spyglass.spikesorting.v2._selection_identity import (
    assert_supplied_id_matches,
    deterministic_id,
)
from spyglass.spikesorting.v2.curation import CurationV2
from spyglass.spikesorting.v2.exceptions import (
    DuplicateSelectionError,
    FigPackCurationNamespaceError,
    FigPackRetrievalError,
    FigPackUploadError,
    SchemaBypassError,
)
from spyglass.spikesorting.v2.sorting import Sorting
from spyglass.spikesorting.v2.utils import (
    SelectionMasterInsertGuard,
    _is_duplicate_key_error,
)
from spyglass.utils import SpyglassMixin, logger

schema = dj.schema("spikesorting_v2_figpack_curation")


# ---- figpack access + storage helpers (lazy / DB-light) ------------------


def _require_figpack():
    """Return ``(figpack.views, figpack_spike_sorting.views)`` or raise.

    Raises ``ImportError`` with the install hint if either optional package is
    missing, so the failure is actionable rather than a bare ``ModuleNotFound``.
    """
    try:
        import figpack.views as figpack_views
        import figpack_spike_sorting.views as figpack_ss_views
    except ImportError as exc:  # pragma: no cover - exercised via gated tests
        raise ImportError(FIGPACK_INSTALL_HINT) from exc
    return figpack_views, figpack_ss_views


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


def _load_annotations_json(uri: str) -> dict:
    """Fetch a figure's ``annotations.json`` (HTTP or local path); fail closed.

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
    annotations_url = base + "/annotations.json"

    if base.startswith(("http://", "https://")):
        try:
            # Bound the fetch so a stalled host fails closed instead of hanging
            # indefinitely (a read timeout raises a bare TimeoutError, not a
            # URLError, so both are caught below).
            with urllib.request.urlopen(
                annotations_url, timeout=30
            ) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return {}  # no annotations file yet == pristine figure
            raise FigPackRetrievalError(
                f"Failed to fetch {annotations_url}: HTTP {exc.code}."
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            reason = getattr(exc, "reason", exc)
            raise FigPackRetrievalError(
                f"Could not reach {annotations_url}: {reason}. Refusing to "
                "treat an unreachable figure as having no edits."
            ) from exc
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise FigPackRetrievalError(
                f"Malformed annotations at {annotations_url}: {exc}."
            ) from exc

    figure_dir = Path(base)
    if not figure_dir.exists():
        raise FigPackRetrievalError(
            f"FigPack figure path does not exist: {figure_dir}. Refusing to "
            "treat a missing/typoed figure as having no edits."
        )
    path = figure_dir / "annotations.json"
    if not path.exists():
        return {}  # existing figure dir, never edited == pristine
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise FigPackRetrievalError(
            f"Malformed annotations at {path}: {exc}."
        ) from exc


def _coerce_units_table_ids(view) -> None:
    """Coerce ``UnitsTable`` unit ids to Python ``int`` in place (bug workaround).

    ``figpack_spike_sorting``'s ``UnitsTable.write_to_zarr_group`` ``json.dumps``
    its rows directly, and SpikeInterface's ``generate_unit_table_view`` builds
    each ``UnitsTableRow`` from ``sorting.unit_ids`` WITHOUT coercion -- so a real
    v2 analyzer (integer unit ids stored as ``numpy.int32``) raises
    ``TypeError: Object of type int32 is not JSON serializable`` at save/upload.
    Walk the composed view and coerce every ``UnitsTableRow.unit_id`` and
    ``UnitSimilarityScore`` id to ``int``. Remove once upstream serializes these
    (e.g. via ``check_json``).
    """
    import figpack_spike_sorting.views as figpack_ss_views

    seen: set[int] = set()

    def walk(obj):
        if obj is None or id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, figpack_ss_views.UnitsTable):
            for row in obj.rows:
                row.unit_id = int(row.unit_id)
            for score in obj.similarity_scores or []:
                score.unit_id1 = int(score.unit_id1)
                score.unit_id2 = int(score.unit_id2)
        for attr in ("item1", "item2", "view"):
            walk(getattr(obj, attr, None))
        for child in getattr(obj, "items", None) or []:
            walk(child)

    walk(view)


def _build_curation_view(curation_key: dict, *, label_options, metrics):
    """Build the FigPack curation view for a curation (minimal-attach).

    Lets SpikeInterface compose the whole sorting summary over the sort's
    display analyzer (ensuring the curation-view extensions), then attaches only
    the ``SortingCuration`` control as a sibling. Returns the composed
    ``figpack.views`` object.
    """
    import spikeinterface.widgets as sw  # noqa: F401 - imported by visualization

    from spyglass.spikesorting.v2 import visualization

    figpack_views, figpack_ss_views = _require_figpack()

    summary_kwargs = {}
    if metrics:
        summary_kwargs["displayed_unit_properties"] = list(metrics)

    sorting_key = {"sorting_id": curation_key["sorting_id"]}
    summary = visualization.plot_sorting_summary(
        sorting_key,
        compute_missing=True,
        backend="figpack",
        curation=False,
        generate_url=False,
        display=False,
        **summary_kwargs,
    ).view

    control = figpack_ss_views.SortingCuration(
        default_label_options=list(label_options)
    )
    view = figpack_views.Box(
        direction="vertical",
        items=[
            figpack_views.LayoutItem(
                view=summary, title="Sorting summary", stretch=1
            ),
            figpack_views.LayoutItem(
                view=control, title="Curation", max_size=260
            ),
        ],
    )
    _coerce_units_table_ids(view)
    return view


def _curation_matches_raw_namespace(curation_key: dict) -> bool:
    """Whether a curation's unit set equals the raw sort's unit set.

    The FigPack view is built over the raw-sort display analyzer, so it is only
    correct when the curation lives in the raw ``Sorting.Unit`` namespace. A
    merged curation (or a label-only child of a merged parent) has a different
    unit set, so this returns ``False`` and the view is rejected upstream. Label-
    only curations of a non-merged sort keep the raw unit set, so they pass.
    """
    raw = {
        int(unit_id)
        for unit_id in (
            Sorting.Unit & {"sorting_id": curation_key["sorting_id"]}
        ).fetch("unit_id")
    }
    curated = {
        int(unit_id)
        for unit_id in (CurationV2.Unit & curation_key).fetch("unit_id")
    }
    return curated == raw


def _assert_figpack_curatable(curation_key: dict) -> None:
    """Assert a curation is committed and in the raw-sort unit namespace.

    Enforced at BOTH ``insert_selection`` (early, friendly) and ``make`` (the
    integrity boundary): ``SelectionMasterInsertGuard`` has an
    ``allow_direct_insert`` escape hatch, so a row bypassing ``insert_selection``
    must still be re-validated before the view is built -- otherwise a preview or
    merged-namespace curation could render the raw-sort namespace. Mirrors
    ``CurationEvaluation.make_fetch`` re-asserting its preview guard.
    """
    CurationV2.assert_committed_curation(
        curation_key, context="FigPackCuration"
    )
    if not _curation_matches_raw_namespace(curation_key):
        raise FigPackCurationNamespaceError(
            "FigPack curation of a merged curation (or a label-only child of a "
            "merged curation) is not supported: the view renders the raw "
            "sort's unit namespace, not this curation's units. "
            f"curation_key={curation_key}. Curate the root curation instead."
        )


def _reject_unsupported_metrics(metrics) -> None:
    """Reject a non-empty metric-column request (unsupported).

    The builder ensures only the four summary extensions, not
    ``quality_metrics`` / ``template_metrics``, so a requested metric absent from
    the display analyzer would be silently warned-and-dropped by SpikeInterface
    -- producing a populated row whose unit table does not show it. Refuse rather
    than mislead: the view shows only the four summary extensions.
    """
    if metrics:
        raise ValueError(
            "FigPackCuration does not support metric-column selection: a "
            f"requested metric in {list(metrics)} absent from the display "
            "analyzer would be silently dropped. Omit `metrics`."
        )


def _assert_selection_identity(selection: dict, key: dict) -> None:
    """Recheck the content-addressed identity at consume time (bypass guard).

    ``figpack_config_hash`` and ``figpack_curation_id`` are derived from the
    selection's own fields, but ``allow_direct_insert`` can store a row whose PK
    / hash disagree with its label options, metrics, and upload mode. Recompute
    both from the row and raise ``SchemaBypassError`` on mismatch, mirroring the
    consume-time hash recheck the other v2 content-addressed selections do.
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
    expected_hash = figpack_config_hash(
        sorting_id=selection["sorting_id"],
        curation_id=selection["curation_id"],
        label_options=list(selection["label_options"]),
        metrics=list(selection["metrics"]),
        upload=bool(selection["upload"]),
        ephemeral=bool(selection["ephemeral"]),
    )
    if selection["figpack_config_hash"] != expected_hash:
        raise SchemaBypassError(
            f"FigPackCurationSelection {key['figpack_curation_id']} has a "
            "figpack_config_hash that does not match its own label_options / "
            "metrics / upload / ephemeral (a raw-insert bypass). Drop the row "
            "and re-insert via insert_selection()."
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
    """Return ``(labels, merge_groups)`` a curation already carries.

    Used both to seed an offline view and to detect (and refuse) a hosted upload
    of a curation with pre-existing state before cloud seeding is verified. Reads
    the curation's own namespace; for the raw-namespace curations FigPack
    accepts, ``merge_groups`` is empty and only labels can be present.
    """
    labels = CurationV2._labels_by_unit(curation_key)
    merge_groups = [
        sorted({kept, *contributors})
        for kept, contributors in CurationV2.get_merge_groups(
            curation_key
        ).items()
        if len({kept, *contributors}) > 1
    ]
    return labels, merge_groups


def _write_seed_annotations(
    bundle: Path, labels: dict, merge_groups: list, label_options
) -> None:
    """Write ``annotations.json`` so a saved bundle opens pre-curated."""
    payload = labels_and_merges_to_annotations(
        labels, merge_groups, label_options=label_options
    )
    (bundle / "annotations.json").write_text(json.dumps(payload, indent=2))


def _publish_view(
    view, *, upload: bool, ephemeral: bool, title: str, figpack_curation_id
) -> str:
    """Publish a built view and return its URI (cloud URL or saved bundle path).

    ``upload=True`` publishes to figpack.org and requires ``FIGPACK_API_KEY``
    (unless ``ephemeral``); ``upload=False`` saves a durable static bundle and
    returns its folder path.
    """
    if upload:
        if not ephemeral and not os.environ.get("FIGPACK_API_KEY"):
            raise FigPackUploadError(
                "FigPack upload=True requires the FIGPACK_API_KEY environment "
                "variable (or ephemeral=True for a temporary figure). Set it "
                "to publish to figpack.org, or use upload=False to save a local "
                "bundle."
            )
        return view.show(
            upload=True,
            ephemeral=ephemeral,
            open_in_browser=False,
            wait_for_input=False,
            inline=False,
            title=title,
        )

    bundle = figpack_bundle_path(figpack_curation_id)
    bundle.parent.mkdir(parents=True, exist_ok=True)
    if bundle.exists():
        shutil.rmtree(bundle)
    view.save(str(bundle), title=title)
    return str(bundle)


# ---- tables --------------------------------------------------------------


@schema
class FigPackCurationSelection(
    SelectionMasterInsertGuard, SpyglassMixin, dj.Manual
):
    """A committed ``CurationV2`` row paired with a FigPack UI configuration.

    The ``figpack_curation_id`` PK is content-addressed over the curation plus
    the UI config (label palette, metric columns, upload/ephemeral mode), so the
    same configuration always maps to one row and distinct configurations of one
    curation coexist. A raw ``insert`` / ``insert1`` is blocked; use
    ``insert_selection``.
    """

    definition = """
    figpack_curation_id: uuid
    ---
    -> CurationV2
    figpack_config_hash: char(64)  # sha256 over label_options, metrics, upload, ephemeral
    label_options: blob            # curation label palette, in display order
    metrics: blob                  # metric column names to display
    upload: bool                   # True publishes a hosted figpack.org URI
    ephemeral: bool                # temporary hosted figure (no API key needed)
    """

    @classmethod
    def insert_selection(
        cls,
        curation_key: dict,
        *,
        label_options: list[str] | None = None,
        metrics: list[str] | None = None,
        upload: bool = False,
        ephemeral: bool = False,
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
        metrics : list of str, optional
            Metric columns to display. Not supported -- a non-empty value
            raises ``ValueError`` (a requested metric absent from the display
            analyzer would be silently dropped). Defaults to ``[]``.
        upload : bool, optional
            Publish a hosted figpack.org figure (requires ``FIGPACK_API_KEY``
            unless ``ephemeral``). Default ``False`` (save a local bundle).
        ephemeral : bool, optional
            For ``upload=True``, publish a temporary figure (no API key needed).
            Default ``False``.
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
        metrics = list(metrics) if metrics else []
        _reject_unsupported_metrics(metrics)
        config_hash = figpack_config_hash(
            sorting_id=parent_key["sorting_id"],
            curation_id=parent_key["curation_id"],
            label_options=label_options,
            metrics=metrics,
            upload=upload,
            ephemeral=ephemeral,
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
            "metrics": metrics,
            "upload": bool(upload),
            "ephemeral": bool(ephemeral),
        }
        try:
            cls.insert1(new_row, allow_direct_insert=True)
        except Exception as exc:  # noqa: BLE001 - re-raised unless dup-PK race
            if not _is_duplicate_key_error(exc):
                raise
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
    figpack_uri: varchar(512)
    figpack_version: varchar(32)
    figpack_spike_sorting_version: varchar(32)
    spikeinterface_version: varchar(32)
    """

    def make(self, key):
        """Build, publish, and record one FigPack curation view."""
        import figpack
        import figpack_spike_sorting
        import spikeinterface

        selection = (FigPackCurationSelection & key).fetch1()
        curation_key = {
            "sorting_id": selection["sorting_id"],
            "curation_id": selection["curation_id"],
        }
        label_options = list(selection["label_options"])
        upload = bool(selection["upload"])

        # Re-validate at the integrity boundary: insert_selection's guard is
        # bypassable (allow_direct_insert), so re-check everything it enforced
        # before any view is built -- the curation namespace, the content-
        # addressed identity, and the unsupported metric-column request.
        _assert_figpack_curatable(curation_key)
        _assert_selection_identity(selection, key)
        _reject_unsupported_metrics(list(selection["metrics"]))

        seed_labels, seed_merges = _existing_curation_state(curation_key)
        if upload and (seed_labels or seed_merges):
            raise FigPackUploadError(
                "Hosted upload (upload=True) of a curation that already carries "
                "labels/merges is not supported: the initial curation state is "
                "not written into the hosted figure's annotations.json. Use "
                "upload=False (the seeded local bundle), or open a fresh root "
                "curation."
            )

        view = _build_curation_view(
            curation_key,
            label_options=label_options,
            metrics=list(selection["metrics"]),
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
        )
        if not upload:
            _write_seed_annotations(
                Path(uri), seed_labels, seed_merges, label_options
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
        metrics: list[str] | None = None,
        upload: bool = False,
        ephemeral: bool = False,
    ) -> str:
        """Insert the selection, populate the view, and return its URI.

        The v2 analog of v1's ``FigURLCurationSelection.generate_curation_uri``:
        a one-call convenience that creates/inserts the
        ``FigPackCurationSelection`` row, populates ``FigPackCuration``, and
        returns the stored ``figpack_uri``.
        """
        selection = FigPackCurationSelection.insert_selection(
            curation_key,
            label_options=label_options,
            metrics=metrics,
            upload=upload,
            ephemeral=ephemeral,
        )
        # An offline bundle lives under a temp dir that can be purged; if the
        # row exists but its local bundle is gone, drop the stale row so populate
        # rebuilds it rather than returning a dead path (mirrors run_v2_pipeline's
        # figpack reuse guard). A hosted (upload=True) URI is remote, so the
        # on-disk check does not apply.
        built = cls & selection
        if (
            not upload
            and built
            and not Path(built.fetch1("figpack_uri")).exists()
        ):
            built.delete_quick()
        cls.populate(selection)
        return (cls & selection).fetch1("figpack_uri")

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

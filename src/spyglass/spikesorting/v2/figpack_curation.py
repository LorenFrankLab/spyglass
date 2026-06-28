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
    FigPackUploadError,
)
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
    """Fetch a figure's ``annotations.json`` (HTTP or local path); {} if absent.

    Mirrors how the FigPack frontend loads annotations: a GET on
    ``<figure>/annotations.json`` for a hosted figure, or a file read for a
    saved bundle directory. A missing file / 404 yields ``{}`` (a pristine,
    never-edited figure round-trips to no curation).
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
            with urllib.request.urlopen(annotations_url) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return {}
            raise
        except urllib.error.URLError:
            # Server unreachable (e.g. an expired offline local server) -> no
            # retrievable edits rather than a hard failure.
            return {}

    path = Path(base) / "annotations.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


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


def _seed_offline_annotations(bundle: Path, curation_key: dict, label_options):
    """Pre-write ``annotations.json`` so a saved bundle opens pre-curated.

    Best-effort: seeds the FigPack curation control with the curation's existing
    labels and merge groups so a reviewer continues from the committed state.
    A failure here never blocks publishing the view.
    """
    try:
        labels = CurationV2._labels_by_unit(curation_key)
        merge_groups = [
            sorted({kept, *contributors})
            for kept, contributors in CurationV2.get_merge_groups(
                curation_key
            ).items()
            if len({kept, *contributors}) > 1
        ]
        payload = labels_and_merges_to_annotations(
            labels, merge_groups, label_options=label_options
        )
        (bundle / "annotations.json").write_text(json.dumps(payload, indent=2))
    except Exception as exc:  # noqa: BLE001 - seeding is a non-fatal nicety
        logger.warning(
            f"FigPack: could not seed initial curation annotations: {exc}"
        )


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
            Metric column names to display. Defaults to ``[]`` (SI's default
            unit-table columns).
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
        CurationV2.assert_committed_curation(
            parent_key, context="FigPackCuration"
        )

        label_options = (
            list(label_options) if label_options else default_label_options()
        )
        metrics = list(metrics) if metrics else []
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
            upload=bool(selection["upload"]),
            ephemeral=bool(selection["ephemeral"]),
            title=title,
            figpack_curation_id=key["figpack_curation_id"],
        )
        if not bool(selection["upload"]):
            _seed_offline_annotations(Path(uri), curation_key, label_options)

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

"""Compose the FigPack review view: SI sorting summary + curation control.

The pieces of the review figure that do not need a database: the
``SortingCuration`` control seeded with the committed labels, and the
vertical layout that gives that control a fixed, reachable strip under the
SpikeInterface sorting summary. ``figpack_curation._build_curation_view``
(schema-bound) resolves the analyzer and the review table, then composes
through these helpers; the browser regression tests compose the same layout
over an in-memory analyzer.

DB-free: imports FigPack lazily (the optional curation extra) and nothing
else beyond the standard library.
"""

from __future__ import annotations

from spyglass.spikesorting.v2._figpack_curation import FIGPACK_INSTALL_HINT

#: Fixed height (px) reserved for the curation control when expanded. A
#: LayoutItem with only ``max_size`` reserves NO space when a sibling has
#: ``stretch``, which collapsed the control to its title and made its buttons
#: unclickable; the pane is collapsible so the scientific views can take the
#: whole height while inspecting.
CURATION_CONTROL_HEIGHT = 280

CURATION_PANE_TITLE = "Curation"
REVIEW_HELP = (
    "**Edit:** Curate Figure → select units → labels or Merge Selected. "
    "**Save Annotations** saves edits to this bundle. **Finalize Curation** "
    "sets a browser flag. **Commit in Python:** `preview_import()` → `commit()`.\n\n"
    "Metrics describe the committed curation; pending merges do not update them. "
    "After committing a merge, inspect its reevaluated child with `continue_review()`. "
    "Blank metrics are unavailable, not zero. An unflagged unit is not necessarily "
    "accepted; exclusion labels override acceptance in the shipped selection policies."
)


def require_figpack():
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


def curation_control(label_options, seed_labels=None):
    """The ``SortingCuration`` control seeded with committed labels.

    ``mergeGroups`` starts empty on purpose: applied merge provenance is
    shown in the unit table (``merged_from``), and only NEW browser edits may
    populate the control's merge groups.
    """
    _, figpack_ss_views = require_figpack()
    return figpack_ss_views.SortingCuration(
        default_label_options=list(label_options),
        curation={
            "labelsByUnit": {
                str(unit_id): list(labels)
                for unit_id, labels in (seed_labels or {}).items()
            },
            "mergeGroups": [],
            "isClosed": False,
            "labelChoices": list(label_options),
        },
    )


def compose_review_layout(
    summary, control, *, summary_title: str, context: str = ""
):
    """Stack the SI sorting summary over the curation control.

    The summary (selectable unit table + scientific views) takes all
    remaining height; the control keeps a fixed strip below it so its label
    and merge buttons are clickable at laptop and large viewports alike, and
    collapses (title click) to hand the whole height to the views.
    """
    figpack_views, _ = require_figpack()
    return figpack_views.Box(
        direction="vertical",
        items=[
            figpack_views.LayoutItem(
                view=figpack_views.Markdown(
                    REVIEW_HELP + ("\n\n" + context if context else ""),
                    font_size=12,
                ),
                title="Review instructions and QC (scroll for details)",
                min_size=130,
                max_size=130,
                collapsible=True,
            ),
            figpack_views.LayoutItem(
                view=summary, title=summary_title, stretch=1
            ),
            figpack_views.LayoutItem(
                view=control,
                title=CURATION_PANE_TITLE,
                min_size=CURATION_CONTROL_HEIGHT,
                max_size=CURATION_CONTROL_HEIGHT,
                collapsible=True,
            ),
        ],
    )


def coerce_units_table_ids(view) -> None:
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

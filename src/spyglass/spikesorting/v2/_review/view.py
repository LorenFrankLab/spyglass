"""Build the FigPack review figure without database access.

The schema layer resolves the analyzer and review inputs. FigPack and
SpikeInterface are imported only when rendering.
"""

from __future__ import annotations

from pathlib import Path

from spyglass.spikesorting.v2._review.annotations import (
    FIGPACK_INSTALL_HINT,
    curation_state,
    normalize_displayed_unit_properties,
)
from spyglass.spikesorting.v2.exceptions import (
    FigPackDisplayedUnitPropertyError,
)
from spyglass.spikesorting.v2._review.inspection import (
    defer_time_views,
    inspection_view,
)
from spyglass.spikesorting.v2._review.profile import ReviewDisplayOptions
from spyglass.spikesorting.v2._review.unit_properties import (
    review_unit_properties,
)

#: Fixed control height (px). With only ``max_size``, a LayoutItem beside a
#: ``stretch`` sibling reserves no space and hides the buttons.
CURATION_CONTROL_HEIGHT = 150

CURATION_PANE_TITLE = "Curation"
REVIEW_HELP = (
    "**Edit:** select units → labels or Merge Selected. **Save draft** saves "
    "browser edits; it does not commit a curation. For hosted figures, the "
    "authenticated **Save Annotations** toolbar action also saves a draft. "
    "**Commit and review:** use **Preview and commit** in a connected local review, "
    "or `review.commit_panel()` in the notebook.\n\n"
    "Pending merges do not update metrics. Committing reevaluates merged "
    "units and opens their verification review. Blank metrics are unavailable, not "
    "zero. Exclusion labels override acceptance in the shipped selection policies."
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
    """The Spyglass draft control seeded with committed labels.

    ``mergeGroups`` starts empty on purpose: applied merge provenance is
    shown in the unit table (``merged_from``), and only NEW browser edits may
    populate the control's merge groups.
    """
    import figpack

    extension = figpack.FigpackExtension(
        name="spyglass-review",
        javascript_code=Path(__file__).with_name("controls.js").read_text(),
        version="1.0.0",
    )

    class ReviewControls(figpack.ExtensionView):
        def write_to_zarr_group(self, group):
            super().write_to_zarr_group(group)
            group.attrs["label_options"] = list(label_options)
            group.attrs["curation"] = curation_state(
                seed_labels, [], label_options=label_options
            )

    return ReviewControls(
        extension=extension, view_type="spyglass.ReviewControls"
    )


def compose_review_layout(
    summary, control, *, summary_title: str, context: str = ""
):
    """Stack the SI sorting summary over the curation control.

    The summary takes the remaining height; the control keeps a fixed,
    collapsible strip so its buttons stay clickable.
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
                min_size=60,
                max_size=60,
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

    ``UnitsTable`` ``json.dumps`` rows that SI builds from ``numpy.int32``
    unit ids, raising ``TypeError`` at save. Remove once upstream serializes
    these.
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


def available_displayed_unit_properties(analyzer) -> list[str]:
    """Return unit-table columns SpikeInterface can render for ``analyzer``."""
    from spikeinterface.widgets.utils import make_units_table_from_analyzer

    table = make_units_table_from_analyzer(analyzer)
    return [str(column) for column in table.columns]


def assert_displayed_unit_properties_available(
    analyzer, displayed_unit_properties
) -> None:
    """Fail closed when requested FigPack unit-table columns are unavailable."""
    requested = normalize_displayed_unit_properties(displayed_unit_properties)
    # ``None`` defers to SI's defaults and ``[]`` requests no columns, so
    # neither needs the analyzer's units table.
    if not requested:
        return
    available = available_displayed_unit_properties(analyzer)
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


def build_curation_view(
    analyzer,
    curation_key: dict,
    *,
    timeline: dict | None,
    label_options,
    displayed_unit_properties,
    seed_labels=None,
    review_table=None,
    display_options=None,
):
    """Build the FigPack curation view for a curation (minimal-attach).

    Composes SI inspection widgets over the display ``analyzer`` and attaches
    the draft control. No DB access. ``timeline`` is the curation's
    ``review_timeline``; ``display_options`` (:class:`ReviewDisplayOptions`)
    bounds the payload.

    ``review_table`` (profile-backed reviews: evaluation metrics, annotation
    columns and proposals, indexed by unit id) replaces SI's default unit
    table columns, so the evaluation wins over same-named analyzer
    properties.
    """
    display = ReviewDisplayOptions.from_mapping(display_options)
    if review_table is not None:
        # Profile-backed: exactly the review columns, no SI defaults.
        analyzer_properties: list[str] | None = []
        extra_properties = review_unit_properties(
            review_table, analyzer.unit_ids
        )
    else:
        # Expert path: analyzer-native properties (None: SI defaults).
        analyzer_properties = displayed_unit_properties
        extra_properties = None
    assert_displayed_unit_properties_available(analyzer, analyzer_properties)
    deferred = defer_time_views(analyzer, display)
    summary = inspection_view(
        analyzer,
        display,
        timeline=timeline,
        deferred=deferred,
        displayed_unit_properties=analyzer_properties,
        extra_unit_properties=extra_properties,
        min_similarity_for_correlograms=display.min_similarity_for_correlograms,
    )

    control = curation_control(label_options, seed_labels)
    summary_title = "Sorting summary"
    context = (
        f"Sorting `{curation_key['sorting_id']}`, curation "
        f"`{curation_key['curation_id']}`. {display.describe()}. "
        "Omitted pairs are filtered from the display, not evidence of no correlation. "
        "Select units and use Inspect selected units / pairs for every pair and "
        "an exact raster window. Python alternative: "
        "`review.inspect_units([id1, id2], time_range=(start, stop))`. "
        "Times are seconds from the start of this sorting recording; concatenated "
        "recordings use the concatenated timeline."
    )
    if review_table is not None:
        context += " " + review_table.attrs.get("qc_context", "")
    view = compose_review_layout(
        summary, control, summary_title=summary_title, context=context
    )
    coerce_units_table_ids(view)
    return view

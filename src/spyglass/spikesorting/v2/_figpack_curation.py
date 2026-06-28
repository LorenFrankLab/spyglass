"""DB-free helpers for the FigPack curation tables.

Pure logic split out of ``figpack_curation`` so it is unit-testable without a
DataJoint connection or the optional ``figpack`` packages: the content-addressed
config hash, the default label set, and the translation between FigPack's
``sorting_curation`` annotation state and v2's ``(labels, merge_groups)`` form.

DB-FREE BY CONTRACT. Imports only the standard library plus the pure
``_enums`` module; it never imports DataJoint, SpikeInterface, or figpack, and
opens no database connection at import (mirrors ``_selection_identity``).
"""

from __future__ import annotations

import hashlib
import json

from spyglass.spikesorting.v2._enums import CurationLabel

#: Install hint surfaced when the optional FigPack packages are missing.
FIGPACK_INSTALL_HINT = (
    "FigPack curation requires the optional 'figpack' and "
    "'figpack-spike-sorting' packages. Install them with "
    '`pip install -e ".[spikesorting-v2-curation]"` (or '
    "`pip install figpack figpack-spike-sorting`)."
)

#: FigPack's annotation key holding the JSON-encoded curation state, and the
#: figure-root path it is stored under in ``annotations.json``.
SORTING_CURATION_KEY = "sorting_curation"
ANNOTATION_ROOT_PATH = "/"


def default_label_options() -> list[str]:
    """Return the default FigPack label choices in curation display order.

    The three primary manual labels (``accept`` / ``mua`` / ``noise``) drawn
    from :class:`CurationLabel`, not the FigURL-era ``"good"``. ``artifact`` /
    ``reject`` are valid v2 labels but are omitted from the default UI palette;
    a caller can pass an explicit ``label_options`` to include them.
    """
    return [
        CurationLabel.accept.value,
        CurationLabel.mua.value,
        CurationLabel.noise.value,
    ]


def figpack_config_hash(
    *,
    sorting_id,
    curation_id,
    label_options,
    metrics,
    upload,
    ephemeral,
) -> str:
    """Return the sha256 hex digest content-addressing a FigPack UI config.

    Two ``FigPackCurationSelection`` rows are the same configuration iff they
    target the same curation and request the same label palette, metric set,
    and upload mode. List ORDER is significant (it is the display order), so the
    lists are hashed as-given, not sorted; only the dict keys are sorted for
    byte-stability.

    Parameters
    ----------
    sorting_id, curation_id
        The ``CurationV2`` primary key the view is built for.
    label_options, metrics : list of str
        The curation label palette and the metric columns to display.
    upload, ephemeral : bool
        The publish mode flags.

    Returns
    -------
    str
        The 64-char sha256 hex digest.
    """
    payload = {
        "sorting_id": str(sorting_id),
        "curation_id": int(curation_id),
        "label_options": list(label_options),
        "metrics": list(metrics),
        "upload": bool(upload),
        "ephemeral": bool(ephemeral),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def labels_and_merges_to_annotations(
    labels: dict | None,
    merge_groups: list | None,
    *,
    label_options: list[str] | None = None,
    is_closed: bool = False,
) -> dict:
    """Build a FigPack ``annotations.json`` payload seeding a curation state.

    The inverse of :func:`curation_annotations_to_labels_and_merges`: turns v2's
    ``{unit_id: [label]}`` labels and ``[[unit_id, ...]]`` merge groups into the
    ``{"annotations": {"/": {"sorting_curation": "<json>"}}}`` envelope FigPack
    serves and the curation control reads, so an editable view can open
    pre-seeded with a curation's existing decisions.

    Parameters
    ----------
    labels : dict or None
        ``{unit_id: [label, ...]}`` mapping. ``None`` is treated as empty.
    merge_groups : list or None
        ``[[unit_id, ...], ...]`` merge groups. ``None`` is treated as empty.
    label_options : list of str, optional
        The label palette to record as ``labelChoices`` (omitted if ``None``).
    is_closed : bool, optional
        Whether the seeded curation is marked finalized. Default ``False``.

    Returns
    -------
    dict
        The ``annotations.json`` payload.
    """
    state = {
        "labelsByUnit": {
            str(unit_id): list(unit_labels)
            for unit_id, unit_labels in (labels or {}).items()
        },
        "mergeGroups": [list(group) for group in (merge_groups or [])],
        "isClosed": bool(is_closed),
    }
    if label_options:
        state["labelChoices"] = list(label_options)
    return {
        "annotations": {
            ANNOTATION_ROOT_PATH: {SORTING_CURATION_KEY: json.dumps(state)}
        }
    }


def curation_annotations_to_labels_and_merges(
    annotations: dict | None,
) -> tuple[dict, list]:
    """Parse a FigPack ``annotations.json`` payload into ``(labels, merges)``.

    The retrieval half of the curation round trip: reads the
    ``sorting_curation`` annotation FigPack writes when a user edits and saves a
    curation, and returns it in the exact shape
    ``CurationV2.insert_curation(labels=..., merge_groups=...)`` consumes. Unit
    ids are coerced to ``int`` (v2 unit ids are integers; FigPack stores the
    ``labelsByUnit`` keys as strings). A missing/empty payload, a missing
    ``sorting_curation`` entry, or an empty state all yield ``({}, [])`` rather
    than raising, so a pristine (never-edited) figure round-trips cleanly.

    Parameters
    ----------
    annotations : dict or None
        The parsed ``annotations.json`` payload (or ``None`` if absent).

    Returns
    -------
    tuple[dict, list]
        ``({unit_id: [label, ...]}, [[unit_id, ...], ...])``.
    """
    if not annotations:
        return {}, []
    node = (annotations.get("annotations") or {}).get(
        ANNOTATION_ROOT_PATH
    ) or {}
    raw_state = node.get(SORTING_CURATION_KEY)
    if not raw_state:
        return {}, []
    state = json.loads(raw_state) if isinstance(raw_state, str) else raw_state

    labels = {
        int(unit_id): list(unit_labels)
        for unit_id, unit_labels in (state.get("labelsByUnit") or {}).items()
    }
    merge_groups = [
        [int(unit_id) for unit_id in group]
        for group in (state.get("mergeGroups") or [])
    ]
    return labels, merge_groups

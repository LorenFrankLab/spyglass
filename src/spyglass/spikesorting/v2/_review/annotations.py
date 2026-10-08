"""DB-free helpers for the FigPack curation tables.

Pure logic kept out of ``figpack_curation`` so it is unit-testable without a
DataJoint connection or the optional ``figpack`` packages: the content-addressed
config hash, the default label set, and the translation between FigPack's
``sorting_curation`` annotation state and v2's ``(labels, merge_groups)`` form,
and small FigPack config normalization helpers.

DB-FREE BY CONTRACT. Imports only the standard library and dependency-light
curation helpers; it never imports DataJoint, SpikeInterface, or figpack, and
opens no database connection at import (mirrors ``_core.selection_identity``).
"""

from __future__ import annotations

import json

from spyglass.spikesorting.v2._curation.transforms import parse_curation_unit_id
from spyglass.spikesorting.v2._core.enums import CurationLabel
from spyglass.spikesorting.v2._core.selection_identity import sha256_json

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
FIGURE_CONFIG_FILENAME = "spyglass_curation.json"


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
    curation_uuid,
    label_options,
    displayed_unit_properties,
    upload,
    ephemeral,
    review_config=None,
) -> str:
    """Return the sha256 hex digest content-addressing a FigPack UI config.

    Two ``FigPackCurationSelection`` rows are the same configuration iff they
    target the same curation and request the same label palette, displayed unit
    table columns, and upload mode. List ORDER is significant (it is the display
    order), so lists are hashed as-given, not sorted; only the dict keys are
    sorted for byte-stability.

    Parameters
    ----------
    sorting_id, curation_id, curation_uuid
        The ``CurationV2`` key and immutable row-generation identity the view
        is built for. Every configuration requires its curation generation.
    label_options : list of str
        The curation label palette.
    displayed_unit_properties : list of str or None
        Unit-table properties passed to SpikeInterface
        ``plot_sorting_summary(displayed_unit_properties=...)``. ``None`` keeps
        SpikeInterface's backend defaults; ``[]`` requests no property columns.
    upload, ephemeral : bool
        The publish mode flags.
    review_config : dict or None
        Exact persisted profile/evaluation snapshot for a guided review.

    Returns
    -------
    str
        The 64-char sha256 hex digest.
    """
    if curation_uuid is None:
        raise ValueError("curation_uuid must identify a curation generation.")
    payload = {
        "sorting_id": str(sorting_id),
        "curation_id": int(curation_id),
        "curation_uuid": str(curation_uuid),
        "label_options": list(label_options),
        "displayed_unit_properties": normalize_displayed_unit_properties(
            displayed_unit_properties
        ),
        "upload": bool(upload),
        "ephemeral": bool(ephemeral),
        "review_config": _json_native(review_config),
    }
    return sha256_json(payload)


def _json_native(value):
    """Return a deterministic JSON-native copy of nested config values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    if hasattr(value, "item"):
        return _json_native(value.item())
    return str(value)


def pack_display_config(displayed_unit_properties, review_config=None):
    """Store expert properties and an optional guided-review snapshot together."""
    properties = normalize_displayed_unit_properties(displayed_unit_properties)
    if review_config is not None and not isinstance(review_config, dict):
        raise TypeError(
            "review_config must be a mapping or None; got "
            f"{type(review_config).__name__}."
        )
    return {
        "properties": properties,
        "review": _json_native(review_config),
    }


def unpack_display_config(stored) -> tuple[list[str] | None, dict | None]:
    """Decode the current display mapping for expert and guided reviews."""
    if not isinstance(stored, dict) or set(stored) != {"properties", "review"}:
        raise TypeError(
            "stored display configuration must contain exactly 'properties' "
            "and 'review'."
        )
    if stored["properties"] is not None and not isinstance(
        stored["properties"], list
    ):
        raise TypeError("stored properties must be a list or None.")
    properties = normalize_displayed_unit_properties(stored["properties"])
    review = stored["review"]
    if review is not None and not isinstance(review, dict):
        raise TypeError(
            "stored review configuration must be a mapping or None."
        )
    return properties, _json_native(review)


def annotations_payload_hash(annotations: dict | None) -> str:
    """Hash the complete logical annotations payload independent of spacing."""
    return sha256_json(_json_native(annotations or {}))


def normalize_displayed_unit_properties(
    displayed_unit_properties,
) -> list[str] | None:
    """Normalize optional FigPack unit-table column selection.

    Parameters
    ----------
    displayed_unit_properties : list of str or None
        ``None`` preserves SpikeInterface's default unit-table properties.
        An explicit list requests exactly those properties in display order;
        ``[]`` is distinct from ``None`` and requests no property columns.

    Returns
    -------
    list of str or None
        Normalized value suitable for hashing and storage.

    Raises
    ------
    TypeError
        If the value is not ``None`` or a list/tuple of strings.
    ValueError
        If a property name is empty or duplicated.
    """
    if displayed_unit_properties is None:
        return None
    if not isinstance(displayed_unit_properties, (list, tuple)):
        raise TypeError(
            "displayed_unit_properties must be None or a list of strings; "
            f"got {type(displayed_unit_properties).__name__}."
        )
    normalized = []
    for prop in displayed_unit_properties:
        if not isinstance(prop, str):
            raise TypeError(
                "displayed_unit_properties must contain only strings; "
                f"got {type(prop).__name__} in {displayed_unit_properties!r}."
            )
        if not prop:
            raise ValueError(
                "displayed_unit_properties entries must be non-empty strings."
            )
        normalized.append(prop)
    duplicates = sorted(
        {prop for prop in normalized if normalized.count(prop) > 1}
    )
    if duplicates:
        raise ValueError(
            "displayed_unit_properties contains duplicate entries: "
            f"{duplicates}."
        )
    return normalized


def curation_state(
    labels: dict | None,
    merge_groups: list | None,
    *,
    label_options: list[str] | None = None,
    is_closed: bool = False,
) -> dict:
    """Build the curation state the review control reads and writes.

    ``{"labelsByUnit": {str(unit_id): [label, ...]}, "mergeGroups":
    [[unit_id, ...]], "isClosed": bool}``, plus ``"labelChoices"`` when
    ``label_options`` is not ``None``. ``None`` labels or merge groups are
    treated as empty.
    """
    state = {
        "labelsByUnit": {
            str(unit_id): list(unit_labels)
            for unit_id, unit_labels in (labels or {}).items()
        },
        "mergeGroups": [list(group) for group in (merge_groups or [])],
        "isClosed": bool(is_closed),
    }
    if label_options is not None:
        state["labelChoices"] = list(label_options)
    return state


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
    state = curation_state(
        labels,
        merge_groups,
        label_options=label_options or None,
        is_closed=is_closed,
    )
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
    ids accept integers and integer strings (FigPack stores the ``labelsByUnit``
    keys as strings); floats and booleans are rejected. A missing/empty payload,
    a missing ``sorting_curation`` entry, or an empty state all yield
    ``({}, [])`` rather than raising, so a pristine figure round-trips cleanly.

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
        parse_curation_unit_id(unit_id): list(unit_labels)
        for unit_id, unit_labels in (state.get("labelsByUnit") or {}).items()
    }
    merge_groups = [
        [parse_curation_unit_id(unit_id) for unit_id in group]
        for group in (state.get("mergeGroups") or [])
    ]
    return labels, merge_groups

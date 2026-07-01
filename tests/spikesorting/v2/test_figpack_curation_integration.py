"""Integration tests for the FigPack curation tables (require the curation extra).

Exercises ``FigPackCuration`` end to end against a real populated v2
``SortingAnalyzer``: building the curation view (offline bundle), idempotency,
the edited-curation round trip through ``annotations.json``, and the hosted-upload
credential gate. Skipped unless the ``spikesorting-v2-curation`` extra (``figpack``
+ ``figpack_spike_sorting``) is installed.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_FIGPACK_MISSING = (
    importlib.util.find_spec("figpack") is None
    or importlib.util.find_spec("figpack_spike_sorting") is None
)

pytestmark = pytest.mark.skipif(
    _FIGPACK_MISSING,
    reason="requires the spikesorting-v2-curation extra (figpack)",
)


def _units_table_column_keys(bundle: Path) -> list[list[str]]:
    """Return every FigPack UnitsTable's column keys from an offline bundle."""
    metadata = json.loads((bundle / "data.zarr" / ".zmetadata").read_text())
    tables: list[list[str]] = []
    for path, attrs in metadata.get("metadata", {}).items():
        if not path.endswith(".zattrs"):
            continue
        if attrs.get("view_type") != "spike_sorting.UnitsTable":
            continue
        tables.append([column["key"] for column in attrs.get("columns", [])])
    return tables


def test_build_curation_view_offline_creates_bundle(
    populated_sorting_with_curation,
):
    """build_curation_view(upload=False) saves a real, openable static bundle."""
    from spyglass.spikesorting.v2.figpack_curation import FigPackCuration

    uri = FigPackCuration.build_curation_view(
        populated_sorting_with_curation, upload=False
    )
    bundle = Path(uri)
    assert bundle.is_dir()
    assert (bundle / "index.html").exists()
    assert (bundle / "data.zarr").exists()


def test_displayed_unit_properties_render_in_bundle(
    populated_sorting_with_curation,
):
    """Explicit SI unit-table properties are written into the FigPack bundle."""
    from spyglass.spikesorting.v2.figpack_curation import FigPackCuration

    uri = FigPackCuration.build_curation_view(
        populated_sorting_with_curation,
        upload=False,
        displayed_unit_properties=["x", "y"],
    )
    tables = _units_table_column_keys(Path(uri))
    assert any({"x", "y"} <= set(columns) for columns in tables), tables


def test_displayed_unit_properties_reject_missing(
    populated_sorting_with_curation,
):
    """Explicit unavailable unit-table properties fail instead of disappearing."""
    from spyglass.spikesorting.v2.exceptions import (
        FigPackDisplayedUnitPropertyError,
    )
    from spyglass.spikesorting.v2.figpack_curation import FigPackCuration

    with pytest.raises(FigPackDisplayedUnitPropertyError, match="not_a_column"):
        FigPackCuration.build_curation_view(
            populated_sorting_with_curation,
            upload=False,
            displayed_unit_properties=["not_a_column"],
        )


def test_selection_and_view_are_idempotent(populated_sorting_with_curation):
    """Identical config -> one selection id and one populated view."""
    from spyglass.spikesorting.v2.figpack_curation import (
        FigPackCuration,
        FigPackCurationSelection,
    )

    first = FigPackCurationSelection.insert_selection(
        populated_sorting_with_curation
    )
    second = FigPackCurationSelection.insert_selection(
        populated_sorting_with_curation
    )
    assert first == second

    uri_a = FigPackCuration.build_curation_view(
        populated_sorting_with_curation, upload=False
    )
    uri_b = FigPackCuration.build_curation_view(
        populated_sorting_with_curation, upload=False
    )
    assert uri_a == uri_b
    assert len(FigPackCuration & first) == 1


def test_edited_curation_round_trips(populated_sorting_with_curation):
    """A user's edited annotations.json round-trips to (labels, merge_groups)."""
    from spyglass.spikesorting.v2._figpack_curation import (
        labels_and_merges_to_annotations,
    )
    from spyglass.spikesorting.v2.figpack_curation import FigPackCuration

    uri = FigPackCuration.build_curation_view(
        populated_sorting_with_curation, upload=False
    )

    # Simulate a curator editing in the browser and saving: overwrite the
    # figure's annotations.json with a known state.
    edited_labels = {0: ["noise"], 1: ["accept"]}
    edited_merges = [[0, 1]]
    (Path(uri) / "annotations.json").write_text(
        json.dumps(
            labels_and_merges_to_annotations(edited_labels, edited_merges)
        )
    )

    labels, merge_groups = FigPackCuration.fetch_curation_from_uri(uri)
    assert labels == edited_labels
    assert merge_groups == edited_merges


def test_pristine_view_round_trips_to_empty(populated_sorting_with_curation):
    """A never-edited (root-curation) figure fetches as no curation."""
    from spyglass.spikesorting.v2.figpack_curation import FigPackCuration

    uri = FigPackCuration.build_curation_view(
        populated_sorting_with_curation, upload=False
    )
    # The root curation has no labels/merges, so the seeded state is empty.
    assert FigPackCuration.fetch_curation_from_uri(uri) == ({}, [])


def test_fetch_from_uri_chains_into_save_manual_curation(
    populated_sorting_with_curation,
):
    """The full FigPack return path: fetch a figure's edited annotations and
    commit them as the next curation via ``save_manual_curation``.

    Closes the loop between the two halves already tested separately
    (``fetch_curation_from_uri`` reads annotations; ``save_manual_curation``
    commits a payload): a browser edit becomes a committed child curation.
    """
    from spyglass.spikesorting.v2._figpack_curation import (
        labels_and_merges_to_annotations,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.figpack_curation import FigPackCuration

    root = populated_sorting_with_curation
    unit_ids = sorted(int(u) for u in (CurationV2.Unit & root).fetch("unit_id"))
    if not unit_ids:
        pytest.skip("need >=1 curated unit to label")
    labeled = unit_ids[0]

    uri = FigPackCuration.build_curation_view(root, upload=False)
    # Simulate a curator labeling one unit 'noise' in the browser and saving.
    edited_labels = {labeled: ["noise"]}
    (Path(uri) / "annotations.json").write_text(
        json.dumps(labels_and_merges_to_annotations(edited_labels, []))
    )

    # Return path: fetch the edited state, then commit it as the next curation.
    labels, merge_groups = FigPackCuration.fetch_curation_from_uri(uri)
    assert labels == edited_labels
    child = CurationV2.save_manual_curation(
        {"sorting_id": root["sorting_id"]},
        parent_curation_id=root["curation_id"],
        labels=labels,
        merge_groups=merge_groups,
        curation_source="figpack",
        description="curated in FigPack",
    )

    assert child["curation_id"] != root["curation_id"]
    assert (CurationV2 & child).fetch1("curation_source") == "figpack"
    stored = {
        (int(r["unit_id"]), r["curation_label"])
        for r in (CurationV2.UnitLabel & child).fetch(as_dict=True)
    }
    assert stored == {(labeled, "noise")}


def test_upload_without_api_key_raises(monkeypatch):
    """upload=True without FIGPACK_API_KEY raises a clear, typed error."""
    from spyglass.spikesorting.v2.exceptions import FigPackUploadError
    from spyglass.spikesorting.v2.figpack_curation import _publish_view

    monkeypatch.delenv("FIGPACK_API_KEY", raising=False)
    with pytest.raises(FigPackUploadError):
        # The credential gate fires before the view is ever shown, so a dummy
        # view object never gets touched.
        _publish_view(
            view=object(),
            upload=True,
            ephemeral=False,
            title="x",
            figpack_curation_id="x",
        )


def test_fetch_from_unreachable_uri_fails_closed():
    """An unreachable figure raises, not silently look like 'no edits'."""
    from spyglass.spikesorting.v2.exceptions import FigPackRetrievalError
    from spyglass.spikesorting.v2.figpack_curation import FigPackCuration

    # Port 9 (discard) refuses the connection -> URLError, not a 404.
    with pytest.raises(FigPackRetrievalError):
        FigPackCuration.fetch_curation_from_uri("http://127.0.0.1:9/figure")


def test_ephemeral_normalized_when_offline(populated_sorting_with_curation):
    """ephemeral is inert offline -> normalized away (one identity, stored 0)."""
    from spyglass.spikesorting.v2.figpack_curation import (
        FigPackCurationSelection,
    )

    with_ephemeral = FigPackCurationSelection.insert_selection(
        populated_sorting_with_curation, upload=False, ephemeral=True
    )
    without_ephemeral = FigPackCurationSelection.insert_selection(
        populated_sorting_with_curation, upload=False, ephemeral=False
    )
    assert with_ephemeral == without_ephemeral
    assert (
        int((FigPackCurationSelection & with_ephemeral).fetch1("ephemeral"))
        == 0
    )


def test_merged_curation_rejected(planted_two_unit_sort):
    """A merged curation (non-raw namespace) is refused with a typed error."""
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.exceptions import (
        FigPackCurationNamespaceError,
    )
    from spyglass.spikesorting.v2.figpack_curation import (
        FigPackCurationSelection,
    )
    from spyglass.spikesorting.v2.sorting import Sorting

    clear_curations_for(planted_two_unit_sort)  # package-scoped sort is shared
    unit_ids = sorted(
        int(u) for u in (Sorting.Unit & planted_two_unit_sort).fetch("unit_id")
    )
    merged = CurationV2.insert_curation(
        sorting_key=planted_two_unit_sort,
        merge_groups=[unit_ids[:2]],
        apply_merge=True,
    )
    with pytest.raises(FigPackCurationNamespaceError):
        FigPackCurationSelection.insert_selection(merged)
    clear_curations_for(planted_two_unit_sort)


def test_upload_of_labeled_curation_rejected(planted_two_unit_sort):
    """Hosted upload of a curation with existing labels raises (no blank view)."""
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.exceptions import FigPackUploadError
    from spyglass.spikesorting.v2.figpack_curation import FigPackCuration
    from spyglass.spikesorting.v2.sorting import Sorting

    clear_curations_for(planted_two_unit_sort)  # package-scoped sort is shared
    unit_ids = sorted(
        int(u) for u in (Sorting.Unit & planted_two_unit_sort).fetch("unit_id")
    )
    labeled = CurationV2.insert_curation(
        sorting_key=planted_two_unit_sort, labels={unit_ids[0]: ["noise"]}
    )
    with pytest.raises(FigPackUploadError):
        FigPackCuration.build_curation_view(labeled, upload=True)
    clear_curations_for(planted_two_unit_sort)


def test_fetch_from_nonexistent_local_path_fails_closed():
    """A missing/typoed local figure path raises, not look like 'no edits'."""
    from spyglass.spikesorting.v2.exceptions import FigPackRetrievalError
    from spyglass.spikesorting.v2.figpack_curation import FigPackCuration

    with pytest.raises(FigPackRetrievalError):
        FigPackCuration.fetch_curation_from_uri(
            "/tmp/figpack_does_not_exist_xyz/figure"
        )


def test_displayed_unit_properties_public_but_metrics_not_public(dj_conn):
    """Expose SI-native unit properties, not v1's narrower metrics name.

    Importing the table classes activates DataJoint schemas, so depend on the
    DB fixture even though this assertion only inspects public signatures.
    """
    import inspect

    from spyglass.spikesorting.v2.figpack_curation import (
        FigPackCuration,
        FigPackCurationSelection,
    )

    selection_params = inspect.signature(
        FigPackCurationSelection.insert_selection
    ).parameters
    build_params = inspect.signature(
        FigPackCuration.build_curation_view
    ).parameters
    assert "displayed_unit_properties" in selection_params
    assert "displayed_unit_properties" in build_params
    assert "metrics" not in selection_params
    assert "metrics" not in build_params


def test_make_rejects_tampered_config_hash(populated_sorting_with_curation):
    """A bypassed selection whose config hash != its fields is refused."""
    from spyglass.spikesorting.v2._figpack_curation import (
        default_label_options,
    )
    from spyglass.spikesorting.v2._selection_identity import deterministic_id
    from spyglass.spikesorting.v2.exceptions import SchemaBypassError
    from spyglass.spikesorting.v2.figpack_curation import (
        FigPackCuration,
        FigPackCurationSelection,
    )

    label_options = default_label_options()
    # A config hash that does NOT match the row's own fields; pair it with the
    # id derived from that same wrong hash so the hash recheck (not the id
    # recheck) is what fires.
    wrong_hash = "0" * 64
    identity = {
        **populated_sorting_with_curation,
        "figpack_config_hash": wrong_hash,
    }
    figpack_id = deterministic_id("figpack_curation", identity)
    FigPackCurationSelection.insert1(
        {
            **identity,
            "figpack_curation_id": figpack_id,
            "label_options": label_options,
            "displayed_unit_properties": None,
            "upload": False,
            "ephemeral": False,
        },
        allow_direct_insert=True,
    )
    with pytest.raises(SchemaBypassError):
        FigPackCuration.populate({"figpack_curation_id": figpack_id})


def test_make_rejects_offline_ephemeral_bypass(populated_sorting_with_curation):
    """A bypassed upload=False + ephemeral=True row is refused (inert flag)."""
    from spyglass.spikesorting.v2._figpack_curation import (
        default_label_options,
        figpack_config_hash,
    )
    from spyglass.spikesorting.v2._selection_identity import deterministic_id
    from spyglass.spikesorting.v2.exceptions import SchemaBypassError
    from spyglass.spikesorting.v2.figpack_curation import (
        FigPackCuration,
        FigPackCurationSelection,
    )

    label_options = default_label_options()
    # Hash/id computed FOR the offline+ephemeral combo, so the hash/id rechecks
    # pass and the offline-ephemeral invariant is what fires.
    config_hash = figpack_config_hash(
        sorting_id=populated_sorting_with_curation["sorting_id"],
        curation_id=populated_sorting_with_curation["curation_id"],
        label_options=label_options,
        displayed_unit_properties=None,
        upload=False,
        ephemeral=True,
    )
    identity = {
        **populated_sorting_with_curation,
        "figpack_config_hash": config_hash,
    }
    figpack_id = deterministic_id("figpack_curation", identity)
    FigPackCurationSelection.insert1(
        {
            **identity,
            "figpack_curation_id": figpack_id,
            "label_options": label_options,
            "displayed_unit_properties": None,
            "upload": False,
            "ephemeral": True,
        },
        allow_direct_insert=True,
    )
    with pytest.raises(SchemaBypassError):
        FigPackCuration.populate({"figpack_curation_id": figpack_id})


def test_make_revalidates_a_bypassed_selection(planted_two_unit_sort):
    """A selection bypassing insert_selection is re-validated at populate time."""
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    from spyglass.spikesorting.v2._figpack_curation import (
        default_label_options,
        figpack_config_hash,
    )
    from spyglass.spikesorting.v2._selection_identity import deterministic_id
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.exceptions import (
        FigPackCurationNamespaceError,
    )
    from spyglass.spikesorting.v2.figpack_curation import (
        FigPackCuration,
        FigPackCurationSelection,
    )
    from spyglass.spikesorting.v2.sorting import Sorting

    clear_curations_for(planted_two_unit_sort)
    unit_ids = sorted(
        int(u) for u in (Sorting.Unit & planted_two_unit_sort).fetch("unit_id")
    )
    merged = CurationV2.insert_curation(
        sorting_key=planted_two_unit_sort,
        merge_groups=[unit_ids[:2]],
        apply_merge=True,
    )
    label_options = default_label_options()
    config_hash = figpack_config_hash(
        sorting_id=merged["sorting_id"],
        curation_id=merged["curation_id"],
        label_options=label_options,
        displayed_unit_properties=None,
        upload=False,
        ephemeral=False,
    )
    identity = {**merged, "figpack_config_hash": config_hash}
    figpack_id = deterministic_id("figpack_curation", identity)
    # Bypass insert_selection's guard via the documented escape hatch.
    FigPackCurationSelection.insert1(
        {
            **identity,
            "figpack_curation_id": figpack_id,
            "label_options": label_options,
            "displayed_unit_properties": None,
            "upload": False,
            "ephemeral": False,
        },
        allow_direct_insert=True,
    )
    with pytest.raises(FigPackCurationNamespaceError):
        FigPackCuration.populate({"figpack_curation_id": figpack_id})
    clear_curations_for(planted_two_unit_sort)

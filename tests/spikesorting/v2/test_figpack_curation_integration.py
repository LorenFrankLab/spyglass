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

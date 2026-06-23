"""Tests for ``Sorting.add_extensions`` (in-place, idempotent extension add)."""

from __future__ import annotations

import pytest


@pytest.mark.slow
@pytest.mark.integration
def test_add_extensions_idempotent(populated_sorting):
    """First call computes a new extension; a second call is a no-op.

    ``add_extensions`` returns the list of extensions it actually computed, so
    idempotency is observable: the second request for an already-present
    extension computes nothing (returns ``[]``) and never recomputes
    waveforms/templates (which would cascade-delete derived extensions).
    """
    from spyglass.spikesorting.v2.sorting import Sorting

    sorting = Sorting()
    sorting.get_analyzer(populated_sorting)  # ensure the analyzer exists

    added = sorting.add_extensions(populated_sorting, ["correlograms"])
    assert "correlograms" in added
    assert sorting.get_analyzer(populated_sorting).has_extension("correlograms")

    # Second call: already present -> no recompute.
    again = sorting.add_extensions(populated_sorting, ["correlograms"])
    assert again == []

    # A base (sort-time) extension is never recomputed by add_extensions.
    assert sorting.add_extensions(populated_sorting, ["waveforms"]) == []

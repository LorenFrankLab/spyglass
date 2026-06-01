"""DB-gated test for A34's opt-in legacy-sorter back-compat helper.

``SorterParameters.insert_default_legacy_si_sorters()`` replicates v1's
auto-insert of ``('<sorter>', 'default')`` rows for SI sorters outside
v2's curated set, so v1 workflows that name a non-curated sorter via
``('kilosort2_5', 'default')`` keep resolving after porting.

This test requests ``dj_conn`` directly (function-scoped) so the rest of
the Phase 7 migration tests stay hermetic / DB-free.
"""

from __future__ import annotations

import pytest


def test_insert_default_legacy_si_sorters_skip_on_missing_sorter(dj_conn):
    """A34: the call is safe, idempotent, and lands a back-compat row.

    Asserts that after the opt-in call, an installed non-curated SI
    sorter has a ``('<sorter>', 'default')`` row, that the call raises
    nothing even though ``available_sorters()`` includes uninstalled
    sorters (those are caught + logged + skipped), and that a second
    call is a no-op (idempotent via ``skip_duplicates=True``).
    """
    import spikeinterface.sorters as sis

    from spyglass.spikesorting.v2._params.sorter import _SORTER_SCHEMAS
    from spyglass.spikesorting.v2.sorting import SorterParameters

    curated = set(_SORTER_SCHEMAS)
    candidates = sorted(set(sis.installed_sorters()) - curated)
    if not candidates:
        pytest.skip(
            "no installed non-curated SI sorter available to exercise A34"
        )

    # First call populates; second call must be a no-op (idempotent).
    SorterParameters.insert_default_legacy_si_sorters()
    SorterParameters.insert_default_legacy_si_sorters()

    present = [
        sorter
        for sorter in candidates
        if SorterParameters
        & {"sorter": sorter, "sorter_params_name": "default"}
    ]
    assert present, (
        "expected a ('<sorter>', 'default') row for at least one "
        f"installed non-curated sorter among {candidates}"
    )

    # The inserted row validates against the generic (extra='allow')
    # schema, so its params blob carries schema_version == 1.
    row = (
        SorterParameters
        & {"sorter": present[0], "sorter_params_name": "default"}
    ).fetch1()
    assert row["params_schema_version"] == 1
    assert row["params"]["schema_version"] == 1

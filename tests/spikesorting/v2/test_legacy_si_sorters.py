"""DB-gated test for the opt-in legacy-sorter back-compat helper.

``SorterParameters.insert_default_legacy_si_sorters()`` replicates v1's
auto-insert of ``('<sorter>', 'default')`` rows for SI sorters outside
v2's curated set, so v1 workflows that name a non-curated sorter via
``('kilosort2_5', 'default')`` keep resolving after porting.

This test requests ``dj_conn`` directly (function-scoped) so the rest of
the migration tests stay hermetic / DB-free.
"""

from __future__ import annotations

import pytest


def test_insert_default_legacy_si_sorters_skip_on_missing_sorter(dj_conn):
    """The call is safe, idempotent, and lands a back-compat row.

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
            "no installed non-curated SI sorter available to exercise "
            "the legacy back-compat helper"
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


def test_insert_default_legacy_si_sorters_skips_not_installed(
    dj_conn, monkeypatch
):
    """An available-but-not-installed sorter gets no 'default' row.

    ``get_default_sorter_params`` succeeds for wrapper-only sorters whose
    binary is absent, so the helper must gate on ``installed_sorters()``
    (mirroring ``insert_default``) -- otherwise it would ship a row that
    fails at ``Sorting.populate`` time. A fictitious sorter name proves
    the gate fires *before* ``get_default_sorter_params`` is consulted
    (a real call for that name would raise).
    """
    import spikeinterface.sorters as sis

    from spyglass.spikesorting.v2.sorting import SorterParameters

    fake = "definitely_not_installed_sorter_xyz"
    monkeypatch.setattr(sis, "available_sorters", lambda: [fake])
    monkeypatch.setattr(sis, "installed_sorters", lambda: [])

    # Must not raise even though the sorter is available-but-not-installed.
    SorterParameters.insert_default_legacy_si_sorters()

    assert not (
        SorterParameters & {"sorter": fake, "sorter_params_name": "default"}
    ), "available-but-not-installed sorter must not get a 'default' row"


def test_insert_default_legacy_si_sorters_backfills_local_execution(
    dj_conn, monkeypatch, request
):
    """A legacy back-compat row inserts and backfills local execution_params.

    The helper builds its row WITHOUT execution_params, so the insert hook must
    backfill the default (local) execution blob. Driven deterministically with a
    fake INSTALLED non-curated sorter so the test does not depend on which SI
    sorters happen to be installed -- which is also why the happy-path test above
    can skip. A regression that built a wrong-length positional row (e.g. omitting
    the execution_params columns) would fail the validated insert here.
    """
    import spikeinterface.sorters as sis

    from spyglass.spikesorting.v2.sorting import SorterParameters

    fake = "fake_legacy_sorter_xyz"
    monkeypatch.setattr(sis, "available_sorters", lambda: [fake])
    monkeypatch.setattr(sis, "installed_sorters", lambda: [fake])
    monkeypatch.setattr(sis, "get_default_sorter_params", lambda _sorter: {})
    request.addfinalizer(
        lambda: (
            SorterParameters & {"sorter": fake, "sorter_params_name": "default"}
        ).delete(safemode=False)
    )

    SorterParameters.insert_default_legacy_si_sorters()

    row = (
        SorterParameters & {"sorter": fake, "sorter_params_name": "default"}
    ).fetch1()
    assert row["params"]["schema_version"] == 1
    assert row["execution_params"]["backend"] == "local"
    assert row["execution_params"]["container_image"] is None
    assert int(row["execution_params_schema_version"]) == 1

"""DB-gated test for the opt-in legacy-sorter back-compat helper.

``SorterParameters.insert_default_legacy_si_sorters()`` inserts v1-style
``('<sorter>', 'default')`` rows for SI sorters outside v2's curated set.
"""

from __future__ import annotations

import pytest


def test_insert_default_legacy_si_sorters_skip_on_missing_sorter(dj_conn):
    """The call is safe, idempotent, and lands a back-compat row.

    Uninstalled sorters in ``available_sorters()`` are skipped without
    raising.
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

    The helper gates on ``installed_sorters()`` before the wrapper lookup;
    a fictitious sorter name proves the order.
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

    Uses a fake installed sorter so the result does not depend on which SI
    sorters are installed.
    """
    import spikeinterface.sorters as sis

    from spyglass.spikesorting.v2.sorting import SorterParameters

    fake = "fake_legacy_sorter_xyz"

    class FakeSorter:
        """Stand-in SI wrapper with an empty default-parameter dict."""

        sorter_name = fake

        @classmethod
        def _dynamic_params(cls):
            return {}, {}

    monkeypatch.setattr(sis, "available_sorters", lambda: [fake])
    monkeypatch.setattr(sis, "installed_sorters", lambda: [fake])
    # The helper reads wrappers from ``sis.sorter_dict``.
    monkeypatch.setitem(sis.sorter_dict, fake, FakeSorter)
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


def test_legacy_default_rows_exclude_job_kwargs(dj_conn, monkeypatch, request):
    """Seeded rows carry wrapper params only, no SI global job kwargs.

    SI's ``default_params()`` adds global job kwargs when
    ``requires_binary_data=True``; the insert guard would reject them.
    """
    import spikeinterface.sorters as sis
    from spikeinterface.sorters import sorterlist

    from spyglass.spikesorting.v2._params.sorter import (
        validate_sorter_params_against_wrapper,
    )
    from spyglass.spikesorting.v2.sorting import SorterParameters

    # One dict object behind both import paths, so a single setitem is
    # seen by the helper and by the wrapper-vocabulary lookup alike.
    assert sis.sorter_dict is sorterlist.sorter_dict

    fake = "fake_job_kwargs_sorter_xyz"

    class FakeSorter:
        """Stand-in for an SI wrapper with ``requires_binary_data=True``."""

        sorter_name = fake

        @classmethod
        def _dynamic_params(cls):
            return {"detect_threshold": 5}, {}

        @classmethod
        def default_params(cls):
            params, _ = cls._dynamic_params()
            return {**params, "n_jobs": 4, "chunk_duration": "1s"}

    monkeypatch.setattr(sis, "available_sorters", lambda: [fake])
    monkeypatch.setattr(sis, "installed_sorters", lambda: [fake])
    monkeypatch.setitem(sis.sorter_dict, fake, FakeSorter)
    request.addfinalizer(
        lambda: (
            SorterParameters & {"sorter": fake, "sorter_params_name": "default"}
        ).delete(safemode=False)
    )

    SorterParameters.insert_default_legacy_si_sorters()

    row = (
        SorterParameters & {"sorter": fake, "sorter_params_name": "default"}
    ).fetch1()
    assert "n_jobs" not in row["params"]
    assert "chunk_duration" not in row["params"]
    assert row["params"]["detect_threshold"] == 5
    # The seeded blob survives the guard that the insert itself applies.
    validate_sorter_params_against_wrapper(fake, row["params"])

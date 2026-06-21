"""SorterParameters schema and insert-default behavior.

Covers the MS4 ``adjacency_radius`` sentinel, install-gating on the
``insert_default()`` path, ``params_schema_version`` backfill and drift, and
unknown-sorter-name rejection at insert time.
"""

from __future__ import annotations

import pytest

from spyglass.spikesorting.v2._params.sorter import MountainSort4Schema


def test_ms4_schema_accepts_adjacency_radius_minus_one():
    """``adjacency_radius=-1`` is SI's "use all channels" sentinel.

    SpikeInterface's MountainSort4 wrapper documents ``adjacency_radius=-1``
    as "use all channels in the adjacency graph". The earlier
    ``Field(ge=0.0)`` floor rejected the sentinel, so a v1 user porting
    ``{"adjacency_radius": -1}`` was blocked at insert time. ``ge=-1.0``
    plus a validator that rejects the open interval ``(-1, 0)`` accepts
    the sentinel and any non-negative radius while still catching
    nonsensical negatives.
    """
    assert MountainSort4Schema(adjacency_radius=-1).adjacency_radius == -1.0
    assert MountainSort4Schema(adjacency_radius=0).adjacency_radius == 0.0
    assert MountainSort4Schema(adjacency_radius=75.0).adjacency_radius == 75.0

    # The open interval (-1, 0) is meaningless and must be rejected.
    for bad in (-0.5, -0.999, -0.0001):
        with pytest.raises(ValueError, match="adjacency_radius"):
            MountainSort4Schema(adjacency_radius=bad)

    # Below the sentinel is also invalid.
    with pytest.raises(ValueError):
        MountainSort4Schema(adjacency_radius=-2.0)


@pytest.mark.usefixtures("dj_conn")
def test_sorter_parameters_skips_uninstalled_sorters(monkeypatch):
    """``insert_default()`` skips uninstalled SI sorters at INSERT time.

    The companion ``_gated_default_rows`` test pins the gating DECISION;
    this one pins the INSERT PATH -- that ``insert_default`` actually
    routes the decision into the table, not just computes it. A
    regression that made ``insert_default`` insert the full
    ``_DEFAULT_CONTENTS`` directly (bypassing the gate) would pass the
    decision test but fail here.

    With ``installed_sorters()`` forced empty, every SI-registered default
    row is gated out while the Spyglass-internal ``clusterless_thresholder``
    row (never gated) still inserts. The skip is asserted on default
    sorters genuinely absent from the box, whose rows are therefore
    guaranteed clean (no earlier test inserted them) -- if the gate
    regressed, those rows would appear.
    """
    import spikeinterface.sorters as sis

    from spyglass.spikesorting.v2.sorting import SorterParameters

    real_installed = set(sis.installed_sorters())
    si_default_sorters = {
        row[0]
        for row in SorterParameters._DEFAULT_CONTENTS
        if row[0] not in SorterParameters._NON_SI_SORTERS
    }
    # Canaries: SI default sorters genuinely absent from this box. We
    # delete their rows first (a clean slate the persistent test DB may
    # not give us -- deleting a default Lookup row for an uninstalled,
    # therefore unreferenced, sorter is safe) so that a row reappearing
    # after insert_default unambiguously means the gate was bypassed.
    clean_canaries = si_default_sorters - real_installed
    if not clean_canaries:
        pytest.skip(
            "every SI default sorter is installed; no clean canary row to "
            "verify the gated insert path against"
        )
    for sorter in clean_canaries:
        (SorterParameters & {"sorter": sorter}).delete(safemode=False)

    monkeypatch.setattr(sis, "installed_sorters", lambda: [])
    SorterParameters.insert_default()

    # Non-SI special case -> never gated -> always inserted.
    assert SorterParameters & {
        "sorter": "clusterless_thresholder",
        "sorter_params_name": "default",
    }, "clusterless default row must insert even with no SI sorter installed"

    # Every gated-out SI sorter stays absent: insert_default routed the
    # gating DECISION into the INSERT, it did not insert _DEFAULT_CONTENTS
    # wholesale.
    for sorter in clean_canaries:
        assert not (SorterParameters & {"sorter": sorter}), (
            f"insert_default inserted uninstalled SI sorter {sorter!r} "
            "despite installed_sorters()==[]; the install gate was "
            "bypassed on the insert path."
        )


@pytest.mark.usefixtures("dj_conn")
def test_sorter_parameters_backfills_missing_schema_version(request):
    """A custom row may omit ``params_schema_version``; it is backfilled.

    The column default is the sentinel 0 ("unspecified"). The validated
    ``params`` blob already carries the authoritative ``schema_version``
    (4 for clusterless), so ``insert`` backfills the outer column from it
    when the caller leaves it at 0 -- the user does not copy the number by
    hand. An explicitly-supplied version that DISAGREES with the blob still
    trips the drift guard.
    """
    from spyglass.spikesorting.v2.sorting import SorterParameters

    base = {
        "sorter": "clusterless_thresholder",
        "params": {"detect_threshold": 100.0, "threshold_unit": "uv"},
        "job_kwargs": None,
    }

    # The rows below are default-content forks under custom names (this test
    # exercises schema-version backfill, not the duplicate guard). Delete them
    # after the test so they cannot shadow ``clusterless_thresholder/default``
    # and poison a later ``insert_default()``.
    request.addfinalizer(
        lambda: (
            SorterParameters
            & [
                {
                    "sorter": "clusterless_thresholder",
                    "sorter_params_name": name,
                }
                for name in ("audit_a5_missing", "audit_a5_zero", "audit_a5_ok")
            ]
        ).delete(safemode=False)
    )

    def _stored_version(name):
        return (
            SorterParameters
            & {
                "sorter": "clusterless_thresholder",
                "sorter_params_name": name,
            }
        ).fetch1("params_schema_version")

    # The three rows are intentionally the SAME content under different names
    # (this test exercises schema-version backfill, not the duplicate-content
    # guard), so each opts out of the guard explicitly.
    # Omitted entirely -> backfilled from the blob's schema_version (4).
    omitted = dict(base, sorter_params_name="audit_a5_missing")
    SorterParameters().insert1(
        omitted, skip_duplicates=True, allow_duplicate_params=True
    )
    assert _stored_version("audit_a5_missing") == 4

    # Explicit sentinel 0 is backfilled the same way.
    sentinel = dict(
        base, sorter_params_name="audit_a5_zero", params_schema_version=0
    )
    SorterParameters().insert1(
        sentinel, skip_duplicates=True, allow_duplicate_params=True
    )
    assert _stored_version("audit_a5_zero") == 4

    # The correct explicit version inserts cleanly and is preserved.
    ok = dict(base, sorter_params_name="audit_a5_ok", params_schema_version=4)
    SorterParameters().insert1(
        ok, skip_duplicates=True, allow_duplicate_params=True
    )
    assert _stored_version("audit_a5_ok") == 4

    # A wrong (non-sentinel) version that disagrees with the blob still
    # trips the drift check -- backfill only fills the sentinel 0.
    drift = dict(
        base, sorter_params_name="audit_a5_drift", params_schema_version=2
    )
    with pytest.raises(ValueError, match="schema_version"):
        SorterParameters().insert1(drift, skip_duplicates=True)


@pytest.mark.usefixtures("dj_conn")
def test_sorter_parameters_rejects_unknown_sorter_name(request):
    """A typo'd sorter name is rejected at insert, not at populate time.

    ``_get_sorter_schema`` falls back to the permissive generic schema for
    any unknown sorter, so without the guard a typo like ``"mountainSort4"``
    (capital S) would validate cleanly here and fail only later inside
    ``Sorting.populate`` with an opaque SI "sorter not registered" error.
    The insert guard rejects it up front with a message naming the bad
    sorter, while real registered sorters still insert -- including the
    v1 "try any installed SI sorter" escape hatch (a sorter that is in
    ``sis.available_sorters()`` but NOT in the curated ``_SORTER_SCHEMAS``
    set, dispatched through ``GenericSorterParamsSchema``).
    """
    from spyglass.spikesorting.v2.sorting import SorterParameters

    typo = {
        "sorter": "mountainSort4",  # capital S -> not a real SI sorter
        "sorter_params_name": "audit_a5_typo",
        "params": {},
        "job_kwargs": None,
    }
    with pytest.raises(ValueError, match="mountainSort4"):
        SorterParameters().insert1(typo, skip_duplicates=True)

    # A curated v2 sorter is accepted (covered by the _SORTER_SCHEMAS term
    # of the valid_sorters union).
    ok = {
        "sorter": "tridesclous2",
        "sorter_params_name": "audit_a5_real",
        "params": {},
        "job_kwargs": None,
    }
    # The bare-default content matches the shipped ``tridesclous2`` "default"
    # row; this test checks sorter-name acceptance, not the duplicate guard.
    SorterParameters().insert1(
        ok, skip_duplicates=True, allow_duplicate_params=True
    )
    # Delete it after the test (even if a later assertion fails): this is a
    # same-content fork under a custom name, and if left behind it shadows
    # ``tridesclous2/default``. A later ``SorterParameters.insert_default()``
    # on a box where tridesclous2 is installed would then raise
    # ``DuplicateParameterContentError`` -- poisoning every downstream test
    # that installs defaults.
    request.addfinalizer(
        lambda: (
            SorterParameters
            & {"sorter": "tridesclous2", "sorter_params_name": "audit_a5_real"}
        ).delete(safemode=False)
    )
    assert SorterParameters & {
        "sorter": "tridesclous2",
        "sorter_params_name": "audit_a5_real",
    }

    # The escape hatch: ``simple`` is a real SI sorter that is NOT in
    # ``_SORTER_SCHEMAS`` and NOT in ``_NON_SI_SORTERS``, so it is accepted
    # ONLY via the ``sis.available_sorters()`` term of the valid_sorters
    # union and validated by ``GenericSorterParamsSchema``. A regression
    # that dropped ``available_sorters()`` from the union would still pass
    # the curated/clusterless cases but break this insert. Its
    # ``params_schema_version`` is backfilled from the generic blob (1),
    # exercising backfill from a non-clusterless sorter.
    escape_hatch = {
        "sorter": "simple",
        "sorter_params_name": "audit_a5_escape_hatch",
        "params": {},
        "job_kwargs": None,
    }
    SorterParameters().insert1(escape_hatch, skip_duplicates=True)
    assert (
        SorterParameters
        & {"sorter": "simple", "sorter_params_name": "audit_a5_escape_hatch"}
    ).fetch1("params_schema_version") == 1

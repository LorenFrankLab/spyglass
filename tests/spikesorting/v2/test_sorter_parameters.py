"""SorterParameters schema and insert-default behavior.

Covers the MS4 ``adjacency_radius`` sentinel, install-gating on the
``insert_default()`` path, ``params_schema_version`` backfill and drift, and
unknown-sorter-name rejection at insert time.
"""

from __future__ import annotations

import pytest

from spyglass.spikesorting.v2._params.sorter import MountainSort4Schema


def test_ms4_schema_accepts_adjacency_radius_minus_one():
    """MS4 accepts ``adjacency_radius=-1`` (SI's "use all channels" sentinel)
    and non-negative radii; other negatives are rejected.
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
    """``insert_default()`` inserts no rows for uninstalled SI sorters.

    With ``installed_sorters()`` empty, local SI default rows are skipped
    while ``clusterless_thresholder`` (never gated) inserts.
    """
    import spikeinterface.sorters as sis

    from spyglass.spikesorting.v2._sorting.dispatch import is_container_backend
    from spyglass.spikesorting.v2.sorting import SorterParameters

    real_installed = set(sis.installed_sorters())
    # Canaries: local default rows of sorters absent from this machine
    # (unreferenced, so safe to delete first). Container rows are never gated.
    clean_canaries = [
        {"sorter": row[0], "sorter_params_name": row[1]}
        for row in SorterParameters._DEFAULT_CONTENTS
        if row[0] not in SorterParameters._NON_SI_SORTERS
        and row[0] not in real_installed
        and not is_container_backend(row[5])
    ]
    if not clean_canaries:
        pytest.skip(
            "every SI default sorter is installed; no clean canary row to "
            "verify the gated insert path against"
        )
    for canary in clean_canaries:
        (SorterParameters & canary).delete(safemode=False)

    monkeypatch.setattr(sis, "installed_sorters", lambda: [])
    SorterParameters.insert_default()

    # Non-SI special case -> never gated -> always inserted.
    assert SorterParameters & {
        "sorter": "clusterless_thresholder",
        "sorter_params_name": "default",
    }, "clusterless default row must insert even with no SI sorter installed"

    # Every gated-out SI sorter row stays absent.
    for canary in clean_canaries:
        assert not (SorterParameters & canary), (
            f"insert_default inserted uninstalled SI sorter row {canary!r} "
            "despite installed_sorters()==[]; the install gate was "
            "bypassed on the insert path."
        )


@pytest.mark.usefixtures("dj_conn")
def test_sorter_parameters_backfills_missing_schema_version(request):
    """An omitted or 0 ``params_schema_version`` is backfilled from the
    blob's ``schema_version``; an explicit mismatch raises.
    """
    from spyglass.spikesorting.v2.sorting import SorterParameters

    base = {
        "sorter": "clusterless_thresholder",
        "params": {"detect_threshold": 100.0, "threshold_unit": "uv"},
        "job_kwargs": None,
    }

    # Same-content forks would shadow ``clusterless_thresholder/default``
    # and break a later ``insert_default()``.
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

    # Same content under three names, so each opts out of the duplicate guard.
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

    # A wrong non-sentinel version trips the drift check.
    drift = dict(
        base, sorter_params_name="audit_a5_drift", params_schema_version=2
    )
    with pytest.raises(ValueError, match="schema_version"):
        SorterParameters().insert1(drift, skip_duplicates=True)


@pytest.mark.usefixtures("dj_conn")
def test_sorter_parameters_rejects_unknown_sorter_name(request):
    """A typo'd sorter name is rejected at insert, naming the sorter.

    Curated sorters and any other ``sis.available_sorters()`` sorter (via
    ``GenericSorterParamsSchema``) still insert.
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

    # A curated v2 sorter (in ``_SORTER_SCHEMAS``) is accepted.
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
    # A leftover fork would make a later ``insert_default()`` raise
    # ``DuplicateParameterContentError``.
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

    # ``simple`` is accepted only via ``sis.available_sorters()`` and
    # validated by ``GenericSorterParamsSchema`` (schema version 1).
    escape_hatch = {
        "sorter": "simple",
        "sorter_params_name": "audit_a5_escape_hatch",
        "params": {},
        "job_kwargs": None,
    }
    request.addfinalizer(
        lambda: (
            SorterParameters
            & {
                "sorter": "simple",
                "sorter_params_name": "audit_a5_escape_hatch",
            }
        ).delete(safemode=False)
    )
    SorterParameters().insert1(escape_hatch, skip_duplicates=True)
    assert (
        SorterParameters
        & {"sorter": "simple", "sorter_params_name": "audit_a5_escape_hatch"}
    ).fetch1("params_schema_version") == 1


@pytest.mark.usefixtures("dj_conn")
def test_sorter_parameters_update1_rejected_in_place(request):
    """``update1`` of a param row is rejected unless
    ``allow_param_mutation=True``; the name is part of ``sorting_id``.
    """
    import datajoint as dj

    from spyglass.spikesorting.v2.sorting import SorterParameters

    key = {
        "sorter": "clusterless_thresholder",
        "sorter_params_name": "audit_immutable_guard",
    }
    request.addfinalizer(
        lambda: (SorterParameters & key).delete(safemode=False)
    )
    SorterParameters().insert1(
        {
            **key,
            "params": {"detect_threshold": 100.0, "threshold_unit": "uv"},
            "job_kwargs": None,
        },
        allow_duplicate_params=True,
    )

    # Default path: an in-place blob edit is rejected and the row is unchanged.
    with pytest.raises(dj.errors.DataJointError, match="not supported"):
        SorterParameters().update1(
            {
                **key,
                "params": {"detect_threshold": 50.0, "threshold_unit": "uv"},
            }
        )
    assert (SorterParameters & key).fetch1("params")[
        "detect_threshold"
    ] == 100.0

    # Explicit escape hatch: the deliberate mutation goes through.
    SorterParameters().update1(
        {**key, "params": {"detect_threshold": 50.0, "threshold_unit": "uv"}},
        allow_param_mutation=True,
    )
    assert (SorterParameters & key).fetch1("params")["detect_threshold"] == 50.0


@pytest.mark.usefixtures("dj_conn")
def test_sorter_parameters_tracks_execution_params(request):
    """SorterParameters carries validated execution_params + schema version.

    Local and container rows with identical params coexist, since the
    duplicate-content fingerprint includes execution_params; a second name
    for the same params + execution is rejected.
    """
    from spyglass.spikesorting.v2.exceptions import (
        DuplicateParameterContentError,
    )
    from spyglass.spikesorting.v2.sorting import SorterParameters

    # adjacency_radius=77 is a custom value no shipped row uses, so these rows
    # cannot collide with the catalog's MS4 rows.
    params = {"adjacency_radius": 77.0}
    names = (
        "exec_test_local",
        "exec_test_singularity",
        "exec_test_local_dup",
    )
    request.addfinalizer(
        lambda: (
            SorterParameters
            & [
                {"sorter": "mountainsort4", "sorter_params_name": n}
                for n in names
            ]
        ).delete(safemode=False)
    )

    # Local row: execution_params omitted -> backfilled to default local.
    SorterParameters().insert1(
        {
            "sorter": "mountainsort4",
            "sorter_params_name": "exec_test_local",
            "params": dict(params),
            "job_kwargs": None,
        },
        skip_duplicates=True,
    )
    # Containerized row: SAME scientific params, distinct name + execution.
    SorterParameters().insert1(
        {
            "sorter": "mountainsort4",
            "sorter_params_name": "exec_test_singularity",
            "params": dict(params),
            "job_kwargs": None,
            "execution_params": {
                "backend": "singularity",
                "container_image": "my-image.sif",
            },
        },
        skip_duplicates=True,
    )

    local_exec, local_ver = (
        SorterParameters
        & {"sorter": "mountainsort4", "sorter_params_name": "exec_test_local"}
    ).fetch1("execution_params", "execution_params_schema_version")
    assert local_exec["backend"] == "local"
    assert local_exec["container_image"] is None
    assert int(local_ver) == 1

    cont_exec, cont_ver = (
        SorterParameters
        & {
            "sorter": "mountainsort4",
            "sorter_params_name": "exec_test_singularity",
        }
    ).fetch1("execution_params", "execution_params_schema_version")
    assert cont_exec["backend"] == "singularity"
    assert cont_exec["container_image"] == "my-image.sif"
    assert int(cont_ver) == 1

    # A SECOND name for the same (params + local execution) forks provenance.
    with pytest.raises(DuplicateParameterContentError):
        SorterParameters().insert1(
            {
                "sorter": "mountainsort4",
                "sorter_params_name": "exec_test_local_dup",
                "params": dict(params),
                "job_kwargs": None,
            },
            skip_duplicates=True,
        )


@pytest.mark.usefixtures("dj_conn")
def test_execution_params_schema_version_drift_rejected(request):
    """An explicit execution_params_schema_version that disagrees with the blob
    is rejected; an omitted column is backfilled from the blob.
    """
    from spyglass.spikesorting.v2.sorting import SorterParameters

    request.addfinalizer(
        lambda: (
            SorterParameters
            & {
                "sorter": "mountainsort4",
                "sorter_params_name": "exec_version_drift",
            }
        ).delete(safemode=False)
    )

    # Inner blob schema_version is 1; an explicit outer 2 must trip the guard.
    with pytest.raises(ValueError, match="execution_params_schema_version"):
        SorterParameters().insert1(
            {
                "sorter": "mountainsort4",
                "sorter_params_name": "exec_version_drift",
                "params": {"adjacency_radius": 71.0},
                "job_kwargs": None,
                "execution_params": {
                    "backend": "docker",
                    "container_image": "img:1",
                },
                "execution_params_schema_version": 2,
            },
            skip_duplicates=True,
            allow_duplicate_params=True,
        )


@pytest.mark.usefixtures("dj_conn")
def test_container_kwargs_not_allowed_in_sorter_params(request):
    """Reserved execution keys are rejected from params AND job_kwargs.

    Holds for strict (mountainsort4) and permissive (kilosort4,
    spykingcircus2) schemas. Container backend belongs on ``execution_params``.
    """
    from spyglass.spikesorting.v2.sorting import SorterParameters

    request.addfinalizer(
        lambda: (
            SorterParameters
            & [
                {"sorter": s, "sorter_params_name": n}
                for s, n in (
                    ("mountainsort4", "reserved_strict"),
                    ("kilosort4", "reserved_permissive"),
                    ("spykingcircus2", "reserved_jobkwargs"),
                )
            ]
        ).delete(safemode=False)
    )

    # Strict schema: Pydantic or the guard raises a ValueError naming the key.
    with pytest.raises(ValueError, match="docker_image"):
        SorterParameters().insert1(
            {
                "sorter": "mountainsort4",
                "sorter_params_name": "reserved_strict",
                "params": {"adjacency_radius": 100.0, "docker_image": "x"},
                "job_kwargs": None,
            },
            skip_duplicates=True,
        )

    # Permissive schema: the reserved-key guard catches what Pydantic allows.
    with pytest.raises(ValueError, match="singularity_image"):
        SorterParameters().insert1(
            {
                "sorter": "kilosort4",
                "sorter_params_name": "reserved_permissive",
                "params": {"singularity_image": "x"},
                "job_kwargs": None,
            },
            skip_duplicates=True,
        )

    # Reserved keys are also rejected from job_kwargs.
    with pytest.raises(ValueError, match="installation_mode"):
        SorterParameters().insert1(
            {
                "sorter": "spykingcircus2",
                "sorter_params_name": "reserved_jobkwargs",
                "params": {},
                "job_kwargs": {"installation_mode": "pypi"},
            },
            skip_duplicates=True,
        )


@pytest.mark.usefixtures("dj_conn")
def test_describe_pipeline_preset_surfaces_execution_params():
    """describe_pipeline_preset(name) shows the sorter row's execution_params
    under a ``sorter_execution`` stage.
    """
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.pipeline import describe_pipeline_preset

    initialize_v2_defaults()
    detail = describe_pipeline_preset(
        "franklab_probe_hippocampus_30khz_ms4_singularity_2026_06"
    )
    exec_rows = detail[detail["stage"] == "sorter_execution"]
    kv = dict(zip(exec_rows["key"], exec_rows["value"]))
    assert kv["backend"] == "singularity"
    assert kv["container_image"].startswith(
        "spikeinterface/mountainsort4-base:"
    )
    assert kv["installation_mode"] == "pypi"


@pytest.mark.usefixtures("dj_conn")
def test_container_ms4_default_row_inserts_without_local_ms4(monkeypatch):
    """Container MS4 rows insert when local MS4 is not installed; local MS4
    rows are skipped.
    """
    import spikeinterface.sorters as sis

    from spyglass.spikesorting.v2._core.recipe_catalog import (
        MS4_30KHZ,
        MS4_SINGULARITY_30KHZ,
    )
    from spyglass.spikesorting.v2.sorting import SorterParameters

    real_installed = set(sis.installed_sorters())
    monkeypatch.setattr(
        sis,
        "installed_sorters",
        lambda: sorted(real_installed - {"mountainsort4"}),
    )

    insertable, skipped = SorterParameters._gated_default_rows()
    insertable_names = {(r[0], r[1]) for r in insertable}
    skipped_names = {(r[0], r[1]) for r in skipped}

    # Local MS4 is skipped (its runtime is unavailable on this box).
    assert ("mountainsort4", MS4_30KHZ) in skipped_names
    # The container row is checked by preflight at run time instead.
    assert ("mountainsort4", MS4_SINGULARITY_30KHZ) in insertable_names


@pytest.mark.usefixtures("dj_conn")
def test_clusterless_rejects_container_backend():
    """A clusterless ``SorterParameters`` row cannot claim a container
    backend; it runs in-process.
    """
    from spyglass.spikesorting.v2.sorting import SorterParameters

    row = {
        "sorter": "clusterless_thresholder",
        "sorter_params_name": "clusterless_container_reject",
        "params": {"detect_threshold": 100.0, "threshold_unit": "uv"},
        "job_kwargs": None,
        "execution_params": {
            "backend": "docker",
            "container_image": "spikeinterface/test:latest",
        },
    }
    with pytest.raises(ValueError, match="runs in-process"):
        SorterParameters().insert1(
            row, skip_duplicates=True, allow_duplicate_params=True
        )
    assert not (
        SorterParameters
        & {
            "sorter": "clusterless_thresholder",
            "sorter_params_name": "clusterless_container_reject",
        }
    ), "a rejected clusterless container row must not be inserted"


@pytest.mark.usefixtures("dj_conn")
def test_legacy_seeder_skips_matlab(monkeypatch, request):
    """``insert_default_legacy_si_sorters`` does not create a local ``default``
    row for a MATLAB sorter.

    MATLAB sorters require a container backend. The sorter is forced
    available and installed so only the skip prevents the row.
    """
    import spikeinterface.sorters as sis

    from spyglass.spikesorting.v2._sorting.dispatch import MATLAB_SORTERS
    from spyglass.spikesorting.v2.sorting import SorterParameters

    matlab_sorter = MATLAB_SORTERS[0]  # e.g. "kilosort2_5"
    monkeypatch.setattr(sis, "available_sorters", lambda: [matlab_sorter])
    monkeypatch.setattr(sis, "installed_sorters", lambda: [matlab_sorter])
    monkeypatch.setattr(
        sis, "get_default_sorter_params", lambda s: {"detect_threshold": 6.0}
    )

    # Clean slate so a reappearing row means the skip failed.
    (SorterParameters & {"sorter": matlab_sorter}).delete(safemode=False)
    request.addfinalizer(
        lambda: (SorterParameters & {"sorter": matlab_sorter}).delete(
            safemode=False
        )
    )

    SorterParameters.insert_default_legacy_si_sorters()

    assert not (
        SorterParameters
        & {"sorter": matlab_sorter, "sorter_params_name": "default"}
    ), (
        f"legacy seeder must skip MATLAB sorter {matlab_sorter!r} -- a local "
        "default row is rejected by the dispatcher at populate time"
    )


@pytest.mark.usefixtures("dj_conn")
def test_is_seed_dependent_sort_flags_only_stochastic_paths():
    """SI sorters are always seed-dependent; the clusterless thresholder is
    only on its MAD path without explicit ``noise_levels`` (random chunks).
    """
    from spyglass.spikesorting.v2.sorting import SorterParameters

    dep = SorterParameters._is_seed_dependent_sort

    # SI clustering sorter: always seed-dependent.
    assert dep("mountainsort5", {}) is True

    # Clusterless MAD noise estimated from random chunks: seed-dependent
    # (whether ``noise_levels`` is absent or explicitly None).
    assert dep("clusterless_thresholder", {"threshold_unit": "mad"}) is True
    assert (
        dep(
            "clusterless_thresholder",
            {"threshold_unit": "mad", "noise_levels": None},
        )
        is True
    )

    # Clusterless deterministic paths: NOT seed-dependent.
    assert dep("clusterless_thresholder", {"threshold_unit": "uv"}) is False
    assert (
        dep(
            "clusterless_thresholder",
            {"threshold_unit": "mad", "noise_levels": [1.0, 2.0]},
        )
        is False
    )

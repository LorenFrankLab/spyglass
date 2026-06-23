"""Pure-unit coverage for the sorter execution backend (local vs container).

DB-free: imports only the ``_params.sorter`` schema/guards, the DB-free
``_sorting_dispatch`` builder/policy helpers, and the ``_recipe_catalog``
shipped rows. Covers the ``SorterExecutionParamsSchema`` validation rules, the
reserved-key rule, the ``run_sorter`` container-kwargs builder, the
MATLAB-sorter container policy, and that the shipped/recommended container rows
pin their container-side SpikeInterface runtime.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from spyglass.spikesorting.v2._params.sorter import (
    EXECUTION_RESERVED_KEYS,
    SorterExecutionParamsSchema,
    reject_reserved_execution_keys,
    validate_execution_params,
)
from spyglass.spikesorting.v2._sorting_dispatch import (
    MATLAB_SORTERS,
    assert_matlab_sorter_has_container_backend,
    build_run_sorter_container_kwargs,
    is_container_backend,
)

pytestmark = pytest.mark.unit


def test_sorter_execution_params_schema_local():
    """The default execution row is local; local rows reject container fields."""
    default = SorterExecutionParamsSchema().model_dump()
    assert default["backend"] == "local"
    assert default["container_image"] is None
    assert default["installation_mode"] == "auto"
    assert default["spikeinterface_version"] is None
    assert default["extra_requirements"] == []

    # A local row with a container image is contradictory.
    with pytest.raises(ValidationError, match="container_image=None"):
        SorterExecutionParamsSchema(backend="local", container_image="img:1")

    # Install controls are container-only; a non-default value on a local row
    # is rejected, not silently ignored.
    for kwargs in (
        {"installation_mode": "pypi"},
        {"spikeinterface_version": "0.104.3"},
        {"extra_requirements": ["foo"]},
    ):
        with pytest.raises(ValidationError, match="container-only"):
            SorterExecutionParamsSchema(backend="local", **kwargs)

    # Unknown keys are rejected (extra="forbid"); reserved execution keys can
    # therefore never hide inside the execution blob either.
    with pytest.raises(ValidationError):
        SorterExecutionParamsSchema(docker_image="img:1")

    # validate_execution_params(None) resolves to the default local row.
    assert validate_execution_params(None) == default
    assert validate_execution_params({}) == default


def test_sorter_execution_params_schema_container():
    """Container rows require an explicit non-empty image; True is rejected."""
    for backend in ("docker", "singularity"):
        row = SorterExecutionParamsSchema(
            backend=backend,
            container_image="spikeinterface/mountainsort4-base:0.104.3",
            installation_mode="pypi",
            spikeinterface_version="0.104.3",
        ).model_dump()
        assert row["backend"] == backend
        assert row["container_image"].endswith(":0.104.3")
        assert is_container_backend(row)

        # Missing / empty image -> clear error.
        with pytest.raises(ValidationError, match="non-empty container_image"):
            SorterExecutionParamsSchema(backend=backend)
        with pytest.raises(ValidationError, match="non-empty container_image"):
            SorterExecutionParamsSchema(
                backend=backend, container_image="   "
            )
        # SI accepts docker_image=True; v2 rejects True as untracked provenance.
        with pytest.raises(ValidationError):
            SorterExecutionParamsSchema(backend=backend, container_image=True)

    assert not is_container_backend(validate_execution_params(None))


def test_reject_reserved_execution_keys():
    """Every reserved execution key is rejected from a scientific blob."""
    assert EXECUTION_RESERVED_KEYS == frozenset(
        {
            "docker_image",
            "singularity_image",
            "delete_container_files",
            "installation_mode",
            "spikeinterface_version",
            "spikeinterface_folder_source",
            "extra_requirements",
        }
    )
    for key in EXECUTION_RESERVED_KEYS:
        with pytest.raises(ValueError, match=key):
            reject_reserved_execution_keys({key: "x"}, context="params blob")
    # No-op for None / a clean scientific blob.
    reject_reserved_execution_keys(None, context="params blob")
    reject_reserved_execution_keys({"adjacency_radius": 100.0}, context="x")


def test_build_run_sorter_container_kwargs():
    """The builder maps a validated execution row to SI run_sorter kwargs."""
    # Local -> no container kwargs at all.
    assert build_run_sorter_container_kwargs(validate_execution_params(None)) == {}

    docker = SorterExecutionParamsSchema(
        backend="docker",
        container_image="img:1",
        installation_mode="pypi",
        spikeinterface_version="0.104.3",
        extra_requirements=["pkg==1.0"],
    ).model_dump()
    kw = build_run_sorter_container_kwargs(docker)
    assert kw["docker_image"] == "img:1"
    assert "singularity_image" not in kw
    assert kw["delete_container_files"] is True
    assert kw["installation_mode"] == "pypi"
    assert kw["spikeinterface_version"] == "0.104.3"
    assert kw["extra_requirements"] == ["pkg==1.0"]

    # Singularity, no-install, no pinned version / extra reqs -> those omitted.
    sing = SorterExecutionParamsSchema(
        backend="singularity",
        container_image="img.sif",
        installation_mode="no-install",
    ).model_dump()
    kw = build_run_sorter_container_kwargs(sing)
    assert kw["singularity_image"] == "img.sif"
    assert "docker_image" not in kw
    assert kw["installation_mode"] == "no-install"
    assert "spikeinterface_version" not in kw
    assert "extra_requirements" not in kw


def test_matlab_sorter_container_policy():
    """MATLAB-backed sorters must run on a container backend, never local."""
    local = validate_execution_params(None)
    container = SorterExecutionParamsSchema(
        backend="singularity", container_image="img.sif"
    ).model_dump()
    for sorter in MATLAB_SORTERS:
        with pytest.raises(ValueError, match="container"):
            assert_matlab_sorter_has_container_backend(sorter, local)
        # A container backend passes.
        assert_matlab_sorter_has_container_backend(sorter, container)
    # Non-MATLAB sorters may run local.
    assert_matlab_sorter_has_container_backend("mountainsort4", local)


def test_recommended_container_rows_pin_si_runtime():
    """Shipped container sorter rows pin the container-side SI runtime.

    Either ``installation_mode="no-install"`` with a baked image, or
    ``installation_mode in {"pypi", "github"}`` with an explicit
    ``spikeinterface_version`` -- never the unpinned ``installation_mode="auto"``
    + ``spikeinterface_version=None`` combination, which drifts with the host.
    """
    from spyglass.spikesorting.v2._recipe_catalog import sorter_default_contents

    container_rows = [
        row
        for row in sorter_default_contents()
        if is_container_backend(row[5])
    ]
    assert container_rows, "no shipped container sorter rows found"
    for row in container_rows:
        execution = row[5]  # validated execution_params blob
        name = row[1]
        # Explicit, non-empty, pinned image (name:tag, not True / blank).
        image = execution["container_image"]
        assert image and ":" in image, f"{name} image not pinned: {image!r}"
        mode = execution["installation_mode"]
        if mode == "no-install":
            continue  # baked image carries the SI runtime
        assert mode in ("pypi", "github"), f"{name} uses non-pinned mode {mode!r}"
        assert execution["spikeinterface_version"], (
            f"{name} uses installation_mode={mode!r} without an explicit "
            "spikeinterface_version -- not reproducible by row content"
        )

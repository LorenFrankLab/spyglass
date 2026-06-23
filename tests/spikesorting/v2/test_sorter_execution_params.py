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

    # Container-only controls are rejected on a local row, not silently ignored
    # (a non-default value would be inert and would mislead a reader).
    for kwargs in (
        {"installation_mode": "pypi", "spikeinterface_version": "0.104.3"},
        {"spikeinterface_version": "0.104.3"},
        {"extra_requirements": ["foo"]},
        {"delete_container_files": False},
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
            container_image="example/img:1",  # a placeholder, not a real tag
            installation_mode="pypi",
            spikeinterface_version="0.104.3",
        ).model_dump()
        assert row["backend"] == backend
        assert row["container_image"] == "example/img:1"
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

        # pypi/github install SI into the container at run time, so they MUST
        # pin spikeinterface_version (else the install floats with the host).
        for mode in ("pypi", "github"):
            with pytest.raises(ValidationError, match="spikeinterface_version"):
                SorterExecutionParamsSchema(
                    backend=backend,
                    container_image="img:1",
                    installation_mode=mode,
                )
        # no-install (baked image) and auto (SI picks) legitimately leave the
        # version unset.
        SorterExecutionParamsSchema(
            backend=backend,
            container_image="img:1",
            installation_mode="no-install",
        )
        SorterExecutionParamsSchema(backend=backend, container_image="img:1")

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


def test_container_row_matches_local_sibling_science():
    """The containerized MS4 row is byte-identical SCIENCE to its local sibling.

    The container backend is the ONLY thing that differs: the scientific
    ``params`` of the shipped Singularity-30 kHz row must equal the local 30 kHz
    MS4 row (shared ``_MS4_RATE_PARAMS[30000]`` source). A regression that drifted
    one side would otherwise pass every other test.
    """
    from spyglass.spikesorting.v2._recipe_catalog import (
        MS4_30KHZ,
        MS4_20KHZ,
        MS4_SINGULARITY_30KHZ,
        sorter_default_contents,
    )

    # (sorter, name) -> params blob.
    params_by_name = {(r[0], r[1]): r[2] for r in sorter_default_contents()}
    local_30 = params_by_name[("mountainsort4", MS4_30KHZ)]
    assert params_by_name[("mountainsort4", MS4_SINGULARITY_30KHZ)] == local_30
    # The 30 vs 20 kHz local rows genuinely differ (so the parity is non-trivial).
    assert local_30 != params_by_name[("mountainsort4", MS4_20KHZ)]


def test_parameter_fingerprint_folds_execution_params():
    """execution_params changes the SorterParameters fingerprint but is omitted
    for the single-key Lookups.

    This is the property the duplicate-content guard relies on so a local and a
    containerized row with identical scientific params do NOT collide, while a
    Preprocessing/Artifact row's fingerprint stays byte-identical to the
    pre-execution-params behavior.
    """
    from spyglass.spikesorting.v2._parameter_identity import (
        parameter_fingerprint,
    )

    local = validate_execution_params(None)
    container = SorterExecutionParamsSchema(
        backend="singularity", container_image="img.sif"
    ).model_dump()
    common = dict(
        table="SorterParameters",
        params={"adjacency_radius": 100.0},
        params_schema_version=1,
        sorter="mountainsort4",
    )
    fp_local = parameter_fingerprint(
        **common, execution_params=local, execution_params_schema_version=1
    )
    fp_container = parameter_fingerprint(
        **common, execution_params=container, execution_params_schema_version=1
    )
    assert fp_local != fp_container

    # For a single-key Lookup, omitting execution_params (None) leaves the
    # fingerprint identical to not passing it at all.
    preproc = dict(
        table="PreprocessingParameters",
        params={"x": 1},
        params_schema_version=3,
    )
    assert parameter_fingerprint(**preproc) == parameter_fingerprint(
        **preproc, execution_params=None
    )


def test_reserved_keys_cover_schema_install_controls():
    """EXECUTION_RESERVED_KEYS is a superset of the schema's container-install
    controls (minus the schema-internal fields).

    A future install control added to ``SorterExecutionParamsSchema`` without a
    matching reserved key could be smuggled through a permissive sorter
    ``params`` blob. ``container_image`` maps to ``docker_image`` /
    ``singularity_image`` (both reserved); ``schema_version`` / ``backend`` are
    routing fields, not run_sorter kwargs.
    """
    schema_fields = set(SorterExecutionParamsSchema.model_fields)
    routing = {"schema_version", "backend", "container_image"}
    install_controls = schema_fields - routing
    missing = install_controls - EXECUTION_RESERVED_KEYS
    assert not missing, (
        f"schema install controls not in EXECUTION_RESERVED_KEYS: {missing}"
    )
    # The image maps to both run_sorter image kwargs, which are reserved.
    assert {"docker_image", "singularity_image"} <= EXECUTION_RESERVED_KEYS

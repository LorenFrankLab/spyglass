"""Sorter runtime checks: installed sorters, runtimes and container engines."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, TYPE_CHECKING
import importlib

from spyglass.spikesorting.v2._sorting.dispatch import (
    MATLAB_SORTERS,
    matlab_container_required_message,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


# SI's installed check only calls find_spec, which misses broken packages.
_SORTER_RUNTIME_BACKENDS: dict[str, tuple[str, ...]] = {
    "mountainsort4": ("mountainsort4",),
}


# The engine and Python-package probes ``run_sorter`` checks, with messages.
_CONTAINER_RUNTIME_PROBES: dict[
    str, tuple[tuple[tuple[str, str], ...], str]
] = {
    "docker": (
        (
            ("has_docker", "the Docker engine (`docker` CLI) was not found"),
            (
                "has_docker_python",
                "the Python `docker` package is not installed "
                "(`pip install docker`)",
            ),
        ),
        "Docker engine + Python `docker` package available",
    ),
    "singularity": (
        (
            ("has_singularity", "Singularity/Apptainer was not found"),
            (
                "has_spython",
                "the Python `spython` package is not installed "
                "(`pip install spython`)",
            ),
        ),
        "Singularity + Python `spython` package available",
    ),
}


def _container_runtime_available(backend: str) -> tuple[bool, str]:
    """Return ``(ok, detail)`` for a container execution backend's runtime.

    Requires the container engine and its Python package, as ``run_sorter``
    does.

    Parameters
    ----------
    backend : {"docker", "singularity"}
        The ``execution_params["backend"]`` of a container-backed row.

    Returns
    -------
    ok : bool
        True when both the engine and its Python package are available.
    detail : str
        What is missing, or what was found when ``ok``.
    """
    from spikeinterface.sorters import runsorter

    probes, available_detail = _CONTAINER_RUNTIME_PROBES[backend]
    for probe_name, missing_detail in probes:
        if not getattr(runsorter, probe_name)():
            return False, missing_detail
    return True, available_detail


# Memo for :func:`_installed_sorters_reused` (the probe takes seconds). ``None``
# outside a block, since installs can change in a long-lived kernel.
_installed_sorters_memo: ContextVar["dict[str, frozenset[str]] | None"] = (
    ContextVar("_installed_sorters_memo", default=None)
)


@contextmanager
def _installed_sorters_reused() -> "Iterator[None]":
    """Probe the installed sorters at most once within this block."""
    token = _installed_sorters_memo.set({})
    try:
        yield
    finally:
        _installed_sorters_memo.reset(token)


def _installed_sorters(sis) -> frozenset[str]:
    """Return ``sis.installed_sorters()``, reused inside a reuse block.

    Parameters
    ----------
    sis : module
        ``spikeinterface.sorters``.
    """
    memo = _installed_sorters_memo.get()
    if memo is None:
        return frozenset(sis.installed_sorters())
    if "names" not in memo:
        memo["names"] = frozenset(sis.installed_sorters())
    return memo["names"]


def _check_local_sorter_runtime(bundle, sis, non_si_sorters, check) -> None:
    """Run the LOCAL-execution sorter checks (installed + runtime backend).

    Parameters
    ----------
    bundle : _PipelinePreset
        The resolved preset whose ``sorter`` is being checked.
    sis : module
        ``spikeinterface.sorters``, imported lazily by each caller.
    non_si_sorters : Container[str]
        ``SorterParameters._NON_SI_SORTERS``, never gated on an SI binary.
    check : Callable[[str, Any, str], bool]
        The report's check-recording closure.
    """
    # Same gate as insert_default; non-SI sorters are always available.
    sorter_installed_ok = (
        bundle.sorter in non_si_sorters
        or bundle.sorter in _installed_sorters(sis)
    )
    if sorter_installed_ok:
        check("sorter_installed", True, "")
    elif bundle.sorter in set(sis.available_sorters()):
        check(
            "sorter_installed",
            False,
            f"sorter {bundle.sorter!r} is a known SpikeInterface sorter but its "
            "binary/runtime is not installed here "
            "(spikeinterface.sorters.installed_sorters()). Install it, or pick a "
            "preset whose sorter is installed. (Or use a containerized execution "
            "preset, which runs the sorter runtime inside a container image "
            "instead.)",
        )
    else:
        check(
            "sorter_installed",
            False,
            f"sorter {bundle.sorter!r} is not a known SpikeInterface sorter "
            "(spikeinterface.sorters.available_sorters()) -- check the spelling "
            "or the preset.",
        )

    # Installed packages can still fail to import a compiled dependency.
    backend_modules = _SORTER_RUNTIME_BACKENDS.get(bundle.sorter, ())
    if sorter_installed_ok and backend_modules:
        broken_backends = []
        for mod in backend_modules:
            try:
                importlib.import_module(mod)
            except (
                Exception
            ) as exc:  # noqa: BLE001 - any import failure disqualifies
                broken_backends.append(f"{mod} ({type(exc).__name__}: {exc})")
        check(
            "sorter_runtime_available",
            not broken_backends,
            f"sorter {bundle.sorter!r} is listed as installed but its runtime "
            "backend(s) cannot be imported, so the sort would crash: "
            f"{'; '.join(broken_backends)}. Install/repair the backend "
            '(for MountainSort4, install "spyglass-neuro[spikesorting-v2]"), '
            "or pick a "
            "preset whose sorter runs in this environment -- e.g. a MountainSort5 "
            "preset, or a containerized MountainSort4 preset whose runtime lives "
            "in the container.",
        )


def _check_sorter_execution(
    check: "Callable[[str, Any, str], bool]",
    warnings: list[str],
    bundle,
    pipeline_preset: "str | None",
    sorter_row: "dict | None",
) -> None:
    """Check the sorter can execute on the row's execution backend.

    The backend comes only from the row's ``execution_params``. With no row,
    the sorter name is still checked locally for its spelling hint. A
    runnable container backend appends an advisory to ``warnings``.
    """
    import spikeinterface.sorters as sis

    from spyglass.spikesorting.v2.sorting import SorterParameters

    if sorter_row is None:
        _check_local_sorter_runtime(
            bundle, sis, SorterParameters._NON_SI_SORTERS, check
        )
    else:
        from spyglass.spikesorting.v2._params.sorter import (
            validate_execution_params,
        )

        execution_params = validate_execution_params(
            sorter_row["execution_params"]
        )
        execution_backend = execution_params["backend"]
        container_image = execution_params["container_image"]

        if execution_backend == "local":
            # MATLAB sorters ship only as container images.
            if bundle.sorter.lower() in MATLAB_SORTERS:
                check(
                    "sorter_execution_backend",
                    False,
                    matlab_container_required_message(bundle.sorter),
                )
            else:
                _check_local_sorter_runtime(
                    bundle, sis, SorterParameters._NON_SI_SORTERS, check
                )
        else:
            # The sorter runtime lives in the image; check the container runtime.
            runtime_ok, runtime_detail = _container_runtime_available(
                execution_backend
            )
            preset_label = (
                "the preset"
                if pipeline_preset is None
                else f"pipeline_preset {pipeline_preset!r}"
            )
            check(
                "container_runtime_available",
                runtime_ok,
                f"{preset_label} selects the "
                f"{execution_backend} execution backend (image "
                f"{container_image!r}) for sorter {bundle.sorter!r}, but it is "
                f"not runnable here: {runtime_detail}. Install the container "
                "runtime and its Python package, or pick a local-execution "
                "preset. Preflight does not fall back to local execution.",
            )
            if runtime_ok:
                warnings.append(
                    f"{preset_label} runs sorter "
                    f"{bundle.sorter!r} inside the {execution_backend} image "
                    f"{container_image!r}: the host can stay on the v2 numpy>=2 "
                    "environment because the sorter runtime lives in the "
                    "container, not on the host."
                )

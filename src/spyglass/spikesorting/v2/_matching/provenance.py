"""Producer metadata for matching and input preparation, without database I/O."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from collections.abc import Mapping
from numbers import Integral

from spyglass.spikesorting.v2._core.runtime import (
    runtime_environment_provenance,
    validate_runtime_provenance,
)


def validate_matcher_provenance(values: Mapping) -> None:
    """Require producer identities while allowing unknown external versions."""
    if not isinstance(values, Mapping):
        raise ValueError("matcher provenance must be a mapping.")
    required = {"schema_version", "spyglass_version", "backend", "preparer"}
    missing = sorted(required - values.keys())
    if missing:
        raise ValueError(
            f"matcher provenance is missing required fields: {missing}."
        )
    schema = values["schema_version"]
    if (
        isinstance(schema, bool)
        or not isinstance(schema, Integral)
        or schema != 1
    ):
        raise ValueError("matcher provenance requires schema_version=1.")
    installed_version = values["spyglass_version"]
    if installed_version is not None and (
        not isinstance(installed_version, str) or not installed_version.strip()
    ):
        raise ValueError(
            "matcher provenance.spyglass_version must be a string or None."
        )
    for name in ("backend", "preparer"):
        component = values[name]
        if not isinstance(component, Mapping):
            raise ValueError(f"matcher provenance.{name} must be a mapping.")
        missing = sorted(
            {"qualified_name", "version", "fingerprints"} - component.keys()
        )
        if missing:
            raise ValueError(
                f"matcher provenance.{name} is missing required fields: {missing}."
            )
        if (
            not isinstance(component["qualified_name"], str)
            or not component["qualified_name"].strip()
        ):
            raise ValueError(
                f"matcher provenance.{name}.qualified_name must be a nonempty string."
            )
        component_version = component["version"]
        if component_version is not None and (
            not isinstance(component_version, str)
            or not component_version.strip()
        ):
            raise ValueError(
                f"matcher provenance.{name}.version must be a string or None."
            )
        fingerprints = component["fingerprints"]
        if not isinstance(fingerprints, Mapping) or any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            or not value
            for key, value in fingerprints.items()
        ):
            raise ValueError(
                f"matcher provenance.{name}.fingerprints must contain nonempty string names and fingerprints."
            )

    validate_runtime_provenance(values)


def spyglass_version() -> str | None:
    """Installed Spyglass version; unknown when distribution metadata is absent."""
    try:
        return version("spyglass-neuro")
    except PackageNotFoundError:
        return None


def _component_provenance(component, version_hook: str, params: dict) -> dict:
    component_type = type(component)
    qualified_name = (
        f"{component_type.__module__}.{component_type.__qualname__}"
    )
    get_version = getattr(component, version_hook, None)
    component_version = get_version() if get_version is not None else None
    if component_version is not None and not isinstance(component_version, str):
        raise TypeError(
            f"{qualified_name}.{version_hook}() must return str or None"
        )
    get_fingerprints = getattr(component, "provenance_fingerprints", None)
    fingerprints = (
        get_fingerprints(params) if get_fingerprints is not None else {}
    )
    if not isinstance(fingerprints, dict) or any(
        not isinstance(key, str)
        or not key
        or not isinstance(value, str)
        or not value
        for key, value in fingerprints.items()
    ):
        raise TypeError(
            f"{qualified_name}.provenance_fingerprints(params) must return "
            "a dict of nonempty string names and fingerprints"
        )
    return {
        "qualified_name": qualified_name,
        "version": component_version,
        "fingerprints": dict(fingerprints),
    }


def matcher_provenance(
    backend, preparer, params: dict, *, job_kwargs=None
) -> dict:
    """Snapshot the registered producers and their optional asset fingerprints.

    Versions and fingerprints are observational metadata, not run identity.
    Backends/preparers must also put result-determining asset fingerprints in
    their immutable params schema and verify the assets before using them.
    Missing version hooks record None; no plugin package version is guessed.
    Hook failures propagate rather than writing misleading provenance.
    """
    values = {
        "schema_version": 1,
        "spyglass_version": spyglass_version(),
        "backend": _component_provenance(backend, "backend_version", params),
        "preparer": _component_provenance(preparer, "preparer_version", params),
    }
    values.update(
        runtime_environment_provenance(
            job_kwargs=job_kwargs,
            execution_params={
                "stage": "matching",
                "backend": values["backend"]["qualified_name"],
                "preparer": values["preparer"]["qualified_name"],
            },
        )
    )
    validate_matcher_provenance(values)
    return values

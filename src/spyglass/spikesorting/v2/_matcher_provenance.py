"""Producer metadata for matching and input preparation, without database I/O."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version


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


def matcher_provenance(backend, preparer, params: dict) -> dict:
    """Snapshot the registered producers and their optional asset fingerprints.

    Versions and fingerprints are observational metadata, not run identity.
    Backends/preparers must also put result-determining asset fingerprints in
    their immutable params schema and verify the assets before using them.
    Missing version hooks record None; no plugin package version is guessed.
    Hook failures propagate rather than writing misleading provenance.
    """
    return {
        "schema_version": 1,
        "spyglass_version": spyglass_version(),
        "backend": _component_provenance(backend, "backend_version", params),
        "preparer": _component_provenance(preparer, "preparer_version", params),
    }

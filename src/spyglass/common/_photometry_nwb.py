"""DB-free helpers for fiber-photometry NWB traversal.

Pure functions over an in-memory ``NWBFile`` — no DataJoint, no table imports, no
``ndx_*`` imports (matching is by class-name string) — so they are unit-testable
without a database and safe to import from any schema module. Shared by
``common_photometry`` (reference-scoped device collection) and
``common_optogenetics`` (the fiber-table gate).
"""

from typing import Callable, List, Sequence, Set, Union

from pynwb import NWBFile

from spyglass.utils.nwb_helper_fn import is_nwb_obj_type


def photometry_tables(nwb_file: NWBFile) -> List:
    """Every ``FiberPhotometryTable`` object in the file (empty if none)."""
    return [
        obj
        for obj in nwb_file.objects.values()
        if is_nwb_obj_type(obj, "FiberPhotometryTable")
    ]


def is_photometry_file(nwb_file: NWBFile) -> bool:
    """True iff the file carries at least one ``FiberPhotometryTable``."""
    return len(photometry_tables(nwb_file)) > 0


def referenced_devices(
    nwb_file: NWBFile, column_names: Union[str, Sequence[str]]
) -> List:
    """Distinct device objects referenced by the given ``FiberPhotometryTable``
    column(s), deduped by ``name`` in first-seen order.

    Returns ``[]`` when the file has no ``FiberPhotometryTable`` (a clean no-op
    for non-photometry files). Defensive: a column absent from a table
    (e.g. a file predating the 0.2.3 shape) is skipped rather than raising, and a
    per-row null reference is ignored.
    """
    if isinstance(column_names, str):
        column_names = [column_names]

    by_name = {}
    for table in photometry_tables(nwb_file):
        df = table.to_dataframe()
        for col in column_names:
            if col not in df.columns:
                continue  # column not present on this (possibly older) table
            for value in df[col].tolist():
                name = getattr(value, "name", None)
                if name is not None and name not in by_name:
                    by_name[name] = value
    return list(by_name.values())


def optical_fiber_instances(nwb_file: NWBFile) -> List:
    """All ``OpticalFiber`` *instances* in the file (any modality)."""
    return [
        obj
        for obj in nwb_file.objects.values()
        if is_nwb_obj_type(obj, "OpticalFiber")
    ]


def photometry_fiber_names(nwb_file: NWBFile) -> Set[str]:
    """Names of ``OpticalFiber`` instances referenced by any photometry table."""
    return {f.name for f in referenced_devices(nwb_file, "optical_fiber")}


def model_attr(name: str) -> Callable:
    """Null-safe getter reading ``obj.model.<name>`` (``None`` if no model)."""

    def _get(obj):
        model = getattr(obj, "model", None)
        return getattr(model, name, None) if model is not None else None

    return _get


def model_range(name: str, index: int) -> Callable:
    """Null-safe getter for a ``[2]``-vector model spec, returning one endpoint
    as a scalar float (``None`` if the model or vector is absent)."""

    def _get(obj):
        model = getattr(obj, "model", None)
        vec = getattr(model, name, None) if model is not None else None
        if vec is None:
            return None
        return float(vec[index])

    return _get


def class_discriminator(subclass_to_value: dict, default: str) -> Callable:
    """Discriminator getter: the value for the first ``subclass_to_value`` class
    name the object matches (checked in dict order), else ``default``. Because the
    referenced object *is* the concrete subtype, this derives the enum without any
    installed ``ndx_*`` types."""

    def _get(obj):
        for subclass_name, value in subclass_to_value.items():
            if is_nwb_obj_type(obj, subclass_name):
                return value
        return default

    return _get


def populated_attrs(obj, candidate_names: Sequence[str]) -> List[str]:
    """Of ``candidate_names``, those set to a non-``None`` value on ``obj``."""
    found = []
    for name in candidate_names:
        if getattr(obj, name, None) is not None:
            found.append(name)
    return found

#!/usr/bin/env python3
"""
spyglass_ingestion_autodoc.py

Scan a Python package for SpyglassIngestion tables and auto-generate
documentation of their NWB ingestion mappings.

Outputs Markdown by default; can also emit JSON or CSV.

Example:
    python spyglass_ingestion_autodoc.py
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import io
import json
import pkgutil
import sys
import types
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import datajoint as dj

# ---------- Helpers to safely access @property without running __init__ ----------


def safe_new(cls):
    """Create an uninitialized instance (avoid dj.Table setup / __init__)."""
    try:
        return cls.__new__(cls)
    except Exception:
        # Fallback: plain object (property fget only needs 'self' for method lookup)
        return object.__new__(cls)


def get_property_value(cls, prop_name: str):
    """
    Call a @property (if present) on a dummy instance without invoking __init__.
    Returns (value, error_str) where error_str is None on success.
    """
    try:
        prop = getattr(cls, prop_name, None)
        if prop is None or not isinstance(prop, property):
            return None, f"missing property {prop_name}"
        inst = safe_new(cls)
        return prop.fget(inst), None
    except NotImplementedError as e:
        return None, f"NotImplementedError: {e}"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def callable_name(v: Any, owner: type) -> str:
    """Human-friendly name for callables in mapping."""
    try:
        # bound/unbound function defined on the class?
        if inspect.isfunction(v):
            # Prefer qualified name if it belongs to the owner class
            qual = (
                f"{owner.__name__}.{v.__name__}"
                if getattr(v, "__qualname__", "").startswith(owner.__name__)
                else v.__name__
            )
            return qual
        if inspect.ismethod(v):
            return f"{owner.__name__}.{v.__name__}"
        if hasattr(v, "__name__"):
            return v.__name__
        return repr(v)
    except Exception:
        return repr(v)


def is_spyglass_ingestion_subclass(cls) -> bool:
    """
    Identify SpyglassIngestion subclasses without hard-failing if the package isn't importable.
    We prefer issubclass against spyglass.utils.SpyglassIngestion when available; otherwise,
    duck-type on required properties.
    """
    try:
        from spyglass.utils import SpyglassIngestion  # type: ignore

        return (
            inspect.isclass(cls)
            and issubclass(cls, SpyglassIngestion)
            and cls is not SpyglassIngestion
        )
    except Exception:
        # Duck-typing fallback: has the two required @properties
        return (
            inspect.isclass(cls)
            and isinstance(
                getattr(cls, "table_key_to_obj_attr", None), property
            )
            and isinstance(
                getattr(cls, "_source_nwb_object_type", None), property
            )
        )


# ---------- Introspection across modules ----------


def iter_modules(package_name: str) -> Iterable[types.ModuleType]:
    """Import the package and yield all submodules (breadth-first)."""
    pkg = importlib.import_module(package_name)
    if not hasattr(pkg, "__path__"):
        yield pkg
        return

    # Monkeypatch the `datajoint.schema` to skip activation during crawl
    def no_op_activate(*args, **kwargs):
        pass

    dj.schema.activate = no_op_activate
    dj.Table.is_declared = property(lambda self: True)
    dj.Table.user_is_admin = property(lambda self: True)

    for modinfo in pkgutil.walk_packages(
        pkg.__path__, prefix=pkg.__name__ + "."
    ):
        name = modinfo.name
        try:
            yield importlib.import_module(name)
        except Exception:
            # Skip modules that fail to import; continue
            continue


def discover_ingestion_classes(package_name: str) -> List[type]:
    """Return all SpyglassIngestion subclasses found in package modules."""
    classes: List[type] = []
    for mod in iter_modules(package_name):
        for _, cls in inspect.getmembers(mod, inspect.isclass):
            # Only include classes defined in this module (avoid duplicates from re-exports)
            if cls.__module__ != mod.__name__:
                continue
            if is_spyglass_ingestion_subclass(cls):
                classes.append(cls)
    # Keep deterministic order: by module then class name
    classes.sort(key=lambda c: (c.__module__, c.__name__))
    return classes


# ---------- Documentation builders ----------


def extract_mapping_rows(cls: type) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    For a given SpyglassIngestion class, return:
      - list of rows describing mapping entries
      - list of warnings (if any)
    """
    rows: List[Dict[str, Any]] = []
    warnings: List[str] = []

    # _source_nwb_object_type
    source_type, err = get_property_value(cls, "_source_nwb_object_type")
    if err:
        warnings.append(f"{qualname(cls)}: _source_nwb_object_type -> {err}")
    source_type_name = (
        getattr(source_type, "__name__", str(source_type))
        if source_type is not None
        else "None"
    )

    # table_key_to_obj_attr
    mapping, err = get_property_value(cls, "table_key_to_obj_attr")
    if err:
        warnings.append(f"{qualname(cls)}: table_key_to_obj_attr -> {err}")
        mapping = {}

    if not isinstance(mapping, dict):
        warnings.append(f"{qualname(cls)}: table_key_to_obj_attr is not a dict")
        mapping = {}

    for obj_key, keymap in mapping.items():
        if not isinstance(keymap, dict):
            warnings.append(
                f"{qualname(cls)}: mapping for '{obj_key}' is not a dict"
            )
            continue
        for table_key, value in keymap.items():
            if isinstance(value, str):
                rows.append(
                    {
                        "module": cls.__module__,
                        "class": cls.__name__,
                        "source_nwb_object_type": source_type_name,
                        "object_selector": obj_key,
                        "table_key": table_key,
                        "maps_to": value,
                        "is_callable": False,
                        "callable_name": "",
                    }
                )
            else:
                rows.append(
                    {
                        "module": cls.__module__,
                        "class": cls.__name__,
                        "source_nwb_object_type": source_type_name,
                        "object_selector": obj_key,
                        "table_key": table_key,
                        "maps_to": callable_name(value, cls),
                        "is_callable": True,
                        "callable_name": callable_name(value, cls),
                    }
                )
    return rows, warnings


def qualname(cls: type) -> str:
    return f"{cls.__module__}.{cls.__name__}"


def as_markdown(rows: List[Dict[str, Any]]) -> str:
    """
    Render a single Markdown table, sorted for readability.
    """
    if not rows:
        return "_No SpyglassIngestion classes found._\n"

    cols = [
        "module",
        "class",
        "source_nwb_object_type",
        "object_selector",
        "table_key",
        "maps_to",
        "is_callable",
    ]
    # sort for stable output
    rows_sorted = sorted(
        rows,
        key=lambda r: (
            r["module"],
            r["class"],
            r["object_selector"],
            r["table_key"],
        ),
    )
    out = io.StringIO()
    out.write("# Spyglass Ingestion Mapping\n\n")
    out.write("| " + " | ".join(cols) + " |\n")
    out.write("|" + "|".join(["---"] * len(cols)) + "|\n")
    for r in rows_sorted:
        out.write("| " + " | ".join(str(r[c]) for c in cols) + " |\n")
    out.write("\n")
    return out.getvalue()


def main():
    classes = discover_ingestion_classes("spyglass")
    all_rows: List[Dict[str, Any]] = []
    all_warnings: List[str] = []
    for cls in classes:
        rows, warns = extract_mapping_rows(cls)
        all_rows.extend(rows)
        all_warnings.extend(warns)

    content = as_markdown(all_rows)
    out_path = Path("./docs/src/ForDevelopers/ingestion_mapping.md").resolve()
    with open(out_path, "w") as f:
        f.write(content)


main()

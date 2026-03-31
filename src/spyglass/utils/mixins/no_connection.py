"""NoConnectionMixin: safe fallbacks for every DB-touching method.

When a table class carries ``_no_connection = True`` (set by
``SpyglassSchema.__call__`` on activation failure), this mixin intercepts
every DataJoint method that would otherwise require a live database connection
and returns a safe, clearly-labelled empty result instead.

Two-tier visibility
-------------------
* **Loud** — emitted once by ``SpyglassSchema.activate()`` as a
  ``UserWarning``; not repeated here.
* **Quiet** — the offline notice is embedded in ``__repr__`` /
  ``_repr_html_`` / ``preview()`` output so users always see it when they
  inspect a table in a terminal or Jupyter notebook.

Usage
-----
Include before any DataJoint base class in the MRO so the overrides take
priority::

    class SpyglassMixin(NoConnectionMixin, CautiousDeleteMixin, ...):
        ...
"""

import re
import warnings
from functools import cached_property

import numpy as np
import pyparsing as pp
from datajoint.declare import attribute_parser, is_foreign_key
from datajoint.errors import DataJointError

# ---------------------------------------------------------------------------
# Definition parser — uses DataJoint's own grammar, no context required
# ---------------------------------------------------------------------------


def _parse_definition(definition):
    """Return ``(primary_keys, secondary_attrs)`` parsed from a DJ definition.

    Uses DataJoint's ``attribute_parser`` pyparsing grammar and
    ``is_foreign_key`` directly — the same logic as ``declare.prepare_declare``
    but without the ``context`` lookup that FK lines require (we skip those).
    This means FK-contributed column names are omitted from the display, which
    is acceptable since they can't be resolved without a DB connection.

    Parameters
    ----------
    definition : str
        Raw DataJoint table definition string.

    Returns
    -------
    primary_keys : list[tuple[str, str]]
        Each entry is ``(name, type_str)`` for a primary-key attribute.
    secondary_attrs : list[tuple[str, str]]
        Each entry is ``(name, type_str)`` for a secondary attribute.
    """
    primary_keys: list = []
    secondary_attrs: list = []
    in_secondary = False

    # Mirror prepare_declare's line-splitting strategy.
    for line in re.split(r"\s*\n\s*", definition.strip()):
        if not line or line.startswith("#"):
            continue

        if line.startswith("---") or line.startswith("___"):
            in_secondary = True
            continue

        if is_foreign_key(line):
            continue  # FK attrs can't be resolved offline; skip silently

        if re.match(r"^(unique\s+)?index\s*", line, re.I):
            continue  # index declarations — no attribute name to show

        try:
            # attribute_parser expects a trailing "#" (comment marker)
            match = attribute_parser.parseString(line + "#", parseAll=True)
            name = match["name"]
            # type field may include a default (e.g. "varchar(8) = 'x' ").
            # Strip trailing whitespace; keep default so users see the full spec.
            type_str = match["type"].strip()
        except pp.ParseException:
            continue  # skip any line that doesn't parse cleanly

        entry = (name, type_str)
        if in_secondary:
            secondary_attrs.append(entry)
        else:
            primary_keys.append(entry)

    return primary_keys, secondary_attrs


# ---------------------------------------------------------------------------
# Offline fetch callable
# ---------------------------------------------------------------------------


class _OfflineFetch:
    """Drop-in replacement for ``datajoint.fetch.Fetch`` in offline mode.

    Always returns an empty result in the format the caller expects and emits
    a ``UserWarning`` so callers that don't inspect the return value still see
    that no data was actually retrieved.
    """

    _WARN_MSG = (
        "[Spyglass] No database connection — fetch returning empty result."
    )

    def __call__(
        self,
        *attrs,
        as_dict=None,
        format=None,
        **kwargs,
    ):
        warnings.warn(self._WARN_MSG, UserWarning, stacklevel=2)

        from datajoint.settings import config as dj_config

        if as_dict or attrs:
            return []  # list-of-dicts / tuple style

        fmt = format or dj_config.get("fetch_format", "array")
        if fmt == "frame":
            import pandas as pd

            return pd.DataFrame()
        return np.array([])  # default: structured numpy array

    # Support attribute-style call: Table.fetch["attr"] (rare but valid)
    def __getitem__(self, item):
        warnings.warn(self._WARN_MSG, UserWarning, stacklevel=2)
        return []


# ---------------------------------------------------------------------------
# NoConnectionMixin
# ---------------------------------------------------------------------------


class NoConnectionMixin:
    """Override every DB-touching method to return safe empty results offline.

    This mixin is a no-op when ``self._no_connection`` is ``False`` (the
    normal connected case): all ``super()`` calls pass through unchanged.
    """

    _no_connection: bool = False

    # ------------------------------------------------------------------
    # is_declared — skip SHOW TABLES query
    # ------------------------------------------------------------------

    @property
    def is_declared(self):
        if self._no_connection:
            return False
        return super().is_declared

    # ------------------------------------------------------------------
    # Counting / existence
    # ------------------------------------------------------------------

    def __len__(self):
        if self._no_connection:
            return 0
        return super().__len__()

    def __bool__(self):
        if self._no_connection:
            return False
        return super().__bool__()

    # ------------------------------------------------------------------
    # restrict — &, -, ^ operators all call restrict()
    # ------------------------------------------------------------------

    def restrict(self, restriction):
        """Return self unchanged; restriction cannot be evaluated offline."""
        if self._no_connection:
            return self
        return super().restrict(restriction)

    # ------------------------------------------------------------------
    # fetch / fetch1
    # ------------------------------------------------------------------

    @property
    def fetch(self):
        if self._no_connection:
            return _OfflineFetch()
        return super().fetch

    @property
    def fetch1(self):
        if self._no_connection:
            raise DataJointError(
                f"{self.__class__.__name__} has no database connection. "
                "Cannot fetch1 from an offline table."
            )
        return super().fetch1

    # ------------------------------------------------------------------
    # preview / __repr__ / _repr_html_
    # ------------------------------------------------------------------

    def preview(self, limit=None, width=None):
        """:return: string preview; offline tables show column names only."""
        if self._no_connection:
            return self._offline_preview()
        return super().preview(limit=limit, width=width)

    def __repr__(self):
        if self._no_connection:
            return self._offline_preview()
        return super().__repr__()

    def _repr_html_(self):
        """:return: HTML preview; offline tables show a labelled empty table."""
        if self._no_connection:
            return self._offline_repr_html()
        return super()._repr_html_()

    # ------------------------------------------------------------------
    # Private offline display helpers
    # ------------------------------------------------------------------

    @cached_property
    def _offline_columns(self):
        """Parse column names from ``self.definition`` without touching the DB.

        Returns ``(primary_keys, secondary_attrs)`` as lists of strings.
        Falls back to empty lists when the definition is unavailable.
        """
        try:
            defn = self.definition
            if not isinstance(defn, str):
                return [], []
            return _parse_definition(defn)
        except (NotImplementedError, AttributeError):
            return [], []

    def _offline_preview(self):
        """Text preview for terminal / notebook ``repr``.

        Mirrors DataJoint's heading display: one ``name : type`` line per
        attribute, primary keys prefixed with ``*``, separated by ``---``.
        """
        pk, sec = self._offline_columns
        name = self.__class__.__name__
        db = getattr(self, "database", None)
        schema_label = f"{db}." if db else ""

        lines = [
            f"{schema_label}.{name} [no database connection]",
        ]
        if pk or sec:
            # Align the colon column for readability.
            all_entries = pk + sec
            max_name = max((len(n) for n, _ in all_entries), default=0)
            for col_name, col_type in pk:
                pad = " " * (max_name - len(col_name))
                lines.append(f" *{col_name}{pad} : {col_type}")
            if sec:
                lines.append("  ---")
                for col_name, col_type in sec:
                    pad = " " * (max_name - len(col_name))
                    lines.append(f"  {col_name}{pad} : {col_type}")
        else:
            lines.append(" (definition unavailable)")
        lines.append("")  # trailing newline
        return "\n".join(lines)

    def _offline_repr_html(self):
        """Minimal HTML table for Jupyter display.

        Shows one row per attribute (``name : type``), matching the structure
        a connected user would see for an empty table.
        """
        pk, sec = self._offline_columns
        name = self.__class__.__name__
        db = getattr(self, "database", None)
        schema_label = f" ({db})" if db else ""

        def _attr_row(col_name, col_type, is_pk):
            prefix = "*" if is_pk else "&nbsp;"
            weight = "bold" if is_pk else "normal"
            return (
                f"<tr>"
                f'<td style="font-weight:{weight};color:#bd93f9;">'
                f"{prefix}{col_name}</td>"
                f'<td style="color:#f8f8f2;">{col_type}</td>'
                f"</tr>"
            )

        if pk or sec:
            rows = "".join(_attr_row(n, t, True) for n, t in pk)
            if sec:
                rows += (
                    '<tr><td colspan="2" style="border-top:2px solid #44475a;">'
                    "</td></tr>"
                )
                rows += "".join(_attr_row(n, t, False) for n, t in sec)
        else:
            rows = (
                '<tr><td colspan="2" style="color:#6272a4;">'
                "(definition unavailable)</td></tr>"
            )

        return (
            f'<div style="max-width:800px;">'
            f"<p><b>{name}</b>{schema_label} "
            f'<em style="color:#ff5555;">[no database connection]</em></p>'
            f'<table border="1" style="border-collapse:collapse;">'
            f"<thead><tr>"
            f'<th style="color:#8be9fd;">attribute</th>'
            f'<th style="color:#8be9fd;">type</th>'
            f"</tr></thead>"
            f"<tbody>{rows}"
            f'<tr><td colspan="2" style="text-align:center;color:#6272a4;">'
            f"(no data \u2014 offline)</td></tr>"
            f"</tbody>"
            f"</table></div>"
        )

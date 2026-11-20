"""Insert guards and integrity adapters for v2 DataJoint tables.

Imports no schemas. Parameter validation and integrity queries run at the
explicit table boundary.
"""

from __future__ import annotations

import datajoint as dj

from spyglass.spikesorting.v2._core.lookup_validation import (
    _insert_row_to_dict,
    reject_duplicate_parameter_content,
    validate_lookup_rows,
)
from spyglass.spikesorting.v2.exceptions import SchemaBypassError
from spyglass.utils.mixins import immutable


def stored_source(table_name: str, row: dict) -> tuple[str, dict]:
    """Return the one source a selection row names.

    For selection masters that store a source as the nullable
    ``recording_id`` and ``concat_recording_id`` columns, exactly one set.

    Parameters
    ----------
    table_name : str
        The selection table's name, for the error message.
    row : dict
        The selection row (or the row about to be inserted).

    Returns
    -------
    kind : {"recording", "concatenated_recording"}
        Which column holds the source.
    source_key : dict
        ``{"recording_id": ...}`` or ``{"concat_recording_id": ...}``.

    Raises
    ------
    SchemaBypassError
        If the row names zero or both sources.
    """
    recording_id = row.get("recording_id")
    concat_recording_id = row.get("concat_recording_id")
    if (recording_id is None) == (concat_recording_id is None):
        raise SchemaBypassError(
            f"{table_name} row {row} has recording_id={recording_id} and "
            f"concat_recording_id={concat_recording_id}; exactly one must "
            f"be set. Use {table_name}.insert_selection() to add this "
            "selection."
        )
    if recording_id is not None:
        return "recording", {"recording_id": recording_id}
    return "concatenated_recording", {
        "concat_recording_id": concat_recording_id
    }


def foreign_key_referrers(
    parent, rows: list[dict], *, skip=frozenset()
) -> dict[str, int]:
    """Count rows in other tables whose foreign keys onto ``parent`` match ``rows``.

    The foreign keys come from ``information_schema``, so a referencing
    table counts even when its module is not imported in this process.

    Parameters
    ----------
    parent : dj.Table
        The referenced table.
    rows : list[dict]
        ``parent`` primary keys to look up.
    skip : Collection[str]
        Full names of referencing tables to leave out.

    Returns
    -------
    dict[str, int]
        Matching row count per referencing table, for tables with any.
    """
    if not rows:
        return {}
    schema, table = parent.full_table_name.replace("`", "").split(".")
    columns = parent.connection.query(
        "SELECT table_schema, table_name, constraint_name, column_name, "
        "referenced_column_name FROM information_schema.key_column_usage "
        "WHERE referenced_table_schema = %s AND referenced_table_name = %s",
        args=(schema, table),
    ).fetchall()
    foreign_keys: dict[tuple[str, str], dict[str, str]] = {}
    for child_schema, child_table, constraint, column, referenced in columns:
        name = f"`{child_schema}`.`{child_table}`"
        foreign_keys.setdefault((name, constraint), {})[column] = referenced
    counts: dict[str, int] = {}
    for (name, _), attr_map in foreign_keys.items():
        if name in skip:
            continue
        child = dj.FreeTable(parent.connection, name)
        restriction = [
            {column: row[referenced] for column, referenced in attr_map.items()}
            for row in rows
        ]
        n_rows = len(child & restriction)
        if n_rows:
            counts[name] = counts.get(name, 0) + n_rows
    return counts


class SelectionMasterInsertGuard(immutable._IdentityMasterGuard):
    """Reject a direct ``insert`` into a deterministic-id selection master.

    Only ``insert_selection`` holds the full logical identity the primary key
    is derived from. This catches the easy mistake early; integrity is
    enforced by PK uniqueness and the bypass checks. ``insert_selection``
    passes ``allow_direct_insert=True`` itself. A table that sets
    ``_xor_source = True`` also refuses any row that does not name exactly
    one of ``recording_id`` / ``concat_recording_id`` (:func:`stored_source`).
    """

    _xor_source: bool = False

    def _direct_insert_reason(self) -> str:
        return (
            "the primary key is derived from the selection's full logical "
            "identity. Use "
            f"{type(self).__name__}.insert_selection()."
        )

    def insert(self, rows, *args, allow_direct_insert=False, **kwargs):
        """Insert after the direct-insert guard and, if set, the XOR check.

        Raises
        ------
        SchemaBypassError
            If ``_xor_source`` is set and a row names zero or both sources.
        """
        if self._xor_source and allow_direct_insert:
            rows = [
                _insert_row_to_dict(row, self.heading.names) for row in rows
            ]
            for row in rows:
                stored_source(type(self).__name__, row)
        super().insert(
            rows, *args, allow_direct_insert=allow_direct_insert, **kwargs
        )

    def _update1_create_hint(self) -> str:
        return (
            f"Insert a new selection via {self.__class__.__name__}."
            "insert_selection() instead."
        )


class ImmutableParamsLookup(immutable.ImmutableParamsLookup):
    """The shared parameter-row guard plus v2's shipped-defaults installer."""

    def _insert_missing_defaults(self, rows) -> None:
        """Insert the shipped default rows whose key is not stored yet.

        A stored default keeps its content even if it diverged from the
        shipped one; ``verify_v2_default_catalog`` reports that divergence.

        Parameters
        ----------
        rows : iterable
            Shipped default rows, as mappings or positional sequences.
        """
        names = self.heading.names
        rows = [_insert_row_to_dict(row, names) for row in rows]
        stored = {
            tuple(row[field] for field in self.primary_key)
            for row in self.fetch(as_dict=True)
        }
        missing = [
            row
            for row in rows
            if tuple(row[field] for field in self.primary_key) not in stored
        ]
        # skip_duplicates tolerates a concurrent installer.
        self.insert(missing, skip_duplicates=True)


def _insert_parameter_rows(
    table,
    rows,
    *,
    insert_rows,
    table_name: str,
    name_attr: str,
    schema_for=None,
    per_row_hook=None,
    validate_rows=None,
    sorter_keyed: bool = False,
    matcher_keyed: bool = False,
    allow_duplicate_params: bool = False,
    **kwargs,
):
    """Validate a parameter batch, reject duplicate content, then insert once.

    ``insert_rows`` is the override's bound ``super().insert``, so the MRO
    guards still apply. Pass ``schema_for`` (and optionally ``per_row_hook``),
    or ``validate_rows(rows, names)`` for custom batch validation.
    """
    if validate_rows is None:
        validated = validate_lookup_rows(
            rows,
            table.heading.names,
            schema_for=schema_for,
            table_name=table_name,
            per_row_hook=per_row_hook,
        )
    else:
        validated = validate_rows(rows, table.heading.names)
    reject_duplicate_parameter_content(
        table,
        validated,
        table_name=table_name,
        name_attr=name_attr,
        sorter_keyed=sorter_keyed,
        matcher_keyed=matcher_keyed,
        allow_duplicate_params=allow_duplicate_params,
    )
    insert_rows(validated, **kwargs)

"""Insert guards and integrity adapters for v2 DataJoint tables.

Imports no schemas. DataJoint itself is loaded only for guard exceptions;
parameter validation and integrity queries run at the explicit table boundary.
"""

from __future__ import annotations

from spyglass.spikesorting.v2._core.lookup_validation import (
    reject_duplicate_parameter_content,
    validate_lookup_rows,
)


def single_source_part(table, key: dict) -> tuple[str, dict]:
    """Return the one XOR source part row of a selection master.

    For masters with exactly one ``RecordingSource`` or
    ``ConcatenatedRecordingSource`` part row.

    Parameters
    ----------
    table : type
        The selection master class.
    key : dict
        A key restricting ``table`` to one master; non-primary-key fields
        are ignored.

    Returns
    -------
    kind : {"recording", "concatenated_recording"}
        Which source part holds the row.
    source_key : dict
        ``{"recording_id": ...}`` or ``{"concat_recording_id": ...}``.

    Raises
    ------
    SchemaBypassError
        If zero or more than one source part row exists for ``key``.
    """
    from spyglass.spikesorting.v2.exceptions import SchemaBypassError

    master_key = {k: v for k, v in key.items() if k in table.primary_key}
    recording_ids = (table.RecordingSource & master_key).fetch("recording_id")
    concat_ids = (table.ConcatenatedRecordingSource & master_key).fetch(
        "concat_recording_id"
    )
    total = len(recording_ids) + len(concat_ids)
    if total != 1:
        raise SchemaBypassError(
            f"{table.__name__} {master_key} has {total} source part "
            f"rows; expected exactly one. Use {table.__name__}."
            "insert_selection() to add or remove this selection."
        )
    if len(recording_ids):
        return "recording", {"recording_id": recording_ids[0]}
    return "concatenated_recording", {"concat_recording_id": concat_ids[0]}


def optional_part_value(table, key: dict, part, column: str):
    """Return ``column`` of a selection's zero-or-one part row, or ``None``.

    Parameters
    ----------
    table : type
        The selection master class that owns ``part``.
    key : dict
        A key restricting ``table`` to one master; non-primary-key fields
        are ignored.
    part : type
        The optional part table to read.
    column : str
        The ``part`` attribute to return.

    Raises
    ------
    SchemaBypassError
        If more than one ``part`` row exists for ``key``.
    """
    from spyglass.spikesorting.v2.exceptions import SchemaBypassError

    master_key = {k: v for k, v in key.items() if k in table.primary_key}
    rows = (part & master_key).fetch(column)
    if len(rows) > 1:
        raise SchemaBypassError(
            f"{table.__name__} {master_key} has {len(rows)} "
            f"{part.__name__} rows; expected zero or one."
        )
    return rows[0] if len(rows) else None


class _IdentityMasterGuard:
    """Reject a direct ``insert`` / in-place ``update1`` of an identity master.

    Subclasses supply the rejection text. ``insert`` mirrors
    ``dj.Table.insert``'s positional order; ``insert1`` forwards its kwargs
    here.
    """

    def _direct_insert_reason(self) -> str:
        """The rationale and create-path sentence for a rejected insert."""
        raise NotImplementedError

    def _update1_create_hint(self) -> str:
        """The create path to use instead of a rejected ``update1``."""
        raise NotImplementedError

    def insert(
        self,
        rows,
        replace=False,
        skip_duplicates=False,
        ignore_extra_fields=False,
        *,
        allow_direct_insert=False,
        **kwargs,
    ):
        """Reject a direct insert unless ``allow_direct_insert`` is set.

        Parameters
        ----------
        rows : iterable
            Rows to insert, forwarded to ``dj.Table.insert``.
        replace : bool, optional
            Replace existing rows on key conflict. Default ``False``.
        skip_duplicates : bool, optional
            Silently skip duplicate-key rows. Default ``False``.
        ignore_extra_fields : bool, optional
            Drop row keys not in the table heading. Default ``False``.
        allow_direct_insert : bool, optional
            Escape hatch for a deliberate maintenance or test bypass of
            the create path. Default ``False``.
        **kwargs
            Additional keyword arguments forwarded to
            ``super().insert``.

        Raises
        ------
        datajoint.errors.DataJointError
            If ``allow_direct_insert`` is ``False`` (the default),
            directing the caller to the create path instead.
        """
        if not allow_direct_insert:
            import datajoint as dj

            raise dj.errors.DataJointError(
                f"Direct insert into {type(self).__name__} is not supported: "
                f"{self._direct_insert_reason()} Pass allow_direct_insert=True "
                "only for a deliberate maintenance or test bypass."
            )
        super().insert(
            rows,
            replace=replace,
            skip_duplicates=skip_duplicates,
            ignore_extra_fields=ignore_extra_fields,
            **kwargs,
        )

    def update1(self, row, *, allow_master_mutation=False):
        """Reject an in-place row mutation unless ``allow_master_mutation``.

        Editing identity-bearing columns in place retargets live dependents.

        Parameters
        ----------
        row : dict
            The row to update, forwarded to ``dj.Table.update1``.
        allow_master_mutation : bool, optional
            Escape hatch for a deliberate maintenance or test mutation of a
            row known to have no live downstream references. Default ``False``.

        Raises
        ------
        datajoint.errors.DataJointError
            If ``allow_master_mutation`` is ``False`` (the default).
        """
        if not allow_master_mutation:
            import datajoint as dj

            raise dj.errors.DataJointError(
                f"In-place update1 of {type(self).__name__} is not supported: "
                "its identity-bearing columns feed the deterministic ids (or "
                "are the provenance roots) that live dependents reference, so "
                "editing them in place silently retargets those references. "
                f"{self._update1_create_hint()} Pass "
                "allow_master_mutation=True only for a deliberate maintenance "
                "edit of a row with no live references."
            )
        super().update1(row)


class SelectionMasterInsertGuard(_IdentityMasterGuard):
    """Reject a direct ``insert`` into a deterministic-id selection master.

    Only ``insert_selection`` holds the full logical identity the primary key
    is derived from. This catches the easy mistake early; integrity is
    enforced by PK uniqueness and the bypass checks. ``insert_selection``
    passes ``allow_direct_insert=True`` itself.
    """

    def _direct_insert_reason(self) -> str:
        return (
            "the primary key is derived from the selection's full logical "
            "identity (and, for the part-bearing masters, the source-part "
            "rows are inserted atomically with it). Use "
            f"{type(self).__name__}.insert_selection()."
        )

    def _update1_create_hint(self) -> str:
        return (
            f"Insert a new selection via {self.__class__.__name__}."
            "insert_selection() instead."
        )


class FactoryOnlyMaster(_IdentityMasterGuard):
    """Reject direct ``insert`` / ``update1`` of a factory-constructed master.

    A direct insert skips the factory's atomic master + part construction.
    Must precede ``SpyglassMixin`` / ``dj.Manual`` in the MRO. Subclasses set
    :attr:`_factory_create_call`.
    """

    #: e.g. ``"CurationV2.insert_curation()"``.
    _factory_create_call: str = "its factory classmethod"

    def _direct_insert_reason(self) -> str:
        return (
            "it is an identity / provenance root that downstream rows "
            f"reference. Write it through {self._factory_create_call}, "
            "which constructs the master and its parts atomically."
        )

    def _update1_create_hint(self) -> str:
        return f"Insert a new row via {self._factory_create_call} instead."


class ImmutableParamsLookup:
    """Reject in-place mutation of a content-addressed parameter Lookup row.

    Downstream ids hash the row name, not its content, so editing content in
    place silently redefines existing ids. Guards ``update1`` and
    ``insert(..., replace=True)``. Must precede ``SpyglassMixin`` /
    ``dj.Lookup`` in the MRO.
    """

    def update1(self, row, *, allow_param_mutation=False):
        """Reject an in-place row mutation unless ``allow_param_mutation``.

        Parameters
        ----------
        row : dict
            The row to update, forwarded to ``dj.Table.update1``.
        allow_param_mutation : bool, optional
            Escape hatch for a deliberate maintenance or test mutation of a
            content-addressed parameter row. Default ``False``.

        Raises
        ------
        datajoint.errors.DataJointError
            If ``allow_param_mutation`` is ``False`` (the default),
            directing the caller to insert a new named row instead.
        """
        if not allow_param_mutation:
            import datajoint as dj

            raise dj.errors.DataJointError(
                f"In-place update1 of {self.__class__.__name__} is not "
                "supported: this row's content is folded into the deterministic "
                "id of downstream selections (directly, or via the named set it "
                "belongs to), so editing it under the same key silently "
                "re-defines what existing ids mean. Insert a NEW named row "
                "instead. Pass allow_param_mutation=True only for a deliberate "
                "maintenance or test edit of a row with no live references."
            )
        super().update1(row)

    def insert(self, rows, replace=False, *args, **kwargs):
        """Reject ``replace=True``; otherwise forward to the next ``insert``.

        No escape hatch: a content change needs a new named row. ``replace``
        keeps DataJoint's second position so ``insert(rows, True)`` is caught.

        Parameters
        ----------
        rows
            Forwarded to ``super().insert`` unchanged.
        replace : bool, optional
            Must be ``False`` (the default); ``True`` raises.
        *args
            DataJoint's later positional flags, forwarded in position.
        **kwargs
            Forwarded to ``super().insert``.

        Raises
        ------
        datajoint.errors.DataJointError
            If ``replace=True``.
        """
        if replace:
            import datajoint as dj

            raise dj.errors.DataJointError(
                f"insert(replace=True) on {self.__class__.__name__} is not "
                "supported: this row's content is folded into the "
                "deterministic id of downstream selections, so overwriting "
                "it in place under the same key silently re-defines what "
                "existing ids mean. Insert a NEW named row instead."
            )
        super().insert(rows, replace, *args, **kwargs)


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

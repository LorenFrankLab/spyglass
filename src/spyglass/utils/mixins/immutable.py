"""Mixins that reject in-place edits of rows other rows depend on.

See the "Immutable Parameter Rows" and "Factory-Only Masters" sections of
the Spyglass Mixin docs.
"""

import datajoint as dj


class ImmutableRowError(dj.errors.DataJointError):
    """Raise on an in-place edit of a row that downstream rows depend on.

    Covers ``update1`` and ``insert(replace=True)`` on a guarded table.
    Message names the table and the supported alternative.
    """


class UnsupportedDirectInsertError(dj.errors.DataJointError, RuntimeError):
    """Raise on a direct insert into a table that has its own create path.

    Message names the table and the create path to use instead.
    """


class _IdentityMasterGuard:
    """Reject a direct ``insert`` / in-place ``update1`` of a master.

    For a master whose rows have one create path and live dependents.
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
        UnsupportedDirectInsertError
            If ``allow_direct_insert`` is ``False`` (the default),
            directing the caller to the create path instead.
        """
        if not allow_direct_insert:
            raise UnsupportedDirectInsertError(
                f"Direct insert into {type(self).__name__} is not supported: "
                f"{self._direct_insert_reason()} Pass allow_direct_insert=True "
                "only for a maintenance or test bypass."
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

        Editing a master in place changes what its live dependents reference.

        Parameters
        ----------
        row : dict
            The row to update, forwarded to ``dj.Table.update1``.
        allow_master_mutation : bool, optional
            Escape hatch for a deliberate maintenance or test mutation of a
            row known to have no live downstream references. Default ``False``.

        Raises
        ------
        ImmutableRowError
            If ``allow_master_mutation`` is ``False`` (the default).
        """
        if not allow_master_mutation:
            raise ImmutableRowError(
                f"In-place update1 of {type(self).__name__} is not supported: "
                "downstream rows depend on its content. "
                f"{self._update1_create_hint()} Pass "
                "allow_master_mutation=True only for a maintenance edit of a "
                "row with no live references."
            )
        super().update1(row)


class FactoryOnlyMaster(_IdentityMasterGuard):
    """Reject direct ``insert`` / ``update1`` of a factory-constructed master.

    A factory classmethod inserts the master and its parts together; a
    direct insert can leave a master without its parts, and an in-place edit
    changes what existing parts and downstream rows reference. The factory
    passes ``allow_direct_insert=True``. Goes after ``SpyglassMixin`` and
    before ``dj.Manual``, except on a ``SpyglassMixinPart``, which it must
    precede. Subclasses set :attr:`_factory_create_call`.
    """

    #: e.g. ``"MyGroup.create_group()"``.
    _factory_create_call: str = "its factory classmethod"

    def _direct_insert_reason(self) -> str:
        return (
            "the master and its parts must be inserted together. Use "
            f"{self._factory_create_call} instead."
        )

    def _update1_create_hint(self) -> str:
        return f"Insert a new row via {self._factory_create_call} instead."


class ImmutableParamsLookup:
    """Reject in-place mutation of a parameter Lookup row.

    Downstream rows refer to a parameter row by name, so new content under
    the same name silently changes what existing rows mean. Guards
    ``update1`` and
    ``insert(..., replace=True)``. Goes after ``SpyglassMixin`` and before
    ``dj.Lookup``, except on a ``SpyglassMixinPart``, which it must precede.
    """

    def update1(self, row, *, allow_param_mutation=False):
        """Reject an in-place row mutation unless ``allow_param_mutation``.

        Parameters
        ----------
        row : dict
            The row to update, forwarded to ``dj.Table.update1``.
        allow_param_mutation : bool, optional
            Escape hatch for a deliberate maintenance or test mutation of a
            parameter row. Default ``False``.

        Raises
        ------
        ImmutableRowError
            If ``allow_param_mutation`` is ``False`` (the default),
            directing the caller to insert a new named row instead.
        """
        if not allow_param_mutation:
            raise ImmutableRowError(
                f"In-place update1 of {self.__class__.__name__} is not "
                "supported: downstream rows refer to this row by name, so new "
                "content under the same name changes what they mean. Insert a "
                "new named row instead, or pass allow_param_mutation=True for "
                "a maintenance edit of a row with no live references."
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
        ImmutableRowError
            If ``replace=True``.
        """
        if replace:
            raise ImmutableRowError(
                f"insert(replace=True) on {self.__class__.__name__} is not "
                "supported: downstream rows refer to this row by name, so new "
                "content under the same name changes what they mean. Insert a "
                "new named row instead."
            )
        super().insert(rows, replace, *args, **kwargs)

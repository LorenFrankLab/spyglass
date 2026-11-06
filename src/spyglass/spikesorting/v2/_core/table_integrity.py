"""Insert guards and integrity adapters for v2 DataJoint tables.

Imports no schemas. DataJoint itself is loaded only for guard exceptions;
parameter validation and integrity queries run at the explicit table boundary.
"""

from __future__ import annotations

from collections import Counter

from spyglass.spikesorting.v2._core.lookup_validation import (
    reject_duplicate_parameter_content,
    validate_lookup_rows,
)


def split_leading_restrictions(args: tuple) -> tuple[list, tuple]:
    """Peel leading restriction positionals off a ``delete`` arg tuple.

    DataJoint's ``delete`` takes no positional restrictions; Spyglass's
    cautious-delete layer reads the first positional as the truthy
    ``force_permission`` and would then cascade-delete EVERY row of the
    unrestricted instance. The v2 table ``delete`` overrides
    (``Sorting``, guarding a 5-50 GB per-row analyzer folder; the
    ``RecordingArtifactDetection`` / ``SharedGroupArtifactDetection``, guarding
    owned ``IntervalList`` rows + the merge registration) defend against the
    easy-to-mistype
    ``Table().delete(restriction)`` form by peeling every leading
    ``dict`` / ``list`` / ``str`` positional into a restriction list and
    re-dispatching ``(self & r1 & r2 & ...).delete(*rest)``.

    This is the pure half of that guard: it does not touch the DB.

    Parameters
    ----------
    args : tuple
        The ``*args`` a ``delete`` override received.

    Returns
    -------
    restrictions : list
        The leading restriction positionals, in order. Empty when ``args``
        does not start with a restriction (a normal ``.delete()`` call).
    remaining : tuple
        The rest of ``args`` after the leading restrictions, untouched.
    """
    restrictions = []
    remaining = args
    while remaining and isinstance(remaining[0], (dict, list, str)):
        restrictions.append(remaining[0])
        remaining = remaining[1:]
    return restrictions, remaining


class _IdentityMasterGuard:
    """Reject a direct ``insert`` / in-place ``update1`` of an identity master.

    The shared body of :class:`SelectionMasterInsertGuard` and
    :class:`FactoryOnlyMaster`, which differ only in their rejection text:
    each subclass implements :meth:`_direct_insert_reason` and
    :meth:`_update1_create_hint`. Both are methods, not class attributes,
    because the text names the concrete table or its factory call.

    The ``insert`` signature mirrors ``dj.Table.insert`` so positional
    ``replace`` / ``skip_duplicates`` keep working; only
    ``allow_direct_insert`` is keyword-only (it cannot accidentally bind a
    positional flag). DataJoint forwards ``insert1``'s ``**kwargs`` to
    ``insert``, so an ``insert1(row, allow_direct_insert=True)`` reaches this
    override too.
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

        A master's identity-bearing columns feed the deterministic ids (or
        are the provenance roots) that live dependents reference, so editing
        them in place retargets those references -- the symmetric hazard to a
        direct insert. ``update1`` is the only standard in-place mutation
        path, so guard it the same way :class:`ImmutableParamsLookup` guards
        the param Lookups.

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

    The v2 deterministic-id selection masters (``RecordingSelection`` /
    ``RecordingArtifactSelection`` / ``SharedGroupArtifactSelection`` /
    ``SortingSelection``) derive their primary key from the selection's FULL
    logical identity, and ``insert_selection`` is the only entry point that
    holds that full payload: it computes the deterministic PK, pre-checks the
    lookup-row FKs, and inserts the row(s). Only ``SortingSelection`` is
    part-bearing -- its optional ``ArtifactDetectionSource`` pass genuinely
    CANNOT be verified from the master row alone (it lives in a part table). The
    two artifact selections carry their source as a REQUIRED FK on the
    master (structural exactly-one-source), and ``RecordingSelection`` has no
    source part either; routing all of them through the same boundary keeps one
    consistent create path.

    This guard is a guard-RAIL, not the integrity boundary: it rejects the
    easy mistake (calling ``insert`` / ``insert1`` instead of
    ``insert_selection``) early and loudly. The actual integrity enforcement
    is downstream -- the deterministic-PK uniqueness + the
    ``SchemaBypassError`` / ``DuplicateSelectionError`` checks that detect a
    bypassed or orphaned master. ``allow_direct_insert=True`` is the escape
    hatch for a deliberate maintenance or test bypass -- the SAME keyword
    DataJoint uses to override its own auto-populated-table insert guard
    (note: on a ``dj.Manual`` table that keyword is otherwise inert, so it
    is repurposed here). ``insert_selection`` itself passes
    ``allow_direct_insert=True`` for its already-validated master insert.
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

    ``CurationV2`` (`curation.py`) and ``SessionGroup`` (`session_group.py`)
    are not selection masters, but they are identity / provenance roots that
    downstream rows reference. A direct ``insert`` skips the atomic master +
    part construction the factory classmethod performs (the analysis-file row
    and ``Unit`` / ``UnitLabel`` parts for ``CurationV2``; the ``Member`` rows
    for ``SessionGroup``), and an in-place ``update1`` retargets what existing
    dependents point at. Both are blocked unless an explicit bypass keyword is
    passed; the factory classmethods (``CurationV2.insert_curation`` /
    ``SessionGroup.create_group``) pass ``allow_direct_insert=True`` for their
    already-validated master insert.

    Shares :class:`SelectionMasterInsertGuard`'s ``insert`` /
    ``update1`` guards (``allow_direct_insert`` / ``allow_master_mutation``),
    with factory-specific messages: the mixin must precede
    ``SpyglassMixin`` / ``dj.Manual`` in the MRO so its overrides take
    precedence. Subclasses set :attr:`_factory_create_call` to the factory the
    error message points to.
    """

    #: The factory call named in the rejection messages (e.g.
    #: ``"CurationV2.insert_curation()"``). Subclasses override.
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

    The v2 parameter Lookups
    (``PreprocessingParameters`` / ``ArtifactDetectionParameters`` /
    ``SorterParameters`` / ``AnalyzerWaveformParameters`` /
    ``MatcherParameters``) are keyed by a
    human-chosen NAME, and that name -- not the parameter content -- is what
    flows into the deterministic ``recording_id`` / ``artifact_detection_id``
    / ``sorting_id`` / ``concat_recording_id`` / ``unitmatch_id`` of every
    downstream selection. The ``insert`` overrides already reject a SECOND
    name for identical content (``reject_duplicate_parameter_content``)
    because that forks provenance; the symmetric hazard is editing a row's
    blob IN PLACE under the SAME name, which silently re-defines what every
    already-minted id means. DataJoint offers two standard in-place mutation
    paths -- ``update1`` and ``insert(..., replace=True)`` -- so both are
    guarded here: a backend/parameter change requires a NEW named row, not an
    in-place edit or overwrite.

    ``allow_param_mutation=True`` is the escape hatch for a deliberate
    maintenance or test edit of ``update1`` (mirroring ``allow_direct_insert``
    on :class:`SelectionMasterInsertGuard`) of a row known to have no live
    downstream references; ``insert(..., replace=True)`` has no such escape
    hatch. The mixin must precede ``SpyglassMixin`` / ``dj.Lookup`` in the
    MRO so its ``update1`` / ``insert`` take precedence.
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

        DataJoint's ``insert(..., replace=True)`` overwrites a row's content
        in place under its EXISTING primary key (SQL ``REPLACE``) -- the same
        identity-forking hazard ``update1`` above guards against, reachable
        through a different DataJoint entry point. Unlike ``update1``, there
        is no escape hatch here: a deliberate content change still needs a
        NEW named row, not an in-place overwrite of one every downstream
        selection may already reference.

        Every subclass whose own ``insert`` override forwards ``**kwargs`` to
        ``super().insert(...)`` reaches this check; a subclass that never
        calls ``super().insert`` at all (``AutoCurationRules`` /
        ``AutoCurationRules.Rule`` reject direct inserts unconditionally) or
        that guards ``replace`` itself before calling ``super()``
        (``UnitAnnotationDefinition``, ``CurationReviewProfile``) is already
        covered without reaching here.

        ``replace`` takes DataJoint's own position (second), so a positional
        ``insert(rows, True)`` is rejected like ``replace=True`` on a subclass
        without an ``insert`` override (e.g. ``MotionCorrectionParameters``).
        Subclass overrides forward only ``rows`` and keyword arguments here.

        Parameters
        ----------
        rows
            Forwarded to ``super().insert`` unchanged.
        replace : bool, optional
            Must be ``False`` (the default); ``True`` raises.
        *args
            DataJoint's later positional flags (``skip_duplicates``, ...),
            forwarded after ``replace`` so they keep their positions.

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

    ``insert_rows`` is the table override's bound ``super().insert`` so the
    immutable-parameter guard and DataJoint insertion flags remain in its MRO.
    Ordinary parameter tables supply ``schema_for`` and an optional row hook.
    Sorter/matcher tables may instead supply ``validate_rows(rows, names)`` to
    retain their specialized batch validation and error ordering.
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


def find_orphaned_masters(master_table, part_tables: list) -> list[dict]:
    """Return master PKs whose source-part counts sum to zero.

    Backs ``SortingSelection.prune_orphaned_selections`` and
    ``MotionEstimateSelection.prune_orphaned_selections``: ``part_tables`` is
    that master's XOR source-part set
    ``[RecordingSource, ConcatenatedRecordingSource]``. (The artifact
    selections and ``MotionCorrectedRecordingSelection`` carry their input as
    a REQUIRED FK on the master, so they cannot be orphaned and have no
    ``prune_orphaned_selections``.)

    Source-part atomicity is enforced at insert time by the transactional
    ``insert_selection`` helper, but DataJoint cannot enforce "exactly one
    source per master" across two part tables; an upstream cascade-delete from
    ``Recording`` / ``ConcatenatedRecording`` can leave the master row without
    any source children. This helper finds those orphans so a maintenance
    script can review or remove them.

    Parameters
    ----------
    master_table : datajoint.Table
        The selection master table to scan for orphaned rows.
    part_tables : list
        The source-part tables to count against each master row.

    Returns
    -------
    list[dict]
        The primary-key dicts of masters with zero source-part rows.
    """
    # Each part is keyed by ``-> master`` only: antijoin on the master key.
    orphans = master_table.proj()
    for part in part_tables:
        orphans = orphans - part.proj()
    return orphans.fetch("KEY", as_dict=True)


def audit_source_part_integrity(master_table, part_tables: list) -> list[dict]:
    """Return masters whose source-part row count is not exactly one.

    Complements :func:`find_orphaned_masters`, which flags only the
    zero-source case. A master with TWO source-part rows is an
    AMBIGUOUS-source bug -- ``resolve_source`` would raise lazily, only when
    something happens to read it -- so flag both ``0`` (orphan) and ``>1``
    (ambiguous) here for a maintenance script to review.

    ``part_tables`` must be the recording-source parts ONLY -- the
    exactly-one-of XOR set. For ``SortingSelection`` (its only user) that is
    ``[RecordingSource, ConcatenatedRecordingSource]``; ``ArtifactDetectionSource``
    is deliberately EXCLUDED because it is an independent zero-or-one part (a
    valid artifact-backed sorting carries one, so including it would falsely
    flag every artifact-bearing sorting as ``count == 2``). This is the same
    part list ``find_orphaned_masters`` / ``prune_orphaned_selections`` pass.
    (The ``ArtifactDetectionOutput`` merge has its OWN, separate
    ``audit_source_part_integrity`` classmethod for its two source parts.)

    Parameters
    ----------
    master_table : datajoint.Table
        The selection master table to scan.
    part_tables : list
        The recording-source part tables (the exactly-one-of XOR set) to count
        against each master row.

    Returns
    -------
    list[dict]
        One entry per offending master: its primary-key fields plus
        ``"source_part_count"`` (``0`` = orphan, ``>= 2`` = ambiguous). Masters
        with exactly one source part are omitted.
    """
    pk = master_table.primary_key
    counts = Counter(
        tuple(row[attr] for attr in pk)
        for part in part_tables
        for row in part.fetch("KEY", as_dict=True)
    )
    flagged: list[dict] = []
    for master in master_table.fetch("KEY", as_dict=True):
        count = counts[tuple(master[attr] for attr in pk)]
        if count != 1:
            flagged.append({**master, "source_part_count": count})
    return flagged

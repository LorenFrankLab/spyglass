"""Shared helpers for the spike sorting tables and pipeline."""

from __future__ import annotations

import functools
import numbers
import os
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import datajoint as dj
import spikeinterface as si

# Schema enums live in the stdlib-only _enums.py so dependency-light
# service modules can import the canonical label set without pulling
# DataJoint/SpikeInterface through this module; re-exported here so
# existing ``from .utils import CurationLabel`` call sites are unchanged.
from spyglass.spikesorting.v2._enums import (  # noqa: F401
    CurationLabel,
    CurationSource,
)

# Pure signal/frame/interval math lives in _signal_math.py; re-exported
# here so existing ``from .utils import _consolidate_intervals`` (etc.)
# keep working unchanged.
from spyglass.spikesorting.v2._signal_math import (  # noqa: F401
    _base_intervals_from_timestamps,
    _consolidate_intervals,
    _dedup_merged_spike_times,
    _get_recording_timestamps,
    _spike_times_to_frames,
)

# Pure reference-resolution lives in _reference_resolution.py (DB-free);
# re-exported so ``from .utils import resolve_group_reference`` (etc.) and the
# ``ReferenceMode`` type keep working unchanged.
from spyglass.spikesorting.v2._reference_resolution import (  # noqa: F401
    ReferenceMode,
    _validate_reference_fields,
    assert_reference_not_member,
    resolve_group_reference,
)

# Parameter-Lookup validation lives in _lookup_validation.py and NWB /
# recording-metadata helpers in _nwb_metadata_helpers.py (both import-light);
# re-exported so existing ``from .utils import validate_lookup_rows`` /
# ``electrode_table_region`` (etc.) call sites are unchanged.
from spyglass.spikesorting.v2._lookup_validation import (  # noqa: F401
    _assert_schema_version_matches,
    _ensure_lookup_row_exists,
    _insert_row_to_dict,
    _jsonable_blob,
    _validate_params,
    reject_duplicate_parameter_content,
    reject_duplicate_quality_metric_content,
    validate_lookup_rows,
)
from spyglass.spikesorting.v2._nwb_metadata_helpers import (  # noqa: F401
    electrode_table_region,
    resolve_conversion_and_offset,
)

# The artifact-detection IntervalList naming convention lives in the
# stdlib-only _artifact_naming.py so the DB-free source-routing module
# (_curation_routing) can import the parser without pulling DataJoint /
# SpikeInterface through this barrel; re-exported here so existing
# ``from .utils import artifact_detection_interval_list_name`` (etc.) call
# sites are unchanged.
from spyglass.spikesorting.v2._artifact_naming import (  # noqa: F401
    _ARTIFACT_DETECTION_INTERVAL_LIST_PREFIX,
    artifact_detection_interval_list_name,
    parse_artifact_detection_interval_list_name,
)


@contextmanager
def transaction_or_noop(connection):
    """Open a DataJoint transaction unless one is already active.

    Parameters
    ----------
    connection : datajoint.connection.Connection
        The active DataJoint connection. A transaction is opened on it
        only when one is not already in progress.

    Source-part inserts and curation inserts both want to wrap their
    master + part rows in one transaction; but the same helpers may be
    called from inside an existing populate cascade where DataJoint
    refuses nested transactions. This context manager makes the
    transaction wrap a no-op when the connection is already in one.

    NOTE on duplicate-key recovery: the ``insert_selection`` helpers catch
    a duplicate-PK error raised inside this block and refetch the winner's
    row. That recovery assumes a TOP-LEVEL call. There is no savepoint, so
    in the no-op (already-in-transaction) branch a recovered duplicate
    would rely on the caller's outer transaction for cleanup -- but it is
    safe in practice because (a) the selection helpers are only invoked at
    top level (``run_v2_pipeline``), and (b) the deterministic PK makes the
    duplicate collide on the FIRST statement (the master insert), so no
    part row is ever written and there is nothing to roll back. Do not call
    ``insert_selection`` from inside an outer transaction without revisiting
    this.
    """
    if connection.in_transaction:
        yield
    else:
        with connection.transaction:
            yield


def _is_duplicate_key_error(exc: BaseException) -> bool:
    """Return whether ``exc`` is a duplicate-PRIMARY-KEY violation.

    The deterministic-id selection helpers
    (``RecordingSelection``/``ArtifactDetectionSelection``/``SortingSelection``
    ``insert_selection``) derive each master PK from the selection's
    logical identity, then insert. Two callers racing on the same logical
    selection compute the SAME PK, so the loser's insert violates the PK
    uniqueness constraint; the helper catches that, refetches the winner's
    row, and returns it. This predicate decides whether a caught exception
    is that benign race (refetch + return) or a different integrity
    failure that MUST propagate.

    DataJoint translates MySQL errno 1062 (``ER_DUP_ENTRY``) to
    ``dj.errors.DuplicateError`` -- a SIBLING of, not a subclass of,
    ``dj.errors.IntegrityError`` (reserved for FK violations: errno
    1217/1451/1452). So a duplicate PK is a ``DuplicateError`` and a
    missing-source-part / FK violation is an ``IntegrityError``; only the
    former is the race we recover from.

    The ``IntegrityError`` branch is purely defensive -- for a raw
    connector error surfacing before DataJoint's translation (which strips
    the errno, so a *translated* 1062 never reaches ``IntegrityError`` at
    all). It matches ONLY the structured errno (``exc.args[0] == 1062``),
    not rendered database messages. Message matching is connector- and
    locale-sensitive, and a false positive would silently swallow a genuine
    integrity failure (e.g. an FK violation) that must propagate rather
    than be treated as the recoverable duplicate-PK race.
    """
    if isinstance(exc, dj.errors.DuplicateError):
        return True
    if isinstance(exc, dj.errors.IntegrityError):
        return bool(exc.args and exc.args[0] == 1062)
    return False


def split_leading_restrictions(args: tuple) -> tuple[list, tuple]:
    """Peel leading restriction positionals off a ``delete`` arg tuple.

    DataJoint's ``delete`` takes no positional restrictions; Spyglass's
    cautious-delete layer reads the first positional as the truthy
    ``force_permission`` and would then cascade-delete EVERY row of the
    unrestricted instance. The v2 table ``delete`` overrides
    (``Sorting`` / ``ArtifactDetection``, each guarding a 5-50 GB
    per-row on-disk artifact) defend against the easy-to-mistype
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


class SelectionMasterInsertGuard:
    """Reject a direct ``insert`` into a deterministic-id selection master.

    The three v2 selection masters
    (``RecordingSelection`` / ``ArtifactDetectionSelection`` / ``SortingSelection``)
    derive their primary key from the selection's FULL logical identity, and
    ``insert_selection`` is the only entry point that holds that full
    payload: it computes the deterministic PK, pre-checks the lookup-row
    FKs, and -- for the part-bearing masters (``ArtifactDetectionSelection`` /
    ``SortingSelection``) -- inserts the master + source parts atomically.
    Those two masters genuinely CANNOT be verified from a master row alone
    (their source identity lives in a part table, not the master's own
    columns); ``RecordingSelection`` is not part-bearing, but routing it
    through the same boundary keeps one consistent create path.

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

    The signature mirrors ``dj.Table.insert`` so positional ``replace`` /
    ``skip_duplicates`` keep working; only ``allow_direct_insert`` is
    keyword-only (it cannot accidentally bind a positional flag).
    """

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
            the ``insert_selection`` boundary. Default ``False``.
        **kwargs
            Additional keyword arguments forwarded to
            ``super().insert``.

        Raises
        ------
        datajoint.errors.DataJointError
            If ``allow_direct_insert`` is ``False`` (the default),
            directing the caller to ``insert_selection`` instead.
        """
        if not allow_direct_insert:
            _reject_master_insert(
                self,
                reason=(
                    "the primary key is derived from the selection's full "
                    "logical identity (and, for the part-bearing masters, the "
                    "source-part rows are inserted atomically with it). Use "
                    f"{type(self).__name__}.insert_selection()."
                ),
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

        A selection master's secondary columns ARE its logical identity (the
        deterministic PK is content-addressed from them), so editing them in
        place retargets the id under every live dependent -- the symmetric
        hazard to a direct insert. ``update1`` is the only standard in-place
        mutation path, so guard it the same way
        :class:`ImmutableParamsLookup` guards the param Lookups: insert a NEW
        selection via ``insert_selection`` instead of editing an existing one.

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
            _reject_master_update1(
                self,
                create_hint=(
                    f"Insert a new selection via {self.__class__.__name__}."
                    "insert_selection() instead."
                ),
            )
        super().update1(row)


def _reject_master_update1(table, *, create_hint: str) -> None:
    """Raise the shared "master identity is immutable" ``update1`` rejection.

    Used by both :class:`SelectionMasterInsertGuard` and
    :class:`FactoryOnlyMaster` so the two in-place-mutation guards share one
    message; ``create_hint`` names the create path to use instead.
    """
    raise dj.errors.DataJointError(
        f"In-place update1 of {type(table).__name__} is not supported: its "
        "identity-bearing columns feed the deterministic ids (or are the "
        "provenance roots) that live dependents reference, so editing them in "
        f"place silently retargets those references. {create_hint} Pass "
        "allow_master_mutation=True only for a deliberate maintenance edit of "
        "a row with no live references."
    )


def _reject_master_insert(table, *, reason: str) -> None:
    """Raise the shared "direct insert blocked" rejection for an identity master.

    Used by both :class:`SelectionMasterInsertGuard` and
    :class:`FactoryOnlyMaster` so the lead-in and the ``allow_direct_insert``
    escape-hatch note live once; ``reason`` is the per-guard rationale plus the
    create-path sentence.
    """
    raise dj.errors.DataJointError(
        f"Direct insert into {type(table).__name__} is not supported: {reason} "
        "Pass allow_direct_insert=True only for a deliberate maintenance or "
        "test bypass."
    )


class FactoryOnlyMaster:
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

    Mirrors :class:`SelectionMasterInsertGuard` (``allow_direct_insert``) and
    :class:`ImmutableParamsLookup` (``update1``): the mixin must precede
    ``SpyglassMixin`` / ``dj.Manual`` in the MRO so its overrides take
    precedence. Subclasses set :attr:`_factory_create_call` to the factory the
    error message points to.
    """

    #: The factory call named in the rejection messages (e.g.
    #: ``"CurationV2.insert_curation()"``). Subclasses override.
    _factory_create_call: str = "its factory classmethod"

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

        Signature mirrors ``dj.Table.insert`` so positional ``replace`` /
        ``skip_duplicates`` keep working; ``allow_direct_insert`` is
        keyword-only. DataJoint forwards ``insert1``'s ``**kwargs`` to
        ``insert``, so an ``insert1(row, allow_direct_insert=True)`` reaches
        this override too.
        """
        if not allow_direct_insert:
            _reject_master_insert(
                self,
                reason=(
                    "it is an identity / provenance root that downstream rows "
                    f"reference. Write it through {self._factory_create_call}, "
                    "which constructs the master and its parts atomically."
                ),
            )
        super().insert(
            rows,
            replace=replace,
            skip_duplicates=skip_duplicates,
            ignore_extra_fields=ignore_extra_fields,
            **kwargs,
        )

    def update1(self, row, *, allow_master_mutation=False):
        """Reject an in-place row mutation unless ``allow_master_mutation``."""
        if not allow_master_mutation:
            _reject_master_update1(
                self,
                create_hint=(
                    f"Insert a new row via {self._factory_create_call} instead."
                ),
            )
        super().update1(row)


class ImmutableParamsLookup:
    """Reject in-place mutation of a content-addressed parameter Lookup row.

    The v2 parameter Lookups
    (``PreprocessingParameters`` / ``ArtifactDetectionParameters`` /
    ``SorterParameters`` / ``AnalyzerWaveformParameters`` /
    ``MotionCorrectionParameters`` / ``MatcherParameters``) are keyed by a
    human-chosen NAME, and that name -- not the parameter content -- is what
    flows into the deterministic ``recording_id`` / ``artifact_detection_id``
    / ``sorting_id`` / ``concat_recording_id`` / ``unitmatch_id`` of every
    downstream selection. The ``insert`` overrides already reject a SECOND
    name for identical content (``reject_duplicate_parameter_content``)
    because that forks provenance; the symmetric hazard is editing a row's
    blob IN PLACE under the SAME name, which silently re-defines what every
    already-minted id means. ``update1`` is the only standard in-place
    mutation path, so guard it: a backend/parameter change requires a NEW
    named row, not an in-place edit.

    ``allow_param_mutation=True`` is the escape hatch for a deliberate
    maintenance or test edit (mirroring ``allow_direct_insert`` on
    :class:`SelectionMasterInsertGuard`) of a row known to have no live
    downstream references. The mixin must precede ``SpyglassMixin`` /
    ``dj.Lookup`` in the MRO so its ``update1`` takes precedence.
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


# ``CurationSource`` and ``CurationLabel`` are defined in the stdlib-only
# ``_enums`` module and re-exported at the top of this file; see the
# import there for why they live outside ``utils``.


def find_orphaned_masters(master_table, part_tables: list) -> list[dict]:
    """Return master PKs whose source-part counts sum to zero.

    Shared implementation of ``ArtifactDetectionSelection.prune_orphaned_selections``
    and ``SortingSelection.prune_orphaned_selections``. ``part_tables``
    is the list of source-part tables to count against the master --
    e.g. ``[RecordingSource, SharedGroupSource]`` for artifact
    selection, ``[RecordingSource, ConcatenatedRecordingSource]`` for
    sorting selection.

    Source-part atomicity is enforced at insert time by the
    transactional ``insert_selection`` helpers, but DataJoint cannot
    enforce "exactly one source per master" across two part tables;
    an upstream cascade-delete from ``Recording`` or
    ``SharedArtifactGroup`` / ``ConcatenatedRecording`` can leave the
    master row without any source children. This helper finds those
    orphans so a maintenance script can review or remove them.

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
    orphans: list[dict] = []
    for master in master_table.fetch("KEY", as_dict=True):
        if sum(len(part & master) for part in part_tables) == 0:
            orphans.append(master)
    return orphans


def audit_source_part_integrity(master_table, part_tables: list) -> list[dict]:
    """Return masters whose source-part row count is not exactly one.

    Complements :func:`find_orphaned_masters`, which flags only the
    zero-source case. A master with TWO source-part rows is an
    AMBIGUOUS-source bug -- ``resolve_source`` would raise lazily, only when
    something happens to read it -- so flag both ``0`` (orphan) and ``>1``
    (ambiguous) here for a maintenance script to review.

    ``part_tables`` must be the recording-source parts ONLY -- the
    exactly-one-of XOR set. For ``SortingSelection`` that is
    ``[RecordingSource, ConcatenatedRecordingSource]``; ``ArtifactDetectionSource``
    is deliberately EXCLUDED because it is an independent zero-or-one part (a
    valid artifact-backed sorting carries one, so including it would falsely
    flag every artifact-bearing sorting as ``count == 2``). For
    ``ArtifactDetectionSelection`` the XOR set is
    ``[RecordingSource, SharedGroupSource]``. These are the same part lists
    ``find_orphaned_masters`` / ``prune_orphaned_selections`` already pass.

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
    flagged: list[dict] = []
    for master in master_table.fetch("KEY", as_dict=True):
        count = sum(len(part & master) for part in part_tables)
        if count != 1:
            flagged.append({**master, "source_part_count": count})
    return flagged


def resolve_peak_sign(params) -> str:
    """Map a validated sorter params blob to a SpikeInterface ``peak_sign``.

    Used by ``Sorting._populate_unit_part`` so per-unit peak channel and
    amplitude attribution honor the sorter's configured detection polarity
    instead of hardcoding ``"neg"`` (SpikeInterface's default). Sorters
    express polarity differently:

    * ``clusterless_thresholder`` carries ``peak_sign`` directly
      (``"neg"`` / ``"pos"`` / ``"both"``).
    * MountainSort 4/5 carry ``detect_sign`` (``-1`` neg, ``1`` pos,
      ``0`` both).
    * Kilosort4 / SpykingCircus2 / Tridesclous2 / the generic schema
      carry neither; fall back to ``"neg"`` (SI's default, matching the
      prior behavior for those sorters).

    Parameters
    ----------
    params : Mapping or None
        The validated ``SorterParameters.params`` blob. A non-mapping /
        ``None`` returns ``"neg"`` rather than raising.

    Returns
    -------
    str
        One of ``"neg"`` / ``"pos"`` / ``"both"``.
    """
    if not isinstance(params, Mapping):
        return "neg"
    if params.get("peak_sign") is not None:
        return str(params["peak_sign"])
    if params.get("detect_sign") is not None:
        return {-1: "neg", 0: "both", 1: "pos"}.get(
            int(params["detect_sign"]), "neg"
        )
    return "neg"


def write_buffer_gb(
    n_channels: int,
    sampling_frequency: float,
    max_seconds: float = 30.0,
    itemsize: int = 8,
    cap_gb: float = 5.0,
) -> float:
    """Streaming-write buffer size (GB) bounded to ~``max_seconds`` of data.

    A fixed buffer (the prior 5 GB) buffers the WHOLE recording for narrow
    sort groups -- a 4-channel tetrode is ~87 min in 5 GB -- defeating the
    HDMF streaming write and spiking RAM under parallel per-group workers.
    Scale the buffer with channel count so every group buffers ~the same
    bounded duration, capped at ``cap_gb`` so wide groups are unchanged.

    Parameters
    ----------
    n_channels : int
        Number of channels in the recording being written.
    sampling_frequency : float
        Sampling frequency of the recording, in Hz.
    max_seconds : float, optional
        Target buffered duration, in seconds. Default ``30.0``.
    itemsize : int, optional
        Bytes per sample of the trace dtype. Default ``8`` (float64).
    cap_gb : float, optional
        Upper bound on the returned buffer size, in GB. Default ``5.0``.

    Returns
    -------
    float
        Buffer size in GB, in ``[0, cap_gb]`` (``0`` only for the degenerate
        ``n_channels == 0``; ``> 0`` for any real recording).
    """
    seconds_gb = (
        float(n_channels)
        * float(sampling_frequency)
        * float(max_seconds)
        * itemsize
        / 1e9
    )
    return min(cap_gb, seconds_gb)


def unit_brain_region_df(unit_relation, resolution: str):
    """Join a Unit-part relation against Electrode * BrainRegion.

    Shared implementation of ``Sorting.get_unit_brain_regions`` and
    ``CurationV2.get_unit_brain_regions``. The Unit relation must
    carry an ``Electrode`` FK; the join walks it to ``BrainRegion``
    (non-null FK on ``Electrode``) and returns a DataFrame with the
    standard column set + a ``region_resolution`` literal label so
    concat-backed callers can distinguish anchor-member results.

    Parameters
    ----------
    unit_relation : datajoint.expression.QueryExpression
        A Unit-part relation carrying an ``Electrode`` FK.
    resolution : str
        Literal label written verbatim into the ``region_resolution``
        column of every returned row.

    Returns
    -------
    pandas.DataFrame
        Columns ``unit_id``, ``electrode_id``, ``region_name``,
        ``subregion_name``, ``subsubregion_name``, and the
        ``region_resolution`` literal label. Carries the full schema
        even when empty.
    """
    import pandas as pd

    from spyglass.common.common_ephys import Electrode as _Electrode
    from spyglass.common.common_region import BrainRegion

    columns = [
        "unit_id",
        "electrode_id",
        "region_name",
        "subregion_name",
        "subsubregion_name",
    ]
    joined = (unit_relation * _Electrode * BrainRegion).fetch(
        *columns, as_dict=True
    )
    # Pass ``columns=`` so an empty result still carries the full schema;
    # ``pd.DataFrame([])`` would otherwise drop every column and leave
    # callers a frame with only ``region_resolution``.
    df = pd.DataFrame(joined, columns=columns)
    df["region_resolution"] = resolution
    return df


@dataclass(frozen=True)
class SourceResolution:
    """Result of ``<MasterTable>.resolve_source(key)`` on a source-part master.

    Used at the top of ``Sorting.make()`` and ``ArtifactDetection.make()``
    to dispatch on which source-part row backs the master. ``kind`` is
    the source enum; ``key`` is the source-row PK fields (e.g.
    ``{"recording_id": ...}`` or ``{"shared_artifact_group_name": ...}``)
    so the caller can pass it straight into the upstream table's
    ``get_*`` / fetch helpers.
    """

    kind: Literal[
        "recording",
        "concatenated_recording",
        "shared_artifact_group",
    ]
    key: dict


# ---------------------------------------------------------------------------
# Database-host safety guard
# ---------------------------------------------------------------------------

_SAFE_DB_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_OVERRIDE_ENV = "SPYGLASS_SPIKESORTING_V2_ALLOW_NONLOCAL_DB"


def _assert_v2_db_safe() -> None:
    """Refuse to register v2 schemas against a non-test database.

    Called at module-import time by every v2 schema module before
    ``schema = dj.schema(...)`` runs. While the v2 pipeline is under
    active development we hard-fail any attempt to declare or write to
    v2 schemas on a database other than the isolated test docker
    container (host ``localhost`` / ``127.0.0.1`` / ``::1``).

    The check is intentionally narrow: it pins the database host only.
    Other isolation guarantees (the ``pytests``/``test`` schema prefix,
    the temp ``SPYGLASS_BASE_DIR`` path) are handled by
    ``bootstrap_v2_test_environment``. This guard is the last line of
    defense if some other code path repointed ``dj.config`` at the
    production server after bootstrap ran.

    Raises
    ------
    RuntimeError
        If ``dj.config['database.host']`` is set to a non-local host
        and the override env var is not set.

    Override
    --------
    Set ``SPYGLASS_SPIKESORTING_V2_ALLOW_NONLOCAL_DB=1`` to bypass the
    guard. This requires an explicit deliberate action; do NOT export
    it for a shell session in which you might accidentally run v2
    populate / insert calls.
    """
    if os.environ.get(_OVERRIDE_ENV) == "1":
        return

    host = dj.config.get("database.host")
    if host in _SAFE_DB_HOSTS:
        return

    raise RuntimeError(
        f"v2 spike sorting refuses to register schemas against host "
        f"{host!r}. Expected one of {sorted(_SAFE_DB_HOSTS)} (the "
        f"isolated test docker container). Run "
        f"`bootstrap_v2_test_environment` from "
        f"`tests/spikesorting/v2/test_env.py` before importing v2 "
        f"runtime modules. To bypass with full understanding of the "
        f"risk, set {_OVERRIDE_ENV}=1 in the environment."
    )


def _assert_noise_levels_length(
    noise_levels: list[float] | None, n_channels: int
) -> None:
    """Reject an explicit ``noise_levels`` of an unusable length.

    The clusterless-thresholder broadcasts a singleton ``noise_levels``
    to ``n_channels`` and otherwise indexes it per channel
    (``noise_levels[chan]``). An explicit array whose length is neither
    1 (broadcast) nor ``n_channels`` would either silently truncate the
    channel set or raise an opaque ``IndexError`` deep inside SI's
    ``detect_peaks``. ``None`` is valid (SI estimates per-channel MAD).

    Parameters
    ----------
    noise_levels : list[float] | None
        The resolved noise levels (explicit user value or the
        ``[1.0]`` raw-uV broadcast). ``None`` short-circuits to valid.
    n_channels : int
        Number of channels on the recording the detector runs over.

    Raises
    ------
    ValueError
        If ``noise_levels`` is non-``None`` and ``len(noise_levels)`` is
        neither 1 nor ``n_channels``.
    """
    if noise_levels is None:
        return
    if isinstance(noise_levels, (str, bytes)):
        raise ValueError(
            "clusterless noise_levels must be a numeric sequence with length "
            "1 (broadcast) or n_channels; got a string/bytes value."
        )
    try:
        length = len(noise_levels)
    except TypeError as exc:
        raise ValueError(
            "clusterless noise_levels must be a numeric sequence with length "
            "1 (broadcast) or n_channels; got a scalar value."
        ) from exc
    if length not in (1, n_channels):
        raise ValueError(
            "clusterless noise_levels must have length 1 (broadcast) or "
            f"n_channels={n_channels}; got {length}."
        )


def _ambient_job_kwargs() -> dict:
    """Return the ambient job-kwargs layer: SI globals then dj.config custom.

    The process-global precedence stack beneath any per-row blob -- the
    SpikeInterface global defaults overlaid with
    ``dj.config['custom']['spikesorting_v2_job_kwargs']``. Single source of this
    merge so :func:`_resolved_job_kwargs` and :func:`resolve_effective_seed`
    (which needs the ambient layer in isolation to attribute a seed) cannot
    drift.
    """
    merged = dict(si.get_global_job_kwargs())
    custom = dj.config.get("custom", {}) or {}
    merged.update(custom.get("spikesorting_v2_job_kwargs", {}) or {})
    return merged


def _resolved_job_kwargs(*row_job_kwargs: dict | None) -> dict:
    """Merge SpikeInterface-global, DataJoint-config, and per-row job kwargs.

    Sources are merged in increasing precedence order: the SpikeInterface
    global defaults, then ``dj.config['custom']['spikesorting_v2_job_kwargs']``,
    then each per-row blob in the order given.

    Parameters
    ----------
    *row_job_kwargs : dict or None
        ``job_kwargs`` blob values from the parameter rows that govern this
        compute stage, in increasing precedence order (a later argument wins
        on key conflict). ``None`` and empty-dict entries are skipped.

    Returns
    -------
    dict
        The merged kwargs, ready to splat into a compute call.
    """
    merged = _ambient_job_kwargs()
    for override in row_job_kwargs:
        if override:
            merged.update(override)
    return merged


@functools.lru_cache(maxsize=None)
def _warn_ambient_seed_once(seed: int) -> None:
    """Emit the ambient-seed warning at most once per distinct seed value.

    Deduped per ``seed`` value so a repeated populate (or several sort groups
    sharing one ambient seed) does not spam the log, while a genuinely new
    ambient seed is still surfaced. ``cache_clear()`` resets the dedup (used by
    the unit tests).
    """
    from spyglass.utils import logger

    logger.warning(
        "spikesorting v2: random_seed=%s is supplied via the ambient "
        "SI-global / dj.config['custom']['spikesorting_v2_job_kwargs'] layer, "
        "not a per-row job_kwargs blob. It is used and recorded as the "
        "effective seed, but an ambient seed is process-global rather than "
        "pinned to a parameter row, so the run is harder to reproduce. Prefer "
        "setting random_seed in the per-row job_kwargs.",
        seed,
    )


def resolve_effective_seed(*row_job_kwargs: dict | None) -> int:
    """Return the ``random_seed`` actually used by a v2 compute stage.

    Resolves the seed through the same precedence the dispatch reads --
    SpikeInterface globals, then ``dj.config['custom']['spikesorting_v2_job_kwargs']``,
    then the per-row ``job_kwargs`` blob(s) -- so the value stored on a computed
    row equals the value the sorter / analyzer consumed. The resolution bottoms
    out on :func:`_resolved_job_kwargs`, the exact merge the seed sites read, so
    the stored seed cannot drift from the used seed. Defaults to ``0``.

    Emits a one-time warning (per distinct seed value) when a ``random_seed``
    arrives via the ambient SI-global / ``dj.config`` layer rather than a
    per-row blob -- a process-global ambient seed already takes effect, but it
    is not pinned to a parameter row, so surfacing it keeps a non-reproducible
    ambient seed visible rather than silent.

    Parameters
    ----------
    *row_job_kwargs : dict or None
        The per-row ``job_kwargs`` blob(s) governing this stage, in increasing
        precedence order (a later argument wins). ``None`` / empty entries are
        skipped, matching :func:`_resolved_job_kwargs`.

    Returns
    -------
    int
        The resolved effective random seed (``0`` when unset).

    Raises
    ------
    ValueError
        If the resolved ``random_seed`` is not a non-negative integer (e.g.
        ``"7"``, ``7.9``, or ``-1``). The seed sites consume this resolved value
        directly, so a bad seed is rejected here -- BEFORE compute -- rather than
        silently ``int()``-coerced (which would store ``7`` while the sorter saw
        the original object) or deferred to a later SI/NumPy RNG failure. This is
        the call that runs first in ``make_compute``, so a bad seed aborts the
        populate before any row is written. Mirrors the ``seed >= 0`` constraint
        on the UnitMatch bundle-seed schema.
    """
    resolved = _resolved_job_kwargs(*row_job_kwargs).get("random_seed", 0)
    # ``bool`` is an ``int`` subclass but never a valid seed; ``numbers.Integral``
    # accepts Python and numpy integers (which are seed-equivalent to ``int``).
    # SI/NumPy RNG seeds must be non-negative, so reject ``< 0`` here too.
    if (
        isinstance(resolved, bool)
        or not isinstance(resolved, numbers.Integral)
        or resolved < 0
    ):
        raise ValueError(
            "spikesorting v2 random_seed must be a non-negative integer, got "
            f"{resolved!r} ({type(resolved).__name__}). Set an int random_seed "
            "in the per-row job_kwargs blob (or "
            "dj.config['custom']['spikesorting_v2_job_kwargs'])."
        )
    effective = int(resolved)
    per_row_has_seed = any(
        isinstance(blob, Mapping) and "random_seed" in blob
        for blob in row_job_kwargs
    )
    if not per_row_has_seed:
        ambient = _ambient_job_kwargs()
        if "random_seed" in ambient:
            _warn_ambient_seed_once(int(ambient["random_seed"]))
    return effective


def get_spiking_sorting_v2_merge_ids(
    restriction: dict, as_dict: bool = False
) -> list:
    """Return merge ids for a v2 spike-sorting restriction.

    Notebook-discoverable parallel of ``get_spiking_sorting_v1_merge_ids``;
    thin wrapper over ``SpikeSortingOutput()._get_restricted_merge_ids_v2`` so
    users can do ``get_spiking_sorting_v2_merge_ids(restriction)``
    without poking at the private merge-table method directly.

    ``as_dict=True`` returns ``{"merge_id": uuid}`` dicts; the default
    ``False`` returns a plain list of UUIDs, matching the
    ``get_spiking_sorting_v1_merge_ids`` return shape so a caller copied
    verbatim works unchanged.

    Parameters
    ----------
    restriction : dict
        Restriction on any v2 column (``nwb_file_name``,
        ``sort_group_id``, ``interval_list_name``,
        ``preprocessing_params_name``, ``recording_id``, ``artifact_detection_id``,
        ``sorter``, ``sorter_params_name``, ``sorting_id``,
        ``curation_id``). Unknown keys raise ``ValueError``.
    as_dict : bool, optional
        Return list of ``{"merge_id": uuid}`` dicts when True;
        a list of UUIDs when False (default).
    """
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput

    return SpikeSortingOutput()._get_restricted_merge_ids_v2(
        restriction, as_dict=as_dict
    )

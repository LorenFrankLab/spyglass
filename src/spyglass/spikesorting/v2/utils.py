"""Shared helpers for the spike sorting tables and pipeline."""

from __future__ import annotations

import os
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

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

if TYPE_CHECKING:
    from pydantic import BaseModel


@contextmanager
def transaction_or_noop(connection):
    """Open a DataJoint transaction unless one is already active.

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
    """True iff ``exc`` is a duplicate-PRIMARY-KEY violation.

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
    all). It matches ONLY the structured errno (``exc.args[0] == 1062``) or
    the specific ``"Duplicate entry"`` text, NEVER a bare ``"1062"``
    substring: a free substring would false-positive on an FK message that
    merely contains those digits (a constraint name, a rendered UUID) and
    silently swallow a genuine integrity failure (e.g. an FK violation)
    that must propagate rather than be treated as the recoverable
    duplicate-PK race.
    """
    if isinstance(exc, dj.errors.DuplicateError):
        return True
    if isinstance(exc, dj.errors.IntegrityError):
        # Structured errno from a raw/untranslated connector error.
        if exc.args and exc.args[0] == 1062:
            return True
        # ER_DUP_ENTRY messages render as "Duplicate entry '...' for key
        # '...'". This phrase is specific to duplicate keys; FK-violation
        # messages ("a foreign key constraint fails") never contain it.
        return any("Duplicate entry" in str(a) for a in exc.args)
    return False


class SelectionMasterInsertGuard:
    """Mixin: reject a direct ``insert`` into a deterministic-id selection
    master.

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
        if not allow_direct_insert:
            raise dj.errors.DataJointError(
                f"Direct insert into {self.__class__.__name__} is not "
                "supported: the primary key is derived from the selection's "
                "full logical identity (and, for the part-bearing masters, "
                "the source-part rows are inserted atomically with it). Use "
                f"{self.__class__.__name__}.insert_selection(). Pass "
                "allow_direct_insert=True only for a deliberate maintenance "
                "or test bypass."
            )
        super().insert(
            rows,
            replace=replace,
            skip_duplicates=skip_duplicates,
            ignore_extra_fields=ignore_extra_fields,
            **kwargs,
        )


# ``CurationSource`` and ``CurationLabel`` are defined in the stdlib-only
# ``_enums`` module and re-exported at the top of this file; see the
# import there for why they live outside ``utils``.


# How a ``SortGroupV2`` references its channels before sorting:
#   * ``"none"`` -- no re-referencing.
#   * ``"global_median"`` -- common-median reference across the group.
#   * ``"specific"`` -- subtract a single named reference electrode
#     (``reference_electrode_id``), then drop it from the sorted channel set.
#
# Stored as a ``varchar(32)`` on ``SortGroupV2`` validated against this
# Literal at insert time (NOT a MySQL enum). The set may grow --
# SpikeInterface also supports ``global_average`` / CAR and local per-group
# referencing -- and an enum would trap a future mode behind a forbidden
# ``ALTER TABLE`` under the zero-migration policy. The Literal gives
# identical typo protection at the ``insert1`` boundary without the migration
# risk; same decision as ``CurationLabel``. It replaced a single
# ``sort_reference_electrode_id`` int whose magic sentinels (-1 none, -2
# global median, >=0 specific) conflated the mode with the channel id.
ReferenceMode = Literal["none", "global_median", "specific"]

_VALID_REFERENCE_MODES: frozenset[str] = frozenset(
    ("none", "global_median", "specific")
)


def _validate_reference_fields(row: dict) -> None:
    """Validate a ``SortGroupV2`` row's reference fields before insert.

    Enforces the two invariants the ``reference_mode`` /
    ``reference_electrode_id`` split must hold:

    1. ``reference_mode`` (defaulting to ``"none"`` when absent) is a
       member of the ``ReferenceMode`` Literal -- the varchar's typo
       guard, replacing what a MySQL enum would have done at the DB.
    2. ``reference_electrode_id`` is non-null **iff**
       ``reference_mode == 'specific'``: a specific reference needs a
       channel to subtract, and a non-specific mode must not carry a
       stray channel id the runtime would silently ignore.
    """
    mode = row.get("reference_mode", "none")
    if mode not in _VALID_REFERENCE_MODES:
        raise ValueError(
            f"SortGroupV2: reference_mode {mode!r} is not a valid "
            f"ReferenceMode. Valid modes: {sorted(_VALID_REFERENCE_MODES)}."
        )
    ref_id = row.get("reference_electrode_id")
    if mode == "specific" and ref_id is None:
        raise ValueError(
            "SortGroupV2: reference_mode='specific' requires a non-null "
            "reference_electrode_id (the channel to subtract)."
        )
    if mode != "specific" and ref_id is not None:
        raise ValueError(
            f"SortGroupV2: reference_mode={mode!r} must leave "
            f"reference_electrode_id NULL; got {ref_id!r}. Only "
            "reference_mode='specific' names a reference electrode."
        )


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
    """
    orphans: list[dict] = []
    for master in master_table.fetch("KEY", as_dict=True):
        if sum(len(part & master) for part in part_tables) == 0:
            orphans.append(master)
    return orphans


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


def assert_reference_not_member(
    reference_mode, reference_electrode_id, sort_group_channel_ids
) -> None:
    """Reject a ``'specific'`` reference that is also a sort-group channel.

    The write path subtracts the specific reference from every channel and then
    REMOVES it from the recording. If the reference electrode is itself a member
    of the sort group, that silently drops a channel the user meant to sort
    (a 4-wire tetrode would sort on 3). v1 silently dropped it via
    ``setdiff1d``; v2 fails loud. No-op for non-``'specific'`` modes (which
    carry no reference electrode).
    """
    if reference_mode != "specific":
        return
    group = {int(c) for c in sort_group_channel_ids}
    if int(reference_electrode_id) in group:
        raise ValueError(
            "SortGroupV2: reference_electrode_id "
            f"{int(reference_electrode_id)} is also a sort-group channel; a "
            "'specific' reference is subtracted from every channel and then "
            "removed, which would silently drop that channel from the sort. "
            "Choose a reference electrode outside the sort group."
        )


# Private sentinel for "derive the reference from the per-electrode config"
# in :func:`resolve_group_reference`. A distinct object (not ``None``) so an
# explicit/configured ``None`` can still mean "no reference" -- ``None`` is a
# real, v1-compatible sentinel value (``original_reference_electrode`` is
# nullable), so it cannot double as the "auto-derive" marker.
_AUTO_REFERENCE = object()


def resolve_group_reference(
    configured_ref_ids,
    *,
    explicit_ref=_AUTO_REFERENCE,
    group_label="",
) -> tuple[ReferenceMode, int | None]:
    """Resolve one electrode group's reference to ``(reference_mode, id)``.

    Maps a sort group's configured or explicitly-supplied reference to a
    validated ``(reference_mode, reference_electrode_id)`` pair that
    satisfies :func:`_validate_reference_fields`. Emits only the three real
    ``ReferenceMode`` values -- never ``"auto"`` -- so the resolved pair can
    be written straight into a ``SortGroupV2`` master row.

    Sentinels (v1-compatible): ``None`` or ``-1`` -> ``("none", None)``;
    ``-2`` -> ``("global_median", None)``; ``>= 0`` -> ``("specific", id)``.

    Parameters
    ----------
    configured_ref_ids : iterable of (int or None)
        The ``original_reference_electrode`` value of every member electrode
        in the group. Only read on the auto-derivation path
        (``explicit_ref is _AUTO_REFERENCE``).
    explicit_ref : int, None, or _AUTO_REFERENCE, optional
        A reference electrode id / sentinel from a ``references`` mapping, or
        the private ``_AUTO_REFERENCE`` sentinel (the default) to derive the
        reference from ``configured_ref_ids``. An explicit value -- including
        an explicit ``None`` ("no reference") -- always wins over config.
    group_label : str, optional
        Group name used in error messages.

    Returns
    -------
    (reference_mode, reference_electrode_id) : tuple of (str, int or None)

    Raises
    ------
    ValueError
        On the auto path, if the members carry mixed
        ``original_reference_electrode`` values (cannot pick one reference;
        v1 silently fell through here). For any path, if the resolved
        sentinel is a negative value other than ``-1`` / ``-2``.

    Notes
    -----
    Membership validation is intentionally outside this resolver: helpers that
    support ``omit_ref_electrode_group`` must honor those skips first, then
    call :func:`assert_reference_not_member` for the groups they actually
    insert.
    """
    if explicit_ref is not _AUTO_REFERENCE:
        ref = -1 if explicit_ref is None else int(explicit_ref)
    else:
        uniq = sorted(
            {-1 if r is None else int(r) for r in configured_ref_ids}
        )
        if len(uniq) == 0:
            raise ValueError(
                "SortGroupV2: electrode group "
                f"{group_label!r} has no member electrodes to auto-derive a "
                "reference from. Pass an explicit `references` entry, or check "
                "the group membership."
            )
        if len(uniq) != 1:
            raise ValueError(
                "SortGroupV2: electrode group "
                f"{group_label!r} has mixed original_reference_electrode "
                f"values {uniq}; cannot auto-derive a single reference. "
                "Pass an explicit `references` entry for this group, or fix "
                "the Electrode config."
            )
        ref = uniq[0]

    if ref == -1:
        return "none", None
    if ref == -2:
        return "global_median", None
    if ref < 0:
        raise ValueError(
            f"SortGroupV2: electrode group {group_label!r} has unrecognized "
            f"reference sentinel {ref}; expected -1 (none), -2 "
            "(global_median), or a non-negative electrode id."
        )
    return "specific", ref


def resolve_conversion_and_offset(recording) -> tuple[float, float]:
    """Resolve the ElectricalSeries ``(conversion, offset)`` for a recording.

    v2 writes traces UNSCALED (``return_in_uV=False``), so the persisted
    ElectricalSeries must carry BOTH the gain (as ``conversion``) and the
    per-channel offset (as ``offset``) to recover physical volts on readback:
    ``volts = raw * conversion + offset``. SpikeInterface stores per-channel
    gain/offset in microvolts (``uV = raw*gain + offset``); a single NWB
    ``conversion``/``offset`` scalar can only represent a UNIFORM gain/offset,
    so heterogeneous values are rejected rather than silently mis-scaled.

    Dropping the offset (the prior behavior, inherited from v1) silently biased
    every channel by ``offset`` uV on readback for recordings with a non-zero
    DC offset (e.g. Intan / Open Ephys). A non-positive gain (``0`` ->
    all-zero recording, negative -> sign flip) is also rejected.

    Returns
    -------
    (conversion, offset) : tuple of float
        Volts-per-count and volts, for the ElectricalSeries.
    """
    import numpy as np

    gains = np.unique(recording.get_channel_gains())
    if len(gains) != 1:
        raise ValueError(
            "resolve_conversion_and_offset: recording has heterogeneous "
            f"channel gains {gains.tolist()}; v2 ElectricalSeries write "
            "requires a single conversion factor. Verify probe metadata for "
            "the sort group."
        )
    if gains[0] <= 0:
        raise ValueError(
            "resolve_conversion_and_offset: recording has a non-positive "
            f"channel gain {float(gains[0])}; cannot scale to volts. Verify "
            "probe metadata for the sort group."
        )
    offsets = np.unique(recording.get_channel_offsets())
    if len(offsets) != 1:
        raise ValueError(
            "resolve_conversion_and_offset: recording has heterogeneous "
            f"channel offsets {offsets.tolist()}; a single ElectricalSeries "
            "offset cannot represent them."
        )
    return float(gains[0]) * 1e-6, float(offsets[0]) * 1e-6


def electrode_table_region(nwbf, electrode_ids, description: str):
    """Build an ElectricalSeries electrode-table region from electrode ids.

    ``pynwb.NWBFile.create_electrode_table_region(region=...)`` interprets
    ``region`` as ROW INDICES into the electrodes table, NOT electrode ids.
    Passing spyglass ``electrode_id`` values directly is correct only when
    ``electrode.id == row position``; for an electrodes table whose ids are
    non-contiguous or reordered it silently points the ElectricalSeries at the
    wrong electrode rows -- wrong channel locations and brain-region
    attribution on readback. Map ids -> row indices via
    ``get_electrode_indices`` so the region is correct for any electrodes
    table, and fail loud on an unknown id rather than aliasing it.

    Parameters
    ----------
    nwbf : pynwb.NWBFile
        File whose ``electrodes`` table the region indexes into.
    electrode_ids : iterable of int
        Spyglass electrode ids (e.g. the recording's channel ids).
    description : str
        Region description.

    Returns
    -------
    hdmf.common.table.DynamicTableRegion
    """
    from spyglass.utils.nwb_helper_fn import (
        get_electrode_indices,
        invalid_electrode_index,
    )

    ids = [int(e) for e in electrode_ids]
    indices = get_electrode_indices(nwbf, ids)
    missing = [
        eid for eid, idx in zip(ids, indices) if idx == invalid_electrode_index
    ]
    if missing:
        raise ValueError(
            "electrode_table_region: electrode ids "
            f"{missing} are not in the NWB electrodes table"
        )
    return nwbf.create_electrode_table_region(
        region=[int(i) for i in indices], description=description
    )


def unit_brain_region_df(unit_relation, resolution: str):
    """Join a Unit-part relation against Electrode * BrainRegion.

    Shared implementation of ``Sorting.get_unit_brain_regions`` and
    ``CurationV2.get_unit_brain_regions``. The Unit relation must
    carry an ``Electrode`` FK; the join walks it to ``BrainRegion``
    (non-null FK on ``Electrode``) and returns a DataFrame with the
    standard column set + a ``region_resolution`` literal label so
    concat-backed callers can distinguish anchor-member results.
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


def _validate_params(model_cls: type[BaseModel], payload: dict) -> dict:
    """Validate a parameter payload against a Pydantic model.

    Parameters
    ----------
    model_cls : type[pydantic.BaseModel]
        The schema to validate against.
    payload : dict
        The raw parameter dict, typically a Lookup row's ``params`` blob.

    Returns
    -------
    dict
        The validated, normalized payload (``model_dump()`` output).

    Raises
    ------
    pydantic.ValidationError
        If ``payload`` does not satisfy ``model_cls``.
    """
    return model_cls.model_validate(payload).model_dump()


def _assert_schema_version_matches(
    row: dict, model_cls: type[BaseModel], *, table_name: str
) -> None:
    """Raise if a Lookup row's ``params_schema_version`` disagrees with
    the inner Pydantic schema_version.

    Each v2 Lookup table stores a ``params_schema_version`` column
    alongside the validated ``params`` blob. The blob also carries a
    ``schema_version`` field (Pydantic-validated). If a user inserts
    a row where the two values disagree, downstream code that
    branches on the outer column will silently route v2 rows to v1
    behavior (or vice versa). This helper catches the drift at
    insert time so the row never lands.

    Parameters
    ----------
    row : dict
        The full row dict being inserted. Must have a ``params``
        entry that already contains a ``schema_version`` (i.e. the
        caller has already run ``_validate_params``).
    model_cls : type[pydantic.BaseModel]
        The schema class. Its default ``schema_version`` is used as
        the fallback when the row omits ``params_schema_version``.
    table_name : str
        Human-readable table name for the error message.

    Raises
    ------
    ValueError
        If ``row['params_schema_version']`` is set and does not match
        ``row['params']['schema_version']``.
    """
    inner = int(row["params"]["schema_version"])
    if "params_schema_version" not in row:
        return
    outer = int(row["params_schema_version"])
    if inner != outer:
        raise ValueError(
            f"{table_name}.insert1: params_schema_version={outer} does "
            f"not match the inner Pydantic schema_version={inner} on the "
            "validated params blob. Drop the column or align it with the "
            "blob's schema_version."
        )


def validate_lookup_rows(
    rows, attr_names, *, schema_for, table_name, per_row_hook=None
):
    """Validate + normalize params-Lookup rows before a (bulk) insert.

    Shared body of the four validated-Lookup ``insert`` overrides
    (``PreprocessingParameters``, ``ArtifactDetectionParameters``,
    ``SorterParameters``, ``MotionCorrectionParameters``): for each row,
    normalize to a dict, validate its ``params`` blob against the row's
    schema, run an optional per-row check, and assert the outer
    ``params_schema_version`` agrees with the validated blob. Returns the
    list of validated row dicts to hand to ``super().insert``. Each table's
    ``insert1`` delegates to ``insert`` (so a single code path validates
    both), mirroring ``CurationV2.UnitLabel``.

    Parameters
    ----------
    rows : iterable
        The rows handed to ``insert`` (mappings, positional sequences, or a
        ``QueryExpression``); normalized per row via ``_insert_row_to_dict``.
    attr_names : Sequence[str]
        ``self.heading.names``, used to label positional rows.
    schema_for : Callable[[dict], type[BaseModel]]
        Picks the schema for a row -- a constant for the single-schema
        tables, per-``sorter`` for ``SorterParameters``.
    table_name : str
        Table name for the drift-check error message.
    per_row_hook : Callable[[dict, type[BaseModel]], None], optional
        Extra table-specific step run after validation and before the
        drift assertion. May mutate the row in place: ``SorterParameters``
        uses it to reject unknown sorter names and to backfill
        ``params_schema_version`` from the validated blob.
    """
    validated = []
    for row in rows:
        row = _insert_row_to_dict(row, attr_names)
        schema_cls = schema_for(row)
        row["params"] = _validate_params(schema_cls, row["params"])
        if per_row_hook is not None:
            per_row_hook(row, schema_cls)
        _assert_schema_version_matches(row, schema_cls, table_name=table_name)
        validated.append(row)
    return validated


def _insert_row_to_dict(row, attr_names) -> dict:
    """Normalize a DataJoint insert row to a mutable dict.

    A bulk ``insert`` accepts both mapping rows (``{"sorter": ...}``,
    what user code passes) and positional sequences (``("mountainsort4",
    name, params, 1, None)``, what each Lookup's ``_DEFAULT_CONTENTS``
    ships to ``insert_default``). The ``insert`` validation overrides
    need a dict in both cases so they can read/rewrite ``row["params"]``;
    a positional tuple is zipped against the table heading's attribute
    order to recover the dict form. A ``QueryExpression`` passed to
    ``insert`` is fine -- iterating it yields per-row dicts, which hit the
    mapping branch.

    A bare ``str``/``bytes`` row is rejected loudly: it is the shape that
    leaks through when a caller passes a ``pandas.DataFrame`` or a CSV
    path to ``insert`` (iterating those yields column-name strings /
    characters), neither of which these validated Lookups support.
    ``zip``-ing a string against the heading would otherwise produce a
    silently malformed row.

    Parameters
    ----------
    row : Mapping | Sequence
        One row from the iterable handed to ``insert``.
    attr_names : Sequence[str]
        The table heading's attribute names in definition order
        (``self.heading.names``), used to label a positional sequence.

    Returns
    -------
    dict
        A shallow-copied dict suitable for in-place ``params`` rewrite.

    Raises
    ------
    TypeError
        If ``row`` is a ``str``/``bytes`` (an unsupported insert form
        for these validated Lookups).
    """
    if isinstance(row, Mapping):
        return dict(row)
    if isinstance(row, (str, bytes)):
        raise TypeError(
            "validated Lookup insert expects each row to be a mapping or "
            f"a positional sequence; got {type(row).__name__}. Pass a list "
            "of dicts -- DataFrame / CSV-path inserts are not supported "
            "on these Pydantic-validated parameter tables."
        )
    return dict(zip(attr_names, row))


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
    if noise_levels is not None and len(noise_levels) not in (1, n_channels):
        raise ValueError(
            "clusterless noise_levels must have length 1 (broadcast) or "
            f"n_channels={n_channels}; got {len(noise_levels)}."
        )


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
    merged = dict(si.get_global_job_kwargs())
    custom = dj.config.get("custom", {}) or {}
    merged.update(custom.get("spikesorting_v2_job_kwargs", {}) or {})
    for override in row_job_kwargs:
        if override:
            merged.update(override)
    return merged


def _ensure_lookup_row_exists(
    lookup_table,
    restriction: dict,
    *,
    helper_name: str,
    insert_default_path: str,
) -> None:
    """Pre-check that a Lookup-row FK target exists before insert_selection.

    Without this guard, a missing Lookup row produces an opaque
    DataJoint ``IntegrityError`` ("foreign key constraint fails")
    that gives the user no hint about which Lookup table is empty or
    how to populate it. Raise a clear ``ValueError`` instead so the
    notebook user can fix the setup in one step.

    Parameters
    ----------
    lookup_table
        The Lookup table class whose row is required (e.g.
        ``PreprocessingParameters``).
    restriction
        The dict identifying the required row (e.g.
        ``{"preprocessing_params_name": "default_franklab"}``).
    helper_name
        Name of the insert_selection helper calling us, for the error
        message (e.g. ``"RecordingSelection.insert_selection"``).
    insert_default_path
        Importable path that loads the default rows (e.g.
        ``"PreprocessingParameters.insert_default()"``).
    """
    if not (lookup_table & restriction):
        raise ValueError(
            f"{helper_name}: required Lookup row not found in "
            f"{lookup_table.__name__} for {restriction}. "
            f"Run {insert_default_path} first to install the default "
            "rows, or insert your custom row before retrying. The "
            "one-shot `spyglass.spikesorting.v2.initialize_v2_defaults()`"
            " installs every required default in one call."
        )


_ARTIFACT_DETECTION_INTERVAL_LIST_PREFIX = "artifact_detection_"


def artifact_detection_interval_list_name(artifact_detection_id) -> str:
    """Return the ``IntervalList.interval_list_name`` for an artifact detection.

    Centralizes the v2 convention
    ``f"artifact_detection_{artifact_detection_id}"`` so the prefix lives in
    one place; ``parse_artifact_detection_interval_list_name`` is its inverse.

    NOTE: v2 prefixes the name with ``artifact_detection_``; v1 wrote the bare
    ``str(artifact_detection_id)`` (``v1/artifact.py:200``). The prefix is
    intentional -- it disambiguates artifact-detection-derived IntervalList rows
    from sort_valid_times / lfp / etc. rows when grepping by name. A
    ported v1 query that looks rows up by the bare UUID returns empty;
    use this helper (or its inverse) instead.
    """
    return f"{_ARTIFACT_DETECTION_INTERVAL_LIST_PREFIX}{artifact_detection_id}"


def parse_artifact_detection_interval_list_name(name: str):
    """Return the artifact-detection id encoded in an IntervalList name.

    Returns ``None`` if ``name`` is not in the artifact-named form,
    matching the merge-dispatcher's "leave non-artifact names alone"
    contract.
    """
    if isinstance(name, str) and name.startswith(
        _ARTIFACT_DETECTION_INTERVAL_LIST_PREFIX
    ):
        return name[len(_ARTIFACT_DETECTION_INTERVAL_LIST_PREFIX) :]
    return None


def get_spiking_sorting_v2_merge_ids(
    restriction: dict, as_dict: bool = False
) -> list:
    """Return merge ids for a v2 spike-sorting restriction.

    Notebook-discoverable parallel of v1's
    ``get_spiking_sorting_v1_merge_ids``
    (``src/spyglass/spikesorting/v1/utils.py:37-109``); thin wrapper
    over ``SpikeSortingOutput()._get_restricted_merge_ids_v2`` so
    users can do ``get_spiking_sorting_v2_merge_ids(restriction)``
    without poking at the private merge-table method directly.

    ``as_dict`` is an enhancement over v1's surface (v1 always
    returned a plain list of UUIDs); the default ``False`` matches
    v1's behavior so a v1 caller copied verbatim works unchanged.

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


def _hash_nwb_recording(analysis_file_name: str) -> str:
    """Return a content hash of a recording's AnalysisNwbfile.

    Delegates to ``AnalysisNwbfile().get_hash`` (the project's blessed
    wrapper over ``NwbfileHasher``) so v2 verification uses the same
    hashing path as the v1 recompute machinery.

    Parameters
    ----------
    analysis_file_name : str
        Name of the AnalysisNwbfile holding the preprocessed recording.

    Returns
    -------
    str
        The ``NwbfileHasher`` digest of the file.
    """
    from spyglass.common.common_nwbfile import AnalysisNwbfile

    return AnalysisNwbfile().get_hash(analysis_file_name)

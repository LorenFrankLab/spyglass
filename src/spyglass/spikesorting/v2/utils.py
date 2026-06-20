"""Shared helpers for the spike sorting tables and pipeline."""

from __future__ import annotations

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
    validate_lookup_rows,
)
from spyglass.spikesorting.v2._nwb_metadata_helpers import (  # noqa: F401
    _hash_nwb_recording,
    electrode_table_region,
    resolve_conversion_and_offset,
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

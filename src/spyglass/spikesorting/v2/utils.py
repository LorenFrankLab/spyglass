"""Shared helpers for the spike sorting tables and pipeline."""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import datajoint as dj
import spikeinterface as si

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
    """
    if connection.in_transaction:
        yield
    else:
        with connection.transaction:
            yield


class MetricsSource(str, Enum):
    """Provenance of metrics attached to a ``CurationV2`` row.

    Matches the enum on the table's ``metrics_source`` column. Promoted
    from a runtime set check so a typo at insert time raises a clear
    ``ValueError`` instead of a DataJoint enum-mismatch error from
    MySQL.
    """

    manual = "manual"
    analyzer_curation = "analyzer_curation"
    figpack = "figpack"


class CurationLabel(str, Enum):
    """Curation labels recognized by ``CurationV2.insert_curation``.

    Members match the v1 convention list at
    ``src/spyglass/spikesorting/v1/curation.py``; v2 promotes the
    list from a docstring to a validated set so a typo raises at
    insert time. The backing ``CurationV2.UnitLabel.curation_label``
    column is a ``varchar(32)``, not a MySQL enum: DataJoint *can*
    declare an enum column (``metrics_source`` on ``CurationV2`` is
    one), but v2 chooses varchar because the label set is open-ended --
    a lab adding a custom label later would otherwise need a forbidden
    ``ALTER TABLE`` under the zero-migration policy. The typo guard is
    enforced in Python on every insert path instead: both
    ``CurationV2.insert_curation`` and a direct
    ``CurationV2.UnitLabel.insert1`` / ``insert`` validate against this
    set (pass ``allow_custom_labels=True`` to opt out).
    """

    accept = "accept"
    mua = "mua"
    noise = "noise"
    artifact = "artifact"
    reject = "reject"


ReferenceMode = Literal["none", "global_median", "specific"]
"""How a ``SortGroupV2`` references its channels before sorting.

* ``"none"`` -- no re-referencing.
* ``"global_median"`` -- common-median reference across the group.
* ``"specific"`` -- subtract a single named reference electrode
  (``reference_electrode_id``), then drop it from the sorted channel set.

Stored as a ``varchar(32)`` on ``SortGroupV2`` validated against this
Literal at insert time (NOT a MySQL enum). The set may grow --
SpikeInterface also supports ``global_average`` / CAR and local per-group
referencing -- and an enum would trap a future mode behind a forbidden
``ALTER TABLE`` under the zero-migration policy. The Literal gives
identical typo protection at the ``insert1`` boundary without the
migration risk; same decision as ``CurationLabel``. It replaced a single
``sort_reference_electrode_id`` int whose magic sentinels (-1 none, -2
global median, >=0 specific) conflated the mode with the channel id.
"""

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

    Shared implementation of ``ArtifactSelection.prune_orphaned_selections``
    and ``SortingSelection.prune_orphaned_selections``. ``part_tables``
    is the list of source-part tables to count against the master --
    e.g. ``[RecordingSource, SharedArtifactGroupSource]`` for artifact
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


def _analyzer_path(key: dict) -> Path:
    """Return the on-disk SortingAnalyzer folder for a sorting row.

    The folder holds regeneratable scratch (waveforms, templates, metric
    extensions); it is not the canonical artifact and lives outside the
    AnalysisNwbfile storage tree, under Spyglass's configured temp directory.

    Parameters
    ----------
    key : dict
        A key containing ``sorting_id``.

    Returns
    -------
    pathlib.Path
        ``{temp_dir}/spikesorting_v2/analyzers/{sorting_id}.analyzer``.
    """
    from spyglass.settings import temp_dir

    return (
        Path(temp_dir)
        / "spikesorting_v2"
        / "analyzers"
        / f"{key['sorting_id']}.analyzer"
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
        ``{"preproc_params_name": "default_franklab"}``).
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


def _get_recording_timestamps(
    recording,
    override=None,
):
    """Return the absolute-time vector for a SpikeInterface recording.

    Two responsibilities:

    1. **Multi-segment NWB support.** SpikeInterface's
       ``recording.get_times()`` only returns the active-segment
       times; a multi-segment NWB (epoch-stitched recordings) would
       silently report just segment 0. This helper concatenates the
       per-segment timestamps into one ``(total_frames,)`` array so
       downstream code sees the whole-session timeline.

    2. **Caller-supplied corrected vector.** The non-monotonic
       timestamp repair (Spyglass v1 in-flight on
       ``copilot/fix-populating-artifact-detection``; ported here as
       a defensive measure) computes a corrected timestamps array
       once at the top of ``Recording.make`` and threads it through
       to subsequent helpers. ``override`` is that hook: when not
       ``None``, return it verbatim. ``Recording.make`` passes the
       repaired array; helpers called outside the make path (e.g.
       ``get_recording``) leave ``override=None`` and get the
       segment-aware ``get_times()`` concatenation.

    Mirrors v1's helper at
    ``src/spyglass/spikesorting/utils.py:260-279`` with the
    additional ``override`` plumbing.

    Parameters
    ----------
    recording : si.BaseRecording
        Source recording whose timestamps are needed.
    override : numpy.ndarray, optional
        Pre-computed timestamps (already monotonicity-corrected, if
        applicable) to return verbatim. Callers that have a
        corrected timestamp array pass it here so this helper does
        not re-derive it from the recording.

    Returns
    -------
    numpy.ndarray
        ``(total_frames,)`` float array of wall-clock seconds.
    """
    import numpy as _np

    if override is not None:
        return _np.asarray(override)

    num_segments = recording.get_num_segments()
    if num_segments <= 1:
        return recording.get_times()

    frames_per_segment = [0] + [
        recording.get_num_frames(segment_index=i)
        for i in range(num_segments)
    ]
    cumsum_frames = _np.cumsum(frames_per_segment)
    total_frames = int(cumsum_frames[-1])

    timestamps = _np.zeros((total_frames,), dtype=_np.float64)
    for i in range(num_segments):
        start_index = int(cumsum_frames[i])
        end_index = int(cumsum_frames[i + 1])
        timestamps[start_index:end_index] = recording.get_times(segment_index=i)
    return timestamps


def _consolidate_intervals(intervals, timestamps):
    """Convert ``(start_time, stop_time)`` intervals to frame indices.

    Ports v1's helper at
    ``src/spyglass/spikesorting/v1/recording.py:715`` with the
    off-by-one bug **fixed**: v1 computed
    ``stop_indices = searchsorted(side="right") - 1`` (an INCLUSIVE
    end index), then passed that value to
    ``frame_slice(end_frame=stop)`` whose ``end_frame`` is
    EXCLUSIVE. Net effect: v1 silently dropped the last sample
    (~33 us at 30 kHz) of every disjoint interval. The v2 port
    keeps ``side="right"`` semantics and uses the result directly
    as the exclusive end -- matching SI's convention. Document the
    v1 divergence in the docstring; the join-overlap condition is
    adjusted accordingly (``stop >= next_start`` instead of
    v1's ``stop >= next_start - 1``).

    Parameters
    ----------
    intervals : array-like
        Iterable of ``(start_seconds, stop_seconds)`` tuples. The
        helper sorts and consolidates overlapping/adjacent
        intervals before returning.
    timestamps : numpy.ndarray
        Strictly increasing wall-clock timestamps for the recording.

    Returns
    -------
    numpy.ndarray
        ``(n_consolidated, 2)`` array of ``(start_frame,
        end_frame_exclusive)`` integer pairs suitable for
        ``recording.frame_slice(start_frame=..., end_frame=...)``.
    """
    import numpy as _np

    intervals = _np.asarray(intervals)
    if intervals.ndim == 1:
        intervals = intervals.reshape(-1, 2)
    if intervals.shape[1] != 2:
        raise ValueError("Input array must have shape (N_Intervals, 2).")

    # Sort defensively; v1 did the same. Stable ordering by start.
    if not _np.all(intervals[:-1] <= intervals[1:]):
        intervals = intervals[_np.argsort(intervals[:, 0])]

    start_indices = _np.searchsorted(
        timestamps, intervals[:, 0], side="left"
    )
    # Exclusive end -- ``side="right"`` returns the count of
    # timestamps <= value, which is exactly the half-open end SI's
    # ``frame_slice`` expects. v1 subtracted 1 here (bug).
    stop_indices = _np.searchsorted(
        timestamps, intervals[:, 1], side="right"
    )

    consolidated = []
    start, stop = int(start_indices[0]), int(stop_indices[0])
    for next_start, next_stop in zip(start_indices, stop_indices):
        next_start = int(next_start)
        next_stop = int(next_stop)
        # Overlap / adjacency in exclusive-end form: next_start <= stop
        # (== means strictly adjacent). v1 used ``stop >= next_start - 1``
        # because its ``stop`` was inclusive; the conditions are
        # equivalent under the off-by-one rewrite.
        if next_start <= stop:
            stop = max(stop, next_stop)
        else:
            consolidated.append((start, stop))
            start, stop = next_start, next_stop

    consolidated.append((start, stop))
    return _np.asarray(consolidated, dtype=_np.int64)


def _spike_times_to_frames(recording_times, spike_times, n_samples, unit_id):
    """Map absolute spike times (seconds) to recording frame indices.

    Spike times are persisted in the recording's ABSOLUTE wall-clock
    timeline (``timestamps[frame]``). For disjoint sort intervals that
    timeline is non-uniform (it carries the wall-clock gaps that
    ``concatenate_recordings(ignore_times=True)`` drops), so the inverse
    map must be ``np.searchsorted`` against the actual timestamps -- NOT
    an affine ``round((t - t_start) * fs)``, which would shift every
    frame after a gap by the accumulated gap and can push frames past
    ``n_samples``. This mirrors v1's ``spike_times_to_valid_samples``
    (``v1/sorting.py:29``) so v2 ``get_sorting`` recovers the same frame
    indices v1 did.

    ``side='left'``: a spike exactly equal to ``recording_times[-1]``
    maps to ``n_samples - 1`` (valid). Floating-point rounding can still
    land a near-final spike at ``n_samples`` (one past the end), which
    SpikeInterface rejects; those out-of-bounds frames are dropped with
    a warning, matching v1.

    Parameters
    ----------
    recording_times : np.ndarray, shape (n_samples,)
        Recording timestamps in seconds, monotonically increasing.
    spike_times : np.ndarray, shape (n_spikes,)
        Spike times for a single unit in seconds.
    n_samples : int
        Total number of samples in the recording.
    unit_id : int or str
        Identifier of the unit, used only in the warning message.

    Returns
    -------
    np.ndarray, shape (n_valid_spikes,)
        Frame indices in ``[0, n_samples)``; out-of-bounds frames
        removed (``n_valid_spikes <= n_spikes``).
    """
    import numpy as _np

    from spyglass.utils import logger

    spike_frames = _np.searchsorted(recording_times, spike_times)
    excess_mask = spike_frames >= n_samples
    n_excess = int(excess_mask.sum())
    if n_excess > 0:
        logger.warning(
            f"Unit {unit_id} has {n_excess} spike(s) exceeding the "
            "recording duration. Removing excess spikes. This may be "
            "caused by floating-point rounding during the seconds-to-"
            "samples conversion."
        )
        spike_frames = spike_frames[~excess_mask]
    return spike_frames


def _base_intervals_from_timestamps(timestamps, fs):
    """Split a (possibly gap-preserving) timestamp vector into recorded chunks.

    A disjoint v2 recording persists the concatenated per-sort-interval
    timestamp slices (``Recording._restrict_recording``'s
    ``timestamps_override``), so consecutive samples WITHIN a chunk differ
    by ~1 sample period while an inter-chunk wall-clock gap shows up as a
    diff > 1.5 sample periods (a missing sample). Returns one
    ``[start, end]`` (inclusive first/last sample times, seconds) per
    chunk so an artifact complement built per chunk never spans a gap --
    the v1-parity behavior (v1 subtracts artifacts from the explicit
    ``sort_interval_valid_times`` at ``v1/artifact.py:327``). A contiguous
    recording yields a single ``[t0, t_end]`` interval, so this is a no-op
    for the common single-interval case.

    The ``1.5 / fs`` threshold is robust against sub-sample timestamp
    jitter (within-chunk diffs are ~1 sample period) and matches the
    ±1.5-sample tolerance used in the spike-time readback; any genuine
    disjoint gap is orders of magnitude larger.

    Parameters
    ----------
    timestamps : array-like, shape (n_samples,)
        Recording timestamps in seconds, monotonically increasing.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    list[list[float]]
        ``[[start, end], ...]`` inclusive per-chunk bounds; ``[]`` for an
        empty input.
    """
    import numpy as _np

    ts = _np.asarray(timestamps, dtype=float)
    if ts.size == 0:
        return []
    sample_period = 1.0 / float(fs)
    # Index i marks a gap when ts[i+1] - ts[i] exceeds 1.5 sample periods
    # (i.e. at least one sample of wall-clock time is missing).
    gap_after = _np.flatnonzero(_np.diff(ts) > 1.5 * sample_period)
    starts = [0, *(int(i) + 1 for i in gap_after)]
    ends = [*(int(i) for i in gap_after), ts.size - 1]
    return [[float(ts[s]), float(ts[e])] for s, e in zip(starts, ends)]


_ARTIFACT_INTERVAL_LIST_PREFIX = "artifact_"


def artifact_interval_list_name(artifact_id) -> str:
    """Return the ``IntervalList.interval_list_name`` for an artifact id.

    Centralizes the v2 convention ``f"artifact_{artifact_id}"`` so the
    prefix lives in one place; ``parse_artifact_interval_list_name``
    is its inverse.

    NOTE: v2 prefixes the name with ``artifact_``; v1 wrote the bare
    ``str(artifact_id)`` (``v1/artifact.py:200``). The prefix is
    intentional -- it disambiguates artifact-derived IntervalList rows
    from sort_valid_times / lfp / etc. rows when grepping by name. A
    ported v1 query that looks rows up by the bare UUID returns empty;
    use this helper (or its inverse) instead.
    """
    return f"{_ARTIFACT_INTERVAL_LIST_PREFIX}{artifact_id}"


def parse_artifact_interval_list_name(name: str):
    """Return the artifact id encoded in an artifact IntervalList name.

    Returns ``None`` if ``name`` is not in the artifact-named form,
    matching the merge-dispatcher's "leave non-artifact names alone"
    contract.
    """
    if isinstance(name, str) and name.startswith(
        _ARTIFACT_INTERVAL_LIST_PREFIX
    ):
        return name[len(_ARTIFACT_INTERVAL_LIST_PREFIX) :]
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
        ``preproc_params_name``, ``recording_id``, ``artifact_id``,
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

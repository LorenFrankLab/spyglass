"""Recompute verification and safe storage reclamation for v2 artifacts.

Covers the ``Recording`` artifact and the ``SortingAnalyzer`` folder. Each
trio of tables:

- ``*Versions``: inventories an artifact's dependencies.
- ``*RecomputeSelection``: plans an attempt under a ``UserEnvironment``.
- ``*Recompute``: regenerates and compares content hashes, so the original is
  deleted only after a verified match in the current environment.

Hashes cover reproducible content (preprocessed traces; deterministic analyzer
extension data), not whole-file digests. ``rounding`` sets the float precision
of the analyzer comparison.

``delete_files()`` raises ``StaleEnvMatchedError`` for a ``matched=1`` row from
another ``UserEnvironment`` unless ``force_stale_env=True``.
"""

from __future__ import annotations

import datetime as dt
import shutil
import tempfile
from pathlib import Path
from typing import NamedTuple, Optional

import datajoint as dj

from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.common.common_user import UserEnvironment
from spyglass.spikesorting.v2._storage import analyzer_cache as _analyzer_cache
from spyglass.spikesorting.v2._storage.analyzer_cache import (
    analyzer_cache_lock,
    analyzer_folder_storage_fingerprint,
)
from spyglass.spikesorting.v2._storage.recompute import (
    ANALYZER_CONTENT_HASH_VERSION,
    ANALYZER_RECOMPUTE_EXTENSIONS,
    analyzer_inventory_refresh_needed,
    analyzer_seed_modes,
    combined_hash,
    compare_hash_dicts,
    current_env_namespaces,
    current_nwb_namespaces,
    env_matches,
    hash_extension_data,
)
from spyglass.spikesorting.v2._recording.fingerprint import (
    TRACE_ROUNDING,
    recording_content_fingerprint,
)
from spyglass.spikesorting.v2.exceptions import (
    AnalyzerFolderMissingError,
    StaleEnvMatchedError,
)
from spyglass.spikesorting.v2.recording import (
    _ELECTRICAL_SERIES_PATH,
    Recording,
    RecordingSelection,
)
from spyglass.spikesorting.v2.sorting import (
    AnalyzerWaveformParameters,
    Sorting,
)
from spyglass.utils import SpyglassMixin, SpyglassMixinPart, logger
from spyglass.utils.dj_helper_fn import bytes_to_human_readable

schema = dj.schema("spikesorting_v2_recompute")

_ZERO_HASH = "0" * 64
# "Analyzer folder absent"; _ZERO_HASH is a zero-unit sort. Non-hex, so it
# cannot collide with a sha256.
_MISSING_HASH = "MISSING_ANALYZER_FOLDER".ljust(64, "0")


def _current_env_id() -> Optional[str]:
    """Return the current ``UserEnvironment`` env_id (inserting if needed)."""
    return UserEnvironment().this_env.get("env_id")


# ---------------------------------------------------------------------
# Shared tri-part dispatch carriers + helpers for the recompute QC tables
# ---------------------------------------------------------------------


class RecomputeFetched(NamedTuple):
    """DB inputs for a recompute table's ``make_compute``.

    No regeneration I/O except rebuilding a missing traces file.

    ``parent_key`` carries its UUID PK as a str. ``regen_inputs`` is ``None``
    for an ``xfail`` row; it keeps UUIDs as ``uuid.UUID``, which DeepHash
    hashes by value, so both fetches agree.
    """

    parent_key: dict
    rounding: int
    xfail_reason: Optional[str]
    regen_inputs: Optional[tuple]


class FetchFailure(NamedTuple):
    """A regeneration input ``make_fetch`` could not resolve.

    A regeneration failure is a recorded ``'error'`` outcome (``matched=0``),
    not a failed populate. The message is carried in place of the input, and
    :func:`_resolved` raises it in ``make_compute`` where the input is used.

    DataJoint runs ``make_fetch`` twice and refuses the insert if the results
    differ, so a message that varies between fetches (e.g. a random temp file
    name) records no row.

    Attributes
    ----------
    message : str
        ``str()`` of the exception the resolution raised.
    """

    message: str


def _resolve_or_failure(resolve, *, what, parent_key):
    """Return ``resolve()``, or a :class:`FetchFailure` if it raises.

    The traceback is logged here, where the original exception is caught;
    ``make_compute`` only re-raises its message.
    """
    try:
        return resolve()
    except Exception as err:  # noqa: BLE001 - recorded as the 'error' outcome
        logger.error(
            f"{what} could not resolve its regeneration inputs for "
            f"{parent_key}; recording matched=0 (retryable).",
            exc_info=True,
        )
        return FetchFailure(str(err))


def _resolved(value):
    """Return a fetched regeneration input, raising a carried failure."""
    if isinstance(value, FetchFailure):
        raise RuntimeError(value.message)
    return value


class RecomputeComputed(NamedTuple):
    """``make_compute`` -> ``make_insert`` carrier for the recompute tables.

    ``outcome`` is ``'xfail'``, ``'error'`` or ``'compare'``. A regeneration
    failure is returned as ``'error'`` (``matched=0``), not raised, so the QC
    result is still recorded. ``stored_hashes`` / ``new_hashes`` are empty
    except on ``'compare'``.
    """

    outcome: str
    err_msg: Optional[str]
    stored_hashes: dict
    new_hashes: dict
    parent_key: dict


class RecordingRegenInputs(NamedTuple):
    """What ``RecordingArtifactRecompute.make_compute`` regenerates from.

    Attributes
    ----------
    current : StoredTraces or FetchFailure
        The canonical artifact, rebuilt first if it was missing (the
        ``Recording.get_recording`` self-heal).
    recording : RecordingFetched, FetchFailure or None
        ``Recording.make_fetch`` of the parent, for the fresh rebuild;
        ``None`` when ``current`` failed (the rebuild inputs were never read).
    """

    current: object
    recording: object


class AnalyzerRecipe(NamedTuple):
    """A validated analyzer recipe: its cache folder and params blob."""

    analyzer_folder: str
    waveform_params: dict


class AnalyzerRegenSource(NamedTuple):
    """The recording, sorting and sorter row an analyzer rebuilds from.

    Attributes
    ----------
    recording : CanonicalRecording
        The sort's artifact-masked effective traces (file rebuilt if missing).
    units : StoredUnits
        The sort's units NWB (its statistics spans are read from it too).
    sorter_row : dict
        The sort's ``SorterParameters`` row.
    job_kwargs : dict
        Execution kwargs resolved by the fetch adapter before computation.
    """

    recording: object
    units: object
    sorter_row: dict
    job_kwargs: dict


class AnalyzerRegenInputs(NamedTuple):
    """What ``SortingAnalyzerRecompute.make_compute`` regenerates from.

    Attributes
    ----------
    sorting_id : uuid.UUID
        The sort, as stored (named in the no-rebuild loader's errors). Kept a
        ``uuid.UUID``: DeepHash hashes it by value.
    waveform_params_name : str
        The recipe verified.
    recipe : AnalyzerRecipe, FetchFailure or None
        The validated recipe; ``None`` for a zero-unit sort, which has no
        analyzer (nothing else is resolved).
    source : AnalyzerRegenSource, FetchFailure or None
        ``None`` when the stored folder was absent at fetch (compute reports
        it missing without reading a source, so no traces file is rebuilt)
        or an earlier input is missing.
    """

    sorting_id: object
    waveform_params_name: str
    recipe: object
    source: object


class RecordingVersionsFetched(NamedTuple):
    """DB inputs for ``RecordingArtifactVersions.make_compute`` (no file I/O)."""

    analysis_file_name: str
    content_hash: str


class RecordingVersionsComputed(NamedTuple):
    """``make_compute`` -> ``make_insert`` for ``RecordingArtifactVersions``."""

    nwb_deps: Optional[dict]
    content_hash: str


class AnalyzerVersionsFetched(NamedTuple):
    """DB inputs for ``SortingAnalyzerVersions.make_compute``.

    The analyzer load and hash run in ``make_compute``, off the transaction.

    Attributes
    ----------
    n_units : int
        The sort's unit count; ``0`` means no analyzer exists.
    analyzer_folder : str
        The (sort, recipe) analyzer cache folder.
    """

    n_units: int
    analyzer_folder: str


class AnalyzerVersionsComputed(NamedTuple):
    """``make_compute`` -> ``make_insert`` for ``SortingAnalyzerVersions``."""

    si_deps: dict
    analyzer_manifest: dict
    analyzer_hash: str


def _recompute_compute(
    parent_key,
    xfail_reason,
    *,
    regen,
    what,
) -> RecomputeComputed:
    """Shared off-transaction compute for the recompute QC tables.

    Every outcome is inserted:

    - ``xfail``: explicit skip.
    - ``error``: caught regeneration failure, ``matched=0``, retryable.
    - ``compare``: hash comparison.

    ``regen`` is a no-arg callable returning ``(stored_hashes, new_hashes)``.
    """
    if xfail_reason:
        return RecomputeComputed(
            outcome="xfail",
            err_msg=f"xfail: {xfail_reason}"[:255],
            stored_hashes={},
            new_hashes={},
            parent_key=parent_key,
        )
    try:
        stored_hashes, new_hashes = regen()
    except Exception as err:  # noqa: BLE001 - record the failure, retryable
        # Log the full traceback; err_msg truncates to 255 chars.
        logger.error(
            f"{what} regeneration failed for {parent_key}; matched=0 "
            "(retryable).",
            exc_info=True,
        )
        return RecomputeComputed(
            outcome="error",
            err_msg=str(err)[:255],
            stored_hashes={},
            new_hashes={},
            parent_key=parent_key,
        )
    return RecomputeComputed(
        outcome="compare",
        err_msg=None,
        stored_hashes=stored_hashes,
        new_hashes=new_hashes,
        parent_key=parent_key,
    )


def _insert_recompute_outcome(
    table, key, outcome, err_msg, stored_hashes, new_hashes, created_at
):
    """Shared ``make_insert`` body: write the QC result atomically.

    ``compare`` inserts the master plus Name/Hash diff rows; ``xfail`` /
    ``error`` insert one ``matched=0`` master row.
    """

    with table._safe_context():
        if outcome == "compare":
            _insert_comparison(
                table, key, stored_hashes, new_hashes, created_at
            )
        else:
            table.insert1(
                {
                    **key,
                    "matched": False,
                    "err_msg": err_msg,
                    "created_at": created_at,
                }
            )


class _RecomputeSelectionMixin:
    """``remove_matched`` for the ``*RecomputeSelection`` tables.

    Subclasses set ``_versions_table`` (the ``*Versions`` table whose primary
    key identifies an artifact) and ``_recompute_table_name`` (the name of the
    ``*Recompute`` table in this module, declared after the selection).
    """

    _versions_table: type
    _recompute_table_name: str

    @classmethod
    def remove_matched(cls, restriction=True, *, dry_run: bool = True) -> int:
        """Remove redundant selection rows for already-verified artifacts.

        Drops selections whose artifact matched in any environment but that
        are not themselves the matched attempt. Uses ``delete`` so a failed
        (``matched=0``) recompute child cascades. Matched selections are kept
        as the verification record.

        Parameters
        ----------
        restriction : optional
            Restricts the selection rows considered. Default all.
        dry_run : bool, optional
            Only count. Default ``True``.

        Returns
        -------
        int
            The number of redundant rows.
        """
        matched = globals()[cls._recompute_table_name] & "matched=1"
        artifact_pk = cls._versions_table.primary_key
        matched_artifacts = (dj.U(*artifact_pk) & matched).fetch(
            "KEY", as_dict=True
        )
        redundant = (cls & restriction & matched_artifacts) - matched.proj()
        # Fetch keys first: deleting the antijoin directly hits MySQL 1093.
        redundant_keys = redundant.fetch("KEY")
        count = len(redundant_keys)
        if dry_run or count == 0:
            logger.info(
                f"remove_matched: {count} redundant rows (dry_run={dry_run})."
            )
            return count
        (cls & redundant_keys).delete(safemode=False)
        return count


class _RecomputeMixin:
    """``recheck`` for the ``*Recompute`` tables."""

    def recheck(self, key) -> bool:
        """Rerun the comparison for one row (after env/file changes).

        Deletes the row (cascading its diff parts, without a prompt) and
        re-populates. Returns whether it now matches.
        """
        (self & key).delete(safemode=False)
        self.populate(key, reserve_jobs=False)
        return bool((self & key & "matched=1"))


# =====================================================================
# Recording artifact recompute
# =====================================================================


@schema
class RecordingArtifactVersions(SpyglassMixin, dj.Computed):
    """Dependency + content inventory for a ``Recording`` artifact."""

    definition = """
    -> Recording
    ---
    nwb_deps=null: blob       # pynwb namespace versions embedded in the file
    content_hash: char(64)      # stored Recording.content_hash (provenance)
    """

    # Tri-part keeps the NWB read outside the populate transaction.
    _parallel_make = True

    def make_fetch(self, key) -> RecordingVersionsFetched:
        """Read the artifact's file name + stored content_hash (no file I/O)."""
        analysis_file_name, content_hash = (Recording & key).fetch1(
            "analysis_file_name", "content_hash"
        )
        return RecordingVersionsFetched(
            analysis_file_name=str(analysis_file_name),
            content_hash=str(content_hash),
        )

    def make_compute(
        self, key, analysis_file_name, content_hash
    ) -> RecordingVersionsComputed:
        """Read embedded pynwb namespace versions off the transaction."""
        abs_path = AnalysisNwbfile.get_abs_path(analysis_file_name)
        nwb_deps = (
            current_nwb_namespaces(abs_path)
            if Path(abs_path).exists()
            else None
        )
        return RecordingVersionsComputed(
            nwb_deps=nwb_deps, content_hash=content_hash
        )

    def make_insert(self, key, nwb_deps, content_hash):
        """Insert the inventory row."""
        self.insert1(
            {**key, "nwb_deps": nwb_deps, "content_hash": content_hash}
        )

    def this_env(self) -> dj.expression.QueryExpression:
        """Restrict to artifacts reproducible in the current environment.

        Keeps rows whose shared pynwb namespaces all match the live
        environment's versions (see
        :func:`~spyglass.spikesorting.v2._storage.recompute.env_matches`).
        Restrict ``self`` first to scope the scan.
        """
        env_deps = current_env_namespaces()
        pk = self.primary_key
        compatible = [
            {field: row[field] for field in pk}
            for row in self.fetch(as_dict=True)
            if env_matches(row["nwb_deps"], env_deps)
        ]
        return self & compatible


@schema
class RecordingArtifactRecomputeSelection(
    _RecomputeSelectionMixin, SpyglassMixin, dj.Manual
):
    """Plan a recording recompute attempt under a labeled environment."""

    _versions_table = RecordingArtifactVersions
    _recompute_table_name = "RecordingArtifactRecompute"

    definition = """
    -> RecordingArtifactVersions
    -> UserEnvironment
    ---
    logged_at_creation=0: bool
    xfail_reason=NULL: varchar(127)
    """

    @classmethod
    def attempt_all(
        cls,
        restriction=True,
        *,
        limit: Optional[int] = None,
        force_attempt: bool = False,
        check_xfail: bool = True,
    ) -> None:
        """Insert a recompute attempt for every eligible artifact (current env).

        Only artifacts passing :meth:`RecordingArtifactVersions.this_env` are
        planned unless ``force_attempt=True``. Known structural impossibilities
        are recorded in ``xfail_reason`` so the recompute records ``matched=0``
        without regenerating.

        There is no ``rounding`` argument: the stored ``content_hash`` fixes
        the precision (``TRACE_ROUNDING`` / ``TIMESTAMP_ROUNDING``).

        Parameters
        ----------
        restriction : dict, str, or list, optional
            Restriction on ``RecordingArtifactVersions``. Default all.
        limit : int, optional
            Plan at most this many artifacts, sampled at random from the
            eligible set. Default None (all).
        force_attempt : bool, optional
            Plan even env-incompatible artifacts (skip the compatibility gate).
            Default False.
        check_xfail : bool, optional
            Screen each artifact for known structural impossibilities and record
            the reason in ``xfail_reason``. Default True.
        """
        env_id = _current_env_id()
        if not env_id:
            logger.warning(
                "No UserEnvironment available; cannot plan recompute attempts."
            )
            return
        versions = RecordingArtifactVersions()
        eligible = versions if force_attempt else versions.this_env()
        source = eligible & restriction
        if limit:
            source = source & dj.condition.Top(limit=limit, order_by="RAND()")
        rows = []
        for version_key in source.fetch("KEY", as_dict=True):
            xfail_reason = None
            if check_xfail:
                _is_xfail, xfail_reason = cls._check_xfail(version_key)
            rows.append(
                {**version_key, "env_id": env_id, "xfail_reason": xfail_reason}
            )
        cls.insert(rows, skip_duplicates=True)

    @classmethod
    def _check_xfail(
        cls,
        key: dict,
        *,
        skip_probe: bool = True,
        skip_pynwb_api: bool = True,
        skip_nwb_spec: bool = True,
    ) -> tuple[bool, Optional[str]]:
        """Detect known structural impossibilities for one recording.

        Flags only failures a recompute can never overcome: missing probe
        info, or a PyNWB-API or NWB-spec incompatibility. Anything else is
        attempted normally.

        Uses no file I/O: prior ``matched=0`` ``err_msg`` patterns, plus an
        ``Electrode * Probe`` presence query for probe info.

        Parameters
        ----------
        key : dict
            A key carrying ``recording_id`` (e.g. a ``RecordingArtifactVersions``
            primary key).
        skip_probe, skip_pynwb_api, skip_nwb_spec : bool, optional
            Enable each xfail pattern. All default True.

        Returns
        -------
        tuple[bool, str | None]
            ``(is_xfail, reason)``; ``reason`` is one of ``"missing_probe_info"``,
            ``"pynwb_api_incompatible"``, ``"nwb_spec_incompatible"``, or None.
        """
        rec_key = {"recording_id": key["recording_id"]}
        prev_runs = RecordingArtifactRecompute & rec_key & "matched=0"

        if skip_probe:
            # Wording raised by _recording.geometry / _recording.preprocessing.
            if bool(prev_runs & 'err_msg LIKE "%probe geometry%"'):
                return True, "missing_probe_info"
            try:
                nwb_file_name = (RecordingSelection & rec_key).fetch1(
                    "nwb_file_name"
                )
                if _recording_missing_probe_info(nwb_file_name):
                    return True, "missing_probe_info"
            except Exception as exc:  # noqa: BLE001 - can't check -> don't flag
                logger.warning(
                    f"Unable to check probe info for {rec_key}; attempting "
                    f"normally: {exc!r}"
                )

        if skip_pynwb_api and bool(
            prev_runs & 'err_msg LIKE "%unexpected keyword%dtype%"'
        ):
            return True, "pynwb_api_incompatible"

        if skip_nwb_spec and bool(
            prev_runs & 'err_msg LIKE "%No spec%namespace%"'
        ):
            return True, "nwb_spec_incompatible"

        return False, None


@schema
class RecordingArtifactRecompute(_RecomputeMixin, SpyglassMixin, dj.Computed):
    """Regenerate a recording artifact and compare trace content hashes."""

    definition = """
    -> RecordingArtifactRecomputeSelection
    ---
    matched: bool
    err_msg=NULL: varchar(255)
    created_at=NULL: datetime
    deleted=0: bool
    """

    class Name(SpyglassMixinPart):
        definition = """
        -> master
        name: varchar(255)
        missing_from: enum('old', 'new')
        """

    class Hash(SpyglassMixinPart):
        definition = """
        -> master
        name: varchar(255)
        """

    @property
    def with_names(self):
        """Join recompute rows to their artifact ``analysis_file_name``."""
        return self * Recording.proj("analysis_file_name")

    def get_parent_key(self, key) -> dict:
        """Return the upstream ``Recording`` key for a recompute row.

        Read off ``key`` without a join: ``Recording`` and the versions table
        both carry a ``content_hash`` attribute.
        """
        return {"recording_id": key["recording_id"]}

    # Tri-part keeps the slow regeneration outside the populate transaction.
    _parallel_make = True

    def make_fetch(self, key) -> RecomputeFetched:
        """Read the recompute inputs.

        Unless the row is ``xfail``, resolves the canonical artifact (rebuilt
        if missing) and ``Recording.make_fetch`` for the fresh rebuild. A
        failure is carried as a :class:`FetchFailure`. ``rounding`` is
        ``TRACE_ROUNDING`` and unused by the recording path.
        """
        xfail_reason = (RecordingArtifactRecomputeSelection & key).fetch1(
            "xfail_reason"
        )
        parent = self.get_parent_key(key)
        # str UUID for a DeepHash-stable carrier.
        parent_key = {"recording_id": str(parent["recording_id"])}
        return RecomputeFetched(
            parent_key=parent_key,
            rounding=TRACE_ROUNDING,
            xfail_reason=xfail_reason,
            regen_inputs=(
                None
                if xfail_reason
                else _resolve_recording_regen_inputs(parent_key)
            ),
        )

    def make_compute(
        self,
        key,
        parent_key,
        rounding,
        xfail_reason,
        regen_inputs,
    ) -> RecomputeComputed:
        """Fingerprint the current file + a fresh rebuild, off the transaction.

        ``stored_hashes`` fingerprints the current on-disk file and
        ``new_hashes`` a fresh temp rebuild. The only DB access is staging the
        rebuild (see :mod:`._storage.nwb`).
        """
        from spyglass.spikesorting.v2._storage.nwb import read_stored_traces

        def _regen():
            current_traces = _resolved(regen_inputs.current)
            read_stored_traces(current_traces)
            current = recording_content_fingerprint(
                current_traces.abs_path,
                electrical_series_path=_ELECTRICAL_SERIES_PATH,
            )
            fresh = _recompute_recording_fingerprint(
                _resolved(regen_inputs.recording)
            )
            return current, fresh

        return _recompute_compute(
            parent_key,
            xfail_reason,
            regen=_regen,
            what="RecordingArtifactRecompute",
        )

    def make_insert(
        self, key, outcome, err_msg, stored_hashes, new_hashes, parent_key
    ):
        """Record the QC result; ``matched`` anchors on ``content_hash``.

        ``created_at`` is the file mtime, read here because it can differ
        between the two ``make_fetch`` calls. See
        :func:`_insert_recording_comparison` for ``matched``.
        """
        created_at = _artifact_created_at(parent_key)
        if outcome != "compare":
            _insert_recompute_outcome(
                self, key, outcome, err_msg, {}, {}, created_at
            )
            return
        content_hash = (Recording & parent_key).fetch1("content_hash")
        _insert_recording_comparison(
            self, key, stored_hashes, new_hashes, content_hash, created_at
        )

    def get_disk_space(self, restriction=True) -> str:
        """Report reclaimable disk for matched, not-yet-deleted artifacts."""
        return _reclaimable_disk(self.with_names & restriction)

    def update_secondary(self, restriction=True) -> None:
        """Backfill ``created_at`` from the artifact file mtime."""
        for key in (self & restriction).fetch("KEY", as_dict=True):
            self.update1(
                {
                    **key,
                    "created_at": _artifact_created_at(
                        self.get_parent_key(key)
                    ),
                }
            )

    def delete_files(
        self,
        restriction=True,
        *,
        dry_run: bool = True,
        force_stale_env: bool = False,
        days_since_creation: int = 7,
    ) -> list:
        """Delete recording artifacts verified in the CURRENT environment.

        Refuses ``matched=0`` rows and, by default, rows matched only in a
        stale ``UserEnvironment``. Removes the preprocessed ``AnalysisNwbfile``
        and sets ``deleted``; ``Recording.get_recording`` rebuilds it on
        demand.

        Parameters
        ----------
        restriction : optional
            Restricts the recompute rows. Default all.
        dry_run : bool, optional
            Report without deleting. Default ``True``.
        force_stale_env : bool, optional
            Allow rows matched in another environment. Default ``False``.
        days_since_creation : int, optional
            Only delete artifacts at least this many days old. Default 7.

        Returns
        -------
        list[str]
            Paths deleted, or that would be deleted in a dry run.

        Raises
        ------
        ValueError
            If ``days_since_creation`` is negative.
        StaleEnvMatchedError
            If a row matched only in another environment and
            ``force_stale_env`` is ``False``.
        """
        _assert_nonnegative_age(days_since_creation)
        return _delete_files(
            self,
            Recording,
            restriction,
            dry_run=dry_run,
            force_stale_env=force_stale_env,
            days_since_creation=days_since_creation,
            file_attr="analysis_file_name",
            path_fn=AnalysisNwbfile.get_abs_path,
            artifact_pk=Recording.primary_key,
        )


def _recording_missing_probe_info(nwb_file_name: str) -> bool:
    """Whether no ``Electrode * Probe`` rows are on record for ``nwb_file_name``.

    The rebuild needs probe geometry, so such an artifact can never be
    regenerated.
    """
    from spyglass.common.common_device import Probe
    from spyglass.common.common_ephys import Electrode

    return not bool(Electrode * Probe & {"nwb_file_name": nwb_file_name})


def _resolve_recording_regen_inputs(parent_key: dict) -> RecordingRegenInputs:
    """Resolve what a recording recompute regenerates from (DB, self-heal).

    Resolves the canonical artifact (rebuilt if missing), then
    ``Recording.make_fetch``. A failure stops resolution and is carried as a
    :class:`FetchFailure`.
    """
    what = "RecordingArtifactRecompute"
    current = _resolve_or_failure(
        lambda: Recording().resolve_stored_traces(parent_key),
        what=what,
        parent_key=parent_key,
    )
    if isinstance(current, FetchFailure):
        return RecordingRegenInputs(current=current, recording=None)
    return RecordingRegenInputs(
        current=current,
        recording=_resolve_or_failure(
            lambda: Recording().make_fetch(parent_key),
            what=what,
            parent_key=parent_key,
        ),
    )


def _recompute_recording_fingerprint(fetched) -> dict:
    """Recompute a recording to a fresh (unregistered) temp file and return its
    content-fingerprint component dict.

    ``fetched`` is ``Recording.make_fetch`` of the recording. The temp file
    is always unlinked and never enters the canonical slot.
    """
    result = Recording._compute_recording_artifact(fetched)
    fresh_abs = AnalysisNwbfile.get_abs_path(result.analysis_file_name)
    try:
        return recording_content_fingerprint(
            fresh_abs, electrical_series_path=_ELECTRICAL_SERIES_PATH
        )
    finally:
        Path(fresh_abs).unlink(missing_ok=True)


def _insert_recording_comparison(
    table, key, current, fresh, content_hash, created_at
):
    """Insert a recording recompute outcome with a content-anchored ``matched``.

    ``matched`` is ``combined_hash(fresh) == content_hash``, the invariant
    ``Recording._rebuild_nwb_artifact`` enforces, so a match authorizes a
    delete the rebuild can honor. The ``Name`` / ``Hash`` parts still report
    which component differs between the current file and the rebuild.
    """

    matched = combined_hash(fresh) == content_hash
    with table._safe_context():
        _insert_comparison(
            table, key, current, fresh, created_at, matched=matched
        )


# =====================================================================
# SortingAnalyzer recompute
# =====================================================================


@schema
class SortingAnalyzerVersions(SpyglassMixin, dj.Computed):
    """Dependency + content inventory for a ``Sorting``'s analyzer folders.

    One row per (sort, analyzer recipe): the sort's display recipe plus any
    whitened metric recipe a ``CurationEvaluationSelection`` references. Each
    recipe has its own folder, ``{sorting_id}__{waveform_params_name}.analyzer``.
    """

    definition = """
    -> Sorting
    -> AnalyzerWaveformParameters.proj(waveform_params_name="waveform_params_name")
    ---
    si_deps=null: blob          # spikeinterface version, etc.
    analyzer_manifest=null: blob # extension_content_hashes + base_extension_seed_modes
    analyzer_hash: char(64)
    """

    @property
    def key_source(self):
        """The (sort, recipe) pairs that have an analyzer folder.

        Every sort's display recipe plus every metric recipe a curation
        evaluation that requests PCs references.
        """
        from spyglass.spikesorting.v2.metric_curation import (
            CurationEvaluationSelection,
        )

        # OR-list semijoin, not a dj.U union (its headings cannot be joined).
        all_pairs = (AnalyzerWaveformParameters * Sorting).proj()
        is_display = Sorting.proj(
            waveform_params_name="display_waveform_params_name"
        )
        # sorting_id is a secondary attribute here; a bare proj() would drop it
        # and match every sort on waveform_params_name alone.
        is_metric = CurationEvaluationSelection.pc_requesting().proj(
            "sorting_id", waveform_params_name="metric_waveform_params_name"
        )
        return all_pairs & [is_display, is_metric]

    # Tri-part keeps the analyzer load and hashing outside the transaction.
    _parallel_make = True

    def make_fetch(self, key) -> AnalyzerVersionsFetched:
        """Read the unit count and resolve the analyzer folder.

        Validates the recipe (:func:`_validated_analyzer_recipe`) for a sort
        with units before any folder is read.
        """
        sorting_id = key["sorting_id"]
        name = key["waveform_params_name"]
        n_units = int((Sorting & {"sorting_id": sorting_id}).fetch1("n_units"))
        analyzer_folder = (
            _validated_analyzer_recipe(sorting_id, name).analyzer_folder
            if n_units > 0
            else str(_analyzer_cache.analyzer_path(sorting_id, name))
        )
        return AnalyzerVersionsFetched(
            n_units=n_units, analyzer_folder=analyzer_folder
        )

    def make_compute(
        self, key, n_units, analyzer_folder
    ) -> AnalyzerVersionsComputed:
        """Load the analyzer and hash its extensions, without DB access.

        An absent or unloadable folder is inventoried as ``_MISSING_HASH``,
        not rebuilt, so reclaimed disk is not re-materialized.
        """
        import spikeinterface as si

        from spyglass.spikesorting.v2._sorting.analyzer import (
            load_analyzer_folder_no_rebuild,
        )

        si_deps = {"spikeinterface": si.__version__}
        if n_units == 0:
            # SI cannot build an analyzer over zero units.
            return AnalyzerVersionsComputed(
                si_deps=si_deps, analyzer_manifest={}, analyzer_hash=_ZERO_HASH
            )
        try:
            analyzer = load_analyzer_folder_no_rebuild(
                Path(analyzer_folder),
                recipe_label=key["waveform_params_name"],
                sorting_id=key["sorting_id"],
            )
            # Seed modes are provenance only; content hashes set analyzer_hash.
            content_hashes = hash_extension_data(analyzer)
            manifest = {
                "extension_content_hash_version": ANALYZER_CONTENT_HASH_VERSION,
                "extension_content_hashes": content_hashes,
                "base_extension_seed_modes": analyzer_seed_modes(analyzer),
                "storage_fingerprint": analyzer_folder_storage_fingerprint(
                    analyzer_folder
                ),
            }
        except AnalyzerFolderMissingError as exc:
            # Also catches AnalyzerFolderInvalidError (a subclass).
            logger.warning(
                "SortingAnalyzerVersions: analyzer folder unavailable for "
                f"sorting_id={key['sorting_id']}, "
                f"recipe={key['waveform_params_name']}; inventorying as MISSING "
                f"(not rebuilding). {type(exc).__name__}: {exc}"
            )
            return AnalyzerVersionsComputed(
                si_deps=si_deps,
                analyzer_manifest={},
                analyzer_hash=_MISSING_HASH,
            )
        return AnalyzerVersionsComputed(
            si_deps=si_deps,
            analyzer_manifest=manifest,
            analyzer_hash=(
                combined_hash(content_hashes) if content_hashes else _ZERO_HASH
            ),
        )

    def make_insert(self, key, si_deps, analyzer_manifest, analyzer_hash):
        """Insert the analyzer inventory row."""
        self.insert1(
            {
                **key,
                "si_deps": si_deps,
                "analyzer_manifest": analyzer_manifest,
                "analyzer_hash": analyzer_hash,
            }
        )

    @classmethod
    def refresh_changed_folders(cls, restriction=True) -> int:
        """Refresh inventories whose analyzer-folder generation changed.

        Catches folders rebuilt by DB-free workers. Dependent selections and
        verdicts describe the prior folder, so they are removed and the
        current folder is inventoried.

        Parameters
        ----------
        restriction : optional
            Restricts the inventories checked. Default all.

        Returns
        -------
        int
            The number of inventories refreshed.
        """
        refreshed = 0
        inventories = cls & restriction
        # A folder freed by delete_files is absent on purpose.
        reclaimed_keys = {
            (row["sorting_id"], row["waveform_params_name"])
            for row in (
                SortingAnalyzerRecompute & "deleted=1" & inventories.proj()
            ).fetch("sorting_id", "waveform_params_name", as_dict=True)
        }
        for row in inventories.fetch(as_dict=True):
            key = {field: row[field] for field in cls.primary_key}
            folder = _analyzer_cache.analyzer_path(
                key["sorting_id"], key["waveform_params_name"]
            )
            reclaimed = (
                key["sorting_id"],
                key["waveform_params_name"],
            ) in reclaimed_keys
            with analyzer_cache_lock(key["sorting_id"]):
                current = (
                    analyzer_folder_storage_fingerprint(folder)
                    if folder.exists()
                    else None
                )
                if not analyzer_inventory_refresh_needed(
                    row.get("analyzer_manifest"), current, reclaimed=reclaimed
                ):
                    continue
                invalidate_sorting_analyzer_inventory(
                    key["sorting_id"], key["waveform_params_name"]
                )
                repopulate_sorting_analyzer_inventory(
                    key["sorting_id"], key["waveform_params_name"]
                )
            refreshed += 1
        return refreshed


@schema
class SortingAnalyzerRecomputeSelection(
    _RecomputeSelectionMixin, SpyglassMixin, dj.Manual
):
    """Plan an analyzer recompute attempt under a labeled environment."""

    _versions_table = SortingAnalyzerVersions
    _recompute_table_name = "SortingAnalyzerRecompute"

    definition = """
    -> SortingAnalyzerVersions
    -> UserEnvironment
    rounding=4: int
    ---
    logged_at_creation=0: bool
    xfail_reason=NULL: varchar(127)
    """

    @classmethod
    def attempt_all(cls, restriction=True, *, rounding: int = 4) -> None:
        """Insert an analyzer recompute attempt for every eligible sort."""
        if rounding < 0:
            raise ValueError(
                f"rounding must be a non-negative np.round precision; "
                f"got {rounding}."
            )
        env_id = _current_env_id()
        if not env_id:
            logger.warning(
                "No UserEnvironment available; cannot plan recompute attempts."
            )
            return
        SortingAnalyzerVersions.refresh_changed_folders(restriction)
        rows = [
            {**version_key, "env_id": env_id, "rounding": rounding}
            for version_key in (SortingAnalyzerVersions & restriction).fetch(
                "KEY", as_dict=True
            )
        ]
        cls.insert(rows, skip_duplicates=True)


@schema
class SortingAnalyzerRecompute(_RecomputeMixin, SpyglassMixin, dj.Computed):
    """Regenerate an analyzer folder and compare extension content hashes."""

    definition = """
    -> SortingAnalyzerRecomputeSelection
    ---
    matched: bool
    err_msg=NULL: varchar(255)
    created_at=NULL: datetime
    deleted=0: bool
    """

    class Name(SpyglassMixinPart):
        definition = """
        -> master
        name: varchar(255)
        missing_from: enum('old', 'new')
        """

    class Hash(SpyglassMixinPart):
        definition = """
        -> master
        name: varchar(255)
        """

    @property
    def with_names(self):
        """Join recompute rows to the upstream ``sorting_id``."""
        return self * Sorting.proj()

    def get_parent_key(self, key) -> dict:
        """Return the upstream ``Sorting`` key for a recompute row.

        ``sorting_id`` is in every recompute primary key.
        """
        return {"sorting_id": key["sorting_id"]}

    # Tri-part keeps the slow regeneration outside the populate transaction.
    _parallel_make = True

    def make_fetch(self, key) -> RecomputeFetched:
        """Read the recompute inputs.

        Unless the row is ``xfail``, resolves
        :func:`_resolve_analyzer_regen_inputs`. A failure is carried as a
        :class:`FetchFailure`.
        """
        rounding, xfail_reason = (
            SortingAnalyzerRecomputeSelection & key
        ).fetch1("rounding", "xfail_reason")
        parent = self.get_parent_key(key)
        # str UUID for a DeepHash-stable carrier.
        parent_key = {"sorting_id": str(parent["sorting_id"])}
        return RecomputeFetched(
            parent_key=parent_key,
            rounding=int(rounding),
            xfail_reason=xfail_reason,
            regen_inputs=(
                None
                if xfail_reason
                else _resolve_analyzer_regen_inputs(
                    parent_key, key["waveform_params_name"]
                )
            ),
        )

    def make_compute(
        self,
        key,
        parent_key,
        rounding,
        xfail_reason,
        regen_inputs,
    ) -> RecomputeComputed:
        """Regenerate the analyzer folder + hash extensions off the transaction.

        Reads only ``regen_inputs``; no DB access.
        """
        return _recompute_compute(
            parent_key,
            xfail_reason,
            regen=lambda: _recompute_analyzer_hashes(regen_inputs, rounding),
            what="SortingAnalyzerRecompute",
        )

    def make_insert(
        self, key, outcome, err_msg, stored_hashes, new_hashes, parent_key
    ):
        """Record the QC result (created_at = populate time for analyzers)."""
        _insert_recompute_outcome(
            self,
            key,
            outcome,
            err_msg,
            stored_hashes,
            new_hashes,
            dt.datetime.now(),
        )

    def get_disk_space(self, restriction=True) -> str:
        """Report reclaimable disk for matched, not-yet-deleted analyzers.

        Counts each folder once across env rows.
        """
        total = 0
        reclaimable = self & restriction & "matched=1 AND deleted=0"
        for sid, name in {
            (key["sorting_id"], key["waveform_params_name"])
            for key in reclaimable.fetch("KEY", as_dict=True)
        }:
            folder = _analyzer_cache.analyzer_path(sid, name)
            if folder.exists():
                total += sum(
                    f.stat().st_size for f in folder.rglob("*") if f.is_file()
                )
        return f"Total: {bytes_to_human_readable(total)}"

    def update_secondary(self, restriction=True) -> None:
        """Backfill ``created_at`` (analyzer folders use populate time)."""
        for key in (self & restriction).fetch("KEY", as_dict=True):
            self.update1({**key, "created_at": dt.datetime.now()})

    def delete_files(
        self,
        restriction=True,
        *,
        dry_run: bool = True,
        force_stale_env: bool = False,
        days_since_creation: int = 7,
    ) -> list:
        """Delete analyzer folders verified in the current environment.

        Same gates as :meth:`RecordingArtifactRecompute.delete_files`.
        ``Sorting.get_analyzer`` rebuilds a deleted folder.

        Parameters
        ----------
        restriction : optional
            Restricts the recompute rows. Default all.
        dry_run : bool, optional
            Report without deleting. Default ``True``.
        force_stale_env : bool, optional
            Allow rows matched in another environment. Default ``False``.
        days_since_creation : int, optional
            Only delete rows at least this many days old. Default 7.

        Returns
        -------
        list[str]
            Folders deleted, or that would be deleted in a dry run.

        Raises
        ------
        ValueError
            If ``days_since_creation`` is negative.
        StaleEnvMatchedError
            If a row matched only in another environment and
            ``force_stale_env`` is ``False``.
        """
        _assert_nonnegative_age(days_since_creation)
        return _delete_analyzer_folders(
            self,
            restriction,
            dry_run=dry_run,
            force_stale_env=force_stale_env,
            days_since_creation=days_since_creation,
            folder_fn=_analyzer_cache.analyzer_path,
            artifact_pk=SortingAnalyzerVersions.primary_key,
        )


def _validated_analyzer_recipe(
    sorting_id, waveform_params_name: str
) -> AnalyzerRecipe:
    """Validate a sort's analyzer recipe and resolve its cache folder.

    Checks the name is path-safe and names a tracked
    ``AnalyzerWaveformParameters`` row, as ``Sorting.get_analyzer`` does,
    before any folder is read.
    """
    from spyglass.spikesorting.v2._storage.analyzer_cache import (
        assert_path_safe_waveform_params_name,
    )
    from spyglass.spikesorting.v2._sorting.analyzer import fetch_waveform_params

    assert_path_safe_waveform_params_name(waveform_params_name)
    return AnalyzerRecipe(
        analyzer_folder=str(
            _analyzer_cache.analyzer_path(sorting_id, waveform_params_name)
        ),
        waveform_params=fetch_waveform_params(waveform_params_name),
    )


def invalidate_sorting_analyzer_inventory(
    sorting_id, waveform_params_name
) -> bool:
    """Remove an inventory and verdicts tied to a folder being rebuilt.

    The inventory describes one on-disk generation, so its verdicts are stale
    after a rebuild. Returns whether a row existed, so the rebuild path can
    restore the inventory afterwards.
    """
    restriction = {
        "sorting_id": sorting_id,
        "waveform_params_name": waveform_params_name,
    }
    inventory = SortingAnalyzerVersions & restriction
    if not inventory:
        return False
    # Delete selections first so per-env recompute rows cascade.
    selections = SortingAnalyzerRecomputeSelection & restriction
    if selections:
        selections.delete(safemode=False)
    inventory.delete(safemode=False)
    return True


def repopulate_sorting_analyzer_inventory(
    sorting_id, waveform_params_name
) -> None:
    """Inventory a freshly published analyzer generation immediately."""
    SortingAnalyzerVersions.populate(
        {
            "sorting_id": sorting_id,
            "waveform_params_name": waveform_params_name,
        },
        reserve_jobs=False,
    )


def _resolve_analyzer_regen_inputs(
    sort_key: dict, waveform_params_name: str
) -> AnalyzerRegenInputs:
    """Resolve what an analyzer recompute regenerates from (DB, self-heal).

    In the order :func:`_recompute_analyzer_hashes` uses them:

    - the unit count (a zero-unit sort has nothing to verify);
    - the validated recipe;
    - only if the stored folder exists: the canonical recording (traces
      rebuilt if missing), the units NWB and the sorter row.

    A failure stops resolution and is carried as a :class:`FetchFailure`.
    """
    from spyglass.spikesorting.v2._sorting.analyzer import (
        resolve_canonical_recording,
    )
    from spyglass.spikesorting.v2.sorting import SortingSelection
    from spyglass.spikesorting.v2._sorting.fetch import (
        fetch_sorter_analyzer_inputs,
    )

    what = "SortingAnalyzerRecompute"
    sorting_id, n_units = (Sorting & sort_key).fetch1("sorting_id", "n_units")
    if int(n_units) == 0:
        return AnalyzerRegenInputs(
            sorting_id, waveform_params_name, recipe=None, source=None
        )

    def _source():
        canonical = resolve_canonical_recording(sort_key)
        sorter_row, job_kwargs = fetch_sorter_analyzer_inputs(sort_key)
        return AnalyzerRegenSource(
            recording=canonical,
            units=SortingSelection.resolve_stored_units(
                (Sorting & sort_key).fetch1("analysis_file_name"),
                canonical.source,
            ),
            sorter_row=sorter_row,
            job_kwargs=job_kwargs,
        )

    recipe = _resolve_or_failure(
        lambda: _validated_analyzer_recipe(sorting_id, waveform_params_name),
        what=what,
        parent_key=sort_key,
    )
    source = None
    if (
        not isinstance(recipe, FetchFailure)
        and Path(recipe.analyzer_folder).exists()
    ):
        source = _resolve_or_failure(_source, what=what, parent_key=sort_key)
    return AnalyzerRegenInputs(
        sorting_id, waveform_params_name, recipe=recipe, source=source
    )


def _recompute_analyzer_hashes(inputs: AnalyzerRegenInputs, rounding: int):
    """Hash a recipe's stored analyzer and a fresh temp rebuild; no DB access.

    ``inputs.waveform_params_name`` selects the recipe (display or whitened
    metric); the rebuild uses the same recipe's params.
    """
    from spyglass.spikesorting.v2._sorting.analyzer import (
        build_analyzer,
        load_analyzer_folder_no_rebuild,
        read_canonical_recording,
    )
    from spyglass.spikesorting.v2._storage.units_nwb import (
        read_sorting_statistics_spans,
        read_stored_units,
    )

    if inputs.recipe is None:
        return {}, {}  # zero-unit sort: trivially matched
    recipe = _resolved(inputs.recipe)
    # No self-heal: rebuilding a missing original would compare two fresh
    # builds. AnalyzerFolderMissingError records matched=0.
    stored = load_analyzer_folder_no_rebuild(
        Path(recipe.analyzer_folder),
        recipe_label=inputs.waveform_params_name,
        sorting_id=inputs.sorting_id,
    )
    stored_hashes = hash_extension_data(stored, rounding=rounding)

    if inputs.source is None:
        # make_fetch found no stored folder and resolved no source; one has
        # appeared since.
        raise RuntimeError(
            "SortingAnalyzerRecompute: the analyzer folder "
            f"{recipe.analyzer_folder} was absent when the inputs were "
            "fetched and present at compute; recheck the row to verify it."
        )
    source = _resolved(inputs.source)
    # Read canonical inputs so the display analyzer is never rebuilt here.
    recording = read_canonical_recording(source.recording)
    sorting = read_stored_units(source.units)

    from spyglass.settings import temp_dir as spyglass_temp_dir

    with tempfile.TemporaryDirectory(
        prefix="v2_analyzer_recompute_",
        dir=spyglass_temp_dir,
        ignore_cleanup_errors=True,
    ) as tmp:
        # Build only the hashed extensions; all are seed-pinned.
        from spyglass.spikesorting.v2._storage.analyzer_cache import (
            ANALYZER_FOLDER_SUFFIX,
            load_analyzer_folder,
        )

        fresh_folder = Path(tmp) / f"analyzer{ANALYZER_FOLDER_SUFFIX}"
        build_analyzer(
            sorting,
            recording,
            {"sorting_id": str(inputs.sorting_id)},
            sorter_row=source.sorter_row,
            job_kwargs=source.job_kwargs,
            analyzer_folder=fresh_folder,
            waveform_params=recipe.waveform_params,
            extensions=ANALYZER_RECOMPUTE_EXTENSIONS,
            # Same spans as the stored build, so noise_levels matches.
            statistics_spans=read_sorting_statistics_spans(
                source.units.abs_path, sorting_id=inputs.sorting_id
            ),
        )
        fresh = load_analyzer_folder(fresh_folder)
        new_hashes = hash_extension_data(fresh, rounding=rounding)
    return stored_hashes, new_hashes


# =====================================================================
# Shared helpers
# =====================================================================


def _artifact_created_at(rec_key: dict):
    """Return the recording artifact file mtime (now() if missing)."""
    fname = (Recording & rec_key).fetch1("analysis_file_name")
    # Only the mtime is read, so skip get_abs_path's whole-file checksum.
    abs_path = Path(AnalysisNwbfile.get_abs_path(fname, from_schema=True))
    if not abs_path.exists():
        return dt.datetime.now()
    return dt.datetime.fromtimestamp(abs_path.stat().st_mtime)


def _insert_comparison(
    table, key, stored_hashes, new_hashes, created_at, *, matched=None
):
    """Insert the master row plus Name/Hash diff part rows.

    ``matched`` defaults to the two hash dicts being equal.
    """
    dicts_match, missing_old, missing_new, differing = compare_hash_dicts(
        stored_hashes, new_hashes
    )
    if matched is None:
        matched = dicts_match
    table.insert1({**key, "matched": matched, "created_at": created_at})
    name_rows = [
        {**key, "name": n, "missing_from": "old"} for n in missing_old
    ] + [{**key, "name": n, "missing_from": "new"} for n in missing_new]
    if name_rows:
        table.Name().insert(name_rows)
    if differing:
        table.Hash().insert([{**key, "name": n} for n in differing])


def _authorize_artifacts_for_deletion(
    recompute_table, restriction, *, force_stale_env, artifact_pk
):
    """Return ``(authorized, matched_query)`` for deletion.

    ``authorized`` lists ``(artifact_key, authorizing_rows)``. An artifact
    is authorized by a ``matched=1``, not-deleted row in the current
    ``UserEnvironment``; a stale-env row never blocks it. If all its matched
    rows are stale, raises ``StaleEnvMatchedError`` unless
    ``force_stale_env`` (logged). ``authorizing_rows`` are the rows that
    grant it, and the only ones the age gate checks.
    """
    current_env = _current_env_id()
    matched = recompute_table & restriction & "matched=1 AND deleted=0"
    authorized = []
    for artifact in (dj.U(*artifact_pk) & matched).fetch("KEY", as_dict=True):
        artifact_rows = matched & artifact
        current_rows = artifact_rows & {"env_id": current_env}
        if current_rows:
            authorized.append((artifact, current_rows))
            continue
        stale = sorted(set(artifact_rows.fetch("env_id")))
        if force_stale_env:
            logger.warning(
                f"force_stale_env=True: {artifact} authorized by stale-env "
                f"matches {stale} (current env {current_env!r} has no match). "
                "Audit-logged."
            )
            authorized.append((artifact, artifact_rows))
        else:
            raise StaleEnvMatchedError(
                f"No matched recompute in current env {current_env!r} for "
                f"{artifact}. Stale-env matches: {stale}. Rerun the recompute "
                "under the current environment, or pass force_stale_env=True "
                "(audit-logged)."
            )
    return authorized, matched


def _assert_nonnegative_age(days_since_creation: int) -> None:
    """Reject a negative ``days_since_creation``, which would pass every age."""
    if days_since_creation < 0:
        raise ValueError(
            "days_since_creation must be >= 0; a negative value moves the age "
            f"floor into the future (got {days_since_creation})."
        )


def _recent_cutoff(days_since_creation: int):
    return dt.datetime.now() - dt.timedelta(days=days_since_creation)


def _too_recent_or_unknown(matched_rows, cutoff) -> bool:
    """True if any matched row's ``created_at`` is NULL or newer than cutoff."""
    return any(
        created_at is None or created_at > cutoff
        for created_at in matched_rows.fetch("created_at")
    )


def _delete_files(
    recompute_table,
    parent_table,
    restriction,
    *,
    dry_run,
    force_stale_env,
    days_since_creation,
    file_attr,
    path_fn,
    artifact_pk,
):
    """Unlink each recording artifact that passes the env-match and age gates.

    Each unlink and ``deleted=1`` update holds ``recording_artifact_lock`` so
    it cannot interleave with a ``get_recording`` rebuild.
    """
    from spyglass.spikesorting.v2._recording.fingerprint import (
        recording_artifact_lock,
    )

    cutoff = _recent_cutoff(days_since_creation)
    authorized, matched = _authorize_artifacts_for_deletion(
        recompute_table,
        restriction,
        force_stale_env=force_stale_env,
        artifact_pk=artifact_pk,
    )
    deleted = []
    for artifact, authorizing_rows in authorized:
        # Age-gate only the authorizing rows; other envs' rows must not block.
        if _too_recent_or_unknown(authorizing_rows, cutoff):
            continue
        fname = (parent_table & artifact).fetch1(file_attr)
        abs_path = Path(path_fn(fname))
        if dry_run:
            deleted.append(str(abs_path))
            continue
        with recording_artifact_lock(artifact["recording_id"]):
            abs_path.unlink(missing_ok=True)
            if abs_path.exists():
                logger.warning(
                    f"delete_files: file not removed: {abs_path}; leaving "
                    "deleted=0 so a later cleanup retries."
                )
                continue
            for key in (matched & artifact).fetch("KEY", as_dict=True):
                recompute_table.update1({**key, "deleted": 1})
        deleted.append(str(abs_path))
    return deleted


def _delete_analyzer_folders(
    recompute_table,
    restriction,
    *,
    dry_run,
    force_stale_env,
    days_since_creation,
    folder_fn,
    artifact_pk,
):
    """Remove each analyzer folder that passes the env-match and age gates.

    Removes the whole folder, including extensions the audit does not hash;
    ``get_analyzer`` rebuilds it in full on the next read.
    """
    from spyglass.spikesorting.v2._storage.analyzer_cache import (
        analyzer_cache_lock,
    )

    cutoff = _recent_cutoff(days_since_creation)
    authorized, matched = _authorize_artifacts_for_deletion(
        recompute_table,
        restriction,
        force_stale_env=force_stale_env,
        artifact_pk=artifact_pk,
    )
    deleted = []
    for artifact, authorizing_rows in authorized:
        if _too_recent_or_unknown(authorizing_rows, cutoff):
            continue
        folder = Path(
            folder_fn(artifact["sorting_id"], artifact["waveform_params_name"])
        )
        if dry_run:
            deleted.append(str(folder))
            continue
        with analyzer_cache_lock(artifact["sorting_id"]):
            logger.info(
                "delete_files: removing the FULL analyzer folder "
                f"{folder} -- this drops ALL extensions, including the derived "
                "curation/visualization extensions the audit does not hash "
                "(amplitudes, correlograms, principal_components, "
                "quality_metrics), not just random_spikes/templates/waveforms. "
                "The analyzer cache is regeneratable scratch, rebuilt in full "
                "by the next get_analyzer."
            )
            shutil.rmtree(folder, ignore_errors=True)
            if folder.exists():
                logger.warning(
                    f"delete_files: analyzer folder not removed: {folder}; "
                    "leaving deleted=0 so a later cleanup retries."
                )
                continue
            for key in (matched & artifact).fetch("KEY", as_dict=True):
                recompute_table.update1({**key, "deleted": 1})
            deleted.append(str(folder))
    return deleted


def _reclaimable_disk(query) -> str:
    """Sum on-disk bytes of matched (reclaimable), not-yet-deleted artifacts.

    Counts each ``analysis_file_name`` once across matched env rows.
    """
    total = 0
    file_names = set(
        (query & "matched=1 AND deleted=0").fetch("analysis_file_name")
    )
    for file_name in file_names:
        # Only the size is read, so skip get_abs_path's whole-file checksum.
        abs_path = Path(
            AnalysisNwbfile.get_abs_path(file_name, from_schema=True)
        )
        if abs_path.exists():
            total += abs_path.stat().st_size
    return f"Total: {bytes_to_human_readable(total)}"

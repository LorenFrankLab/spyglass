"""Recompute verification + safe storage reclamation for v2 artifacts.

Ports v1's ``RecordingRecompute`` pattern to v2's ``Recording`` artifact and
its ``SortingAnalyzer`` folder. Each trio inventories an artifact's
dependencies (``*Versions``), plans a recompute attempt under a labeled
``UserEnvironment`` (``*RecomputeSelection``), then regenerates and compares
content hashes (``*Recompute``) so the original is deleted only after a
verified, current-environment match.

Comparison uses reproducible CONTENT (preprocessed ``ElectricalSeries`` traces
for recordings; deterministic analyzer extension data for analyzers) -- the same
reproducible content the recording ``content_hash`` captures, never a volatile
whole-file digest (see ``_recompute`` / ``_recording_fingerprint`` for why).
``rounding`` sets the float precision of the analyzer comparison.

``delete_files()`` is current-environment-aware: a ``matched=1`` row from a
different ``UserEnvironment`` does NOT authorize deletion (it raises
``StaleEnvMatchedError`` unless ``force_stale_env=True``), because a recompute
that succeeded under an older SpikeInterface pin is not evidence the current
environment can regenerate the artifact.
"""

from __future__ import annotations

import datetime as dt
import shutil
import tempfile
from pathlib import Path
from typing import NamedTuple, Optional, Union

import datajoint as dj

from spyglass.common.common_nwbfile import AnalysisNwbfile, Nwbfile
from spyglass.common.common_user import UserEnvironment
from spyglass.spikesorting.v2._recompute import (
    ANALYZER_RECOMPUTE_EXTENSIONS,
    combined_hash,
    compare_hash_dicts,
    current_env_namespaces,
    current_nwb_namespaces,
    env_matches,
    hash_extension_data,
    hash_recording_traces,
)
from spyglass.spikesorting.v2._recording_fingerprint import (
    TRACE_ROUNDING,
    recording_content_fingerprint,
)
from spyglass.spikesorting.v2.exceptions import (
    AnalyzerFolderInvalidError,
    AnalyzerFolderMissingError,
    StaleEnvMatchedError,
    ZeroUnitAnalyzerError,
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
# Explicit "analyzer folder absent" inventory sentinel, distinct from
# _ZERO_HASH (a legitimately zero-unit sort with no analyzer). Non-hex
# characters guarantee it never collides with a real sha256 content hash.
_MISSING_HASH = "MISSING_ANALYZER_FOLDER".ljust(64, "0")


def _current_env_id() -> Optional[str]:
    """Return the current ``UserEnvironment`` env_id (inserting if needed)."""
    return UserEnvironment().this_env.get("env_id")


# ---------------------------------------------------------------------
# Shared tri-part dispatch carriers + helpers for the recompute QC tables
# ---------------------------------------------------------------------


class RecomputeFetched(NamedTuple):
    """DB inputs for a recompute table's ``make_compute`` (no regen I/O).

    ``parent_key`` carries its UUID PK (``recording_id`` / ``sorting_id``) as a
    str so the carrier is DeepHash-stable for the tri-part integrity check.
    """

    parent_key: dict
    rounding: int
    xfail_reason: Optional[str]


class RecomputeComputed(NamedTuple):
    """``make_compute`` -> ``make_insert`` carrier for the recompute tables.

    ``outcome`` is ``'xfail'`` | ``'error'`` | ``'compare'``. A regeneration
    failure is a VALID ``matched=0`` outcome (``'error'``) encoded here rather
    than raised, so ``make_insert`` still records the QC result instead of the
    populate aborting. ``stored_hashes`` / ``new_hashes`` are empty except on
    ``'compare'``.
    """

    outcome: str
    err_msg: Optional[str]
    stored_hashes: dict
    new_hashes: dict
    parent_key: dict


class RecordingVersionsFetched(NamedTuple):
    """DB inputs for ``RecordingArtifactVersions.make_compute`` (no file I/O)."""

    analysis_file_name: str
    content_hash: str


class RecordingVersionsComputed(NamedTuple):
    """``make_compute`` -> ``make_insert`` for ``RecordingArtifactVersions``."""

    nwb_deps: Optional[dict]
    content_hash: str


class AnalyzerVersionsFetched(NamedTuple):
    """No upstream DB state to read -- the populate key already carries
    ``sorting_id`` + ``waveform_params_name``; the heavy analyzer load + hash is
    deferred to ``make_compute`` (off the framework transaction)."""


class AnalyzerVersionsComputed(NamedTuple):
    """``make_compute`` -> ``make_insert`` for ``SortingAnalyzerVersions``."""

    si_deps: dict
    analyzer_manifest: dict
    analyzer_hash: str


def _recompute_compute(
    parent_key, rounding, xfail_reason, *, regen, what
) -> RecomputeComputed:
    """Shared off-transaction compute for the recompute QC tables.

    Branches the three outcomes the monolithic ``make()`` handled, all of which
    end in an INSERT: ``xfail`` (skip the regen), ``error`` (a caught
    regeneration failure -> ``matched=0``, retryable), and ``compare`` (a real
    hash comparison). ``regen`` is a no-arg callable returning
    ``(stored_hashes, new_hashes)``.
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
        # A regeneration failure (SI pin mismatch, missing probe info, ...) is a
        # legitimate matched=0 outcome for this QC table. Log the full traceback
        # first so a masked code defect / disk error is still debuggable -- the
        # err_msg column truncates to 255 chars.
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

    ``compare`` routes through :func:`_insert_comparison` (master + Name/Hash
    diff rows); ``xfail`` / ``error`` insert a single ``matched=0`` master row.
    ``transaction_or_noop`` no-ops inside the framework transaction but keeps a
    direct (non-populate) call atomic.
    """
    from spyglass.spikesorting.v2.utils import transaction_or_noop

    with transaction_or_noop(table.connection):
        if outcome == "compare":
            _insert_comparison(table, key, stored_hashes, new_hashes, created_at)
        else:  # 'xfail' / 'error': a single matched=0 row, no diff parts
            table.insert1(
                {
                    **key,
                    "matched": False,
                    "err_msg": err_msg,
                    "created_at": created_at,
                }
            )


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

    # Tri-part: the namespace read opens the analysis NWB; keep that file I/O
    # OUTSIDE the framework transaction (make_compute) rather than in a
    # monolithic make holding row locks.
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

        The v2 analog of v1 ``RecordingRecomputeVersions.this_env``: an artifact
        is eligible only when every pynwb namespace it and the live environment
        have in common agrees on version (so a re-preprocess writes
        namespace-comparable output). Reads the live catalog once, then keeps the
        rows whose inventoried ``nwb_deps`` are compatible (see
        :func:`~spyglass.spikesorting.v2._recompute.env_matches`). Operates on
        ``self`` (restrict first to scope the scan).
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
class RecordingArtifactRecomputeSelection(SpyglassMixin, dj.Manual):
    """Plan a recording recompute attempt under a labeled environment."""

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

        The v2 analog of v1 ``RecordingRecomputeSelection.attempt_all``: bulk
        plan attempts for all (or restricted) ``RecordingArtifactVersions``
        rows under the current ``UserEnvironment``.

        By default only **env-compatible** artifacts are planned: an artifact is
        skipped unless every pynwb namespace its file and the current
        environment share agrees on version (mirrors v1's ``this_env`` gate).
        This avoids scheduling attempts that cannot reproduce the artifact in
        the current environment. ``force_attempt=True`` overrides the gate for a
        deliberate audit. Each planned artifact is also screened for known
        structural impossibilities (missing probe info, PyNWB-API / NWB-spec
        incompatibility); a match is recorded in ``xfail_reason`` so the
        recompute short-circuits to ``matched=0`` instead of wasting a regen
        (set ``check_xfail=False`` to skip the screen).

        There is no per-attempt ``rounding`` knob: the recording identity is the
        content fingerprint, whose precision is the fixed ``TRACE_ROUNDING`` /
        ``TIMESTAMP_ROUNDING`` constants. A per-attempt rounding would hash
        differently from the stored ``content_hash`` and spuriously never match.

        Parameters
        ----------
        restriction : dict, str, or list, optional
            Restriction on ``RecordingArtifactVersions``. Default all.
        limit : int, optional
            Plan at most this many artifacts, drawn at RANDOM from the eligible
            set (``dj.condition.Top(order_by="RAND()")``). For large
            retrospective audits where attempting every artifact is too costly;
            a random sample exercises a diverse spread. Default None (all).
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
        """Detect known STRUCTURAL impossibilities for one recording.

        Ports v1 ``RecordingRecomputeSelection._check_xfail`` and is kept
        deliberately NARROW -- it flags only impossibilities a recompute can
        never overcome (missing probe info, a PyNWB-API or NWB-spec
        incompatibility), so a flagged artifact is scheduled-but-marked rather
        than re-attempted. It is **not** a general skip mechanism: anything not
        matching these patterns returns ``(False, None)`` and is attempted
        normally.

        Recognition has two cheap (no file I/O) layers: prior ``matched=0``
        recompute runs whose ``err_msg`` names the pattern, and -- for probe
        info -- a direct ``Electrode * Probe`` presence query. (PyNWB-API /
        NWB-spec incompatibilities are recognized only from a prior failure's
        message; unlike v1 there is no proactive SpikeInterface re-read here, so
        planning stays I/O-free and the ``limit`` throttle is meaningful.)

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
            if bool(prev_runs & 'err_msg LIKE "%probe info%"'):
                return True, "missing_probe_info"
            try:  # proactive: is probe metadata on record for this recording?
                nwb_file_name = (RecordingSelection & rec_key).fetch1(
                    "nwb_file_name"
                )
                if _recording_missing_probe_info(nwb_file_name):
                    return True, "missing_probe_info"
            except Exception:  # noqa: BLE001 - can't check -> don't flag
                logger.warning(f"Unable to check probe info for {rec_key}")

        if skip_pynwb_api and bool(
            prev_runs & 'err_msg LIKE "%unexpected keyword%dtype%"'
        ):
            return True, "pynwb_api_incompatible"

        if skip_nwb_spec and bool(
            prev_runs & 'err_msg LIKE "%No spec%namespace%"'
        ):
            return True, "nwb_spec_incompatible"

        return False, None

    @classmethod
    def remove_matched(cls, restriction=True, *, dry_run: bool = True) -> int:
        """Remove redundant selection rows for already-verified artifacts.

        Mirrors v1 ``remove_matched``: drop selections that target a recording
        with a matched recompute (in ANY env) but are NOT themselves the
        matched attempt. Those redundant selections carry no dependent
        recompute row, so ``delete_quick`` cannot hit the Recompute->Selection
        FK. Selections whose own recompute matched are kept -- they are the
        verification record.
        """
        matched = RecordingArtifactRecompute & "matched=1"
        artifact_pk = RecordingArtifactVersions.primary_key
        matched_artifacts = (dj.U(*artifact_pk) & matched).fetch(
            "KEY", as_dict=True
        )
        redundant = (cls & restriction & matched_artifacts) - matched.proj()
        count = len(redundant)
        if dry_run or count == 0:
            logger.info(
                f"remove_matched: {count} redundant rows (dry_run={dry_run})."
            )
            return count
        redundant.delete_quick()
        return count


@schema
class RecordingArtifactRecompute(SpyglassMixin, dj.Computed):
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

        Projects to the ``recording_id`` PK rather than joining ``Recording``
        directly -- both tables carry a ``content_hash`` secondary attr, which
        would otherwise trigger a join on a dependent attribute.
        """
        recording_id = (RecordingArtifactVersions & key).fetch1("recording_id")
        return {"recording_id": recording_id}

    # Tri-part dispatch: the trace regeneration (re-preprocess to a fresh NWB +
    # hash) is the long step and stays OUTSIDE the framework transaction
    # (mirroring Recording / Sorting). A regen failure is encoded as the 'error'
    # outcome (matched=0), not raised, preserving the monolithic behavior.
    _parallel_make = True

    def make_fetch(self, key) -> RecomputeFetched:
        """Read the recompute inputs (no regeneration I/O).

        There is no per-attempt ``rounding`` -- the content fingerprint's
        precision is fixed (``TRACE_ROUNDING`` / ``TIMESTAMP_ROUNDING``). The
        shared ``RecomputeFetched.rounding`` field carries ``TRACE_ROUNDING`` for
        carrier shape only; the recording compute path ignores it.
        """
        xfail_reason = (
            RecordingArtifactRecomputeSelection & key
        ).fetch1("xfail_reason")
        parent = self.get_parent_key(key)
        return RecomputeFetched(
            # str the recording_id UUID for a DeepHash-stable carrier.
            parent_key={"recording_id": str(parent["recording_id"])},
            rounding=TRACE_ROUNDING,
            xfail_reason=xfail_reason,
        )

    def make_compute(
        self, key, parent_key, rounding, xfail_reason
    ) -> RecomputeComputed:
        """Fingerprint the current file + a fresh rebuild, off the transaction.

        ``stored_hashes`` is the CURRENT on-disk file's fingerprint components
        (``get_recording`` self-heals a missing file first) and ``new_hashes``
        is an independent FRESH temp rebuild's components. ``make_insert``
        anchors ``matched`` on ``combined_hash(new) == Recording.content_hash``
        (recoverability) and reports the current-vs-fresh component diff.
        """

        def _regen():
            # Self-heal a missing/deleted file first (and fail closed on drift),
            # then fingerprint the canonical file; fingerprint a fresh,
            # independent temp rebuild for the match authority.
            Recording().get_recording(parent_key)
            current_abs = AnalysisNwbfile.get_abs_path(
                (Recording & parent_key).fetch1("analysis_file_name")
            )
            current = recording_content_fingerprint(
                current_abs, electrical_series_path=_ELECTRICAL_SERIES_PATH
            )
            fresh = _recompute_recording_fingerprint(parent_key)
            return current, fresh

        return _recompute_compute(
            parent_key,
            rounding,
            xfail_reason,
            regen=_regen,
            what="RecordingArtifactRecompute",
        )

    def make_insert(
        self, key, outcome, err_msg, stored_hashes, new_hashes, parent_key
    ):
        """Record the QC result; ``matched`` anchors on ``content_hash``.

        ``created_at`` is the artifact file mtime, read here (not in
        ``make_fetch``) because the mtime is non-deterministic across the
        framework's two ``make_fetch`` calls and would trip the DeepHash
        integrity check.

        ``matched`` is ``combined_hash(new_hashes) == Recording.content_hash``
        -- the same invariant ``_rebuild_nwb_artifact`` enforces, so a
        ``matched`` recompute authorizes a delete the rebuild can honor -- NOT
        the current-vs-fresh dict diff. The diff parts still report which
        component drifted.
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

    def recheck(self, key) -> bool:
        """Rerun the comparison for one row (after env/file changes).

        Uses cautious ``delete`` (not ``delete_quick``) so the row's
        ``Name`` / ``Hash`` diff part rows cascade and the team-permission
        guard applies; ``safemode=False`` skips the prompt for this
        programmatic recheck. Then re-populates.
        """
        (self & key).delete(safemode=False)
        self.populate(key, reserve_jobs=False)
        return bool((self & key & "matched=1"))

    def update_secondary(self, restriction=True) -> None:
        """Backfill ``created_at`` from the artifact file mtime."""
        for key in (self & restriction).fetch("KEY", as_dict=True):
            self.update1(
                {**key, "created_at": _artifact_created_at(self.get_parent_key(key))}
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

        Refuses ``matched=0`` rows and, by default, refuses rows matched only
        in a stale ``UserEnvironment`` (raises ``StaleEnvMatchedError``). The
        regeneratable artifact (the preprocessed ``AnalysisNwbfile``) is removed
        and ``deleted`` set; ``Recording.get_recording`` rebuilds it on demand.
        """
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

    A structural impossibility for recompute: the rebuild needs probe geometry,
    so an empty join (probe metadata never ingested, or stripped) means the
    artifact can never be regenerated. Mirrors v1's probe-presence check.
    """
    from spyglass.common.common_device import Probe
    from spyglass.common.common_ephys import Electrode

    return not bool(Electrode * Probe & {"nwb_file_name": nwb_file_name})


def _recompute_recording_fingerprint(rec_key: dict) -> dict:
    """Recompute a recording to a fresh (unregistered) temp file and return its
    content-fingerprint component dict.

    The fresh temp is unlinked on success, mismatch, and error -- it never
    enters the canonical slot.
    """
    recording_table = Recording()
    fetched = recording_table.make_fetch(rec_key)
    raw_path = Nwbfile().get_abs_path(fetched.sel["nwb_file_name"])
    result = recording_table._compute_recording_artifact(
        raw_path=raw_path,
        raw_object_id=fetched.raw_object_id,
        nwb_file_name=fetched.sel["nwb_file_name"],
        interval_list_name=fetched.sel["interval_list_name"],
        channel_ids=fetched.channel_ids,
        reference_mode=fetched.reference_mode,
        reference_electrode_id=fetched.reference_electrode_id,
        sort_valid_times=fetched.sort_valid_times,
        raw_valid_times=fetched.raw_valid_times,
        preprocessing_params=fetched.preprocessing_params,
        probe_types=fetched.probe_types,
        electrode_group_names=fetched.electrode_group_names,
        bad_channel_ids=fetched.bad_channel_ids,
        existing_analysis_file_name=None,  # fresh, unregistered temp file
    )
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

    ``matched = combined_hash(fresh) == content_hash`` -- the FRESH rebuild
    reproducing the row identity (recoverability), the same invariant
    ``Recording._rebuild_nwb_artifact`` enforces -- NOT the current-vs-fresh
    dict diff. The ``Name`` / ``Hash`` diff parts still report which fingerprint
    component differs between the current served file and the fresh rebuild
    (non-determinism / current-file drift), so an operator can localize a
    problem even on a matched row.
    """
    from spyglass.spikesorting.v2.utils import transaction_or_noop

    matched = combined_hash(fresh) == content_hash
    _, missing_old, missing_new, differing = compare_hash_dicts(current, fresh)
    with transaction_or_noop(table.connection):
        table.insert1({**key, "matched": matched, "created_at": created_at})
        name_rows = [
            {**key, "name": n, "missing_from": "old"} for n in missing_old
        ] + [{**key, "name": n, "missing_from": "new"} for n in missing_new]
        if name_rows:
            table.Name().insert(name_rows)
        if differing:
            table.Hash().insert([{**key, "name": n} for n in differing])


# =====================================================================
# SortingAnalyzer recompute
# =====================================================================


@schema
class SortingAnalyzerVersions(SpyglassMixin, dj.Computed):
    """Dependency + content inventory for a ``Sorting``'s analyzer folders.

    One row per (sort, analyzer recipe): a sort's stored DISPLAY recipe plus
    any whitened METRIC recipe an ``AnalyzerCurationSelection`` references. The
    whitened and unwhitened analyzers for one ``sorting_id`` are inventoried
    (and recomputed) independently, keyed by ``waveform_params_name`` -- their
    folders are ``{sorting_id}__{waveform_params_name}.zarr`` and never collide.
    """

    definition = """
    -> Sorting
    -> AnalyzerWaveformParameters.proj(waveform_params_name="waveform_params_name")
    ---
    si_deps=null: blob          # spikeinterface version, etc.
    analyzer_manifest=null: blob # extension_name -> content_hash mapping
    analyzer_hash: char(64)
    """

    @property
    def key_source(self):
        """The (sort, recipe) pairs that have an analyzer folder.

        Every sort's stored display recipe, unioned with every metric recipe a
        curation selection references (the only way a whitened analyzer comes
        into existence). A sort with no curation has just its one display row.
        """
        from spyglass.spikesorting.v2.metric_curation import (
            AnalyzerCurationSelection,
        )

        # All (sort, recipe) pairs, restricted to those actually in use: a
        # sort's stored display recipe OR a metric recipe a PC-requesting
        # curation references. An OR-list semijoin on the clean
        # (sorting_id, waveform_params_name) cross product -- not a union of
        # dj.U aggregations, whose headings cannot be joined (DataJoint
        # Union.create -> heading.join KeyError).
        all_pairs = (AnalyzerWaveformParameters * Sorting).proj()
        is_display = Sorting.proj(
            waveform_params_name="display_waveform_params_name"
        )
        # pc_requesting() is the single source of "which metric recipes were
        # actually built" (shared with the orphan-folder audit). Project to
        # (sorting_id, waveform_params_name); sorting_id is carried explicitly
        # because it is a SECONDARY (CurationV2) FK attr, not the selection's
        # uuid PK -- a bare proj() would drop it and the semijoin would match on
        # waveform_params_name alone (leaking a recipe onto every sort).
        is_metric = AnalyzerCurationSelection.pc_requesting().proj(
            "sorting_id", waveform_params_name="metric_waveform_params_name"
        )
        return all_pairs & [is_display, is_metric]

    # Tri-part: loading the analyzer folder + hashing its full extension arrays
    # is the heavy step and must stay OUTSIDE the framework transaction
    # (make_compute), not hold row locks in a monolithic make. make_fetch is
    # empty -- the populate key already carries sorting_id + waveform_params_name.
    _parallel_make = True

    def make_fetch(self, key) -> AnalyzerVersionsFetched:
        """No upstream DB state to read (key is self-sufficient)."""
        return AnalyzerVersionsFetched()

    def make_compute(self, key) -> AnalyzerVersionsComputed:
        """Load the analyzer + hash its extensions off the transaction.

        Uses the NO-REBUILD loader: an absent analyzer folder is inventoried as
        an explicit MISSING state (``_MISSING_HASH``) rather than silently
        rebuilt and hashed as if present -- so the inventory distinguishes a
        reclaimed/missing analyzer from a legitimately zero-unit one
        (``_ZERO_HASH``), and reclaimed disk is not re-materialized just to
        record a hash.
        """
        import spikeinterface as si

        si_deps = {"spikeinterface": si.__version__}
        try:
            analyzer = Sorting().get_analyzer(
                {"sorting_id": key["sorting_id"]},
                waveform_params_name=key["waveform_params_name"],
                rebuild=False,
            )
            manifest = hash_extension_data(analyzer)
        except ZeroUnitAnalyzerError:
            manifest = {}
        except AnalyzerFolderInvalidError as exc:
            logger.warning(
                "SortingAnalyzerVersions: analyzer folder invalid for "
                f"sorting_id={key['sorting_id']}, "
                f"recipe={key['waveform_params_name']}; inventorying as MISSING "
                f"(not rebuilding). Error: {exc}"
            )
            return AnalyzerVersionsComputed(
                si_deps=si_deps,
                analyzer_manifest={},
                analyzer_hash=_MISSING_HASH,
            )
        except AnalyzerFolderMissingError:
            logger.warning(
                "SortingAnalyzerVersions: analyzer folder missing for "
                f"sorting_id={key['sorting_id']}, "
                f"recipe={key['waveform_params_name']}; inventorying as MISSING "
                "(not rebuilding)."
            )
            return AnalyzerVersionsComputed(
                si_deps=si_deps,
                analyzer_manifest={},
                analyzer_hash=_MISSING_HASH,
            )
        return AnalyzerVersionsComputed(
            si_deps=si_deps,
            analyzer_manifest=manifest,
            analyzer_hash=combined_hash(manifest) if manifest else _ZERO_HASH,
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


@schema
class SortingAnalyzerRecomputeSelection(SpyglassMixin, dj.Manual):
    """Plan an analyzer recompute attempt under a labeled environment."""

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
        rows = [
            {**version_key, "env_id": env_id, "rounding": rounding}
            for version_key in (
                SortingAnalyzerVersions & restriction
            ).fetch("KEY", as_dict=True)
        ]
        cls.insert(rows, skip_duplicates=True)

    @classmethod
    def remove_matched(cls, restriction=True, *, dry_run: bool = True) -> int:
        """Remove redundant selection rows for already-verified analyzers.

        Mirrors v1 ``remove_matched`` (see the recording variant): drop
        selections targeting a sort with a matched recompute (any env) that are
        not themselves the matched attempt, so ``delete_quick`` cannot hit the
        Recompute->Selection FK.
        """
        matched = SortingAnalyzerRecompute & "matched=1"
        artifact_pk = SortingAnalyzerVersions.primary_key
        matched_artifacts = (dj.U(*artifact_pk) & matched).fetch(
            "KEY", as_dict=True
        )
        redundant = (cls & restriction & matched_artifacts) - matched.proj()
        count = len(redundant)
        if dry_run or count == 0:
            logger.info(
                f"remove_matched: {count} redundant rows (dry_run={dry_run})."
            )
            return count
        redundant.delete_quick()
        return count


@schema
class SortingAnalyzerRecompute(SpyglassMixin, dj.Computed):
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
        """Return the upstream ``Sorting`` key for a recompute row."""
        sorting_id = (SortingAnalyzerVersions & key).fetch1("sorting_id")
        return {"sorting_id": sorting_id}

    # Tri-part dispatch: the analyzer-folder regeneration + extension hashing is
    # the long step and stays OUTSIDE the framework transaction. A regen failure
    # is encoded as the 'error' outcome (matched=0), not raised.
    _parallel_make = True

    def make_fetch(self, key) -> RecomputeFetched:
        """Read the recompute inputs (no regeneration I/O)."""
        rounding, xfail_reason = (
            SortingAnalyzerRecomputeSelection & key
        ).fetch1("rounding", "xfail_reason")
        parent = self.get_parent_key(key)
        return RecomputeFetched(
            # str the sorting_id UUID for a DeepHash-stable carrier.
            parent_key={"sorting_id": str(parent["sorting_id"])},
            rounding=int(rounding),
            xfail_reason=xfail_reason,
        )

    def make_compute(
        self, key, parent_key, rounding, xfail_reason
    ) -> RecomputeComputed:
        """Regenerate the analyzer folder + hash extensions off the transaction."""
        return _recompute_compute(
            parent_key,
            rounding,
            xfail_reason,
            regen=lambda: _recompute_analyzer_hashes(
                parent_key, rounding, key["waveform_params_name"]
            ),
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

        Only ``matched=1`` analyzers are deletable by ``delete_files``, so only
        those are reclaimable; one folder is counted once even across multiple
        env rows.
        """
        total = 0
        reclaimable = self & restriction & "matched=1 AND deleted=0"
        # Each (sorting_id, recipe) is a distinct analyzer folder; count each
        # once across env rows.
        for sid, name in {
            (key["sorting_id"], key["waveform_params_name"])
            for key in reclaimable.fetch("KEY", as_dict=True)
        }:
            folder = _analyzer_folder(sid, name)
            if folder.exists():
                total += sum(
                    f.stat().st_size for f in folder.rglob("*") if f.is_file()
                )
        return f"Total: {bytes_to_human_readable(total)}"

    def recheck(self, key) -> bool:
        """Rerun the comparison for one row.

        Uses cautious ``delete`` (not ``delete_quick``) so the diff part rows
        cascade and the team-permission guard applies, then re-populates.
        """
        (self & key).delete(safemode=False)
        self.populate(key, reserve_jobs=False)
        return bool((self & key & "matched=1"))

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
        """Delete analyzer folders verified in the CURRENT environment.

        Same current-environment gate as the recording recompute. The deleted
        analyzer folder is regeneratable via ``Sorting.get_analyzer``.
        """
        return _delete_analyzer_folders(
            self,
            restriction,
            dry_run=dry_run,
            force_stale_env=force_stale_env,
            days_since_creation=days_since_creation,
            folder_fn=_analyzer_folder,
            artifact_pk=SortingAnalyzerVersions.primary_key,
        )


def _analyzer_folder(sorting_id, waveform_params_name):
    """Return the analyzer cache folder for a (sort, recipe).

    Recompute inventories one folder per (sort, recipe); the folder-size
    accounting and delete target resolve the explicit ``waveform_params_name``
    (display or whitened metric), keyed ``{sorting_id}__{name}.zarr``.
    """
    from spyglass.spikesorting.v2._analyzer_cache import analyzer_path

    return analyzer_path(sorting_id, waveform_params_name)


def _recompute_analyzer_hashes(
    sort_key: dict, rounding: int, waveform_params_name: str
):
    """Hash a recipe's stored analyzer and a fresh temp rebuild.

    ``waveform_params_name`` selects which recipe to verify (display or
    whitened metric); the rebuild uses that recipe's params, so the fresh
    analyzer is byte-comparable to the cached one for the SAME recipe.
    """
    from spyglass.spikesorting.v2._sorting_analyzer import (
        build_analyzer,
        fetch_waveform_params,
        reconstruct_recording_and_sorting,
    )

    try:
        # NO-REBUILD: an absent stored analyzer must NOT be self-healed here --
        # rebuilding it would compare a fresh build to another fresh build and
        # report a tautological match, authorizing deletion of a folder that was
        # reconstructed for the audit. Instead AnalyzerFolderMissingError
        # propagates to _recompute_compute, which records matched=0 (the audit
        # cannot verify reproducibility against an original that is gone).
        stored = Sorting().get_analyzer(
            sort_key,
            waveform_params_name=waveform_params_name,
            rebuild=False,
        )
    except ZeroUnitAnalyzerError:
        return {}, {}  # zero-unit: nothing to verify -> trivially matched
    stored_hashes = hash_extension_data(stored, rounding=rounding)

    import spikeinterface as si

    params = fetch_waveform_params(waveform_params_name)
    # Source sorting + recording from the CANONICAL units NWB + recording (the
    # shared resolver), NOT a self-healing analyzer load. This (a) never rebuilds
    # the DISPLAY analyzer cache as a side effect -- so verifying a metric
    # analyzer while the display folder is reclaimed still produces a real
    # comparison (and never re-materializes a large display folder during the
    # audit) -- and (b) reconstructs the SAME artifact-masked, unwhitened
    # recording build_analyzer starts from (it 2D-projects + whitens per recipe,
    # so a whitened metric analyzer is not double-whitened). The sorting is
    # recipe-independent. (Recording.get_recording can still rebuild a reclaimed
    # RECORDING cache -- a separate, known self-heal, out of scope here.)
    recording, sorting = reconstruct_recording_and_sorting(Sorting(), sort_key)

    tmp = tempfile.mkdtemp(prefix="v2_analyzer_recompute_")
    try:
        # Rebuild from the SAME sorting + recording with build_analyzer's exact
        # seed/param logic, to a temp folder, so the comparison is a genuine
        # regeneration rather than the stored folder compared to itself. Build
        # only the extensions this verify actually hashes -- noise_levels is not
        # hashed (and not a dependency of templates/waveforms), so computing it
        # would be wasted work on every recompute.
        build_analyzer(
            sorting,
            recording,
            sort_key,
            analyzer_folder=Path(tmp) / "analyzer.zarr",
            waveform_params=params,
            extensions=ANALYZER_RECOMPUTE_EXTENSIONS,
        )
        fresh = si.load_sorting_analyzer(Path(tmp) / "analyzer.zarr")
        new_hashes = hash_extension_data(fresh, rounding=rounding)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return stored_hashes, new_hashes


# =====================================================================
# Shared helpers
# =====================================================================


def _artifact_created_at(rec_key: dict):
    """Return the recording artifact file mtime (now() if missing)."""
    fname = (Recording & rec_key).fetch1("analysis_file_name")
    abs_path = Path(AnalysisNwbfile.get_abs_path(fname))
    if not abs_path.exists():
        return dt.datetime.now()
    return dt.datetime.fromtimestamp(abs_path.stat().st_mtime)


def _insert_comparison(table, key, stored_hashes, new_hashes, created_at):
    """Insert the matched row plus Name/Hash diff part rows."""
    matched, missing_old, missing_new, differing = compare_hash_dicts(
        stored_hashes, new_hashes
    )
    table.insert1(
        {**key, "matched": matched, "created_at": created_at}
    )
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

    ``authorized`` is a list of ``(artifact_key, authorizing_rows)`` tuples.
    Authorization is at the ARTIFACT level, not per recompute row: an artifact
    (one ``recording_id`` / ``sorting_id``) is authorized when it has a
    ``matched=1``, not-yet-deleted recompute row in the CURRENT
    ``UserEnvironment``. An artifact whose matched rows are ALL in stale envs
    raises ``StaleEnvMatchedError`` (unless ``force_stale_env``, which
    authorizes it with an audit log line). Keying on the artifact -- rather than
    the full recompute row including ``env_id`` -- means a stale-env row never
    blocks an artifact that also has a current-env match.

    ``authorizing_rows`` is the subset of rows that grant the authorization
    (the current-env rows when present, else the stale rows under
    ``force_stale_env``). The age gate is applied to ONLY these rows, so a
    stale/recent sibling row never blocks an otherwise-authorized artifact.
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


def _recent_cutoff(days_since_creation: int):
    return dt.datetime.now() - dt.timedelta(days=days_since_creation)


def _too_recent_or_unknown(matched_rows, cutoff) -> bool:
    """True if any matched row's ``created_at`` is NULL or newer than cutoff.

    A destructive op should not proceed on an unknown (NULL) or too-recent age.
    """
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
    """Artifact-level recording delete gate: current-env match + age, unlink once.

    The unlink + ``deleted=1`` update for each artifact runs under
    ``recording_artifact_lock(recording_id)`` so a reclamation can never
    interleave with a concurrent ``get_recording`` rebuild of the same recording
    -- no unlink racing a write, no reader seeing a half-state.
    """
    from spyglass.spikesorting.v2._recording_fingerprint import (
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
        # Age-gate the AUTHORIZING rows only -- a stale/recent sibling row must
        # not block an artifact a valid current-env row authorizes.
        if _too_recent_or_unknown(authorizing_rows, cutoff):
            continue
        fname = (parent_table & artifact).fetch1(file_attr)
        abs_path = Path(path_fn(fname))
        if dry_run:
            deleted.append(str(abs_path))
            continue
        # Serialize against a concurrent rebuild of THIS recording.
        with recording_artifact_lock(artifact["recording_id"]):
            abs_path.unlink(missing_ok=True)
            if abs_path.exists():
                logger.warning(
                    f"delete_files: file not removed: {abs_path}; leaving "
                    "deleted=0 so a later cleanup retries."
                )
                continue
            # Mark every matched row for this artifact deleted, only after the
            # file is gone (the file is per-artifact; the flag is per row).
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
    """Artifact-level analyzer delete gate: current-env match + age, rmtree once."""
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
        shutil.rmtree(folder, ignore_errors=True)
        if folder.exists():
            # Do NOT mark deleted -- the folder is still on disk; leaving
            # deleted=0 lets a later cleanup retry rather than silently
            # suppressing it.
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

    Dedupes by ``analysis_file_name`` so an artifact with multiple matched env
    rows is counted once (the file is per-artifact, not per recompute row).
    """
    total = 0
    file_names = set(
        (query & "matched=1 AND deleted=0").fetch("analysis_file_name")
    )
    for file_name in file_names:
        abs_path = Path(AnalysisNwbfile.get_abs_path(file_name))
        if abs_path.exists():
            total += abs_path.stat().st_size
    return f"Total: {bytes_to_human_readable(total)}"

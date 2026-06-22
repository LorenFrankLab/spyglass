"""Recompute verification + safe storage reclamation for v2 artifacts.

Ports v1's ``RecordingRecompute`` pattern to v2's ``Recording`` artifact and
its ``SortingAnalyzer`` folder. Each trio inventories an artifact's
dependencies (``*Versions``), plans a recompute attempt under a labeled
``UserEnvironment`` (``*RecomputeSelection``), then regenerates and compares
content hashes (``*Recompute``) so the original is deleted only after a
verified, current-environment match.

Comparison uses reproducible CONTENT (preprocessed ``ElectricalSeries`` traces
for recordings; deterministic analyzer extension data for analyzers), not the
volatile whole-file ``cache_hash`` -- see ``_recompute`` for why. ``rounding``
sets the float precision of the comparison.

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
from typing import Optional, Union

import datajoint as dj

from spyglass.common.common_nwbfile import AnalysisNwbfile, Nwbfile
from spyglass.common.common_user import UserEnvironment
from spyglass.spikesorting.v2._recompute import (
    combined_hash,
    compare_hash_dicts,
    current_nwb_namespaces,
    hash_extension_data,
    hash_recording_traces,
)
from spyglass.spikesorting.v2.exceptions import (
    StaleEnvMatchedError,
    ZeroUnitAnalyzerError,
)
from spyglass.spikesorting.v2.recording import (
    _ELECTRICAL_SERIES_PATH,
    Recording,
)
from spyglass.spikesorting.v2.sorting import Sorting
from spyglass.utils import SpyglassMixin, SpyglassMixinPart, logger
from spyglass.utils.dj_helper_fn import bytes_to_human_readable

schema = dj.schema("spikesorting_v2_recompute")

_ZERO_HASH = "0" * 64


def _current_env_id() -> Optional[str]:
    """Return the current ``UserEnvironment`` env_id (inserting if needed)."""
    return UserEnvironment().this_env.get("env_id")


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
    cache_hash: char(64)      # stored Recording.cache_hash (provenance)
    """

    def make(self, key):
        row = (Recording & key).fetch1()
        abs_path = AnalysisNwbfile.get_abs_path(row["analysis_file_name"])
        nwb_deps = (
            current_nwb_namespaces(abs_path) if Path(abs_path).exists() else None
        )
        self.insert1(
            {**key, "nwb_deps": nwb_deps, "cache_hash": row["cache_hash"]}
        )


@schema
class RecordingArtifactRecomputeSelection(SpyglassMixin, dj.Manual):
    """Plan a recording recompute attempt under a labeled environment."""

    definition = """
    -> RecordingArtifactVersions
    -> UserEnvironment
    rounding=4: int           # float precision for the trace comparison
    ---
    logged_at_creation=0: bool
    xfail_reason=NULL: varchar(127)
    """

    @classmethod
    def attempt_all(cls, restriction=True, *, rounding: int = 4) -> None:
        """Insert a recompute attempt for every eligible artifact (current env).

        The v2 analog of v1 ``RecordingRecomputeSelection.attempt_all``: bulk
        plan attempts for all (or restricted) ``RecordingArtifactVersions``
        rows under the current ``UserEnvironment``.
        """
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
                RecordingArtifactVersions & restriction
            ).fetch("KEY", as_dict=True)
        ]
        cls.insert(rows, skip_duplicates=True)

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
        directly -- both tables carry a ``cache_hash`` secondary attr, which
        would otherwise trigger a join on a dependent attribute.
        """
        recording_id = (RecordingArtifactVersions & key).fetch1("recording_id")
        return {"recording_id": recording_id}

    def make(self, key):
        sel = (RecordingArtifactRecomputeSelection & key).fetch1()
        rec_key = self.get_parent_key(key)
        created_at = _artifact_created_at(rec_key)
        if sel["xfail_reason"]:
            self.insert1(
                {
                    **key,
                    "matched": False,
                    "err_msg": f"xfail: {sel['xfail_reason']}"[:255],
                    "created_at": created_at,
                }
            )
            return
        rounding = sel["rounding"]
        try:
            stored = Recording().get_recording(rec_key)
            stored_hashes = hash_recording_traces(stored, rounding=rounding)
            new_hashes = _recompute_recording_trace_hashes(rec_key, rounding)
        except Exception as err:  # noqa: BLE001 - record the failure, retryable
            # A regeneration failure (SI pin mismatch, missing probe info, ...)
            # is a legitimate matched=0 outcome for this QC table. Log the full
            # traceback first so a masked code defect / disk error is still
            # debuggable -- the err_msg column truncates to 255 chars.
            logger.error(
                f"RecordingArtifactRecompute regeneration failed for "
                f"{rec_key}; recording matched=0 (retryable).",
                exc_info=True,
            )
            self.insert1(
                {
                    **key,
                    "matched": False,
                    "err_msg": str(err)[:255],
                    "created_at": created_at,
                }
            )
            return
        _insert_comparison(self, key, stored_hashes, new_hashes, created_at)

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


def _recompute_recording_trace_hashes(rec_key: dict, rounding: int) -> dict:
    """Recompute a recording to a fresh (unregistered) file and hash traces."""
    import spikeinterface.extractors as se

    recording_table = Recording()
    fetched = recording_table.make_fetch(rec_key)
    raw_path = Nwbfile().get_abs_path(fetched.sel["nwb_file_name"])
    result = recording_table._compute_recording_artifact(
        raw_path=raw_path,
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
        existing_analysis_file_name=None,  # fresh, unregistered file
    )
    fresh_abs = AnalysisNwbfile.get_abs_path(result.analysis_file_name)
    try:
        fresh = se.read_nwb_recording(
            fresh_abs,
            electrical_series_path=_ELECTRICAL_SERIES_PATH,
            load_time_vector=True,
        )
        return hash_recording_traces(fresh, rounding=rounding)
    finally:
        Path(fresh_abs).unlink(missing_ok=True)


# =====================================================================
# SortingAnalyzer recompute
# =====================================================================


@schema
class SortingAnalyzerVersions(SpyglassMixin, dj.Computed):
    """Dependency + content inventory for a ``Sorting``'s analyzer folder."""

    definition = """
    -> Sorting
    ---
    si_deps=null: blob          # spikeinterface version, etc.
    analyzer_manifest=null: blob # extension_name -> content_hash mapping
    analyzer_hash: char(64)
    """

    def make(self, key):
        import spikeinterface as si

        si_deps = {"spikeinterface": si.__version__}
        try:
            analyzer = Sorting().get_analyzer(key)
            manifest = hash_extension_data(analyzer)
        except ZeroUnitAnalyzerError:
            manifest = {}
        self.insert1(
            {
                **key,
                "si_deps": si_deps,
                "analyzer_manifest": manifest,
                "analyzer_hash": combined_hash(manifest)
                if manifest
                else _ZERO_HASH,
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

    def make(self, key):
        sel = (SortingAnalyzerRecomputeSelection & key).fetch1()
        sort_key = self.get_parent_key(key)
        created_at = dt.datetime.now()
        if sel["xfail_reason"]:
            self.insert1(
                {
                    **key,
                    "matched": False,
                    "err_msg": f"xfail: {sel['xfail_reason']}"[:255],
                    "created_at": created_at,
                }
            )
            return
        rounding = sel["rounding"]
        try:
            stored_hashes, new_hashes = _recompute_analyzer_hashes(
                sort_key, rounding
            )
        except Exception as err:  # noqa: BLE001 - record the failure
            logger.error(
                f"SortingAnalyzerRecompute regeneration failed for "
                f"{sort_key}; analyzer matched=0 (retryable).",
                exc_info=True,
            )
            self.insert1(
                {
                    **key,
                    "matched": False,
                    "err_msg": str(err)[:255],
                    "created_at": created_at,
                }
            )
            return
        _insert_comparison(self, key, stored_hashes, new_hashes, created_at)

    def get_disk_space(self, restriction=True) -> str:
        """Report reclaimable disk for matched, not-yet-deleted analyzers.

        Only ``matched=1`` analyzers are deletable by ``delete_files``, so only
        those are reclaimable; one folder is counted once even across multiple
        env rows.
        """
        from spyglass.spikesorting.v2._analyzer_cache import analyzer_path

        total = 0
        reclaimable = self & restriction & "matched=1 AND deleted=0"
        for sid in {
            self.get_parent_key(key)["sorting_id"]
            for key in reclaimable.fetch("KEY", as_dict=True)
        }:
            folder = analyzer_path(sid)
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
        from spyglass.spikesorting.v2._analyzer_cache import analyzer_path

        return _delete_analyzer_folders(
            self,
            restriction,
            dry_run=dry_run,
            force_stale_env=force_stale_env,
            days_since_creation=days_since_creation,
            folder_fn=analyzer_path,
            artifact_pk=Sorting.primary_key,
        )


def _recompute_analyzer_hashes(sort_key: dict, rounding: int):
    """Hash the stored analyzer and a fresh temp rebuild; return both dicts."""
    from spyglass.spikesorting.v2._sorting_analyzer import build_analyzer

    try:
        stored = Sorting().get_analyzer(sort_key)
    except ZeroUnitAnalyzerError:
        return {}, {}  # zero-unit: nothing to verify -> trivially matched
    stored_hashes = hash_extension_data(stored, rounding=rounding)

    import spikeinterface as si

    tmp = tempfile.mkdtemp(prefix="v2_analyzer_recompute_")
    try:
        # Rebuild from the SAME sorting + recording with build_analyzer's exact
        # seed/param logic, to a temp folder, so the comparison is a genuine
        # regeneration rather than the stored folder compared to itself.
        build_analyzer(
            stored.sorting,
            stored.recording,
            sort_key,
            analyzer_folder=Path(tmp) / "analyzer",
        )
        fresh = si.load_sorting_analyzer(Path(tmp) / "analyzer")
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
    """Artifact-level recording delete gate: current-env match + age, unlink once."""
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
        abs_path.unlink(missing_ok=True)
        if abs_path.exists():
            logger.warning(
                f"delete_files: file not removed: {abs_path}; leaving "
                "deleted=0 so a later cleanup retries."
            )
            continue
        # Mark every matched row for this artifact deleted, only after the file
        # is gone (the file is per-artifact; the flag is per recompute row).
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
        folder = Path(folder_fn(artifact["sorting_id"]))
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

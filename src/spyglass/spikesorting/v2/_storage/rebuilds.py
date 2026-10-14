"""Recording artifact self-heal, verified rebuilding and atomic publication.

This module owns file verification memoization, missing-artifact reconstruction,
content and span drift checks, checksum reconciliation, and rebuild rollback.
Single-session, concatenated and motion-corrected recordings use the same
publication contract. Readers and NWB writers live in ``_storage.nwb``.

DB-free at import. Database rows, persisted file locations and metadata are
resolved only when a lifecycle operation runs. Rebuilding calls the owning
table's fetch/compute methods and publishes only verified staged output.
"""

from __future__ import annotations

from spyglass.spikesorting.v2._storage.nwb import (
    StoredTraces,
    recording_provenance_table,
)
from spyglass.spikesorting.v2._storage.staged_outputs import (
    unlink_staged_analysis_file as _unlink_staged_analysis_file,
)

#: ``{analysis_file_name: (abs_path, file_identity)}`` for each cached trace
#: artifact this process resolved through ``AnalysisNwbfile.get_abs_path``
#: (which checksums the file) outside a transaction; ``file_identity`` is
#: :func:`_file_identity` at that time. See :func:`ensure_artifact_file`.
_VERIFIED_ARTIFACT_PATHS: dict[str, tuple[str, tuple[int, int, int]]] = {}


def install_rebuilt_recording(
    temp_abs: str, canonical_abs: str, analysis_file_name: str
) -> None:
    """Install a verified rebuild and refresh its tracked byte checksum.

    The caller holds the recording's artifact lock and has checked that the
    temp's content fingerprint matches the stored recording. Replace the file
    atomically, refresh its checksum, then verify that it resolves. On failure,
    remove the temp or installed file so the next read can retry the rebuild.
    """
    import os
    from pathlib import Path

    from spyglass.common.common_nwbfile import AnalysisNwbfile

    try:
        os.replace(temp_abs, canonical_abs)
    except Exception:
        Path(temp_abs).unlink(missing_ok=True)
        raise
    try:
        AnalysisNwbfile()._resolve_external(analysis_file_name)
        AnalysisNwbfile.get_abs_path(analysis_file_name)
    except Exception:
        Path(canonical_abs).unlink(missing_ok=True)
        raise


def _reject_irreproducible_rebuild(
    computed, row: dict, *, context: str, explanation: str
) -> None:
    """Discard a staged rebuild whose content differs from the stored row.

    Parameters
    ----------
    computed
        The rebuild result, carrying the staged ``analysis_file_name`` and its
        readback ``content_hash``.
    row : dict
        The stored artifact row (``analysis_file_name``, ``content_hash``).
    context : str
        The rebuilding method, prefixed to the error and cleanup log.
    explanation : str
        Why the environment may no longer reproduce the artifact and how to
        recover; appended to the error message.

    Raises
    ------
    RecordingContentDriftError
        If ``computed.content_hash`` differs from ``row["content_hash"]``. The
        staged file is removed first; the canonical slot is never touched.
    """
    from spyglass.spikesorting.v2.exceptions import (
        RecordingContentDriftError,
    )

    if computed.content_hash == row["content_hash"]:
        return
    _unlink_staged_analysis_file(computed.analysis_file_name, context=context)
    raise RecordingContentDriftError(
        f"{context}: rebuilt content_hash {computed.content_hash} does not "
        f"match the stored content_hash {row['content_hash']} for "
        f"{row['analysis_file_name']!r}. {explanation}"
    )


def _file_identity(abs_path: str) -> tuple[int, int, int] | None:
    """``(inode, size, mtime_ns)`` of a file, or ``None`` if it is absent."""
    import os

    try:
        stat = os.stat(abs_path)
    except FileNotFoundError:
        return None
    return stat.st_ino, stat.st_size, stat.st_mtime_ns


def ensure_artifact_file(table, key: dict, analysis_file_name: str) -> str:
    """Absolute path of a cached trace artifact, rebuilt first if missing.

    The one self-heal every trace-artifact table shares: when the file is
    gone, ``table()._rebuild_nwb_artifact(key)`` restores it (a locked,
    content-verified rebuild; the DataJoint row is never deleted), and the
    path is resolved again.

    ``AnalysisNwbfile.get_abs_path`` checksums the whole file, about 1.3 s per
    GiB (measured on a 1 GiB file with a warm page cache, dominated by
    DataJoint's ``uuid_from_file``). DataJoint runs a tri-part ``make_fetch``
    twice, the second time inside the insert transaction, so that checksum
    would run twice per populate, once while the transaction is open. A
    resolution outside a transaction always goes through ``get_abs_path``.
    Inside a transaction, a file this process already resolved that way and
    whose inode, size and modification time are unchanged reuses that result:
    in a populate, that is the first ``make_fetch``'s check of the same file.
    A rebuilt or rewritten file differs and is checked again.

    Parameters
    ----------
    table : type
        The owning table class (``Recording``, ``ConcatenatedRecording`` or
        ``MotionCorrectedRecording``); it must define
        ``_rebuild_nwb_artifact(key)``.
    key : dict
        The artifact row's primary key.
    analysis_file_name : str
        The row's ``analysis_file_name``.

    Returns
    -------
    str
        Absolute path of the (present) artifact file.
    """
    from pathlib import Path

    from spyglass.common.common_nwbfile import AnalysisNwbfile

    in_transaction = AnalysisNwbfile().connection.in_transaction
    if in_transaction and analysis_file_name in _VERIFIED_ARTIFACT_PATHS:
        abs_path, identity = _VERIFIED_ARTIFACT_PATHS[analysis_file_name]
        if _file_identity(abs_path) == identity:
            return abs_path

    abs_path = AnalysisNwbfile.get_abs_path(analysis_file_name)
    if not Path(abs_path).exists():
        table()._rebuild_nwb_artifact(key)
        abs_path = AnalysisNwbfile.get_abs_path(analysis_file_name)
    identity = _file_identity(abs_path)
    if not in_transaction and identity is not None:
        _VERIFIED_ARTIFACT_PATHS[analysis_file_name] = (abs_path, identity)
    return abs_path


def stored_traces(table, key: dict, row: dict) -> StoredTraces:
    """Self-heal a cached trace artifact and resolve it for a DB-free read.

    Idempotent: a missing file is rebuilt on the first call, so a second call
    (DataJoint's in-transaction re-fetch) finds it and returns equal values.

    Parameters
    ----------
    table : type
        The owning table class; see :func:`ensure_artifact_file`.
    key : dict
        The artifact row's primary key.
    row : dict
        The artifact row (``analysis_file_name``, ``electrical_series_path``,
        ``content_hash``).

    Returns
    -------
    StoredTraces
    """
    return StoredTraces(
        abs_path=ensure_artifact_file(table, key, row["analysis_file_name"]),
        electrical_series_path=row["electrical_series_path"],
        content_hash=row["content_hash"],
    )


def rebuild_nwb_artifact(table, key) -> None:
    """Rebuild a missing recording artifact -- locked, atomic, reconciled.

    Locked + atomic-publish: acquire
    ``recording_artifact_lock(recording_id)``, double-check the file is
    still missing under the lock (a peer may have rebuilt while we waited),
    then rebuild to a PRIVATE temp file on the same filesystem as the
    canonical slot, fingerprint it, and only on a ``content_hash`` match
    ``os.replace`` it into the slot and refresh the DataJoint ``~external``
    byte checksum. A rebuild whose fingerprint diverges from the stored
    ``content_hash`` (SpikeInterface/BLAS drift, an edited raw NWB, or
    changed upstream inputs) raises ``RecordingContentDriftError`` and never
    touches the canonical slot -- drifted bytes are never served.

    Cleanup contract (all-or-nothing): the temp is unlinked on any failure;
    if ``os.replace`` ran but the checksum refresh then failed, the
    canonical is unlinked to return the slot to the missing state for the
    next (locked) ``get_recording``. Ordering is load-bearing -- the atomic
    ``os.replace`` precedes ``_resolve_external``.

    Calls ``make_fetch`` to re-derive every DB input -- the same fetch the
    populate path uses -- so a rebuild cannot drift from the original
    write's inputs. Safe to call directly (it takes the lock itself) and
    from ``get_recording`` (which does not hold the lock).

    Parameters
    ----------
    table : Recording
        A ``Recording`` instance. ``make_fetch`` and
        ``_compute_recording_artifact`` are called on it so patched
        methods take effect.
    key : dict
        Restriction selecting a single ``Recording`` row.
    """
    import time
    from pathlib import Path

    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2._recording.fingerprint import (
        recording_artifact_lock,
    )
    from spyglass.utils import logger

    row = (table & key).fetch1()
    recording_id = row["recording_id"]
    analysis_file_name = row["analysis_file_name"]
    canonical_abs = AnalysisNwbfile.get_abs_path(analysis_file_name)

    with recording_artifact_lock(recording_id):
        # Double-checked: a peer rebuilt (or the file was never gone) while
        # we waited for the lock -- nothing to do. Two readers cannot both
        # rebuild.
        if Path(canonical_abs).exists():
            return

        started = time.monotonic()
        logger.info(
            "Recording.get_recording: cache miss for "
            f"{analysis_file_name!r} (reason=missing cache); rebuilding "
            "the preprocessed artifact..."
        )
        fetched = table.make_fetch(key)
        # Rebuild to a FRESH, unregistered temp file -- same analysis dir
        # (and filesystem) as the canonical slot, so the install is an
        # atomic rename. ``_compute_recording_artifact`` returns the temp's
        # readback content fingerprint as ``content_hash``.
        rebuilt = table._compute_recording_artifact(
            fetched,
            existing_analysis_file_name=None,  # fresh temp, not the slot
            provenance_tables=recording_provenance_table(
                recording_id=fetched.sel["recording_id"],
                raw_object_id=fetched.raw_object_id,
                preprocessing_params_name=fetched.sel[
                    "preprocessing_params_name"
                ],
                sort_group_id=fetched.sel["sort_group_id"],
                reference_mode=fetched.reference_mode,
                bad_channel_handling=(
                    fetched.preprocessing_params.bad_channel_handling
                ),
            ),
        )
        _reject_irreproducible_rebuild(
            rebuilt,
            row,
            context="Recording._rebuild_nwb_artifact",
            explanation=(
                "The current environment no longer reproduces this recording "
                "(e.g. a SpikeInterface/BLAS upgrade, an edited raw NWB, or "
                "changed upstream inputs). The canonical artifact was NOT "
                "modified. Recover by restoring a backup of the artifact, "
                "rerunning the recompute under the original environment, or "
                "deleting and repopulating the Recording row (and its "
                "downstream)."
            ),
        )
        install_rebuilt_recording(
            AnalysisNwbfile.get_abs_path(rebuilt.analysis_file_name),
            canonical_abs,
            analysis_file_name,
        )

        logger.info(
            "Recording.get_recording: rebuilt + reconciled "
            f"{analysis_file_name!r} in "
            f"{time.monotonic() - started:.1f}s"
        )

    # Best-effort (outside the all-or-nothing block): clear a stale
    # RecordingArtifactRecompute deleted=1 flag now the file is back on
    # disk. File tracking is presence-aware, so a failed clear cannot hide
    # the rebuilt file -- this only keeps the flag accurate.
    clear_recompute_deleted_flag(recording_id)


def rebuild_motion_corrected_artifact(table, key) -> None:
    """Rebuild a missing corrected artifact from the SAVED motion.

    Locked on the corrected recording, double-checked under the lock,
    then ``make_fetch`` / ``make_compute`` write a fresh temp artifact:
    the saved estimate is reapplied, never estimated again, under the
    installed SpikeInterface even if it differs from the one the
    selection was made with. Only a temp
    whose ``content_hash`` equals the stored one is installed
    (``install_rebuilt_recording``); otherwise it is removed,
    ``RecordingContentDriftError`` is raised and the canonical slot is
    left untouched. A selection whose interpolation recipe now resolves
    differently, or whose application algorithm version changed, cannot
    reproduce the stored traces: ``RecordingContentDriftError`` names
    the missing file and the repair before anything is computed.

    Parameters
    ----------
    table : MotionCorrectedRecording
        A ``MotionCorrectedRecording`` instance; ``make_fetch`` and
        ``make_compute`` are called on it.
    key : dict
        Restriction selecting a single ``MotionCorrectedRecording`` row.
    """
    from pathlib import Path

    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2._motion.estimation import (
        motion_corrected_recording_artifact_lock,
        resolve_interpolation_params,
    )
    from spyglass.spikesorting.v2._motion.compute import (
        stale_corrected_selection_fields,
    )
    from spyglass.spikesorting.v2.exceptions import (
        RecordingContentDriftError,
    )
    from spyglass.utils import logger

    row = (table & key).fetch1()
    analysis_file_name = row["analysis_file_name"]
    canonical_abs = AnalysisNwbfile.get_abs_path(analysis_file_name)
    with motion_corrected_recording_artifact_lock(
        row["motion_corrected_recording_id"]
    ):
        if Path(canonical_abs).exists():
            return
        logger.info(
            "MotionCorrectedRecording.get_recording: cache miss for "
            f"{analysis_file_name!r}; reapplying the saved motion..."
        )
        master_key = {
            "motion_corrected_recording_id": row[
                "motion_corrected_recording_id"
            ]
        }
        fetched = table.make_fetch(master_key)
        # The content hash is the guard (as for Recording), so a rebuild
        # under another SpikeInterface version is allowed and installed
        # only if it reproduces the stored traces. A changed recipe
        # resolution or application algorithm cannot reproduce them.
        stale = stale_corrected_selection_fields(
            fetched.selection,
            resolve_interpolation_params(fetched.interpolation_params),
            check_spikeinterface_version=False,
        )
        if stale:
            raise RecordingContentDriftError(
                "MotionCorrectedRecording._rebuild_nwb_artifact: the "
                f"corrected recording file {analysis_file_name!r} is "
                "missing and cannot be rebuilt: its selection is stale "
                f"({'; '.join(stale)}), so reapplying the saved motion "
                "would not reproduce the stored traces. Restore the file "
                "from a backup, or delete this MotionCorrectedRecording "
                "row and everything made from it (sorts, curations) and "
                "repopulate them from a new "
                "MotionCorrectedRecordingSelection."
            )
        computed = table.make_compute(
            master_key,
            *fetched,
            allow_spikeinterface_version_change=True,
        )
        _reject_irreproducible_rebuild(
            computed,
            row,
            context="MotionCorrectedRecording._rebuild_nwb_artifact",
            explanation=(
                "The current environment no longer reproduces this corrected "
                "recording (e.g. a SpikeInterface/BLAS upgrade). The canonical "
                "artifact was NOT modified. Recover by restoring a backup or "
                "deleting and repopulating the MotionCorrectedRecording row."
            ),
        )
        install_rebuilt_recording(
            AnalysisNwbfile.get_abs_path(computed.analysis_file_name),
            canonical_abs,
            analysis_file_name,
        )


def clear_recompute_deleted_flag(recording_id) -> None:
    """Best-effort clear of stale ``RecordingArtifactRecompute.deleted``.

    After an on-demand rebuild restores the file, clear any ``deleted=1``
    recompute rows for this recording so the flag stays semantically true
    ("intentionally removed and still absent"). Best-effort: a failed clear
    is logged, never raised, and cannot hide the rebuilt file because file
    tracking is presence-aware.
    """
    from spyglass.utils import logger

    try:
        from spyglass.spikesorting.v2.recompute import (
            RecordingArtifactRecompute,
            RecordingArtifactVersions,
        )

        versions = RecordingArtifactVersions & {"recording_id": recording_id}
        flagged = (RecordingArtifactRecompute & versions & "deleted=1").fetch(
            "KEY", as_dict=True
        )
        for flagged_key in flagged:
            RecordingArtifactRecompute.update1({**flagged_key, "deleted": 0})
    except Exception as exc:  # pragma: no cover -- best-effort accuracy
        logger.warning(
            "Recording._rebuild_nwb_artifact: could not clear stale "
            f"deleted flag for recording_id={recording_id}: {exc!r}"
        )


def rebuild_concat_nwb_artifact(table, key) -> None:
    """Rebuild a missing concat artifact -- locked, atomic, content-verified.

    The concat analog of ``Recording._rebuild_nwb_artifact``. Acquire
    ``concat_recording_artifact_lock(concat_recording_id)``, double-check the
    file is still missing under the lock (a peer may have rebuilt while we
    waited), then re-run the materialization (``make_fetch`` -> ``make_compute``
    -- which also re-verifies the frozen member set against the live
    recordings) to a FRESH temp analysis file, fingerprint it, and only on a
    ``content_hash`` match ``os.replace`` it into the canonical slot and
    refresh the DataJoint ``~external`` byte checksum. A rebuild whose
    fingerprint diverges from the stored ``content_hash`` raises
    ``RecordingContentDriftError`` and never touches the canonical slot --
    drifted bytes are never served.

    Cleanup contract (all-or-nothing): the temp is unlinked on any failure;
    if ``os.replace`` ran but the checksum refresh then failed, the canonical
    is unlinked to return the slot to the missing state for the next (locked)
    ``get_recording``. Ordering is load-bearing -- the atomic ``os.replace``
    precedes ``_resolve_external``.

    An irreproducible rebuild (e.g. a changed SpikeInterface or NWB read
    path) surfaces loudly as ``RecordingContentDriftError`` rather than
    silently serving different bytes (the fingerprint's trace rounding
    absorbs sub-µV noise).
    Safe to call directly (it takes the lock itself) and from
    ``get_recording`` (which does not hold the lock).

    Parameters
    ----------
    table : ConcatenatedRecording
        A ``ConcatenatedRecording`` instance; ``make_fetch`` and
        ``make_compute`` are called on it.
    key : dict
        Restriction selecting a single ``ConcatenatedRecording`` row.
    """
    from pathlib import Path

    import numpy as np

    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2._recording.concat import (
        concat_recording_artifact_lock,
    )
    from spyglass.spikesorting.v2.exceptions import (
        RecordingContentDriftError,
    )
    from spyglass.utils import logger

    row = (table & key).fetch1()
    concat_recording_id = row["concat_recording_id"]
    analysis_file_name = row["analysis_file_name"]
    canonical_abs = AnalysisNwbfile.get_abs_path(analysis_file_name)

    with concat_recording_artifact_lock(concat_recording_id):
        # Double-checked: a peer rebuilt (or the file was never gone) while
        # we waited for the lock -- nothing to do.
        if Path(canonical_abs).exists():
            return

        logger.info(
            "ConcatenatedRecording.get_recording: cache miss for "
            f"{analysis_file_name!r} (reason=missing cache); rebuilding "
            "the concatenated artifact..."
        )
        fetched = table.make_fetch(key)
        # make_compute writes a FRESH (unregistered) temp analysis file and
        # returns its readback content fingerprint as ``content_hash``.
        computed = table.make_compute(key, *fetched)
        _reject_irreproducible_rebuild(
            computed,
            row,
            context="ConcatenatedRecording._rebuild_nwb_artifact",
            explanation=(
                "The current environment no longer reproduces this "
                "concatenated recording (e.g. a SpikeInterface/BLAS upgrade "
                "or a changed member recording). The canonical artifact was "
                "NOT modified. Recover by restoring a backup, rerunning under "
                "the original environment, or deleting and repopulating the "
                "ConcatenatedRecording row (and its downstream)."
            ),
        )
        # The traces fingerprint does not include the stored spans:
        # downstream sorts estimate noise from the statistics spans and
        # motion estimation reads the continuity spans and their first
        # and last timestamps, so a rebuild must reproduce them exactly.
        drifted = [
            name
            for name, shape in (
                ("statistics_spans", (-1, 2)),
                ("continuity_spans", (-1, 2)),
                ("continuity_start_s", (-1,)),
                ("continuity_end_s", (-1,)),
            )
            if not np.array_equal(
                np.asarray(getattr(computed, name)).reshape(shape),
                np.asarray(row[name]).reshape(shape),
            )
        ]
        if drifted:
            _unlink_staged_analysis_file(
                computed.analysis_file_name,
                context="ConcatenatedRecording._rebuild_nwb_artifact",
            )
            raise RecordingContentDriftError(
                "ConcatenatedRecording._rebuild_nwb_artifact: rebuilt "
                f"{drifted} do not match the stored values for "
                f"{analysis_file_name!r}. The canonical artifact was NOT "
                "modified. Delete and repopulate the "
                "ConcatenatedRecording row (and its downstream)."
            )

        install_rebuilt_recording(
            AnalysisNwbfile.get_abs_path(computed.analysis_file_name),
            canonical_abs,
            analysis_file_name,
        )

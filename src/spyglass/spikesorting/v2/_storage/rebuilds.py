"""Recording artifact self-heal, verified rebuilding and atomic publication.

Rebuilds call the owning table's fetch/compute methods and publish only
output whose content hash matches the stored row. DB-free at import.
"""

from __future__ import annotations

from spyglass.spikesorting.v2._storage.nwb import (
    StoredTraces,
    recording_provenance_table,
)
from spyglass.spikesorting.v2._storage.staged_outputs import (
    unlink_staged_analysis_file as _unlink_staged_analysis_file,
)

#: ``{analysis_file_name: (abs_path, file_identity)}`` for files checksummed
#: outside a transaction. See :func:`ensure_artifact_file`.
_VERIFIED_ARTIFACT_PATHS: dict[str, tuple[str, tuple[int, int, int]]] = {}


def _restore_object_id(path: str, rebuilt_id: str, stored_id: str) -> None:
    """Give the rebuilt series the object id the stored row records.

    Spyglass ``fetch_nwb`` finds each ``*object_id`` column's object by that
    id.
    """
    import h5py

    if rebuilt_id == stored_id:
        return
    with h5py.File(path, "r+") as file:
        found = []
        file.visititems(
            lambda name, obj: (
                found.append(name)
                if obj.attrs.get("object_id")
                in (rebuilt_id, rebuilt_id.encode())
                else None
            )
        )
        if len(found) != 1:
            raise ValueError(
                f"Rebuilt file {path!r} has {len(found)} objects with the "
                f"rebuilt object id {rebuilt_id!r}; expected 1."
            )
        file[found[0]].attrs["object_id"] = stored_id


def install_rebuilt_recording(
    temp_abs: str,
    canonical_abs: str,
    analysis_file_name: str,
    *,
    rebuilt_object_id: str,
    stored_object_id: str,
) -> None:
    """Install a verified rebuild and refresh its tracked byte checksum.

    The caller holds the artifact lock and has verified the content hash. On
    failure the temp or installed file is removed so the next read retries.
    """
    import os
    from pathlib import Path

    from spyglass.common.common_nwbfile import AnalysisNwbfile

    try:
        _restore_object_id(temp_abs, rebuilt_object_id, stored_object_id)
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

    ``get_abs_path`` checksums the whole file (~1.3 s/GiB) and DataJoint runs
    ``make_fetch`` twice. Inside a transaction, a file already checksummed by
    this process with unchanged inode, size and mtime is not checksummed
    again.

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
    """Rebuild a missing recording artifact under its lock.

    Rebuilds via ``make_fetch`` to a temp file and installs it only on a
    ``content_hash`` match. Takes the lock itself.

    Parameters
    ----------
    table : Recording
        A ``Recording`` instance.
    key : dict
        Restriction selecting a single ``Recording`` row.

    Raises
    ------
    RecordingContentDriftError
        If the rebuild does not reproduce the stored ``content_hash``.
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
        # A peer may have rebuilt it while we waited.
        if Path(canonical_abs).exists():
            return

        started = time.monotonic()
        logger.info(
            "Recording.get_recording: cache miss for "
            f"{analysis_file_name!r} (reason=missing cache); rebuilding "
            "the preprocessed artifact..."
        )
        fetched = table.make_fetch(key)
        # Same filesystem as the canonical slot, so install is an atomic rename.
        rebuilt = table._compute_recording_artifact(
            fetched,
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
            rebuilt_object_id=rebuilt.object_id,
            stored_object_id=row["object_id"],
        )

        logger.info(
            "Recording.get_recording: rebuilt + reconciled "
            f"{analysis_file_name!r} in "
            f"{time.monotonic() - started:.1f}s"
        )

    clear_recompute_deleted_flag(recording_id)


def rebuild_motion_corrected_artifact(table, key) -> None:
    """Rebuild a missing corrected artifact from the SAVED motion.

    The saved estimate is reapplied, under any SpikeInterface version; only a
    ``content_hash`` match is installed. A stale selection is refused before
    computing.

    Parameters
    ----------
    table : MotionCorrectedRecording
        A ``MotionCorrectedRecording`` instance; ``make_fetch`` and
        ``make_compute`` are called on it.
    key : dict
        Restriction selecting a single ``MotionCorrectedRecording`` row.

    Raises
    ------
    RecordingContentDriftError
        If the selection is stale or the rebuild does not reproduce the
        stored ``content_hash``.
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
        # The content hash guards SI version changes; a changed recipe or
        # algorithm cannot reproduce the traces.
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
            rebuilt_object_id=computed.object_id,
            stored_object_id=row["object_id"],
        )


def clear_recompute_deleted_flag(recording_id) -> None:
    """Best-effort clear of stale ``RecordingArtifactRecompute.deleted``.

    ``deleted=1`` means "removed and still absent". A failed clear is logged;
    file tracking is presence-aware, so it cannot hide the rebuilt file.
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
    """Rebuild a missing concat artifact under its lock.

    Like :func:`rebuild_nwb_artifact`; the stored spans must also match.

    Parameters
    ----------
    table : ConcatenatedRecording
        A ``ConcatenatedRecording`` instance; ``make_fetch`` and
        ``make_compute`` are called on it.
    key : dict
        Restriction selecting a single ``ConcatenatedRecording`` row.

    Raises
    ------
    RecordingContentDriftError
        If the rebuilt content hash or spans differ from the stored row.
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
        if Path(canonical_abs).exists():
            return

        logger.info(
            "ConcatenatedRecording.get_recording: cache miss for "
            f"{analysis_file_name!r} (reason=missing cache); rebuilding "
            "the concatenated artifact..."
        )
        fetched = table.make_fetch(key)
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
        # The content hash omits the spans, which noise and motion
        # estimation read.
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
            rebuilt_object_id=computed.object_id,
            stored_object_id=row["object_id"],
        )

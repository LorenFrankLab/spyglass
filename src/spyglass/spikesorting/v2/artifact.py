"""Artifact detection over a preprocessed Recording.

Tables (all final-shape under the zero-migration policy):
    ArtifactDetectionParameters       -- threshold detection parameters.
    SharedArtifactGroup (+ Member)    -- opt-in cross-recording detection.
    ArtifactSelection                 -- source parts encode input shape.
        .RecordingSource              -- single-recording path (default).
        .SharedArtifactGroupSource    -- cross-recording path (#928).
    ArtifactDetection                 -- writes IntervalList rows; no part.

The master is named ``ArtifactSelection`` (a shorter alternative to the
verbose v1 ``ArtifactDetectionSelection`` pattern) so that the
auto-generated FK constraint name for the source part referencing
``SharedArtifactGroup`` fits inside MySQL's 64-character identifier
limit -- the longer master + the longer part together overflow. The
shorter master also matches the v2 single-master-per-topic convention:
there is only one v2 artifact-related ``Selection`` table, so the
``Detection`` qualifier is redundant.

Artifact-removed valid times live in ``common.IntervalList`` rather than
a dedicated part table -- the UUID-suffixed name prevents collision with
human-authored session intervals while letting downstream
IntervalList-querying code consume them through the standard interface.

``insert1`` on ``ArtifactDetectionParameters`` is live and
Pydantic-validates the ``params`` blob; ``insert_selection``, ``make``,
``get_artifact_removed_intervals``, ``delete``, and
``SharedArtifactGroup.insert_group`` are forward-declared stubs that
raise ``NotImplementedError`` until the matching runtime change lands.
"""

from __future__ import annotations

import uuid
from typing import NamedTuple

import datajoint as dj
import numpy as np
from spikeinterface.core.job_tools import (
    ChunkRecordingExecutor,
    ensure_n_jobs,
    job_keys,
)

from spyglass.common import IntervalList, Session  # noqa: F401
from spyglass.spikesorting.v2._params.artifact_detection import (
    ArtifactDetectionParamsSchema,
)
from spyglass.spikesorting.v2.recording import Recording
from spyglass.spikesorting.v2.utils import (
    SourceResolution,
    _assert_schema_version_matches,
    _assert_v2_db_safe,
    _get_recording_timestamps,
    _insert_row_to_dict,
    _validate_params,
    find_orphaned_masters,
    transaction_or_noop,
)
from spyglass.utils import SpyglassMixin, SpyglassMixinPart, logger

_assert_v2_db_safe()
schema = dj.schema("spikesorting_v2_artifact")


def _init_artifact_worker(
    recording,
    zscore_thresh,
    amplitude_thresh_uV,
    proportion_above_thresh,
):
    """Per-worker initializer for the chunked artifact scan.

    Mirrors v1's ``spikesorting/utils.py:_init_artifact_worker`` (this is a
    self-contained v2 copy -- v2 must not import the v1 helper). On a
    multi-process pool the ``recording`` arrives as a ``to_dict()`` blob and is
    re-hydrated with ``si.load`` (SI 0.104 renamed v1's ``load_extractor``);
    on the single-process / thread path the live recording object is passed
    straight through. ``n_required`` and the per-channel ``gains`` are constant
    across chunks, so they are resolved once here and cached in the worker
    context rather than recomputed per chunk.
    """
    import spikeinterface as si

    recording = (
        si.load(recording) if isinstance(recording, dict) else recording
    )
    n_channels = len(recording.get_channel_ids())
    return {
        "recording": recording,
        "zscore_thresh": zscore_thresh,
        "amplitude_thresh_uV": amplitude_thresh_uV,
        "n_required": int(np.ceil(proportion_above_thresh * n_channels)),
        "gains": recording.get_channel_gains(),
    }


def _compute_artifact_chunk(segment_index, start_frame, end_frame, worker_ctx):
    """Flag artifact frame indices within a ``[start_frame, end_frame)`` chunk.

    Reproduces the EXACT per-frame detection math of the former full-in-memory
    scan, applied to a single chunk: scale to µV with the stored per-channel
    gains, then OR-combine the amplitude and across-channel z-score detectors.
    The across-channel (``axis=1``) z-score uses only the chunk row's own
    columns, so it is identical regardless of where the chunk boundaries fall --
    this is what makes the chunked output frame-identical to the in-memory one.
    Returns the GLOBAL (``+ start_frame``) ascending frame indices flagged in
    this chunk.

    Peak working set per chunk ≈ ``4 × (end_frame - start_frame) × n_channels ×
    4 bytes`` (raw int slice + float32 µV copy + abs + z-score intermediate),
    independent of the full recording length.
    """
    recording = worker_ctx["recording"]
    zscore_thresh = worker_ctx["zscore_thresh"]
    amplitude_thresh_uV = worker_ctx["amplitude_thresh_uV"]
    n_required = worker_ctx["n_required"]
    gains = worker_ctx["gains"]

    traces = recording.get_traces(
        segment_index=segment_index,
        start_frame=start_frame,
        end_frame=end_frame,
        return_in_uV=False,
    )
    traces_uv = traces.astype(np.float32) * gains[None, :]
    absolute = np.abs(traces_uv)

    if amplitude_thresh_uV is not None:
        above_amp = absolute > amplitude_thresh_uV
    else:
        above_amp = np.zeros_like(absolute, dtype=bool)
    if zscore_thresh is not None:
        ch_mean = traces_uv.mean(axis=1, keepdims=True)
        ch_std = traces_uv.std(axis=1, keepdims=True) + 1e-12
        zscores = np.abs((traces_uv - ch_mean) / ch_std)
        above_z = zscores > zscore_thresh
    else:
        above_z = np.zeros_like(absolute, dtype=bool)

    if amplitude_thresh_uV is not None and zscore_thresh is not None:
        channel_hit = above_amp | above_z
    elif amplitude_thresh_uV is not None:
        channel_hit = above_amp
    else:
        channel_hit = above_z

    local = (channel_hit.sum(axis=1) >= n_required).nonzero()[0]
    return local + start_frame


class ArtifactFetched(NamedTuple):
    """DB-side inputs gathered by :meth:`ArtifactDetection.make_fetch`.

    For ``source.kind == "recording"`` only the single-recording
    fields are populated and ``member_recording_ids`` /
    ``member_nwb_file_names`` are ``None``. For
    ``source.kind == "shared_artifact_group"`` the per-recording
    fields are populated as ordered tuples (length n_members) and
    the single-recording ``nwb_file_name`` is the common parent
    session validated at ``insert_group`` time.
    """

    source: SourceResolution
    validated: ArtifactDetectionParamsSchema
    nwb_file_name: str
    artifact_job_kwargs: dict | None
    member_recording_ids: tuple | None = None
    member_nwb_file_names: tuple | None = None


class ArtifactComputed(NamedTuple):
    """Outputs of :meth:`ArtifactDetection.make_compute`.

    ``per_member_nwb_files`` is a tuple of distinct
    ``nwb_file_name`` strings the ``make_insert`` step must write
    one ``IntervalList`` row per. For the single-recording path it
    is ``(nwb_file_name,)`` of length 1; for the shared-group path
    it is the distinct set across the member recordings.
    """

    valid_times: np.ndarray
    nwb_file_name: str
    per_member_nwb_files: tuple = ()


@schema
class ArtifactDetectionParameters(SpyglassMixin, dj.Lookup):
    """Validated artifact-detection parameter blob.

    The ``params`` blob is validated by
    :class:`ArtifactDetectionParamsSchema`. ``insert_default`` ships two
    presets: ``"none"`` (detect=False, skip artifact scanning) and
    ``"default"`` (amplitude threshold + proportion-above-thresh).

    ``job_kwargs`` is the optional per-row SpikeInterface job-kwargs blob that
    governs the chunked detection scan
    (``ArtifactDetection._scan_artifact_frames``). It is merged over the
    SI-global and ``dj.config['custom']
    ['spikesorting_v2_job_kwargs']`` defaults by ``_resolved_job_kwargs``. The
    memory-relevant key is the chunk size -- ``chunk_duration`` (e.g. ``"1s"``,
    the default), ``chunk_size`` (frames), or ``chunk_memory`` -- which bounds
    peak working set at ``~4 × chunk_frames × n_channels × 4 bytes``. ``n_jobs``
    controls the worker-pool size (default 1, serial in-process, matching v1).
    """

    definition = """
    artifact_params_name: varchar(64)
    ---
    params: blob
    params_schema_version=2: int
    job_kwargs=null: blob  # SI job-kwargs for chunked scan; see docstring
    """

    # Row-level ``params_schema_version`` matches the inner
    # ``ArtifactDetectionParamsSchema.schema_version`` (bumped to 2
    # for ``min_length_s``).
    _DEFAULT_CONTENTS: tuple = (
        (
            "none",
            ArtifactDetectionParamsSchema(
                detect=False, amplitude_thresh_uV=None
            ).model_dump(),
            2,
            None,
        ),
        (
            "default",
            ArtifactDetectionParamsSchema().model_dump(),
            2,
            None,
        ),
    )

    def insert1(self, row, **kwargs):
        row = dict(row)
        row["params"] = _validate_params(
            ArtifactDetectionParamsSchema, row["params"]
        )
        _assert_schema_version_matches(
            row,
            ArtifactDetectionParamsSchema,
            table_name="ArtifactDetectionParameters",
        )
        super().insert1(row, **kwargs)

    def insert(self, rows, **kwargs):
        # Mirror ``insert1``'s validation across a bulk insert so an
        # ``insert([...])`` (including ``insert_default``'s positional
        # rows) cannot bypass schema validation or the
        # params_schema_version check.
        rows = [_insert_row_to_dict(r, self.heading.names) for r in rows]
        for row in rows:
            row["params"] = _validate_params(
                ArtifactDetectionParamsSchema, row["params"]
            )
            _assert_schema_version_matches(
                row,
                ArtifactDetectionParamsSchema,
                table_name="ArtifactDetectionParameters",
            )
        super().insert(rows, **kwargs)

    @classmethod
    def insert_default(cls):
        """Insert v2 default artifact-detection presets if missing."""
        cls.insert(cls._DEFAULT_CONTENTS, skip_duplicates=True)


@schema
class SharedArtifactGroup(SpyglassMixin, dj.Manual):
    """Named bundle of Recording rows that share an artifact-detection pass.

    Addresses Spyglass issue #928 (behavioral artifacts visible on every
    probe -- chewing, licking, head-bumps). Per-recording artifact
    detection misses these because each sort group is processed
    independently. ``SharedArtifactGroup`` lets users declare a set of
    Recording rows from the same session whose artifact intervals should
    be unioned; one detection pass over the union of channels produces a
    shared set of valid times applied to every member.

    All members must belong to one session (enforced by the master row's
    Session FK and re-checked by ``insert_group``).
    """

    definition = """
    shared_artifact_group_name: varchar(64)
    ---
    -> Session
    """

    class Member(SpyglassMixinPart):
        definition = """
        -> master
        -> Recording
        """

    @classmethod
    def insert_group(cls, name: str, members: list[dict]) -> None:
        """Insert master + Member rows; validate session consistency.

        A ``SharedArtifactGroup`` is a named bundle of populated
        ``Recording`` rows whose artifact-detection pass should run
        ONCE over the union of channels. The matching
        ``ArtifactDetection.make_compute`` branch unions the channels
        across all members, runs the same threshold scan as the
        single-recording path, and ``make_insert`` writes one
        ``IntervalList`` row per member ``nwb_file_name`` so each
        member session sees the artifact times in its own namespace.

        Parameters
        ----------
        name
            Group name (PK on the master). Must be unique within the
            installation.
        members
            List of dicts identifying member recordings. Each dict
            must contain at least ``recording_id`` (other fields are
            ignored so the caller can pass arbitrary upstream rows
            / PKs).

        Raises
        ------
        ValueError
            If ``members`` is empty, if any member ``recording_id``
            is not a populated ``Recording``, or if members span
            more than one session. The shared-group detection
            requires all members to share a time axis -- mixing
            sessions makes the artifact-removed valid times
            undefined.
        """
        from spyglass.spikesorting.v2.recording import (
            Recording,
            RecordingSelection,
        )

        if not members:
            raise ValueError(
                "SharedArtifactGroup.insert_group: members list is empty. "
                "Pass at least one recording_id dict."
            )

        member_recording_ids = []
        for m in members:
            if "recording_id" not in m:
                raise ValueError(
                    "SharedArtifactGroup.insert_group: every member dict "
                    "must include 'recording_id'. Got: " + str(m)
                )
            member_recording_ids.append(m["recording_id"])

        # All recording_ids must reference populated Recording rows.
        missing = [
            rid
            for rid in member_recording_ids
            if not (Recording & {"recording_id": rid})
        ]
        if missing:
            raise ValueError(
                "SharedArtifactGroup.insert_group: recording_id(s) "
                f"{missing} are not in Recording. Populate Recording for "
                "those selections first."
            )

        # Time-axis compatibility check. ``RecordingSelection`` is
        # keyed by (nwb_file_name, sort_group_id, interval_list_name,
        # preproc_params_name, team_name); same NWB does NOT imply
        # same time axis. ``si.aggregate_channels`` requires every
        # member to share ``n_samples``, ``sampling_frequency``, and
        # dtype, otherwise the union construction crashes deep inside
        # SI with an opaque "shape mismatch" error at populate time.
        # Catch the invariant at insert time so the user gets a
        # clear diagnostic before populate.
        per_member_recording_rows = (
            Recording
            & [{"recording_id": rid} for rid in member_recording_ids]
        ).fetch("recording_id", "sampling_frequency", as_dict=True)
        per_member_selection_rows = (
            RecordingSelection
            & [{"recording_id": rid} for rid in member_recording_ids]
        ).fetch(
            "recording_id",
            "nwb_file_name",
            "interval_list_name",
            as_dict=True,
        )
        # Index by recording_id so we can compose the two relations.
        rec_by_id = {
            str(r["recording_id"]): r for r in per_member_recording_rows
        }
        sel_by_id = {
            str(r["recording_id"]): r for r in per_member_selection_rows
        }

        sessions = {sel_by_id[str(rid)]["nwb_file_name"] for rid in member_recording_ids}
        if len(sessions) != 1:
            raise ValueError(
                "SharedArtifactGroup.insert_group: members span "
                f"{len(sessions)} sessions ({sorted(sessions)}); a shared "
                "artifact-detection pass only makes sense within one "
                "session because the detection writes IntervalList rows "
                "keyed by (nwb_file_name, interval_list_name)."
            )
        (nwb_file_name,) = sessions

        # Sampling frequency must match: SI's ``aggregate_channels``
        # requires identical fs. A typo in upstream preproc params
        # could silently produce different ``sampling_frequency``
        # values on the same NWB; catch it here.
        sampling_frequencies = {
            float(rec_by_id[str(rid)]["sampling_frequency"])
            for rid in member_recording_ids
        }
        if len(sampling_frequencies) != 1:
            raise ValueError(
                "SharedArtifactGroup.insert_group: members have "
                f"differing sampling frequencies "
                f"{sorted(sampling_frequencies)}; "
                "``si.aggregate_channels`` requires identical fs."
            )

        # Exact time-axis check. Equal sample counts alone are not enough:
        # two interval selections from the same session can have the same
        # length, sampling frequency, and dtype while starting at different
        # wall-clock times. ``si.aggregate_channels`` stacks by frame index,
        # so the full timestamp vector must match exactly for every member.
        # Load each member's preprocessed recording through ``Recording.
        # get_recording`` (the same path ``make_compute`` uses) and require
        # EXACT ``get_num_samples()``, ``get_dtype()``, and timestamps.
        per_member_sizes: dict[str, tuple[int, str]] = {}
        reference_timestamps = None
        reference_rid = None
        for rid in member_recording_ids:
            try:
                rec_obj = Recording().get_recording({"recording_id": rid})
            except Exception as exc:
                raise RuntimeError(
                    "SharedArtifactGroup.insert_group: failed to load "
                    f"preprocessed recording for recording_id={rid!r} "
                    f"({type(exc).__name__}: {exc}). The strict "
                    "n_samples / dtype check at insert time requires "
                    "every member recording to be readable."
                ) from exc
            n_samples = int(rec_obj.get_num_samples())
            timestamps = np.asarray(
                _get_recording_timestamps(rec_obj), dtype=np.float64
            )
            if len(timestamps) != n_samples:
                raise ValueError(
                    "SharedArtifactGroup.insert_group: recording_id="
                    f"{rid!r} reports get_num_samples()={n_samples}, "
                    f"but its timestamp vector has length "
                    f"{len(timestamps)}. The preprocessed recording is "
                    "internally inconsistent and cannot be used for a "
                    "shared artifact group."
                )
            if reference_timestamps is None:
                reference_timestamps = timestamps
                reference_rid = rid
            elif not np.array_equal(reference_timestamps, timestamps):
                if reference_timestamps.shape != timestamps.shape:
                    mismatch_detail = (
                        "timestamp vector shapes differ: "
                        f"recording_id={reference_rid!r} has "
                        f"{reference_timestamps.shape}, recording_id="
                        f"{rid!r} has {timestamps.shape}"
                    )
                else:
                    mismatch_indices = np.flatnonzero(
                        reference_timestamps != timestamps
                    )
                    first = int(mismatch_indices[0])
                    mismatch_detail = (
                        f"first mismatch at sample {first}: "
                        f"recording_id={reference_rid!r} has "
                        f"{reference_timestamps[first]!r}, "
                        f"recording_id={rid!r} has {timestamps[first]!r}"
                    )
                raise ValueError(
                    "SharedArtifactGroup.insert_group: members have "
                    f"differing exact timestamps ({mismatch_detail}). "
                    "Shared artifact detection requires time-aligned "
                    "recordings; use recordings from the same interval "
                    "identity or run separate artifact detections."
                )
            per_member_sizes[str(rid)] = (
                n_samples,
                str(rec_obj.get_dtype()),
            )
        distinct_n_samples = {n for (n, _) in per_member_sizes.values()}
        if len(distinct_n_samples) != 1:
            raise ValueError(
                "SharedArtifactGroup.insert_group: members have "
                f"differing exact n_samples "
                f"{sorted(distinct_n_samples)}; "
                "``si.aggregate_channels`` requires identical sample "
                "counts. Likely cause: different "
                "``interval_list_name`` values on the upstream "
                "RecordingSelection rows."
            )
        distinct_dtypes = {dt for (_, dt) in per_member_sizes.values()}
        if len(distinct_dtypes) != 1:
            raise ValueError(
                "SharedArtifactGroup.insert_group: members have "
                f"differing dtypes {sorted(distinct_dtypes)}; "
                "``si.aggregate_channels`` requires identical dtype."
            )

        master_row = {
            "shared_artifact_group_name": name,
            "nwb_file_name": nwb_file_name,
        }
        member_rows = [
            {
                "shared_artifact_group_name": name,
                "recording_id": rid,
            }
            for rid in member_recording_ids
        ]

        with transaction_or_noop(cls.connection):
            cls.insert1(master_row)
            cls.Member.insert(member_rows)


@schema
class ArtifactSelection(SpyglassMixin, dj.Manual):
    """One row per (parameters, source) artifact detection request.

    Source part rows make the input shape explicit: exactly one of
    ``RecordingSource`` (single-recording, default) or
    ``SharedArtifactGroupSource`` (cross-recording, opt-in) must exist
    for each selection row. Enforced by ``insert_selection`` and
    re-checked at the start of ``ArtifactDetection.make()`` per the
    shared-contracts Source Part Pattern.
    """

    definition = """
    artifact_id: uuid
    ---
    -> ArtifactDetectionParameters
    """

    class RecordingSource(SpyglassMixinPart):
        definition = """
        -> master
        ---
        -> Recording
        """

    class SharedArtifactGroupSource(SpyglassMixinPart):
        definition = """
        -> master
        ---
        -> SharedArtifactGroup
        """

    @classmethod
    def insert_selection(cls, key: dict) -> dict:
        """Insert master + exactly one source part; return PK-only dict.

        Reads exactly one source key from ``key`` (``recording_id`` xor
        ``shared_artifact_group_name``), finds an existing master row
        that matches both the master fields and the source row, or
        mints a new UUID and inserts master + source part together.

        The find step joins the master to the chosen source part so a
        prior ``ArtifactSelection`` with a different source is not
        silently reused. Source-part atomicity is enforced via the
        ``SchemaBypassError`` re-check at the top of
        ``ArtifactDetection.make()`` (Layer 2 of the source-part
        pattern).

        Raises
        ------
        ValueError
            If zero or two source keys are supplied in ``key``.
        DuplicateSelectionError
            If more than one master+source row matches the supplied
            logical identity (an integrity bug; v2 inserts via this
            helper should never produce duplicates).
        """
        from spyglass.spikesorting.v2.exceptions import (
            DuplicateSelectionError,
        )

        has_recording = "recording_id" in key
        has_shared = "shared_artifact_group_name" in key
        if has_recording == has_shared:
            raise ValueError(
                "ArtifactSelection.insert_selection requires exactly one "
                "source key. Provide either recording_id (single-recording "
                "path) or shared_artifact_group_name (cross-recording "
                "path), not both and not neither. Got: "
                f"recording_id={'set' if has_recording else 'unset'}, "
                f"shared_artifact_group_name="
                f"{'set' if has_shared else 'unset'}."
            )

        master_field = "artifact_params_name"
        if master_field not in key:
            raise ValueError(
                "ArtifactSelection.insert_selection requires "
                f"{master_field!r} in key."
            )
        master_restriction = {master_field: key[master_field]}

        if has_recording:
            source_part = cls.RecordingSource
            source_restriction = {"recording_id": key["recording_id"]}
        else:
            source_part = cls.SharedArtifactGroupSource
            source_restriction = {
                "shared_artifact_group_name": key[
                    "shared_artifact_group_name"
                ]
            }

        # Find existing master+source rows by joining the master to the
        # selected source part.
        joined = (cls * source_part) & master_restriction & source_restriction
        existing = joined.fetch("KEY", as_dict=True)
        existing_master_keys = [
            {k: v for k, v in row.items() if k in cls.primary_key}
            for row in existing
        ]
        # Dedupe master PKs across the join (one master * source row may
        # appear once per source-part row but our integrity test pins
        # exactly-one source per master).
        unique = {tuple(sorted(d.items())) for d in existing_master_keys}
        if len(unique) == 1:
            return dict(next(iter(unique)))
        if len(unique) > 1:
            raise DuplicateSelectionError(
                f"ArtifactSelection has {len(unique)} master rows for "
                f"{master_restriction | source_restriction}. v2 inserts "
                "via this helper should not produce duplicates."
            )

        # Translate the would-be DataJoint FK IntegrityError into a
        # clear "missing default row" message before the inserts attempt.
        from spyglass.spikesorting.v2.utils import _ensure_lookup_row_exists

        _ensure_lookup_row_exists(
            ArtifactDetectionParameters,
            master_restriction,
            helper_name="ArtifactSelection.insert_selection",
            insert_default_path="ArtifactDetectionParameters.insert_default()",
        )

        new_master_key = {
            **master_restriction,
            "artifact_id": uuid.uuid4(),
        }
        new_part_key = {
            "artifact_id": new_master_key["artifact_id"],
            **source_restriction,
        }
        with transaction_or_noop(cls.connection):
            cls.insert1(new_master_key)
            source_part.insert1(new_part_key)
        return {k: new_master_key[k] for k in cls.primary_key}

    @classmethod
    def prune_orphaned_selections(cls, dry_run: bool = True) -> list[dict]:
        """Find or delete master rows that have no source-part row.

        The source-part pattern's transactional ``insert_selection``
        guarantees every master is born with exactly one source row,
        but DataJoint cannot enforce that invariant across two part
        tables -- if a maintenance script cascade-deletes from
        ``Recording`` or ``SharedArtifactGroup`` it leaves a master row
        with zero source children. This helper finds and (optionally)
        removes those orphans via cautious_delete so downstream
        ``ArtifactDetection`` / ``Sorting`` / ``CurationV2`` rows are
        reviewed by the normal cascade preview.

        Parameters
        ----------
        dry_run
            If True (default), return the orphan master PKs without
            deleting. If False, cautious_delete the orphans.

        Returns
        -------
        list of dict
            Orphan master PKs (``{"artifact_id": ...}``). Empty when
            no orphans remain.
        """
        orphans = find_orphaned_masters(
            cls,
            [cls.RecordingSource, cls.SharedArtifactGroupSource],
        )
        if dry_run or not orphans:
            return orphans
        for orphan in orphans:
            (cls & orphan).cautious_delete()
        return orphans

    @classmethod
    def resolve_source(cls, key: dict) -> SourceResolution:
        """Return the source-resolution record for an artifact selection.

        Layer 2 of the source-part pattern: fetches source part rows for
        the master key and asserts exactly one exists. Called at the top
        of ``ArtifactDetection.make()`` to catch rows inserted via raw
        ``dj.Manual.insert1()`` that bypassed ``insert_selection``.

        Raises
        ------
        SchemaBypassError
            If zero or multiple source part rows exist for ``key``.
        """
        from spyglass.spikesorting.v2.exceptions import SchemaBypassError

        master_key = {k: v for k, v in key.items() if k in cls.primary_key}
        rec_rows = (cls.RecordingSource & master_key).fetch(as_dict=True)
        shared_rows = (cls.SharedArtifactGroupSource & master_key).fetch(
            as_dict=True
        )
        total = len(rec_rows) + len(shared_rows)
        if total != 1:
            raise SchemaBypassError(
                f"ArtifactSelection {master_key} has {total} source part "
                "rows; expected exactly one. Use "
                "ArtifactSelection.insert_selection() to add or remove "
                "this selection."
            )
        if rec_rows:
            return SourceResolution(
                kind="recording",
                key={"recording_id": rec_rows[0]["recording_id"]},
            )
        return SourceResolution(
            kind="shared_artifact_group",
            key={
                "shared_artifact_group_name": shared_rows[0][
                    "shared_artifact_group_name"
                ]
            },
        )


@schema
class ArtifactDetection(SpyglassMixin, dj.Computed):
    """Artifact-removed valid times for a Recording or SharedArtifactGroup.

    The valid times are written to ``common.IntervalList`` under name
    ``f"artifact_{artifact_id}"`` at the end of ``make()`` -- one row per
    affected session. Single-recording detections write exactly one row;
    cross-recording detections write one row per distinct member
    ``nwb_file_name``.
    """

    definition = """
    -> ArtifactSelection
    """

    # Tri-part dispatch enables non-daemon parallel populate via
    # Spyglass's ``PopulateMixin`` and -- more importantly -- moves
    # the long-running detection loop OUTSIDE the framework
    # transaction. See ``Recording`` for the same pattern and the
    # background on DataJoint #1170 / Spyglass #1030.
    _parallel_make = True

    def make_fetch(self, key):
        """Read every DB input the compute step needs.

        Layer-2 source re-check happens here so a row whose source
        part was deleted (cascade orphan) fails fast inside
        ``make_fetch``; raising here is cheap and deterministic.
        For the shared-artifact-group source, fetch the ordered
        member ``recording_id`` + ``nwb_file_name`` tuples so
        ``make_compute`` can union their channels without further
        DB I/O.  Tuple return is DeepHash-stable across the two
        calls DataJoint makes (before compute and again inside the
        transaction).
        """
        from spyglass.spikesorting.v2.recording import RecordingSelection

        source = ArtifactSelection.resolve_source(key)

        params_blob, artifact_job_kwargs = (
            ArtifactDetectionParameters * (ArtifactSelection & key)
        ).fetch1("params", "job_kwargs")
        validated = ArtifactDetectionParamsSchema.model_validate(params_blob)

        if source.kind == "recording":
            nwb_file_name = (
                RecordingSelection
                & {"recording_id": source.key["recording_id"]}
            ).fetch1("nwb_file_name")
            return ArtifactFetched(
                source=source,
                validated=validated,
                nwb_file_name=nwb_file_name,
                artifact_job_kwargs=artifact_job_kwargs,
                member_recording_ids=None,
                member_nwb_file_names=None,
            )

        if source.kind == "shared_artifact_group":
            # Members come from SharedArtifactGroup.Member ordered by
            # ``recording_id`` so the deterministic tuple shape is
            # DeepHash-stable across both make_fetch calls. The
            # per-member ``nwb_file_name`` lookup is bulk-fetched
            # via a join.
            members = (
                SharedArtifactGroup.Member
                * RecordingSelection
                & {
                    "shared_artifact_group_name": source.key[
                        "shared_artifact_group_name"
                    ]
                }
            ).fetch("recording_id", "nwb_file_name", as_dict=True, order_by="recording_id")
            if not members:
                raise RuntimeError(
                    "ArtifactDetection.make: SharedArtifactGroup "
                    f"{source.key['shared_artifact_group_name']!r} has "
                    "zero members; insert_group should have rejected "
                    "the empty case."
                )
            member_recording_ids = tuple(
                str(m["recording_id"]) for m in members
            )
            member_nwb_file_names = tuple(
                m["nwb_file_name"] for m in members
            )
            # ``insert_group`` validated single-session, so the
            # ``nwb_file_name`` is unique; surface the canonical one
            # so callers that only need the master nwb_file_name
            # (single-recording-shaped consumers) keep working.
            nwb_file_name = member_nwb_file_names[0]
            return ArtifactFetched(
                source=source,
                validated=validated,
                nwb_file_name=nwb_file_name,
                artifact_job_kwargs=artifact_job_kwargs,
                member_recording_ids=member_recording_ids,
                member_nwb_file_names=member_nwb_file_names,
            )

        raise RuntimeError(
            f"ArtifactDetection.make: unexpected source kind {source.kind!r}."
        )

    def make_compute(
        self,
        key,
        source,
        validated,
        nwb_file_name,
        artifact_job_kwargs,
        member_recording_ids=None,
        member_nwb_file_names=None,
    ):
        """Run artifact detection outside any DB transaction.

        For ``source.kind == "recording"``: load the single cached
        preprocessed recording and scan it.

        For ``source.kind == "shared_artifact_group"``: load each
        member's preprocessed recording, ``si.aggregate_channels``
        them into a single recording (column-stack along the
        channel axis -- the union of channels), then scan the
        unioned recording ONCE. The same ``valid_times`` array
        gets written into every member's ``IntervalList`` so each
        member session sees the artifact times in its own
        namespace.

        Returns ``valid_times`` plus the per-member-nwb-file
        target list ``make_insert`` writes to.
        """
        # Merge the SI-global, DataJoint-config, and per-row job_kwargs so
        # the chunked ``_scan_artifact_frames`` pass honors ``n_jobs`` /
        # ``chunk_duration`` overrides. The stored per-row blob (the audit's
        # formerly-dead ``job_kwargs`` column) is now functional here.
        from spyglass.spikesorting.v2.utils import _resolved_job_kwargs

        resolved_job_kwargs = _resolved_job_kwargs(artifact_job_kwargs)

        if source.kind == "recording":
            recording = Recording().get_recording(
                {"recording_id": source.key["recording_id"]}
            )
            valid_times = self._detect_artifacts(
                recording,
                validated,
                context=(
                    f" for artifact_id={key['artifact_id']}, "
                    f"recording_id={source.key['recording_id']}"
                ),
                job_kwargs=resolved_job_kwargs,
            )
            return ArtifactComputed(
                valid_times=valid_times,
                nwb_file_name=nwb_file_name,
                per_member_nwb_files=(nwb_file_name,),
            )

        # shared_artifact_group source.
        import spikeinterface as si

        if not member_recording_ids:
            raise RuntimeError(
                "ArtifactDetection.make_compute: shared-group source "
                "has no member_recording_ids; make_fetch contract "
                "violated."
            )
        per_member_recordings = [
            Recording().get_recording({"recording_id": rid})
            for rid in member_recording_ids
        ]
        # ``aggregate_channels`` column-stacks along the channel
        # axis (column = channel). All members must share the same
        # n_samples + fs + dtype, which the same-session check at
        # ``insert_group`` guarantees -- members live in one
        # ``nwb_file_name`` so their time axes match by
        # construction.
        unioned = si.aggregate_channels(per_member_recordings)
        valid_times = self._detect_artifacts(
            unioned,
            validated,
            context=(
                f" for artifact_id={key['artifact_id']}, "
                f"shared_artifact_group="
                f"{source.key['shared_artifact_group_name']}"
            ),
            job_kwargs=resolved_job_kwargs,
        )
        # One IntervalList row per distinct member nwb_file_name;
        # ``insert_group`` enforces single-session, so the distinct
        # set has length 1 today, but keep the tuple shape so a
        # future cross-session relaxation does not need a make-body
        # change.
        per_member_nwb_files = tuple(dict.fromkeys(member_nwb_file_names))
        return ArtifactComputed(
            valid_times=valid_times,
            nwb_file_name=nwb_file_name,
            per_member_nwb_files=per_member_nwb_files,
        )

    def make_insert(
        self,
        key,
        valid_times,
        nwb_file_name,
        per_member_nwb_files=(),
    ):
        """Write the artifact ``IntervalList`` + master row atomically.

        DataJoint's tri-part dispatch opens the framework transaction
        around this method; the inner ``transaction_or_noop`` is a
        no-op (yields without re-opening when a transaction is
        already active). Kept defensively so an out-of-populate
        caller still gets atomic registration.

        Writes ONE ``IntervalList`` row per distinct member
        ``nwb_file_name`` so each session sees the artifact times.
        For the single-recording path this is one row keyed by the
        master's ``nwb_file_name``.
        """
        from spyglass.spikesorting.v2.utils import (
            artifact_interval_list_name,
            transaction_or_noop,
        )

        interval_list_name = artifact_interval_list_name(key["artifact_id"])
        # Backwards compat: callers that constructed
        # ``ArtifactComputed`` without ``per_member_nwb_files``
        # (older test stubs) fall back to the single
        # ``nwb_file_name``.
        targets = per_member_nwb_files or (nwb_file_name,)
        interval_rows = [
            {
                "nwb_file_name": member_nwb,
                "interval_list_name": interval_list_name,
                "valid_times": valid_times,
                "pipeline": "spikesorting_artifact_v2",
            }
            for member_nwb in targets
        ]
        # no-op when framework transaction is active; kept defensively
        # so an out-of-populate caller still gets atomic registration.
        with transaction_or_noop(self.connection):
            IntervalList.insert(interval_rows)
            self.insert1(key)

    @staticmethod
    def _scan_artifact_frames(recording, validated, job_kwargs=None):
        """Flag artifact frame indices via a chunked ``ChunkRecordingExecutor``.

        Scans the recording chunk by chunk (defaulting to
        ``chunk_duration='1s'`` when ``job_kwargs`` carries no chunk-size key,
        so the scan is bounded
        for every caller, not only the production path that merges SI's global
        job kwargs) and concatenates the per-chunk flagged frame
        indices into one ascending array. Peak working set is bounded by the
        chunk size -- roughly ``4 × chunk_frames × n_channels × 4 bytes`` (raw
        int slice + float32 µV copy + abs + z-score intermediate) -- rather than
        by the full recording, which the predecessor full-``get_traces`` load
        materialized at ``~4 × n_samples × n_channels × 4 bytes`` (≈27 GB for a
        1-hour 64-channel 30 kHz recording). Restores v1's chunked path
        (``v1/artifact.py:_get_artifact_times`` +
        ``spikesorting/utils.py:_compute_artifact_chunk``) self-contained in v2.

        ``job_kwargs`` is the merged SI job-kwargs blob; only recognized
        ``job_keys`` (``n_jobs``, ``chunk_duration``, ``pool_engine``, ...) are
        forwarded to the executor. ``n_jobs=1`` (the default, matching v1) runs
        serially in-process and passes the live recording to each worker; a
        multi-process pool receives the ``to_dict()`` blob instead.

        Returns the ascending ndarray of flagged frame indices.
        """
        resolved = dict(job_kwargs or {})
        exec_kwargs = {k: resolved[k] for k in job_keys if k in resolved}
        # Chunk BY DEFAULT. ChunkRecordingExecutor with no chunk-size key
        # processes the whole recording in a single chunk -- i.e. the exact
        # full-traces memory profile this restoration removed. A caller that
        # reaches here without a chunk key (any direct ``_detect_artifacts``
        # call, not just the production ``make_compute`` path that merges SI's
        # global ``chunk_duration='1s'``) must still be bounded, so default the
        # chunk to 1 s of data. Callers can override via job_kwargs (and a
        # single-pass-in-memory mode is still reachable with an explicit
        # ``chunk_duration=-1`` / ``chunk_size`` spanning the recording).
        _CHUNK_KEYS = (
            "chunk_size",
            "chunk_duration",
            "chunk_memory",
            "total_memory",
        )
        if not any(k in exec_kwargs for k in _CHUNK_KEYS):
            exec_kwargs["chunk_duration"] = "1s"
        n_jobs = ensure_n_jobs(recording, n_jobs=exec_kwargs.get("n_jobs", 1))
        rec_arg = recording if n_jobs == 1 else recording.to_dict()
        init_args = (
            rec_arg,
            validated.zscore_thresh,
            validated.amplitude_thresh_uV,
            validated.proportion_above_thresh,
        )
        executor = ChunkRecordingExecutor(
            recording=recording,
            func=_compute_artifact_chunk,
            init_func=_init_artifact_worker,
            init_args=init_args,
            handle_returns=True,
            job_name="detect_artifact_frames",
            **exec_kwargs,
        )
        per_chunk = executor.run()
        if not per_chunk:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(per_chunk)

    @staticmethod
    def _detect_artifacts(recording, validated, context="", job_kwargs=None):
        """Run amplitude / z-score artifact scan on a SI recording.

        Returns an ``ndarray`` of shape ``(n_intervals, 2)`` containing
        the artifact-removed valid times in seconds. When ``detect`` is
        False the full recording window is returned untouched.

        The threshold scan runs chunk by chunk via ``_scan_artifact_frames``
        (SpikeInterface's ``ChunkRecordingExecutor``) so peak memory is bounded
        by the chunk size rather than the full recording -- see that method's
        docstring for the per-chunk memory formula and the default chunk size.

        ``job_kwargs`` is the merged per-row / config / global SI job-kwargs
        blob; the caller resolves it via ``_resolved_job_kwargs`` and passes it
        through to the executor. ``context`` is an optional caller-supplied
        string (e.g. ``" for artifact_id=... recording_id=..."``) appended to
        the zero-frames warning so an operator can identify which selection
        scanned empty.
        """
        import numpy as _np

        from spyglass.spikesorting.v2.utils import (
            _base_intervals_from_timestamps,
        )

        timestamps = recording.get_times()
        fs = recording.get_sampling_frequency()
        # Recorded chunks, split at wall-clock discontinuities. For a
        # contiguous recording this is a single ``[t0, t_end]``; for
        # disjoint sort intervals it is one interval per chunk, so the
        # detect=False / zero-artifact returns AND the artifact complement
        # below never span an inter-chunk gap (which would inflate
        # obs_intervals duration and let sub-min_length slivers survive by
        # borrowing gap time). v1 subtracts artifacts from the explicit
        # ``sort_interval_valid_times`` (``v1/artifact.py:327``).
        base_intervals = _base_intervals_from_timestamps(timestamps, fs)
        if not validated.detect:
            logger.info(
                "ArtifactDetection: detect=False; returning the recorded "
                "window(s) as valid intervals."
            )
            return _np.asarray(base_intervals)

        # Log the threshold configuration so a population report can
        # audit which detection mode actually fired per row. Matches
        # v1's logger.info at v1/artifact.py:258-261.
        logger.info(
            "ArtifactDetection: scanning with "
            f"amplitude_thresh_uV={validated.amplitude_thresh_uV}, "
            f"zscore_thresh={validated.zscore_thresh}, "
            f"proportion_above_thresh={validated.proportion_above_thresh}, "
            f"removal_window_ms={validated.removal_window_ms}, "
            f"join_window_ms={validated.join_window_ms}, "
            f"min_length_s={validated.min_length_s}."
        )

        # Chunked scan via ChunkRecordingExecutor (see
        # ``_scan_artifact_frames``). The per-frame detection math --
        # µV scaling with the stored gains, amplitude threshold, and the
        # across-channel (``axis=1``) z-score with its ``+1e-12`` std
        # epsilon -- lives in the module-level ``_compute_artifact_chunk``
        # worker. The z-score is computed on each frame's own columns, so
        # chunk boundaries (which split the time axis) leave the flagged
        # frame set unchanged; this is the equivalence the regression test
        # pins. v1 OR-combined the two detectors at
        # ``spikesorting/utils.py:198``; the worker preserves that (an AND
        # would make the dual-threshold mode strictly less sensitive than
        # either single-threshold mode).
        frames_above = ArtifactDetection._scan_artifact_frames(
            recording, validated, job_kwargs
        )
        if len(frames_above) == 0:
            # Matches v1's warning at v1/artifact.py:318 so a
            # downstream consumer noticing "all valid times" can see
            # whether detection was attempted-and-empty vs skipped.
            logger.warning(
                "ArtifactDetection: scan found zero artifact frames"
                f"{context} (amplitude_thresh_uV="
                f"{validated.amplitude_thresh_uV}, zscore_thresh="
                f"{validated.zscore_thresh}, proportion_above_thresh="
                f"{validated.proportion_above_thresh}); returning the "
                "recorded window(s) as valid intervals."
            )
            return _np.asarray(base_intervals)

        half_window_frames = int(
            _np.ceil(validated.removal_window_ms * 1e-3 * fs / 2)
        )
        join_window_frames = int(
            _np.ceil(validated.join_window_ms * 1e-3 * fs)
        )

        # Inter-chunk wall-clock gaps: frame index of the last sample
        # before each gap (diff > 1.5 sample periods). Frame indices are
        # contiguous across a gap, so BOTH the join below and the
        # removal-window expansion further down must be gap-aware --
        # otherwise an artifact near a chunk edge bridges/spills into the
        # neighboring chunk across the gap.
        n = len(timestamps)
        gap_after = _np.flatnonzero(
            _np.diff(timestamps) > 1.5 * (1.0 / fs)
        )

        # Build artifact intervals in frame indices, then convert to
        # seconds and subtract per base chunk. Join nearby artifact frames
        # into spans, but NEVER across a timestamp gap: two artifacts in
        # different chunks are frame-adjacent yet seconds apart in
        # wall-clock, so joining them would create a span straddling the
        # gap that over-masks the earlier chunk's tail. v1's effective
        # join is in timestamp space (after time-based window expansion),
        # so a large disjoint gap is never bridged.
        spans = []
        cur_start = frames_above[0]
        cur_end = frames_above[0]
        for f in frames_above[1:]:
            crosses_gap = bool(
                _np.any((gap_after >= cur_end) & (gap_after < f))
            )
            if f - cur_end <= join_window_frames and not crosses_gap:
                cur_end = f
            else:
                spans.append((cur_start, cur_end))
                cur_start = f
                cur_end = f
        spans.append((cur_start, cur_end))

        # Cap the removal-window expansion at the CHUNK boundary on each
        # side, for the same reason: a frame-space window (``end_f +
        # half_window_frames``) on an artifact near a chunk edge would
        # reach into the neighboring chunk's first samples across the gap
        # and -- after per-chunk clipping -- remove them. v1's window is
        # time-based (``timestamps[end] + half_win`` seconds), which
        # cannot cross a gap orders of magnitude larger than the window;
        # capping at the chunk frame bounds reproduces that.
        artifact_intervals = []
        for start_f, end_f in spans:
            left = gap_after[gap_after < start_f]
            chunk_start = int(left[-1]) + 1 if left.size else 0
            right = gap_after[gap_after >= end_f]
            chunk_end = int(right[0]) if right.size else n - 1
            start_f = max(chunk_start, start_f - half_window_frames)
            end_f = min(chunk_end, end_f + half_window_frames)
            # ``end_f`` is the INCLUSIVE last artifact sample, but the
            # interval is stored as a half-open ``[start, end)`` where
            # ``end`` is the first non-artifact sample. Otherwise the
            # complement (saved as valid_times) would silently include
            # ``timestamps[end_f]`` -- an artifact sample -- in the
            # next valid interval, and ``Sorting._apply_artifact_mask``
            # would fail to mask that sample before the sort.
            end_time = timestamps[min(end_f + 1, n - 1)]
            artifact_intervals.append([timestamps[start_f], end_time])

        # Subtract artifact intervals from each recorded base chunk
        # SEPARATELY so a kept valid interval never spans an inter-chunk
        # wall-clock gap. Subtracting from one envelope
        # ``[timestamps[0], timestamps[-1]]`` would reintroduce the gaps
        # ``Recording.make`` excluded -- inflating obs_intervals duration
        # and letting a sub-min_length sliver survive by borrowing gap
        # time. ``artifact_intervals`` is start-sorted; clip each to the
        # current chunk and walk the complement within it.
        kept = []
        for base_start, base_end in base_intervals:
            cursor = base_start
            for art_start, art_end in artifact_intervals:
                clipped_start = max(art_start, base_start)
                clipped_end = min(art_end, base_end)
                if clipped_end <= clipped_start:
                    continue  # artifact does not overlap this chunk
                if clipped_start > cursor:
                    kept.append([cursor, clipped_start])
                cursor = max(cursor, clipped_end)
            if cursor < base_end:
                kept.append([cursor, base_end])

        # Drop valid-interval slivers shorter than ``min_length_s``
        # (default 1.0 s) before returning. Without this, a noisy
        # recording with frequent artifacts leaves millisecond-scale
        # slivers between artifact intervals that downstream
        # ``Sorting._apply_artifact_mask`` iterates one by one; SI
        # sorters may also crash on micro-intervals. Matches v1's
        # hardcoded ``min_length=1`` at
        # ``src/spyglass/spikesorting/v1/artifact.py:327-328``.
        if kept:
            kept = [
                [start, end]
                for start, end in kept
                if (end - start) >= validated.min_length_s
            ]
        return _np.asarray(kept) if kept else _np.empty((0, 2))

    def get_artifact_removed_intervals(self, key):
        """Return the artifact-removed ``valid_times`` for ``key``.

        Single-recording source: one ``IntervalList`` row keyed by
        the recording's parent ``nwb_file_name`` -- returned as a
        plain ``(n_intervals, 2)`` ndarray.

        Shared-artifact-group source: ``make_insert`` writes one
        ``IntervalList`` row per distinct member ``nwb_file_name``
        (today single-session, so length 1). Returns a dict
        keyed by ``nwb_file_name`` mapping to the per-member
        ``valid_times`` array. All values are equal across keys
        (the detection ran ONCE over the unioned channels and
        ``make_insert`` wrote the same array per member), so a
        caller that wants a single array can ``next(iter(d.values()))``.
        """
        from spyglass.spikesorting.v2.recording import RecordingSelection
        from spyglass.spikesorting.v2.utils import (
            artifact_interval_list_name,
        )

        source = ArtifactSelection.resolve_source(key)
        if "artifact_id" not in key:
            raise ValueError(
                "ArtifactDetection.get_artifact_removed_intervals: key "
                "must include 'artifact_id'."
            )
        interval_list_name = artifact_interval_list_name(key["artifact_id"])

        if source.kind == "recording":
            nwb_file_name = (
                RecordingSelection
                & {"recording_id": source.key["recording_id"]}
            ).fetch1("nwb_file_name")
            return (
                IntervalList
                & {
                    "nwb_file_name": nwb_file_name,
                    "interval_list_name": interval_list_name,
                }
            ).fetch1("valid_times")

        # shared_artifact_group source: collect the member
        # nwb_file_names through the Member -> RecordingSelection
        # join, fetch each member's IntervalList row, return a
        # name->valid_times dict.
        members = (
            SharedArtifactGroup.Member
            * RecordingSelection
            & {
                "shared_artifact_group_name": source.key[
                    "shared_artifact_group_name"
                ]
            }
        ).fetch("nwb_file_name", as_dict=True)
        member_nwb_files = list(
            dict.fromkeys(m["nwb_file_name"] for m in members)
        )
        rows = (
            IntervalList
            & [
                {
                    "nwb_file_name": nwb,
                    "interval_list_name": interval_list_name,
                }
                for nwb in member_nwb_files
            ]
        ).fetch("nwb_file_name", "valid_times", as_dict=True)
        return {row["nwb_file_name"]: row["valid_times"] for row in rows}

    def delete(self, *args, safemode=None, **kwargs):
        """Override that also removes the matching IntervalList rows.

        DataJoint does not cascade through ``interval_list_name``-keyed
        dependencies, so the cleanup is explicit. We fetch the
        ``artifact_id`` (and, for shared-group sources, the member
        ``nwb_file_name``s) up front, delete the master row(s) via
        ``super().delete``, then drop the matching IntervalList rows.
        """
        from spyglass.spikesorting.v2.exceptions import SchemaBypassError
        from spyglass.spikesorting.v2.recording import RecordingSelection

        from spyglass.spikesorting.v2.utils import (
            artifact_interval_list_name,
        )

        # Collect the IntervalList rows to clean up BEFORE we delete the
        # master rows -- the source-part join no longer resolves after
        # the master is gone.
        interval_rows_to_remove = []
        for row in self.fetch(as_dict=True):
            try:
                source = ArtifactSelection.resolve_source(row)
            except SchemaBypassError as resolve_exc:
                # A genuinely broken source (zero/multiple source part
                # rows -- the only thing ``resolve_source`` raises) cannot
                # be resolved to its IntervalList target. Log, skip ITS
                # paired cleanup, and let the master delete proceed. Any
                # OTHER exception is an unexpected bug: DO NOT swallow it
                # (a broad ``except`` would silently orphan IntervalList
                # rows). Let it propagate to abort the delete BEFORE the
                # master is removed, so the operator sees the failure.
                logger.error(
                    "ArtifactDetection.delete: resolve_source failed for "
                    f"{ {k: row[k] for k in ('artifact_id',)} }; leaving "
                    f"IntervalList rows in place. Underlying error: "
                    f"{resolve_exc!r}"
                )
                continue
            interval_list_name = artifact_interval_list_name(
                row["artifact_id"]
            )
            if source.kind == "recording":
                nwb_file_name = (
                    RecordingSelection
                    & {"recording_id": source.key["recording_id"]}
                ).fetch1("nwb_file_name")
                interval_rows_to_remove.append(
                    {
                        "nwb_file_name": nwb_file_name,
                        "interval_list_name": interval_list_name,
                    }
                )
            elif source.kind == "shared_artifact_group":
                # ``make_insert`` wrote one IntervalList row per
                # distinct member ``nwb_file_name``; collect them
                # all for cleanup. The Member -> RecordingSelection
                # join surfaces the per-member nwb_file_name set.
                members = (
                    SharedArtifactGroup.Member
                    * RecordingSelection
                    & {
                        "shared_artifact_group_name": source.key[
                            "shared_artifact_group_name"
                        ]
                    }
                ).fetch("nwb_file_name", as_dict=True)
                seen = set()
                for m in members:
                    nwb = m["nwb_file_name"]
                    if nwb in seen:
                        continue
                    seen.add(nwb)
                    interval_rows_to_remove.append(
                        {
                            "nwb_file_name": nwb,
                            "interval_list_name": interval_list_name,
                        }
                    )

        if safemode is None:
            super().delete(*args, **kwargs)
        else:
            super().delete(*args, safemode=safemode, **kwargs)

        # Clean up the matching IntervalList rows through cautious_delete
        # (the user already passed the same team-permission check on the
        # parent ArtifactDetection rows above keyed by the same
        # nwb_file_name, so the IntervalList check is a re-verification,
        # not a bypass). super_delete here would silently delete other
        # users' rows under shared lab sessions.
        for restriction in interval_rows_to_remove:
            rows = IntervalList & restriction
            if len(rows) == 0:
                continue
            rows.delete(safemode=False)

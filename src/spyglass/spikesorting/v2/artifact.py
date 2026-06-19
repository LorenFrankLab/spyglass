"""Artifact detection over a preprocessed Recording.

Tables (all final-shape under the zero-migration policy):
    ArtifactDetectionParameters       -- threshold detection parameters.
    SharedArtifactGroup (+ Member)    -- opt-in cross-recording detection.
    ArtifactDetectionSelection        -- source parts encode input shape.
        .RecordingSource              -- single-recording path (default).
        .SharedGroupSource          -- cross-recording path (#928).
    ArtifactDetection                 -- writes IntervalList rows; no part.

Artifact-removed valid times live in ``common.IntervalList`` rather than
a dedicated part table -- the UUID-suffixed name prevents collision with
human-authored session intervals while letting downstream
IntervalList-querying code consume them through the standard interface.

``ArtifactDetectionParameters.insert1`` Pydantic-validates the
``params`` blob. ``insert_selection`` resolves a selection to a single
``artifact_detection_id``, ``make`` runs threshold detection and
writes the artifact-removed ``IntervalList`` rows,
``get_artifact_removed_intervals`` reads them back, ``delete`` removes
them, and ``SharedArtifactGroup.insert_group`` declares a cross-recording
detection bundle.
"""

from __future__ import annotations

from typing import NamedTuple

import datajoint as dj
import numpy as np

from spyglass.common import IntervalList, Session  # noqa: F401

# Pure-compute worker kernels live in a DB-free module so that a spawned
# ``n_jobs>1`` artifact-detection worker (macOS ``spawn`` re-imports the
# function's defining module) does NOT open a DB connection at import. Re-export
# here so existing ``from ...v2.artifact import _compute_artifact_chunk`` call
# sites keep working. See ``_artifact_compute`` for the rationale.
from spyglass.spikesorting.v2._artifact_compute import (  # noqa: F401
    _compute_artifact_chunk,
    _init_artifact_worker,
)

# Artifact-removed interval construction + IntervalList persistence live in a
# DB-free service module so ``ArtifactDetection`` stays a thin orchestrator.
# The class keeps thin delegators where tests pin the surface
# (``_detect_artifacts`` / ``_scan_artifact_frames`` are called directly on the
# class, and ``get_artifact_removed_intervals`` is called on instances).
from spyglass.spikesorting.v2._artifact_intervals import (
    build_artifact_interval_rows,
    collect_artifact_interval_rows_to_remove,
    detect_artifacts,
    read_artifact_removed_intervals,
    remove_artifact_interval_rows,
    scan_artifact_frames,
)
from spyglass.spikesorting.v2._params.artifact_detection import (
    ArtifactDetectionParamsSchema,
)
from spyglass.spikesorting.v2.recording import Recording
from spyglass.spikesorting.v2.utils import (
    SelectionMasterInsertGuard,
    SourceResolution,
    _assert_v2_db_safe,
    _get_recording_timestamps,
    _validate_params,
    find_orphaned_masters,
    reject_duplicate_parameter_content,
    transaction_or_noop,
    validate_lookup_rows,
)
from spyglass.utils import SpyglassMixin, SpyglassMixinPart, logger

_assert_v2_db_safe()
schema = dj.schema("spikesorting_v2_artifact")


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
    artifact_detection_params_name: varchar(64)
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
                detect=False, amplitude_threshold_uv=None
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
        (
            # Production artifact recipe (June 2026): 100 uV amplitude
            # threshold, 0.7 proportion-above-threshold, 1.0 ms removal window
            # -- far more aggressive than the 500 uV shipped "default".
            "franklab_100uv_p07_2026_06",
            ArtifactDetectionParamsSchema(
                amplitude_threshold_uv=100.0,
                proportion_above_threshold=0.7,
            ).model_dump(),
            2,
            None,
        ),
        (
            # More aggressive 50 uV production variant (same proportion-above
            # and removal window).
            "franklab_50uv_p07_2026_06",
            ArtifactDetectionParamsSchema(
                amplitude_threshold_uv=50.0,
                proportion_above_threshold=0.7,
            ).model_dump(),
            2,
            None,
        ),
    )

    def insert1(self, row, allow_duplicate_params=False, **kwargs):
        """Insert one validated artifact-detection parameter row."""
        # Delegate to ``insert`` so one validated path serves both.
        self.insert(
            [row], allow_duplicate_params=allow_duplicate_params, **kwargs
        )

    def insert(self, rows, allow_duplicate_params=False, **kwargs):
        """Insert artifact-detection parameter rows after validation.

        ``allow_duplicate_params=True`` opts out of the duplicate-content
        guard (a second name for an existing blob); see
        ``reject_duplicate_parameter_content``.
        """
        # Validate every row (incl. ``insert_default``'s positional
        # ``_DEFAULT_CONTENTS``) so a bulk insert can't bypass schema
        # validation or the params_schema_version drift check.
        validated = validate_lookup_rows(
            rows,
            self.heading.names,
            schema_for=lambda _row: ArtifactDetectionParamsSchema,
            table_name="ArtifactDetectionParameters",
        )
        reject_duplicate_parameter_content(
            self,
            validated,
            table_name="ArtifactDetectionParameters",
            name_attr="artifact_detection_params_name",
            allow_duplicate_params=allow_duplicate_params,
        )
        super().insert(validated, **kwargs)

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
        """One member Recording of a shared artifact-detection group."""

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
        # preprocessing_params_name, team_name); same NWB does NOT imply
        # same time axis. ``si.aggregate_channels`` requires every
        # member to share ``n_samples``, ``sampling_frequency``, and
        # dtype, otherwise the union construction crashes deep inside
        # SI with an opaque "shape mismatch" error at populate time.
        # Catch the invariant at insert time so the user gets a
        # clear diagnostic before populate.
        per_member_recording_rows = (
            Recording & [{"recording_id": rid} for rid in member_recording_ids]
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

        sessions = {
            sel_by_id[str(rid)]["nwb_file_name"] for rid in member_recording_ids
        }
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
        # requires identical fs. A typo in upstream preprocessing params
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
class ArtifactDetectionSelection(
    SelectionMasterInsertGuard, SpyglassMixin, dj.Manual
):
    """One row per (parameters, source) artifact detection request.

    Source part rows make the input shape explicit: exactly one of
    ``RecordingSource`` (single-recording, default) or
    ``SharedGroupSource`` (cross-recording, opt-in) must exist
    for each selection row. Enforced by ``insert_selection`` and
    re-checked at the start of ``ArtifactDetection.make()`` so a row
    inserted via raw ``insert1`` (bypassing ``insert_selection``) is
    caught before populate proceeds.
    """

    definition = """
    artifact_detection_id: uuid
    ---
    -> ArtifactDetectionParameters
    """

    class RecordingSource(SpyglassMixinPart):
        """Single-recording source for an artifact-detection selection."""

        definition = """
        -> master
        ---
        -> Recording
        """

    class SharedGroupSource(SpyglassMixinPart):
        """Shared-group source for an artifact-detection selection."""

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
        that matches both the master fields and the source row, or, if
        absent, inserts master + source part keyed by a deterministic,
        content-addressed ``artifact_detection_id`` derived from the params + source
        identity. A concurrent caller that wins the duplicate-PK race is
        handled by refetching the winner's row.

        The find step joins the master to the chosen source part so a
        prior ``ArtifactDetectionSelection`` with a different source is not
        silently reused. Source-part atomicity is enforced via the
        ``SchemaBypassError`` re-check at the top of
        ``ArtifactDetection.make()`` (Layer 2 of the source-part
        pattern).

        Raises
        ------
        ValueError
            If zero or two source keys are supplied in ``key``.
        DuplicateSelectionError
            If any matching master has a non-deterministic ``artifact_detection_id``
            (a raw ``insert`` bypass or a pre-determinism legacy row) --
            even a single one; an integrity bug, not user error.
        SchemaBypassError
            If a deterministic master exists but its expected source part
            is missing/mismatched (a raw-insert orphan).
        """
        has_recording = "recording_id" in key
        has_shared = "shared_artifact_group_name" in key
        if has_recording == has_shared:
            raise ValueError(
                "ArtifactDetectionSelection.insert_selection requires exactly one "
                "source key. Provide either recording_id (single-recording "
                "path) or shared_artifact_group_name (cross-recording "
                "path), not both and not neither. Got: "
                f"recording_id={'set' if has_recording else 'unset'}, "
                f"shared_artifact_group_name="
                f"{'set' if has_shared else 'unset'}."
            )

        master_field = "artifact_detection_params_name"
        if master_field not in key:
            raise ValueError(
                "ArtifactDetectionSelection.insert_selection requires "
                f"{master_field!r} in key."
            )
        master_restriction = {master_field: key[master_field]}

        if has_recording:
            source_part = cls.RecordingSource
            source_restriction = {"recording_id": key["recording_id"]}
        else:
            source_part = cls.SharedGroupSource
            source_restriction = {
                "shared_artifact_group_name": key["shared_artifact_group_name"]
            }

        # Deterministic, content-addressed artifact_detection_id from the logical
        # identity (params + source), built via the shared payload helper so
        # preflight derives the identical id. ``source_kind`` is explicit so a
        # recording source and a shared-group source never alias even if their
        # source-identifier strings were to collide.
        from spyglass.spikesorting.v2._selection_identity import (
            artifact_detection_identity_payload,
            assert_supplied_id_matches,
            deterministic_id,
        )

        identity = artifact_detection_identity_payload(
            artifact_detection_params_name=key[master_field],
            recording_id=key.get("recording_id"),
            shared_artifact_group_name=key.get("shared_artifact_group_name"),
        )
        artifact_detection_id = deterministic_id("artifact_detection", identity)
        assert_supplied_id_matches(
            key.get("artifact_detection_id"),
            artifact_detection_id,
            field="artifact_detection_id",
        )

        existing = cls._find_existing_pk(
            master_restriction,
            source_part,
            source_restriction,
            artifact_detection_id,
        )
        if existing is not None:
            return existing

        # Translate the would-be DataJoint FK IntegrityError into a
        # clear "missing default row" message before the inserts attempt.
        from spyglass.spikesorting.v2.exceptions import SchemaBypassError
        from spyglass.spikesorting.v2.utils import (
            _ensure_lookup_row_exists,
            _is_duplicate_key_error,
        )

        _ensure_lookup_row_exists(
            ArtifactDetectionParameters,
            master_restriction,
            helper_name="ArtifactDetectionSelection.insert_selection",
            insert_default_path="ArtifactDetectionParameters.insert_default()",
        )

        new_master_key = {
            **master_restriction,
            "artifact_detection_id": artifact_detection_id,
        }
        new_part_key = {
            "artifact_detection_id": artifact_detection_id,
            **source_restriction,
        }
        try:
            with transaction_or_noop(cls.connection):
                # allow_direct_insert: this helper IS the validation boundary.
                cls.insert1(new_master_key, allow_direct_insert=True)
                source_part.insert1(new_part_key)
        except Exception as exc:  # noqa: BLE001 -- re-raised unless dup-PK
            if not _is_duplicate_key_error(exc):
                raise
            # Lost a concurrent race on the same deterministic artifact_detection_id;
            # refetch and return the winner's master+source row.
            # (Top-level recovery only -- see transaction_or_noop.)
            logger.debug(
                "ArtifactDetectionSelection.insert_selection: lost deterministic-id "
                "race on %s; returning the existing row.",
                artifact_detection_id,
            )
            existing = cls._find_existing_pk(
                master_restriction,
                source_part,
                source_restriction,
                artifact_detection_id,
            )
            if existing is not None:
                return existing
            # The deterministic master exists (that is the duplicate key)
            # but the expected source part is missing/mismatched -- a
            # raw-insert orphan. Surface a clear schema-bypass error
            # instead of the opaque duplicate-key error.
            raise SchemaBypassError(
                f"ArtifactDetectionSelection master {artifact_detection_id} exists but has no "
                f"matching {source_part.__name__} row for "
                f"{source_restriction}: the master was inserted without "
                "insert_selection (raw-insert orphan). Use insert_selection() "
                "to create master+source atomically, or drop the orphan "
                "master."
            ) from exc
        return {k: new_master_key[k] for k in cls.primary_key}

    @classmethod
    def _find_existing_pk(
        cls,
        master_restriction,
        source_part,
        source_restriction,
        deterministic_id,
    ) -> dict | None:
        """Return the canonical master PK for this master+source, or None.

        Joins the master to the chosen source part (so a prior selection
        with a DIFFERENT source is not reused) and splits the matches by
        primary key:

        * the master at ``deterministic_id`` is the canonical, content-
          addressed selection -> return ``{"artifact_detection_id": ...}``;
        * ANY master with a different ``artifact_detection_id`` is non-deterministic
          (a raw ``insert`` bypass or pre-determinism legacy row) and
          violates the content-addressed-identity invariant -> raise
          ``DuplicateSelectionError`` so it is reset rather than silently
          returned.

        Used by ``insert_selection`` for both the pre-insert lookup and
        the post-duplicate-key refetch.
        """
        from spyglass.spikesorting.v2.exceptions import (
            DuplicateSelectionError,
        )

        joined = (cls * source_part) & master_restriction & source_restriction
        master_ids = {
            row["artifact_detection_id"]
            for row in joined.fetch("KEY", as_dict=True)
        }
        bypassed = [aid for aid in master_ids if aid != deterministic_id]
        if bypassed:
            raise DuplicateSelectionError(
                f"ArtifactDetectionSelection has {len(master_ids)} master rows for "
                f"{master_restriction | source_restriction} whose artifact_detection_id "
                f"is not the deterministic id {deterministic_id}: {bypassed}. "
                "This is a non-deterministic selection row (a raw insert or "
                "pre-determinism legacy row); drop it and re-insert via "
                "insert_selection."
            )
        return (
            {"artifact_detection_id": deterministic_id} if master_ids else None
        )

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
            Orphan master PKs (``{"artifact_detection_id": ...}``). Empty when
            no orphans remain.
        """
        orphans = find_orphaned_masters(
            cls,
            [cls.RecordingSource, cls.SharedGroupSource],
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
        shared_rows = (cls.SharedGroupSource & master_key).fetch(as_dict=True)
        total = len(rec_rows) + len(shared_rows)
        if total != 1:
            raise SchemaBypassError(
                f"ArtifactDetectionSelection {master_key} has {total} source part "
                "rows; expected exactly one. Use "
                "ArtifactDetectionSelection.insert_selection() to add or remove "
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
    ``f"artifact_detection_{artifact_detection_id}"`` at the end of ``make()`` -- one row per
    affected session. Single-recording detections write exactly one row;
    cross-recording detections write one row per distinct member
    ``nwb_file_name``.
    """

    definition = """
    -> ArtifactDetectionSelection
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

        Parameters
        ----------
        key : dict
            Primary-key restriction selecting one
            ``ArtifactDetectionSelection`` row to populate.

        Returns
        -------
        ArtifactFetched
            The resolved source, validated params, parent
            ``nwb_file_name``, per-row job kwargs, and -- for the
            shared-group source -- the ordered member
            ``recording_id`` / ``nwb_file_name`` tuples.
        """
        from spyglass.spikesorting.v2.recording import RecordingSelection

        source = ArtifactDetectionSelection.resolve_source(key)

        params_blob, artifact_job_kwargs = (
            ArtifactDetectionParameters * (ArtifactDetectionSelection & key)
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
                SharedArtifactGroup.Member * RecordingSelection
                & {
                    "shared_artifact_group_name": source.key[
                        "shared_artifact_group_name"
                    ]
                }
            ).fetch(
                "recording_id",
                "nwb_file_name",
                as_dict=True,
                order_by="recording_id",
            )
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
            member_nwb_file_names = tuple(m["nwb_file_name"] for m in members)
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

        Parameters
        ----------
        key : dict
            Primary-key restriction for the populating row.
        source : SourceResolution
            Resolved source record (``recording`` or
            ``shared_artifact_group`` kind) from ``make_fetch``.
        validated : ArtifactDetectionParamsSchema
            Validated detection parameters.
        nwb_file_name : str
            Parent session for the single-recording path; the
            canonical member session for the shared-group path.
        artifact_job_kwargs : dict or None
            Per-row SI job-kwargs blob, merged over the global and
            config defaults before the chunked scan.
        member_recording_ids : tuple, optional
            Ordered member ``recording_id`` s for the shared-group
            source; ``None`` for the single-recording source.
        member_nwb_file_names : tuple, optional
            Ordered member ``nwb_file_name`` s for the shared-group
            source; ``None`` for the single-recording source.

        Returns
        -------
        ArtifactComputed
            The artifact-removed ``valid_times`` plus the
            per-member ``nwb_file_name`` target list that
            ``make_insert`` writes one ``IntervalList`` row per.
        """
        # Merge the SI-global, DataJoint-config, and per-row job_kwargs so
        # the chunked ``_scan_artifact_frames`` pass honors ``n_jobs`` /
        # ``chunk_duration`` overrides. The stored per-row ``job_kwargs``
        # blob is honored here, merged over the global and config defaults.
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
                    f" for artifact_detection_id={key['artifact_detection_id']}, "
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
                f" for artifact_detection_id={key['artifact_detection_id']}, "
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
        master's ``nwb_file_name``. The IntervalList row contents are
        built by :func:`._artifact_intervals.build_artifact_interval_rows`;
        the transaction + master ``self.insert1`` stay here because they
        are the table's Computed-make contract.

        Parameters
        ----------
        key : dict
            Primary-key restriction for the row being inserted.
        valid_times : np.ndarray
            Artifact-removed valid times, shape ``(n_intervals, 2)``.
        nwb_file_name : str
            Parent session used as the fallback IntervalList target
            when ``per_member_nwb_files`` is empty.
        per_member_nwb_files : tuple, optional
            Distinct member ``nwb_file_name`` s to write one
            ``IntervalList`` row each. Default ``()`` falls back to
            ``(nwb_file_name,)``.
        """
        from spyglass.spikesorting.v2.utils import transaction_or_noop

        interval_rows = build_artifact_interval_rows(
            key, valid_times, nwb_file_name, per_member_nwb_files
        )
        # no-op when framework transaction is active; kept defensively
        # so an out-of-populate caller still gets atomic registration.
        with transaction_or_noop(self.connection):
            IntervalList.insert(interval_rows)
            self.insert1(key)

    @staticmethod
    def _scan_artifact_frames(recording, validated, job_kwargs=None):
        """Flag artifact frame indices via a chunked ``ChunkRecordingExecutor``.

        Thin delegator to
        :func:`._artifact_intervals.scan_artifact_frames`; kept as an
        ``ArtifactDetection`` staticmethod because ``test_audit_parity``
        calls ``ArtifactDetection._scan_artifact_frames(...)`` directly.
        The chunked-scan memory contract (default 1 s chunk,
        ``ChunkRecordingExecutor`` over the ``_artifact_compute`` kernels)
        lives in the service module.
        """
        return scan_artifact_frames(recording, validated, job_kwargs)

    @staticmethod
    def _detect_artifacts(recording, validated, context="", job_kwargs=None):
        """Run amplitude / z-score artifact scan on a SI recording.

        Thin delegator to
        :func:`._artifact_intervals.detect_artifacts`; kept as an
        ``ArtifactDetection`` staticmethod because ``make_compute`` calls
        ``self._detect_artifacts(...)`` and the v2 tests call
        ``ArtifactDetection._detect_artifacts(...)`` directly. The chunked
        scan, gap-aware span join, removal-window clamp, per-chunk
        subtraction, and min-length sliver filter live in the service
        module.
        """
        return detect_artifacts(
            recording, validated, context=context, job_kwargs=job_kwargs
        )

    def get_artifact_removed_intervals(
        self, key: dict, as_dict: bool = False
    ) -> "np.ndarray | dict[str, np.ndarray]":
        """Return the artifact-removed ``valid_times`` for ``key``.

        Single-recording source: one ``IntervalList`` row keyed by
        the recording's parent ``nwb_file_name`` -- returned as a
        plain ``(n_intervals, 2)`` ndarray (or, with ``as_dict=True``,
        a one-key ``{nwb_file_name: ndarray}`` dict).

        Shared-artifact-group source: ``make_insert`` writes one
        ``IntervalList`` row per distinct member ``nwb_file_name``
        (today single-session, so length 1). Returns a dict
        keyed by ``nwb_file_name`` mapping to the per-member
        ``valid_times`` array. All values are equal across keys
        (the detection ran ONCE over the unioned channels and
        ``make_insert`` wrote the same array per member), so a
        caller that wants a single array can ``next(iter(d.values()))``.

        Parameters
        ----------
        key : dict
            Restriction selecting a single ``ArtifactDetection`` row;
            must include ``artifact_detection_id``.
        as_dict : bool, optional
            If ``False`` (default), the return type depends on the
            source: a plain ``(n_intervals, 2)`` ndarray for a
            single-recording source, a ``{nwb_file_name: ndarray}`` dict
            for a shared-artifact-group source. If ``True``, BOTH sources
            return the dict shape (a single-recording result is wrapped
            as a one-key dict), so source-agnostic callers can avoid
            branching on the return type.

        Returns
        -------
        np.ndarray or dict[str, np.ndarray]
            For a single-recording source with ``as_dict=False``, the
            ``(n_intervals, 2)`` artifact-removed ``valid_times`` array.
            For a shared-artifact-group source, or any source with
            ``as_dict=True``, a dict mapping each member ``nwb_file_name``
            to its ``(n_intervals, 2)`` array.

        Thin delegator to
        :func:`._artifact_intervals.read_artifact_removed_intervals`; kept
        as an ``ArtifactDetection`` method because the v2 tests call
        ``ArtifactDetection().get_artifact_removed_intervals(...)`` on
        instances. Source resolution, the per-session ``IntervalList``
        reads, and the missing-member loudness live in the service module.
        """
        return read_artifact_removed_intervals(key, as_dict=as_dict)

    def delete(self, *args, safemode=None, **kwargs):
        """Override that also removes the matching IntervalList rows.

        DataJoint does not cascade through ``interval_list_name``-keyed
        dependencies, so the cleanup is explicit. We collect the artifact
        ``IntervalList`` rows up front (via
        :func:`._artifact_intervals.collect_artifact_interval_rows_to_remove`,
        BEFORE the master delete, while the source-part join still
        resolves), delete the master row(s) via ``super().delete``, then --
        only if the cascade actually removed the masters -- drop the
        matching IntervalList rows via
        :func:`._artifact_intervals.remove_artifact_interval_rows`.
        """
        # Collect the IntervalList rows to clean up BEFORE we delete the
        # master rows -- the source-part join no longer resolves after
        # the master is gone.
        interval_rows_to_remove = collect_artifact_interval_rows_to_remove(
            self.fetch(as_dict=True)
        )

        if safemode is None:
            super().delete(*args, **kwargs)
        else:
            super().delete(*args, safemode=safemode, **kwargs)

        # If the cascade was cancelled (user answered "no" to the prompt) or
        # the restriction matched nothing, the master rows remain in place --
        # the delete is atomic, so a single check suffices. Do NOT orphan the
        # artifact IntervalList rows for masters the user chose to keep.
        if len(self) > 0:
            return

        remove_artifact_interval_rows(interval_rows_to_remove)

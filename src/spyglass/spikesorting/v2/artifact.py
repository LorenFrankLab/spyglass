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

from spyglass.common import IntervalList, Session  # noqa: F401
from spyglass.spikesorting.v2._params.artifact_detection import (
    ArtifactDetectionParamsSchema,
)
from spyglass.spikesorting.v2.recording import Recording
from spyglass.spikesorting.v2.utils import (
    SourceResolution,
    _assert_schema_version_matches,
    _assert_v2_db_safe,
    _validate_params,
    find_orphaned_masters,
    transaction_or_noop,
)
from spyglass.utils import SpyglassMixin, SpyglassMixinPart, logger

_assert_v2_db_safe()
schema = dj.schema("spikesorting_v2_artifact")


class ArtifactFetched(NamedTuple):
    """DB-side inputs gathered by :meth:`ArtifactDetection.make_fetch`."""

    source: SourceResolution
    validated: ArtifactDetectionParamsSchema
    nwb_file_name: str
    artifact_job_kwargs: dict | None


class ArtifactComputed(NamedTuple):
    """Outputs of :meth:`ArtifactDetection.make_compute`."""

    valid_times: np.ndarray
    nwb_file_name: str


@schema
class ArtifactDetectionParameters(SpyglassMixin, dj.Lookup):
    """Validated artifact-detection parameter blob.

    The ``params`` blob is validated by
    :class:`ArtifactDetectionParamsSchema`. ``insert_default`` ships two
    presets: ``"none"`` (detect=False, skip artifact scanning) and
    ``"default"`` (amplitude threshold + proportion-above-thresh).
    """

    definition = """
    artifact_params_name: varchar(64)
    ---
    params: blob
    params_schema_version=2: int
    job_kwargs=null: blob
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

        Currently gated: ``ArtifactDetection.make`` does not yet
        implement the matching make-body branch for the
        ``SharedArtifactGroupSource`` part, so inserting a group here
        would create rows that cannot populate downstream. The schema
        is declared (final-shape) so v2 won't need a migration when
        the shared-group path lands; ``insert_group`` raises
        ``NotImplementedError`` until then so users get a clear
        message instead of an opaque ``ArtifactDetection.make``
        failure later.

        Parameters
        ----------
        name
            Group name (PK on the master). Must be unique within the
            installation.
        members
            List of dicts identifying member recordings. Each dict must
            contain at least ``recording_id`` (other fields are ignored
            so the caller can pass arbitrary upstream rows / PKs).

        Raises
        ------
        NotImplementedError
            Always, until ``ArtifactDetection.make`` ships the
            shared-group branch. The validation logic below is
            preserved (under ``# pragma: no cover``) so it can be
            re-enabled then.
        ValueError
            If ``members`` is empty, if any member ``recording_id`` is
            not a populated ``Recording``, or if members span more than
            one session. The shared-group detection assumes all members
            share a time axis -- mixing sessions makes the
            artifact-removed valid times undefined.
        """
        raise NotImplementedError(
            "SharedArtifactGroup.insert_group is gated until "
            "ArtifactDetection.make ships the shared-group branch. "
            "Use the single-recording artifact path "
            "(ArtifactSelection.insert_selection with recording_id) "
            "until then."
        )

        from spyglass.spikesorting.v2.recording import (  # pragma: no cover
            Recording,
            RecordingSelection,
        )

        if not members:  # pragma: no cover
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

        sessions = list(
            {
                (
                    RecordingSelection & {"recording_id": rid}
                ).fetch1("nwb_file_name")
                for rid in member_recording_ids
            }
        )
        if len(sessions) != 1:
            raise ValueError(
                "SharedArtifactGroup.insert_group: members span "
                f"{len(sessions)} sessions ({sorted(sessions)}); a shared "
                "artifact-detection pass only makes sense within one "
                "session because the detection writes IntervalList rows "
                "keyed by (nwb_file_name, interval_list_name)."
            )
        (nwb_file_name,) = sessions

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
        The shared-artifact-group source path is still gated by
        ``NotImplementedError`` -- the runtime body lands once
        ``SharedArtifactGroup.insert_group`` ships. Tuple return is
        DeepHash-stable across the two calls DataJoint makes (before
        compute and again inside the transaction).
        """
        from spyglass.spikesorting.v2.recording import RecordingSelection

        source = ArtifactSelection.resolve_source(key)
        if source.kind == "shared_artifact_group":
            raise NotImplementedError(
                "ArtifactDetection.make for SharedArtifactGroup is not yet "
                "implemented; populate SharedArtifactGroup.insert_group "
                "first."
            )
        if source.kind != "recording":
            raise RuntimeError(
                f"ArtifactDetection.make: unexpected source kind "
                f"{source.kind!r}."
            )

        # Pre-fetch both the params blob and the job_kwargs blob so
        # ``make_compute`` can call ``_resolved_job_kwargs(...)`` per
        # shared-contracts.md "Job-Kwargs Resolution". The in-memory
        # ``_detect_artifacts`` scan does not consume SI-style
        # job_kwargs, but the resolver call is required so the
        # override channels are testable here. ``fetch1(a, b)``
        # returns a tuple; unpack directly.
        params_blob, artifact_job_kwargs = (
            ArtifactDetectionParameters * (ArtifactSelection & key)
        ).fetch1("params", "job_kwargs")
        validated = ArtifactDetectionParamsSchema.model_validate(params_blob)
        nwb_file_name = (
            RecordingSelection & {"recording_id": source.key["recording_id"]}
        ).fetch1("nwb_file_name")
        return ArtifactFetched(
            source=source,
            validated=validated,
            nwb_file_name=nwb_file_name,
            artifact_job_kwargs=artifact_job_kwargs,
        )

    def make_compute(
        self, key, source, validated, nwb_file_name, artifact_job_kwargs
    ):
        """Run artifact detection outside any DB transaction.

        Loads the cached preprocessed recording via
        ``Recording.get_recording`` (regenerates the artifact on
        disk if missing -- still inside ``make_compute``, no DB
        lock), scans for amplitude / z-score threshold crossings,
        expands them by the configured removal window. Returns the
        artifact-removed ``valid_times`` plus the upstream metadata
        ``make_insert`` needs.
        """
        # Resolve job_kwargs even though the in-memory
        # ``_detect_artifacts`` scan does not consume them. The
        # resolver call exercises the DataJoint-config and per-row
        # override channels so tests can monkey-patch
        # ``_resolved_job_kwargs`` and confirm the artifact compute
        # stage was wired through. See the Recording.make_compute
        # comment for the same pattern.
        from spyglass.spikesorting.v2.utils import _resolved_job_kwargs

        _resolved_job_kwargs(artifact_job_kwargs)

        recording = Recording().get_recording(
            {"recording_id": source.key["recording_id"]}
        )
        valid_times = self._detect_artifacts(recording, validated)
        return ArtifactComputed(
            valid_times=valid_times, nwb_file_name=nwb_file_name
        )

    def make_insert(self, key, valid_times, nwb_file_name):
        """Write the artifact ``IntervalList`` + master row atomically.

        DataJoint's tri-part dispatch opens the framework transaction
        around this method; the inner ``transaction_or_noop`` is a
        no-op (yields without re-opening when a transaction is
        already active). Kept defensively so an out-of-populate
        caller still gets atomic registration.
        """
        from spyglass.spikesorting.v2.utils import transaction_or_noop

        interval_list_name = f"artifact_{key['artifact_id']}"
        # no-op when framework transaction is active; kept defensively
        # so an out-of-populate caller still gets atomic registration.
        with transaction_or_noop(self.connection):
            IntervalList.insert1(
                {
                    "nwb_file_name": nwb_file_name,
                    "interval_list_name": interval_list_name,
                    "valid_times": valid_times,
                    "pipeline": "spikesorting_artifact_v2",
                }
            )
            self.insert1(key)

    @staticmethod
    def _detect_artifacts(recording, validated):
        """Run amplitude / z-score artifact scan on a SI recording.

        Returns an ``ndarray`` of shape ``(n_intervals, 2)`` containing
        the artifact-removed valid times in seconds. When ``detect`` is
        False the full recording window is returned untouched.
        """
        import numpy as _np

        timestamps = recording.get_times()
        if not validated.detect:
            logger.info(
                "ArtifactDetection: detect=False; returning full "
                "recording window as a single valid interval."
            )
            return _np.asarray([[timestamps[0], timestamps[-1]]])

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

        # In-memory scan: acceptable for recordings up to a few minutes;
        # the smoke / 60s polymer fixtures fit easily. Chunked iteration
        # is follow-up work alongside the recompute pipeline. SI 0.104
        # renamed the ``return_scaled`` kwarg to ``return_in_uV``.
        traces = recording.get_traces(return_in_uV=False)
        # Scale to microvolts using SI's stored gain so the threshold
        # comparison is in physical units.
        gains = recording.get_channel_gains()
        traces_uv = traces.astype(_np.float32) * gains[None, :]
        absolute = _np.abs(traces_uv)

        n_channels = traces.shape[1]
        n_required = int(
            _np.ceil(validated.proportion_above_thresh * n_channels)
        )

        if validated.amplitude_thresh_uV is not None:
            above_amp = absolute > validated.amplitude_thresh_uV
        else:
            above_amp = _np.zeros_like(absolute, dtype=bool)
        if validated.zscore_thresh is not None:
            # Z-score ACROSS channels per frame (not per channel over
            # time). Detects common-mode artifact events (chewing,
            # head movement, behavioral artifacts that hit all
            # channels at once); per-channel-over-time z-score
            # detects per-channel baseline drift, which is what an
            # ``axis=0`` z-score would have computed.
            # Matches v1's ``stats.zscore(traces, axis=1)`` at
            # ``v1/utils.py:185,193``: ``axis=1`` is the channels
            # axis (rows are time frames), so the same z-score value
            # is computed for every channel within a single frame
            # using THAT FRAME's across-channel statistics. A
            # 50x common-mode spike on all channels surfaces; a
            # per-channel baseline shift does not.
            ch_mean = traces_uv.mean(axis=1, keepdims=True)
            ch_std = traces_uv.std(axis=1, keepdims=True) + 1e-12
            zscores = _np.abs((traces_uv - ch_mean) / ch_std)
            above_z = zscores > validated.zscore_thresh
        else:
            above_z = _np.zeros_like(absolute, dtype=bool)

        # OR combine logic (belt-and-suspenders): flag a frame if
        # EITHER detector trips. v1 at ``v1/utils.py:198`` uses
        # ``np.logical_or(above_z, above_a)``; an AND would make the
        # dual-threshold mode strictly less sensitive than either
        # single-threshold mode -- the opposite of the v1 intent.
        # The single-threshold paths (only one of amplitude / zscore
        # set) are unchanged because ``zeros | above_x == above_x``.
        if (
            validated.amplitude_thresh_uV is not None
            and validated.zscore_thresh is not None
        ):
            channel_hit = above_amp | above_z
        elif validated.amplitude_thresh_uV is not None:
            channel_hit = above_amp
        else:
            channel_hit = above_z

        frames_above = (channel_hit.sum(axis=1) >= n_required).nonzero()[0]
        if len(frames_above) == 0:
            # Matches v1's warning at v1/artifact.py:318 so a
            # downstream consumer noticing "all valid times" can see
            # whether detection was attempted-and-empty vs skipped.
            logger.warning(
                "ArtifactDetection: scan found zero artifact frames; "
                "returning the full recording window as a single "
                "valid interval."
            )
            return _np.asarray([[timestamps[0], timestamps[-1]]])

        fs = recording.get_sampling_frequency()
        half_window_frames = int(
            _np.ceil(validated.removal_window_ms * 1e-3 * fs / 2)
        )
        join_window_frames = int(
            _np.ceil(validated.join_window_ms * 1e-3 * fs)
        )

        # Build artifact intervals in frame indices, then convert to
        # seconds and subtract from the recording window.
        spans = []
        cur_start = frames_above[0]
        cur_end = frames_above[0]
        for f in frames_above[1:]:
            if f - cur_end <= join_window_frames:
                cur_end = f
            else:
                spans.append((cur_start, cur_end))
                cur_start = f
                cur_end = f
        spans.append((cur_start, cur_end))

        artifact_intervals = []
        for start_f, end_f in spans:
            start_f = max(0, start_f - half_window_frames)
            end_f = min(len(timestamps) - 1, end_f + half_window_frames)
            # ``end_f`` is the INCLUSIVE last artifact sample, but the
            # interval is stored as a half-open ``[start, end)`` where
            # ``end`` is the first non-artifact sample. Otherwise the
            # complement (saved as valid_times) would silently include
            # ``timestamps[end_f]`` -- an artifact sample -- in the
            # next valid interval, and ``Sorting._apply_artifact_mask``
            # would fail to mask that sample before the sort.
            end_time = timestamps[min(end_f + 1, len(timestamps) - 1)]
            artifact_intervals.append(
                [timestamps[start_f], end_time]
            )

        # Subtract artifact intervals from the recording window.
        valid_start = timestamps[0]
        valid_end = timestamps[-1]
        kept = []
        cursor = valid_start
        for art_start, art_end in artifact_intervals:
            if art_start > cursor:
                kept.append([cursor, art_start])
            cursor = max(cursor, art_end)
        if cursor < valid_end:
            kept.append([cursor, valid_end])

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
        """Thin ``IntervalList.fetch1('valid_times')`` wrapper.

        Returns the artifact-removed valid times array for the
        ``artifact_id`` keyed by ``key``. The IntervalList row's
        ``nwb_file_name`` is resolved through the upstream source part
        so this works for both single-recording and (once implemented)
        shared-artifact-group selections.
        """
        from spyglass.spikesorting.v2.recording import RecordingSelection

        source = ArtifactSelection.resolve_source(key)
        if "artifact_id" not in key:
            raise ValueError(
                "ArtifactDetection.get_artifact_removed_intervals: key "
                "must include 'artifact_id'."
            )
        interval_list_name = f"artifact_{key['artifact_id']}"

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

        raise NotImplementedError(
            "ArtifactDetection.get_artifact_removed_intervals for "
            "SharedArtifactGroup is not yet implemented."
        )

    def delete(self, *args, safemode=None, **kwargs):
        """Override that also removes the matching IntervalList rows.

        DataJoint does not cascade through ``interval_list_name``-keyed
        dependencies, so the cleanup is explicit. We fetch the
        ``artifact_id`` (and, for shared-group sources, the member
        ``nwb_file_name``s) up front, delete the master row(s) via
        ``super().delete``, then drop the matching IntervalList rows.
        """
        from spyglass.spikesorting.v2.recording import RecordingSelection

        # Collect the IntervalList rows to clean up BEFORE we delete the
        # master rows -- the source-part join no longer resolves after
        # the master is gone.
        interval_rows_to_remove = []
        for row in self.fetch(as_dict=True):
            try:
                source = ArtifactSelection.resolve_source(row)
            except Exception as resolve_exc:
                # Surface the swallow so a broken ``resolve_source``
                # does not silently leave ``IntervalList`` rows
                # behind. The row's master is still deleted by the
                # cascade below; only the paired IntervalList cleanup
                # is skipped here.
                logger.error(
                    "ArtifactDetection.delete: resolve_source failed for "
                    f"{ {k: row[k] for k in ('artifact_id',)} }; leaving "
                    f"IntervalList rows in place. Underlying error: "
                    f"{resolve_exc!r}"
                )
                continue
            interval_list_name = f"artifact_{row['artifact_id']}"
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
            # shared_artifact_group branch lands with the group helper.

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

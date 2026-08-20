"""Artifact detection over a preprocessed Recording.

Tables (source-specific split; unified downstream by the
``ArtifactDetectionOutput`` merge in ``artifact_output.py``):
    ArtifactDetectionParameters       -- threshold detection parameters.
    SharedArtifactGroup (+ Member)    -- opt-in cross-recording detection (#928).
    RecordingArtifactSelection        -- single-recording request (Recording FK).
    SharedGroupArtifactSelection      -- cross-recording request (+ member_set_hash).
    RecordingArtifactDetection        -- single-recording result; writes IntervalList.
        .RemovedInterval              -- relational owner of each written row.
    SharedGroupArtifactDetection      -- cross-recording result; one row per member.
        .RemovedInterval              -- relational owner of each written row.

The recording source is a required FK on each selection master, so
"exactly one source" is STRUCTURAL (a row cannot exist with zero or two
sources) rather than a runtime-asserted XOR. ``insert_artifact_detection``
dispatches a single-entry request to the matching selection.

Artifact-removed valid times live in ``common.IntervalList`` (the
UUID-suffixed name prevents collision with human-authored session intervals,
and downstream IntervalList-querying code consumes them through the standard
interface); each detection's ``RemovedInterval`` part table owns its generated
rows relationally so generic IntervalList cleanup treats them as live children,
not orphans.

``ArtifactDetectionParameters.insert1`` Pydantic-validates the ``params`` blob.
Each detection's ``make`` runs threshold detection and writes the
artifact-removed ``IntervalList`` rows, ``get_artifact_removed_intervals`` reads
them back, ``delete`` removes them, and ``SharedArtifactGroup.insert_group``
declares a cross-recording detection bundle. The source-agnostic machinery (the
scan, the IntervalList write, the read-back, the delete cleanup) lives on the
shared ``_ArtifactDetectionMixin`` both result tables inherit.
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
    build_artifact_interval_part_rows,
    build_artifact_interval_rows,
    collect_artifact_interval_rows_to_remove,
    detect_artifacts,
    read_owned_artifact_intervals,
    remove_artifact_interval_rows,
    scan_artifact_frames,
)
from spyglass.spikesorting.v2._params.artifact_detection import (
    ARTIFACT_DETECTION_SCHEMA_VERSION,
    ArtifactDetectionParamsSchema,
)
from spyglass.spikesorting.v2._recipe_catalog import artifact_default_contents
from spyglass.spikesorting.v2._signal_math import timestamp_fingerprint
from spyglass.spikesorting.v2.recording import Recording
from spyglass.spikesorting.v2.utils import (
    ImmutableParamsLookup,
    SelectionMasterInsertGuard,
    _validate_params,
    reject_duplicate_parameter_content,
    split_leading_restrictions,
    transaction_or_noop,
    validate_lookup_rows,
)
from spyglass.utils import SpyglassMixin, SpyglassMixinPart, logger

schema = dj.schema("spikesorting_v2_artifact")


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
class ArtifactDetectionParameters(
    ImmutableParamsLookup, SpyglassMixin, dj.Lookup
):
    """Validated artifact-detection parameter blob.

    The ``params`` blob is validated by
    :class:`ArtifactDetectionParamsSchema`. ``insert_default`` ships four rows:
    ``"none"`` (detect=False, skip artifact scanning), ``"default"`` (the 500 uV
    schema default: amplitude threshold + proportion-above-thresh), and the two
    production Frank-lab recipes ``"franklab_100uv_p07_2026_06"`` (100 uV, 0.7
    proportion) and ``"franklab_50uv_p07_2026_06"`` (50 uV, 0.7). A preset that
    omits ``artifact_detection_params_name`` runs no detection at all (no
    ``ArtifactDetectionSource`` row), which is the only valid shape for a concat
    sort.

    ``job_kwargs`` is the optional per-row SpikeInterface job-kwargs blob that
    governs the chunked detection scan
    (``ArtifactDetection._scan_artifact_frames``). It is merged over the
    SI-global and ``dj.config['custom']
    ['spikesorting_v2_job_kwargs']`` defaults by ``_resolved_job_kwargs``. The
    memory-relevant key is the chunk size -- ``chunk_duration`` (e.g. ``"1s"``,
    the default), ``chunk_size`` (frames), or ``chunk_memory`` -- which bounds
    peak working set at ``~4 × chunk_frames × n_channels × 4 bytes``. ``n_jobs``
    controls the worker-pool size (default 1, serial in-process).
    """

    definition = f"""
    artifact_detection_params_name: varchar(64)
    ---
    params: blob
    params_schema_version={ARTIFACT_DETECTION_SCHEMA_VERSION}: int
    job_kwargs=null: blob  # SI job-kwargs for chunked scan; see docstring
    """

    # Row-level ``params_schema_version`` matches the inner
    # ``ArtifactDetectionParamsSchema.schema_version``.
    # The shipped rows are defined in
    # ``_recipe_catalog.artifact_default_contents`` (single source).
    _DEFAULT_CONTENTS: tuple = artifact_default_contents()

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
        from spyglass.spikesorting.v2._shared_artifact_group import (
            validate_shared_artifact_group_members,
        )
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
                    f"must include 'recording_id'. Got: {m!r}."
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

        # Cheap pre-load consistency: one session AND one sampling frequency
        # (SI's ``aggregate_channels`` requires identical fs). Validated BEFORE
        # loading any member's (5-50 GB) recording for the exact-timestamp check
        # below, so a misconfigured group is rejected without the load cost.
        nwb_file_name = validate_shared_artifact_group_members(
            [
                {
                    "nwb_file_name": sel_by_id[str(rid)]["nwb_file_name"],
                    "sampling_frequency": rec_by_id[str(rid)][
                        "sampling_frequency"
                    ],
                }
                for rid in member_recording_ids
            ]
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
        reference_fingerprint = None
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
            # Internal consistency without materializing: the lazy time_vector's
            # length (h5py shape[0], no data read) must match get_num_samples().
            explicit_time_vector = rec_obj.get_time_info(segment_index=0).get(
                "time_vector"
            )
            if (
                explicit_time_vector is not None
                and len(explicit_time_vector) != n_samples
            ):
                raise ValueError(
                    "SharedArtifactGroup.insert_group: recording_id="
                    f"{rid!r} reports get_num_samples()={n_samples}, "
                    f"but its timestamp vector has length "
                    f"{len(explicit_time_vector)}. The preprocessed recording "
                    "is internally inconsistent and cannot be used for a "
                    "shared artifact group."
                )
            # Time-alignment check via a chunked SHA-256 fingerprint of the
            # timestamp vector rather than holding two full ~824 MB float64
            # vectors for an ``np.array_equal``. ``timestamp_fingerprint``
            # streams the vector in ~1 s slices (bounded peak memory) and
            # prefixes n_samples, so a length difference also yields differing
            # fingerprints and is caught here as a timestamp mismatch.
            fingerprint = timestamp_fingerprint(rec_obj)
            if reference_fingerprint is None:
                reference_fingerprint = fingerprint
                reference_rid = rid
            elif fingerprint != reference_fingerprint:
                raise ValueError(
                    "SharedArtifactGroup.insert_group: members have "
                    "differing exact timestamps (recording_id="
                    f"{reference_rid!r} and recording_id={rid!r} have "
                    "non-identical timestamp vectors). Shared artifact "
                    "detection requires time-aligned recordings; use "
                    "recordings from the same interval identity or run "
                    "separate artifact detections."
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


def _insert_artifact_selection(
    selection_cls,
    *,
    artifact_detection_params_name,
    recording_id=None,
    shared_artifact_group_name=None,
    supplied_id=None,
    extra_row,
) -> dict:
    """Idempotently insert one split artifact-detection selection row.

    Shared body for ``RecordingArtifactSelection.insert_selection`` and
    ``SharedGroupArtifactSelection.insert_selection``. Derives the
    content-addressed ``artifact_detection_id`` from the same
    ``artifact_detection_identity_payload`` + ``deterministic_id`` -- so the id
    is content-addressed purely by params + source (keeping artifact-backed
    ``sorting_id`` parity and the ``artifact_detection_{id}`` IntervalList name
    stable) -- validates any caller-supplied id, ensures the params row exists,
    and inserts the single selection row.

    Because the recording source is now a required FK on the master row (not
    an XOR part), a selection cannot exist with zero or two sources: the
    orphan / ambiguous states the pre-split ``_find_existing_pk`` /
    ``resolve_source`` guarded against are structurally unrepresentable, so
    this insert only handles the concurrent duplicate-id race.

    Parameters
    ----------
    selection_cls : dj.Manual
        The split selection table to insert into.
    artifact_detection_params_name : str
        The ``ArtifactDetectionParameters`` row name.
    recording_id : uuid or str, optional
        Single-recording source id (mutually exclusive with
        ``shared_artifact_group_name``).
    shared_artifact_group_name : str, optional
        Shared-group source name (mutually exclusive with ``recording_id``).
    supplied_id : optional
        A caller-supplied ``artifact_detection_id`` to validate against the
        derived one, or ``None``.
    extra_row : dict
        Source-specific columns to add to the inserted row (the source FK,
        plus ``member_set_hash`` for the shared-group table).

    Returns
    -------
    dict
        ``{"artifact_detection_id": ...}`` -- the content-addressed PK.
    """
    from spyglass.spikesorting.v2._selection_identity import (
        artifact_detection_identity_payload,
        assert_supplied_id_matches,
        deterministic_id,
    )
    from spyglass.spikesorting.v2.utils import (
        _ensure_lookup_row_exists,
        _is_duplicate_key_error,
    )

    payload = artifact_detection_identity_payload(
        artifact_detection_params_name=artifact_detection_params_name,
        recording_id=recording_id,
        shared_artifact_group_name=shared_artifact_group_name,
    )
    artifact_detection_id = deterministic_id("artifact_detection", payload)
    assert_supplied_id_matches(
        supplied_id, artifact_detection_id, field="artifact_detection_id"
    )
    pk = {"artifact_detection_id": artifact_detection_id}
    if selection_cls & pk:
        return pk

    _ensure_lookup_row_exists(
        ArtifactDetectionParameters,
        {"artifact_detection_params_name": artifact_detection_params_name},
        helper_name=f"{selection_cls.__name__}.insert_selection",
        insert_default_path="ArtifactDetectionParameters.insert_default()",
    )
    row = {
        **pk,
        "artifact_detection_params_name": artifact_detection_params_name,
        **extra_row,
    }
    try:
        with transaction_or_noop(selection_cls.connection):
            # allow_direct_insert: insert_selection IS the validation boundary.
            selection_cls.insert1(row, allow_direct_insert=True)
    except Exception as exc:  # noqa: BLE001 -- re-raised unless a dup-PK race
        if not _is_duplicate_key_error(exc):
            raise
        # A concurrent caller won the deterministic-id race; its row is the
        # same content-addressed selection, so return the shared PK.
        logger.debug(
            "%s.insert_selection: lost deterministic-id race on %s; "
            "returning the existing row.",
            selection_cls.__name__,
            artifact_detection_id,
        )
    return pk


@schema
class RecordingArtifactSelection(
    SelectionMasterInsertGuard, SpyglassMixin, dj.Manual
):
    """Single-recording artifact-detection request.

    Source-specific split of the former
    ``ArtifactDetectionSelection.RecordingSource``: the recording source is a
    required FK on the master row, so "exactly one recording source" is
    structural (a row cannot exist with zero or two sources) rather than a
    runtime-asserted invariant.
    """

    definition = """
    artifact_detection_id: uuid
    ---
    -> ArtifactDetectionParameters
    -> Recording
    """

    @classmethod
    def insert_selection(cls, key: dict) -> dict:
        """Idempotently register a single-recording artifact selection.

        Parameters
        ----------
        key : dict
            Must carry ``artifact_detection_params_name`` and
            ``recording_id``. An explicit ``artifact_detection_id`` is
            validated against the derived deterministic id.

        Returns
        -------
        dict
            ``{"artifact_detection_id": ...}`` -- the content-addressed PK.
        """
        recording_id = key["recording_id"]
        return _insert_artifact_selection(
            cls,
            artifact_detection_params_name=key[
                "artifact_detection_params_name"
            ],
            recording_id=recording_id,
            supplied_id=key.get("artifact_detection_id"),
            extra_row={"recording_id": recording_id},
        )


@schema
class SharedGroupArtifactSelection(
    SelectionMasterInsertGuard, SpyglassMixin, dj.Manual
):
    """Cross-recording (shared-group) artifact-detection request.

    Source-specific split of the former
    ``ArtifactDetectionSelection.SharedGroupSource``. ``member_set_hash``
    snapshots the group's ordered member ``recording_id`` set at selection
    time; ``SharedGroupArtifactDetection.make_fetch`` re-derives it from the
    current members and rejects a drift (the identity is ``{params, group}``
    only, so a live member edit could otherwise change the scanned set under a
    fixed ``artifact_detection_id``).
    """

    definition = """
    artifact_detection_id: uuid
    ---
    -> ArtifactDetectionParameters
    -> SharedArtifactGroup
    member_set_hash: char(64)   # frozen sha256 of the ordered member recording_id set
    """

    @classmethod
    def insert_selection(cls, key: dict) -> dict:
        """Idempotently register a shared-group artifact selection.

        Snapshots the group's CURRENT member set onto ``member_set_hash`` so a
        later ``SharedArtifactGroup.Member`` edit cannot silently change the
        scanned set under this fixed ``artifact_detection_id``.

        Parameters
        ----------
        key : dict
            Must carry ``artifact_detection_params_name`` and
            ``shared_artifact_group_name``. An explicit
            ``artifact_detection_id`` is validated against the derived id.

        Returns
        -------
        dict
            ``{"artifact_detection_id": ...}`` -- the content-addressed PK.
        """
        from spyglass.spikesorting.v2._selection_identity import (
            shared_group_member_set_hash,
        )

        group_name = key["shared_artifact_group_name"]
        member_recording_ids = (
            SharedArtifactGroup.Member
            & {"shared_artifact_group_name": group_name}
        ).fetch("recording_id")
        member_set_hash = shared_group_member_set_hash(member_recording_ids)
        return _insert_artifact_selection(
            cls,
            artifact_detection_params_name=key[
                "artifact_detection_params_name"
            ],
            shared_artifact_group_name=group_name,
            supplied_id=key.get("artifact_detection_id"),
            extra_row={
                "shared_artifact_group_name": group_name,
                "member_set_hash": member_set_hash,
            },
        )


def insert_artifact_detection(key: dict) -> dict:
    """Insert an artifact-detection selection, dispatching on source kind.

    Single-entry UX over the two split selections: a single-recording key
    (``recording_id``) routes to :class:`RecordingArtifactSelection`, a
    cross-recording key (``shared_artifact_group_name``) to
    :class:`SharedGroupArtifactSelection`. Preserves the call shape of the
    former ``ArtifactDetectionSelection.insert_selection``.

    Parameters
    ----------
    key : dict
        Carries ``artifact_detection_params_name`` plus exactly one of
        ``recording_id`` or ``shared_artifact_group_name``.

    Returns
    -------
    dict
        ``{"artifact_detection_id": ...}`` -- the content-addressed PK.

    Raises
    ------
    ValueError
        If neither or both source keys are supplied.
    """
    has_recording = "recording_id" in key
    has_shared = "shared_artifact_group_name" in key
    if has_recording == has_shared:
        raise ValueError(
            "insert_artifact_detection requires exactly one source key: "
            "recording_id (single-recording) xor shared_artifact_group_name "
            "(cross-recording), not both and not neither."
        )
    if has_recording:
        return RecordingArtifactSelection.insert_selection(key)
    return SharedGroupArtifactSelection.insert_selection(key)


class RecordingArtifactFetched(NamedTuple):
    """DB-side inputs for :meth:`RecordingArtifactDetection.make_fetch`.

    The single-recording source is structural, so the fetched shape carries
    just the resolved ``recording_id`` + its parent ``nwb_file_name`` (no
    source-kind branch, no member tuples).
    """

    validated: ArtifactDetectionParamsSchema
    recording_id: object
    nwb_file_name: str
    artifact_job_kwargs: dict | None


class SharedGroupArtifactFetched(NamedTuple):
    """DB-side inputs for :meth:`SharedGroupArtifactDetection.make_fetch`.

    The per-member fields are ordered tuples (length n_members) so
    ``make_compute`` can union their channels without further DB I/O.
    ``nwb_file_name`` is the common parent session (``insert_group``
    validated single-session).
    """

    validated: ArtifactDetectionParamsSchema
    shared_artifact_group_name: str
    member_recording_ids: tuple
    member_nwb_file_names: tuple
    nwb_file_name: str
    artifact_job_kwargs: dict | None


class _ArtifactDetectionMixin:
    """Shared source-agnostic detection machinery for the split result tables.

    ``RecordingArtifactDetection`` and ``SharedGroupArtifactDetection`` differ
    only in how ``make_fetch`` / ``make_compute`` resolve and LOAD their source
    (a single cached recording vs the channel-union of a shared group's
    members). Everything downstream of the loaded recording -- the chunked
    threshold scan, the ``IntervalList`` write + ownership part rows, the
    read-back, and the delete-time cleanup -- is identical, so it lives here
    and both tables inherit it. The scan itself is the source-agnostic helper
    both ``make_compute``s funnel through (:meth:`_run_artifact_scan`).

    ``_single_source`` records whether the table owns exactly one
    ``IntervalList`` row (recording source) so
    :meth:`get_artifact_removed_intervals` can return a bare array for the
    single-recording case while the shared-group case always returns the
    per-member dict.
    """

    # Tri-part dispatch moves the long-running detection loop OUTSIDE the
    # framework transaction (see the pre-split ``ArtifactDetection`` for the
    # DataJoint #1170 / Spyglass #1030 background).
    _parallel_make = True
    _single_source: bool = True

    @staticmethod
    def _scan_artifact_frames(recording, validated, job_kwargs=None):
        """Flag contiguous artifact-frame RUNS via a chunked executor.

        Thin delegator to
        :func:`._artifact_intervals.scan_artifact_frames`; kept as a
        staticmethod for the public/tested chunked artifact-scan boundary.
        """
        return scan_artifact_frames(recording, validated, job_kwargs)

    @staticmethod
    def _detect_artifacts(recording, validated, context="", job_kwargs=None):
        """Run amplitude / z-score artifact scan on a SI recording.

        Thin delegator to :func:`._artifact_intervals.detect_artifacts`; kept
        as a staticmethod because ``make_compute`` calls
        ``self._detect_artifacts(...)`` and the v2 tests call it on the class.
        """
        return detect_artifacts(
            recording, validated, context=context, job_kwargs=job_kwargs
        )

    def _run_artifact_scan(
        self, recording, validated, artifact_job_kwargs, context
    ):
        """Resolve job kwargs and scan a loaded recording for artifacts.

        The source-agnostic detection helper both ``make_compute``s call once
        their (single or unioned) recording is loaded: merges the per-row
        ``job_kwargs`` over the SI-global and DataJoint-config defaults, then
        runs the chunked threshold scan.

        Parameters
        ----------
        recording : si.BaseRecording
            The loaded single or channel-unioned recording to scan.
        validated : ArtifactDetectionParamsSchema
            Validated detection parameters.
        artifact_job_kwargs : dict or None
            Per-row SI job-kwargs blob.
        context : str
            Diagnostic suffix identifying the selection for empty-scan logs.

        Returns
        -------
        np.ndarray
            Artifact-removed ``valid_times``, shape ``(n_intervals, 2)``.
        """
        from spyglass.spikesorting.v2.utils import _resolved_job_kwargs

        resolved_job_kwargs = _resolved_job_kwargs(artifact_job_kwargs)
        return self._detect_artifacts(
            recording,
            validated,
            context=context,
            job_kwargs=resolved_job_kwargs,
        )

    def make_insert(
        self, key, valid_times, nwb_file_name, per_member_nwb_files=()
    ):
        """Write the artifact ``IntervalList`` + master + part rows atomically.

        Source-agnostic: writes ONE ``IntervalList`` row per distinct member
        ``nwb_file_name`` (one row for the single-recording path), registers
        the master row, and records ownership through
        ``RemovedInterval`` so generic ``IntervalList`` cleanup treats
        the generated intervals as live children. The inner
        ``transaction_or_noop`` is a no-op under the framework transaction the
        tri-part dispatch already opened; kept so an out-of-populate caller
        still gets atomic registration.

        Parameters
        ----------
        key : dict
            Primary-key restriction for the row being inserted.
        valid_times : np.ndarray
            Artifact-removed valid times, shape ``(n_intervals, 2)``.
        nwb_file_name : str
            Fallback ``IntervalList`` target when ``per_member_nwb_files`` is
            empty.
        per_member_nwb_files : tuple, optional
            Distinct member ``nwb_file_name`` s to write one row each.
        """
        from spyglass.spikesorting.v2.artifact_output import (
            ArtifactDetectionOutput,
        )

        interval_rows = build_artifact_interval_rows(
            key, valid_times, nwb_file_name, per_member_nwb_files
        )
        part_rows = build_artifact_interval_part_rows(key, interval_rows)
        detection_key = {"artifact_detection_id": key["artifact_detection_id"]}
        with transaction_or_noop(self.connection):
            IntervalList.insert(interval_rows)
            self.insert1(key)
            self.RemovedInterval.insert(part_rows)
            # Producer-owned registration: a materialized detection registers
            # itself into the ArtifactDetectionOutput merge as an available
            # source, so a later SortingSelection just resolves its merge id and
            # never has to register (no merge insert inside the sorting
            # transaction). Idempotent + inside this atomic block, so a detection
            # is registered iff it is materialized.
            ArtifactDetectionOutput.insert_detection(detection_key)

    def get_artifact_removed_intervals(self, key, as_dict=False):
        """Return the artifact-removed ``valid_times`` for ``key``.

        Reads the detection row's OWN ``RemovedInterval`` part rows.
        A single-recording source (``_single_source``) owns exactly one row,
        returned as a plain ``(n_intervals, 2)`` array (or a one-key dict with
        ``as_dict=True``); a shared-group source returns the
        ``{nwb_file_name: array}`` dict (values equal across members -- the
        scan ran ONCE over the unioned channels).

        Parameters
        ----------
        key : dict
            Restriction selecting one detection row; must include
            ``artifact_detection_id``.
        as_dict : bool, optional
            Force the dict shape even for a single-recording source.

        Returns
        -------
        np.ndarray or dict[str, np.ndarray]
            The ``valid_times`` array (single-recording, ``as_dict=False``)
            or the per-member dict otherwise.
        """
        result = read_owned_artifact_intervals(type(self), key)
        if self._single_source:
            if len(result) != 1:
                raise ValueError(
                    f"{type(self).__name__}.get_artifact_removed_intervals: "
                    f"recording-backed {key!r} has {len(result)} "
                    "RemovedInterval part rows; expected exactly one."
                )
            if not as_dict:
                return next(iter(result.values()))
        return result

    def _merge_registration(self, row):
        """Return ``(merge_id, n_referencing_sortings)`` for a detection row.

        ``(None, 0)`` when the detection is not registered in
        ``ArtifactDetectionOutput`` (so nothing references it through the merge).
        """
        from spyglass.spikesorting.v2.artifact_output import (
            ArtifactDetectionOutput,
        )
        from spyglass.spikesorting.v2.sorting import SortingSelection

        det_key = {"artifact_detection_id": row["artifact_detection_id"]}
        try:
            merge_id = ArtifactDetectionOutput.get_merge_id(det_key)
        except KeyError:
            return None, 0
        n = len(
            SortingSelection.ArtifactDetectionSource
            & {"artifact_detection_merge_id": merge_id}
        )
        return merge_id, n

    def delete(self, *args, safemode=None, _cascade_sorts=False, **kwargs):
        """Delete detection rows, their ``ArtifactDetectionOutput`` registration,
        and their owned ``IntervalList`` rows -- coordinated with the merge.

        Deletion is REFUSED (``ValueError``) when a detection is referenced by a
        ``SortingSelection`` through the merge, so an existing sort does not
        silently lose its artifact pass. Delete the dependent sorts first, or use
        :meth:`cascade_delete` to remove them too. A per-detection advisory lock
        (the same one ``SortingSelection.insert_selection`` takes when it links
        an artifact) serializes delete-vs-select on a detection, so a CONCURRENT
        ``insert_selection`` cannot slip a new referrer between the check and the
        cascade -- it either commits first (and is refused) or blocks until the
        detection is gone (and fails on the FK). The lock is FAIL-CLOSED: a lock
        it cannot take within the short lifecycle timeout aborts the delete with
        ``AdvisoryLockError`` rather than proceeding unserialized. The per-
        detection locks are taken in a deterministic (``artifact_detection_id``-
        sorted) order, so two overlapping bulk deletes acquire them in the same
        order and cannot deadlock.

        For an unreferenced detection, the cautious cascade (which sets
        ``force_masters`` on modern DataJoint) removes the detection together
        with its ``ArtifactDetectionOutput`` source part + merge master -- gated
        by ``safemode``, so a cancelled delete leaves both in place (no orphaned
        merge row). In ``cascade`` mode the same cascade continues on through
        ``SortingSelection.ArtifactDetectionSource`` to the dependent sorts.
        DataJoint does not cascade through ``interval_list_name``-keyed
        dependencies, so the owned artifact ``IntervalList`` rows are collected
        up front and dropped for masters the cascade actually removed. A leading
        positional restriction is accepted for the easy-to-mistype
        ``Table().delete(restriction)`` form.
        """
        restriction_args, args = split_leading_restrictions(args)
        if restriction_args:
            target = self
            for restriction in restriction_args:
                target = target & restriction
            return target.delete(
                *args,
                safemode=safemode,
                _cascade_sorts=_cascade_sorts,
                **kwargs,
            )

        from contextlib import ExitStack

        from spyglass.spikesorting.v2._db_locking import required_advisory_lock
        from spyglass.spikesorting.v2.artifact_output import (
            ArtifactDetectionOutput,
        )

        detection_cls = type(self)
        # Sort by artifact_detection_id so the per-detection locks below are
        # acquired in a deterministic order: two overlapping bulk deletes then
        # take them in the SAME order and cannot deadlock (A->B vs B->A).
        rows = sorted(
            self.fetch(as_dict=True),
            key=lambda r: str(r["artifact_detection_id"]),
        )

        # Serialize delete-vs-insert_selection on each detection with the same
        # advisory lock insert_selection takes when it links an artifact, keyed
        # on ArtifactDetectionOutput + artifact_detection_id so both sides derive
        # the same lock name. This closes the check-then-cascade race: a
        # concurrent insert_selection linking one of these detections either runs
        # fully BEFORE the referrer check (so it is seen and refused) or fully
        # AFTER the cascade (so it fails on the deleted merge FK), never in
        # between where the force_masters cascade would silently drop its sort.
        # FAIL-CLOSED: required_advisory_lock RAISES AdvisoryLockError if it
        # cannot take the lock within the lifecycle timeout, so a delete never
        # proceeds unserialized (ExitStack releases any locks already taken).
        with ExitStack() as locks:
            for row in rows:
                locks.enter_context(
                    required_advisory_lock(
                        ArtifactDetectionOutput,
                        {"artifact_detection_id": row["artifact_detection_id"]},
                    )
                )

            # Refuse to delete a detection a sorting references (unless
            # cascading): a sort must not silently lose its artifact pass.
            # ``_merge_registration`` returns the referencing-sort count via the
            # merge; under the lock the count is stable through the cascade.
            if not _cascade_sorts:
                referenced = [
                    str(row["artifact_detection_id"])
                    for row in rows
                    if self._merge_registration(row)[1]
                ]
                if referenced:
                    raise ValueError(
                        f"{detection_cls.__name__}.delete: refusing to delete "
                        "artifact detection(s) referenced by a SortingSelection:"
                        f" {referenced}. Delete those sorts first, or call "
                        "cascade_delete() to remove them too."
                    )

            delete_targets = [
                (
                    {k: row[k] for k in self.primary_key},
                    collect_artifact_interval_rows_to_remove(
                        [row], detection_cls=detection_cls
                    ),
                )
                for row in rows
            ]

            # The cautious cascade (force_masters on modern DataJoint) walks
            # detection -> ArtifactDetectionOutput source part -> merge master
            # (and, when cascading, on to the dependent sorts), so the merge
            # registration is removed WITH the detection -- atomically and
            # safemode-gated.
            if safemode is None:
                super().delete(*args, **kwargs)
            else:
                super().delete(*args, safemode=safemode, **kwargs)

        for master_key, interval_rows_to_remove in delete_targets:
            if not (detection_cls & master_key):
                remove_artifact_interval_rows(interval_rows_to_remove)

    def cascade_delete(self, *args, safemode=None, **kwargs):
        """Delete these detections AND every sorting that references them.

        The explicit force path for :meth:`delete`'s refuse-if-referenced
        default: removes the dependent ``SortingSelection`` rows (and their
        downstream ``Sorting`` / ``CurationV2`` / ``SpikeSortingOutput``
        cascade), then the merge registration, then the detection.
        """
        return self.delete(
            *args, safemode=safemode, _cascade_sorts=True, **kwargs
        )


@schema
class RecordingArtifactDetection(
    _ArtifactDetectionMixin, SpyglassMixin, dj.Computed
):
    """Artifact-removed valid times for a single ``Recording``.

    Source-specific split of the pre-split ``ArtifactDetection`` recording
    branch. The recording source is structural (the selection master carries a
    required ``Recording`` FK), so ``make_fetch`` resolves it directly without
    a source-kind branch. Writes exactly one ``IntervalList`` row keyed by the
    recording's parent ``nwb_file_name``.
    """

    definition = """
    -> RecordingArtifactSelection
    """

    _single_source = True

    class RemovedInterval(SpyglassMixinPart):
        """Generated artifact-removed ``IntervalList`` row (relational owner).

        Named ``RemovedInterval`` (not ``ArtifactRemovedInterval``) so the
        derived part-table + FK identifier stays under MySQL's 64-char limit
        for the long ``shared_group_artifact_detection`` sibling master.
        """

        definition = """
        -> master
        -> IntervalList
        """

    def make_fetch(self, key):
        """Read the params blob, resolved recording, and parent session.

        Returns
        -------
        RecordingArtifactFetched
            Validated params, the resolved ``recording_id``, its parent
            ``nwb_file_name``, and the per-row job kwargs.
        """
        from spyglass.spikesorting.v2.recording import RecordingSelection

        params_blob, artifact_job_kwargs = (
            ArtifactDetectionParameters * (RecordingArtifactSelection & key)
        ).fetch1("params", "job_kwargs")
        validated = ArtifactDetectionParamsSchema.model_validate(params_blob)
        recording_id = (RecordingArtifactSelection & key).fetch1("recording_id")
        nwb_file_name = (
            RecordingSelection & {"recording_id": recording_id}
        ).fetch1("nwb_file_name")
        return RecordingArtifactFetched(
            validated=validated,
            recording_id=recording_id,
            nwb_file_name=nwb_file_name,
            artifact_job_kwargs=artifact_job_kwargs,
        )

    def make_compute(
        self, key, validated, recording_id, nwb_file_name, artifact_job_kwargs
    ):
        """Load the single recording and scan it for artifacts.

        Returns
        -------
        ArtifactComputed
            The ``valid_times`` plus the one-element per-member target list.
        """
        recording = Recording().get_recording({"recording_id": recording_id})
        valid_times = self._run_artifact_scan(
            recording,
            validated,
            artifact_job_kwargs,
            context=(
                f" for artifact_detection_id={key['artifact_detection_id']}, "
                f"recording_id={recording_id}"
            ),
        )
        return ArtifactComputed(
            valid_times=valid_times,
            nwb_file_name=nwb_file_name,
            per_member_nwb_files=(nwb_file_name,),
        )


@schema
class SharedGroupArtifactDetection(
    _ArtifactDetectionMixin, SpyglassMixin, dj.Computed
):
    """Artifact-removed valid times for a ``SharedArtifactGroup``.

    Source-specific split of the pre-split ``ArtifactDetection`` shared-group
    branch. ``make_fetch`` re-derives the group's member set and rejects a
    drift from the ``member_set_hash`` frozen on the selection (the identity is
    ``{params, group}`` only). ``make_compute`` unions the members' channels
    (``si.aggregate_channels``) and scans ONCE; ``make_insert`` writes the same
    ``valid_times`` into every distinct member session's ``IntervalList``.
    """

    definition = """
    -> SharedGroupArtifactSelection
    """

    _single_source = False

    class RemovedInterval(SpyglassMixinPart):
        """Generated artifact-removed ``IntervalList`` row (relational owner).

        Named ``RemovedInterval`` (not ``ArtifactRemovedInterval``) so the
        derived part-table + FK identifier stays under MySQL's 64-char limit
        for this long ``shared_group_artifact_detection`` master.
        """

        definition = """
        -> master
        -> IntervalList
        """

    def make_fetch(self, key):
        """Read params, resolve members, and reject a member-set drift.

        Returns
        -------
        SharedGroupArtifactFetched
            Validated params, the group name, the ordered member
            ``recording_id`` / ``nwb_file_name`` tuples, the canonical parent
            ``nwb_file_name``, and the per-row job kwargs.

        Raises
        ------
        SharedArtifactGroupMemberDriftError
            If the live member set differs from the ``member_set_hash`` frozen
            on the selection (scanning it would silently change the result
            under a fixed ``artifact_detection_id``).
        """
        from spyglass.spikesorting.v2._selection_identity import (
            shared_group_member_set_hash,
        )
        from spyglass.spikesorting.v2.exceptions import (
            SharedArtifactGroupMemberDriftError,
        )
        from spyglass.spikesorting.v2.recording import RecordingSelection

        group_name, frozen_hash = (SharedGroupArtifactSelection & key).fetch1(
            "shared_artifact_group_name", "member_set_hash"
        )
        params_blob, artifact_job_kwargs = (
            ArtifactDetectionParameters * (SharedGroupArtifactSelection & key)
        ).fetch1("params", "job_kwargs")
        validated = ArtifactDetectionParamsSchema.model_validate(params_blob)

        # Members ordered by recording_id so the tuple shape is DeepHash-stable
        # across the two make_fetch calls; the per-member nwb_file_name is
        # bulk-fetched through the join.
        members = (
            SharedArtifactGroup.Member * RecordingSelection
            & {"shared_artifact_group_name": group_name}
        ).fetch(
            "recording_id",
            "nwb_file_name",
            as_dict=True,
            order_by="recording_id",
        )
        if not members:
            raise RuntimeError(
                "SharedGroupArtifactDetection.make: SharedArtifactGroup "
                f"{group_name!r} has zero members; insert_group should have "
                "rejected the empty case."
            )
        member_recording_ids = tuple(str(m["recording_id"]) for m in members)
        # Reject a member-set drift: the artifact_detection_id was minted for
        # the set frozen on member_set_hash. Hashing the set just resolved
        # (rather than re-querying) yields the same digest insert_selection
        # minted (the helper sorts; every member has a RecordingSelection so
        # the join is total).
        current_hash = shared_group_member_set_hash(member_recording_ids)
        if current_hash != frozen_hash:
            raise SharedArtifactGroupMemberDriftError(
                "SharedGroupArtifactDetection.make: SharedArtifactGroup "
                f"{group_name!r} member set changed since "
                f"artifact_detection_id={key['artifact_detection_id']} was "
                f"created (frozen member_set_hash {frozen_hash[:12]} != "
                f"current {current_hash[:12]}). The artifact_detection_id "
                "identity is params+group only, so insert_selection() returns "
                "this same id with its stale snapshot -- to scan the new "
                "membership, DELETE this selection and re-run "
                "insert_selection() (which re-snapshots the current members), "
                "or restore the group's members."
            )
        member_nwb_file_names = tuple(m["nwb_file_name"] for m in members)
        return SharedGroupArtifactFetched(
            validated=validated,
            shared_artifact_group_name=group_name,
            member_recording_ids=member_recording_ids,
            member_nwb_file_names=member_nwb_file_names,
            nwb_file_name=member_nwb_file_names[0],
            artifact_job_kwargs=artifact_job_kwargs,
        )

    def make_compute(
        self,
        key,
        validated,
        shared_artifact_group_name,
        member_recording_ids,
        member_nwb_file_names,
        nwb_file_name,
        artifact_job_kwargs,
    ):
        """Union the members' channels and scan the union ONCE.

        Returns
        -------
        ArtifactComputed
            The shared ``valid_times`` plus the distinct member
            ``nwb_file_name`` target list ``make_insert`` writes one row per.
        """
        import spikeinterface as si

        if not member_recording_ids:
            raise RuntimeError(
                "SharedGroupArtifactDetection.make_compute: shared-group "
                "source has no member_recording_ids; make_fetch contract "
                "violated."
            )
        per_member_recordings = [
            Recording().get_recording({"recording_id": rid})
            for rid in member_recording_ids
        ]
        # aggregate_channels column-stacks along the channel axis. insert_group
        # enforces session + n_samples + fs + dtype + timestamp equality, but a
        # direct SharedGroupArtifactSelection insert can bypass that, so
        # re-assert the invariants over the loaded recordings here.
        from spyglass.spikesorting.v2._shared_artifact_group import (
            assert_shared_group_recordings_aggregatable,
        )

        assert_shared_group_recordings_aggregatable(
            per_member_recordings,
            member_recording_ids,
            member_nwb_file_names,
        )
        unioned = si.aggregate_channels(per_member_recordings)
        valid_times = self._run_artifact_scan(
            unioned,
            validated,
            artifact_job_kwargs,
            context=(
                f" for artifact_detection_id={key['artifact_detection_id']}, "
                f"shared_artifact_group={shared_artifact_group_name}"
            ),
        )
        # One IntervalList row per distinct member nwb_file_name (single-session
        # today, so length 1, but keep the tuple shape for a future relaxation).
        per_member_nwb_files = tuple(dict.fromkeys(member_nwb_file_names))
        return ArtifactComputed(
            valid_times=valid_times,
            nwb_file_name=nwb_file_name,
            per_member_nwb_files=per_member_nwb_files,
        )

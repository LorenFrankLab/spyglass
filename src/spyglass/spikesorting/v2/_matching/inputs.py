"""Matching-input resolution and checks behind ``UnitMatchSelection``.

:func:`insert_inputs` is the body of ``UnitMatchSelection.insert_inputs``: it
validates the requested curations, resolves each input's pinned generation,
source and constituent recordings (:func:`_resolve_match_input`), numbers the
inputs chronologically, freezes their recordings' frames and kept intervals,
and inserts the master with its ``Input`` / ``InputRecording`` parts in one
transaction. It returns an existing selection when one matches
(:func:`find_existing_pk`, the body behind the patched
``UnitMatchSelection._find_existing_pk``, which it calls through the class).
``table_cls`` is ``UnitMatchSelection``.

``UnitMatch.make_fetch`` and the ``TrackedUnit`` readers re-run the same input
checks on the frozen rows and compare them with a fresh resolution
(:func:`_snapshot_mismatches`); ``make_fetch`` also re-checks the session start
times (:func:`_session_start_mismatches`).
``UnitMatchSelection.insert_selection`` validates its per-member choices with
:func:`normalize_curation_choices` and :func:`_validate_member_curations`.

Imports without the DB layer: the DataJoint tables are imported inside the
functions.
"""

from __future__ import annotations

import functools
import uuid

from spyglass.spikesorting.v2._core.lookup_validation import lossless_int


@functools.lru_cache(maxsize=None)
def _warn_clusterless_match_once(sorting_id: str) -> None:
    """Warn once per clusterless sort that cross-session matching is degenerate.

    A clusterless sort's single "unit" is a threshold-crossing event stream, not
    a sorted neuron, so tracking it across sessions has no biological meaning.
    Deduped per ``sorting_id`` so a repeated selection does not spam the log;
    ``cache_clear()`` resets it (used by the tests).
    """
    from spyglass.utils import logger

    logger.warning(
        "UnitMatchSelection: sort %s was produced by the clusterless "
        "thresholder -- its units are threshold-crossing events, NOT sorted "
        "neurons (CurationV2.get_unit_semantics == "
        "'clusterless_threshold_crossings'). Cross-session matching tracks "
        "sorted neurons, so matching this input is degenerate.",
        sorting_id,
    )


def insert_inputs(
    table_cls,
    curations,
    matcher_params_name: str,
    session_group: tuple[str, str] | None = None,
) -> dict:
    """Find-existing-or-insert a match over explicit inputs; return its key.

    The body of ``UnitMatchSelection.insert_inputs`` (see its docstring for
    the inputs, ordering and errors). ``table_cls`` is the
    ``UnitMatchSelection`` class; ``_find_existing_pk``, the electrode-space
    warning and the backend geometry preflight are called on it.

    Duplicate-key recovery has no savepoint: it relies on the deterministic
    primary key colliding on the master insert, the transaction's first
    statement, before any part row is written. Keep the master insert first,
    or the recovery would leave part rows inside a caller's transaction.
    """
    import datajoint as dj

    from spyglass.spikesorting.v2._matching.graph import (
        chronological_input_order,
        input_set_hash,
    )
    from spyglass.spikesorting.v2._core.selection_identity import (
        deterministic_id,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.session_group import SessionGroup
    from spyglass.spikesorting.v2.unit_matching import MatcherParameters

    curations = list(curations)
    requested = _normalize_input_curations(curations)
    _check_input_count_and_sortings(requested, ValueError)
    group_key = None
    if session_group is not None:
        owner, name = session_group
        group_key = {
            "session_group_owner": owner,
            "session_group_name": name,
        }
        if not (SessionGroup & group_key):
            raise ValueError(
                "UnitMatchSelection.insert_inputs: SessionGroup "
                f"{group_key} does not exist."
            )
    for sorting_id, curation_id in requested:
        _check_input_curation(sorting_id, curation_id, ValueError)

    resolved = [
        _resolve_match_input(sorting_id, curation_id, ValueError)
        for sorting_id, curation_id in requested
    ]
    for pin, item in zip(curations, resolved, strict=True):
        if "curation_uuid" not in pin:
            continue
        expected = uuid.UUID(str(pin["curation_uuid"]))
        if expected != uuid.UUID(str(item["curation_uuid"])):
            raise ValueError(
                "UnitMatchSelection.insert_inputs: input "
                f"{_input_label(item['sorting_id'], item['curation_id'])} "
                f"has changed curation_uuid since planning (expected "
                f"{expected}, found {item['curation_uuid']}). The curation "
                "was deleted and recreated; rebuild and review the plan "
                "before matching."
            )
    start_times = _session_start_times(
        {
            recording["nwb_file_name"]
            for item in resolved
            for recording in item["recordings"]
        }
    )
    for item in resolved:
        for recording in item["recordings"]:
            recording["session_start_time"] = start_times[
                recording["nwb_file_name"]
            ]
        item["input_start_time"] = min(
            recording["session_start_time"] for recording in item["recordings"]
        )
    _check_input_sessions(resolved, ValueError)

    ordered = chronological_input_order(resolved)
    for input_index, item in enumerate(ordered):
        item["input_index"] = input_index
        # A single recording's frames and kept intervals come from its
        # persisted traces; they are frozen and hashed like a
        # concatenation member's.
        _add_single_recording_frames(item)
    input_rows, recording_rows = _input_part_rows(ordered)
    set_hash = input_set_hash(input_rows, recording_rows)
    identity = {
        "matcher_params_name": matcher_params_name,
        "input_set_hash": set_hash,
    }
    unitmatch_id = deterministic_id("unitmatch", identity)

    existing = table_cls._find_existing_pk(identity, unitmatch_id)
    if existing is not None:
        return existing

    # Honor unit semantics: warn (don't block -- the cheap clusterless
    # thresholder is a valid sort) when an input's units are threshold
    # crossings rather than sorted neurons, since matching them across
    # sessions is biologically degenerate.
    for item in ordered:
        if (
            CurationV2.get_unit_semantics({"sorting_id": item["sorting_id"]})
            == "clusterless_threshold_crossings"
        ):
            _warn_clusterless_match_once(str(item["sorting_id"]))

    # Preflight NOW, at selection time, before UnitMatch.make's expensive
    # dense bundle extraction: warn if inputs map to different electrode
    # identities (advisory -- group names / ids are not lab-stable), and
    # let the selected backend enforce its own geometry requirements. Only
    # on the new-insert path (an idempotent re-call of an already-validated
    # selection skips the I/O).
    choices_by_input = {
        item["input_index"]: (item["sorting_id"], item["curation_id"])
        for item in ordered
    }
    table_cls._warn_on_divergent_electrode_space(choices_by_input)
    matcher_name, params = (
        MatcherParameters & {"matcher_params_name": matcher_params_name}
    ).fetch1("matcher", "params")
    table_cls._validate_matcher_geometry(choices_by_input, matcher_name, params)

    master_row = {**identity, "unitmatch_id": unitmatch_id}
    if group_key is not None:
        master_row.update(group_key)
    try:
        with table_cls._safe_context():
            # allow_direct_insert: this helper IS the validation boundary
            # (it has validated the inputs and minted the deterministic
            # id), so it bypasses the master insert guard.
            table_cls().insert1(master_row, allow_direct_insert=True)
            table_cls.Input.insert(
                [{**row, "unitmatch_id": unitmatch_id} for row in input_rows]
            )
            table_cls.InputRecording.insert(
                [
                    {**row, "unitmatch_id": unitmatch_id}
                    for row in recording_rows
                ]
            )
    except dj.errors.DuplicateError:
        # Lost a concurrent race on the same deterministic unitmatch_id;
        # refetch and return the winner's row.
        existing = table_cls._find_existing_pk(identity, unitmatch_id)
        if existing is not None:
            return existing
        raise
    return {"unitmatch_id": unitmatch_id}


def find_existing_pk(
    table_cls, identity: dict, deterministic_unitmatch_id
) -> dict | None:
    """Return the canonical PK for this selection identity, or None.

    The full logical identity (matcher + ``input_set_hash``) lives in the
    master's own columns. A master matching the identity whose
    ``unitmatch_id`` is NOT the deterministic id is a raw-insert bypass and
    is rejected. When the deterministic master DOES exist, its ``Input`` /
    ``InputRecording`` parts are verified to be well formed and to realize
    the identity's ``input_set_hash``: a forged master can carry the right
    id with missing / stale / orphaned parts, and that must be rejected
    HERE rather than returning a "valid" PK that only ``make_fetch`` later
    rejects.

    Used by ``insert_inputs`` for both the pre-insert lookup and the
    post-duplicate-key refetch.
    """
    from spyglass.spikesorting.v2._matching.graph import (
        frozen_order_errors,
        input_part_structure_errors,
        input_set_hash,
    )
    from spyglass.spikesorting.v2._core.selection_identity import (
        existing_selection_pk,
    )
    from spyglass.spikesorting.v2.exceptions import SchemaBypassError

    master_ids = {
        row["unitmatch_id"]
        for row in (table_cls & identity).fetch("KEY", as_dict=True)
    }
    existing = existing_selection_pk(
        master_ids,
        deterministic_unitmatch_id,
        pk_field="unitmatch_id",
        bypass_message=lambda bypassed: (
            f"UnitMatchSelection has {len(master_ids)} master row(s) for "
            f"identity {identity} whose unitmatch_id is not the "
            f"deterministic id {deterministic_unitmatch_id}: {bypassed}. "
            "This is a non-deterministic selection row (a raw insert); "
            "drop it and re-insert via insert_inputs."
        ),
    )
    if existing is None:
        return None
    restriction = {"unitmatch_id": deterministic_unitmatch_id}
    input_rows = (table_cls.Input & restriction).fetch(as_dict=True)
    recording_rows = (table_cls.InputRecording & restriction).fetch(
        as_dict=True
    )
    structure_errors = input_part_structure_errors(input_rows, recording_rows)
    if not structure_errors:
        structure_errors = frozen_order_errors(input_rows, recording_rows)
    if structure_errors:
        raise SchemaBypassError(
            f"UnitMatchSelection master {deterministic_unitmatch_id} has "
            f"malformed or misordered Input / InputRecording parts "
            f"({'; '.join(structure_errors)}; a raw-insert orphan or "
            "forgery). Drop the master and re-insert via insert_inputs()."
        )
    if input_set_hash(input_rows, recording_rows) != identity["input_set_hash"]:
        raise SchemaBypassError(
            f"UnitMatchSelection master {deterministic_unitmatch_id} exists "
            "but its Input / InputRecording parts do not realize its "
            "input_set_hash (missing / stale parts -- a raw-insert orphan "
            "or forgery). Drop the master and re-insert via "
            "insert_inputs()."
        )
    return existing


def _input_label(sorting_id, curation_id) -> str:
    """Name one matching input in an error message."""
    return f"(sorting_id={sorting_id}, curation_id={curation_id})"


def _normalize_input_curations(curations) -> list[tuple]:
    """``[{sorting_id, curation_id}, ...]`` -> ``[(uuid.UUID, int), ...]``.

    Caller-supplied curation ids go through the lossless integer rule (a
    fractional or boolean id is rejected, not truncated).
    """
    return [
        (
            uuid.UUID(str(curation["sorting_id"])),
            lossless_int(curation["curation_id"], "curation_id"),
        )
        for curation in curations
    ]


def _check_input_count_and_sortings(pairs, exc_class) -> None:
    """Reject an empty input set or a sorting pinned more than once.

    A single input is valid: its run writes zero pairs, and every matchable
    unit becomes a singleton tracked unit.

    Parameters
    ----------
    pairs : list of (sorting_id, curation_id)
        The matching inputs.
    exc_class : type
        Exception raised on a violation.
    """
    if not pairs:
        raise exc_class(
            "UnitMatchSelection: a match selection needs at least one "
            "matching input; got none."
        )
    curations_by_sorting: dict = {}
    for sorting_id, curation_id in pairs:
        curations_by_sorting.setdefault(str(sorting_id), []).append(
            int(curation_id)
        )
    repeated = {
        sorting_id: curation_ids
        for sorting_id, curation_ids in curations_by_sorting.items()
        if len(curation_ids) > 1
    }
    if repeated:
        detail = "; ".join(
            f"sorting_id={sorting_id}: curation_id {curation_ids}"
            for sorting_id, curation_ids in sorted(repeated.items())
        )
        raise exc_class(
            "UnitMatchSelection: each sorting may be matched through one "
            "curation generation only, but these sortings are pinned more "
            f"than once -- {detail}. Pick one curation per sorting."
        )


def _check_input_curation(sorting_id, curation_id, exc_class) -> None:
    """Reject a missing curation or one with unapplied proposed merges."""
    from spyglass.spikesorting.v2.curation import CurationV2

    key = {"sorting_id": sorting_id, "curation_id": curation_id}
    if not (CurationV2 & key):
        raise exc_class(
            f"UnitMatchSelection: input {_input_label(sorting_id, curation_id)} "
            "pins a curation that does not exist."
        )
    # Matching UNMERGED units across sessions is unambiguously wrong -- a
    # curation created with apply_merge=False (proposed merges recorded but
    # not applied) would feed oversplit units into the matcher.
    if CurationV2.has_unapplied_proposed_merges(key):
        raise exc_class(
            f"UnitMatchSelection: input {_input_label(sorting_id, curation_id)} "
            "pins a curation with proposed merges that are NOT applied "
            "(apply_merge=False); matching unmerged (oversplit) units across "
            "sessions is wrong. Apply or drop the proposed merges first "
            "(CurationV2.insert_curation(..., apply_merge=True)) before adding "
            "the curation to a UnitMatch selection."
        )


def _resolve_match_input(
    sorting_id,
    curation_id,
    exc_class,
    *,
    context: str = "UnitMatchSelection",
    input_index: int | None = None,
) -> dict:
    """Resolve one matching input's pinned generation, source and recordings.

    Database reads only (no trace file is opened and ``Session`` is not
    read). A single-recording sort resolves to its one ``Recording``. A
    concatenation sort resolves to its frozen members
    (``ConcatenatedRecordingSelection.MemberSnapshot``) in member order;
    each member must still resolve to a ``Recording`` with the content hash
    the concatenation froze
    (:func:`._recording.concat_fetch.resolve_snapshot_recordings`), and carries
    its frames in the concatenation (``ConcatenatedRecording.MemberBoundary``:
    the cumulative exclusive ``end_sample``, a member starting where the
    previous one ended) and its kept intervals on its own clock
    (``member_valid_times``). A single recording's frames and kept intervals
    need its persisted traces and are added by :func:`_recording_n_samples` /
    :func:`_recording_valid_times`.

    Parameters
    ----------
    sorting_id : uuid.UUID
    curation_id : int
    exc_class : type
        Exception raised when the concatenation's boundaries do not match its
        frozen members, or (chained from the ``ConcatMemberDriftError`` /
        ``MissingRecordingForConcatError`` it catches) when a member
        ``Recording`` changed or is gone.
    context : str, optional
        Caller named at the start of the member-drift message. Default
        ``"UnitMatchSelection"``.
    input_index : int, optional
        The input's frozen ``input_index``, named in the member-drift message
        when known. Default ``None``.

    Returns
    -------
    dict
        ``sorting_id``, ``curation_id``, ``curation_uuid``, ``source_kind``
        (the ``SortingSelection`` source kind), ``source_id``,
        ``motion_corrected_recording_id`` (or ``None``),
        ``artifact_detection_id`` (the sort's pinned detection, or ``None``),
        and ``recordings``: one dict per constituent recording with
        ``recording_index``, ``nwb_file_name``, ``sort_group_id``,
        ``interval_list_name``, ``recording_id``, ``recording_content_hash``
        and, for a concatenation member, ``start_sample``, ``end_sample`` and
        ``valid_times``.
    """
    import numpy as np

    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
    )
    from spyglass.spikesorting.v2._recording.concat_fetch import (
        resolve_snapshot_recordings,
    )
    from spyglass.spikesorting.v2.exceptions import (
        ConcatMemberDriftError,
        MissingRecordingForConcatError,
    )
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecording,
        ConcatenatedRecordingSelection,
    )
    from spyglass.spikesorting.v2.sorting import SortingSelection

    curation_uuid = (
        CurationV2 & {"sorting_id": sorting_id, "curation_id": curation_id}
    ).fetch1("curation_uuid")
    source = SortingSelection.resolve_effective_source(
        {"sorting_id": sorting_id}
    )
    lineage = source.lineage
    if lineage.kind == "recording":
        source_id = lineage.key["recording_id"]
        nwb_file_name, sort_group_id, interval_list_name = (
            RecordingSelection & {"recording_id": source_id}
        ).fetch1("nwb_file_name", "sort_group_id", "interval_list_name")
        recordings = [
            {
                "recording_index": 0,
                "nwb_file_name": nwb_file_name,
                "sort_group_id": int(sort_group_id),
                "interval_list_name": interval_list_name,
                "recording_id": source_id,
                "recording_content_hash": str(
                    (Recording & {"recording_id": source_id}).fetch1(
                        "content_hash"
                    )
                ),
            }
        ]
    else:
        source_id = lineage.key["concat_recording_id"]
        concat_key = {"concat_recording_id": source_id}
        snapshot = (
            ConcatenatedRecordingSelection.MemberSnapshot & concat_key
        ).fetch(as_dict=True, order_by="member_index")
        boundaries = (ConcatenatedRecording.MemberBoundary & concat_key).fetch(
            as_dict=True, order_by="member_index"
        )
        if [int(row["member_index"]) for row in snapshot] != [
            int(row["member_index"]) for row in boundaries
        ]:
            raise exc_class(
                "UnitMatchSelection: concatenation "
                f"{source_id} of input {_input_label(sorting_id, curation_id)} "
                "has MemberBoundary rows that do not match its frozen members."
            )
        # Every member Recording must still exist with the content hash its
        # MemberSnapshot froze -- the check ConcatenatedRecording.make_fetch
        # and ConcatMemberCuration.make run. So a member's content hash below
        # (the snapshot value) is also its live Recording's, for selection,
        # UnitMatch.make and the TrackedUnit readers (which read each member
        # Recording's timestamps) alike; a member repopulated with other
        # content after the concatenation was built is refused here.
        try:
            resolve_snapshot_recordings(snapshot)
        except (ConcatMemberDriftError, MissingRecordingForConcatError) as exc:
            position = (
                "" if input_index is None else f"input_index {input_index} "
            )
            raise exc_class(
                f"{context}: {position}"
                f"{_input_label(sorting_id, curation_id)} is a sort of "
                f"concatenation {source_id}, whose member recordings no "
                f"longer match its frozen members: {exc}"
            ) from exc
        recordings = []
        start_sample = 0
        for member, boundary in zip(snapshot, boundaries, strict=True):
            end_sample = int(boundary["end_sample"])
            recordings.append(
                {
                    "recording_index": int(member["member_index"]),
                    "nwb_file_name": member["nwb_file_name"],
                    "sort_group_id": int(member["sort_group_id"]),
                    "interval_list_name": member["interval_list_name"],
                    "recording_id": member["recording_id"],
                    "recording_content_hash": str(
                        member["recording_content_hash"]
                    ),
                    "start_sample": start_sample,
                    "end_sample": end_sample,
                    "valid_times": np.asarray(
                        boundary["member_valid_times"], dtype=np.float64
                    ).reshape(-1, 2),
                }
            )
            start_sample = end_sample
    return {
        "sorting_id": sorting_id,
        "curation_id": int(curation_id),
        "curation_uuid": curation_uuid,
        "source_kind": lineage.kind,
        "source_id": source_id,
        "motion_corrected_recording_id": source.traces.key.get(
            "motion_corrected_recording_id"
        ),
        "artifact_detection_id": lineage.artifact_detection_id,
        "recordings": recordings,
    }


def _session_start_times(nwb_file_names) -> dict:
    """``{nwb_file_name: Session.session_start_time}`` in one query."""
    from spyglass.common import Session

    if not nwb_file_names:
        return {}
    rows = (
        Session & [{"nwb_file_name": name} for name in nwb_file_names]
    ).fetch("nwb_file_name", "session_start_time", as_dict=True)
    return {row["nwb_file_name"]: row["session_start_time"] for row in rows}


def _session_start_mismatches(recording_rows, live_start_times) -> list[str]:
    """Describe frozen session start times that differ from ``Session``.

    Compares each ``UnitMatchSelection.InputRecording`` row's frozen
    ``session_start_time`` with the live ``Session.session_start_time`` of
    its ``nwb_file_name`` (both read as UTC). An input's
    ``input_start_time`` is checked separately to be the earliest of its
    frozen session times, so it is covered too.

    Parameters
    ----------
    recording_rows : iterable of dict
        ``InputRecording`` rows (``recording_index``, ``nwb_file_name``,
        ``session_start_time``).
    live_start_times : dict
        :func:`_session_start_times` of the rows' sessions.

    Returns
    -------
    list of str
        One message per differing or missing session; empty when every
        frozen time equals the live one.
    """
    from spyglass.spikesorting.v2._matching.graph import utc_datetime

    mismatches = []
    for row in recording_rows:
        name = row["nwb_file_name"]
        label = f"recording {row['recording_index']} ({name})"
        frozen = utc_datetime(row["session_start_time"])
        if name not in live_start_times:
            mismatches.append(f"{label} has no Session row")
        elif utc_datetime(live_start_times[name]) != frozen:
            mismatches.append(
                f"{label} session_start_time frozen {frozen.isoformat()}, "
                f"now {utc_datetime(live_start_times[name]).isoformat()}"
            )
    return mismatches


def _add_single_recording_frames(item) -> None:
    """Add a single-recording input's frames and kept intervals, in place.

    Sets the recording's ``start_sample`` (0), ``end_sample`` (the persisted
    traces' frame count) and ``valid_times``; a concatenation input already
    carries its members' values from :func:`_resolve_match_input`.

    Parameters
    ----------
    item : dict
        A :func:`_resolve_match_input` result.
    """
    if item["source_kind"] != "recording":
        return
    recording = item["recordings"][0]
    recording["start_sample"] = 0
    recording["end_sample"] = _recording_n_samples(recording["recording_id"])
    recording["valid_times"] = _recording_valid_times(
        recording["recording_id"],
        recording["nwb_file_name"],
        item["artifact_detection_id"],
    )


def _recording_n_samples(recording_id) -> int:
    """Frame count of a single recording's persisted traces.

    A sort of the recording, or of its motion correction (same frames), sees
    frames ``[0, n_samples)``.
    """
    from spyglass.spikesorting.v2.recording import Recording

    return int(
        Recording()
        .get_recording({"recording_id": recording_id})
        .get_num_samples()
    )


def _recording_valid_times(recording_id, nwb_file_name, artifact_detection_id):
    """Kept intervals of a single recording on its own clock, in seconds.

    The same intervals the sort records as its observation times
    (``Sorting.make_fetch`` / ``_units_nwb``): the artifact-removed valid
    times when the sort pins an artifact detection, else the recorded chunks
    of the persisted traces (gaps between disjoint intervals kept) -- the
    rule ``ConcatenatedRecording`` uses for ``member_valid_times``.

    Returns
    -------
    numpy.ndarray, shape (n_intervals, 2)
    """
    import numpy as np

    if artifact_detection_id is not None:
        from spyglass.spikesorting.v2._artifacts.readers import (
            read_recording_artifact_valid_times,
        )

        valid_times = read_recording_artifact_valid_times(
            artifact_detection_id,
            nwb_file_name,
            caller="UnitMatchSelection.insert_inputs",
        )
    else:
        from spyglass.spikesorting.v2._storage.units_nwb import (
            _base_intervals_from_recording,
        )
        from spyglass.spikesorting.v2.recording import Recording

        recording = Recording().get_recording({"recording_id": recording_id})
        valid_times = _base_intervals_from_recording(
            recording, recording.get_sampling_frequency()
        )
    return np.asarray(valid_times, dtype=np.float64).reshape(-1, 2)


def _check_input_sessions(inputs, exc_class) -> None:
    """Reject a multi-day concatenation input or inputs sharing a session.

    Parameters
    ----------
    inputs : list of dict
        Each with ``sorting_id``, ``curation_id``, ``source_kind`` and
        ``recordings`` (``nwb_file_name``, ``session_start_time``).
    exc_class : type
        Exception raised for a multi-day concatenation input. Shared
        sessions raise ``SameSessionMatchError``.
    """
    from spyglass.spikesorting.v2._matching.graph import (
        assert_disjoint_input_sessions,
    )
    from spyglass.spikesorting.v2.session_group import (
        distinct_recording_dates,
    )

    for item in inputs:
        if item["source_kind"] == "recording":
            continue
        dates = distinct_recording_dates(
            recording["session_start_time"] for recording in item["recordings"]
        )
        if len(dates) > 1:
            raise exc_class(
                "UnitMatchSelection: concatenation input "
                f"{_input_label(item['sorting_id'], item['curation_id'])} "
                f"spans {len(dates)} recording dates ({dates}); a matching "
                "input must lie within one day. Match the days as separate "
                "inputs instead."
            )
    assert_disjoint_input_sessions(
        {
            _input_label(item["sorting_id"], item["curation_id"]): [
                recording["nwb_file_name"] for recording in item["recordings"]
            ]
            for item in inputs
        }
    )


def _input_part_rows(ordered) -> tuple[list[dict], list[dict]]:
    """Build the ``Input`` / ``InputRecording`` part rows (without the PK).

    Parameters
    ----------
    ordered : list of dict
        Resolved inputs carrying ``input_index`` and ``input_start_time``;
        each recording carries ``session_start_time``, ``start_sample``,
        ``end_sample`` and ``valid_times`` (``None`` until read).

    Returns
    -------
    tuple of (list of dict, list of dict)
    """
    input_rows = []
    recording_rows = []
    for item in ordered:
        input_rows.append(
            {
                "input_index": item["input_index"],
                "sorting_id": item["sorting_id"],
                "curation_id": item["curation_id"],
                "curation_uuid": item["curation_uuid"],
                "source_kind": item["source_kind"],
                "source_id": item["source_id"],
                "motion_corrected_recording_id": item[
                    "motion_corrected_recording_id"
                ],
                "input_start_time": item["input_start_time"],
            }
        )
        for recording in item["recordings"]:
            recording_rows.append(
                {
                    "input_index": item["input_index"],
                    "recording_index": recording["recording_index"],
                    "nwb_file_name": recording["nwb_file_name"],
                    "sort_group_id": recording["sort_group_id"],
                    "interval_list_name": recording["interval_list_name"],
                    "recording_id": recording["recording_id"],
                    "recording_content_hash": recording[
                        "recording_content_hash"
                    ],
                    "session_start_time": recording["session_start_time"],
                    "start_sample": recording["start_sample"],
                    "end_sample": recording["end_sample"],
                    "valid_times": recording.get("valid_times"),
                }
            )
    return input_rows, recording_rows


def _snapshot_mismatches(input_row, recording_rows, live) -> list[str]:
    """Compare an input's frozen rows with its live resolution.

    Parameters
    ----------
    input_row : dict
        The frozen ``UnitMatchSelection.Input`` row.
    recording_rows : list of dict
        Its frozen ``InputRecording`` rows in ``recording_index`` order.
    live : dict
        :func:`_resolve_match_input` for the same curation now. Frames and
        kept intervals are compared for every recording that carries them
        (concatenation members always; a single recording once
        :func:`_add_single_recording_frames` has read its traces).

    Returns
    -------
    list of str
        One message per differing field; empty when the live state matches.
    """
    import numpy as np

    def _text(value):
        return None if value is None else str(value)

    mismatches = []
    for field in (
        "curation_uuid",
        "source_kind",
        "source_id",
        "motion_corrected_recording_id",
    ):
        if _text(input_row[field]) != _text(live[field]):
            mismatches.append(
                f"{field} frozen {_text(input_row[field])}, now "
                f"{_text(live[field])}"
            )
    frozen_indexes = [int(row["recording_index"]) for row in recording_rows]
    live_indexes = [
        int(recording["recording_index"]) for recording in live["recordings"]
    ]
    if frozen_indexes != live_indexes:
        mismatches.append(
            f"recordings frozen {frozen_indexes}, now {live_indexes}"
        )
        return mismatches
    fields = [
        "nwb_file_name",
        "sort_group_id",
        "interval_list_name",
        "recording_id",
        "recording_content_hash",
    ]
    for frozen, current in zip(recording_rows, live["recordings"], strict=True):
        with_frames = "valid_times" in current
        for field in fields + (
            ["start_sample", "end_sample"] if with_frames else []
        ):
            if _text(frozen[field]) != _text(current[field]):
                mismatches.append(
                    f"recording {frozen['recording_index']} {field} frozen "
                    f"{_text(frozen[field])}, now {_text(current[field])}"
                )
        if with_frames and not np.array_equal(
            np.asarray(frozen["valid_times"], dtype=np.float64).reshape(-1, 2),
            current["valid_times"],
        ):
            mismatches.append(
                f"recording {frozen['recording_index']} valid_times changed"
            )
    return mismatches


def normalize_curation_choices(curation_choices) -> dict[int, tuple]:
    """``{member_index: {sorting_id, curation_id}}`` -> ``{int: (sid, int)}``.

    Caller-supplied ids go through the lossless integer rule (a fractional
    or boolean member index / curation id is rejected, not truncated).
    """
    return {
        lossless_int(idx, "member_index"): (
            choice["sorting_id"],
            lossless_int(choice["curation_id"], f"member {idx} curation_id"),
        )
        for idx, choice in curation_choices.items()
    }


def _validate_member_curations(members, choices_by_member) -> None:
    """Validate per-member curation coverage + ownership for a group.

    Raises ``ValueError`` when the choices do not exactly cover the group's
    members, when a chosen curation does not exist, or when a chosen curation
    is not a sort of the member's own recording (another member's sort, or a
    concatenation sort -- pass those to ``insert_inputs``).
    """
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.recording import RecordingSelection
    from spyglass.spikesorting.v2.sorting import SortingSelection

    member_indices = {int(member["member_index"]) for member in members}
    chosen = set(choices_by_member)
    missing = member_indices - chosen
    extra = chosen - member_indices
    if missing or extra:
        raise ValueError(
            "UnitMatchSelection: per-member curation choices must exactly cover "
            f"the group's members. Missing member_index {sorted(missing)}; "
            f"extra member_index {sorted(extra)}."
        )
    for member in members:
        member_index = int(member["member_index"])
        sorting_id, curation_id = choices_by_member[member_index]
        if not (
            CurationV2 & {"sorting_id": sorting_id, "curation_id": curation_id}
        ):
            raise ValueError(
                f"UnitMatchSelection: member_index {member_index} pins curation "
                f"(sorting_id={sorting_id}, curation_id={curation_id}) that does "
                "not exist."
            )
        member_identity = (
            str(member["nwb_file_name"]),
            int(member["sort_group_id"]),
            str(member["interval_list_name"]),
            str(member["team_name"]),
        )
        source = SortingSelection.resolve_source({"sorting_id": sorting_id})
        if source.kind != "recording":
            raise ValueError(
                f"UnitMatchSelection: member_index {member_index} "
                f"({member_identity}) was pinned to curation "
                f"(sorting_id={sorting_id}, curation_id={curation_id}), a "
                "concatenation sort, not a sort of the member's recording. "
                "Match concatenation sorts with "
                "UnitMatchSelection.insert_inputs()."
            )
        nwb_file_name, sort_group_id, interval_list_name, team_name = (
            RecordingSelection & source.key
        ).fetch1(
            "nwb_file_name", "sort_group_id", "interval_list_name", "team_name"
        )
        curation_identity = (
            str(nwb_file_name),
            int(sort_group_id),
            str(interval_list_name),
            str(team_name),
        )
        if curation_identity != member_identity:
            raise ValueError(
                f"UnitMatchSelection: member_index {member_index} "
                f"({member_identity}) was pinned to a curation that belongs to "
                f"{curation_identity}. A curation from another member cannot be "
                "pinned here."
            )

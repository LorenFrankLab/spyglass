"""Readers over a populated ``UnitMatch`` run.

:func:`get_input_provenance` is the body of ``UnitMatch.get_input_provenance``
(the run's per-input and per-recording provenance, from the frozen selection
rows or the run's NWB). :func:`tracked_unit_rows` reads a run's frozen
matchable-unit universe, its ``Pair`` graph and per-recording spike counts
(:func:`_node_detections`) and derives the rows ``TrackedUnit.make`` inserts.
:func:`get_unit_brain_regions` and :func:`get_member_spike_times` are the bodies
of the ``TrackedUnit`` accessors; both first compare each input's live
curation and source recordings with the frozen rows
(:func:`_assert_run_input_unchanged`). ``table`` is the ``UnitMatch`` or
``TrackedUnit`` instance the method was called on.

Imports without the DB layer: the DataJoint tables are imported inside the
functions.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from spyglass.spikesorting.v2._matching.inputs import (
    _input_label,
    _resolve_match_input,
    _snapshot_mismatches,
)
from spyglass.spikesorting.v2.exceptions import (
    UnitMatchSelectionIntegrityError,
)

if TYPE_CHECKING:
    import pandas as pd


#: Columns of ``UnitMatch.get_input_provenance``'s per-input frame.
_INPUT_PROVENANCE_COLUMNS = (
    "input_index",
    "sorting_id",
    "curation_id",
    "curation_uuid",
    "source_kind",
    "source_id",
    "input_start_time",
    "waveform_traces",
    "motion_corrected",
    "motion_corrected_recording_id",
)


#: Columns shared by both forms of ``UnitMatch.get_input_provenance``'s
#: per-recording frame.
_INPUT_RECORDING_PROVENANCE_COLUMNS = (
    "input_index",
    "recording_index",
    "nwb_file_name",
    "interval_list_name",
    "recording_id",
    "session_start_time",
    "start_sample",
    "end_sample",
)


def get_input_provenance(
    table, key, *, from_nwb: bool = False
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Per-input and per-recording provenance of one match run.

    The body of ``UnitMatch.get_input_provenance`` (see its docstring for
    the columns). ``table`` is the ``UnitMatch`` instance.
    """
    from datetime import datetime

    import pandas as pd

    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2._matching.graph import utc_datetime
    from spyglass.spikesorting.v2.unit_matching import UnitMatchSelection

    run = (table & key).fetch1()
    restriction = {"unitmatch_id": run["unitmatch_id"]}

    def _uuid(value):
        return None if value in (None, "") else uuid.UUID(str(value))

    if from_nwb:
        from spyglass.spikesorting.v2._storage.matches_nwb import (
            read_input_provenance,
        )

        input_rows, recording_rows = read_input_provenance(
            AnalysisNwbfile.get_abs_path(run["analysis_file_name"])
        )
        inputs = [
            {
                **row,
                "sorting_id": _uuid(row["sorting_id"]),
                "curation_uuid": _uuid(row["curation_uuid"]),
                "source_id": _uuid(row["source_id"]),
                "input_start_time": utc_datetime(
                    datetime.fromisoformat(row["input_start_time"])
                ),
                "motion_corrected": row["waveform_traces"]
                == "motion_corrected_recording",
                "motion_corrected_recording_id": _uuid(
                    row["motion_corrected_recording_id"]
                ),
            }
            for row in input_rows
        ]
        recordings = [
            {
                **row,
                "recording_id": _uuid(row["recording_id"]),
                "session_start_time": utc_datetime(
                    datetime.fromisoformat(row["session_start_time"])
                ),
            }
            for row in recording_rows
        ]
        extra_recording_columns = []
    else:
        inputs = []
        for row in (UnitMatchSelection.Input & restriction).fetch(
            as_dict=True, order_by="input_index"
        ):
            corrected_id = _uuid(row["motion_corrected_recording_id"])
            inputs.append(
                {
                    "input_index": int(row["input_index"]),
                    "sorting_id": _uuid(row["sorting_id"]),
                    "curation_id": int(row["curation_id"]),
                    "curation_uuid": _uuid(row["curation_uuid"]),
                    "source_kind": row["source_kind"],
                    "source_id": _uuid(row["source_id"]),
                    "input_start_time": utc_datetime(row["input_start_time"]),
                    "waveform_traces": (
                        row["source_kind"]
                        if corrected_id is None
                        else "motion_corrected_recording"
                    ),
                    "motion_corrected": corrected_id is not None,
                    "motion_corrected_recording_id": corrected_id,
                }
            )
        recordings = [
            {
                "input_index": int(row["input_index"]),
                "recording_index": int(row["recording_index"]),
                "nwb_file_name": row["nwb_file_name"],
                "interval_list_name": row["interval_list_name"],
                "recording_id": _uuid(row["recording_id"]),
                "session_start_time": utc_datetime(row["session_start_time"]),
                "start_sample": int(row["start_sample"]),
                "end_sample": int(row["end_sample"]),
                "sort_group_id": int(row["sort_group_id"]),
                "recording_content_hash": row["recording_content_hash"],
                "valid_times": row["valid_times"],
            }
            for row in (UnitMatchSelection.InputRecording & restriction).fetch(
                as_dict=True, order_by=("input_index", "recording_index")
            )
        ]
        extra_recording_columns = [
            "sort_group_id",
            "recording_content_hash",
            "valid_times",
        ]
    return (
        pd.DataFrame(inputs, columns=list(_INPUT_PROVENANCE_COLUMNS)),
        pd.DataFrame(
            recordings,
            columns=list(_INPUT_RECORDING_PROVENANCE_COLUMNS)
            + extra_recording_columns,
        ),
    )


def tracked_unit_rows(key) -> tuple[list[dict], list[dict]]:
    """Derive one run's ``TrackedUnit`` master and ``Member`` rows.

    The derivation behind ``TrackedUnit.make`` (see its docstring), which
    inserts the returned rows in one transaction.

    Parameters
    ----------
    key : dict
        Primary key of one ``UnitMatch`` row.

    Returns
    -------
    master_rows : list of dict
        One ``TrackedUnit`` row per derived tracked unit.
    member_rows : list of dict
        One ``TrackedUnit.Member`` row per (tracked unit, member unit).
    """
    from spyglass.spikesorting.v2._matching.graph import (
        STRICT_POLICY,
        derive_tracked_units,
    )
    from spyglass.spikesorting.v2._params.tracking import (
        resolve_tracking_params,
    )
    from spyglass.spikesorting.v2.unit_matching import (
        MatcherParameters,
        UnitMatch,
        UnitMatchSelection,
    )

    sel = (UnitMatchSelection & key).fetch1()
    params = (
        MatcherParameters & {"matcher_params_name": sel["matcher_params_name"]}
    ).fetch1("params")
    tracking = resolve_tracking_params(params)

    # Canonicalize on read to derive_tracked_units' node identity:
    # MatchableUnit stores (sorting_id uuid, curation_id, unit_id); the graph
    # keys on (str(sorting_id), int(curation_id), int(unit_id)).
    node_universe = [
        (
            str(row["sorting_id"]),
            int(row["curation_id"]),
            int(row["unit_id"]),
        )
        for row in (UnitMatch.MatchableUnit & key).fetch(as_dict=True)
    ]
    # A populated UnitMatch always wrote a non-empty MatchableUnit snapshot
    # (make_fetch rejects an input with zero matchable units), so an empty
    # snapshot under an existing UnitMatch means the row was written by a
    # Spyglass version without the MatchableUnit part. Fail loud rather than silently deriving zero tracked
    # units (or raising obscurely on a Pair edge outside an empty universe).
    if not node_universe:
        raise ValueError(
            "TrackedUnit.make: UnitMatch row "
            f"{key} has no UnitMatch.MatchableUnit snapshot (it was written "
            "by a Spyglass version that did not record one). Re-populate "
            "UnitMatch (delete + populate) "
            "so the matchable set is recorded before deriving tracked units."
        )

    edges = [
        (
            (
                str(pair["session_a_sorting_id"]),
                int(pair["session_a_curation_id"]),
                int(pair["unit_a_id"]),
            ),
            (
                str(pair["session_b_sorting_id"]),
                int(pair["session_b_curation_id"]),
                int(pair["unit_b_id"]),
            ),
            float(pair["match_probability"]),
        )
        for pair in (UnitMatch.Pair & key).fetch(as_dict=True)
    ]

    input_by_node, detected_sessions_by_node = _node_detections(key)
    tracked = derive_tracked_units(
        node_universe,
        edges,
        threshold=tracking.tracked_unit_threshold,
        max_strict_nodes=tracking.max_strict_nodes,
        policy=STRICT_POLICY,
        input_by_node=input_by_node,
        detected_sessions_by_node=detected_sessions_by_node,
    )

    master_rows = []
    member_rows = []
    for tracked_unit_id, unit in enumerate(tracked):
        master_rows.append(
            {
                **key,
                "tracked_unit_id": tracked_unit_id,
                "n_sessions_detected": unit["n_sessions_detected"],
                "n_matching_inputs": unit["n_matching_inputs"],
                "median_match_probability": unit["median_match_probability"],
                "policy_used": unit["policy_used"],
            }
        )
        for sorting_id, curation_id, unit_id in unit["members"]:
            member_rows.append(
                {
                    **key,
                    "tracked_unit_id": tracked_unit_id,
                    "sorting_id": sorting_id,
                    "curation_id": curation_id,
                    "unit_id": unit_id,
                }
            )

    return master_rows, member_rows


def _node_detections(key) -> tuple[dict, dict]:
    """Each matchable unit's input and the sessions it was detected in.

    Reads one run's frozen ``UnitMatch.MatchableUnit``,
    ``UnitMatch.RecordingSpikeCount`` and
    ``UnitMatchSelection.InputRecording`` rows. A unit is detected in a
    recording's session (``nwb_file_name``) when it has at least one spike
    in that recording's frame span.

    Parameters
    ----------
    key : dict
        Restriction selecting one ``UnitMatch`` row.

    Returns
    -------
    input_by_node : dict
        ``{(sorting_id str, curation_id, unit_id): input_index}``.
    detected_sessions_by_node : dict
        ``{node: set of nwb_file_name}``; a unit with no spikes maps to an
        empty set.

    Raises
    ------
    ValueError
        A matchable unit has no spike counts for its input's recordings (the
        ``UnitMatch`` row was written by a Spyglass version that did not
        record them); re-populate ``UnitMatch``.
    """
    from spyglass.spikesorting.v2.unit_matching import (
        UnitMatch,
        UnitMatchSelection,
    )

    nwb_by_recording = {
        (int(row["input_index"]), int(row["recording_index"])): row[
            "nwb_file_name"
        ]
        for row in (UnitMatchSelection.InputRecording & key).fetch(
            "input_index", "recording_index", "nwb_file_name", as_dict=True
        )
    }
    counts: dict = {}
    for row in (UnitMatch.RecordingSpikeCount & key).fetch(as_dict=True):
        counts.setdefault((int(row["input_index"]), int(row["unit_id"])), {})[
            int(row["recording_index"])
        ] = int(row["n_spikes"])
    input_by_node, detected_sessions_by_node = {}, {}
    missing = []
    for row in (UnitMatch.MatchableUnit & key).fetch(as_dict=True):
        input_index = int(row["input_index"])
        node = (
            str(row["sorting_id"]),
            int(row["curation_id"]),
            int(row["unit_id"]),
        )
        per_recording = counts.get((input_index, node[2]), {})
        recordings = {
            recording_index
            for index, recording_index in nwb_by_recording
            if index == input_index
        }
        if set(per_recording) != recordings:
            missing.append(node)
            continue
        input_by_node[node] = input_index
        detected_sessions_by_node[node] = {
            nwb_by_recording[(input_index, recording_index)]
            for recording_index, n_spikes in per_recording.items()
            if n_spikes > 0
        }
    if missing:
        raise ValueError(
            f"TrackedUnit.make: UnitMatch row {key} has no "
            "UnitMatch.RecordingSpikeCount rows for every recording of "
            f"matchable units {sorted(missing)} (the row was written by a "
            "Spyglass version that did not record per-recording counts). "
            "Re-populate UnitMatch (delete + populate) "
            "before deriving tracked units."
        )
    return input_by_node, detected_sessions_by_node


def get_unit_brain_regions(table, tracked_unit_key) -> "pd.DataFrame":
    """Brain regions of tracked units' member units, per original recording.

    The body of ``TrackedUnit.get_unit_brain_regions`` (see its docstring).
    ``table`` is the ``TrackedUnit`` instance.
    """
    import datajoint as dj
    import pandas as pd

    from spyglass.spikesorting.v2._recording.unit_metadata import (
        sort_group_electrode_regions,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.unit_matching import (
        UnitMatch,
        UnitMatchSelection,
    )

    region_columns = [
        "electrode_group_name",
        "electrode_id",
        "region_name",
        "subregion_name",
        "subsubregion_name",
    ]
    member_rel = table.Member & tracked_unit_key
    members = member_rel * CurationV2.Unit.proj(
        "electrode_group_name", "electrode_id"
    )
    # Bulk-read everything the member loop looks up: the members' matchable
    # units, their per-recording spike counts, their inputs' recordings, and
    # those recordings' sort-group electrode regions.
    matchable = UnitMatch.MatchableUnit & member_rel
    input_indices: dict[tuple, list[int]] = {}
    for row in matchable.fetch(as_dict=True):
        unit = (row["unitmatch_id"], row["sorting_id"])
        unit += (int(row["curation_id"]), int(row["unit_id"]))
        input_indices.setdefault(unit, []).append(int(row["input_index"]))
    n_spikes_by_key = {
        (
            row["unitmatch_id"],
            int(row["input_index"]),
            int(row["unit_id"]),
            int(row["recording_index"]),
        ): int(row["n_spikes"])
        for row in (UnitMatch.RecordingSpikeCount & matchable).fetch(
            as_dict=True
        )
    }
    input_recordings = UnitMatchSelection.InputRecording & matchable
    recordings_by_input: dict[tuple, list[dict]] = {}
    for row in input_recordings.fetch(as_dict=True, order_by="recording_index"):
        recordings_by_input.setdefault(
            (row["unitmatch_id"], int(row["input_index"])), []
        ).append(row)
    regions_by_electrode = {
        (
            row["nwb_file_name"],
            int(row["sort_group_id"]),
            row["electrode_group_name"],
            int(row["electrode_id"]),
        ): {column: row[column] for column in region_columns}
        for row in sort_group_electrode_regions(
            input_recordings.proj("nwb_file_name", "sort_group_id")
        ).fetch("nwb_file_name", "sort_group_id", *region_columns, as_dict=True)
    }

    rows = []
    checked = set()
    for member in members.fetch(as_dict=True):
        run = {"unitmatch_id": member["unitmatch_id"]}
        unit_key = {
            "sorting_id": member["sorting_id"],
            "curation_id": int(member["curation_id"]),
            "unit_id": int(member["unit_id"]),
        }
        indices = input_indices.get(
            (run["unitmatch_id"], *unit_key.values()), []
        )
        if len(indices) != 1:
            raise dj.DataJointError(
                f"UnitMatch.MatchableUnit has {len(indices)} rows for unit "
                f"{unit_key} of run {run}; expected exactly one."
            )
        input_index = indices[0]
        input_key = {**run, "input_index": input_index}
        if (str(run["unitmatch_id"]), input_index) not in checked:
            _assert_run_input_unchanged("get_unit_brain_regions", input_key)
            checked.add((str(run["unitmatch_id"]), input_index))
        for recording in recordings_by_input.get(
            (run["unitmatch_id"], input_index), []
        ):
            electrode = {
                "electrode_group_name": member["electrode_group_name"],
                "electrode_id": int(member["electrode_id"]),
            }
            regions = regions_by_electrode.get(
                (
                    recording["nwb_file_name"],
                    int(recording["sort_group_id"]),
                    *electrode.values(),
                )
            )
            if regions is None:
                raise ValueError(
                    "TrackedUnit.get_unit_brain_regions: electrode "
                    f"{electrode} of unit {unit_key} is not in sort group "
                    f"{recording['sort_group_id']} of "
                    f"{recording['nwb_file_name']} (input_index "
                    f"{input_index}, recording_index "
                    f"{recording['recording_index']})."
                )
            n_spikes = n_spikes_by_key[
                (
                    run["unitmatch_id"],
                    input_index,
                    unit_key["unit_id"],
                    int(recording["recording_index"]),
                )
            ]
            rows.append(
                {
                    "unitmatch_id": str(member["unitmatch_id"]),
                    "tracked_unit_id": int(member["tracked_unit_id"]),
                    "input_index": input_index,
                    "recording_index": int(recording["recording_index"]),
                    "nwb_file_name": recording["nwb_file_name"],
                    "interval_list_name": recording["interval_list_name"],
                    "recording_date": recording["session_start_time"],
                    "sorting_id": str(unit_key["sorting_id"]),
                    "curation_id": unit_key["curation_id"],
                    "unit_id": unit_key["unit_id"],
                    "n_spikes": n_spikes,
                    "detected": n_spikes > 0,
                    **regions,
                }
            )
    columns = [
        "unitmatch_id",
        "tracked_unit_id",
        "input_index",
        "recording_index",
        "nwb_file_name",
        "interval_list_name",
        "recording_date",
        "sorting_id",
        "curation_id",
        "unit_id",
        "n_spikes",
        "detected",
        *region_columns,
    ]
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(
            [
                "unitmatch_id",
                "tracked_unit_id",
                "input_index",
                "unit_id",
                "recording_index",
            ]
        )
        .reset_index(drop=True)
    )


def get_member_spike_times(table, tracked_unit_key) -> "pd.DataFrame":
    """Tracked units' member spike times on each original recording's clock.

    The body of ``TrackedUnit.get_member_spike_times`` (see its docstring).
    ``table`` is the ``TrackedUnit`` instance.
    """
    import pandas as pd

    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2._recording.concat import (
        member_spike_times,
        split_spike_frames_by_spans,
    )
    from spyglass.spikesorting.v2._storage.units_nwb import (
        read_units_abs_times_and_sample_indices,
        recording_timestamps,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.recording import Recording
    from spyglass.spikesorting.v2.unit_matching import (
        UnitMatch,
        UnitMatchSelection,
    )

    timestamps_by_recording: dict = {}

    def _timestamps(recording_id):
        if recording_id not in timestamps_by_recording:
            timestamps_by_recording[recording_id] = recording_timestamps(
                (Recording & {"recording_id": recording_id}).fetch1()
            )
        return timestamps_by_recording[recording_id]

    rows = []
    checked = set()
    for member in (table.Member & tracked_unit_key).fetch(as_dict=True):
        run = {"unitmatch_id": member["unitmatch_id"]}
        curation_key = {
            "sorting_id": member["sorting_id"],
            "curation_id": int(member["curation_id"]),
        }
        unit_id = int(member["unit_id"])
        input_index = int(
            (
                UnitMatch.MatchableUnit
                & run
                & curation_key
                & {"unit_id": unit_id}
            ).fetch1("input_index")
        )
        input_key = {**run, "input_index": input_index}
        if (str(run["unitmatch_id"]), input_index) not in checked:
            _assert_run_input_unchanged("get_member_spike_times", input_key)
            checked.add((str(run["unitmatch_id"]), input_index))
        source_kind = (UnitMatchSelection.Input & input_key).fetch1(
            "source_kind"
        )
        recordings = (UnitMatchSelection.InputRecording & input_key).fetch(
            as_dict=True, order_by="recording_index"
        )
        abs_times, sample_indices, _obs = (
            read_units_abs_times_and_sample_indices(
                AnalysisNwbfile.get_abs_path(
                    (CurationV2 & curation_key).fetch1("analysis_file_name")
                ),
                unit_ids=[unit_id],
            )
        )
        if source_kind == "recording":
            times_per_recording = [abs_times[unit_id]]
        else:
            local_frames = split_spike_frames_by_spans(
                {unit_id: sample_indices[unit_id]},
                [
                    (int(r["start_sample"]), int(r["end_sample"]))
                    for r in recordings
                ],
            )
            times_per_recording = [
                member_spike_times(
                    frames,
                    _timestamps(recording["recording_id"]),
                    context=(
                        "TrackedUnit.get_member_spike_times "
                        f"(input_index {input_index}, recording_index "
                        f"{recording['recording_index']})"
                    ),
                )[unit_id]
                for recording, frames in zip(
                    recordings, local_frames, strict=True
                )
            ]
        for recording, spike_times in zip(
            recordings, times_per_recording, strict=True
        ):
            rows.append(
                {
                    "unitmatch_id": str(member["unitmatch_id"]),
                    "tracked_unit_id": int(member["tracked_unit_id"]),
                    "input_index": input_index,
                    "recording_index": int(recording["recording_index"]),
                    "nwb_file_name": recording["nwb_file_name"],
                    "interval_list_name": recording["interval_list_name"],
                    "sorting_id": str(member["sorting_id"]),
                    "curation_id": int(member["curation_id"]),
                    "unit_id": unit_id,
                    "spike_times": spike_times,
                }
            )
    columns = [
        "unitmatch_id",
        "tracked_unit_id",
        "input_index",
        "recording_index",
        "nwb_file_name",
        "interval_list_name",
        "sorting_id",
        "curation_id",
        "unit_id",
        "spike_times",
    ]
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(
            [
                "unitmatch_id",
                "tracked_unit_id",
                "input_index",
                "unit_id",
                "recording_index",
            ]
        )
        .reset_index(drop=True)
    )


def _assert_run_input_unchanged(reader: str, input_key: dict) -> None:
    """Refuse to read a run's input whose curation or sources differ.

    Runs the comparison ``UnitMatch.make_fetch`` runs before matching:
    :func:`_snapshot_mismatches` of the frozen ``Input`` /
    ``InputRecording`` rows against :func:`_resolve_match_input` now. A
    single recording's frames are not re-read from its traces (its content
    hash covers its timestamps). Database reads only.

    Parameters
    ----------
    reader : str
        The ``TrackedUnit`` method name, for the message.
    input_key : dict
        ``{"unitmatch_id", "input_index"}`` of one matching input.

    Raises
    ------
    UnitMatchSelectionIntegrityError
        The pinned curation or the input's source differs from its frozen
        rows, including a concatenation member ``Recording`` that changed or
        is gone (chained from the concatenation's drift error).
    """
    from spyglass.spikesorting.v2.unit_matching import UnitMatchSelection

    input_row = (UnitMatchSelection.Input & input_key).fetch1()
    recording_rows = (UnitMatchSelection.InputRecording & input_key).fetch(
        as_dict=True, order_by="recording_index"
    )
    live = _resolve_match_input(
        input_row["sorting_id"],
        int(input_row["curation_id"]),
        UnitMatchSelectionIntegrityError,
        context=f"TrackedUnit.{reader}",
        input_index=int(input_key["input_index"]),
    )
    mismatches = _snapshot_mismatches(input_row, recording_rows, live)
    if mismatches:
        raise UnitMatchSelectionIntegrityError(
            f"TrackedUnit.{reader}: input_index {input_key['input_index']} "
            f"{_input_label(input_row['sorting_id'], input_row['curation_id'])}"
            f" of match run {input_key['unitmatch_id']} differs from the "
            f"snapshot the run was made from: {'; '.join(mismatches)}. Its "
            "live data no longer describe the matched units. Restore the "
            "original curation and source, or sort and curate the changed "
            "data and match the new curation."
        )

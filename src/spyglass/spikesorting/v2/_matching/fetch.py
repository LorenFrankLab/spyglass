"""DB inputs for a ``UnitMatch`` populate (``UnitMatch.make_fetch``).

DataJoint tables are imported inside the functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from spyglass.spikesorting.v2._matching.inputs import (
    _add_single_recording_frames,
    _check_input_count_and_sortings,
    _check_input_curation,
    _check_input_sessions,
    _check_selection_universe,
    _input_label,
    _resolve_match_input,
    _session_start_mismatches,
    _session_start_times,
    _snapshot_mismatches,
    _validate_frozen_source_coverage,
    _validate_frozen_unit_selections,
)
from spyglass.spikesorting.v2.exceptions import (
    UnitMatchSelectionIntegrityError,
)

if TYPE_CHECKING:
    from spyglass.spikesorting.v2.unit_matching import UnitMatchFetched


def fetch_unit_match_inputs(key) -> UnitMatchFetched:
    """Fetch and re-validate a match run's frozen inputs.

    The body of ``UnitMatch.make_fetch`` (see its docstring for the checks).

    Parameters
    ----------
    key : dict
        Primary key restricting to one ``UnitMatchSelection`` row.

    Returns
    -------
    UnitMatchFetched
        The matcher recipe, the ``input_index``-ordered ``input_plan`` and the
        selection provenance for the compute step.
    """
    from spyglass.spikesorting.v2._matching.graph import (
        frozen_order_errors,
        input_part_structure_errors,
        input_set_hash,
        utc_datetime,
    )
    from spyglass.spikesorting.v2.sorting import SortingSelection
    from spyglass.spikesorting.v2.unit_matching import (
        MatcherParameters,
        UnitMatchFetched,
        UnitMatchSelection,
    )

    exc_class = UnitMatchSelectionIntegrityError
    sel = (UnitMatchSelection & key).fetch1()
    input_rows = (UnitMatchSelection.Input & key).fetch(
        as_dict=True, order_by="input_index"
    )
    recording_rows = (UnitMatchSelection.InputRecording & key).fetch(
        as_dict=True, order_by=("input_index", "recording_index")
    )
    structure_errors = input_part_structure_errors(input_rows, recording_rows)
    if structure_errors:
        raise exc_class(
            f"UnitMatch.make: selection {key} has malformed Input / "
            f"InputRecording parts ({'; '.join(structure_errors)}). A "
            "direct insert bypassed UnitMatchSelection.insert_inputs()."
        )
    # Before the snapshot check, which would mask a missing/preview curation.
    for row in input_rows:
        _check_input_curation(
            row["sorting_id"], int(row["curation_id"]), exc_class
        )
    _validate_frozen_unit_selections(input_rows, exc_class)
    _validate_frozen_source_coverage(recording_rows, exc_class)
    recordings_by_input: dict[int, list[dict]] = {}
    for row in recording_rows:
        recordings_by_input.setdefault(int(row["input_index"]), []).append(row)
    _check_input_count_and_sortings(
        [(row["sorting_id"], row["curation_id"]) for row in input_rows],
        exc_class,
    )
    _check_input_sessions(
        [
            {
                **row,
                "recordings": recordings_by_input[int(row["input_index"])],
            }
            for row in input_rows
        ],
        exc_class,
    )
    order_errors = frozen_order_errors(input_rows, recording_rows)
    if order_errors:
        raise exc_class(
            f"UnitMatch.make: selection {key} has frozen start times or "
            f"input numbering that disagree ({'; '.join(order_errors)}). "
            "Use UnitMatchSelection.insert_inputs()."
        )
    recomputed_hash = input_set_hash(input_rows, recording_rows)
    if recomputed_hash != sel["input_set_hash"]:
        raise exc_class(
            "UnitMatch.make: the selection's stored input_set_hash "
            f"{sel['input_set_hash']} does not match the hash recomputed "
            f"from its Input / InputRecording rows ({recomputed_hash}). "
            "The master and its inputs were not created together by "
            "insert_inputs (a raw-insert bypass that lets a master claim "
            "one input set while matching on another). Use "
            "UnitMatchSelection.insert_inputs()."
        )
    # A single input extracts no bundle and reads no traces file.
    extracts_bundles = len(input_rows) >= 2
    # A changed live session time refuses the run; it never re-orders it.
    live_start_times = _session_start_times(
        {row["nwb_file_name"] for row in recording_rows}
    )
    for row in input_rows:
        sorting_id, curation_id = row["sorting_id"], int(row["curation_id"])
        live = _resolve_match_input(
            sorting_id,
            curation_id,
            exc_class,
            context="UnitMatch.make",
            input_index=int(row["input_index"]),
        )
        if extracts_bundles:
            _add_single_recording_frames(live)
            from spyglass.spikesorting.v2._matching.coverage import (
                enrich_recording_source_coverage,
            )

            enrich_recording_source_coverage(
                live["recordings"], allow_rebuild=True
            )
        frozen_recordings = recordings_by_input[int(row["input_index"])]
        mismatches = _snapshot_mismatches(
            row, frozen_recordings, live
        ) + _session_start_mismatches(frozen_recordings, live_start_times)
        if mismatches:
            raise exc_class(
                f"UnitMatch.make: input_index {row['input_index']} "
                f"{_input_label(sorting_id, curation_id)} no longer "
                f"matches its frozen snapshot: {'; '.join(mismatches)}. "
                "The curation was recreated or its source changed after "
                "the selection was made; select the inputs again with "
                "UnitMatchSelection.insert_inputs()."
            )
    UnitMatchSelection._warn_on_divergent_electrode_space(
        {
            int(row["input_index"]): (row["sorting_id"], row["curation_id"])
            for row in input_rows
        }
    )

    matcher_name, params, job_kwargs = (
        MatcherParameters & {"matcher_params_name": sel["matcher_params_name"]}
    ).fetch1("matcher", "params", "job_kwargs")

    # Resolved here so a relabel before compute cannot change matchable
    # units. Times are UTC ISO strings for a stable DeepHash.
    input_plan = []
    input_curation_keys = []
    for row in input_rows:
        input_index = int(row["input_index"])
        sorting_id, curation_id = row["sorting_id"], int(row["curation_id"])
        curation_key = {
            "sorting_id": sorting_id,
            "curation_id": curation_id,
        }
        recordings = [
            {
                "recording_index": int(recording["recording_index"]),
                "nwb_file_name": recording["nwb_file_name"],
                "sort_group_id": int(recording["sort_group_id"]),
                "interval_list_name": recording["interval_list_name"],
                "recording_id": str(recording["recording_id"]),
                "recording_content_hash": recording["recording_content_hash"],
                "session_start_time": utc_datetime(
                    recording["session_start_time"]
                ).isoformat(),
                "start_sample": int(recording["start_sample"]),
                "end_sample": int(recording["end_sample"]),
                "source_coverage": recording["source_coverage"],
                "valid_times": [
                    [float(start), float(stop)]
                    for start, stop in recording["valid_times"]
                ],
            }
            for recording in recordings_by_input[input_index]
        ]
        snapshot = row["selection_snapshot"]
        matchable = [int(u) for u in snapshot["included_unit_ids"]]
        _check_selection_universe(
            curation_key,
            snapshot,
            exc_class,
            "Frozen unit selection universe differs from the pinned curation.",
        )
        if not matchable:
            raise ValueError(
                f"UnitMatch.make: input_index {input_index} "
                f"{_input_label(sorting_id, curation_id)} has no matchable "
                "units under its frozen selection policy; a matcher "
                "cannot run on an empty input. Review the population selection "
                "and create a new match selection, or drop the input."
            )
        source = SortingSelection.resolve_effective_source(
            {"sorting_id": sorting_id}
        )
        input_plan.append(
            {
                "input_index": input_index,
                "sorting_id": str(sorting_id),
                "curation_id": curation_id,
                "curation_uuid": str(row["curation_uuid"]),
                "source_kind": row["source_kind"],
                "source_id": str(row["source_id"]),
                "input_start_time": utc_datetime(
                    row["input_start_time"]
                ).isoformat(),
                "recordings": recordings,
                "matchable_unit_ids": matchable,
                "selection_snapshot": snapshot,
                **_member_waveform_traces(source.traces),
            }
        )
        input_curation_keys.append(curation_key)
    # Resolve files only after every input passed its checks.
    for plan, curation_key in zip(input_plan, input_curation_keys, strict=True):
        if len(input_plan) >= 2:
            plan.update(_member_match_files(curation_key))
        else:
            plan["units"] = _input_stored_units(curation_key)
    return UnitMatchFetched(
        matcher_name=matcher_name,
        params=dict(params),
        job_kwargs=dict(job_kwargs or {}),
        input_plan=input_plan,
        session_group_owner=sel["session_group_owner"],
        session_group_name=sel["session_group_name"],
        matcher_params_name=sel["matcher_params_name"],
    )


def _member_waveform_traces(traces) -> dict:
    """Name the traces an input's matcher waveforms are extracted from.

    Records whether the waveforms came from motion-corrected or original
    traces.

    Parameters
    ----------
    traces : EffectiveTraces
        The input sort's effective traces.

    Returns
    -------
    dict
        ``{"waveform_traces": <effective traces kind>,
        "motion_corrected_recording_id": <str id or None>}``.
    """
    corrected_id = traces.key.get("motion_corrected_recording_id")
    return {
        "waveform_traces": traces.kind,
        "motion_corrected_recording_id": (
            None if corrected_id is None else str(corrected_id)
        ),
    }


def _member_match_files(curation_key: dict) -> dict:
    """Resolve the files an input's bundle is read from, for a DB-free read.

    - Traces: the sort's effective traces as every analyzer rebuild resolves
      them (``resolve_canonical_recording``), rebuilt if missing.
    - Units: the curated units NWB, read against the sort's clock.
    - Statistics spans: bound the bundle's waveform windows.

    Parameters
    ----------
    curation_key : dict
        ``{"sorting_id", "curation_id"}`` of the input's pinned curation.

    Returns
    -------
    dict
        ``{"sorting_input": CanonicalRecording, "units": StoredUnits,
        "statistics_spans": list of [start, end]}``.
    """
    from spyglass.spikesorting.v2._sorting.analyzer import (
        resolve_canonical_recording,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection

    sorting_key = {"sorting_id": curation_key["sorting_id"]}
    sorting_input = resolve_canonical_recording(sorting_key)
    return {
        "sorting_input": sorting_input,
        "units": SortingSelection.resolve_stored_units(
            (CurationV2 & curation_key).fetch1("analysis_file_name"),
            sorting_input.source,
        ),
        "statistics_spans": [
            [int(start), int(end)]
            for start, end in Sorting().get_statistics_spans(sorting_key)
        ],
    }


def _input_stored_units(curation_key: dict):
    """Resolve an input's curated units NWB without touching its traces.

    For a single-input selection: the units file stores each spike's sort
    frame, so the traces file is neither rebuilt nor checksummed.

    Parameters
    ----------
    curation_key : dict
        ``{"sorting_id", "curation_id"}`` of the input's pinned curation.

    Returns
    -------
    StoredUnits
        For :func:`._storage.units_nwb.read_stored_units`.
    """
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import SortingSelection

    source = SortingSelection.resolve_effective_source(
        {"sorting_id": curation_key["sorting_id"]}
    )
    units_file = (CurationV2 & curation_key).fetch1("analysis_file_name")

    return SortingSelection.resolve_stored_units(units_file, source)

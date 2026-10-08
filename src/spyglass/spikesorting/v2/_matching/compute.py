"""Compute steps of ``UnitMatch.make_compute`` that need no database.

:func:`extract_and_match` is the body of ``UnitMatch._extract_and_match``: it
extracts a waveform bundle per matching input from the files ``make_fetch``
resolved, runs the matcher in chronological input order and canonicalizes the
pairs. :func:`_input_recording_spike_counts` counts each matchable unit's
spikes per constituent recording, and :func:`unit_match_provenance_tables`
builds the provenance tables written beside the pairs table.
``make_compute`` stays on ``UnitMatch`` and keeps staging the pairs NWB.

Imports without the DB layer, and no function queries the database or stages
a file: the only files read are those named in ``UnitMatchFetched.input_plan``.
"""

from __future__ import annotations

import time

from spyglass.spikesorting.v2._matching.inputs import _input_label


def extract_and_match(input_plan, matcher_name, params, job_kwargs):
    """Extract per-input bundles, run the matcher, canonicalize the pairs.

    Returns ``(oriented_pairs, runtime_s)``. The wrapper extracts dense
    split-half waveform bundles from each input's curated, matchable
    sorting + recording (resolving ``MatcherParameters.job_kwargs`` into the
    analyzer compute calls) and feeds the matcher self-contained directories
    in ``input_index`` order, which is chronological; the matcher never sees
    a recording, analyzer, or Spyglass key.

    Only spikes whose whole waveform window lies inside one of the sort's
    statistics spans are sampled, so no window runs across a
    concatenation join, an acquisition gap or an artifact exclusion. A
    matchable unit with fewer than two sampled spikes is left out of its
    input's bundle, so it gets no match pair; one warning per input names
    those units. They stay in the frozen matchable universe
    (``make_insert`` writes ``MatchableUnit`` from the plan, not the
    bundles) and become unmatched tracked units.

    Raises
    ------
    NoMatchableUnitsError
        Every matchable unit of an input was left out of its bundle; the
        message names the input.
    """
    import tempfile
    from pathlib import Path

    from spyglass.settings import temp_dir as spyglass_temp_dir
    from spyglass.spikesorting.v2._core.lookup_validation import lossless_int
    from spyglass.spikesorting.v2._matching.graph import (
        canonicalize_match_pairs,
    )
    from spyglass.spikesorting.v2._sorting.analyzer import (
        read_canonical_recording,
    )
    from spyglass.spikesorting.v2._storage.units_nwb import read_stored_units
    from spyglass.spikesorting.v2._matching.waveforms import (
        NoMatchableUnitsError,
    )
    from spyglass.spikesorting.v2.matcher_protocol import (
        MatcherInputSource,
        PreparedMatcherInput,
        SessionMatcherInput,
        get_input_preparer,
        get_matcher,
    )
    from spyglass.spikesorting.v2._core.job_config import _resolved_job_kwargs
    from spyglass.utils import logger

    def _input_description(plan):
        nwb_file_names = sorted(
            {recording["nwb_file_name"] for recording in plan["recordings"]}
        )
        return (
            f"input_index {plan['input_index']} "
            f"(sorting_id={plan['sorting_id']}, "
            f"curation_id={plan['curation_id']}, "
            f"nwb_file_name {nwb_file_names})"
        )

    matcher = get_matcher(matcher_name)
    preparer = get_input_preparer(matcher_name)
    resolved_job_kwargs = _resolved_job_kwargs(job_kwargs)
    input_index_by_curation = {
        (str(plan["sorting_id"]), int(plan["curation_id"])): plan["input_index"]
        for plan in input_plan
    }
    # Feed the matcher in input_index order, the chronological order frozen
    # at selection: UnitMatch's drift correction aligns each session to the
    # previous one, so an out-of-chronology order would mis-align drift.
    # Pair orientation (side a = lower input_index) is applied by
    # canonicalize_match_pairs below.
    ordered_plan = sorted(input_plan, key=lambda plan: plan["input_index"])
    with tempfile.TemporaryDirectory(
        prefix="unitmatch_", dir=spyglass_temp_dir
    ) as tmp_root:
        session_inputs = []
        eligible_unit_ids_by_curation = {}
        for plan in ordered_plan:
            # Build the SI objects (NWB I/O) here from the files make_fetch
            # resolved; the matchable unit set was already resolved +
            # validated there and threaded in via the plan, so compute does
            # not re-derive curation-label state. The recording is the
            # traces the sorter read, opened by the DB-free half of
            # CurationV2.get_sorting_input_recording: the sort's effective
            # traces (a selected motion-corrected recording included, as
            # the plan's ``waveform_traces`` records), silenced over the
            # sort's artifact periods by the same mask the sorter input and
            # every analyzer rebuild use; the sorting is
            # CurationV2.get_sorting's.
            recording = read_canonical_recording(plan["sorting_input"])
            full_sorting = read_stored_units(plan["units"])
            sorting = full_sorting.select_units(plan["matchable_unit_ids"])
            session_dir = Path(tmp_root) / f"input_{plan['input_index']}"
            source = MatcherInputSource(
                curation_key={
                    "sorting_id": plan["sorting_id"],
                    "curation_id": plan["curation_id"],
                },
                recording=recording,
                sorting=sorting,
                recording_date=plan["input_start_time"],
                statistics_spans=plan["statistics_spans"],
            )
            try:
                prepared = preparer.prepare(
                    source, session_dir, params, resolved_job_kwargs
                )
            except NoMatchableUnitsError as exc:
                raise NoMatchableUnitsError(
                    f"UnitMatch.make: {_input_description(plan)} cannot "
                    f"prepare input for matcher {matcher_name!r}: {exc.reason}"
                ) from exc
            if not isinstance(prepared, PreparedMatcherInput) or not isinstance(
                prepared.session_input, SessionMatcherInput
            ):
                raise TypeError(
                    "Matcher input preparers must return PreparedMatcherInput."
                )
            if (
                prepared.session_input.curation_key
                != {
                    "sorting_id": plan["sorting_id"],
                    "curation_id": plan["curation_id"],
                }
                or prepared.session_input.recording_date
                != plan["input_start_time"]
            ):
                raise ValueError(
                    "Matcher input preparation must preserve the frozen curation identity and date."
                )
            excluded = [
                lossless_int(unit, "excluded_unit_id")
                for unit in prepared.excluded_unit_ids
            ]
            if not set(excluded).issubset(plan["matchable_unit_ids"]):
                raise ValueError(
                    "Matcher input preparer excluded units outside the frozen matchable universe."
                )
            eligible_unit_ids_by_curation[
                (str(plan["sorting_id"]), int(plan["curation_id"]))
            ] = set(plan["matchable_unit_ids"]) - set(excluded)
            if excluded:
                reason = prepared.exclusion_reason or (
                    f"were excluded by input preparation for matcher {matcher_name!r}"
                )
                logger.warning(
                    f"UnitMatch.make: {_input_description(plan)}: units "
                    f"{excluded} {reason} and will have no match pairs; "
                    "they remain in the matchable universe as unmatched "
                    "units."
                )
            session_inputs.append(prepared.session_input)
        start = time.perf_counter()
        raw_pairs = matcher.match(session_inputs, params)
        runtime_s = time.perf_counter() - start
    oriented_pairs = canonicalize_match_pairs(
        raw_pairs,
        input_index_by_curation,
        eligible_unit_ids_by_curation=eligible_unit_ids_by_curation,
    )
    return oriented_pairs, runtime_s


def _input_recording_spike_counts(plan: dict) -> list[dict]:
    """Count each matchable unit's spikes per constituent recording; no DB.

    Reads the input's curated units (sort frames) from ``plan["units"]``
    and splits every matchable unit's spikes by the recordings' frozen
    ``[start_sample, end_sample)`` spans (:func:`._matcher_graph.
    count_recording_spikes`), which conserves every spike.

    Parameters
    ----------
    plan : dict
        One ``UnitMatchFetched.input_plan`` entry.

    Returns
    -------
    list[dict]
        ``{"input_index", "recording_index", "unit_id", "n_spikes"}`` per
        (constituent recording, matchable unit).

    Raises
    ------
    ConcatSplitError
        A spike of a matchable unit lies outside every frozen span of its
        input's recordings.
    """
    from spyglass.spikesorting.v2._matching.graph import count_recording_spikes
    from spyglass.spikesorting.v2._storage.units_nwb import read_stored_units
    from spyglass.spikesorting.v2.exceptions import ConcatSplitError

    sorting = read_stored_units(plan["units"])
    trains = {
        int(unit_id): sorting.get_unit_spike_train(unit_id=unit_id)
        for unit_id in plan["matchable_unit_ids"]
    }
    recordings = plan["recordings"]
    try:
        counts = count_recording_spikes(
            trains,
            [
                (int(recording["start_sample"]), int(recording["end_sample"]))
                for recording in recordings
            ],
        )
    except ConcatSplitError as exc:
        raise ConcatSplitError(
            f"UnitMatch.make: input_index {plan['input_index']} "
            f"{_input_label(plan['sorting_id'], plan['curation_id'])}: its "
            "curated spikes do not fit the frozen frame spans of its "
            f"recordings ({exc})"
        ) from exc
    return [
        {
            "input_index": int(plan["input_index"]),
            "recording_index": int(recording["recording_index"]),
            "unit_id": unit_id,
            "n_spikes": int(n_spikes),
        }
        for unit_id, per_recording in counts.items()
        for recording, n_spikes in zip(recordings, per_recording, strict=True)
    ]


def unit_match_provenance_tables(
    key,
    input_plan,
    *,
    session_group_owner,
    session_group_name,
    matcher_params_name,
    matcher_backend,
    matcher_backend_version,
    spikeinterface_version,
    matcher_provenance=None,
) -> list:
    """Provenance tables that make a run's pairs NWB self-describing.

    The run/group/matcher header (re-emitting the producer provenance the
    ``UnitMatch`` row stores), the per-input map and the per-recording map,
    so the pairs table -- side ids only -- is interpretable without the DB.

    Parameters
    ----------
    key : dict
        The ``UnitMatch`` key (``unitmatch_id``).
    input_plan : list of dict
        ``UnitMatchFetched.input_plan``.
    session_group_owner, session_group_name : str or None
        The ``SessionGroup`` the inputs were discovered from.
    matcher_params_name : str
        The ``MatcherParameters`` row.
    matcher_backend, matcher_backend_version, spikeinterface_version : str
        Producer provenance (``matcher_backend_version`` may be None).
    matcher_provenance : dict or None
        Both producer class names, versions and optional asset fingerprints.

    Returns
    -------
    list of hdmf.common.DynamicTable
        For ``_unitmatch_nwb.write_pairs_table(provenance_tables=...)``.
    """
    from spyglass.spikesorting.v2._storage.provenance import (
        UNITMATCH_INPUT_COLUMNS,
        UNITMATCH_INPUT_RECORDING_COLUMNS,
        UNITMATCH_INPUT_RECORDINGS,
        UNITMATCH_INPUTS,
        UNITMATCH_PROVENANCE,
        build_long_provenance_table,
        build_provenance_table,
    )

    return [
        build_provenance_table(
            UNITMATCH_PROVENANCE,
            {
                "unitmatch_id": str(key["unitmatch_id"]),
                "session_group_owner": session_group_owner,
                "session_group_name": session_group_name,
                "matcher_params_name": matcher_params_name,
                "matcher_backend": matcher_backend,
                "matcher_backend_version": matcher_backend_version,
                "spikeinterface_version": spikeinterface_version,
                "matcher_provenance": matcher_provenance,
            },
        ),
        build_long_provenance_table(
            UNITMATCH_INPUTS,
            [
                {
                    "input_index": int(plan["input_index"]),
                    "sorting_id": str(plan["sorting_id"]),
                    "curation_id": int(plan["curation_id"]),
                    "curation_uuid": str(plan["curation_uuid"]),
                    "source_kind": str(plan["source_kind"]),
                    "source_id": str(plan["source_id"]),
                    "input_start_time": str(plan["input_start_time"]),
                    "waveform_traces": str(plan["waveform_traces"]),
                    # Empty for an input whose waveforms come from its
                    # source's own traces (typed column: no None).
                    "motion_corrected_recording_id": str(
                        plan["motion_corrected_recording_id"] or ""
                    ),
                }
                for plan in input_plan
            ],
            UNITMATCH_INPUT_COLUMNS,
        ),
        build_long_provenance_table(
            UNITMATCH_INPUT_RECORDINGS,
            [
                {
                    "input_index": int(plan["input_index"]),
                    "recording_index": int(recording["recording_index"]),
                    "nwb_file_name": str(recording["nwb_file_name"]),
                    "interval_list_name": str(recording["interval_list_name"]),
                    "recording_id": str(recording["recording_id"]),
                    "session_start_time": str(recording["session_start_time"]),
                    "start_sample": int(recording["start_sample"]),
                    "end_sample": int(recording["end_sample"]),
                }
                for plan in input_plan
                for recording in plan["recordings"]
            ],
            UNITMATCH_INPUT_RECORDING_COLUMNS,
        ),
    ]

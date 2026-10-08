"""Read exposure from committed outputs, keeping interval math DB-free."""

from __future__ import annotations

import hashlib
import json
from itertools import pairwise
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from spyglass.spikesorting.v2._core.observed_time import (
    compact_observation_intervals,
    observed_intervals,
)

if TYPE_CHECKING:
    from spyglass.spikesorting.v2._recording.source import EffectiveTraces


def canonical_unit_intervals(recording, intervals_by_unit):
    """Map and store each distinct mask once; units reference its index."""
    compact = compact_observation_intervals(intervals_by_unit)
    return {
        "masks": [
            observed_intervals(recording, mask).tolist()
            for mask in compact["masks"]
        ],
        "unit_mask_ids": compact["unit_mask_ids"],
    }


def observation_fingerprint(observation):
    """Keep the original per-unit content hash without expanding shared masks."""
    encoded = [json.dumps(mask).encode() for mask in observation["masks"]]
    digest = hashlib.sha256(b"{")
    for index, (unit, mask_id) in enumerate(
        sorted(observation["unit_mask_ids"].items())
    ):
        if index:
            digest.update(b", ")
        digest.update(json.dumps(unit).encode() + b": ")
        digest.update(encoded[mask_id])
    digest.update(b"}")
    return digest.hexdigest()


def observation_metrics_from_nwb(path, recording, *, bin_duration_s):
    """Read one unit's spikes at a time; retain only metrics and small spans."""
    import pandas as pd
    from pynwb import NWBHDF5IO

    from spyglass.spikesorting.v2._core.observed_time import observed_metrics
    from spyglass.spikesorting.v2._core.signal_math import _segment_times_at

    origin = float(_segment_times_at(recording, np.array([0]))[0])
    with NWBHDF5IO(path, "r", load_namespaces=True) as io:
        units = io.read().units
        ids = list(map(int, units.id[:]))
        intervals = canonical_unit_intervals(
            recording,
            {
                uid: units["obs_intervals"][index]
                for index, uid in enumerate(ids)
            },
        )
        metrics = {
            uid: observed_metrics(
                units["spike_times"][index],
                intervals["masks"][intervals["unit_mask_ids"][str(uid)]],
                bin_duration_s=bin_duration_s,
                origin=origin,
            )
            for index, uid in enumerate(ids)
        }
    return pd.DataFrame.from_dict(
        metrics, orient="index"
    ), observation_fingerprint(intervals)


def selection_observations(merge_id, unit_ids):
    """Freeze v2 output availability without reading the NWB spike trains."""
    from pynwb import NWBHDF5IO

    from spyglass.common import AnalysisNwbfile
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput

    if not unit_ids:
        return {"masks": [], "unit_mask_ids": {}}
    parent = SpikeSortingOutput.merge_get_parent({"merge_id": merge_id})
    path = AnalysisNwbfile.get_abs_path(parent.fetch1("analysis_file_name"))
    wanted = set(unit_ids)
    with NWBHDF5IO(path, "r", load_namespaces=True) as io:
        units = io.read().units
        if "obs_intervals" not in units.colnames:
            return None
        intervals = {
            int(unit): units["obs_intervals"][index]
            for index, unit in enumerate(units.id[:])
            if unit in wanted
        }
    recording = SpikeSortingOutput.get_recording({"merge_id": merge_id})
    return canonical_unit_intervals(recording, intervals)


def cached_review_timeline(curation_key, *, cache_path, curation_uuid):
    """Reuse timeline metadata within a review pinned to a curation generation.

    The cache contains only small interval/mapping lists, never timestamps.
    File replacement is atomic so notebook inspection and HTTP workers can
    share it. A new curation generation cannot reuse an earlier one's mask.
    """
    from spyglass.spikesorting.v2._core.json_io import read_json, write_json

    cached = read_json(cache_path)
    identity = str(curation_uuid)
    if cached.get("curation_uuid") == identity:
        timeline = cached["timeline"]
        timeline["excluded"] = np.asarray(
            timeline["excluded"], dtype=float
        ).reshape(-1, 2)
        return timeline
    timeline = review_timeline(curation_key)
    write_json(
        cache_path,
        {
            "curation_uuid": identity,
            "timeline": {**timeline, "excluded": timeline["excluded"].tolist()},
        },
    )
    return timeline


class ReviewTimelineInputs(NamedTuple):
    """DB-resolved inputs of :func:`review_timeline_from_inputs`.

    Attributes
    ----------
    traces : EffectiveTraces
        The sort's effective traces, opened as persisted (no load-time mask).
    traces_path : str
        Absolute path of the traces' analysis NWB (rebuilt first if missing).
    units_path : str
        Absolute path of the curation's Units NWB.
    concatenated : bool
        Whether the sort read a concatenated recording.
    members : list of dict
        One entry per original session in timeline order, each with ``name``
        (the member's ``nwb_file_name``). A concatenated source's entries also
        carry ``traces``, the member's ``Recording.resolve_stored_traces``
        (rebuilt first if missing), and ``end_sample``, its exclusive end frame
        on the concatenated timeline.
    """

    traces: "EffectiveTraces"
    traces_path: str
    units_path: str
    concatenated: bool
    members: list


def resolve_review_timeline_inputs(curation_key) -> ReviewTimelineInputs:
    """Resolve every DB input :func:`review_timeline_from_inputs` reads.

    Rebuilds a missing traces or member ``Recording`` artifact through its
    table's verified self-heal, as ``get_recording`` would.
    """
    from spyglass.common import AnalysisNwbfile
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
    )
    from spyglass.spikesorting.v2.sorting import SortingSelection

    sorting_id, units_file = (CurationV2 & curation_key).fetch1(
        "sorting_id", "analysis_file_name"
    )
    source, traces = SortingSelection.resolve_effective_source(
        {"sorting_id": sorting_id}
    )
    traces_path = SortingSelection.ensure_effective_traces(traces)
    units_path = AnalysisNwbfile.get_abs_path(units_file)
    concatenated = source.kind == "concatenated_recording"
    if concatenated:
        from spyglass.spikesorting.v2.session_group import (
            ConcatenatedRecording,
            ConcatenatedRecordingSelection,
        )

        rows = (
            ConcatenatedRecordingSelection.MemberSnapshot
            * ConcatenatedRecording.MemberBoundary
            & source.key
        ).fetch(as_dict=True, order_by="member_index")
        members = [
            {
                "name": row["nwb_file_name"],
                "traces": Recording().resolve_stored_traces(
                    {"recording_id": row["recording_id"]}
                ),
                "end_sample": int(row["end_sample"]),
            }
            for row in rows
        ]
    else:
        members = [
            {"name": (RecordingSelection & source.key).fetch1("nwb_file_name")}
        ]
    return ReviewTimelineInputs(
        traces=traces,
        traces_path=traces_path,
        units_path=units_path,
        concatenated=concatenated,
        members=members,
    )


def review_timeline(curation_key):
    """Excluded frame spans and original-session mapping for browser inspection.

    V2 currently applies one shared observation mask to every unit. Read that
    mask directly, without materializing the NWB spike trains for a display.
    """
    return review_timeline_from_inputs(
        resolve_review_timeline_inputs(curation_key)
    )


def review_timeline_from_inputs(inputs: ReviewTimelineInputs):
    """Build :func:`review_timeline` from resolved inputs; no DB access."""
    from pynwb import NWBHDF5IO

    from spyglass.spikesorting.v2._storage.nwb import read_stored_traces
    from spyglass.spikesorting.v2._core.signal_math import (
        _segment_times_at,
        base_intervals_and_gaps,
    )
    from spyglass.spikesorting.v2._sorting.artifact_mask import (
        artifact_frame_ranges,
    )
    from spyglass.spikesorting.v2._recording.source import read_persisted_traces

    recording = read_persisted_traces(inputs.traces_path, inputs.traces)
    fs = recording.sampling_frequency
    with NWBHDF5IO(inputs.units_path, "r", load_namespaces=True) as io:
        units = io.read().units
        valid = units["obs_intervals"][0] if len(units) else None
    excluded = (
        np.asarray(
            artifact_frame_ranges(recording, valid), dtype=float
        ).reshape(-1, 2)
        / fs
        if valid is not None
        else np.empty((0, 2))
    )
    mappings = []

    def add_member(member, name, offset):
        boundaries = np.r_[
            0,
            base_intervals_and_gaps(member).gap_after + 1,
            member.get_num_samples(),
        ]
        for start, stop in pairwise(boundaries):
            original = float(_segment_times_at(member, np.array([start]))[0])
            mappings.append(
                (
                    name,
                    (offset + start) / fs,
                    (offset + stop) / fs,
                    original,
                    original + (stop - start) / fs,
                )
            )

    if inputs.concatenated:
        offset = 0
        for entry in inputs.members:
            member = read_stored_traces(entry["traces"])
            add_member(member, entry["name"], offset)
            offset = entry["end_sample"]
    else:
        add_member(recording, inputs.members[0]["name"], 0)
    return {
        "excluded": excluded,
        "mappings": mappings,
        "concatenated": inputs.concatenated,
    }

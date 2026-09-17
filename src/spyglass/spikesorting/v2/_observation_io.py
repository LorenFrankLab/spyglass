"""Read exposure from committed outputs, keeping interval math DB-free."""

from __future__ import annotations

import hashlib
import json
from itertools import pairwise

import numpy as np

from spyglass.spikesorting.v2._observed_time import (
    compact_observation_intervals,
    observed_intervals,
)


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

    from spyglass.spikesorting.v2._observed_time import observed_metrics
    from spyglass.spikesorting.v2._signal_math import _segment_times_at

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
    from spyglass.spikesorting.v2._json_io import read_json, write_json

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


def review_timeline(curation_key):
    """Excluded frame spans and original-session mapping for browser inspection.

    V2 currently applies one shared observation mask to every unit. Read that
    mask directly, without materializing the NWB spike trains for a display.
    """
    from pynwb import NWBHDF5IO

    from spyglass.common import AnalysisNwbfile
    from spyglass.spikesorting.v2._signal_math import (
        _segment_times_at,
        base_intervals_and_gaps,
    )
    from spyglass.spikesorting.v2._sorting_artifact_mask import (
        artifact_frame_ranges,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
    )
    from spyglass.spikesorting.v2.sorting import SortingSelection

    recording = CurationV2.get_recording(curation_key)
    fs = recording.sampling_frequency
    path = AnalysisNwbfile.get_abs_path(
        (CurationV2 & curation_key).fetch1("analysis_file_name")
    )
    with NWBHDF5IO(path, "r", load_namespaces=True) as io:
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
    source = SortingSelection.resolve_source(curation_key)
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

    if source.kind == "concatenated_recording":
        from spyglass.spikesorting.v2.session_group import (
            ConcatenatedRecording,
            ConcatenatedRecordingSelection,
        )

        rows = (
            ConcatenatedRecordingSelection.MemberSnapshot
            * ConcatenatedRecording.MemberBoundary
            & source.key
        ).fetch(as_dict=True, order_by="member_index")
        offset = 0
        for row in rows:
            member = Recording().get_recording(
                {"recording_id": row["recording_id"]}
            )
            add_member(member, row["nwb_file_name"], offset)
            offset = row["end_sample"]
    else:
        add_member(
            recording,
            (RecordingSelection & source.key).fetch1("nwb_file_name"),
            0,
        )
    return {
        "excluded": excluded,
        "mappings": mappings,
        "concatenated": source.kind == "concatenated_recording",
    }

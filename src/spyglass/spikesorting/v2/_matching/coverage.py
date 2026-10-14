"""Original acquisition sample coverage of independently sorted epochs.

Coverage is taken from unmasked persisted recordings, never artifact-adjusted
observation intervals. Integer ranges refer to the parent raw ElectricalSeries;
their stop is exclusive. Time ranges retain inclusive first/last sample times.
The timestamp-only scan uses bounded buffers and does not read signal traces.
"""

from __future__ import annotations

import numpy as np

from spyglass.spikesorting.v2._core.signal_math import (
    _segment_times_at,
    _segment_times_slice,
    frames_for_times,
)
from spyglass.spikesorting._numerical import integer_vector

_AMBIGUOUS_CLOCK_ERROR = (
    "Duplicate acquisition timestamps do not uniquely identify raw samples. "
    "Repair the original acquisition clock and recreate its recordings, or "
    "choose epochs with unambiguous timestamps."
)


def recording_source_coverage(recording, raw_recording, *, source_object_id):
    """Map selected original timestamps to exact, half-open raw sample ranges.

    Contiguous chunks use one binary search and a sequential timestamp read.
    A chunk crossing a selected-interval gap falls back to bounded vectorized
    binary searches. Every mapped timestamp is checked against the raw clock.
    Duplicate timestamps within a selected epoch or touching its raw range
    boundaries are rejected: timestamp equality cannot prove sample identity.
    """
    if (
        recording.get_num_segments() != 1
        or raw_recording.get_num_segments() != 1
    ):
        raise ValueError(
            "Matching source coverage requires one persisted segment."
        )
    count = int(recording.get_num_samples())
    raw_count = int(raw_recording.get_num_samples())
    if count <= 0:
        raise ValueError("Matching source coverage has no selected samples.")
    ranges, times = [], []
    previous_time = None
    chunk_size = max(1, min(65536, round(recording.get_sampling_frequency())))
    for start in range(0, count, chunk_size):
        selected = np.asarray(
            _segment_times_slice(
                recording, start, min(count, start + chunk_size)
            ),
            dtype=np.float64,
        )
        if not np.all(np.isfinite(selected)) or np.any(np.diff(selected) < 0):
            raise ValueError(
                "Matching source timestamps must be finite and ordered."
            )
        if np.any(np.diff(selected) == 0) or (
            previous_time is not None and selected[0] == previous_time
        ):
            raise ValueError(_AMBIGUOUS_CLOCK_ERROR)
        previous_time = float(selected[-1])
        first = int(frames_for_times(raw_recording, selected[:1])[0])
        stop = first + len(selected)
        if stop <= raw_count and np.array_equal(
            _segment_times_slice(raw_recording, first, stop), selected
        ):
            mapped = np.arange(first, stop, dtype=np.int64)
        else:
            mapped = frames_for_times(raw_recording, selected)
            if np.any(mapped >= raw_count) or not np.array_equal(
                _segment_times_at(raw_recording, mapped), selected
            ):
                raise ValueError(
                    "Selected recording timestamps no longer map exactly to "
                    "their original raw acquisition."
                )
        if np.any(np.diff(mapped) <= 0):
            raise ValueError(
                "Selected timestamps do not identify unique raw samples."
            )
        boundaries = np.r_[
            0, np.flatnonzero(np.diff(mapped) != 1) + 1, len(mapped)
        ]
        for begin, end in zip(boundaries[:-1], boundaries[1:], strict=True):
            lo, hi = int(mapped[begin]), int(mapped[end - 1]) + 1
            t0, t1 = float(selected[begin]), float(selected[end - 1])
            # A singleton or range boundary can select only part of a raw
            # duplicate group even when selected timestamps are unique.
            neighbors = ([lo - 1] if lo else []) + (
                [hi] if hi < raw_count else []
            )
            adjacent = _segment_times_at(raw_recording, neighbors)
            if (lo and adjacent[0] == t0) or (
                hi < raw_count and adjacent[-1] == t1
            ):
                raise ValueError(_AMBIGUOUS_CLOCK_ERROR)
            if ranges and lo < ranges[-1][1]:
                raise ValueError(
                    "Selected source sample ranges overlap or run backward."
                )
            if ranges and lo == ranges[-1][1]:
                ranges[-1][1], times[-1][1] = hi, t1
            else:
                ranges.append([lo, hi])
                times.append([t0, t1])
    return validate_source_coverage(
        {
            "version": 1,
            "source_object_id": str(source_object_id),
            "sample_ranges": ranges,
            "time_ranges": times,
        }
    )


def validate_source_coverage(coverage):
    """Validate and canonicalize JSON-safe source coverage metadata."""
    if (
        not isinstance(coverage, dict)
        or type(coverage.get("version")) is not int
        or coverage["version"] != 1
    ):
        raise ValueError(
            "Missing or unsupported matching source coverage version."
        )
    source = coverage.get("source_object_id")
    if not isinstance(source, str) or not source:
        raise ValueError("Matching source coverage requires its raw object ID.")
    ranges = np.asarray(coverage.get("sample_ranges"))
    times = np.asarray(coverage.get("time_ranges"), dtype=np.float64)
    if ranges.ndim != 2 or ranges.shape[1] != 2 or not len(ranges):
        raise ValueError(
            "Matching source coverage requires nonempty sample ranges."
        )
    ranges = integer_vector(
        ranges.reshape(-1), name="source sample ranges", nonnegative=True
    ).reshape(-1, 2)
    if times.shape != ranges.shape or not np.all(np.isfinite(times)):
        raise ValueError(
            "Source time ranges must be finite and align with sample ranges."
        )
    if np.any(ranges[:, 0] >= ranges[:, 1]) or np.any(
        ranges[1:, 0] < ranges[:-1, 1]
    ):
        raise ValueError(
            "Source sample ranges must be ordered and nonoverlapping."
        )
    if np.any(times[:, 0] > times[:, 1]) or np.any(
        times[1:, 0] < times[:-1, 1]
    ):
        raise ValueError(
            "Source time ranges must be ordered inclusive endpoints."
        )
    return {
        "version": 1,
        "source_object_id": source,
        "sample_ranges": ranges.tolist(),
        "time_ranges": times.tolist(),
    }


def stored_analysis_path(analysis_file_name):
    """Read an exact registered analysis path without directory/file fallbacks."""
    from pathlib import Path

    from spyglass.common.common_nwbfile import AnalysisNwbfile

    table = AnalysisNwbfile()
    # Restrict in SQL, then require the exact basename: underscores in generated
    # names are LIKE wildcards and must not select another registration.
    escaped = str(analysis_file_name).replace("\\", "\\\\").replace("'", "''")
    candidates = (table._ext_tbl & f"filepath LIKE '%{escaped}'").fetch(
        "filepath"
    )
    paths = [
        str(path)
        for path in candidates
        if Path(str(path)).name == analysis_file_name
    ]
    if len(paths) != 1:
        raise FileNotFoundError(
            f"Expected one registered analysis path for {analysis_file_name!r}; "
            f"found {len(paths)}. Restore its registration before planning."
        )
    return str(Path(table._analysis_dir) / paths[0])


def _base_artifact(recording_row, *, allow_rebuild):
    """Resolve a pinned artifact without reconstructing it in read-only plans."""
    from pathlib import Path

    from spyglass.spikesorting.v2.recording import Recording, RecordingSelection

    key = {"recording_id": recording_row["recording_id"]}
    row = (Recording & key).fetch1()
    selected_nwb = (RecordingSelection & key).fetch1("nwb_file_name")
    if str(selected_nwb) != str(recording_row["nwb_file_name"]) or str(
        row["content_hash"]
    ) != str(recording_row["recording_content_hash"]):
        raise ValueError(
            "Matching source coverage no longer matches its pinned recording."
        )
    if allow_rebuild:
        # Reuse the transaction's verified-path memo. A direct get_abs_path
        # would checksum large traces again during populate's second fetch.
        path = Recording().resolve_stored_traces(key).abs_path
    else:
        path = stored_analysis_path(row["analysis_file_name"])
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"Source coverage for {recording_row['nwb_file_name']!r} "
                f"({recording_row.get('interval_list_name', '')!r}) needs the "
                "missing recording cache. Rebuild it explicitly before planning."
            )
    return row, path, selected_nwb


def open_base_recording(recording_row, *, allow_rebuild=False):
    """Open the unmasked, content-pinned recording; plans never heal a cache."""
    from spyglass.spikesorting.v2._storage.nwb import read_recording_nwb

    row, path, _nwb = _base_artifact(recording_row, allow_rebuild=allow_rebuild)
    return read_recording_nwb(
        path, electrical_series_path=row["electrical_series_path"]
    )


def resolve_recording_source_coverage(recording_row, *, allow_rebuild=False):
    """Resolve exact acquisition coverage; a missing plan cache stays missing.

    ``recording_row`` contains ``recording_id``, ``nwb_file_name`` and
    ``recording_content_hash``. Only execution may opt into reconstruction.
    """
    from pathlib import Path

    from spyglass.common.common_ephys import Raw
    from spyglass.common.common_nwbfile import Nwbfile
    from spyglass.spikesorting.v2._storage.nwb import read_recording_nwb
    from spyglass.spikesorting.v2._storage.provenance import (
        RECORDING_PROVENANCE,
        read_provenance_values,
    )
    from spyglass.spikesorting.v2._storage.spikeinterface import (
        close_nwb_recording,
    )
    from spyglass.utils.nwb_helper_fn import raw_eseries_path_and_timestamp_mode

    row, path, selected_nwb = _base_artifact(
        recording_row, allow_rebuild=allow_rebuild
    )
    provenance = read_provenance_values(path, RECORDING_PROVENANCE)
    source_object_id = str(provenance["raw_object_id"])
    live_object_id = str(
        (Raw & {"nwb_file_name": selected_nwb}).fetch1("raw_object_id")
    )
    if source_object_id != live_object_id:
        raise ValueError(
            "Matching source coverage raw acquisition identity changed."
        )
    raw_path = Nwbfile.get_abs_path(selected_nwb)
    if not Path(raw_path).is_file():
        raise FileNotFoundError(
            f"Matching source coverage needs the local raw NWB {selected_nwb!r}; no download is attempted."
        )
    series_path, explicit = raw_eseries_path_and_timestamp_mode(
        raw_path, object_id=source_object_id
    )
    selected = read_recording_nwb(
        path, electrical_series_path=row["electrical_series_path"]
    )
    raw = None
    try:
        raw = read_recording_nwb(
            raw_path,
            electrical_series_path=series_path,
            load_time_vector=explicit,
        )
        return recording_source_coverage(
            selected, raw, source_object_id=source_object_id
        )
    finally:
        close_nwb_recording(selected)
        if raw is not None:
            close_nwb_recording(raw)


def enrich_recording_source_coverage(recording_rows, *, allow_rebuild=False):
    """Attach source coverage to resolved rows, reusing duplicate recording reads."""
    cache = {}
    for row in recording_rows:
        key = (str(row["recording_id"]), str(row["recording_content_hash"]))
        if key not in cache:
            cache[key] = resolve_recording_source_coverage(
                row, allow_rebuild=allow_rebuild
            )
        row["source_coverage"] = cache[key]
    return recording_rows

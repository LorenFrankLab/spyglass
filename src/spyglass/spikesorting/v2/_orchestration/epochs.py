"""Read-only epoch discovery before selecting recordings or reviewed sorts."""

from __future__ import annotations

from spyglass.spikesorting.v2._core.lookup_validation import lossless_int

_EPOCH_COLUMNS = (
    "nwb_file_name",
    "kind",
    "epoch",
    "epoch_label",
    "task_name",
    "task_epoch_key",
    "interval_list_name",
    "interval_key",
    "interval_bounds",
    "start_time",
    "stop_time",
    "duration_s",
    "interval_status",
    "interval_error",
    "recording_keys",
    "ready_recording_keys",
    "recordings",
    "recording_readiness",
)


def _interval_bounds(interval):
    """Preserve disjoint acquisition bounds; report malformed lists explicitly."""
    import numpy as np

    if interval is None:
        return (), None, None, None, "missing", "IntervalList row is missing."
    try:
        values = np.asarray(interval["valid_times"], dtype=float)
        if not values.size:
            return (
                (),
                None,
                None,
                0.0,
                "empty",
                "Interval list contains no times.",
            )
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError("valid_times must have shape (n, 2)")
        if not np.all(np.isfinite(values)) or np.any(
            values[:, 0] >= values[:, 1]
        ):
            raise ValueError("intervals must have finite start < stop")
        if np.any(values[1:, 0] < values[:-1, 1]):
            raise ValueError("intervals must be ordered and nonoverlapping")
        return (
            tuple(tuple(map(float, bound)) for bound in values),
            float(values[0, 0]),
            float(values[-1, 1]),
            float(np.sum(values[:, 1] - values[:, 0])),
            "valid",
            "",
        )
    except (ValueError, TypeError, KeyError) as exc:
        return (), None, None, None, "invalid", str(exc)


def epoch_rows_to_dataframe(nwb_file_name, tasks, intervals, recordings):
    """Pure chooser from fetched rows; preserve labels, keys, bounds, and gaps."""
    import pandas as pd

    by_name = {row["interval_list_name"]: row for row in intervals}
    tasks_by_interval = {}
    for task in tasks:
        tasks_by_interval.setdefault(task["interval_list_name"], []).append(
            task
        )
    candidates_by_interval = {}
    for recording in recordings:
        candidates_by_interval.setdefault(
            recording["interval_list_name"], []
        ).append(recording)
    rows = []
    for name in sorted(set(by_name) | set(tasks_by_interval)):
        bounds, start, stop, duration, status, error = _interval_bounds(
            by_name.get(name)
        )
        candidates = tuple(
            sorted(
                (dict(row) for row in candidates_by_interval.get(name, [])),
                key=lambda row: (
                    int(row["sort_group_id"]),
                    str(row["recording_id"]),
                ),
            )
        )
        ready = tuple(
            {"recording_id": row["recording_id"]}
            for row in candidates
            if row.get("recording_computed") and row.get("cache_present")
        )
        readiness = (
            "cache_present"
            if ready
            else (
                "cache_missing"
                if any(row.get("recording_computed") for row in candidates)
                else "not_computed" if candidates else "not_selected"
            )
        )
        for task in tasks_by_interval.get(name, [None]):
            epoch = int(task["epoch"]) if task is not None else None
            task_name = task.get("task_name", "") if task is not None else ""
            label = (
                f"epoch {epoch}: {task_name or name}"
                if task is not None
                else name
            )
            rows.append(
                {
                    "nwb_file_name": nwb_file_name,
                    "kind": (
                        "task_epoch" if task is not None else "interval_list"
                    ),
                    "epoch": epoch,
                    "epoch_label": label,
                    "task_name": task_name,
                    "task_epoch_key": (
                        {"nwb_file_name": nwb_file_name, "epoch": epoch}
                        if task is not None
                        else None
                    ),
                    "interval_list_name": name,
                    "interval_key": {
                        "nwb_file_name": nwb_file_name,
                        "interval_list_name": name,
                    },
                    "interval_bounds": bounds,
                    "start_time": start,
                    "stop_time": stop,
                    "duration_s": duration,
                    "interval_status": status,
                    "interval_error": error,
                    "recording_keys": tuple(
                        {"recording_id": row["recording_id"]}
                        for row in candidates
                    ),
                    "ready_recording_keys": ready,
                    "recordings": candidates,
                    "recording_readiness": readiness,
                }
            )
    frame = pd.DataFrame(rows, columns=list(_EPOCH_COLUMNS))
    frame["epoch"] = frame["epoch"].astype("Int64")
    return frame.sort_values(
        ["start_time", "epoch", "interval_list_name"], na_position="last"
    ).reset_index(drop=True)


def _recording_candidates(selection_rows):
    """Read computed rows and registered paths, never reconstructing a cache."""
    from pathlib import Path

    from spyglass.spikesorting.v2._matching.coverage import stored_analysis_path
    from spyglass.spikesorting.v2.recording import Recording

    if not selection_rows:
        return []
    keys = [{"recording_id": row["recording_id"]} for row in selection_rows]
    computed = {
        str(row["recording_id"]): row
        for row in (Recording & keys).fetch(as_dict=True)
    }
    candidates = []
    for selected in selection_rows:
        row = computed.get(str(selected["recording_id"]))
        candidate = {
            "recording_id": selected["recording_id"],
            "sort_group_id": int(selected["sort_group_id"]),
            "interval_list_name": selected["interval_list_name"],
            "preprocessing_params_name": selected["preprocessing_params_name"],
            "team_name": selected["team_name"],
            "recording_computed": row is not None,
            "cache_present": False,
            "analysis_file_name": (
                row["analysis_file_name"] if row is not None else None
            ),
            "cache_error": "",
        }
        if row is not None:
            try:
                candidate["cache_present"] = Path(
                    stored_analysis_path(row["analysis_file_name"])
                ).is_file()
                if not candidate["cache_present"]:
                    candidate["cache_error"] = (
                        "Recording cache is missing; rebuild it explicitly before matching."
                    )
            except (FileNotFoundError, LookupError, ValueError, OSError) as exc:
                candidate["cache_error"] = str(exc)
        candidates.append(candidate)
    return candidates


def describe_epochs(nwb_file_name: str, sort_group_id: int | None = None):
    """List exact epoch/interval choices and existing recording-cache readiness.

    TaskEpoch-linked intervals are tagged ``kind='task_epoch'`` with their
    original task key and label. Other IntervalList rows remain visible as
    ``kind='interval_list'`` so files without task metadata can still be used.
    ``interval_key`` can be copied directly into an explicit recording request.

    Bounds and start/stop times use original acquisition seconds, not a
    recording-relative or concatenated timeline. ``duration_s`` sums the
    disjoint bounds and excludes gaps. Empty, missing, and malformed interval
    lists carry explicit status/error fields.

    ``recording_keys`` lists all selected v2 recording candidates for the
    interval (optionally restricted to ``sort_group_id``); nested ``recordings``
    includes preprocessing/team identifiers and computed/cache status.
    ``ready_recording_keys`` means the computed row and registered local file
    are present. Matching preflight separately validates content, coverage,
    geometry, and eligibility. This helper never seeds defaults, inserts rows,
    downloads files, rebuilds caches, or extracts data.
    """
    selected_sort_group = (
        lossless_int(sort_group_id, "sort_group_id")
        if sort_group_id is not None
        else None
    )
    from spyglass.common.common_interval import IntervalList
    from spyglass.common.common_task import TaskEpoch
    from spyglass.spikesorting.v2.recording import RecordingSelection

    key = {"nwb_file_name": nwb_file_name}
    recording_key = dict(key)
    if selected_sort_group is not None:
        recording_key["sort_group_id"] = selected_sort_group
    return epoch_rows_to_dataframe(
        nwb_file_name,
        (TaskEpoch & key).fetch(as_dict=True, order_by="epoch"),
        (IntervalList & key).fetch(as_dict=True),
        _recording_candidates(
            (RecordingSelection & recording_key).fetch(as_dict=True)
        ),
    )

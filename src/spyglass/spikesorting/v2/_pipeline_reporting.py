"""Notebook-facing reporting helpers extracted from ``pipeline.py``.

Behavior-preserving extraction: ``describe_parameter_rows``, ``describe_units``
(+ its ``_observed_duration_s`` helper), and ``describe_run`` (+ its
``_run_*`` summary helpers) move here verbatim. ``pipeline.py`` re-exports the
public names so user import paths are unchanged. ``_run_warnings`` /
``_run_metadata`` are shared with ``run_v2_pipeline_session`` and imported
back by the run module (keeping the run -> reporting dependency acyclic).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from spyglass.spikesorting.v2._pipeline_presets import (
    _PIPELINE_PRESETS,
    describe_pipeline_presets,
)

if TYPE_CHECKING:
    import pandas as pd


_PARAMETER_ROW_COLUMNS = [
    "table",
    "parameter_name",
    "sorter",
    "probe_type",
    "sampling_rate_hz",
    "adjacency_radius_um",
    "params_schema_version",
    "fingerprint",
    "is_shipped_default",
    "recommendation_status",
    "used_by_pipeline_presets",
    "duplicate_of",
    "name_warnings",
    "summary",
]


def describe_parameter_rows() -> "pd.DataFrame":
    """Catalog the parameter-Lookup rows currently in the database.

    One row per ``PreprocessingParameters`` / ``ArtifactDetectionParameters``
    / ``SorterParameters`` row, with its content fingerprint (the row name
    excluded; ``SorterParameters`` scoped per sorter), whether it is a shipped
    catalog default, which pipeline presets reference it, and -- when its
    content duplicates another row's -- the name it duplicates. Discovery
    metadata that lives on the *presets* (``probe_type`` / ``sampling_rate_hz``
    / ``recommendation_status``) is folded down onto each row from the presets
    that use it (left blank / ``None`` when those presets disagree);
    ``adjacency_radius_um`` is read straight from the ``SorterParameters``
    blob.

    Unlike the DB-free :func:`describe_pipeline_presets`, this reads the **live
    tables**, so user-added rows appear -- call ``insert_default()`` on the
    three parameter tables first to populate the shipped catalog.

    Returns
    -------
    pandas.DataFrame
        Columns ``table``, ``parameter_name``, ``sorter``, ``probe_type``,
        ``sampling_rate_hz``, ``adjacency_radius_um``, ``params_schema_version``,
        ``fingerprint`` (short), ``is_shipped_default``,
        ``recommendation_status``, ``used_by_pipeline_presets`` (list of preset
        names), ``duplicate_of``, ``name_warnings``, and ``summary``; sorted by
        ``(table, sorter, parameter_name)``.
    """
    import pandas as pd

    from spyglass.spikesorting.v2._parameter_identity import (
        parameter_fingerprint,
        short_fingerprint,
    )
    from spyglass.spikesorting.v2.artifact import ArtifactDetectionParameters
    from spyglass.spikesorting.v2.recording import PreprocessingParameters
    from spyglass.spikesorting.v2.sorting import SorterParameters
    from spyglass.spikesorting.v2.utils import _jsonable_blob

    # Which presets reference each parameter row, per stage.
    preproc_use: dict[str, list[str]] = {}
    artifact_use: dict[str, list[str]] = {}
    sorter_use: dict[tuple[str, str], list[str]] = {}
    for preset_name, preset in _PIPELINE_PRESETS.items():
        preproc_use.setdefault(
            preset.preprocessing_params_name, []
        ).append(preset_name)
        artifact_use.setdefault(
            preset.artifact_detection_params_name, []
        ).append(preset_name)
        sorter_use.setdefault(
            (preset.sorter, preset.sorter_params_name), []
        ).append(preset_name)

    shipped_preproc = {r[0] for r in PreprocessingParameters._DEFAULT_CONTENTS}
    shipped_artifact = {
        r[0] for r in ArtifactDetectionParameters._DEFAULT_CONTENTS
    }
    shipped_sorter = {
        (r[0], r[1]) for r in SorterParameters._DEFAULT_CONTENTS
    }

    def _str_axis(used_by: list[str], attr: str) -> str:
        """Distinct non-blank preset values for ``attr``, comma-joined."""
        vals = {getattr(_PIPELINE_PRESETS[u], attr) for u in used_by}
        vals.discard("")
        vals.discard(None)
        return ", ".join(sorted(vals))

    def _num_axis(used_by: list[str], attr: str):
        """Return the single agreed preset value for a numeric ``attr``, else None."""
        vals = {getattr(_PIPELINE_PRESETS[u], attr) for u in used_by}
        vals.discard(None)
        return next(iter(vals)) if len(vals) == 1 else None

    def _num(value) -> str:
        # describe reads extra="allow" blobs, so a knob can legitimately be a
        # string / list / dict. Format numerics compactly, but fall back to
        # str() instead of letting one odd user row raise and take down the
        # whole catalog.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return str(value)
        return f"{value:g}"

    def _preproc_summary(params: dict) -> str:
        band = params.get("bandpass_filter")
        seg = params.get("min_segment_length")
        seg_str = f", min_segment {_num(seg)} s" if seg is not None else ""
        if band:
            return (
                f"bandpass {_num(band['freq_min'])}-"
                f"{_num(band['freq_max'])} Hz" + seg_str
            )
        return "no bandpass" + seg_str

    def _artifact_summary(params: dict) -> str:
        if not params.get("detect", True):
            return "artifact detection off"
        amp = params.get("amplitude_threshold_uv")
        zscore = params.get("zscore_threshold")
        prop = params.get("proportion_above_threshold")
        thresholds = []
        if amp is not None:
            thresholds.append(f"{_num(amp)} uV")
        if zscore is not None:
            thresholds.append(f"{_num(zscore)} z-score")
        threshold = " + ".join(thresholds) if thresholds else "no threshold"
        prop_str = (
            f" @ {_num(prop)} proportion-above-threshold"
            if prop is not None
            else ""
        )
        return threshold + prop_str

    def _sorter_summary(sorter: str, params: dict) -> str:
        bits = [sorter]
        threshold = params.get("detect_threshold")
        if threshold is not None:
            bits.append(f"detect_threshold {_num(threshold)}")
        radius = params.get("adjacency_radius")
        if radius is not None:
            bits.append(f"adjacency_radius {_num(radius)} um")
        return ", ".join(bits)

    records: list[dict] = []
    for row in PreprocessingParameters.fetch(as_dict=True):
        params = _jsonable_blob(row["params"])
        used = sorted(preproc_use.get(row["preprocessing_params_name"], []))
        records.append(
            {
                "table": "PreprocessingParameters",
                "parameter_name": row["preprocessing_params_name"],
                "sorter": "",
                "probe_type": _str_axis(used, "probe_type"),
                "sampling_rate_hz": _num_axis(used, "sampling_rate_hz"),
                "adjacency_radius_um": None,
                "params_schema_version": int(row["params_schema_version"]),
                "_fp": parameter_fingerprint(
                    "PreprocessingParameters",
                    params=params,
                    params_schema_version=int(row["params_schema_version"]),
                    job_kwargs=_jsonable_blob(row["job_kwargs"]),
                ),
                "is_shipped_default": (
                    row["preprocessing_params_name"] in shipped_preproc
                ),
                "recommendation_status": _str_axis(
                    used, "recommendation_status"
                ),
                "used_by_pipeline_presets": used,
                "summary": _preproc_summary(params),
            }
        )
    for row in ArtifactDetectionParameters.fetch(as_dict=True):
        params = _jsonable_blob(row["params"])
        used = sorted(
            artifact_use.get(row["artifact_detection_params_name"], [])
        )
        records.append(
            {
                "table": "ArtifactDetectionParameters",
                "parameter_name": row["artifact_detection_params_name"],
                "sorter": "",
                "probe_type": _str_axis(used, "probe_type"),
                "sampling_rate_hz": _num_axis(used, "sampling_rate_hz"),
                "adjacency_radius_um": None,
                "params_schema_version": int(row["params_schema_version"]),
                "_fp": parameter_fingerprint(
                    "ArtifactDetectionParameters",
                    params=params,
                    params_schema_version=int(row["params_schema_version"]),
                    job_kwargs=_jsonable_blob(row["job_kwargs"]),
                ),
                "is_shipped_default": (
                    row["artifact_detection_params_name"] in shipped_artifact
                ),
                "recommendation_status": _str_axis(
                    used, "recommendation_status"
                ),
                "used_by_pipeline_presets": used,
                "summary": _artifact_summary(params),
            }
        )
    for row in SorterParameters.fetch(as_dict=True):
        params = _jsonable_blob(row["params"])
        key = (row["sorter"], row["sorter_params_name"])
        used = sorted(sorter_use.get(key, []))
        radius = params.get("adjacency_radius")
        records.append(
            {
                "table": "SorterParameters",
                "parameter_name": row["sorter_params_name"],
                "sorter": row["sorter"],
                "probe_type": _str_axis(used, "probe_type"),
                "sampling_rate_hz": _num_axis(used, "sampling_rate_hz"),
                "adjacency_radius_um": (
                    float(radius) if radius is not None else None
                ),
                "params_schema_version": int(row["params_schema_version"]),
                "_fp": parameter_fingerprint(
                    "SorterParameters",
                    params=params,
                    params_schema_version=int(row["params_schema_version"]),
                    job_kwargs=_jsonable_blob(row["job_kwargs"]),
                    sorter=row["sorter"],
                ),
                "is_shipped_default": key in shipped_sorter,
                "recommendation_status": _str_axis(
                    used, "recommendation_status"
                ),
                "used_by_pipeline_presets": used,
                "summary": _sorter_summary(row["sorter"], params),
            }
        )

    # Duplicate-content detection: rows sharing a fingerprint within the same
    # (table, sorter) scope are content duplicates under different names.
    names_by_fp: dict[tuple[str, str, str], list[str]] = {}
    for rec in records:
        names_by_fp.setdefault(
            (rec["table"], rec["sorter"], rec["_fp"]), []
        ).append(rec["parameter_name"])

    for rec in records:
        group = sorted(
            names_by_fp[(rec["table"], rec["sorter"], rec["_fp"])]
        )
        others = [n for n in group if n != rec["parameter_name"]]
        rec["duplicate_of"] = others[0] if others else None
        rec["fingerprint"] = short_fingerprint(rec["_fp"])
        warnings = []
        if rec["duplicate_of"]:
            warnings.append(f"duplicate content of {rec['duplicate_of']!r}")
        if "franklab" in rec["parameter_name"] and not rec[
            "is_shipped_default"
        ]:
            warnings.append("non-catalog row using the 'franklab' name")
        rec["name_warnings"] = "; ".join(warnings)

    frame = pd.DataFrame(
        [{c: rec.get(c) for c in _PARAMETER_ROW_COLUMNS} for rec in records],
        columns=_PARAMETER_ROW_COLUMNS,
    )
    return frame.sort_values(
        ["table", "sorter", "parameter_name"]
    ).reset_index(drop=True)


_UNIT_COLUMNS = [
    "sorting_id",
    "unit_id",
    "n_spikes",
    "firing_rate_hz",
    "peak_amplitude_uv",
    "peak_electrode_id",
    "brain_region",
]


def _observed_duration_s(sorting_id) -> float:
    """Seconds the sort actually observed, for a firing-rate denominator.

    Mirrors ``Sorting.make_fetch``: when the sort ran artifact detection the
    denominator is the artifact-removed ``valid_times`` total (the segments the
    sorter actually saw), otherwise the materialized ``Recording.duration_s``.
    Using the raw recording length would overstate the denominator -- and so
    understate firing rate -- for artifact-masked sorts.
    """
    import numpy as np

    from spyglass.common.common_interval import IntervalList
    from spyglass.spikesorting.v2.recording import Recording, RecordingSelection
    from spyglass.spikesorting.v2.sorting import SortingSelection
    from spyglass.spikesorting.v2.utils import (
        artifact_detection_interval_list_name,
    )

    sorting_key = {"sorting_id": sorting_id}
    source = SortingSelection.resolve_source(sorting_key)
    if source.kind == "concatenated_recording":
        # A concat sort observes the whole concatenated recording (concat sorts
        # carry no artifact pass), so its observed duration is the materialized
        # cache's total_duration_s -- there is no per-member RecordingSelection
        # to fetch for a concat-only key.
        from spyglass.spikesorting.v2.session_group import (
            ConcatenatedRecording,
        )

        return float(
            (ConcatenatedRecording & source.key).fetch1("total_duration_s")
        )
    recording_id = source.key["recording_id"]
    artifact_detection_id = SortingSelection.resolve_artifact_detection(
        sorting_key
    )
    if artifact_detection_id is not None:
        nwb_file_name = (
            RecordingSelection & {"recording_id": recording_id}
        ).fetch1("nwb_file_name")
        valid_times = (
            IntervalList
            & {
                "nwb_file_name": nwb_file_name,
                "interval_list_name": artifact_detection_interval_list_name(
                    artifact_detection_id
                ),
            }
        ).fetch1("valid_times")
        return float(np.sum(np.diff(np.asarray(valid_times), axis=1)))
    return float(
        (Recording & {"recording_id": recording_id}).fetch1("duration_s")
    )


def describe_units(sorting_id) -> "pd.DataFrame":
    """Return a per-unit, sort-time quality snapshot for one sort.

    A read-only "what did this sort produce?" receipt built entirely from
    metadata committed at sort time -- it computes no SpikeInterface extension
    and loads no recording. Use it right after ``run_v2_pipeline``
    (``describe_units(run_summary["sorting_id"])``) to sanity-check a sort
    before the deeper analyzer-backed quality metrics (SNR / ISI / nearest-
    neighbour), which are computed by the analyzer-driven curation step
    (``AnalyzerCuration``).

    ``firing_rate_hz`` uses the duration the sort actually OBSERVED -- the
    artifact-removed ``valid_times`` total when the sort ran artifact detection,
    otherwise the materialized recording duration (see
    :func:`_observed_duration_s`). ``peak_electrode_id`` and ``brain_region``
    come from each unit's peak-amplitude ``Electrode`` row (anchored to the
    first member for concat sorts).

    Parameters
    ----------
    sorting_id
        ``sorting_id`` of a populated ``Sorting`` row (e.g.
        ``run_summary["sorting_id"]``).

    Returns
    -------
    pandas.DataFrame
        One row per unit, sorted by ``unit_id``. Empty, with the documented
        columns, for a zero-unit sort. Columns are ``sorting_id``, ``unit_id``,
        ``n_spikes``, ``firing_rate_hz``, ``peak_amplitude_uv``,
        ``peak_electrode_id`` (the unit's peak-amplitude ``electrode_id``), and
        ``brain_region``.
    """
    import pandas as pd

    from spyglass.common.common_ephys import Electrode
    from spyglass.common.common_region import BrainRegion
    from spyglass.spikesorting.v2.sorting import Sorting

    sorting_key = {"sorting_id": sorting_id}
    if not (Sorting & sorting_key):
        raise ValueError(
            f"describe_units: sorting_id {sorting_id!r} is not in Sorting. "
            "Populate the sort first (e.g. via run_v2_pipeline) and pass "
            "run_summary['sorting_id']."
        )

    unit_rows = ((Sorting.Unit & sorting_key) * Electrode * BrainRegion).fetch(
        as_dict=True
    )
    if not unit_rows:
        return pd.DataFrame(columns=_UNIT_COLUMNS)

    duration_s = _observed_duration_s(sorting_id)
    if duration_s <= 0:
        # Units present but zero observed seconds is a contradiction (an
        # all-artifact mask or a truncated recording). Raise loudly rather than
        # emitting an all-NaN firing_rate column that reads as a display quirk
        # and hides the upstream defect.
        raise ValueError(
            f"describe_units: sorting_id {sorting_id!r} has {len(unit_rows)} "
            f"unit(s) but its observed duration is {duration_s}s. A sort with "
            "units cannot span zero observed time -- check the recording "
            "duration and the artifact-detection interval."
        )
    rows = []
    for unit in sorted(unit_rows, key=lambda row: int(row["unit_id"])):
        n_spikes = int(unit["n_spikes"])
        rows.append(
            {
                "sorting_id": sorting_id,
                "unit_id": int(unit["unit_id"]),
                "n_spikes": n_spikes,
                "firing_rate_hz": n_spikes / duration_s,
                "peak_amplitude_uv": float(unit["peak_amplitude_uv"]),
                "peak_electrode_id": int(unit["electrode_id"]),
                "brain_region": str(unit["region_name"]),
            }
        )
    return pd.DataFrame(rows, columns=_UNIT_COLUMNS)


_RUN_COLUMNS = [
    "row_type",
    "sort_group_id",
    "stage",
    "status",
    "seconds",
    "n_units",
    "merge_id",
    "warning",
    "error",
]

# Canonical stage order for the run receipt; extras (if any) append after.
_RUN_STAGE_ORDER = (
    "recording",
    "artifact_detection",
    "sorting",
    "curation",
    "merge",
)


def _run_blank_row() -> dict[str, Any]:
    return {col: None for col in _RUN_COLUMNS}


def _run_stage_seconds_total(summary) -> "float | None":
    """Total wall-clock across a run summary's ``stage_seconds`` dict."""
    if not isinstance(summary, dict):
        return None
    stage_seconds = summary.get("stage_seconds")
    if not isinstance(stage_seconds, dict) or not stage_seconds:
        return None
    return float(sum(stage_seconds.values()))


def _run_warnings(entry: dict, partial: dict | None = None) -> list[str]:
    """Warnings from a session-result entry plus any partial summary."""
    warnings: list[str] = []
    for source in (entry.get("warnings"), (partial or {}).get("warnings")):
        if not source:
            continue
        if isinstance(source, str):
            warnings.append(source)
        else:
            warnings.extend(str(warning) for warning in source)
    return warnings


def _run_metadata(entry: dict, partial: dict | None, key: str):
    """Read top-level metadata, falling back to a partial run summary."""
    value = entry.get(key)
    if value is None and isinstance(partial, dict):
        value = partial.get(key)
    return value


def _describe_run_single_rows(
    run_summary: dict, *, sort_group_id=None
) -> list[dict[str, Any]]:
    """Receipt rows for one ``run_v2_pipeline`` summary (summary, stages, warnings)."""
    stage_seconds = run_summary.get("stage_seconds") or {}
    stage_names = [
        stage
        for stage in _RUN_STAGE_ORDER
        if f"{stage}_status" in run_summary or stage in stage_seconds
    ]
    stage_names += [s for s in stage_seconds if s not in stage_names]

    rows = []
    header = _run_blank_row()
    header.update(
        row_type="summary",
        sort_group_id=sort_group_id,
        seconds=_run_stage_seconds_total(run_summary),
        n_units=run_summary.get("n_units"),
        merge_id=run_summary.get("merge_id"),
    )
    rows.append(header)
    for stage in stage_names:
        row = _run_blank_row()
        row.update(
            row_type="stage",
            sort_group_id=sort_group_id,
            stage=stage,
            status=run_summary.get(f"{stage}_status"),
            seconds=stage_seconds.get(stage),
        )
        rows.append(row)
    for warning in run_summary.get("warnings") or []:
        row = _run_blank_row()
        row.update(
            row_type="warning",
            sort_group_id=sort_group_id,
            warning=str(warning),
        )
        rows.append(row)
    return rows


def describe_run(result) -> "pd.DataFrame":
    """Render a ``run_v2_pipeline`` summary (or session result) as a receipt.

    The post-run companion to the ``describe_*`` discovery helpers: it turns the
    plain dict / list those runners return into a long-format DataFrame whose
    rows are explicit, so the things easiest to overlook -- a zero-unit sort, a
    warning, a failed group -- are first-class rows, not a value buried in a
    nested dict. Warnings never disappear into a print statement.

    Pass either a single ``run_v2_pipeline`` run summary (``dict``) or a
    ``run_v2_pipeline_session`` result (``list`` of dicts). For the session
    form, a leading ``summary`` row carries the ok / failed / zero-unit /
    with-warnings counts and ``seconds`` is ``sum(stage_seconds.values())`` per
    group; partial / missing summaries on failed groups are tolerated.

    Returns
    -------
    pandas.DataFrame
        Columns ``row_type`` (``"summary"`` / ``"stage"`` / ``"group"`` /
        ``"warning"``), ``sort_group_id``, ``stage``, ``status``, ``seconds``,
        ``n_units``, ``merge_id``, ``warning``, ``error``.
    """
    import pandas as pd

    if isinstance(result, dict):
        return pd.DataFrame(
            _describe_run_single_rows(result), columns=_RUN_COLUMNS
        )
    if isinstance(result, list):
        rows = []
        n_ok = n_failed = n_zero = n_warn = 0
        total_seconds = 0.0
        have_seconds = False
        for entry in result:
            outcome = entry.get("outcome")
            sort_group_id = entry.get("sort_group_id")
            partial = (
                entry.get("partial_run_summary")
                if outcome == "failed"
                and isinstance(entry.get("partial_run_summary"), dict)
                else None
            )
            warnings = _run_warnings(entry, partial)
            n_units = _run_metadata(entry, partial, "n_units")
            merge_id = _run_metadata(entry, partial, "merge_id")
            if outcome == "failed":
                n_failed += 1
            elif outcome == "ok":
                n_ok += 1
            else:
                # Don't silently count an unrecognized entry as ok -- that would
                # inflate the ok tally and give false reassurance (e.g. a raw
                # run summary mistakenly wrapped in a list has no 'outcome').
                raise ValueError(
                    "describe_run: session-result entry for sort_group_id="
                    f"{sort_group_id!r} has outcome={outcome!r}; expected 'ok' "
                    "or 'failed'. Pass a run_v2_pipeline_session result (list), "
                    "or a single run_v2_pipeline summary as a dict."
                )
            # A ``require_units=True`` zero-unit group raises with no partial
            # summary, so its n_units is None (not 0) and it is tallied under
            # n_failed only; n_zero counts groups that completed with zero units.
            if n_units == 0:
                n_zero += 1
            if warnings:
                n_warn += 1

            seconds = _run_stage_seconds_total(
                entry if outcome != "failed" else partial
            )
            if seconds is not None:
                total_seconds += seconds
                have_seconds = True

            group = _run_blank_row()
            group.update(
                row_type="group",
                sort_group_id=sort_group_id,
                status=outcome,
                seconds=seconds,
                n_units=n_units,
                merge_id=merge_id,
                error=entry.get("error"),
            )
            rows.append(group)
            for warning in warnings:
                row = _run_blank_row()
                row.update(
                    row_type="warning",
                    sort_group_id=sort_group_id,
                    warning=str(warning),
                )
                rows.append(row)

        header = _run_blank_row()
        header.update(
            row_type="summary",
            status=(
                f"{n_ok} ok, {n_failed} failed, {n_zero} zero-unit, "
                f"{n_warn} with warnings"
            ),
            seconds=total_seconds if have_seconds else None,
        )
        return pd.DataFrame([header] + rows, columns=_RUN_COLUMNS)

    raise TypeError(
        "describe_run: expected a run_v2_pipeline run summary (dict) or a "
        "run_v2_pipeline_session result (list of dicts); got "
        f"{type(result).__name__}."
    )

"""Notebook-facing reporting helpers for the v2 pipeline.

Reporting does not import the runners.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from spyglass.spikesorting.v2._orchestration.presets import (
    _PIPELINE_PRESETS,
)
from spyglass.spikesorting.v2._artifacts.readers import (
    read_recording_artifact_valid_times,
)
from spyglass.spikesorting.v2._core.lookup_validation import (
    _jsonable_blob,
    parameter_row_fingerprint,
)
from spyglass.spikesorting.v2._core.observed_time import observed_intervals
from spyglass.spikesorting.v2._core.parameter_identity import (
    parameter_fingerprint,
    short_fingerprint,
)
from spyglass.spikesorting.v2._params.metric_curation import (
    prepare_auto_curation_rules,
    prepare_quality_metric_row,
)
from spyglass.spikesorting.v2.exceptions import DuplicateParameterContentError

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

    One row per row of every parameter table ``initialize_v2_defaults``
    seeds, with its content fingerprint (name excluded), whether it is a
    shipped default, the presets that use it, and any row it duplicates.
    Preset metadata columns are blank for tables presets do not name.
    Reads the live tables, so user-added rows appear.

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
    from spyglass.spikesorting.v2.artifact import ArtifactDetectionParameters
    from spyglass.spikesorting.v2.recording import PreprocessingParameters
    from spyglass.spikesorting.v2.sorting import SorterParameters

    preproc_use, artifact_use, sorter_use = _preset_parameter_use()
    records: list[dict] = [
        *_preset_parameter_records(
            PreprocessingParameters,
            "preprocessing_params_name",
            preproc_use,
            lambda row, params: _preproc_summary(params),
        ),
        *_preset_parameter_records(
            ArtifactDetectionParameters,
            "artifact_detection_params_name",
            artifact_use,
            lambda row, params: _artifact_summary(params),
        ),
        *_preset_parameter_records(
            SorterParameters,
            "sorter_params_name",
            sorter_use,
            lambda row, params: _sorter_summary(row["sorter"], params),
            sorter_keyed=True,
        ),
        *_downstream_parameter_records(),
    ]
    _annotate_duplicate_parameter_records(records)

    frame = pd.DataFrame(
        [{c: rec.get(c) for c in _PARAMETER_ROW_COLUMNS} for rec in records],
        columns=_PARAMETER_ROW_COLUMNS,
    )
    return frame.sort_values(["table", "sorter", "parameter_name"]).reset_index(
        drop=True
    )


def _preset_parameter_use() -> tuple[dict, dict, dict]:
    """Which presets reference each parameter row, per stage.

    Returns
    -------
    preproc_use : dict[str, list[str]]
        Preprocessing params name to preset names.
    artifact_use : dict[str, list[str]]
        Artifact-detection params name to preset names.
    sorter_use : dict[tuple[str, str], list[str]]
        ``(sorter, sorter_params_name)`` to preset names.
    """
    preproc_use: dict[str, list[str]] = {}
    artifact_use: dict[str, list[str]] = {}
    sorter_use: dict[tuple[str, str], list[str]] = {}
    for preset_name, preset in _PIPELINE_PRESETS.items():
        preproc_use.setdefault(preset.preprocessing_params_name, []).append(
            preset_name
        )
        artifact_use.setdefault(
            preset.artifact_detection_params_name, []
        ).append(preset_name)
        sorter_use.setdefault(
            (preset.sorter, preset.sorter_params_name), []
        ).append(preset_name)
    return preproc_use, artifact_use, sorter_use


def _presets_using(attr: str) -> dict[str, list[str]]:
    """Preset names keyed by their ``attr`` parameter-row name."""
    use: dict[str, list[str]] = {}
    for preset_name, preset in _PIPELINE_PRESETS.items():
        use.setdefault(getattr(preset, attr), []).append(preset_name)
    return use


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
    """Format a parameter value compactly for a row summary."""
    # extra="allow" blobs can hold non-numeric knobs.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    return f"{value:g}"


def _preproc_summary(params: dict) -> str:
    """One-line summary of a ``PreprocessingParameters`` row."""
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
    """One-line summary of an ``ArtifactDetectionParameters`` row."""
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
    """One-line summary of a ``SorterParameters`` row."""
    bits = [sorter]
    threshold = params.get("detect_threshold")
    if threshold is not None:
        bits.append(f"detect_threshold {_num(threshold)}")
    radius = params.get("adjacency_radius")
    if radius is not None:
        bits.append(f"adjacency_radius {_num(radius)} um")
    return ", ".join(bits)


def _preset_parameter_records(
    table, name_attr, preset_use, summarize, *, sorter_keyed=False
) -> list[dict]:
    """Catalog records for every row of a preset-referenced param Lookup.

    ``preset_use`` maps each row's key (``(sorter, name)`` when
    ``sorter_keyed``) to the presets that name it. ``sorter_keyed`` also
    folds the sorter and ``execution_params`` into the fingerprint, as the
    duplicate-content guard does.
    """
    table_name = table.__name__
    if sorter_keyed:
        shipped = {(r[0], r[1]) for r in table._DEFAULT_CONTENTS}
    else:
        shipped = {r[0] for r in table._DEFAULT_CONTENTS}
    records: list[dict] = []
    for row in table.fetch(as_dict=True):
        params = _jsonable_blob(row["params"])
        key = (
            (row["sorter"], row[name_attr]) if sorter_keyed else row[name_attr]
        )
        used = sorted(preset_use.get(key, []))
        radius = params.get("adjacency_radius") if sorter_keyed else None
        records.append(
            {
                "table": table_name,
                "parameter_name": row[name_attr],
                "sorter": row["sorter"] if sorter_keyed else "",
                "probe_type": _str_axis(used, "probe_type"),
                "sampling_rate_hz": _num_axis(used, "sampling_rate_hz"),
                "adjacency_radius_um": (
                    float(radius) if radius is not None else None
                ),
                "params_schema_version": int(row["params_schema_version"]),
                "_fp": parameter_row_fingerprint(
                    table_name, row, sorter_keyed=sorter_keyed
                ),
                "is_shipped_default": key in shipped,
                "recommendation_status": _str_axis(
                    used, "recommendation_status"
                ),
                "used_by_pipeline_presets": used,
                "summary": summarize(row, params),
            }
        )
    return records


def _downstream_parameter_records() -> list[dict]:
    """Catalog records for the parameter Lookups outside the stage tables.

    Preset metadata columns stay blank; metric and auto-curation rows still
    fill ``used_by_pipeline_presets``.
    """
    from spyglass.spikesorting.v2.metric_curation import (
        AutoCurationRules,
        QualityMetricParameters,
    )
    from spyglass.spikesorting.v2.motion import (
        MotionCorrectionParameters,
        MotionEstimationParameters,
        MotionInterpolationParameters,
    )
    from spyglass.spikesorting.v2.sorting import AnalyzerWaveformParameters
    from spyglass.spikesorting.v2.unit_matching import MatcherParameters

    records: list[dict] = []
    records += _simple_parameter_records(
        AnalyzerWaveformParameters,
        "waveform_params_name",
        {r[0] for r in AnalyzerWaveformParameters._DEFAULT_CONTENTS},
        lambda r: "",
    )
    records += _simple_parameter_records(
        QualityMetricParameters,
        "metric_params_name",
        {
            r["metric_params_name"]
            for r in QualityMetricParameters._default_rows()
        },
        lambda r: f"{len(_jsonable_blob(r.get('metric_names')) or [])} metrics",
        used_by=_presets_using("metric_params_name"),
    )
    records += _simple_parameter_records(
        AutoCurationRules,
        "auto_curation_rules_name",
        {
            master["auto_curation_rules_name"]
            for master, _ in AutoCurationRules._default_payloads()
        },
        lambda r: f"merge preset {r.get('auto_merge_preset', '')!r}",
        extra_content=_autocuration_rule_content,
        used_by=_presets_using("auto_curation_rules_name"),
    )
    records += _simple_parameter_records(
        MatcherParameters,
        "matcher_params_name",
        {r["matcher_params_name"] for r in MatcherParameters._default_rows()},
        lambda r: f"matcher {r.get('matcher', '')!r}",
    )
    records += _simple_parameter_records(
        MotionEstimationParameters,
        "motion_estimation_params_name",
        {r[0] for r in MotionEstimationParameters._DEFAULT_CONTENTS},
        _motion_estimation_summary,
    )
    records += _simple_parameter_records(
        MotionInterpolationParameters,
        "motion_interpolation_params_name",
        {r[0] for r in MotionInterpolationParameters._DEFAULT_CONTENTS},
        _motion_interpolation_summary,
    )
    records += _simple_parameter_records(
        MotionCorrectionParameters,
        "motion_correction_params_name",
        {r[0] for r in MotionCorrectionParameters._DEFAULT_CONTENTS},
        lambda r: (
            f"{r['motion_estimation_params_name']} + "
            f"{r['motion_interpolation_params_name']}"
        ),
    )
    return records


def _simple_parameter_records(
    table, name_attr, shipped, summarize, extra_content=None, used_by=None
) -> list[dict]:
    """List a name-keyed param Lookup with preset-fold columns blank.

    ``shipped`` is the set of shipped default names. ``used_by`` maps a row
    name to the presets that use it. ``extra_content(row)`` folds part-table
    rows into the fingerprint so masters differing only in parts are not
    duplicates.
    """
    table_name = table.__name__
    records: list[dict] = []
    for simple_row in table.fetch(as_dict=True):
        version = int(simple_row.get("params_schema_version", 0) or 0)
        content = {
            column: _jsonable_blob(value)
            for column, value in simple_row.items()
            if column != name_attr
        }
        if extra_content is not None:
            content["__part_content__"] = extra_content(simple_row)
        records.append(
            {
                "table": table_name,
                "parameter_name": simple_row[name_attr],
                "sorter": "",
                "probe_type": None,
                "sampling_rate_hz": None,
                "adjacency_radius_um": None,
                "params_schema_version": version,
                "_fp": parameter_fingerprint(
                    table_name,
                    params=content,
                    params_schema_version=version,
                    job_kwargs=None,
                ),
                "is_shipped_default": simple_row[name_attr] in shipped,
                "recommendation_status": None,
                "used_by_pipeline_presets": (used_by or {}).get(
                    simple_row[name_attr], []
                ),
                "summary": summarize(simple_row),
            }
        )
    return records


def _autocuration_rule_content(rules_row) -> list[dict]:
    """The ruleset's ``AutoCurationRules.Rule`` rows, for its fingerprint."""
    from spyglass.spikesorting.v2.metric_curation import AutoCurationRules

    # Drop the name FK so identical rule sets under different names match.
    return [
        {
            k: _jsonable_blob(v)
            for k, v in rule.items()
            if k != "auto_curation_rules_name"
        }
        for rule in (
            AutoCurationRules.Rule
            & {
                "auto_curation_rules_name": rules_row[
                    "auto_curation_rules_name"
                ]
            }
        ).fetch(as_dict=True, order_by="rule_index")
    ]


def _motion_estimation_summary(row) -> str:
    """One-line summary of a ``MotionEstimationParameters`` row."""
    params = _jsonable_blob(row["params"])
    return (
        f"preset {params.get('preset')!r}, max_gap_s "
        f"{_num(params.get('max_gap_s'))}"
    )


def _motion_interpolation_summary(row) -> str:
    """One-line summary of a ``MotionInterpolationParameters`` row."""
    params = _jsonable_blob(row["params"])
    return (
        f"{params.get('spatial_interpolation_method')}, border_mode "
        f"{params.get('border_mode')!r}"
    )


def _annotate_duplicate_parameter_records(records: list[dict]) -> None:
    """Fill each record's ``duplicate_of``, ``fingerprint``, ``name_warnings``.

    Rows sharing a fingerprint within the same ``(table, sorter)`` scope are
    content duplicates under different names.
    """
    names_by_fp: dict[tuple[str, str, str], list[str]] = {}
    for rec in records:
        names_by_fp.setdefault(
            (rec["table"], rec["sorter"], rec["_fp"]), []
        ).append(rec["parameter_name"])

    for rec in records:
        group = sorted(names_by_fp[(rec["table"], rec["sorter"], rec["_fp"])])
        others = [n for n in group if n != rec["parameter_name"]]
        rec["duplicate_of"] = others[0] if others else None
        rec["fingerprint"] = short_fingerprint(rec["_fp"])
        warnings = []
        if rec["duplicate_of"]:
            warnings.append(f"duplicate content of {rec['duplicate_of']!r}")
        if (
            "franklab" in rec["parameter_name"]
            and not rec["is_shipped_default"]
        ):
            warnings.append("non-catalog row using the 'franklab' name")
        rec["name_warnings"] = "; ".join(warnings)


def _content_equal(left, right) -> bool:
    """Deep equality for fetched-vs-shipped default content.

    Fetched bool columns come back as int, so ``False`` equals ``0`` here,
    unlike the strict ``_curation.metrics._values_match``.
    """
    if isinstance(left, bool) or isinstance(right, bool):
        return left == right
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _content_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _content_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _canonical_quality_metric_rows(table) -> list[dict]:
    """Shipped ``QualityMetricParameters`` defaults in their STORED shape.

    ``_default_rows`` is insert input; prepare it as the insert does so every
    stored column is compared.
    """
    return [prepare_quality_metric_row(row) for row in table._default_rows()]


def _canonical_autocuration_master_rows(table) -> list[dict]:
    """Shipped ``AutoCurationRules`` MASTER defaults in their STORED shape.

    Prepared as ``insert_rules`` does so defaulted columns are compared.
    ``Rule`` parts are compared separately.
    """
    return [
        prepare_auto_curation_rules(master, rule_rows)[0]
        for master, rule_rows in table._default_payloads()
    ]


def _shipped_default_rows(table) -> list[dict]:
    """Shipped default rows for a Lookup in their STORED (canonical) shape.

    Returns ``[]`` when a table has no static default source.
    """
    contents = getattr(table, "_DEFAULT_CONTENTS", None)
    if contents:
        names = list(table().heading.names)
        return [dict(zip(names, _jsonable_blob(list(row)))) for row in contents]
    if table.__name__ == "QualityMetricParameters":
        return _canonical_quality_metric_rows(table)
    if table.__name__ == "AutoCurationRules":
        return _canonical_autocuration_master_rows(table)
    rows_fn = getattr(table, "_default_rows", None)
    if callable(rows_fn):
        return [dict(row) for row in rows_fn()]
    payloads_fn = getattr(table, "_default_payloads", None)
    if callable(payloads_fn):
        # Rule parts are compared separately in verify_v2_default_catalog.
        return [dict(master) for master, _parts in payloads_fn()]
    return []


def _v2_default_catalog_tables():
    """The default-bearing v2 parameter Lookups + their name attribute.

    Every table ``initialize_v2_defaults`` seeds.
    """
    from spyglass.spikesorting.v2.artifact import ArtifactDetectionParameters
    from spyglass.spikesorting.v2.metric_curation import (
        AutoCurationRules,
        QualityMetricParameters,
    )
    from spyglass.spikesorting.v2.motion import (
        MotionCorrectionParameters,
        MotionEstimationParameters,
        MotionInterpolationParameters,
    )
    from spyglass.spikesorting.v2.recording import PreprocessingParameters
    from spyglass.spikesorting.v2.sorting import (
        AnalyzerWaveformParameters,
        SorterParameters,
    )
    from spyglass.spikesorting.v2.unit_matching import MatcherParameters

    return [
        (PreprocessingParameters, "preprocessing_params_name"),
        (ArtifactDetectionParameters, "artifact_detection_params_name"),
        (SorterParameters, "sorter_params_name"),
        (AnalyzerWaveformParameters, "waveform_params_name"),
        (QualityMetricParameters, "metric_params_name"),
        (AutoCurationRules, "auto_curation_rules_name"),
        (MatcherParameters, "matcher_params_name"),
        (MotionEstimationParameters, "motion_estimation_params_name"),
        (MotionInterpolationParameters, "motion_interpolation_params_name"),
        (MotionCorrectionParameters, "motion_correction_params_name"),
    ]


def _diverged_autocuration_rules(table) -> list[dict]:
    """Stale ``AutoCurationRules.Rule`` part rows vs the shipped rule sets.

    A different rule count, a missing ``rule_index``, or a diverged field is
    stale.
    """
    stale: list[dict] = []
    for master, shipped_rules in table._default_payloads():
        name = master["auto_curation_rules_name"]
        restriction = {"auto_curation_rules_name": name}
        if not (table & restriction):
            continue  # not seeded
        stored_by_index = {
            int(rule["rule_index"]): rule
            for rule in (table.Rule & restriction).fetch(as_dict=True)
        }
        if len(stored_by_index) != len(shipped_rules):
            stale.append(
                {
                    "table": "AutoCurationRules.Rule",
                    "name": name,
                    "fields": ["rule_count"],
                }
            )
            continue
        for shipped_rule in shipped_rules:
            index = int(shipped_rule["rule_index"])
            stored_rule = stored_by_index.get(index)
            if stored_rule is None:
                stale.append(
                    {
                        "table": "AutoCurationRules.Rule",
                        "name": name,
                        "fields": [f"rule_index_{index}_missing"],
                    }
                )
                continue
            diverged = [
                field
                for field, value in shipped_rule.items()
                if field not in ("auto_curation_rules_name", "rule_index")
                and field in stored_rule
                and not _content_equal(
                    _jsonable_blob(value), _jsonable_blob(stored_rule[field])
                )
            ]
            if diverged:
                stale.append(
                    {
                        "table": "AutoCurationRules.Rule",
                        "name": f"{name}[{index}]",
                        "fields": sorted(diverged),
                    }
                )
    return stale


def verify_v2_default_catalog(*, strict: bool = False) -> list[dict]:
    """Flag stored same-name default rows whose content diverged from shipped.

    ``initialize_v2_defaults`` skips existing names without comparing content,
    so a hand-edited or older-default row keeps its stale content. Compares
    each seeded default per primary key, including ``AutoCurationRules.Rule``
    parts.

    Parameters
    ----------
    strict : bool, optional
        Raise ``DuplicateParameterContentError`` listing the stale defaults
        instead of returning them. Default ``False`` (return the list).

    Returns
    -------
    list[dict]
        One entry per stale default: ``{"table", "name", "fields"}`` where
        ``fields`` are the diverged content columns. Empty when the catalog is
        clean.
    """
    stale: list[dict] = []
    for table, name_attr in _v2_default_catalog_tables():
        pk_fields = list(table.primary_key)
        stored_by_pk = {
            tuple(row[field] for field in pk_fields): row
            for row in table.fetch(as_dict=True)
        }
        for shipped in _shipped_default_rows(table):
            try:
                pk = tuple(shipped[field] for field in pk_fields)
            except KeyError:
                continue
            stored = stored_by_pk.get(pk)
            if stored is None:
                continue  # not seeded
            diverged = [
                field
                for field, shipped_value in shipped.items()
                if field not in pk_fields
                and field in stored
                and not _content_equal(
                    _jsonable_blob(shipped_value),
                    _jsonable_blob(stored[field]),
                )
            ]
            if diverged:
                stale.append(
                    {
                        "table": table.__name__,
                        "name": shipped[name_attr],
                        "fields": sorted(diverged),
                    }
                )
        if table.__name__ == "AutoCurationRules":
            stale.extend(_diverged_autocuration_rules(table))
    if strict and stale:
        raise DuplicateParameterContentError(
            "verify_v2_default_catalog: stored default row(s) diverge from the "
            f"shipped content: {stale}. Drop the stale row(s) and re-seed via "
            "insert_default(), or reconcile the divergence deliberately."
        )
    return stale


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

    Mirrors ``Sorting.make_fetch``: the artifact-retained duration if masked,
    else sample count / sampling frequency.
    """
    from spyglass.spikesorting.v2.recording import RecordingSelection
    from spyglass.spikesorting.v2.sorting import SortingSelection

    sorting_key = {"sorting_id": sorting_id}
    source, traces = SortingSelection.resolve_effective_source(sorting_key)
    # Only samples and timestamps are read; no mask needed.
    recording = SortingSelection.load_stored_traces(traces)
    if source.kind == "concatenated_recording":
        from spyglass.spikesorting.v2.session_group import (
            ConcatenatedRecording,
        )

        intervals = (ConcatenatedRecording & source.key).fetch1("obs_intervals")
        return float(
            np.diff(observed_intervals(recording, intervals), axis=1).sum()
        )
    recording_id = source.key["recording_id"]
    artifact_detection_id = source.artifact_detection_id
    if artifact_detection_id is not None:
        nwb_file_name = (
            RecordingSelection & {"recording_id": recording_id}
        ).fetch1("nwb_file_name")
        valid_times = read_recording_artifact_valid_times(
            artifact_detection_id, nwb_file_name, caller="describe_units"
        )
        return float(
            np.diff(observed_intervals(recording, valid_times), axis=1).sum()
        )
    return recording.get_num_samples() / recording.sampling_frequency


def describe_units(sorting_id) -> "pd.DataFrame":
    """Return a per-unit, sort-time quality snapshot for one sort.

    Read-only, from sort-time metadata. ``firing_rate_hz`` divides by the
    observed (artifact-removed) duration. ``peak_electrode_id`` and
    ``brain_region`` come from the peak-amplitude electrode (the first
    member's for concat sorts).

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
        # Units with zero observed time indicate an upstream defect.
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
    "root_merge_id",
    "auto_labeled_merge_id",
    "member_index",
    "nwb_file_name",
    "member_merge_id",
    "warning",
    "error",
    "setting",
    "value",
]

# Canonical stage order for the run receipt; extras (if any) append after.
_RUN_STAGE_ORDER = (
    "recording",
    "artifact_detection",
    "member_recording",
    "member_artifact_detection",
    "concat_recording",
    "motion_estimate",
    "motion_corrected_recording",
    "sorting",
    "curation",
    "auto_curation",
    "member_curation",
)


#: Motion receipt fields ``describe_run`` shows, in display order.
_RUN_MOTION_FIELDS = (
    "motion_mode",
    "motion_correction_params_name",
    "motion_estimate_id",
    "motion_estimate_supplied",
    "motion_estimation_preset",
    "motion_corrected_recording_id",
    "motion_removed_channel_ids",
    "motion_spans_without_evidence",
)


def _run_row(**fields: Any) -> dict[str, Any]:
    """One receipt row: every ``_RUN_COLUMNS`` key, ``None`` unless given."""
    return {**dict.fromkeys(_RUN_COLUMNS), **fields}


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


def _failed_partial_summary(entry: dict) -> "dict | None":
    """A failed session entry's partial run summary, or ``None``."""
    partial = entry.get("partial_run_summary")
    if entry.get("outcome") == "failed" and isinstance(partial, dict):
        return partial
    return None


def _session_outcome_counts(results: list) -> tuple[int, int, int, int]:
    """Count a session result's ok, failed, zero-unit and warned groups.

    ``n_units`` and warnings fall back to a failed group's partial run
    summary. A ``require_units=True`` zero-unit group raises with no partial
    summary, so its ``n_units`` is None (not 0) and it counts as failed only;
    zero-unit counts groups that completed with zero units.

    Returns
    -------
    tuple of int
        ``(n_ok, n_failed, n_zero_unit, n_with_warnings)``.
    """
    n_ok = n_failed = n_zero = n_warn = 0
    for entry in results:
        partial = _failed_partial_summary(entry)
        if entry.get("outcome") == "ok":
            n_ok += 1
        elif entry.get("outcome") == "failed":
            n_failed += 1
        if _run_metadata(entry, partial, "n_units") == 0:
            n_zero += 1
        if _run_warnings(entry, partial):
            n_warn += 1
    return n_ok, n_failed, n_zero, n_warn


def _describe_match_input(match_input) -> str:
    """One line for a unit-match receipt input: kind, sessions, curation, traces.

    ``match_input`` is a ``UnitMatchInputSummary``.
    """
    if match_input.motion_corrected_recording_id is None:
        traces = f"{match_input.waveform_traces} traces (not motion corrected)"
    else:
        traces = (
            f"{match_input.waveform_traces} traces "
            "(motion corrected, motion_corrected_recording_id="
            f"{match_input.motion_corrected_recording_id})"
        )
    return (
        f"{match_input.source_kind} {match_input.source_id}; "
        f"{match_input.n_recordings} recording(s) from "
        f"{', '.join(match_input.nwb_file_names)}; "
        f"curation sorting_id={match_input.sorting_id}, "
        f"curation_id={match_input.curation_id} "
        f"(curation_uuid={match_input.curation_uuid}); {traces}"
    )


def _describe_run_single_rows(run_summary: dict) -> list[dict[str, Any]]:
    """Receipt rows for one ``run_v2_pipeline`` summary (summary, stages, warnings)."""
    stage_seconds = run_summary.get("stage_seconds") or {}
    stage_names = [
        stage
        for stage in _RUN_STAGE_ORDER
        if f"{stage}_status" in run_summary or stage in stage_seconds
    ]
    stage_names += [s for s in stage_seconds if s not in stage_names]

    rows = []
    auto_labeled_merge_id = run_summary.get("auto_labeled_merge_id")
    # Only run_v2_pipeline summaries have a root_merge_id.
    if "root_merge_id" in run_summary:
        summary_status = (
            "auto-labeled"
            if run_summary.get("auto_labeled_curation_id") is not None
            else "root only"
        )
    else:
        summary_status = None
    header = _run_row(
        row_type="summary",
        status=summary_status,
        seconds=_run_stage_seconds_total(run_summary),
        n_units=run_summary.get("n_units"),
        root_merge_id=run_summary.get("root_merge_id"),
        auto_labeled_merge_id=auto_labeled_merge_id,
    )
    rows.append(header)
    for stage in stage_names:
        row = _run_row(
            row_type="stage",
            stage=stage,
            status=run_summary.get(f"{stage}_status"),
            seconds=stage_seconds.get(stage),
        )
        rows.append(row)
    for setting, value in (run_summary.get("scientific_config") or {}).items():
        row = _run_row(
            row_type="config",
            setting=setting,
            value=repr(value),
        )
        rows.append(row)
    for artifact in run_summary.get("member_artifacts", []):
        row = _run_row(
            row_type="member_artifact",
            member_index=artifact["member_index"],
            status=artifact["status"],
            setting="artifact_detection_id",
            value=f"{artifact['artifact_detection_id']}; masked {artifact['masked_duration_s']:.6g} s",
        )
        rows.append(row)
    if "artifact_masked_duration_s" in run_summary:
        row = _run_row(
            row_type="config",
            setting="artifact_masked_duration_s",
            value=str(run_summary["artifact_masked_duration_s"]),
        )
        rows.append(row)
    if "motion_mode" in run_summary:
        for setting in _RUN_MOTION_FIELDS:
            row = _run_row(
                row_type="config",
                setting=setting,
                value=str(run_summary.get(setting)),
            )
            rows.append(row)
    for member_index, member_merge_id in sorted(
        (run_summary.get("member_merge_ids") or {}).items()
    ):
        row = _run_row(
            row_type="member",
            member_index=int(member_index),
            member_merge_id=member_merge_id,
        )
        rows.append(row)
    for match_input in run_summary.get("inputs") or ():
        row = _run_row(
            row_type="input",
            nwb_file_name=", ".join(match_input.nwb_file_names),
            setting=f"input_{match_input.input_index}",
            value=_describe_match_input(match_input),
        )
        rows.append(row)
    for warning in run_summary.get("warnings") or []:
        row = _run_row(
            row_type="warning",
            warning=str(warning),
        )
        rows.append(row)
    sorter_config = run_summary.get("sorter_config")
    if isinstance(sorter_config, dict):
        for setting in (
            "sorter",
            "external_whiten",
            "random_seed",
            "job_kwargs",
            "execution_backend",
            "container_image",
            "si_sorter_params",
        ):
            if setting not in sorter_config:
                continue
            row = _run_row(
                row_type="config",
                setting=setting,
                value=repr(sorter_config[setting]),
            )
            rows.append(row)
    return rows


def describe_run(result) -> "pd.DataFrame":
    """Render a ``run_v2_pipeline`` summary (or session result) as a receipt.

    Long format, so zero-unit sorts, concat member merge IDs, warnings and
    failed groups are explicit rows. Accepts a ``run_v2_pipeline`` summary or
    ``run_v2_unit_match`` manifest (``dict``), or a
    ``run_v2_pipeline_session`` result (``list``), whose leading ``summary``
    row counts ok / failed / zero-unit / warned groups.

    Returns
    -------
    pandas.DataFrame
        Columns ``row_type`` (``"summary"`` / ``"stage"`` / ``"config"`` /
        ``"member_artifact"`` / ``"member"`` / ``"input"`` / ``"group"`` /
        ``"warning"``), ``sort_group_id``, ``stage``, ``status``,
        ``seconds``, ``n_units``, ``root_merge_id``, ``auto_labeled_merge_id``,
        ``member_index``, ``nwb_file_name``, ``member_merge_id``, ``warning``,
        ``error``, ``setting``, ``value``.

        - ``config`` rows: scientific setup, effective sorter configuration,
          and motion fields (``"None"`` where the mode produces none).
        - ``summary`` status: ``"root only"`` / ``"auto-labeled"`` for a
          ``run_v2_pipeline`` summary, blank otherwise.
        - Concat runs leave ``auto_labeled_merge_id`` unset and report
          ``member`` rows instead.
        - ``input`` rows (``run_v2_unit_match``): one per matching input, in
          ``input_index`` order.
    """
    if isinstance(result, dict):
        return pd.DataFrame(
            _describe_run_single_rows(result), columns=_RUN_COLUMNS
        )
    if isinstance(result, list):
        rows = []
        total_seconds = 0.0
        have_seconds = False
        for entry in result:
            outcome = entry.get("outcome")
            sort_group_id = entry.get("sort_group_id")
            partial = _failed_partial_summary(entry)
            warnings = _run_warnings(entry, partial)
            n_units = _run_metadata(entry, partial, "n_units")
            root_merge_id = _run_metadata(entry, partial, "root_merge_id")
            auto_labeled_merge_id = _run_metadata(
                entry, partial, "auto_labeled_merge_id"
            )
            if outcome not in ("ok", "failed"):
                # E.g. a run summary wrongly wrapped in a list.
                raise ValueError(
                    "describe_run: session-result entry for sort_group_id="
                    f"{sort_group_id!r} has outcome={outcome!r}; expected 'ok' "
                    "or 'failed'. Pass a run_v2_pipeline_session result (list), "
                    "or a single run_v2_pipeline summary as a dict."
                )

            seconds = _run_stage_seconds_total(
                entry if outcome != "failed" else partial
            )
            if seconds is not None:
                total_seconds += seconds
                have_seconds = True

            group = _run_row(
                row_type="group",
                sort_group_id=sort_group_id,
                status=outcome,
                seconds=seconds,
                n_units=n_units,
                root_merge_id=root_merge_id,
                auto_labeled_merge_id=auto_labeled_merge_id,
                error=entry.get("error"),
            )
            rows.append(group)
            for warning in warnings:
                row = _run_row(
                    row_type="warning",
                    sort_group_id=sort_group_id,
                    warning=str(warning),
                )
                rows.append(row)

        n_ok, n_failed, n_zero, n_warn = _session_outcome_counts(result)
        header = _run_row(
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

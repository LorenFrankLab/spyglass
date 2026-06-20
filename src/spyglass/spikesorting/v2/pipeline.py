"""Convenience orchestration across the spike sorting tables.

``run_v2_pipeline`` chains the recording -> artifact -> sort ->
curation stages into one call so notebook users can populate an
end-to-end single-session sort without writing the per-stage
insert_selection / populate boilerplate. The orchestrator focuses on
the minimum-viable single-session path; richer surfaces (metrics +
auto-curation, concat sorts, cross-session matching, UI hooks) come
in later versions.

Pipeline presets are Pydantic-validated bundles of Lookup-row names; the
orchestrator looks them up at first call. The shipped presets are the dated
franklab production recipes -- the MountainSort4 family by target region
(hippocampus 600 Hz / cortex 300 Hz high-pass) and sampling rate, a
MountainSort5 preset, and a clusterless preset. The ``run_v2_pipeline``
default is the MountainSort5 tetrode-hippocampus recipe: it runs under the v2
``numpy>=2`` baseline, whereas MountainSort4's ``ml_ms4alg`` backend needs
``numpy<2`` (see the MS4 preset notes). Call ``describe_pipeline_presets()``
for the catalog and ``list_pipeline_presets()`` for the names.

The orchestrator is idempotent: re-running with the same inputs finds
existing rows via the insert_selection helpers and returns the same
run summary (with the same merge_id) without inserting duplicates.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    import pandas as pd


class _PipelinePreset(BaseModel):
    """A v2 pipeline preset: a bundle of Lookup-row names.

    The orchestrator consults this bundle once, then drives the
    ``insert_selection`` -> ``populate`` chain on each stage. ``params``
    Lookup row names must exist in the database before
    ``run_v2_pipeline`` is called; the preset itself does NOT insert
    Lookup rows (default rows are inserted explicitly by callers via
    ``*.insert_default()``).

    The human-facing fields (``intended_use``, ``threshold_units``,
    ``notes``) carry no runtime behavior; they describe the preset for
    ``describe_pipeline_presets()`` so a scientist can choose one without reading
    module source. They default to ``""`` so external presets need not
    supply them, but the built-ins below populate them.
    """

    model_config = ConfigDict(extra="forbid")
    preprocessing_params_name: str
    artifact_detection_params_name: str
    sorter: str
    sorter_params_name: str
    # Discovery metadata (no runtime behavior) -- the axes a scientist picks a
    # preset by. probe_type is informational: the recipe is set by target
    # region (the preproc high-pass) and sampling rate (the sorter window),
    # not by probe geometry, so e.g. the tetrode- and probe-hippocampus 30 kHz
    # presets resolve to the SAME parameter rows.
    probe_type: str = ""  # "tetrode" | "probe" | "neuropixels" | "" (n/a)
    target_region: str = ""  # "hippocampus" | "cortex" -> preproc high-pass
    sampling_rate_hz: "int | None" = None
    sorter_family: str = ""
    adjacency_radius_um: "float | None" = (
        None  # informational MS4 spatial radius
    )
    recommendation_status: str = ""  # production | alternative | experimental
    intended_use: str = ""  # one-line "when to reach for this pipeline preset"
    threshold_units: str = ""  # detection-threshold units (sigma / µV)
    notes: str = ""  # key assumptions (probe geometry, sampling rate, etc.)


def list_pipeline_presets() -> list[str]:
    """Return the sorted pipeline-preset names accepted by ``run_v2_pipeline``.

    Notebook-discoverable accessor so users don't need to read the
    module source to learn what's available.

    Examples
    --------
    >>> from spyglass.spikesorting.v2.pipeline import list_pipeline_presets
    >>> any("ms4_2026_06" in p for p in list_pipeline_presets())
    True
    """
    return sorted(_PIPELINE_PRESETS)


def describe_pipeline_presets() -> "pd.DataFrame":
    """Return a table describing each pipeline preset accepted by ``run_v2_pipeline``.

    Companion to :func:`list_pipeline_presets` that adds the detail a scientist
    needs to choose a preset -- the sorter, the parameter-row names each
    stage uses, the intended use, and (a known footgun) the units of the
    detection threshold -- without reading module source. Pure and
    database-free: it reads only the in-module preset metadata and
    queries/inserts nothing. ``pandas`` is imported lazily so
    ``import spyglass.spikesorting.v2.pipeline`` stays cheap.

    Returns
    -------
    pandas.DataFrame
        One row per pipeline preset, sorted by name, with discovery
        metadata (``recommendation_status``, ``probe_type``,
        ``target_region``, ``sampling_rate_hz``, ``sorter``,
        ``sorter_family``, ``adjacency_radius_um``), the resolved row names
        (``preprocessing_params_name`` / ``artifact_detection_params_name``
        / ``sorter_params_name``), ``intended_use``, ``threshold_units``,
        and ``notes``. Call ``.to_dict("records")`` for raw rows.

    Examples
    --------
    >>> from spyglass.spikesorting.v2.pipeline import describe_pipeline_presets
    >>> "recommendation_status" in describe_pipeline_presets().columns
    True
    """
    import pandas as pd

    columns = [
        "pipeline_preset",
        "recommendation_status",
        "probe_type",
        "target_region",
        "sampling_rate_hz",
        "sorter",
        "sorter_family",
        "adjacency_radius_um",
        "preprocessing_params_name",
        "artifact_detection_params_name",
        "sorter_params_name",
        "intended_use",
        "threshold_units",
        "notes",
    ]
    rows = [
        {
            "pipeline_preset": name,
            "recommendation_status": preset.recommendation_status,
            "probe_type": preset.probe_type,
            "target_region": preset.target_region,
            "sampling_rate_hz": preset.sampling_rate_hz,
            "sorter": preset.sorter,
            "sorter_family": preset.sorter_family,
            "adjacency_radius_um": preset.adjacency_radius_um,
            "preprocessing_params_name": preset.preprocessing_params_name,
            "artifact_detection_params_name": (
                preset.artifact_detection_params_name
            ),
            "sorter_params_name": preset.sorter_params_name,
            "intended_use": preset.intended_use,
            "threshold_units": preset.threshold_units,
            "notes": preset.notes,
        }
        for name, preset in sorted(_PIPELINE_PRESETS.items())
    ]
    return pd.DataFrame(rows, columns=columns)


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
        """The single agreed preset value for a numeric ``attr``, else None."""
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


_PRESET_DETAIL_COLUMNS = [
    "stage",
    "params_row_name",
    "key",
    "value",
    "params_schema_version",
    "job_kwargs",
]


def describe_pipeline_preset(name: str) -> "pd.DataFrame":
    """Unpack one pipeline preset into its full, validated parameter values.

    The singular companion to :func:`describe_pipeline_presets`: where the
    plural helper lists every preset and the parameter-row *names* each stage
    uses, this resolves ONE preset to the actual VALUES of its preprocessing,
    artifact-detection, and sorter parameter rows -- so you can see exactly what
    ``run_v2_pipeline(..., pipeline_preset=name)`` will do before running it.

    Unlike the DB-free :func:`describe_pipeline_presets`, this reads the live
    parameter Lookup tables. If a referenced row is missing it raises a clear
    "run ``initialize_v2_defaults()``" message rather than failing opaquely.

    Parameters
    ----------
    name : str
        A pipeline-preset name from :func:`list_pipeline_presets` /
        :func:`describe_pipeline_presets`.

    Returns
    -------
    pandas.DataFrame
        Long-format, one row per parameter, columns ``stage`` (``"preset"`` /
        ``"preprocessing"`` / ``"artifact_detection"`` / ``"sorter"``),
        ``params_row_name``, ``key`` (dotted path into the validated blob), and
        ``value``. Parameter-row entries also carry ``params_schema_version`` and
        ``job_kwargs`` so the display shows the full row resolved by the preset,
        not just the core params blob. Preset entries include the human-facing
        ``threshold_units`` field so detection thresholds are never unitless.
    """
    import pandas as pd

    # Validate the name BEFORE importing the table modules: importing them
    # triggers DataJoint ``@schema`` decoration (a DB connection), so an unknown
    # name must reject first to keep that path database-free.
    if name not in _PIPELINE_PRESETS:
        raise ValueError(
            f"unknown pipeline_preset {name!r}. Available pipeline presets: "
            f"{sorted(_PIPELINE_PRESETS)}. Call describe_pipeline_presets() to "
            "see what each one does."
        )
    preset = _PIPELINE_PRESETS[name]

    from spyglass.spikesorting.v2.artifact import ArtifactDetectionParameters
    from spyglass.spikesorting.v2.recording import PreprocessingParameters
    from spyglass.spikesorting.v2.sorting import SorterParameters
    from spyglass.spikesorting.v2.utils import _jsonable_blob

    def _flatten(prefix, value):
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                key = f"{prefix}.{sub_key}" if prefix else str(sub_key)
                yield from _flatten(key, sub_value)
        else:
            yield prefix, value

    def _fetch_params(table, restriction, label):
        rel = table & restriction
        if not rel:
            raise ValueError(
                f"describe_pipeline_preset: {label} row {restriction!r} "
                f"referenced by preset {name!r} is not in the database. Run "
                "initialize_v2_defaults() to install the shipped parameter "
                "catalog."
            )
        params, params_schema_version, job_kwargs = rel.fetch1(
            "params", "params_schema_version", "job_kwargs"
        )
        return {
            "params": _jsonable_blob(params),
            "params_schema_version": int(params_schema_version),
            "job_kwargs": _jsonable_blob(job_kwargs),
        }

    def _detail_row(
        *,
        stage,
        params_row_name,
        key,
        value,
        params_schema_version=None,
        job_kwargs=None,
    ):
        return {
            "stage": stage,
            "params_row_name": params_row_name,
            "key": key,
            "value": value,
            "params_schema_version": params_schema_version,
            "job_kwargs": job_kwargs,
        }

    rows = [
        _detail_row(
            stage="preset",
            params_row_name=name,
            key=key,
            value=value,
        )
        for key, value in (
            ("sorter", preset.sorter),
            ("target_region", preset.target_region),
            ("sampling_rate_hz", preset.sampling_rate_hz),
            ("recommendation_status", preset.recommendation_status),
            ("threshold_units", preset.threshold_units),
            ("intended_use", preset.intended_use),
            ("notes", preset.notes),
        )
    ]
    for stage, row_name, row in (
        (
            "preprocessing",
            preset.preprocessing_params_name,
            _fetch_params(
                PreprocessingParameters,
                {"preprocessing_params_name": preset.preprocessing_params_name},
                "PreprocessingParameters",
            ),
        ),
        (
            "artifact_detection",
            preset.artifact_detection_params_name,
            _fetch_params(
                ArtifactDetectionParameters,
                {
                    "artifact_detection_params_name": (
                        preset.artifact_detection_params_name
                    )
                },
                "ArtifactDetectionParameters",
            ),
        ),
        (
            "sorter",
            preset.sorter_params_name,
            _fetch_params(
                SorterParameters,
                {
                    "sorter": preset.sorter,
                    "sorter_params_name": preset.sorter_params_name,
                },
                "SorterParameters",
            ),
        ),
    ):
        for key, value in _flatten("", row["params"]):
            rows.append(
                _detail_row(
                    stage=stage,
                    params_row_name=row_name,
                    key=key,
                    value=value,
                    params_schema_version=row["params_schema_version"],
                    job_kwargs=row["job_kwargs"],
                )
            )
    return pd.DataFrame(rows, columns=_PRESET_DETAIL_COLUMNS)


# Alias: shorter discovery name for the same helper.
describe_preset = describe_pipeline_preset


_SORT_GROUP_COLUMNS = [
    "nwb_file_name",
    "sort_group_id",
    "n_electrodes",
    "electrode_ids",
    "electrode_group_names",
    "probe_shanks",
    "brain_regions",
    "bad_channel_count",
    "reference_mode",
    "reference_electrode_id",
]


def describe_sort_groups(nwb_file_name: str) -> "pd.DataFrame":
    """Return a notebook-friendly summary of v2 sort groups for a session.

    Use this after creating ``SortGroupV2`` rows, before choosing a
    ``sort_group_id`` for ``run_v2_pipeline``. The table surfaces the
    scientific context a user normally needs at that decision point:
    electrode membership, electrode groups, probe shanks, brain regions,
    bad-channel membership, and reference mode. The helper is read-only:
    it restricts existing ``SortGroupV2`` / ``Electrode`` / ``BrainRegion``
    rows and never creates sort groups.

    Parameters
    ----------
    nwb_file_name : str
        Session whose existing ``SortGroupV2`` rows should be summarized.

    Returns
    -------
    pandas.DataFrame
        One row per sort group, sorted by ``sort_group_id``. Empty, with
        the documented columns, when the session has no sort groups.
        Columns are ``nwb_file_name``, ``sort_group_id``, ``n_electrodes``,
        ``electrode_ids``, ``electrode_group_names``, ``probe_shanks``,
        ``brain_regions``, ``bad_channel_count``, ``reference_mode``, and
        ``reference_electrode_id``.
    """
    import pandas as pd

    from spyglass.common.common_ephys import Electrode
    from spyglass.common.common_region import BrainRegion
    from spyglass.spikesorting.v2.recording import SortGroupV2

    def _nullable_int(value):
        if value is None or pd.isna(value):
            return None
        return int(value)

    def _sorted_nullable_ints(values):
        normalized = {_nullable_int(value) for value in values}
        return tuple(
            sorted(
                normalized,
                key=lambda value: (
                    value is None,
                    value if value is not None else 0,
                ),
            )
        )

    master_rows = (SortGroupV2 & {"nwb_file_name": nwb_file_name}).fetch(
        as_dict=True
    )
    if not master_rows:
        return pd.DataFrame(columns=_SORT_GROUP_COLUMNS)

    rows = []
    for master in sorted(
        master_rows, key=lambda row: int(row["sort_group_id"])
    ):
        sort_group_id = int(master["sort_group_id"])
        restriction = {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": sort_group_id,
        }
        electrode_rows = (
            (SortGroupV2.SortGroupElectrode & restriction)
            * Electrode
            * BrainRegion
        ).fetch(as_dict=True)

        reference_mode = master["reference_mode"]
        reference_electrode_id = (
            _nullable_int(master["reference_electrode_id"])
            if reference_mode == "specific"
            else None
        )
        rows.append(
            {
                "nwb_file_name": nwb_file_name,
                "sort_group_id": sort_group_id,
                "n_electrodes": len(electrode_rows),
                "electrode_ids": tuple(
                    sorted(int(row["electrode_id"]) for row in electrode_rows)
                ),
                "electrode_group_names": tuple(
                    sorted(
                        {
                            str(row["electrode_group_name"])
                            for row in electrode_rows
                        }
                    )
                ),
                "probe_shanks": _sorted_nullable_ints(
                    row.get("probe_shank") for row in electrode_rows
                ),
                "brain_regions": tuple(
                    sorted({str(row["region_name"]) for row in electrode_rows})
                ),
                "bad_channel_count": sum(
                    str(row.get("bad_channel")) == "True"
                    for row in electrode_rows
                ),
                "reference_mode": reference_mode,
                "reference_electrode_id": reference_electrode_id,
            }
        )
    return pd.DataFrame(rows, columns=_SORT_GROUP_COLUMNS)


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
    recording_id = (SortingSelection.RecordingSource & sorting_key).fetch1(
        "recording_id"
    )
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
    neighbour), which arrive with the analyzer-driven curation in a later
    release.

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


def _sort_group_geometry_rows(nwb_file_name: str) -> list[dict[str, Any]]:
    """Return DB-backed electrode geometry rows for sort-group plotting."""
    import pandas as pd

    from spyglass.common.common_device import Probe
    from spyglass.common.common_ephys import Electrode
    from spyglass.common.common_region import BrainRegion
    from spyglass.spikesorting.v2.recording import SortGroupV2

    def _missing(value) -> bool:
        return value is None or bool(pd.isna(value))

    def _nullable_int(value):
        if _missing(value):
            return None
        return int(value)

    master_rows = (SortGroupV2 & {"nwb_file_name": nwb_file_name}).fetch(
        as_dict=True
    )
    master_by_group = {
        int(row["sort_group_id"]): row
        for row in sorted(
            master_rows, key=lambda row: int(row["sort_group_id"])
        )
    }
    if not master_by_group:
        return []

    member_rows = (
        (SortGroupV2.SortGroupElectrode & {"nwb_file_name": nwb_file_name})
        * Electrode
        * BrainRegion
    ).fetch(as_dict=True)

    probe_restrictions = []
    for electrode in member_rows:
        probe_key = {
            "probe_id": electrode.get("probe_id"),
            "probe_shank": electrode.get("probe_shank"),
            "probe_electrode": electrode.get("probe_electrode"),
        }
        if all(not _missing(value) for value in probe_key.values()):
            probe_restrictions.append(
                {
                    key: _nullable_int(value) if key != "probe_id" else value
                    for key, value in probe_key.items()
                }
            )
    probe_geometry = {}
    if probe_restrictions:
        probe_rows = (Probe.Electrode & probe_restrictions).fetch(
            "probe_id",
            "probe_shank",
            "probe_electrode",
            "rel_x",
            "rel_y",
            "rel_z",
            "contact_size",
            as_dict=True,
        )
        probe_geometry = {
            (
                row["probe_id"],
                int(row["probe_shank"]),
                int(row["probe_electrode"]),
            ): row
            for row in probe_rows
        }

    rows: list[dict[str, Any]] = []
    for sort_group_id, master in master_by_group.items():
        sort_group_id = int(master["sort_group_id"])
        reference_mode = master["reference_mode"]
        reference_electrode_id = (
            _nullable_int(master["reference_electrode_id"])
            if reference_mode == "specific"
            else None
        )
        group_electrodes = sorted(
            (
                row
                for row in member_rows
                if int(row["sort_group_id"]) == sort_group_id
            ),
            key=lambda row: int(row["electrode_id"]),
        )
        for electrode in group_electrodes:
            rel_x = rel_y = rel_z = contact_size = None
            probe_key = {
                "probe_id": electrode.get("probe_id"),
                "probe_shank": electrode.get("probe_shank"),
                "probe_electrode": electrode.get("probe_electrode"),
            }
            if all(not _missing(value) for value in probe_key.values()):
                geometry = probe_geometry.get(
                    (
                        probe_key["probe_id"],
                        _nullable_int(probe_key["probe_shank"]),
                        _nullable_int(probe_key["probe_electrode"]),
                    )
                )
                if geometry:
                    rel_x = geometry["rel_x"]
                    rel_y = geometry["rel_y"]
                    rel_z = geometry["rel_z"]
                    contact_size = geometry["contact_size"]

            # Pick plot coordinates from a SINGLE source so plot_x/plot_y and
            # coordinate_source can never disagree (e.g. plot probe rel_x
            # against electrode y, or label "electrode" while plotting a probe
            # coord). Probe rel_x/rel_y are populated together, but pairing
            # here keeps the contract explicit if only one were present.
            electrode_x = electrode.get("x")
            electrode_y = electrode.get("y")
            if not _missing(rel_x) and not _missing(rel_y):
                plot_x, plot_y, coord_source = rel_x, rel_y, "probe"
            elif not _missing(electrode_x) and not _missing(electrode_y):
                plot_x, plot_y, coord_source = (
                    electrode_x,
                    electrode_y,
                    "electrode",
                )
            else:
                plot_x, plot_y, coord_source = None, None, None
            rows.append(
                {
                    "nwb_file_name": nwb_file_name,
                    "sort_group_id": sort_group_id,
                    "electrode_id": int(electrode["electrode_id"]),
                    "electrode_group_name": str(
                        electrode["electrode_group_name"]
                    ),
                    "probe_id": electrode.get("probe_id"),
                    "probe_shank": _nullable_int(electrode.get("probe_shank")),
                    "probe_electrode": _nullable_int(
                        electrode.get("probe_electrode")
                    ),
                    "brain_region": str(electrode["region_name"]),
                    "bad_channel": str(electrode.get("bad_channel")),
                    "reference_mode": reference_mode,
                    "reference_electrode_id": reference_electrode_id,
                    "is_reference": reference_electrode_id
                    == int(electrode["electrode_id"]),
                    "x": electrode_x,
                    "y": electrode_y,
                    "z": electrode.get("z"),
                    "rel_x": rel_x,
                    "rel_y": rel_y,
                    "rel_z": rel_z,
                    "contact_size": contact_size,
                    "plot_x": plot_x,
                    "plot_y": plot_y,
                    "coordinate_source": coord_source,
                }
            )
    return rows


def plot_sort_group_geometry(
    nwb_file_name: str,
    *,
    ax=None,
    sort_group_ids: list[int] | tuple[int, ...] | set[int] | None = None,
    label_electrodes: bool = False,
    show_bad_channels: bool = True,
    show_reference: bool = True,
    title: str | None = None,
):
    """Plot a DB-backed geometry view of existing v2 sort groups.

    Use this immediately after ``describe_sort_groups`` and before choosing a
    ``sort_group_id``. Contacts are colored by ``sort_group_id``; bad-channel
    members and specific reference electrodes are overlaid when present. The
    helper reads Spyglass metadata only -- it does not open the raw recording or
    create SpikeInterface objects.

    When the session spans **more than one probe**, ``Probe.Electrode``
    rel_x/rel_y are each probe's own coordinate frame (all near the origin), so
    the probes are laid out **side-by-side** along x -- each probe's contacts
    are display-shifted into their own column (annotated with the ``probe_id``)
    and a ``UserWarning`` is emitted. y depths and within-probe geometry are
    unchanged; the underlying ``rel_x`` is not mutated.

    Parameters
    ----------
    nwb_file_name : str
        Session whose existing ``SortGroupV2`` rows should be visualized.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes to draw into. A new figure/axes is created when
        omitted (the default).
    sort_group_ids : list of int or tuple of int or set of int, optional
        Subset of sort-group ids to display. All sort groups are shown
        when omitted (the default).
    label_electrodes : bool, optional
        If ``True``, annotate each plotted contact with its
        ``electrode_id``. Defaults to ``False``.
    show_bad_channels : bool, optional
        If ``True``, overlay bad-channel members with red ``x`` markers.
        Defaults to ``True``.
    show_reference : bool, optional
        If ``True``, overlay ``reference_mode='specific'`` electrodes with a
        black star marker. Defaults to ``True``.
    title : str, optional
        Axes title. A default title naming the session is used when
        omitted.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.

    Warns
    -----
    UserWarning
        When the session spans more than one probe. ``Probe.Electrode``
        rel_x/rel_y are per-probe coordinates, so the probes are laid out
        side-by-side along x (x positions are display-shifted per probe;
        within-probe geometry and y depths are unchanged).
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    def _missing(value) -> bool:
        return value is None or bool(pd.isna(value))

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)

    rows = _sort_group_geometry_rows(nwb_file_name)
    if sort_group_ids is not None:
        wanted = {int(sort_group_id) for sort_group_id in sort_group_ids}
        rows = [row for row in rows if row["sort_group_id"] in wanted]

    if not rows:
        ax.text(
            0.5,
            0.5,
            "No SortGroupV2 rows",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return ax

    plottable = [
        row
        for row in rows
        if not _missing(row["plot_x"]) and not _missing(row["plot_y"])
    ]
    if not plottable:
        ax.text(
            0.5,
            0.5,
            "No plottable electrode geometry",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return ax

    # ``Probe.Electrode`` rel_x/rel_y are each probe's OWN coordinate frame
    # (every probe starts near the origin), so plotting multiple probes on one
    # shared axis would overlap them. Lay probes out side-by-side along x by
    # offsetting each probe's contacts into its own column; y (depth) is left
    # untouched. ``display_x`` carries the (possibly shifted) plot coordinate so
    # the raw per-probe ``plot_x`` is preserved on each row.
    probe_ids = sorted(
        {row["probe_id"] for row in plottable},
        key=lambda probe_id: (probe_id is None, str(probe_id)),
    )
    by_probe = {
        probe_id: [row for row in plottable if row["probe_id"] == probe_id]
        for probe_id in probe_ids
    }
    multi_probe = len(probe_ids) > 1
    if multi_probe:
        import warnings

        # Gap scales with the overall geometry extent so it is non-zero even
        # for single-column (linear) probes whose contacts share one rel_x.
        all_x = [row["plot_x"] for row in plottable]
        all_y = [row["plot_y"] for row in plottable]
        scale = max(
            max(all_x) - min(all_x),
            max(all_y) - min(all_y),
            1.0,
        )
        gap = 0.15 * scale
        cursor = 0.0
        for probe_id in probe_ids:
            probe_rows = by_probe[probe_id]
            xs = [row["plot_x"] for row in probe_rows]
            min_x, max_x = min(xs), max(xs)
            offset = cursor - min_x
            for row in probe_rows:
                row["display_x"] = row["plot_x"] + offset
            cursor += (max_x - min_x) + gap
        warnings.warn(
            f"plot_sort_group_geometry: {len(probe_ids)} probes present. "
            "Probe.Electrode rel_x/rel_y are per-probe coordinates, so the "
            "probes are laid out side-by-side along x (x positions are "
            "display-shifted per probe; within-probe geometry and y depths are "
            "unchanged).",
            UserWarning,
            stacklevel=2,
        )
    else:
        for row in plottable:
            row["display_x"] = row["plot_x"]

    cmap = plt.get_cmap("tab10")
    sort_group_ids = sorted({row["sort_group_id"] for row in plottable})
    for color_index, sort_group_id in enumerate(sort_group_ids):
        group_rows = [
            row for row in plottable if row["sort_group_id"] == sort_group_id
        ]
        color = cmap(color_index % cmap.N)
        ax.scatter(
            [row["display_x"] for row in group_rows],
            [row["plot_y"] for row in group_rows],
            s=50,
            color=color,
            edgecolors="black",
            linewidths=0.35,
            alpha=0.85,
            label=f"sort_group_id {sort_group_id}",
        )

    if show_bad_channels:
        bad_rows = [
            row for row in plottable if str(row["bad_channel"]) == "True"
        ]
        if bad_rows:
            ax.scatter(
                [row["display_x"] for row in bad_rows],
                [row["plot_y"] for row in bad_rows],
                s=90,
                marker="x",
                color="red",
                linewidths=1.2,
                label="bad channel",
            )

    if show_reference:
        reference_rows = [row for row in plottable if row["is_reference"]]
        if reference_rows:
            ax.scatter(
                [row["display_x"] for row in reference_rows],
                [row["plot_y"] for row in reference_rows],
                s=150,
                marker="*",
                facecolors="none",
                edgecolors="black",
                linewidths=1.2,
                label="specific reference",
            )

    if label_electrodes:
        for row in plottable:
            ax.annotate(
                str(row["electrode_id"]),
                (row["display_x"], row["plot_y"]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
            )

    # Label each probe's column so the side-by-side layout is interpretable.
    if multi_probe:
        for probe_id in probe_ids:
            probe_rows = by_probe[probe_id]
            center_x = sum(row["display_x"] for row in probe_rows) / len(
                probe_rows
            )
            top_y = max(row["plot_y"] for row in probe_rows)
            ax.annotate(
                str(probe_id),
                (center_x, top_y),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                fontweight="bold",
            )

    coordinate_sources = {
        row["coordinate_source"]
        for row in plottable
        if row["coordinate_source"]
    }
    x_suffix = ", offset per probe" if multi_probe else ""
    if coordinate_sources == {"probe"}:
        ax.set_xlabel(f"Probe rel_x (um{x_suffix})")
        ax.set_ylabel("Probe rel_y (um)")
    elif coordinate_sources == {"electrode"}:
        ax.set_xlabel(f"Electrode x (um{x_suffix})")
        ax.set_ylabel("Electrode y (um)")
    else:
        ax.set_xlabel(f"x coordinate (um{x_suffix})")
        ax.set_ylabel("y coordinate (um)")

    missing_count = len(rows) - len(plottable)
    plot_title = title or f"Sort groups for {nwb_file_name}"
    if missing_count:
        plot_title = f"{plot_title} ({missing_count} contact(s) hidden)"
    ax.set_title(plot_title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.25)

    ax.legend(fontsize="small", loc="best")
    return ax


# Production region preproc + rate-keyed MS4 sorter rows the franklab MS4
# presets bundle. probe_type is informational (the recipe is region + rate),
# so the tetrode- and probe-hippocampus-30 kHz presets resolve identically.
_REGION_PREPROC = {
    "hippocampus": "franklab_hippocampus_2026_06",
    "cortex": "franklab_cortex_2026_06",
}
_RATE_MS4_SORTER = {
    30000: "franklab_30khz_ms4_2026_06",
    20000: "franklab_20khz_ms4_2026_06",
}
_MS4_THRESHOLD_UNITS = "sigma of the whitened signal (~3)"
_MS4_NOTES = (
    "MountainSort4 detect_threshold is a multiple of the standard deviation "
    "of the ZCA-whitened signal (~3), not an absolute voltage and not a MAD "
    "multiplier. MS4 oversplits and does not track drift, so merge curation "
    "is expected (Kilosort is the Neuropixels-density alternative). Runtime "
    "note: MS4's algorithm backend (ml_ms4alg) is a numpy<2-era package that "
    "does not install under the v2 numpy>=2 baseline, so these presets need a "
    "numpy<2 environment; the shipped run_v2_pipeline default is the "
    "MountainSort5 recipe, which runs as-is (preflight reports this via the "
    "sorter_runtime_available check)."
)


def _franklab_ms4_preset(
    probe_type: str, region: str, rate_hz: int
) -> "_PipelinePreset":
    """Build a production MS4 preset for a (probe_type, region, rate)."""
    return _PipelinePreset(
        preprocessing_params_name=_REGION_PREPROC[region],
        artifact_detection_params_name="franklab_100uv_p07_2026_06",
        sorter="mountainsort4",
        sorter_params_name=_RATE_MS4_SORTER[rate_hz],
        probe_type=probe_type,
        target_region=region,
        sampling_rate_hz=rate_hz,
        sorter_family="mountainsort4",
        adjacency_radius_um=100.0,
        recommendation_status="production",
        intended_use=(
            f"Frank-lab {region} {probe_type}s at {rate_hz // 1000} kHz "
            "(MountainSort4 production recipe)."
        ),
        threshold_units=_MS4_THRESHOLD_UNITS,
        notes=_MS4_NOTES,
    )


_PIPELINE_PRESETS: dict[str, _PipelinePreset] = {
    "franklab_tetrode_hippocampus_30khz_ms4_2026_06": _franklab_ms4_preset(
        "tetrode", "hippocampus", 30000
    ),
    "franklab_probe_hippocampus_30khz_ms4_2026_06": _franklab_ms4_preset(
        "probe", "hippocampus", 30000
    ),
    "franklab_probe_cortex_30khz_ms4_2026_06": _franklab_ms4_preset(
        "probe", "cortex", 30000
    ),
    "franklab_probe_hippocampus_20khz_ms4_2026_06": _franklab_ms4_preset(
        "probe", "hippocampus", 20000
    ),
    "franklab_probe_cortex_20khz_ms4_2026_06": _franklab_ms4_preset(
        "probe", "cortex", 20000
    ),
    "franklab_tetrode_hippocampus_30khz_ms5_2026_06": _PipelinePreset(
        preprocessing_params_name="franklab_hippocampus_2026_06",
        artifact_detection_params_name="franklab_100uv_p07_2026_06",
        sorter="mountainsort5",
        sorter_params_name="franklab_30khz_ms5_2026_06",
        probe_type="tetrode",
        target_region="hippocampus",
        sampling_rate_hz=30000,
        sorter_family="mountainsort5",
        recommendation_status="alternative",
        intended_use=(
            "Frank-lab hippocampal tetrodes at 30 kHz, MountainSort5 -- the "
            "shipped run_v2_pipeline default because it runs under the v2 "
            "numpy>=2 baseline (the MS4 production recipe needs numpy<2)."
        ),
        threshold_units="sigma of the whitened signal (~5.5)",
        notes=(
            "MountainSort5 detect_threshold is a multiple of the standard "
            "deviation of the whitened signal (~5.5, more conservative than "
            "MS4's 3) -- the same sigma scale, not a MAD multiplier. MS5 is the "
            "shipped run_v2_pipeline default because it runs under numpy>=2; "
            "MS4 is the Frank-lab production recipe but its ml_ms4alg backend "
            "needs numpy<2. recommendation_status stays 'alternative' (MS5 has "
            "no attested probe usage); the function default is a separate, "
            "runnability-driven choice."
        ),
    ),
    "franklab_clusterless_2026_06": _PipelinePreset(
        preprocessing_params_name="default",
        artifact_detection_params_name="default",
        sorter="clusterless_thresholder",
        sorter_params_name="default",
        target_region="hippocampus",
        sorter_family="clusterless_thresholder",
        recommendation_status="production",
        intended_use=(
            "Peak detection only (no clustering); feeds the clusterless "
            "decoding pipeline."
        ),
        threshold_units="µV (100 µV)",
        notes=(
            "The 'default' clusterless SorterParameters row sets "
            "threshold_unit='uv' with detect_threshold=100, so traces are "
            "scaled to microvolts before detection -- a true 100 µV "
            "threshold, not a MAD multiplier. Preproc/artifact rows are "
            "unchanged from the prior clusterless preset."
        ),
    ),
    "franklab_neuropixels_ks4_2026_06": _PipelinePreset(
        preprocessing_params_name="default_neuropixels",
        artifact_detection_params_name="none",
        sorter="kilosort4",
        sorter_params_name="franklab_neuropixels_default",
        probe_type="neuropixels",
        sampling_rate_hz=30000,
        sorter_family="kilosort4",
        recommendation_status="experimental",
        intended_use=(
            "Neuropixels (30 kHz) with Kilosort4, matched to the AIND "
            "aind-ephys-spikesort-kilosort4 recipe. Experimental -- "
            "community-grounded, not Frank-lab-attested; KS4 is "
            "non-deterministic and typically needs a GPU."
        ),
        threshold_units=(
            "KS4 template-matching projection thresholds "
            "(Th_universal=9 / Th_learned=8), not µV or a noise multiple"
        ),
        notes=(
            "Sorter params match the AIND aind-ephys-spikesort-kilosort4 "
            "params.json (and agree with int-brain-lab/ibl-sorter): the only "
            "scientifically-meaningful deviation from stock KS4 is non-rigid "
            "drift correction (nblocks=5 vs stock 1). KS4 does its own "
            "high-pass + common-reference + ZCA whitening internally "
            "(skip_kilosort_preprocessing=False, whitening_range=32); the row "
            "carries no 'whiten' key, so v2's external float64 whitening stays "
            "off and the signal is whitened EXACTLY ONCE (by KS4). The preproc "
            "row applies the ADC phase-shift KS4 cannot, plus a bandpass; "
            "because KS4 also common-references (do_CAR=true), set the sort "
            "group's reference_mode='none' to avoid double-referencing before "
            "KS4. Artifact detection is 'none' (KS4's internal preprocessing "
            "and drift handling stand in for amplitude masking)."
        ),
    ),
}


# SpikeInterface's ``installed_sorters()`` reports a sorter as installed when
# its thin wrapper imports, but some sorters call a separate algorithm backend
# at run time that the wrapper does NOT import -- so the check over-reports. The
# live example is ``mountainsort4``: its wrapper imports (so it appears in
# ``installed_sorters()``), but the actual algorithm package ``ml_ms4alg`` is a
# numpy<2-era build that no longer installs under the v2 ``numpy>=2`` baseline.
# Map each such sorter to the backend module(s) preflight must additionally
# verify, so a green ``sorter_installed`` cannot precede a sort-time
# ``ModuleNotFoundError``.
_SORTER_RUNTIME_BACKENDS: dict[str, tuple[str, ...]] = {
    "mountainsort4": ("ml_ms4alg",),
}


@dataclass(frozen=True)
class PreflightCheck:
    """One preflight check outcome: name, pass/fail, and the fix on fail."""

    name: str  # e.g. "session_exists", "sorter_installed"
    ok: bool
    fix: str  # empty when ok; the actionable fix when not ok


@dataclass(frozen=True)
class PreflightReport:
    """Result of ``preflight_v2_pipeline``: a pre-populate config check.

    Truthy when the configuration is runnable (``ok is True``), so a
    notebook can ``if not preflight_v2_pipeline(...): ...``. ``errors``
    lists each blocking problem with its fix; ``warnings`` holds
    non-blocking advisories; ``expected_ids`` carries the deterministic
    selection PKs the run would produce.

    Attributes
    ----------
    ok
        True when no blocking problem was found (``errors`` is empty).
    errors
        Blocking-problem messages; non-empty iff ``ok`` is False.
    warnings
        Non-blocking advisories (e.g. ``artifact_detection_params_name="none"``).
    resolved_pipeline_preset
        The pipeline-preset name that was checked.
    expected_ids
        The selection PKs a subsequent ``run_v2_pipeline`` would produce,
        each annotated with whether the row already exists, e.g.
        ``{"recording_id": {"id": UUID(...), "exists": False}, ...}``. IDs
        are computed DB-free via ``deterministic_id``; ``exists`` is a
        ``& pk`` restriction. Empty when the preset is unknown (the param
        names needed to derive the IDs are then unavailable). For an ``ok``
        report each ``id`` equals the PK ``run_v2_pipeline`` returns.
        ``curation_id`` is intentionally excluded: it is assigned by
        ``CurationV2.insert_curation``, not content-addressed.
    checks
        Per-check detail; every check runs (the report is complete, not
        first-failure-only).
    """

    ok: bool
    errors: list[str]
    warnings: list[str]
    resolved_pipeline_preset: str
    expected_ids: dict
    checks: list["PreflightCheck"]

    def __bool__(self) -> bool:
        """Return ``True`` when the configuration is runnable (``ok``)."""
        return self.ok


@dataclass(frozen=True)
class PreflightSessionReport:
    """Result of ``preflight_v2_pipeline_session``: a whole-session check.

    Aggregates a per-sort-group :class:`PreflightReport` into one object.
    Truthy when *every* target group is runnable (``ok is True``), so a
    notebook can ``if not preflight_v2_pipeline_session(...): ...``.

    Attributes
    ----------
    ok
        True when no target group has a blocking problem (``errors`` empty).
    errors
        Blocking-problem messages across all groups, each prefixed with its
        ``sort_group_id``; non-empty iff ``ok`` is False.
    warnings
        Non-blocking advisories across all groups, each prefixed with its
        ``sort_group_id``.
    resolved_pipeline_preset
        The pipeline-preset name that was checked for every group.
    group_reports
        One plain-dict entry per target sort group, with keys
        ``sort_group_id``, ``ok``, ``errors``, ``warnings``, ``expected_ids``,
        and ``checks`` (from the underlying :class:`PreflightReport`).
        ``pandas`` is intentionally not imported here; wrap with
        ``pd.DataFrame(report.group_reports)`` in a notebook when useful.
    """

    ok: bool
    errors: list[str]
    warnings: list[str]
    resolved_pipeline_preset: str
    group_reports: list[dict[str, Any]]

    def __bool__(self) -> bool:
        """Return ``True`` when every target group is runnable (``ok``)."""
        return self.ok


def preflight_v2_pipeline(
    nwb_file_name: str,
    sort_group_id: int,
    interval_list_name: str,
    team_name: str,
    pipeline_preset: str = "franklab_tetrode_hippocampus_30khz_ms5_2026_06",
) -> PreflightReport:
    """Read-only pre-populate configuration check for ``run_v2_pipeline``.

    Verifies -- in ~1 s, inserting nothing and never calling ``populate``
    -- that every prerequisite a subsequent ``run_v2_pipeline(...,
    pipeline_preset=pipeline_preset)`` needs is in place: the session /
    interval / team / sort-group rows exist, the pipeline preset's
    parameter Lookup rows exist, and
    the sorter binary is installed. Returns a structured
    :class:`PreflightReport` instead of failing minutes into ``populate``
    with an opaque foreign-key or SpikeInterface error.

    Every check is a read-only restriction (``& {...}``) or a pure call;
    all checks run even after one fails, so the report lists every problem
    at once. An unknown ``pipeline_preset`` short-circuits before any database
    access (the later checks need the resolved param names).

    Parameters
    ----------
    nwb_file_name, sort_group_id, interval_list_name, team_name, pipeline_preset
        The same inputs as :func:`run_v2_pipeline`.

    Returns
    -------
    PreflightReport
        Truthy when the configuration is runnable. ``report.errors`` lists
        each blocking problem with the action to fix it;
        ``report.expected_ids`` holds the deterministic selection PKs the
        run would produce (see :class:`PreflightReport`).
    """
    checks: list[PreflightCheck] = []
    warnings: list[str] = []

    def _check(name: str, ok, fix: str) -> bool:
        ok = bool(ok)
        checks.append(PreflightCheck(name, ok, "" if ok else fix))
        return ok

    # 1. pipeline_preset_known. Short-circuit before any DB access on failure: the
    # remaining checks (and expected_ids) all derive from the resolved
    # bundle's param names, which are unknown for a bogus pipeline preset.
    if pipeline_preset not in _PIPELINE_PRESETS:
        _check(
            "pipeline_preset_known",
            False,
            f"unknown pipeline_preset {pipeline_preset!r}. Available pipeline presets: "
            f"{sorted(_PIPELINE_PRESETS)}. Call describe_pipeline_presets() to see what each "
            "one does.",
        )
        return PreflightReport(
            ok=False,
            errors=[c.fix for c in checks if not c.ok],
            warnings=warnings,
            resolved_pipeline_preset=pipeline_preset,
            expected_ids={},
            checks=checks,
        )
    _check("pipeline_preset_known", True, "")
    bundle = _PIPELINE_PRESETS[pipeline_preset]

    import spikeinterface.sorters as sis

    from spyglass.common import IntervalList, LabTeam, Raw, Session
    from spyglass.spikesorting.v2._selection_identity import (
        artifact_detection_identity_payload,
        deterministic_id,
        recording_identity_payload,
        sorting_identity_payload,
    )
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetectionParameters,
        ArtifactDetectionSelection,
    )
    from spyglass.spikesorting.v2.recording import (
        PreprocessingParameters,
        RecordingSelection,
        SortGroupV2,
    )
    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        SortingSelection,
    )

    sort_group_id = int(sort_group_id)

    # 2-6. Upstream session/raw/interval/team/sort-group rows.
    _check(
        "session_exists",
        Session & {"nwb_file_name": nwb_file_name},
        f"session {nwb_file_name!r} is not ingested. Ingest it with "
        "insert_sessions(...) first.",
    )
    # ``RecordingSelection`` FKs ``Raw`` (not ``Session``), so a session whose
    # ``Raw`` row is missing (e.g. a partial ingestion) would pass
    # ``session_exists`` yet fail the recording insert with an opaque
    # foreign-key error. Check ``Raw`` explicitly so preflight stays honest.
    _check(
        "raw_exists",
        Raw & {"nwb_file_name": nwb_file_name},
        f"Raw electrical-series row for {nwb_file_name!r} is missing (the "
        "session is ingested but its Raw data is not). Re-run ingestion "
        "(e.g. populate_all_common / insert_sessions) so Raw is populated.",
    )
    _check(
        "interval_exists",
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": interval_list_name,
        },
        f"interval_list_name {interval_list_name!r} not found for "
        f"{nwb_file_name!r}. A full-session sort typically uses "
        "'raw data valid times'.",
    )
    _check(
        "team_exists",
        LabTeam & {"team_name": team_name},
        f"LabTeam {team_name!r} does not exist. Create it with "
        "LabTeam.insert1({'team_name': ..., 'team_description': ...}).",
    )
    sort_group_exists = _check(
        "sort_group_exists",
        SortGroupV2
        & {"nwb_file_name": nwb_file_name, "sort_group_id": sort_group_id},
        f"SortGroupV2 sort_group_id={sort_group_id} not found for "
        f"{nwb_file_name!r}. Create sort groups first with "
        "SortGroupV2.set_group_by_shank(nwb_file_name=...).",
    )
    # A SortGroupV2 master row can exist with ZERO electrode members (created
    # then partially deleted, or a shank that resolved to an empty group);
    # Recording.populate then raises "has zero electrodes" minutes into the
    # run. Checking the master alone is a false-green. Only run this when the
    # master exists, to avoid a confusing second failure when it is absent.
    if sort_group_exists:
        _check(
            "sort_group_has_electrodes",
            SortGroupV2.SortGroupElectrode
            & {
                "nwb_file_name": nwb_file_name,
                "sort_group_id": sort_group_id,
            },
            f"SortGroupV2 sort_group_id={sort_group_id} for {nwb_file_name!r} "
            "has zero electrode members; Recording.populate would raise 'has "
            "zero electrodes'. Recreate it with "
            "SortGroupV2.set_group_by_shank(nwb_file_name=...).",
        )

    # 6-8. The preset's parameter Lookup rows.
    _check(
        "preprocessing_params_exist",
        PreprocessingParameters
        & {"preprocessing_params_name": bundle.preprocessing_params_name},
        f"PreprocessingParameters row {bundle.preprocessing_params_name!r} is "
        "missing. Run initialize_v2_defaults().",
    )
    _check(
        "artifact_detection_params_exist",
        ArtifactDetectionParameters
        & {
            "artifact_detection_params_name": bundle.artifact_detection_params_name
        },
        f"ArtifactDetectionParameters row {bundle.artifact_detection_params_name!r} "
        "is missing. Run initialize_v2_defaults().",
    )
    _check(
        "sorter_params_exist",
        SorterParameters
        & {
            "sorter": bundle.sorter,
            "sorter_params_name": bundle.sorter_params_name,
        },
        f"SorterParameters row (sorter={bundle.sorter!r}, "
        f"sorter_params_name={bundle.sorter_params_name!r}) is missing. "
        "Run initialize_v2_defaults().",
    )

    # 8b. sampling_rate_matches. The MS4/MS5 snippet window (clip_size /
    # detect_interval on the rate-keyed sorter row) assumes a specific
    # acquisition rate, so a 30 kHz preset on a 20 kHz recording (or the
    # reverse) silently sorts with a mistuned window. The clusterless preset
    # is rate-agnostic (sampling_rate_hz is None) and is skipped; the check is
    # also skipped if Raw is not ingested yet (raw_exists already reports that,
    # so this would only add a confusing second failure).
    if bundle.sampling_rate_hz is not None:
        raw = Raw & {"nwb_file_name": nwb_file_name}
        if raw:
            actual_rate = float(raw.fetch1("sampling_rate"))
            # 0.5% tolerance absorbs float drift in an estimated rate.
            rate_ok = (
                abs(actual_rate - bundle.sampling_rate_hz)
                <= 0.005 * bundle.sampling_rate_hz
            )
            _check(
                "sampling_rate_matches",
                rate_ok,
                f"recording {nwb_file_name!r} samples at {actual_rate:g} Hz "
                f"but pipeline_preset {pipeline_preset!r} is tuned for "
                f"{bundle.sampling_rate_hz} Hz: the rate-keyed sorter row "
                f"{bundle.sorter_params_name!r} holds its clip_size / "
                "detect_interval snippet window at that rate. Pick the "
                "rate-matched preset (call describe_pipeline_presets() and "
                "match sampling_rate_hz to the recording).",
            )

    # 9. sorter_installed. Reuse the SAME strict gate insert_default uses:
    # the internal clusterless_thresholder (_NON_SI_SORTERS) is never an SI
    # binary and is always available; otherwise the sorter must be in
    # installed_sorters(), distinguishing "known but not installed" from
    # "misspelled / unknown" for the fix message.
    sorter_installed_ok = (
        bundle.sorter in SorterParameters._NON_SI_SORTERS
        or bundle.sorter in set(sis.installed_sorters())
    )
    if sorter_installed_ok:
        _check("sorter_installed", True, "")
    elif bundle.sorter in set(sis.available_sorters()):
        _check(
            "sorter_installed",
            False,
            f"sorter {bundle.sorter!r} is a known SpikeInterface sorter but "
            "its binary/runtime is not installed here "
            "(spikeinterface.sorters.installed_sorters()). Install it, or "
            "pick a preset whose sorter is installed.",
        )
    else:
        _check(
            "sorter_installed",
            False,
            f"sorter {bundle.sorter!r} is not a known SpikeInterface sorter "
            "(spikeinterface.sorters.available_sorters()) -- check the "
            "spelling or the preset.",
        )

    # 9b. sorter_runtime_available. installed_sorters() only checks that the SI
    # wrapper imports; for sorters that call a SEPARATE algorithm backend at run
    # time (see _SORTER_RUNTIME_BACKENDS) actually import that backend, so a
    # green sorter_installed cannot precede a sort-time failure -- whether the
    # backend is absent OR present-but-broken (e.g. a numpy<2-era ml_ms4alg
    # under the numpy>=2 baseline raising at import). Only runs when
    # sorter_installed passed: if the wrapper itself is missing, a second
    # "backend missing" failure would be contradictory ("listed as installed").
    backend_modules = _SORTER_RUNTIME_BACKENDS.get(bundle.sorter, ())
    if sorter_installed_ok and backend_modules:
        import importlib

        broken_backends = []
        for mod in backend_modules:
            try:
                importlib.import_module(mod)
            except Exception as exc:  # noqa: BLE001 - any import failure disqualifies
                broken_backends.append(f"{mod} ({type(exc).__name__}: {exc})")
        _check(
            "sorter_runtime_available",
            not broken_backends,
            f"sorter {bundle.sorter!r} is listed as installed but its runtime "
            "backend(s) cannot be imported, so the sort would crash: "
            f"{'; '.join(broken_backends)}. Install/repair the backend "
            "(mountainsort4 needs ml_ms4alg, which requires numpy<2), or pick a "
            "preset whose sorter runs in this environment (e.g. a MountainSort5 "
            "preset).",
        )

    # Non-blocking advisory: the "none" artifact params are a no-op
    # pass-through (no masking). "default" performs real amplitude-threshold
    # detection and is the legitimate built-in choice, so it is NOT warned.
    if bundle.artifact_detection_params_name == "none":
        warnings.append(
            "artifact_detection_params_name='none': no artifact masking will be "
            "applied for this run."
        )

    # expected_ids: the deterministic selection PKs this run would produce.
    # Pure hashes of (preset params + inputs) via the SAME payload builders
    # insert_selection uses, so they cannot drift; computable once the
    # preset resolves, regardless of whether the rows exist yet. ``exists``
    # is a read-only & pk check.
    recording_id = deterministic_id(
        "recording",
        recording_identity_payload(
            {
                "nwb_file_name": nwb_file_name,
                "sort_group_id": sort_group_id,
                "interval_list_name": interval_list_name,
                "preprocessing_params_name": bundle.preprocessing_params_name,
                "team_name": team_name,
            }
        ),
    )
    artifact_detection_id = deterministic_id(
        "artifact_detection",
        artifact_detection_identity_payload(
            artifact_detection_params_name=bundle.artifact_detection_params_name,
            recording_id=recording_id,
        ),
    )
    sorting_id = deterministic_id(
        "sorting",
        sorting_identity_payload(
            recording_id=recording_id,
            sorter=bundle.sorter,
            sorter_params_name=bundle.sorter_params_name,
            artifact_detection_id=artifact_detection_id,
        ),
    )
    expected_ids = {
        "recording_id": {
            "id": recording_id,
            "exists": bool(RecordingSelection & {"recording_id": recording_id}),
        },
        "artifact_detection_id": {
            "id": artifact_detection_id,
            "exists": bool(
                ArtifactDetectionSelection
                & {"artifact_detection_id": artifact_detection_id}
            ),
        },
        "sorting_id": {
            "id": sorting_id,
            "exists": bool(SortingSelection & {"sorting_id": sorting_id}),
        },
    }

    errors = [c.fix for c in checks if not c.ok]
    return PreflightReport(
        ok=not errors,
        errors=errors,
        warnings=warnings,
        resolved_pipeline_preset=pipeline_preset,
        expected_ids=expected_ids,
        checks=checks,
    )


def _resolve_session_sort_group_ids(
    nwb_file_name: str,
    pipeline_preset: "str | None",
    sort_group_ids: "list[int] | None",
    caller: str,
) -> list[int]:
    """Validate session-runner inputs; return the target ``sort_group_id`` list.

    Shared by :func:`preflight_v2_pipeline_session` and
    :func:`run_v2_pipeline_session` so their input validation -- and its
    wording -- is identical. Unlike the single-group helpers, the session
    helpers do **not** infer a default preset: an explicit ``pipeline_preset``
    is required. The preset checks run before any ``SortGroupV2`` access, so an
    unknown/missing preset is rejected DB-free.

    Parameters
    ----------
    nwb_file_name
        Session whose ``SortGroupV2`` rows define the candidate targets.
    pipeline_preset
        Required pipeline-preset name; ``None`` is rejected.
    sort_group_ids
        Optional explicit subset. ``None`` means "every sort group in the
        session". Normalized to a sorted, de-duplicated ``list[int]``.
    caller
        Name of the calling helper, used to prefix error messages.

    Returns
    -------
    list[int]
        Target ``sort_group_id`` values in ascending order.

    Raises
    ------
    PipelineInputError
        If ``pipeline_preset`` is ``None`` or unknown, the session has no
        ``SortGroupV2`` rows, or a requested ``sort_group_ids`` entry is
        absent.
    """
    from spyglass.spikesorting.v2.exceptions import PipelineInputError

    if pipeline_preset is None:
        raise PipelineInputError(
            f"{caller}: pipeline_preset is required -- a whole-session run does "
            "not infer a default. Call describe_pipeline_presets() to choose "
            "one, then pass pipeline_preset=..."
        )
    if pipeline_preset not in _PIPELINE_PRESETS:
        raise PipelineInputError(
            f"{caller}: unknown pipeline_preset {pipeline_preset!r}. Available "
            f"pipeline presets: {sorted(_PIPELINE_PRESETS)}. Call "
            "describe_pipeline_presets() to see what each preset does."
        )

    from spyglass.spikesorting.v2.recording import SortGroupV2

    available = sorted(
        int(g)
        for g in (SortGroupV2 & {"nwb_file_name": nwb_file_name}).fetch(
            "sort_group_id"
        )
    )
    if not available:
        raise PipelineInputError(
            f"{caller}: no SortGroupV2 rows for {nwb_file_name!r}. Create sort "
            "groups first with "
            "SortGroupV2.set_group_by_shank(nwb_file_name=...)."
        )

    if sort_group_ids is None:
        return available

    available_set = set(available)
    requested = sorted({int(g) for g in sort_group_ids})
    missing = [g for g in requested if g not in available_set]
    if missing:
        raise PipelineInputError(
            f"{caller}: sort_group_ids {missing} not found for "
            f"{nwb_file_name!r}. Available sort_group_ids: {available}."
        )
    return requested


def preflight_v2_pipeline_session(
    nwb_file_name: str,
    interval_list_name: str,
    team_name: str,
    pipeline_preset: "str | None" = None,
    sort_group_ids: "list[int] | None" = None,
) -> PreflightSessionReport:
    """Read-only preflight for every target sort group in a session.

    Runs :func:`preflight_v2_pipeline` once per target ``SortGroupV2`` row and
    aggregates the per-group reports. Read-only and cheap: it inserts nothing,
    calls no ``populate``, and -- unlike the single-group helper -- requires an
    explicit ``pipeline_preset`` (it infers no default). It reuses the
    single-group checks rather than duplicating them, so the two helpers cannot
    drift.

    Parameters
    ----------
    nwb_file_name, interval_list_name, team_name, pipeline_preset
        The same inputs as :func:`run_v2_pipeline`, except ``pipeline_preset``
        is required (no default).
    sort_group_ids
        Optional explicit subset of sort groups to check. ``None`` (default)
        checks every ``SortGroupV2`` row for the session.

    Returns
    -------
    PreflightSessionReport
        Truthy when every target group is runnable. ``report.group_reports``
        holds one plain-dict entry per group; ``report.errors`` /
        ``report.warnings`` aggregate the per-group messages, each prefixed
        with its ``sort_group_id``.

    Raises
    ------
    PipelineInputError
        From the shared target resolver: ``pipeline_preset`` is ``None`` or
        unknown, the session has no sort groups, or a requested
        ``sort_group_ids`` entry is absent. This is *not* swallowed -- a
        misconfigured request is a caller error, not a per-group preflight
        failure.
    """
    targets = _resolve_session_sort_group_ids(
        nwb_file_name=nwb_file_name,
        pipeline_preset=pipeline_preset,
        sort_group_ids=sort_group_ids,
        caller="preflight_v2_pipeline_session",
    )

    group_reports: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []
    for sort_group_id in targets:
        report = preflight_v2_pipeline(
            nwb_file_name=nwb_file_name,
            sort_group_id=sort_group_id,
            interval_list_name=interval_list_name,
            team_name=team_name,
            pipeline_preset=pipeline_preset,
        )
        group_reports.append(
            {
                "sort_group_id": sort_group_id,
                "ok": report.ok,
                "errors": report.errors,
                "warnings": report.warnings,
                "expected_ids": report.expected_ids,
                "checks": report.checks,
            }
        )
        errors.extend(
            f"sort_group_id={sort_group_id}: {e}" for e in report.errors
        )
        warnings.extend(
            f"sort_group_id={sort_group_id}: {w}" for w in report.warnings
        )

    return PreflightSessionReport(
        ok=not errors,
        errors=errors,
        warnings=warnings,
        resolved_pipeline_preset=pipeline_preset,
        group_reports=group_reports,
    )


# Closed vocabulary for the per-stage ``*_status`` run-summary keys. A stage is
# ``"computed"`` when its row did not exist before this call and populate /
# insert_curation created it this call; ``"reused"`` when the row already
# existed and the call no-opped. Test code asserts each status is a member.
_STAGE_STATUSES = frozenset({"computed", "reused"})


def _run_stage(
    stage: str, exists: bool, work, partial: dict
) -> tuple[Any, str, float]:
    """Time a pipeline stage's ``work()``; classify it; wrap failures.

    ``exists`` is the result of a pre-``work`` existence check on the stage's
    output row, so the status is ``"reused"`` when the row was already present
    (``work`` no-ops) and ``"computed"`` otherwise. ``work`` is a zero-arg
    closure over the stage's ``populate`` / ``insert_curation`` call; its
    return value is passed back (the curation stage needs the returned key).
    A failure is re-raised as a chained :class:`PipelineStageError` carrying a
    snapshot of ``partial`` (the run summary accumulated from earlier stages)
    so
    the caller sees which stage broke and what was already built.

    Returns ``(work_result, status, seconds)`` where ``seconds`` is monotonic
    wall-clock spent in ``work`` THIS call (≈0 on a reused no-op), not
    cumulative compute cost.
    """
    from spyglass.spikesorting.v2.exceptions import PipelineStageError

    status = "reused" if exists else "computed"
    start = time.perf_counter()
    try:
        result = work()
    except Exception as exc:  # noqa: BLE001 - re-raised as typed + chained
        raise PipelineStageError(stage, dict(partial), str(exc)) from exc
    return result, status, time.perf_counter() - start


def run_v2_pipeline(
    nwb_file_name: str,
    sort_group_id: int,
    interval_list_name: str,
    team_name: str,
    pipeline_preset: str = "franklab_tetrode_hippocampus_30khz_ms5_2026_06",
    description: str = "",
    require_units: bool = False,
    preflight: bool = True,
) -> dict[str, Any]:
    """End-to-end single-session sort: recording -> artifact detection -> sort -> curation.

    Chains the v2 ``insert_selection`` + ``populate`` calls into one
    call. Idempotent: re-running with the same inputs returns the same
    run summary (same merge_id, same intermediate PKs) without
    duplicating rows.

    Prerequisites (set these up first, in order)
    --------------------------------------------
    1. ``initialize_v2_defaults()`` -- seed the default Lookup rows.
    2. ``LabTeam`` row for ``team_name`` -- the owning team must already
       exist in ``common.LabTeam``.
    3. ``SortGroupV2.set_group_by_shank(nwb_file_name=...)`` (or
       ``set_group_by_electrode_table_column``) -- sort-group structure is
       session-specific user input the orchestrator does not auto-create.

    So a single-session sort is ~4 user touchpoints (the three setup steps
    above plus this call), not 2: this orchestrator collapses the per-stage
    ``insert_selection`` / ``populate`` boilerplate, not the upstream
    session/team/sort-group setup. With ``preflight=True`` (the default)
    this call verifies those prerequisites in ~1 s before any populate and
    raises ``PreflightError`` with the exact fix if one is missing; call
    ``preflight_v2_pipeline(...)`` directly to inspect the report without
    running.

    Parameters
    ----------
    nwb_file_name
        Session whose data will be sorted. The session must already be
        ingested via ``insert_sessions``.
    sort_group_id
        ID of an existing ``SortGroupV2`` row for this session.
        Callers create sort groups via
        ``SortGroupV2.set_group_by_shank`` (or
        ``set_group_by_electrode_table_column``) before calling this
        helper; the orchestrator does not auto-create them because
        sort-group structure is session-specific user input.
    interval_list_name
        Name of the IntervalList row to sort. Typically ``"raw data
        valid times"`` for a full-session sort.
    team_name
        LabTeam owning the sort. Must already exist in
        ``common.LabTeam``.
    pipeline_preset
        Pipeline-preset name from ``_PIPELINE_PRESETS``. The default is
        ``franklab_tetrode_hippocampus_30khz_ms5_2026_06`` (MountainSort5),
        which runs under the v2 ``numpy>=2`` baseline; the MountainSort4
        production recipe is selectable but its ``ml_ms4alg`` backend needs
        ``numpy<2`` (preflight reports this via ``sorter_runtime_available``).
        Call ``describe_pipeline_presets()`` for a table of what each one does
        (sorter, parameter rows, intended use, and threshold units), or
        ``list_pipeline_presets()`` for just the names.
    description
        Free-text description passed to ``CurationV2.insert_curation``.
    require_units
        If False (default), a sort that finds zero units still produces
        an EMPTY (but real) curation + merge row, with a loud warning --
        zero units is a legitimate result on a quiet shank, and the empty
        row lets downstream code treat it like any other
        ``SpikeSortingOutput`` row. If True, a zero-unit sort raises
        ``ZeroUnitSortError`` instead (for callers that treat zero units
        as a hard error).
    preflight
        If True (default), run ``preflight_v2_pipeline`` first as a fast,
        read-only check that the session / interval / team / sort-group
        rows, the pipeline preset's parameter rows, and the sorter binary are all
        present; a failure raises ``PreflightError`` (with the exact fix)
        before any populate. Pass ``preflight=False`` to skip the check and
        attempt the run directly (e.g. to see the raw underlying error).

    Returns
    -------
    dict
        Run summary with the following stage keys:
            ``pipeline_preset``          : the pipeline-preset name
            ``recording_id``             : RecordingSelection PK
            ``artifact_detection_id``    : ArtifactDetectionSelection PK
            ``sorting_id``               : SortingSelection PK
            ``curation_id``              : CurationV2 PK
            ``merge_id``                 : SpikeSortingOutput master PK
            ``n_units``                  : unit count (0 on a zero-unit sort)
        Downstream consumers should key off ``merge_id``. A zero-unit
        sort yields an empty curation/merge row (matching v1's empty
        Units table), not ``None``, so the result is always
        merge-keyable.

        Plus per-stage observability keys (additive; the keys above are
        unchanged):
            ``recording_status`` / ``artifact_detection_status`` /
            ``sorting_status`` / ``curation_status`` : ``"computed"`` if the stage did work
                this call, ``"reused"`` if its row already existed and the
                call no-opped (see ``_STAGE_STATUSES``).
            ``stage_seconds``     : dict of monotonic wall-clock seconds
                spent per stage (keys ``"recording"`` / ``"artifact_detection"``
                / ``"sorting"`` / ``"curation"``) **this call** -- ≈0 on an
                idempotent re-run, NOT cumulative compute cost.
            ``warnings``          : list of human-readable advisories raised
                during the run (e.g. the zero-unit message); empty when
                clean.
        Two identical calls return equal run summaries except for
        ``stage_seconds`` and the ``*_status`` values (the second reports
        ``"reused"``), inserting no duplicate rows.

    Raises
    ------
    PipelineInputError
        If ``pipeline_preset`` is not a known name.
    PreflightError
        If ``preflight=True`` and a prerequisite is missing (the message
        lists every failed check and its fix). Bypass with
        ``preflight=False``.
    PipelineStageError
        If a compute stage's ``populate`` / ``insert_curation`` fails. Names
        the failing stage and carries the partial run summary of the stages
        that completed before it (the original error is chained). Only the
        compute
        stages are wrapped; an error from the cheap ``insert_selection``
        prelude surfaces as its own native exception (e.g.
        ``DuplicateSelectionError``).
    ZeroUnitSortError
        If the sort finds zero units and ``require_units=True``.
    ValueError
        If a required parameter Lookup row is missing (e.g.
        ``PreprocessingParameters`` / ``SorterParameters`` defaults not
        installed); the insert helpers translate the would-be
        foreign-key error into this clear message. Run
        ``initialize_v2_defaults()`` first.
    datajoint.errors.IntegrityError
        If an upstream sort group / session / interval list / team does
        not exist when ``preflight=False`` -- the foreign-key violation
        surfaces untranslated. ``preflight=True`` catches these earlier
        as a ``PreflightError`` with the exact fix.
    """
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetection,
        ArtifactDetectionSelection,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.exceptions import (
        PipelineInputError,
        PreflightError,
        ZeroUnitSortError,
    )
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
    )
    from spyglass.spikesorting.v2.sorting import (
        Sorting,
        SortingSelection,
    )
    from spyglass.utils import logger

    if pipeline_preset not in _PIPELINE_PRESETS:
        raise PipelineInputError(
            f"run_v2_pipeline: unknown pipeline_preset {pipeline_preset!r}. "
            f"Available pipeline presets: {sorted(_PIPELINE_PRESETS)}. "
            "Call spyglass.spikesorting.v2.pipeline.describe_pipeline_presets() to see "
            "what each preset does, or list_pipeline_presets() for just the names."
        )
    bundle = _PIPELINE_PRESETS[pipeline_preset]

    # Fail fast: a read-only config check before any insert/populate, so a
    # missing team / interval / sort group / param row / sorter binary
    # surfaces in ~1 s with the exact fix, not minutes into populate() with
    # an opaque FK or SpikeInterface error. Bypass with preflight=False.
    if preflight:
        report = preflight_v2_pipeline(
            nwb_file_name=nwb_file_name,
            sort_group_id=sort_group_id,
            interval_list_name=interval_list_name,
            team_name=team_name,
            pipeline_preset=pipeline_preset,
        )
        if not report.ok:
            raise PreflightError("\n".join(report.errors))

    # Per-stage observability. For each stage: derive computed-vs-reused from
    # an existence check on the output row BEFORE populate, time the
    # populate/insert with a monotonic clock, and on failure raise a stage-
    # aware PipelineStageError carrying the run summary built so far. The
    # stable run-summary keys are stable; *_status / stage_seconds / warnings
    # are
    # additive. DataJoint's ``populate()`` is idempotent (no-ops on present
    # rows), so no separate guard is needed before each call.
    run_summary: dict[str, Any] = {"pipeline_preset": pipeline_preset}
    stage_seconds: dict[str, float] = {}
    warnings_list: list[str] = []

    recording_key = RecordingSelection.insert_selection(
        {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": int(sort_group_id),
            "interval_list_name": interval_list_name,
            "preprocessing_params_name": bundle.preprocessing_params_name,
            "team_name": team_name,
        }
    )
    _, run_summary["recording_status"], stage_seconds["recording"] = _run_stage(
        "recording",
        bool(Recording & recording_key),
        lambda: Recording.populate(recording_key, reserve_jobs=False),
        run_summary,
    )
    run_summary["recording_id"] = recording_key["recording_id"]

    artifact_detection_key = ArtifactDetectionSelection.insert_selection(
        {
            "recording_id": recording_key["recording_id"],
            "artifact_detection_params_name": bundle.artifact_detection_params_name,
        }
    )
    (
        _,
        run_summary["artifact_detection_status"],
        stage_seconds["artifact_detection"],
    ) = _run_stage(
        "artifact_detection",
        bool(ArtifactDetection & artifact_detection_key),
        lambda: ArtifactDetection.populate(
            artifact_detection_key, reserve_jobs=False
        ),
        run_summary,
    )
    run_summary["artifact_detection_id"] = artifact_detection_key[
        "artifact_detection_id"
    ]

    sorting_key = SortingSelection.insert_selection(
        {
            "recording_id": recording_key["recording_id"],
            "sorter": bundle.sorter,
            "sorter_params_name": bundle.sorter_params_name,
            "artifact_detection_id": artifact_detection_key[
                "artifact_detection_id"
            ],
        }
    )
    _, run_summary["sorting_status"], stage_seconds["sorting"] = _run_stage(
        "sorting",
        bool(Sorting & sorting_key),
        lambda: Sorting.populate(sorting_key, reserve_jobs=False),
        run_summary,
    )
    run_summary["sorting_id"] = sorting_key["sorting_id"]

    # Zero units is a legitimate result on a quiet shank. Unless the
    # caller set require_units=True, proceed to build an empty (but real)
    # curation + merge row so the result is merge-keyable like any other.
    n_units = int((Sorting & sorting_key).fetch1("n_units"))
    if n_units == 0:
        recording_id = recording_key["recording_id"]
        if require_units:
            raise ZeroUnitSortError(
                "run_v2_pipeline: sort found zero units for "
                f"recording_id={recording_id} (sorting_id="
                f"{sorting_key['sorting_id']}); require_units=True. Check "
                "detect_threshold / the artifact mask, or call with "
                "require_units=False to accept the empty result."
            )
        # Fall through to the normal curation + merge insert. A
        # zero-unit sort yields an EMPTY (but real) curation + merge row
        # -- matching v1, which writes an empty Units table -- so
        # downstream consumers treat it like any other
        # SpikeSortingOutput row instead of special-casing a None
        # merge_id. The warning is both logged (console) and recorded on
        # the run summary's ``warnings`` list (programmatic access).
        zero_unit_warning = (
            "run_v2_pipeline: zero units for recording_id="
            f"{recording_id} (sorting_id={sorting_key['sorting_id']}); "
            "writing an EMPTY curation + merge row. Check "
            "detect_threshold / the artifact mask if you expected output."
        )
        logger.warning(zero_unit_warning)
        warnings_list.append(zero_unit_warning)

    # Record the now-known stable ``n_units`` and the ``warnings`` before the
    # curation stage runs, so a curation-stage failure's partial run summary
    # carries them (not just the pre-sorting keys).
    run_summary["n_units"] = n_units
    run_summary["warnings"] = warnings_list

    # Idempotent curation: ``insert_curation`` owns the root-reuse logic.
    # With ``reuse_existing=True`` it returns the canonical (lowest
    # curation_id) existing root if one is present -- deterministically,
    # and through the same source-part / guard / merge-registration path a
    # fresh insert uses -- otherwise it stages a fresh root. Routing through
    # it (rather than a raw fetch-or-insert here) avoids bypassing that
    # guard and silently reusing a root whose description/labels differ.
    # ``curation_id`` is not content-addressed, so classify reused/computed
    # from whether a root curation already exists for this sorting (the same
    # check insert_curation's root-reuse path uses).
    curation_exists = bool(
        CurationV2
        & {"sorting_id": sorting_key["sorting_id"], "parent_curation_id": -1}
    )

    def _curate_and_register():
        curation_key = CurationV2.insert_curation(
            sorting_key=sorting_key,
            labels={},
            parent_curation_id=-1,
            description=(
                description
                or f"run_v2_pipeline pipeline_preset={pipeline_preset}"
            ),
            reuse_existing=True,
        )
        # The CurationV2 part on the merge table is auto-registered inside
        # insert_curation (atomically), so the merge_id read-back is part of
        # the curation stage: a reused root whose registration is missing
        # (e.g. deleted out-of-band) then surfaces as a stage-aware
        # PipelineStageError carrying the partial run summary, not a raw
        # fetch1.
        merge_id = (SpikeSortingOutput.CurationV2 & curation_key).fetch1(
            "merge_id"
        )
        return curation_key, merge_id

    (curation_key, merge_id), curation_status, curation_seconds = _run_stage(
        "curation", curation_exists, _curate_and_register, run_summary
    )
    run_summary["curation_status"] = curation_status
    stage_seconds["curation"] = curation_seconds
    run_summary["curation_id"] = curation_key["curation_id"]
    run_summary["merge_id"] = merge_id
    run_summary["stage_seconds"] = stage_seconds
    return run_summary


def run_v2_pipeline_session(
    nwb_file_name: str,
    interval_list_name: str,
    team_name: str,
    pipeline_preset: "str | None" = None,
    sort_group_ids: "list[int] | None" = None,
    description: str = "",
    require_units: bool = False,
    preflight: bool = True,
    continue_on_error: bool = False,
) -> list[dict[str, Any]]:
    """Sort every (or selected) sort group in a session in one call.

    A thin batch wrapper over :func:`run_v2_pipeline`: it resolves the
    session's target ``SortGroupV2`` rows and runs the single-group
    orchestrator on each, returning one result entry per group. ``run_v2_pipeline``
    already parallelizes the heavy ``populate`` internally; this wrapper loops
    the groups **sequentially**.

    Unlike :func:`run_v2_pipeline`, an explicit ``pipeline_preset`` is required
    (a whole-session run infers no default). Choose one from
    ``describe_pipeline_presets()``.

    Preflight and error handling
    ----------------------------
    With ``preflight=True`` (default), :func:`preflight_v2_pipeline_session`
    runs once up front. If any group fails preflight and
    ``continue_on_error=False``, a :class:`PreflightError` is raised before any
    group is sorted; with ``continue_on_error=True``, the failed groups get a
    ``outcome="failed"`` entry and only the preflight-passing groups are run.
    Either way, the groups that *are* run pass ``preflight=False`` to
    :func:`run_v2_pipeline` (the session preflight already covered the checks;
    with ``preflight=False`` here the caller has opted out entirely).

    ``continue_on_error`` makes the batch resilient to per-group *preflight* and
    *sort* failures only. Exactly :class:`PipelineStageError`,
    :class:`PreflightError`, and :class:`ZeroUnitSortError` are caught per
    group; everything else -- :class:`PipelineInputError` from input validation,
    a bare ``ValueError`` from a missing Lookup row, ``datajoint``'s
    ``IntegrityError`` from a missing upstream when ``preflight=False``, or any
    unexpected bug -- propagates and stops the batch, since those signal a
    misconfiguration or DB-state change that should not be silently skipped.

    Parameters
    ----------
    nwb_file_name, interval_list_name, team_name, description, require_units
        As in :func:`run_v2_pipeline`; applied to every group.
    pipeline_preset
        Required pipeline-preset name (no default). See
        ``describe_pipeline_presets()``.
    sort_group_ids
        Optional explicit subset of sort groups to run. ``None`` (default)
        runs every ``SortGroupV2`` row for the session.
    preflight
        If True (default), run the whole-session preflight once before compute
        (see above). If False, skip it and run each group with
        ``preflight=False``.
    continue_on_error
        If False (default), the first per-group preflight/sort failure
        propagates (fail-fast). If True, failed groups yield an
        ``outcome="failed"`` entry and the batch continues.

    Returns
    -------
    list of dict
        One entry per target group, in ascending ``sort_group_id`` order.
        A successful entry is the single-group run summary (see
        :func:`run_v2_pipeline`) plus ``sort_group_id`` and ``outcome="ok"``.
        A failed entry is ``{"sort_group_id", "pipeline_preset",
        "outcome": "failed", "error_type", "error", "partial_run_summary"}``
        -- the ``partial_run_summary`` carries the stages completed before a
        sort failure (from :class:`PipelineStageError`) or ``None`` when the
        caught error carries none (a preflight or zero-unit failure). Wrap with
        ``describe_run(results)`` for a receipt table.

    Raises
    ------
    PipelineInputError
        From the shared target resolver: ``pipeline_preset`` is ``None`` or
        unknown, the session has no sort groups, or a requested
        ``sort_group_ids`` entry is absent. Never suppressed by
        ``continue_on_error``.
    PreflightError
        If ``preflight=True``, a group fails preflight, and
        ``continue_on_error=False`` -- raised before any group is sorted, with
        the aggregated per-group fixes.
    PipelineStageError, ZeroUnitSortError
        Propagated from a per-group :func:`run_v2_pipeline` when
        ``continue_on_error=False``.
    """
    from spyglass.spikesorting.v2.exceptions import (
        PipelineStageError,
        PreflightError,
        ZeroUnitSortError,
    )
    from spyglass.utils import logger

    targets = _resolve_session_sort_group_ids(
        nwb_file_name=nwb_file_name,
        pipeline_preset=pipeline_preset,
        sort_group_ids=sort_group_ids,
        caller="run_v2_pipeline_session",
    )

    results: list[dict[str, Any]] = []
    failed_preflight_ids: set[int] = set()

    # Up-front, read-only whole-session preflight (when requested).
    if preflight:
        session_report = preflight_v2_pipeline_session(
            nwb_file_name=nwb_file_name,
            interval_list_name=interval_list_name,
            team_name=team_name,
            pipeline_preset=pipeline_preset,
            sort_group_ids=targets,
        )
        if not session_report.ok:
            if not continue_on_error:
                raise PreflightError("\n".join(session_report.errors))
            # continue_on_error: record the failed groups, run only the rest.
            for row in session_report.group_reports:
                if row["ok"]:
                    continue
                sort_group_id = row["sort_group_id"]
                failed_preflight_ids.add(sort_group_id)
                logger.warning(
                    "run_v2_pipeline_session: sort_group_id="
                    f"{sort_group_id} failed preflight; skipping. "
                    f"{row['errors']}"
                )
                results.append(
                    {
                        "sort_group_id": sort_group_id,
                        "pipeline_preset": pipeline_preset,
                        "outcome": "failed",
                        "error_type": "PreflightError",
                        "error": "\n".join(row["errors"]),
                        "partial_run_summary": None,
                    }
                )

    # Per-group compute. Groups covered by the session preflight (or skipped via
    # preflight=False) run with preflight=False so the DB-only checks are not
    # repeated; a per-group failure is caught only for the three pipeline error
    # types, and only when continue_on_error is set.
    for sort_group_id in targets:
        if sort_group_id in failed_preflight_ids:
            continue
        try:
            summary = run_v2_pipeline(
                nwb_file_name=nwb_file_name,
                sort_group_id=sort_group_id,
                interval_list_name=interval_list_name,
                team_name=team_name,
                pipeline_preset=pipeline_preset,
                description=description,
                require_units=require_units,
                preflight=False,
            )
        except (
            PipelineStageError,
            PreflightError,
            ZeroUnitSortError,
        ) as exc:
            if not continue_on_error:
                raise
            logger.warning(
                "run_v2_pipeline_session: sort_group_id="
                f"{sort_group_id} failed: {exc!r}"
            )
            results.append(
                {
                    "sort_group_id": sort_group_id,
                    "pipeline_preset": pipeline_preset,
                    "outcome": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "partial_run_summary": getattr(
                        exc, "partial_run_summary", None
                    ),
                }
            )
        else:
            results.append(
                {**summary, "sort_group_id": sort_group_id, "outcome": "ok"}
            )

    # Stable, group-ordered result (preflight-failed entries were appended
    # first; restore ascending sort_group_id order).
    results.sort(key=lambda entry: entry["sort_group_id"])

    # End-of-batch receipt in the log. Surfaces the outcomes that are easy to
    # miss when scrolling a long run -- not just failures but zero-unit sorts
    # and warnings too. ``describe_run(results)`` is the richer table form.
    n_ok = sum(entry["outcome"] == "ok" for entry in results)
    n_failed = sum(entry["outcome"] == "failed" for entry in results)
    n_zero = 0
    n_warn = 0
    failed_details = []
    for entry in results:
        partial = (
            entry.get("partial_run_summary")
            if isinstance(entry.get("partial_run_summary"), dict)
            else {}
        )
        n_units = _run_metadata(entry, partial, "n_units")
        if n_units == 0:
            n_zero += 1
        if _run_warnings(entry, partial):
            n_warn += 1
        if entry["outcome"] == "failed":
            error_type = entry.get("error_type") or "Error"
            failed_details.append(
                f"sort_group_id={entry['sort_group_id']}: {error_type}"
            )
    failed_suffix = (
        f" ({', '.join(failed_details)})" if failed_details else ""
    )
    logger.info(
        f"run_v2_pipeline_session: {len(results)} group(s): {n_ok} ok, "
        f"{n_failed} failed{failed_suffix}, {n_zero} zero-unit, "
        f"{n_warn} with warnings. "
        "Call describe_run(results) for the per-group receipt."
    )
    return results


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

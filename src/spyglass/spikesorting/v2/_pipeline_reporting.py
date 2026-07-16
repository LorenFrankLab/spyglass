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

    One row per parameter-Lookup row across ALL eight v2 parameter tables that
    ``initialize_v2_defaults`` seeds -- the three preset-referenced tables
    (``PreprocessingParameters`` / ``ArtifactDetectionParameters`` /
    ``SorterParameters``) plus the downstream / cross-session ones
    (``AnalyzerWaveformParameters`` / ``MotionCorrectionParameters`` /
    ``QualityMetricParameters`` / ``AutoCurationRules`` / ``MatcherParameters``)
    -- each with its content fingerprint (the row name excluded;
    ``SorterParameters`` scoped per sorter), whether it is a shipped catalog
    default, which pipeline presets reference it, and -- when its content
    duplicates another row's -- the name it duplicates. Discovery metadata that
    lives on the *presets* (``probe_type`` / ``sampling_rate_hz`` /
    ``recommendation_status`` / ``used_by_pipeline_presets``) applies only to
    the three preset-referenced tables and is left blank / ``None`` for the
    others (they are resolved downstream of preset selection);
    ``adjacency_radius_um`` is read straight from the ``SorterParameters`` blob.

    Unlike the DB-free :func:`describe_pipeline_presets`, this reads the **live
    tables**, so user-added rows appear -- call ``initialize_v2_defaults()`` (or
    each table's ``insert_default()``) first to populate the shipped catalog.

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
        preproc_use.setdefault(preset.preprocessing_params_name, []).append(
            preset_name
        )
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
    shipped_sorter = {(r[0], r[1]) for r in SorterParameters._DEFAULT_CONTENTS}

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
                    # execution_params is part of SorterParameters' real
                    # identity (local vs container backend); omitting it here
                    # would falsely flag two backend variants as duplicates.
                    execution_params=_jsonable_blob(row["execution_params"]),
                    execution_params_schema_version=int(
                        row["execution_params_schema_version"]
                    ),
                ),
                "is_shipped_default": key in shipped_sorter,
                "recommendation_status": _str_axis(
                    used, "recommendation_status"
                ),
                "used_by_pipeline_presets": used,
                "summary": _sorter_summary(row["sorter"], params),
            }
        )

    # The remaining parameter Lookups are not referenced by the pipeline
    # PRESETS (they are resolved downstream of preset selection, or used only by
    # cross-session matching), so the preset-fold columns (probe_type /
    # sampling_rate_hz / adjacency_radius_um / used_by_pipeline_presets /
    # recommendation_status) stay blank. They ARE content-addressed by name and
    # user-populatable, so listing them keeps this report aligned with the full
    # ``initialize_v2_defaults`` surface (eight Lookups, not three).
    from spyglass.spikesorting.v2.metric_curation import (
        AutoCurationRules,
        QualityMetricParameters,
    )
    from spyglass.spikesorting.v2.session_group import (
        MotionCorrectionParameters,
    )
    from spyglass.spikesorting.v2.sorting import AnalyzerWaveformParameters
    from spyglass.spikesorting.v2.unit_matching import MatcherParameters

    def _shipped_names(table, name_attr):
        # Static-tuple catalogs (``_DEFAULT_CONTENTS``) and the dynamic-default
        # tables that build rows in code (``_default_rows`` ->
        # QualityMetricParameters; ``_default_payloads`` -> AutoCurationRules).
        # Both default builders are pure, so calling them here is safe. Returns
        # None only when shipped status genuinely cannot be determined.
        contents = getattr(table, "_DEFAULT_CONTENTS", None)
        if contents:
            return {r[0] for r in contents}
        rows_fn = getattr(table, "_default_rows", None)
        if callable(rows_fn):
            try:
                return {row[name_attr] for row in rows_fn()}
            except Exception:  # pragma: no cover -- defensive
                return None
        payloads_fn = getattr(table, "_default_payloads", None)
        if callable(payloads_fn):
            try:
                return {master[name_attr] for master, _ in payloads_fn()}
            except Exception:  # pragma: no cover -- defensive
                return None
        return None

    def _append_simple_param_records(
        table, table_name, name_attr, summarize, extra_content=None
    ):
        """List a name-keyed param Lookup with preset-fold columns blank.

        ``is_shipped_default`` is resolved from ``_DEFAULT_CONTENTS`` when the
        table ships it, else left ``None`` (undetermined) -- not every Lookup
        exposes a static default-row tuple. ``extra_content(row)`` folds
        part-table content (e.g. ``AutoCurationRules.Rule`` rows) into the
        fingerprint so a name-keyed master with identical scalar columns but
        different part rows is not falsely flagged as a content duplicate.
        """
        shipped = _shipped_names(table, name_attr)
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
                    "is_shipped_default": (
                        simple_row[name_attr] in shipped
                        if shipped is not None
                        else None
                    ),
                    "recommendation_status": None,
                    "used_by_pipeline_presets": [],
                    "summary": summarize(simple_row),
                }
            )

    _append_simple_param_records(
        AnalyzerWaveformParameters,
        "AnalyzerWaveformParameters",
        "waveform_params_name",
        lambda r: "",
    )
    _append_simple_param_records(
        MotionCorrectionParameters,
        "MotionCorrectionParameters",
        "motion_correction_params_name",
        lambda r: str(_jsonable_blob(r.get("params") or {}).get("preset", "")),
    )
    _append_simple_param_records(
        QualityMetricParameters,
        "QualityMetricParameters",
        "metric_params_name",
        lambda r: f"{len(_jsonable_blob(r.get('metric_names')) or [])} metrics",
    )

    def _autocuration_rule_content(rules_row):
        # The Rule part rows define the named ruleset's content, so fold them
        # (ordered by rule_index) into the fingerprint -- two rulesets with the
        # same master columns but different Rule rows are NOT duplicates.
        # Exclude the auto_curation_rules_name FK (it IS the name being
        # abstracted away) so two IDENTICAL rule sets under different names share
        # a fingerprint and surface as duplicate_of.
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

    _append_simple_param_records(
        AutoCurationRules,
        "AutoCurationRules",
        "auto_curation_rules_name",
        lambda r: f"merge preset {r.get('auto_merge_preset', '')!r}",
        extra_content=_autocuration_rule_content,
    )
    _append_simple_param_records(
        MatcherParameters,
        "MatcherParameters",
        "matcher_params_name",
        lambda r: f"matcher {r.get('matcher', '')!r}",
    )

    # Duplicate-content detection: rows sharing a fingerprint within the same
    # (table, sorter) scope are content duplicates under different names.
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
            and rec["is_shipped_default"] is False
        ):
            # Only flag when shipped status is KNOWN False; ``None`` means
            # undetermined (don't emit a false "non-catalog" warning).
            warnings.append("non-catalog row using the 'franklab' name")
        rec["name_warnings"] = "; ".join(warnings)

    frame = pd.DataFrame(
        [{c: rec.get(c) for c in _PARAMETER_ROW_COLUMNS} for rec in records],
        columns=_PARAMETER_ROW_COLUMNS,
    )
    return frame.sort_values(["table", "sorter", "parameter_name"]).reset_index(
        drop=True
    )


def _content_equal(left, right) -> bool:
    """Float-tolerant deep equality for fetched-vs-shipped default content.

    Deliberately NOT ``_metric_curation._values_match`` despite the near-identical
    shape: this audit compares a SHIPPED Python value against a DB-FETCHED value,
    where a ``bool`` column comes back as ``int`` (DataJoint stores bool as
    tinyint), so ``False`` shipped must equal ``0`` stored -- the ``left == right``
    bool branch here conflates them on purpose. ``_values_match`` does the
    opposite (strict ``type(a) is type(b)``) because its own callers compare two
    in-memory blobs where a bool-vs-int distinction is real. Float columns are
    single precision, so floats compare with a tolerance and dicts/lists recurse;
    both sides are pre-normalized via ``_jsonable_blob``.
    """
    import math

    if isinstance(left, bool) or isinstance(right, bool):
        return left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(
            float(left), float(right), rel_tol=1e-6, abs_tol=1e-9
        )
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

    ``_default_rows`` is the raw insert INPUT -- it omits the columns the
    validated insert fills (``template_metric_columns``, ``params_schema_version``,
    ``job_kwargs``). Mirror ``QualityMetricParameters.insert``'s row-building so
    every stored column joins the drift comparison, not just the raw inputs.
    """
    from spyglass.spikesorting.v2._params.metric_curation import (
        QUALITY_METRIC_SCHEMA_VERSION,
        QualityMetricParamsSchema,
    )
    from spyglass.spikesorting.v2.utils import _validate_params

    rows = []
    for row in table._default_rows():
        payload = {
            "schema_version": row.get(
                "params_schema_version", QUALITY_METRIC_SCHEMA_VERSION
            ),
            "metric_names": row["metric_names"],
            "metric_kwargs": row.get("metric_kwargs", {}),
            "skip_pc_metrics": row.get("skip_pc_metrics", True),
        }
        if "template_metric_columns" in row:
            payload["template_metric_columns"] = row["template_metric_columns"]
        clean = _validate_params(QualityMetricParamsSchema, payload)
        rows.append(
            {
                "metric_params_name": row["metric_params_name"],
                "metric_names": clean["metric_names"],
                "metric_kwargs": clean["metric_kwargs"],
                "template_metric_columns": clean["template_metric_columns"],
                "skip_pc_metrics": clean["skip_pc_metrics"],
                "params_schema_version": clean["schema_version"],
                "job_kwargs": row.get("job_kwargs"),
            }
        )
    return rows


def _canonical_autocuration_master_rows(table) -> list[dict]:
    """Shipped ``AutoCurationRules`` MASTER defaults in their STORED shape.

    ``_default_payloads`` masters omit the validated-fill columns
    (``auto_merge_kwargs``, ``params_schema_version``, ``job_kwargs``). Mirror
    ``AutoCurationRules.insert_rules``'s master-building so a drift in those
    defaulted master columns is compared too (the ``Rule`` parts are compared
    separately in ``verify_v2_default_catalog``).
    """
    from spyglass.spikesorting.v2._params.metric_curation import (
        AUTO_CURATION_RULES_SCHEMA_VERSION,
        AutoCurationRulesSchema,
    )
    from spyglass.spikesorting.v2.utils import _validate_params

    rows = []
    for master, rule_rows in table._default_payloads():
        payload = {
            "schema_version": master.get(
                "params_schema_version", AUTO_CURATION_RULES_SCHEMA_VERSION
            ),
            "auto_merge_preset": master["auto_merge_preset"],
            "auto_merge_kwargs": master.get("auto_merge_kwargs", {}),
            "rules": rule_rows,
        }
        clean = _validate_params(AutoCurationRulesSchema, payload)
        rows.append(
            {
                "auto_curation_rules_name": master["auto_curation_rules_name"],
                "auto_merge_preset": clean["auto_merge_preset"],
                "auto_merge_kwargs": clean["auto_merge_kwargs"],
                "params_schema_version": clean["schema_version"],
                "job_kwargs": master.get("job_kwargs"),
            }
        )
    return rows


def _shipped_default_rows(table) -> list[dict]:
    """Shipped default rows for a Lookup in their STORED (canonical) shape.

    ``_DEFAULT_CONTENTS`` positional tuples are zipped to the table heading (and
    are already canonical). The dynamic-default tables expose raw insert INPUT
    (``_default_rows`` / ``_default_payloads``), so their canonical stored rows
    are rebuilt via the table's own validation so EVERY stored column -- not just
    the raw inputs -- joins the drift comparison. Returns ``[]`` when a table
    exposes no static default source.
    """
    from spyglass.spikesorting.v2.utils import _jsonable_blob

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
        # Master scalars only; the ``Rule`` parts are compared separately in
        # ``verify_v2_default_catalog`` (a drifted rule threshold would otherwise
        # be invisible).
        return [dict(master) for master, _parts in payloads_fn()]
    return []


def _v2_default_catalog_tables():
    """The default-bearing v2 parameter Lookups + their name attribute.

    Every table ``initialize_v2_defaults`` seeds, paired with its params-name
    column. Imported lazily so the package stays import-light.
    """
    from spyglass.spikesorting.v2.artifact import ArtifactDetectionParameters
    from spyglass.spikesorting.v2.metric_curation import (
        AutoCurationRules,
        QualityMetricParameters,
    )
    from spyglass.spikesorting.v2.recording import PreprocessingParameters
    from spyglass.spikesorting.v2.session_group import (
        MotionCorrectionParameters,
    )
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
        (MotionCorrectionParameters, "motion_correction_params_name"),
        (QualityMetricParameters, "metric_params_name"),
        (AutoCurationRules, "auto_curation_rules_name"),
        (MatcherParameters, "matcher_params_name"),
    ]


def _diverged_autocuration_rules(table) -> list[dict]:
    """Stale ``AutoCurationRules.Rule`` part rows vs the shipped rule sets.

    The master scalar compare misses a drifted rule (threshold / operator / label
    / shape column), so compare the shipped ``_default_payloads`` rule rows to the
    stored ``Rule`` rows per name: a different rule count, a missing
    ``rule_index``, or a diverged rule field is a stale default.
    """
    from spyglass.spikesorting.v2.utils import _jsonable_blob

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

    ``initialize_v2_defaults`` re-runs each ``insert_default()`` idempotently and
    NEVER compares a stored same-name row's content to the shipped content
    (``reject_duplicate_parameter_content`` skips existing-PK rows). So a stored
    shipped-default row whose content has since diverged from the shipped content
    -- e.g. a hand-edited stored blob or a row seeded under an older default --
    silently keeps its stale content. This audit compares, for each shipped
    default present in the DB, the stored content to the shipped content (per PK,
    so the per-sorter ``SorterParameters`` keys correctly) and returns the
    divergences. For ``QualityMetricParameters`` the canonical
    ``template_metric_columns`` default is included, and for ``AutoCurationRules``
    the ``Rule`` part rows are compared too, so a drifted rule threshold or
    waveform-shape column is not invisible.

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
    from spyglass.spikesorting.v2.exceptions import (
        DuplicateParameterContentError,
    )
    from spyglass.spikesorting.v2.utils import _jsonable_blob

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
                continue  # shipped row missing a PK field -> not comparable
            stored = stored_by_pk.get(pk)
            if stored is None:
                continue  # not seeded (e.g. sorter not installed); skip
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
    (``CurationEvaluation``).

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
    "root_merge_id",
    "analysis_merge_id",
    "warning",
    "error",
]

# Canonical stage order for the run receipt; extras (if any) append after.
_RUN_STAGE_ORDER = (
    "recording",
    "artifact_detection",
    # Concat-mode source stages run BEFORE sorting (a single-session run has
    # recording/artifact_detection instead; the receipt lists only the stages a
    # run actually has), so order them here to keep the receipt chronological.
    "member_recording",
    "concat_recording",
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
    analysis_merge_id = run_summary.get("analysis_merge_id")
    # The root-only / auto-curated status is a run_v2_pipeline concept, keyed on
    # its root_merge_id. Other dict summaries that flow through describe_run
    # (e.g. a run_v2_unit_match manifest) have no root_merge_id -- leave their
    # summary status blank rather than mislabeling them "root only".
    if "root_merge_id" in run_summary:
        summary_status = "auto-curated" if analysis_merge_id else "root only"
    else:
        summary_status = None
    header = _run_blank_row()
    header.update(
        row_type="summary",
        sort_group_id=sort_group_id,
        status=summary_status,
        seconds=_run_stage_seconds_total(run_summary),
        n_units=run_summary.get("n_units"),
        root_merge_id=run_summary.get("root_merge_id"),
        analysis_merge_id=analysis_merge_id,
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

    Pass a single ``dict`` run summary -- a ``run_v2_pipeline`` summary or a
    ``run_v2_unit_match`` manifest (both carry ``stage_seconds`` + per-stage
    ``*_status`` keys) -- or a ``run_v2_pipeline_session`` result (``list`` of
    dicts). For the session form, a leading ``summary`` row carries the ok /
    failed / zero-unit / with-warnings counts and ``seconds`` is
    ``sum(stage_seconds.values())`` per group; partial / missing summaries on
    failed groups are tolerated.

    Returns
    -------
    pandas.DataFrame
        Columns ``row_type`` (``"summary"`` / ``"stage"`` / ``"group"`` /
        ``"warning"``), ``sort_group_id``, ``stage``, ``status``, ``seconds``,
        ``n_units``, ``root_merge_id``, ``analysis_merge_id``, ``warning``,
        ``error``. For a ``run_v2_pipeline`` summary the ``summary`` row's
        ``status`` is ``"root only"`` / ``"auto-curated"`` (and
        ``analysis_merge_id`` is ``None`` until a curation makes the sort
        analysis-ready); for any other dict summary (e.g. ``run_v2_unit_match``,
        which has no ``root_merge_id``) the ``summary`` status is blank and its
        stages render from its own ``*_status`` keys.
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
            root_merge_id = _run_metadata(entry, partial, "root_merge_id")
            analysis_merge_id = _run_metadata(
                entry, partial, "analysis_merge_id"
            )
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
                root_merge_id=root_merge_id,
                analysis_merge_id=analysis_merge_id,
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

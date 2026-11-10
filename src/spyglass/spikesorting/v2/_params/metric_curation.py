"""Validated parameter schemas for analyzer-driven quality-metric curation.

- ``QualityMetricParamsSchema``: which SI quality metrics to compute, their
  kwargs, and whether PCA metrics are skipped.
- ``AutoCurationRulesSchema``: an auto-merge preset plus ordered threshold
  rules that map a metric column to a unit label.

Concurrency settings live in each row's ``job_kwargs`` blob.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

QUALITY_METRIC_SCHEMA_VERSION = 2
AUTO_CURATION_RULES_SCHEMA_VERSION = 1

# Must match the ``operator`` enum on ``AutoCurationRules.Rule``.
RuleOperator = Literal["<", "<=", ">", ">=", "==", "!="]

# Spyglass policy for a non-finite rule metric; not SI's ``nan_policy``.
MissingMetricPolicy = Literal["error", "fail", "pass"]

# SI 0.104.3 ``compute_merge_unit_groups`` presets plus ``"none"`` (skip).
AutoMergePreset = Literal[
    "similarity_correlograms",
    "temporal_splits",
    "x_contaminations",
    "feature_neighbors",
    "slay",
    "none",
]

# Output columns of SI 0.104's ``nn_advanced``; invalid as metric names.
_RENAMED_NN_METRICS = {"nn_isolation", "nn_noise_overlap"}

# SI output column (not metric) names. Trough-local, so it fits the 0.5 ms
# hippocampus window; SpikeSortingV2Architecture.md explains the opt-in columns.
DEFAULT_TEMPLATE_METRIC_COLUMNS = [
    "trough_half_width",
]


def _available_quality_metric_names() -> list[str]:
    """Return the installed SpikeInterface's exported quality-metric names."""
    from spikeinterface.metrics.quality import get_quality_metric_list

    return list(get_quality_metric_list())


@lru_cache(maxsize=1)
def _available_pca_metric_names() -> tuple[str, ...]:
    """Return the installed SI's PCA-based metrics (need the whitened analyzer).

    Returns a tuple because cached results must not be mutated.
    """
    from spikeinterface.metrics.quality import get_quality_pca_metric_list

    return tuple(get_quality_pca_metric_list())


def required_extensions_for_metrics(
    metric_names, present_extensions
) -> list[str]:
    """Extensions the requested metrics need that are not already present.

    SI silently skips a metric whose extension is absent, so dependencies are
    read from SI's ``depend_on``. An either-or ``"a|b"`` dependency is met by
    any present option, else the first is requested. Callers pass only voltage
    (non-PCA) metric names. Order-preserving.
    """
    from spikeinterface.metrics.quality.quality_metrics import (
        ComputeQualityMetrics,
    )

    present = set(present_extensions)
    needed: list[str] = []
    for name in metric_names:
        metric_class = ComputeQualityMetrics.get_metric_by_name(name)
        for dep in metric_class.depend_on:
            options = dep.split("|")
            if any(o in present or o in needed for o in options):
                continue
            needed.append(options[0])
    return needed


def _available_template_metric_columns() -> list[str]:
    """Installed SI's single-channel template-metric output column names.

    Column names, not metric names (``half_width`` emits two columns).
    Single-channel only so surfaced columns stay scalar.
    """
    from spikeinterface.metrics import (
        ComputeTemplateMetrics,
        get_single_channel_template_metric_names,
    )

    return list(
        ComputeTemplateMetrics.get_metric_columns(
            get_single_channel_template_metric_names()
        )
    )


class QualityMetricParamsSchema(BaseModel):
    """Validated schema for a ``QualityMetricParameters`` row.

    - ``metric_names``: SI metric names, validated against the installed SI.
    - ``metric_kwargs``: per-metric kwargs for ``compute_quality_metrics``.
    - ``skip_pc_metrics``: must be ``False`` exactly when a PCA metric such as
      ``nn_advanced`` is requested.
    - ``template_metric_columns``: SI waveform-shape output columns surfaced
      for cell typing; not thresholded.

    Rules threshold output columns (``nn_noise_overlap``), not metric names.
    """

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[QUALITY_METRIC_SCHEMA_VERSION] = (
        QUALITY_METRIC_SCHEMA_VERSION
    )
    metric_names: list[str] = Field(min_length=1)
    metric_kwargs: dict[str, dict] = Field(default_factory=dict)
    skip_pc_metrics: bool = True
    observed_presence_bin_duration_s: float = Field(
        default=60.0, gt=0, allow_inf_nan=False
    )
    template_metric_columns: list[str] = Field(
        default_factory=lambda: list(DEFAULT_TEMPLATE_METRIC_COLUMNS)
    )

    @field_validator("metric_names")
    @classmethod
    def _check_metric_names(cls, names: list[str]) -> list[str]:
        renamed = sorted(set(names) & _RENAMED_NN_METRICS)
        if renamed:
            raise ValueError(
                f"Quality metric name(s) {renamed} were removed in "
                "SpikeInterface 0.104: nn_isolation / nn_noise_overlap are now "
                "the two output COLUMNS of the single metric 'nn_advanced'. "
                "Request 'nn_advanced' in metric_names (with "
                "skip_pc_metrics=False), then threshold the 'nn_noise_overlap' "
                "column in an AutoCurationRules rule."
            )
        available = _available_quality_metric_names()
        unknown = sorted(name for name in names if name not in available)
        if unknown:
            raise ValueError(
                f"Unknown quality metric name(s) {unknown}. Available metrics "
                f"in this SpikeInterface: {sorted(available)}."
            )
        return names

    @field_validator("template_metric_columns")
    @classmethod
    def _check_template_metric_columns(cls, cols: list[str]) -> list[str]:
        available = _available_template_metric_columns()
        unknown = sorted(c for c in cols if c not in available)
        if unknown:
            raise ValueError(
                f"Unknown template metric column(s) {unknown}. These are SI "
                "OUTPUT COLUMN names, not metric names (e.g. the 'half_width' "
                "metric emits 'trough_half_width'/'peak_half_width'). Available "
                f"single-channel template columns: {sorted(available)}."
            )
        return cols

    @model_validator(mode="after")
    def _check_pc_metrics_consistent(self) -> "QualityMetricParamsSchema":
        # Recompute / orphan gating treat skip_pc_metrics=False as proof a
        # metric analyzer exists.
        pca = set(_available_pca_metric_names())
        requested_pca = set(self.metric_names) & pca
        if not self.skip_pc_metrics and not requested_pca:
            raise ValueError(
                "skip_pc_metrics=False but metric_names requests no PCA "
                f"metric (one of {sorted(pca)}). Either request a PCA "
                "metric (e.g. nn_advanced) or set skip_pc_metrics=True -- "
                "skip_pc_metrics=False with no PCA metric builds no metric "
                "analyzer and is a contradiction."
            )
        if self.skip_pc_metrics and requested_pca:
            raise ValueError(
                "skip_pc_metrics=True but metric_names requests PCA metric(s) "
                f"{sorted(requested_pca)}, which would be skipped. Set "
                "skip_pc_metrics=False to compute them on the whitened metric "
                "analyzer, or remove the PCA metric(s) from metric_names."
            )
        return self

    @model_validator(mode="after")
    def _check_kwargs_target_requested_metrics(
        self,
    ) -> "QualityMetricParamsSchema":
        # A kwargs key for an unrequested metric is a silent no-op (a typo).
        orphan = sorted(set(self.metric_kwargs) - set(self.metric_names))
        if orphan:
            raise ValueError(
                f"metric_kwargs has entries for metric(s) {orphan} not in "
                f"metric_names {sorted(self.metric_names)}. Per-metric kwargs "
                "apply only to a requested metric -- add the metric to "
                "metric_names or remove the stray kwargs key."
            )
        return self


class AutoCurationRuleSchema(BaseModel):
    """Validated schema for a single ``AutoCurationRules.Rule`` row.

    A rule applies ``operator`` / ``threshold`` to one metric *column*
    (``metric_name``) and emits ``label`` for matching units. ``metric_name``
    is free text because a column can differ from its metric name
    (``nn_noise_overlap`` from ``nn_advanced``); a missing column raises at
    populate.

    ``missing_policy`` covers only units SI cannot assess (see
    ``expected_missing_units`` in ``_curation.metrics``):

    - ``error``: raise.
    - ``fail``: apply the label.
    - ``pass``: leave the unit unlabelled; right for metrics with a validity
      floor such as ``nn_advanced``'s ``min_spikes``.

    Any other NaN or SI error aborts regardless of policy. ``pass`` and
    ``fail`` warn when the metric is missing for every unit.
    """

    model_config = ConfigDict(extra="forbid")
    rule_index: int = Field(ge=0)
    rule_name: str = Field(min_length=1, max_length=64)
    metric_name: str = Field(min_length=1, max_length=64)
    operator: RuleOperator
    threshold: float = Field(allow_inf_nan=False)
    label: str = Field(min_length=1, max_length=32)
    missing_policy: MissingMetricPolicy = "error"


class AutoCurationRulesSchema(BaseModel):
    """Validated schema for an ``AutoCurationRules`` master + its rule rows.

    Master and rules validate together so a master never lands with unchecked
    rules. ``rule_index`` values must be unique.
    """

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[AUTO_CURATION_RULES_SCHEMA_VERSION] = (
        AUTO_CURATION_RULES_SCHEMA_VERSION
    )
    auto_merge_preset: AutoMergePreset = "none"
    auto_merge_kwargs: dict = Field(default_factory=dict)
    rules: list[AutoCurationRuleSchema] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_rule_indices_unique(self) -> "AutoCurationRulesSchema":
        indices = [rule.rule_index for rule in self.rules]
        if len(indices) != len(set(indices)):
            duplicates = sorted(
                {idx for idx in indices if indices.count(idx) > 1}
            )
            raise ValueError(
                "AutoCurationRules rule_index values must be unique; "
                f"duplicates: {duplicates}. rule_index sets the order rules "
                "are applied, so two rules cannot share one."
            )
        return self


def prepare_quality_metric_row(row: dict) -> dict:
    """Validate one ``QualityMetricParameters`` insert input into its stored row.

    Shared by ``QualityMetricParameters.insert`` and the default-catalog audit.
    Unset optional fields take the schema default.

    Parameters
    ----------
    row : dict
        Insert input: ``metric_params_name``, ``metric_names``, optional
        ``metric_kwargs`` / ``skip_pc_metrics`` / ``template_metric_columns``
        / ``params_schema_version`` / ``job_kwargs``.

    Returns
    -------
    dict
        The row in stored shape (every table column present).
    """
    payload = {
        "schema_version": row.get(
            "params_schema_version", QUALITY_METRIC_SCHEMA_VERSION
        ),
        "metric_names": row["metric_names"],
    }
    for field in (
        "metric_kwargs",
        "skip_pc_metrics",
        "observed_presence_bin_duration_s",
        "template_metric_columns",
    ):
        if field in row:
            payload[field] = row[field]
    clean = QualityMetricParamsSchema.model_validate(payload).model_dump()
    return {
        "metric_params_name": row["metric_params_name"],
        "metric_names": clean["metric_names"],
        "metric_kwargs": clean["metric_kwargs"],
        "template_metric_columns": clean["template_metric_columns"],
        "skip_pc_metrics": clean["skip_pc_metrics"],
        "observed_presence_bin_duration_s": clean[
            "observed_presence_bin_duration_s"
        ],
        "params_schema_version": clean["schema_version"],
        "job_kwargs": row.get("job_kwargs"),
    }


def prepare_auto_curation_rules(
    row: dict, rule_rows: list[dict]
) -> tuple[dict, list[dict]]:
    """Validate an ``AutoCurationRules`` master + rule inputs into stored rows.

    Shared by ``AutoCurationRules.insert_rules`` and the default-catalog audit.

    Parameters
    ----------
    row : dict
        Master insert input: ``auto_curation_rules_name``,
        ``auto_merge_preset``, optional ``auto_merge_kwargs`` /
        ``params_schema_version`` / ``job_kwargs``.
    rule_rows : list[dict]
        Ordered ``Rule`` inputs (``rule_index`` values unique).

    Returns
    -------
    tuple[dict, list[dict]]
        ``(master, rule_inserts)`` in stored shape; each rule insert carries
        the master's ``auto_curation_rules_name`` FK, in validated order.
    """
    name = row["auto_curation_rules_name"]
    payload = {
        "schema_version": row.get(
            "params_schema_version", AUTO_CURATION_RULES_SCHEMA_VERSION
        ),
        "auto_merge_preset": row["auto_merge_preset"],
        "auto_merge_kwargs": row.get("auto_merge_kwargs", {}),
        "rules": rule_rows,
    }
    clean = AutoCurationRulesSchema.model_validate(payload).model_dump()
    master = {
        "auto_curation_rules_name": name,
        "auto_merge_preset": clean["auto_merge_preset"],
        "auto_merge_kwargs": clean["auto_merge_kwargs"],
        "params_schema_version": clean["schema_version"],
        "job_kwargs": row.get("job_kwargs"),
    }
    rule_inserts = [
        {"auto_curation_rules_name": name, **rule} for rule in clean["rules"]
    ]
    return master, rule_inserts

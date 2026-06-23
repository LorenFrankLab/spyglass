"""Validated parameter schemas for analyzer-driven quality-metric curation.

Two structured parameter tables back analyzer curation:

``QualityMetricParameters``
    which SpikeInterface quality metrics to compute, the per-metric kwargs
    passed to ``compute_quality_metrics(..., metric_params=...)``, and whether
    PCA-based metrics are skipped. ``QualityMetricParamsSchema`` validates that
    payload, including checking every requested metric name against the
    installed SpikeInterface's exported metric list.

``AutoCurationRules`` (+ its ``Rule`` part table)
    an auto-merge preset plus ordered threshold rules that turn a metric
    column into a unit label. ``AutoCurationRulesSchema`` validates the whole
    master-plus-rules payload before either table is written.

Concurrency parameters (``n_jobs``, ``chunk_duration``, ``progress_bar``) do
NOT live on these schemas. They are stored on the per-row ``job_kwargs`` blob
column and resolved at populate time by ``_resolved_job_kwargs`` per the
shared Job-Kwargs Resolution convention.
"""

from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

QUALITY_METRIC_SCHEMA_VERSION = 1
AUTO_CURATION_RULES_SCHEMA_VERSION = 1

# Comparison operators a threshold rule may use. Mirrors the ``operator`` enum
# on ``AutoCurationRules.Rule`` so the Pydantic guard and the table column
# agree on the allowed set.
RuleOperator = Literal["<", "<=", ">", ">=", "==", "!="]

# Auto-merge presets. The first five are SpikeInterface 0.104.3's
# ``compute_merge_unit_groups`` presets (verified against the installed
# source -- see tests/spikesorting/v2/resolver/si0104-quality-metrics.md);
# ``"none"`` is the Spyglass-level "skip auto-merge" sentinel, not an SI
# preset.
AutoMergePreset = Literal[
    "similarity_correlograms",
    "temporal_splits",
    "x_contaminations",
    "feature_neighbors",
    "slay",
    "none",
]

# The two output columns of the 0.104 ``nn_advanced`` PCA metric. Requesting
# either as a metric *name* raises in SI 0.104; we surface a targeted hint.
_RENAMED_NN_METRICS = {"nn_isolation", "nn_noise_overlap"}


def _available_quality_metric_names() -> list[str]:
    """Return the installed SpikeInterface's exported quality-metric names.

    Imported lazily so this module stays importable without paying the
    SpikeInterface import cost until a metric name is actually validated.
    """
    try:
        from spikeinterface.metrics.quality import get_quality_metric_list
    except ImportError:  # pragma: no cover - pinned 0.104 always has this
        from spikeinterface.qualitymetrics import get_quality_metric_list

    return list(get_quality_metric_list())


def _available_pca_metric_names() -> list[str]:
    """Return the installed SI's PCA-based (``principal_components``) metrics.

    These (``d_prime``, ``mahalanobis``, ``nearest_neighbor``, ``nn_advanced``,
    ``silhouette`` in SI 0.104) are the only metrics that need the whitened
    metric analyzer; ``skip_pc_metrics=False`` is meaningful only if at least
    one is requested. Read from SI, not hardcoded, like the full list above.
    """
    try:
        from spikeinterface.metrics.quality import (
            get_quality_pca_metric_list,
        )
    except ImportError:  # pragma: no cover - pinned 0.104 always has this
        from spikeinterface.qualitymetrics import get_quality_pca_metric_list

    return list(get_quality_pca_metric_list())


class QualityMetricParamsSchema(BaseModel):
    """Validated schema for a ``QualityMetricParameters`` row.

    ``metric_names`` lists the SpikeInterface metric *names* to compute (e.g.
    ``snr``, ``isi_violation``, ``firing_rate``, ``nn_advanced``). Each name is
    validated against the installed SI's ``get_quality_metric_list()`` so a
    typo or a 0.99-era name fails at insert, not at populate. ``metric_kwargs``
    is the per-metric kwargs dict passed straight through to
    ``compute_quality_metrics(..., metric_params=...)``. ``skip_pc_metrics``
    defaults to ``True`` (PCA metrics off) -- a row feeding an ``nn_advanced``
    rule must set it ``False`` and request ``nn_advanced`` explicitly.

    Note ``nn_advanced`` produces *columns* named ``nn_isolation`` and
    ``nn_noise_overlap``; the rule that reads those columns lives in
    ``AutoCurationRules.Rule`` and references the column name, not the metric
    name (see ``AutoCurationRulesSchema``).
    """

    model_config = ConfigDict(extra="forbid")
    schema_version: int = QUALITY_METRIC_SCHEMA_VERSION
    metric_names: list[str] = Field(min_length=1)
    metric_kwargs: dict[str, dict] = Field(default_factory=dict)
    skip_pc_metrics: bool = True

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

    @model_validator(mode="after")
    def _check_pc_metrics_consistent(self) -> "QualityMetricParamsSchema":
        # Keep the skip_pc_metrics flag consistent with the requested metrics so
        # the row neither builds a whitened analyzer for nothing nor computes
        # nothing at all -- and so skip_pc_metrics=False stays an EXACT signal
        # that a metric analyzer exists, which recompute / orphan gating rely on.
        pca = set(_available_pca_metric_names())
        requested_pca = set(self.metric_names) & pca
        if not self.skip_pc_metrics and not requested_pca:
            # Build the whitened analyzer but compute no PC metric on it.
            raise ValueError(
                "skip_pc_metrics=False but metric_names requests no PCA "
                f"metric (one of {sorted(pca)}). Either request a PCA "
                "metric (e.g. nn_advanced) or set skip_pc_metrics=True -- "
                "skip_pc_metrics=False with no PCA metric builds no metric "
                "analyzer and is a contradiction."
            )
        if self.skip_pc_metrics and requested_pca == set(self.metric_names):
            # Every requested metric is a PCA metric, but PC computation is
            # skipped -> nothing would be computed (fails at populate). Catch it
            # at insert instead. (skip_pc_metrics=True WITH a non-PCA metric is
            # fine -- the PCA names are simply skipped.)
            raise ValueError(
                "skip_pc_metrics=True but every requested metric is a PCA "
                f"metric ({sorted(requested_pca)}); nothing would be computed. "
                "Set skip_pc_metrics=False to compute them, or add a non-PCA "
                "metric (e.g. snr)."
            )
        return self

    @model_validator(mode="after")
    def _check_kwargs_target_requested_metrics(
        self,
    ) -> "QualityMetricParamsSchema":
        # Per-metric kwargs only apply to a metric that is actually computed;
        # a key absent from metric_names is a silent no-op (usually a typo),
        # so reject it at insert rather than discarding it at populate time.
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

    A rule reads one metric *column* (``metric_name``) from the computed
    quality-metrics table, applies ``operator`` against ``threshold``, and
    emits ``label`` for every unit that matches. ``metric_name`` is a free
    string (not validated against the metric-name list) because a rule may
    target a column the metric name does not equal -- e.g. the
    ``nn_noise_overlap`` column produced by the ``nn_advanced`` metric. A rule
    that references a column absent from the computed metrics raises a clear
    error at populate time, not here.
    """

    model_config = ConfigDict(extra="forbid")
    rule_index: int = Field(ge=0)
    rule_name: str = Field(min_length=1, max_length=64)
    metric_name: str = Field(min_length=1, max_length=64)
    operator: RuleOperator
    threshold: float
    label: str = Field(min_length=1, max_length=32)


class AutoCurationRulesSchema(BaseModel):
    """Validated schema for an ``AutoCurationRules`` master + its rule rows.

    The full ``{auto_merge_preset, auto_merge_kwargs, rules}`` payload is
    validated together so a master row never lands without its rule rows being
    checked. ``auto_merge_preset`` is one of SI 0.104's
    ``compute_merge_unit_groups`` presets or the ``"none"`` sentinel;
    ``rules`` is an ordered list whose ``rule_index`` values must be unique.
    """

    model_config = ConfigDict(extra="forbid")
    schema_version: int = AUTO_CURATION_RULES_SCHEMA_VERSION
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

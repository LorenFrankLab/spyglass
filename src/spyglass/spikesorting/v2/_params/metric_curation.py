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

from functools import lru_cache
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

# Default surfaced waveform-shape column -- a single conservative scalar shape
# column used downstream for cell typing (firing rate is already a quality
# metric; this adds spike width). It is a SpikeInterface OUTPUT COLUMN name, NOT
# a metric name: ``trough_half_width`` is one of the two columns SI's
# ``half_width`` metric emits, so selecting columns directly avoids the
# name->column ambiguity (``half_width`` -> ``trough_half_width`` +
# ``peak_half_width``).
#
# ``trough_half_width`` is trough-local: its half-amplitude crossings sit a few
# samples either side of the trough, so it stays inside even the deliberately
# narrow hippocampus display window (``ms_before=ms_after=0.5``) and still
# separates fast-spiking interneurons (narrow) from pyramidal cells (wide).
#
# ``peak_to_trough_duration`` and the slope columns are discoverable but
# deliberately OPT-IN, not shipped defaults: each depends on the post-trough
# repolarization peak (SI's ``recovery_slope`` uses a 0.7 ms post-peak window;
# ``peak_to_trough_duration`` measures to ``peak_after``), which on the 0.5 ms
# hippocampus post-window saturates at the waveform's edge-exclusion boundary --
# the metric stops discriminating cells. The literature trough-to-peak is
# ~0.5-0.8 ms for pyramidal cells (the discriminating end), past the 0.5 ms
# window, so this is not merely a fixture artifact. They become reliable on the
# wider cortex/fallback window (1.0/2.0); a user who wants them adds them
# explicitly to ``template_metric_columns``.
DEFAULT_TEMPLATE_METRIC_COLUMNS = [
    "trough_half_width",
]


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


@lru_cache(maxsize=1)
def _available_pca_metric_names() -> tuple[str, ...]:
    """Return the installed SI's PCA-based (``principal_components``) metrics.

    These (``d_prime``, ``mahalanobis``, ``nearest_neighbor``, ``nn_advanced``,
    ``silhouette`` in SI 0.104) are the only metrics that need the whitened
    metric analyzer; ``skip_pc_metrics=False`` is meaningful only if at least
    one is requested. Read from SI, not hardcoded, like the full list above.
    The PCA set is invariant for the pinned SI version, so it is cached (the
    insert-time validator and the runtime routing both call it); a tuple is
    returned because ``lru_cache`` results must not be mutated -- callers wrap
    it in a ``set``.
    """
    try:
        from spikeinterface.metrics.quality import (
            get_quality_pca_metric_list,
        )
    except ImportError:  # pragma: no cover - pinned 0.104 always has this
        from spikeinterface.qualitymetrics import get_quality_pca_metric_list

    return tuple(get_quality_pca_metric_list())


def _available_template_metric_columns() -> list[str]:
    """Installed SI's single-channel template-metric output COLUMNS (lazy).

    These are output *column* names (``trough_half_width``,
    ``peak_to_trough_duration``, ...), the same vocabulary ``get_metrics``
    returns -- NOT metric names. A metric name is not its column: SI's
    ``half_width`` metric emits TWO columns, ``trough_half_width`` and
    ``peak_half_width``, so ``template_metric_columns`` is validated against
    this set (selecting by column is what-you-configure-is-what-you-get).
    Restricting to the single-channel set also keeps the surfaced columns
    scalar (the multi-channel columns -- ``velocity_above`` etc. -- are
    excluded). Imported lazily so the module stays importable without paying
    the SpikeInterface import cost until a column is validated.
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
    # SI template (waveform-shape) output COLUMNS surfaced in the metric table
    # for downstream cell typing. Exposed, not thresholded (the pipeline ships
    # no cell-type cutoffs). An empty list surfaces no shape columns.
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
        # Validate against the installed SI's single-channel output COLUMNS so a
        # typo -- or a metric *name* (e.g. ``half_width``) passed where a column
        # is expected -- fails at insert, not at populate. An empty list is
        # valid (surfaces no shape columns).
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
        # Keep the skip_pc_metrics flag consistent with the requested metrics:
        # metric_names should name metrics this row actually computes. Also keep
        # skip_pc_metrics=False as an EXACT signal that a metric analyzer exists,
        # which recompute / orphan gating rely on.
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
        if self.skip_pc_metrics and requested_pca:
            # PCA metrics require the metric analyzer. Silently dropping them
            # would make the DB row overstate what was actually computed.
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

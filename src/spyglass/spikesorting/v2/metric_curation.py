"""Analyzer-driven quality-metric curation for spike sorting v2.

Replaces v1's ``MetricCuration`` + ``BurstPair`` with a single
``CurationEvaluation`` computed table that walks a committed ``CurationV2``
row's ``SortingAnalyzer`` extensions to compute quality metrics, suggest
merges, and propose auto-curation labels in the curation's OWN unit namespace
(a merged unit is scored over its merged spike train, never inherited from the
highest-amplitude contributor). The proposed labels/merges are written to NWB;
turning them into a committed child ``CurationV2`` row is an explicit user
action (``CurationEvaluation.accept_evaluation_outputs`` /
``use_evaluation_labels``).

Tables
------
``QualityMetricParameters``
    Which SpikeInterface metrics to compute and their per-metric kwargs.
``AutoCurationRules`` (+ ``Rule`` part)
    An auto-merge preset plus ordered threshold rules that label units.
``CurationEvaluationSelection``
    Pairs a committed ``CurationV2`` row with a metric-params and an
    auto-rules row.
``CurationEvaluation``
    Computes metrics / merge suggestions / proposed labels and stores them in
    three NWB scratch tables; acceptance helpers commit them to a child
    ``CurationV2``.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from typing import NamedTuple

import datajoint as dj

from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.spikesorting.v2._metric_curation import (
    apply_label_rules,
    apply_snr_peak_sign,
    isi_violation_fraction,
    rules_payloads_match,
)
from spyglass.spikesorting.v2._metric_curation_nwb import (
    read_merge_suggestions,
    read_proposed_labels,
    read_quality_metrics,
    write_analyzer_curation_tables,
)
from spyglass.spikesorting.v2._params.metric_curation import (
    AUTO_CURATION_RULES_SCHEMA_VERSION,
    QUALITY_METRIC_SCHEMA_VERSION,
    AutoCurationRulesSchema,
    QualityMetricParamsSchema,
    _available_pca_metric_names,
    required_extensions_for_metrics,
)
from spyglass.spikesorting.v2.curation import CurationV2
from spyglass.spikesorting.v2.exceptions import (
    UnsupportedDirectInsertError,
    ZeroUnitAnalyzerError,
)
from spyglass.spikesorting.v2._recipe_catalog import (
    waveform_params_for_preprocessing,
)
from spyglass.spikesorting.v2.recording import RecordingSelection
from spyglass.spikesorting.v2.sorting import (
    AnalyzerWaveformParameters,
    SorterParameters,
    Sorting,
    SortingSelection,
)
from spyglass.spikesorting.v2.utils import (
    ImmutableParamsLookup,
    SelectionMasterInsertGuard,
    _jsonable_blob,
    _resolved_job_kwargs,
    _validate_params,
    reject_duplicate_quality_metric_content,
)
from spyglass.utils import SpyglassMixin, SpyglassMixinPart, logger

schema = dj.schema("spikesorting_v2_metric_curation")

# Extensions CurationEvaluation adds to the sort-time analyzer before metrics and
# auto-merge. The sort-time base set (random_spikes, noise_levels, templates,
# waveforms) is already present; these derive from it. ``principal_components``
# is added separately, only when PCA metrics are requested.
_CURATION_EXTENSIONS = (
    "spike_amplitudes",
    "correlograms",
    "template_similarity",
    "unit_locations",
    "template_metrics",
)

# Pinned principal_components params for the whitened METRIC analyzer. The
# recording is already SPATIALLY whitened (sip.whiten decorrelates channels);
# SI's PCA then computes components on those decorrelated waveforms and, with
# whiten=True, normalizes the component variances -- the standard input space
# for the PC/NN cluster-separation metrics. These are SI 0.104's defaults
# (including dtype), pinned explicitly so a future SI default change cannot
# silently alter the whitened-space PCA (and therefore the PC/NN metric values).
_PCA_EXTENSION_PARAMS = {
    "n_components": 5,
    "mode": "by_channel_local",
    "whiten": True,
    "dtype": "float32",
}


def _pca_params_match(existing: dict) -> bool:
    """True if a stored ``principal_components`` matches the pinned params.

    Compared key-by-key over ``_PCA_EXTENSION_PARAMS`` only (SI may store extra
    keys); ``dtype`` is normalized through ``np.dtype`` so the stored
    ``dtype('float32')`` matches the pinned ``"float32"`` string.
    """
    import numpy as np

    for key, pinned in _PCA_EXTENSION_PARAMS.items():
        value = existing.get(key)
        if key == "dtype":
            if value is None or np.dtype(value) != np.dtype(pinned):
                return False
        elif value != pinned:
            return False
    return True


_AUTO_MERGE_EXTRA_EXTENSIONS = {
    # SI's ``feature_neighbors`` preset includes the ``knn`` step, whose
    # required extensions are templates, spike_locations, and spike_amplitudes.
    # The sort-time/base curation extension set already covers templates and
    # spike_amplitudes; add spike_locations only for that preset.
    "feature_neighbors": ("spike_locations",),
}


def _nwb_file_name_for_sorting(sorting_key: dict) -> str:
    """Return the analyzer-curation NWB parent ``nwb_file_name`` for a sort.

    Thin alias for ``Sorting.resolve_anchor_nwb_file_name`` (the single owner of
    the source-agnostic anchor dispatch: a single-recording sort reads its
    ``RecordingSelection``; a concat sort anchors to the first
    ``SessionGroup.Member``).
    """
    return Sorting.resolve_anchor_nwb_file_name(sorting_key)


def _requested_pc_metrics(metric_names) -> list[str]:
    """PC/NN metrics among ``metric_names`` (route to the whitened analyzer).

    PCA-based metrics (``d_prime``, ``mahalanobis``, ``nearest_neighbor``,
    ``nn_advanced``, ``silhouette`` in SI 0.104) need the ``principal_components``
    extension and measure cluster separation, so they compute in the
    decorrelated (whitened) space -- the whitened METRIC analyzer; everything
    else stays on the unwhitened DISPLAY analyzer. The PCA set is the same one
    the insert-time validator uses (``_available_pca_metric_names``), so routing
    and validation cannot disagree about which metrics are PCA-based.
    """
    pca = set(_available_pca_metric_names())
    return [name for name in metric_names if name in pca]


def _assert_is_metric_recipe(waveform_params_name: str) -> None:
    """Raise unless ``waveform_params_name`` is a whitened metric recipe.

    The metric analyzer carries the PC/NN cluster-separation metrics, which must
    compute in the WHITENED space. Asserted both at ``insert_selection`` (catch
    a bad explicit override early) and again at the ``make_fetch`` consume
    boundary (a row inserted via ``allow_direct_insert`` bypasses the selection
    guard; re-validating here keeps a display/unwhitened recipe from silently
    building an unwhitened metric analyzer, mirroring the consume-time re-check
    in ``run_clusterless_thresholder``).
    """
    from spyglass.spikesorting.v2._sorting_analyzer import (
        fetch_waveform_params,
    )

    recipe = fetch_waveform_params(waveform_params_name)
    if not recipe.get("whiten") or recipe.get("purpose") != "metric":
        raise ValueError(
            f"metric_waveform_params_name={waveform_params_name!r} is not a "
            "whitened metric recipe (purpose='metric', whiten=True). PC/NN "
            "cluster-separation metrics must compute on a whitened analyzer; "
            "pass a metric recipe (e.g. franklab_cortex_metric_waveforms) or "
            "omit it to use the sort's resolved region metric row."
        )


def _assert_curation_in_raw_namespace(
    sorting_id, curation_id, *, context: str
) -> None:
    """Raise unless the curation's unit set equals the raw sort's unit set.

    The raw-sort DISPLAY analyzer's unit namespace is the raw sort. A merged
    curation -- or a label-only child of a merged parent -- carries merged unit
    ids absent from the raw sort, so reading its waveforms / correlograms /
    peak amplitudes off the raw analyzer would silently mix namespaces. The
    analyzer-backed notebook/plot helpers therefore reject such a curation and
    point the caller at the routed ``get_metrics`` / ``get_suggested_merge_groups``
    accessors (which carry the curation's own unit namespace) or at plotting the
    raw sort directly.
    """
    if not CurationV2.matches_raw_namespace(
        {"sorting_id": sorting_id, "curation_id": curation_id}
    ):
        raise ValueError(
            f"{context}: curation (sorting_id={sorting_id}, "
            f"curation_id={curation_id}) has a unit namespace that differs from "
            "the raw sort (it is a merged curation or a label-only child of "
            "one), so the raw-sort display analyzer cannot render it in the "
            "curation's namespace. Use the routed get_metrics() / "
            "get_suggested_merge_groups() accessors (curation namespace), or plot the raw "
            "sort directly via the sorting_id."
        )


class CurationEvaluationFetched(NamedTuple):
    """DB inputs for ``CurationEvaluation.make_compute`` (no DB INPUT resolution
    in compute; it still stages its OUTPUT via ``AnalysisNwbfile``).

    Everything ``make_compute`` needs to reconstruct the recording + curated /
    raw sorting and load-or-build the analyzers is resolved here (the only stage
    allowed to resolve DB inputs). ``use_fast_path`` records the committed-state routing
    decision (root/label-only -> cached raw-sort analyzer; merged -> temp
    curation analyzer) so the worker does not re-query the curation.
    """

    sorting_id: str
    curation_id: int
    nwb_file_name: str
    source_kind: str
    recording_id: str | None
    artifact_detection_id: str | None
    artifact_valid_times: object  # np.ndarray | None (DeepHashed, not ==)
    recording_row: dict
    fs: float
    raw_units_abs_path: str
    raw_n_units: int
    curated_units_abs_path: str
    expected_unit_ids: list[int]
    use_fast_path: bool
    display_waveform_params: dict
    display_analyzer_folder: str
    metric_waveform_params: dict
    metric_analyzer_folder: str
    metric_names: list[str]
    metric_kwargs: dict[str, dict]
    template_metric_columns: list[str]
    skip_pc_metrics: bool
    auto_merge_preset: str
    auto_merge_kwargs: dict
    rule_rows: list[dict]
    sorter_row: dict
    analyzer_job_kwargs: dict
    metric_job_kwargs: dict
    # Recipe / param names re-emitted into the artifact NWB (provenance, not
    # used in compute -- the resolved params/recipes above drive the work).
    metric_params_name: str
    auto_curation_rules_name: str
    display_waveform_params_name: str
    metric_waveform_params_name: str


class CurationEvaluationComputed(NamedTuple):
    """Compute -> insert carrier (DeepHash-stable strings only)."""

    analysis_file_name: str
    metrics_object_id: str
    merge_suggestions_object_id: str
    proposed_labels_object_id: str
    nwb_file_name: str
    # Producer provenance (secondary, never identity): SI version at eval time
    # and a {role: content_hash} manifest of the canonical analyzers consumed on
    # the fast path (``None`` for the merged-curation temp-analyzer path).
    spikeinterface_version: str
    source_analyzer_hashes: dict | None


@schema
class QualityMetricParameters(ImmutableParamsLookup, SpyglassMixin, dj.Lookup):
    """Which quality metrics to compute and their per-metric kwargs.

    ``metric_names`` is validated against the installed SpikeInterface's
    exported metric list; ``metric_kwargs`` is passed straight through as
    ``compute_quality_metrics(..., metric_params=metric_kwargs)``.
    ``skip_pc_metrics`` defaults True (PCA metrics off); a row feeding an
    ``nn_advanced`` rule must set it False.
    """

    definition = """
    metric_params_name: varchar(64)
    ---
    metric_names: blob              # list[str] of SpikeInterface metric names
    metric_kwargs: blob             # dict[str, dict] per-metric kwargs
    template_metric_columns: blob   # list[str] of SI template output columns
    skip_pc_metrics=1: bool
    params_schema_version=1: int
    job_kwargs=null: blob
    """

    @classmethod
    def _default_rows(cls) -> list[dict]:
        # nn_advanced is a PCA metric -> these rows set skip_pc_metrics=False so
        # the nn_noise_overlap column exists for the default auto-curation rule.
        nn_kwargs = {
            "n_components": 7,
            "n_neighbors": 5,
            "max_spikes": 20000,
            "min_spikes": 10,
            "seed": 0,
        }
        isi_kwargs = {"isi_threshold_ms": 2.0, "min_isi_ms": 0.0}
        full_metrics = [
            "snr",
            "isi_violation",
            "firing_rate",
            "num_spikes",
            "presence_ratio",
            "amplitude_cutoff",
            "nn_advanced",
        ]
        # franklab and neuropixels share the same full metric set today; build
        # both from one payload so they cannot silently drift (a probe-specific
        # divergence would be expressed as an explicit override here).
        full_metric_kwargs = {
            "snr": {"peak_sign": "neg"},
            "isi_violation": isi_kwargs,
            "nn_advanced": nn_kwargs,
        }
        rows = [
            {
                "metric_params_name": name,
                "metric_names": full_metrics,
                "metric_kwargs": full_metric_kwargs,
                "skip_pc_metrics": False,
            }
            for name in ("franklab_default", "neuropixels_default")
        ]
        rows.append(
            {
                "metric_params_name": "minimal",
                "metric_names": ["snr", "isi_violation", "firing_rate"],
                "metric_kwargs": {
                    "snr": {"peak_sign": "neg"},
                    "isi_violation": isi_kwargs,
                },
                "skip_pc_metrics": True,
            }
        )
        return rows

    def insert1(self, row, **kwargs):
        """Validate one row's params then insert it."""
        self.insert([row], **kwargs)

    def insert(self, rows, *, allow_duplicate_params=False, **kwargs):
        """Validate every row's params, reject duplicate content, then insert.

        Like the sibling parameter Lookups, a second NAME for content identical
        to an existing row raises ``DuplicateParameterContentError`` so
        provenance never silently forks. ``allow_duplicate_params=True`` opts
        out (the shipped franklab/neuropixels defaults use it -- see
        ``insert_default``).
        """
        if isinstance(rows, dict):
            rows = [rows]
        validated = []
        for row in rows:
            payload = {
                "schema_version": row.get(
                    "params_schema_version", QUALITY_METRIC_SCHEMA_VERSION
                ),
                "metric_names": row["metric_names"],
                "metric_kwargs": row.get("metric_kwargs", {}),
                "skip_pc_metrics": row.get("skip_pc_metrics", True),
            }
            # Only forward template_metric_columns when the row sets it, so an
            # omitting row picks up the schema default (the conservative,
            # window-safe trough_half_width shape column).
            if "template_metric_columns" in row:
                payload["template_metric_columns"] = row[
                    "template_metric_columns"
                ]
            clean = _validate_params(QualityMetricParamsSchema, payload)
            validated.append(
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
        reject_duplicate_quality_metric_content(
            self.fetch(as_dict=True),
            validated,
            allow_duplicate_params=allow_duplicate_params,
        )
        super().insert(validated, **kwargs)

    @classmethod
    def insert_default(cls):
        """Insert the default quality-metric parameter rows (idempotent)."""
        # franklab_default and neuropixels_default deliberately ship identical
        # content under two names (kept separate so a probe-specific divergence
        # can be expressed later), so the shipped defaults opt out of the
        # duplicate-content guard.
        cls().insert(
            cls._default_rows(),
            skip_duplicates=True,
            allow_duplicate_params=True,
        )

    @classmethod
    def available_quality_metrics(cls) -> list[str]:
        """Return the metric names the installed SpikeInterface exposes."""
        from spyglass.spikesorting.v2._params.metric_curation import (
            _available_quality_metric_names,
        )

        return sorted(_available_quality_metric_names())

    @classmethod
    def show_available_metrics(cls) -> list[str]:
        """Return and log the available SpikeInterface quality-metric names.

        Returning the list keeps the v1 notebook-discovery helper visible in
        Jupyter (a logger-only helper can appear to do nothing depending on the
        notebook's logging configuration).
        """
        names = cls.available_quality_metrics()
        for name in names:
            logger.info(name)
        return names

    @classmethod
    def available_template_metric_columns(cls) -> list[str]:
        """Return the SI template (waveform-shape) output COLUMN names.

        These are the valid values for a row's ``template_metric_columns`` and
        the same vocabulary ``get_metrics`` surfaces -- output *columns*, not
        metric names. SI's ``half_width`` metric, for example, is surfaced as
        the ``trough_half_width`` and ``peak_half_width`` columns (the list
        below includes ``trough_half_width`` but not ``half_width``).
        """
        from spyglass.spikesorting.v2._params.metric_curation import (
            _available_template_metric_columns,
        )

        return sorted(_available_template_metric_columns())


@schema
class AutoCurationRules(ImmutableParamsLookup, SpyglassMixin, dj.Lookup):
    """Auto-merge preset + ordered threshold rules that label units.

    Label rules are part rows (not a blob) so rule order is explicit and a
    metric/label is queryable. Insert through ``insert_rules(row, rule_rows)``
    so the master row and its rule rows are validated together; direct
    ``insert1`` is unsupported.
    """

    definition = """
    auto_curation_rules_name: varchar(64)
    ---
    auto_merge_preset: varchar(32)   # an SI compute_merge_unit_groups preset, or 'none'
    auto_merge_kwargs: blob
    params_schema_version=1: int
    job_kwargs=null: blob
    """

    class Rule(ImmutableParamsLookup, SpyglassMixinPart):
        definition = """
        -> master
        rule_index: int
        ---
        rule_name: varchar(64)
        metric_name: varchar(64)
        operator: enum('<', '<=', '>', '>=', '==', '!=')
        threshold: float
        label: varchar(32)
        """

        def insert(self, rows, **kwargs):
            raise UnsupportedDirectInsertError(
                "AutoCurationRules.Rule.insert is unsupported: rule rows are "
                "only valid together with their master row. Use "
                "AutoCurationRules.insert_rules(row, rule_rows)."
            )

        def insert1(self, row, **kwargs):
            raise UnsupportedDirectInsertError(
                "AutoCurationRules.Rule.insert1 is unsupported. Use "
                "AutoCurationRules.insert_rules(row, rule_rows)."
            )

    def insert1(self, row, **kwargs):
        raise UnsupportedDirectInsertError(
            "AutoCurationRules.insert1 is unsupported: a rules set is only "
            "valid together with its Rule rows. Use "
            "AutoCurationRules.insert_rules(row, rule_rows), which validates "
            "the master row and rule rows together."
        )

    def insert(self, rows, **kwargs):
        raise UnsupportedDirectInsertError(
            "AutoCurationRules.insert is unsupported. Use "
            "AutoCurationRules.insert_rules(row, rule_rows)."
        )

    @classmethod
    def insert_rules(
        cls, row: dict, rule_rows: list[dict], skip_duplicates: bool = False
    ) -> dict:
        """Validate and insert a rules master row plus its ordered rule rows.

        Validates the complete ``{master, rules}`` payload, then inserts the
        master and ``Rule`` rows in one transaction. Returns a PK-only dict.
        Idempotent when the master name already exists with the same payload;
        raises if the existing name maps to different rules.
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
        clean = _validate_params(AutoCurationRulesSchema, payload)
        master = {
            "auto_curation_rules_name": name,
            "auto_merge_preset": clean["auto_merge_preset"],
            "auto_merge_kwargs": clean["auto_merge_kwargs"],
            "params_schema_version": clean["schema_version"],
            "job_kwargs": row.get("job_kwargs"),
        }
        rule_inserts = [
            {"auto_curation_rules_name": name, **rule}
            for rule in clean["rules"]
        ]
        expected_payload = cls._payload_for_compare(master, rule_inserts)
        existing = cls & {"auto_curation_rules_name": name}
        if existing:
            stored_payload = cls._stored_payload_for_compare(name)
            if not rules_payloads_match(expected_payload, stored_payload):
                raise ValueError(
                    f"AutoCurationRules {name!r} already exists with a "
                    "different auto-merge/rule payload. Reuse the stored row, "
                    "choose a new auto_curation_rules_name for changed rules, "
                    "or delete the existing row deliberately before replacing it."
                )
            if not skip_duplicates:
                logger.warning(
                    f"AutoCurationRules {name!r} already exists with the same "
                    "payload; returning it."
                )
            return {"auto_curation_rules_name": name}

        inst = cls()
        with inst.connection.transaction:
            # Bypass the raising insert overrides on both the master and the
            # Rule part by calling the next insert in the MRO (SpyglassMixin /
            # dj.Table). insert_rules is the only validated write path.
            super(AutoCurationRules, inst).insert([master])
            if rule_inserts:
                rule_inst = cls.Rule()
                super(cls.Rule, rule_inst).insert(rule_inserts)
        return {"auto_curation_rules_name": name}

    @classmethod
    def _payload_for_compare(cls, master: dict, rule_rows: list[dict]) -> dict:
        """Normalize master + Rule rows for idempotency comparison."""
        rules = [
            {
                key: rule[key]
                for key in (
                    "rule_index",
                    "rule_name",
                    "metric_name",
                    "operator",
                    "threshold",
                    "label",
                )
            }
            for rule in rule_rows
        ]
        return _jsonable_blob(
            {
                "auto_curation_rules_name": master["auto_curation_rules_name"],
                "auto_merge_preset": master["auto_merge_preset"],
                "auto_merge_kwargs": master.get("auto_merge_kwargs") or {},
                "params_schema_version": master["params_schema_version"],
                "job_kwargs": master.get("job_kwargs"),
                "rules": sorted(rules, key=lambda rule: rule["rule_index"]),
            }
        )

    @classmethod
    def _stored_payload_for_compare(cls, name: str) -> dict:
        """Fetch and normalize one stored rules payload for comparison."""
        master = (cls & {"auto_curation_rules_name": name}).fetch1()
        rules = (cls.Rule & {"auto_curation_rules_name": name}).fetch(
            as_dict=True
        )
        return cls._payload_for_compare(master, list(rules))

    @classmethod
    def _default_payloads(cls) -> list[tuple[dict, list[dict]]]:
        return [
            (
                {
                    "auto_curation_rules_name": "none",
                    "auto_merge_preset": "none",
                },
                [],
            ),
            (
                {
                    "auto_curation_rules_name": "v1_default_nn_noise",
                    "auto_merge_preset": "none",
                },
                [
                    {
                        "rule_index": 0,
                        "rule_name": "nn_noise",
                        "metric_name": "nn_noise_overlap",
                        "operator": ">",
                        "threshold": 0.1,
                        "label": "noise",
                    },
                    {
                        "rule_index": 1,
                        "rule_name": "nn_reject",
                        "metric_name": "nn_noise_overlap",
                        "operator": ">",
                        "threshold": 0.1,
                        "label": "reject",
                    },
                ],
            ),
            (
                {
                    "auto_curation_rules_name": "similarity_merge",
                    "auto_merge_preset": "similarity_correlograms",
                },
                [],
            ),
            (
                # Frank-lab default labeling set: thresholds the lab's ~2% ISI
                # refractory policy in addition to nn_noise_overlap. ISI-violation
                # units are labeled ``reject`` (not ``mua``) so they fall out of
                # the default matchable-unit set (CurationV2.get_matchable_unit_ids
                # excludes reject/noise/artifact). Merges stay a manual step, so
                # this set runs no auto-merge (auto_merge_preset='none'). The
                # metric-params row it pairs with must compute nn_advanced (for
                # the nn_noise_overlap column) and isi_violation -- the shipped
                # ``franklab_default`` QualityMetricParameters row does both.
                {
                    "auto_curation_rules_name": (
                        "franklab_default_auto_curation_2026_06"
                    ),
                    "auto_merge_preset": "none",
                },
                [
                    {
                        "rule_index": 0,
                        "rule_name": "nn_noise",
                        "metric_name": "nn_noise_overlap",
                        "operator": ">",
                        "threshold": 0.1,
                        "label": "noise",
                    },
                    {
                        "rule_index": 1,
                        "rule_name": "isi_reject",
                        "metric_name": "isi_violation",
                        "operator": ">",
                        "threshold": 0.02,
                        "label": "reject",
                    },
                ],
            ),
        ]

    @classmethod
    def insert_default(cls):
        """Insert the default auto-curation rule sets (idempotent)."""
        for master, rules in cls._default_payloads():
            cls.insert_rules(master, rules, skip_duplicates=True)

    @classmethod
    def check_rule_integrity(cls, restriction=True) -> list[dict]:
        """Return master rows whose Rule rows are malformed or missing.

        ``insert``/``insert1`` on ``AutoCurationRules.Rule`` now raise, but
        out-of-band writes (``insert_quick`` / raw SQL) can still bypass
        whole-payload validation; this surfaces a CUSTOM rows-set that does
        nothing (preset ``none`` and no rules) or rule rows that fail the rule
        schema, for use in an integrity check. The shipped ``"none"`` default
        is the documented inert "skip auto-curation" sentinel, so it is exempt
        from the no-effect flag.
        """
        from spyglass.spikesorting.v2._params.metric_curation import (
            AutoCurationRuleSchema,
        )

        offenders = []
        for master in (cls & restriction).fetch(as_dict=True):
            name = master["auto_curation_rules_name"]
            rules = (cls.Rule & {"auto_curation_rules_name": name}).fetch(
                as_dict=True
            )
            if (
                name != "none"
                and master["auto_merge_preset"] == "none"
                and not rules
            ):
                offenders.append(
                    {"auto_curation_rules_name": name, "issue": "no_effect"}
                )
                continue
            for rule in rules:
                try:
                    AutoCurationRuleSchema.model_validate(
                        {
                            k: rule[k]
                            for k in AutoCurationRuleSchema.model_fields
                        }
                    )
                except Exception as err:  # noqa: BLE001 - report, don't raise
                    offenders.append(
                        {
                            "auto_curation_rules_name": name,
                            "rule_index": rule.get("rule_index"),
                            "issue": str(err),
                        }
                    )
        return offenders


class _WaveformsAccessor:
    """Narrow ``WaveformExtractor``-shaped view over a SortingAnalyzer.

    Exposes the two accessors v1 notebook code used: SI's
    ``get_waveforms_one_unit`` and v1's ``get_waveforms``, both reading the
    analyzer's ``waveforms`` extension.
    """

    def __init__(self, analyzer):
        self._analyzer = analyzer
        self._waveforms = analyzer.get_extension("waveforms")

    @property
    def sorting(self):
        return self._analyzer.sorting

    def get_waveforms_one_unit(self, unit_id):
        return self._waveforms.get_waveforms_one_unit(unit_id)

    def get_waveforms(self, unit_id):
        return self._waveforms.get_waveforms_one_unit(unit_id)


@schema
class CurationEvaluationSelection(
    SelectionMasterInsertGuard, SpyglassMixin, dj.Manual
):
    """A committed ``CurationV2`` row paired with metric / auto-rule params.

    ``CurationEvaluation`` scores the curation in ITS OWN unit namespace, so a
    committed applied-merge curation is a valid target (a merged unit is scored
    over its merged spike train, not the raw sort). What it rejects is a PREVIEW
    curation (``apply_merge=False`` with an unapplied proposed merge):
    evaluating that would score the unmerged preview units rather than the final
    merged set.

    Like the other deterministic-id selection masters, a raw ``insert`` /
    ``insert1`` is blocked; use ``insert_selection`` (which passes
    ``allow_direct_insert=True`` for its already-validated insert).
    """

    definition = """
    curation_evaluation_id: uuid
    ---
    -> CurationV2
    -> QualityMetricParameters
    -> AutoCurationRules
    -> AnalyzerWaveformParameters.proj(metric_waveform_params_name="waveform_params_name")
    """

    @classmethod
    def insert_selection(cls, key: dict) -> dict:
        """Insert or find a curation-evaluation selection; return PK-only dict.

        The ``curation_evaluation_id`` PK is content-addressed (a ``uuid5`` of
        the logical identity ``(sorting_id, curation_id, metric_params_name,
        auto_curation_rules_name, metric_waveform_params_name)``) under the
        ``"curation_evaluation"`` deterministic-id namespace, so the same
        logical selection always maps to one id.

        ``metric_waveform_params_name`` (the whitened analyzer recipe the PC/NN
        metrics compute on) defaults to the sort's region metric row, resolved
        source-aware from the sort source's preprocessing recipe (recording or
        concat). Pass it explicitly to override.

        Raises ``ValueError`` when the parent curation is a PREVIEW (unapplied
        proposed merges) -- evaluate a committed curation instead. Raises
        ``DuplicateSelectionError`` if an existing row for this identity carries
        a non-deterministic id.
        """
        from spyglass.spikesorting.v2._selection_identity import (
            assert_supplied_id_matches,
            deterministic_id,
        )
        from spyglass.spikesorting.v2.utils import _is_duplicate_key_error

        metric_waveform_params_name = key.get("metric_waveform_params_name")
        if metric_waveform_params_name is None:
            # Default = the sort's region metric (whitened) recipe, resolved
            # source-aware from the source preprocessing recipe (recording or
            # concat). ``[1]`` is the metric element of the (display, metric)
            # pair.
            preproc = SortingSelection.resolve_source_preprocessing_params_name(
                {"sorting_id": key["sorting_id"]}
            )
            metric_waveform_params_name = waveform_params_for_preprocessing(
                preproc
            )[1]

        # Reject a display/unwhitened recipe before folding it into identity;
        # the shipped default resolves a metric row, so this only fires on a bad
        # explicit value (re-checked in make_fetch for an allow_direct_insert
        # bypass).
        _assert_is_metric_recipe(metric_waveform_params_name)

        parent_key = {
            "sorting_id": key["sorting_id"],
            "curation_id": key["curation_id"],
        }
        # Reject a preview/draft curation BEFORE the find-existing early-return,
        # so a legacy / allow_direct_insert selection over a preview fails loudly
        # here rather than being handed back as a "valid" row (also re-asserted
        # in make_fetch so the preview compute path stays unreachable).
        CurationV2.assert_committed_curation(
            parent_key, context="CurationEvaluation"
        )

        identity = {
            "sorting_id": key["sorting_id"],
            "curation_id": key["curation_id"],
            "metric_params_name": key["metric_params_name"],
            "auto_curation_rules_name": key["auto_curation_rules_name"],
            "metric_waveform_params_name": metric_waveform_params_name,
        }
        curation_evaluation_id = deterministic_id(
            "curation_evaluation", identity
        )
        assert_supplied_id_matches(
            key.get("curation_evaluation_id"),
            curation_evaluation_id,
            field="curation_evaluation_id",
        )

        existing = cls._find_existing_pk(identity, curation_evaluation_id)
        if existing is not None:
            return existing

        new_key = {
            **identity,
            "curation_evaluation_id": curation_evaluation_id,
        }
        try:
            cls.insert1(new_key, allow_direct_insert=True)
        except Exception as exc:  # noqa: BLE001 - re-raised unless dup-PK race
            if not _is_duplicate_key_error(exc):
                raise
            existing = cls._find_existing_pk(identity, curation_evaluation_id)
            if existing is None:
                raise
            return existing
        return {"curation_evaluation_id": curation_evaluation_id}

    @classmethod
    def insert_by_curation_id(
        cls,
        sorting_id,
        curation_id,
        metric_params_name,
        auto_curation_rules_name,
    ) -> dict:
        """Insert a curation-evaluation selection for a curation (v1 analog).

        Assemble the content-addressed selection key from a curation
        (``sorting_id`` + ``curation_id``) and the named quality-metric /
        auto-rule rows, then delegate to :meth:`insert_selection`. Returns the
        PK-only dict.
        """
        return cls.insert_selection(
            {
                "sorting_id": sorting_id,
                "curation_id": curation_id,
                "metric_params_name": metric_params_name,
                "auto_curation_rules_name": auto_curation_rules_name,
            }
        )

    @classmethod
    def _find_existing_pk(cls, identity, deterministic_id):
        """Return the PK-only dict for ``identity`` or None; guard bad ids."""
        from spyglass.spikesorting.v2.exceptions import (
            DuplicateSelectionError,
        )

        existing_ids = (cls & identity).fetch("curation_evaluation_id")
        bypassed = [
            cid
            for cid in existing_ids
            if uuid.UUID(str(cid)) != deterministic_id
        ]
        if bypassed:
            raise DuplicateSelectionError(
                "CurationEvaluationSelection has duplicate selection rows for "
                f"{identity} with non-deterministic id(s) "
                f"{sorted(map(str, bypassed))} (expected the content-addressed "
                f"{deterministic_id}). This is an integrity bug -- a row was "
                "inserted bypassing insert_selection."
            )
        if len(existing_ids):
            return {"curation_evaluation_id": deterministic_id}
        return None

    @classmethod
    def pc_requesting(cls):
        """Selections whose QualityMetricParameters request PC/NN metrics.

        Joined to ``QualityMetricParameters`` and restricted to
        ``skip_pc_metrics=0`` -- the set whose evaluation builds a whitened
        metric analyzer (CurationEvaluation's fast path loads/builds the
        canonical ``analyzer_path(sorting_id, metric_waveform_params_name)``
        folder for PC/NN metrics). The single source of truth for which metric
        recipes are "in use", so the recompute key_source and the orphan-folder
        audit cannot drift apart. Carries ``sorting_id`` (a secondary CurationV2
        FK attr) and ``metric_waveform_params_name`` for the caller to project /
        fetch.
        """
        return cls * QualityMetricParameters & "skip_pc_metrics = 0"


@schema
class CurationEvaluation(SpyglassMixin, dj.Computed):
    """Quality metrics / merge suggestions / labels over a committed curation.

    The post-merge / final-metric path: scores an existing committed
    ``CurationV2`` row in that curation's OWN unit namespace -- a merged unit
    gets metrics recomputed over its merged spike trains/templates, never
    inherited from the highest-amplitude contributor. Outputs (metrics, proposed
    labels, merge suggestions) are written to three NWB scratch tables and are
    PROPOSALS; the acceptance helpers (``accept_evaluation_outputs`` /
    ``use_evaluation_labels`` / ``overlay_evaluation_labels`` /
    ``preview_merges``) commit them into a child ``CurationV2`` row.

    Routing: a committed root / label-only curation (unit set unchanged from the
    raw sort) reuses the cached raw-sort analyzers (the fast path); a committed
    applied-merge curation builds curation-scoped temporary analyzers over the
    merged sorting and cleans them immediately (never published to the canonical
    analyzer cache).
    """

    definition = """
    -> CurationEvaluationSelection
    ---
    -> AnalysisNwbfile
    metrics_object_id: varchar(72)
    merge_suggestions_object_id: varchar(72)
    proposed_labels_object_id: varchar(72)
    spikeinterface_version: varchar(32)  # spikeinterface.__version__ at eval time
    source_analyzer_hashes=null: blob    # role -> content_hash of each canonical analyzer consumed (fast path); NULL for merged-curation temp analyzers
    """
    # ``source_analyzer_hashes`` records the content the metrics were computed
    # over ONLY on the fast path, where the canonical analyzers are regeneratable
    # scratch (not pinned in the schema) -- so ``detect_stale_source`` can re-hash
    # them and flag drift. It is a {role: hash} manifest covering EVERY canonical
    # analyzer actually consumed: ``"display"`` always, plus ``"metric"`` when the
    # evaluation requests PC/NN metrics (which build + read the whitened metric
    # analyzer). On the merged path the temp analyzers are built deterministically
    # from the committed (immutable) curation unit set + the recipe, both
    # reachable through the selection, so no snapshot is needed and the column is
    # NULL. The evaluated curation identity + recipe names live on
    # CurationEvaluationSelection (the row's FK parent), not duplicated here.
    # Secondary provenance, never identity.

    # Tri-part make so the metric/merge compute (and NWB write) run OUTSIDE the
    # DB transaction. make_compute resolves no DB INPUTS: every upstream input
    # is read in make_fetch and threaded through the carrier (no get_sorting /
    # get_analyzer in compute). It does still stage its OUTPUT through
    # AnalysisNwbfile().create()/get_abs_path() -- the same off-transaction
    # write every v2 make_compute uses (Recording / Sorting /
    # ConcatenatedRecording) -- keeping the heavy NWB write off the commit txn.
    _parallel_make = True

    def make_fetch(self, key) -> CurationEvaluationFetched:
        """Resolve every DB input make_compute needs (no SI/NWB compute here).

        Resolves the params, the recording-reconstruction inputs (so the
        DB-free worker can rebuild the recording + sorting), the curated-units
        NWB abs path + expected unit ids, the analyzer recipes + cache folders,
        and the committed-state routing decision. Re-asserts the parent is a
        committed (non-preview) curation and the metric recipe is whitened, so a
        row planted via ``allow_direct_insert`` cannot reach the compute path.
        """
        from spyglass.spikesorting.v2._analyzer_cache import analyzer_path
        from spyglass.spikesorting.v2._artifact_intervals import (
            read_artifact_removed_intervals,
        )
        from spyglass.spikesorting.v2._sorting_analyzer import (
            fetch_waveform_params,
            resolve_display_waveform_params_name,
        )
        from spyglass.spikesorting.v2.recording import Recording
        from spyglass.spikesorting.v2.session_group import (
            ConcatenatedRecording,
        )

        sel = (CurationEvaluationSelection & key).fetch1()
        sorting_id = str(sel["sorting_id"])
        curation_id = int(sel["curation_id"])
        curation_key = {"sorting_id": sorting_id, "curation_id": curation_id}
        sorting_key = {"sorting_id": sorting_id}

        # Re-assert committed (reject a preview planted via allow_direct_insert)
        # and the metric recipe is whitened (same bypass rationale as
        # the sanctioned DB-fetch stage).
        merges_applied = (CurationV2 & curation_key).fetch1("merges_applied")
        CurationV2.assert_committed_curation(
            curation_key,
            context="CurationEvaluation",
            merges_applied=merges_applied,
        )
        _assert_is_metric_recipe(sel["metric_waveform_params_name"])

        qm = (
            QualityMetricParameters
            & {"metric_params_name": sel["metric_params_name"]}
        ).fetch1()
        acr = (
            AutoCurationRules
            & {"auto_curation_rules_name": sel["auto_curation_rules_name"]}
        ).fetch1()
        rule_rows = (
            AutoCurationRules.Rule
            & {"auto_curation_rules_name": sel["auto_curation_rules_name"]}
        ).fetch(as_dict=True)

        metric_names = list(qm["metric_names"])
        sorter_params = (
            SortingSelection * SorterParameters & sorting_key
        ).fetch1("params")
        metric_kwargs = apply_snr_peak_sign(
            metric_names, dict(qm["metric_kwargs"] or {}), sorter_params
        )

        # Recording reconstruction inputs. make_compute rebuilds the recording
        # DB-free from these; self-heal the regeneratable cache here (the DB
        # stage) so that read succeeds, mirroring Recording().get_recording's
        # rebuild-if-missing (the same self-heal get_analyzer provides).
        source = SortingSelection.resolve_source(sorting_key)
        artifact_detection_id = SortingSelection.resolve_artifact_detection(
            sorting_key
        )
        if source.kind == "recording":
            recording_id = source.key["recording_id"]
            Recording().get_recording({"recording_id": recording_id})
            recording_row = (
                Recording & {"recording_id": recording_id}
            ).fetch1()
        else:  # concatenated_recording
            recording_id = None
            ConcatenatedRecording().get_recording(source.key)
            recording_row = (ConcatenatedRecording & source.key).fetch1()
        fs = float(recording_row["sampling_frequency"])

        artifact_valid_times = None
        if source.kind == "recording" and artifact_detection_id is not None:
            from spyglass.spikesorting.v2.recording import RecordingSelection

            nwb_file_name = (
                RecordingSelection & {"recording_id": recording_id}
            ).fetch1("nwb_file_name")
            intervals_by_nwb = read_artifact_removed_intervals(
                {"artifact_detection_id": artifact_detection_id},
                as_dict=True,
            )
            if nwb_file_name not in intervals_by_nwb:
                raise ValueError(
                    "CurationEvaluation.make_fetch: artifact-removed intervals "
                    f"for nwb_file_name={nwb_file_name!r} not found among "
                    f"{sorted(intervals_by_nwb)} for artifact_detection_id="
                    f"{artifact_detection_id!r}; the ArtifactDetection may be "
                    "partially deleted."
                )
            artifact_valid_times = intervals_by_nwb[nwb_file_name]

        raw_units_abs_path = AnalysisNwbfile.get_abs_path(
            (Sorting & sorting_key).fetch1("analysis_file_name")
        )
        raw_n_units = int((Sorting & sorting_key).fetch1("n_units"))
        curated_units_abs_path = AnalysisNwbfile.get_abs_path(
            (CurationV2 & curation_key).fetch1("analysis_file_name")
        )
        expected_unit_ids = sorted(
            int(u) for u in (CurationV2.Unit & curation_key).fetch("unit_id")
        )

        # Routing: the cached raw-sort analyzer fast path is valid ONLY when the
        # curation's unit set IS the raw sort's unit set (a root, or a label-only
        # child of a non-merged ancestor) -- then the cached analyzer already
        # carries exactly these units. ``merges_applied`` is NOT the
        # discriminator: a label-only child of a MERGED parent has
        # merges_applied=False but its namespace includes merged parent ids
        # absent from the raw sort, so it must build a curation-scoped temp
        # analyzer over the curated sorting (same as an applied-merge row).
        use_fast_path = CurationV2.matches_raw_namespace(curation_key)

        display_waveform_params_name = resolve_display_waveform_params_name(
            Sorting(), sorting_id
        )
        display_waveform_params = fetch_waveform_params(
            display_waveform_params_name
        )
        metric_waveform_params_name = sel["metric_waveform_params_name"]
        metric_waveform_params = fetch_waveform_params(
            metric_waveform_params_name
        )

        sorter_row = (
            SorterParameters
            & (
                (SortingSelection & sorting_key).proj(
                    "sorter", "sorter_params_name"
                )
            )
        ).fetch1()

        return CurationEvaluationFetched(
            sorting_id=sorting_id,
            curation_id=curation_id,
            nwb_file_name=_nwb_file_name_for_sorting(sorting_key),
            source_kind=source.kind,
            recording_id=recording_id,
            artifact_detection_id=artifact_detection_id,
            artifact_valid_times=artifact_valid_times,
            recording_row=recording_row,
            fs=fs,
            raw_units_abs_path=raw_units_abs_path,
            raw_n_units=raw_n_units,
            curated_units_abs_path=curated_units_abs_path,
            expected_unit_ids=expected_unit_ids,
            use_fast_path=use_fast_path,
            display_waveform_params=display_waveform_params,
            display_analyzer_folder=str(
                analyzer_path(sorting_id, display_waveform_params_name)
            ),
            metric_waveform_params=metric_waveform_params,
            metric_analyzer_folder=str(
                analyzer_path(sorting_id, metric_waveform_params_name)
            ),
            metric_names=metric_names,
            metric_kwargs=metric_kwargs,
            template_metric_columns=list(qm["template_metric_columns"] or []),
            skip_pc_metrics=bool(qm["skip_pc_metrics"]),
            auto_merge_preset=acr["auto_merge_preset"],
            auto_merge_kwargs=dict(acr["auto_merge_kwargs"] or {}),
            rule_rows=list(rule_rows),
            sorter_row=sorter_row,
            analyzer_job_kwargs=_resolved_job_kwargs(sorter_row["job_kwargs"]),
            metric_job_kwargs=_resolved_job_kwargs(
                qm["job_kwargs"], acr["job_kwargs"]
            ),
            metric_params_name=sel["metric_params_name"],
            auto_curation_rules_name=sel["auto_curation_rules_name"],
            display_waveform_params_name=display_waveform_params_name,
            metric_waveform_params_name=metric_waveform_params_name,
        )

    def make_compute(
        self,
        key,
        sorting_id,
        curation_id,
        nwb_file_name,
        source_kind,
        recording_id,
        artifact_detection_id,
        artifact_valid_times,
        recording_row,
        fs,
        raw_units_abs_path,
        raw_n_units,
        curated_units_abs_path,
        expected_unit_ids,
        use_fast_path,
        display_waveform_params,
        display_analyzer_folder,
        metric_waveform_params,
        metric_analyzer_folder,
        metric_names,
        metric_kwargs,
        template_metric_columns,
        skip_pc_metrics,
        auto_merge_preset,
        auto_merge_kwargs,
        rule_rows,
        sorter_row,
        analyzer_job_kwargs,
        metric_job_kwargs,
        metric_params_name,
        auto_curation_rules_name,
        display_waveform_params_name,
        metric_waveform_params_name,
    ) -> CurationEvaluationComputed:
        """Compute metrics / merges / labels over the committed curation.

        No DB INPUT resolution (tri-part contract): the recording + raw /
        curated sortings are reconstructed from the threaded inputs, never via
        ``CurationV2.get_sorting`` / ``Sorting.get_analyzer`` (both DB-backed).
        The only DB touch is staging the OUTPUT artifact via
        ``AnalysisNwbfile().create()`` / ``get_abs_path()`` -- the standard v2
        off-transaction write (identical to ``Recording`` / ``Sorting`` /
        ``ConcatenatedRecording`` ``make_compute``) that keeps the heavy NWB
        write outside the commit transaction.
        """
        import tempfile
        from pathlib import Path

        import spikeinterface as si

        from spyglass.spikesorting.v2._analyzer_cache import (
            analyzer_curation_lock,
        )
        from spyglass.spikesorting.v2._nwb_provenance import (
            CURATION_EVALUATION_PROVENANCE,
            build_provenance_table,
        )
        from spyglass.spikesorting.v2._recompute import analyzer_role_hashes
        from spyglass.spikesorting.v2._sorting_analyzer import (
            build_analyzer,
            load_or_rebuild_analyzer_from_resolved,
            reconstruct_recording_for_sorting_from_resolved,
        )

        spikeinterface_version = si.__version__
        analysis_file_name = AnalysisNwbfile().create(
            nwb_file_name,
            restrict_permission=True,  # 0o644, not world-writable 0o666
        )
        abs_path = AnalysisNwbfile.get_abs_path(analysis_file_name)
        wants_pc = (not skip_pc_metrics) and bool(
            _requested_pc_metrics(metric_names)
        )

        # Concat sources carry no single recording_id (recording_id is None);
        # record the concat_recording_id (from the ConcatenatedRecording row) so
        # the file identifies its upstream standalone. recording_content_hash is
        # the source row's content_hash either way (the concat's, for a concat).
        concat_recording_id = (
            str(recording_row["concat_recording_id"])
            if source_kind == "concatenated_recording"
            else None
        )

        # Provenance header shared by the populated and zero-unit paths so every
        # CurationEvaluation artifact is self-describing; source_analyzer_hashes
        # is added per-path (a manifest on the fast path, None for a zero-unit
        # or merged-temp-analyzer evaluation).
        base_provenance = {
            "sorting_id": str(sorting_id),
            "curation_id": int(curation_id),
            "metric_params_name": metric_params_name,
            "auto_curation_rules_name": auto_curation_rules_name,
            "display_waveform_params_name": display_waveform_params_name,
            "metric_waveform_params_name": metric_waveform_params_name,
            "metric_names": list(metric_names),
            "metric_kwargs": metric_kwargs,
            "auto_merge_preset": auto_merge_preset,
            "auto_merge_kwargs": auto_merge_kwargs,
            "auto_curation_rules": rule_rows,
            "source_kind": source_kind,
            "recording_id": recording_id,
            "concat_recording_id": concat_recording_id,
            "recording_content_hash": recording_row["content_hash"],
            "spikeinterface_version": spikeinterface_version,
        }

        def _provenance_tables(source_analyzer_hashes):
            return [
                build_provenance_table(
                    CURATION_EVALUATION_PROVENANCE,
                    {
                        **base_provenance,
                        "source_analyzer_hashes": source_analyzer_hashes,
                    },
                )
            ]

        try:
            # Zero-unit committed curation: nothing to analyze; write empty
            # metric/merge/label tables (SI cannot build an analyzer over zero
            # units). Write empty metric/merge/label tables.
            if not expected_unit_ids:
                logger.warning(
                    "CurationEvaluation: curation "
                    f"(sorting_id={sorting_id}, curation_id={curation_id}) has "
                    "zero units; writing empty metric/merge/label tables."
                )
                object_ids = self._write_empty(
                    abs_path, provenance_tables=_provenance_tables(None)
                )
                return CurationEvaluationComputed(
                    analysis_file_name,
                    *object_ids,
                    nwb_file_name,
                    spikeinterface_version,
                    None,
                )

            recording = reconstruct_recording_for_sorting_from_resolved(
                recording_row=recording_row,
                source_kind=source_kind,
                artifact_valid_times=artifact_valid_times,
                artifact_detection_id=artifact_detection_id,
                recording_id=recording_id,
            )

            if use_fast_path:
                # Root / label-only: the cached raw-sort analyzers already carry
                # this curation's unit set. Hold the per-sort lock around the
                # canonical-folder load/rebuild + metric-extension mutation
                # (_compute_metrics / _compute_merge_groups mutate the shared
                # analyzer in place).
                raw_sorting = self._sorting_from_units_nwb(
                    raw_units_abs_path, recording_row, fs
                )
                with analyzer_curation_lock(sorting_id):
                    display_analyzer = load_or_rebuild_analyzer_from_resolved(
                        sorting_id=sorting_id,
                        n_units=raw_n_units,
                        analyzer_folder=Path(display_analyzer_folder),
                        waveform_params=display_waveform_params,
                        recording=recording,
                        sorting=raw_sorting,
                        sorter_row=sorter_row,
                        job_kwargs=analyzer_job_kwargs,
                    )
                    metric_analyzer = None
                    if wants_pc:
                        metric_analyzer = (
                            load_or_rebuild_analyzer_from_resolved(
                                sorting_id=sorting_id,
                                n_units=raw_n_units,
                                analyzer_folder=Path(metric_analyzer_folder),
                                waveform_params=metric_waveform_params,
                                recording=recording,
                                sorting=raw_sorting,
                                sorter_row=sorter_row,
                                job_kwargs=analyzer_job_kwargs,
                            )
                        )
                    metrics_df, labels_by_unit, merge_groups = (
                        self._evaluate_analyzers(
                            display_analyzer,
                            metric_analyzer,
                            metric_names=metric_names,
                            metric_kwargs=metric_kwargs,
                            skip_pc_metrics=skip_pc_metrics,
                            metric_job_kwargs=metric_job_kwargs,
                            template_metric_columns=template_metric_columns,
                            auto_merge_preset=auto_merge_preset,
                            auto_merge_kwargs=auto_merge_kwargs,
                            rule_rows=rule_rows,
                            expected_unit_ids=expected_unit_ids,
                        )
                    )
            else:
                # Applied-merge: build curation-scoped TEMP analyzers over the
                # merged curated sorting. Never published to the canonical
                # analyzer cache (identity is curation-scoped, not sorting-
                # scoped); cleaned on success and failure by TemporaryDirectory.
                curated_sorting = self._sorting_from_units_nwb(
                    curated_units_abs_path, recording_row, fs
                )
                compute_key = {"sorting_id": sorting_id}
                from spyglass.settings import temp_dir as spyglass_temp_dir

                with tempfile.TemporaryDirectory(dir=spyglass_temp_dir) as tmp:
                    # ``.zarr`` suffix matches the SI zarr store create_sorting_
                    # analyzer writes (SI forces it) so load_sorting_analyzer
                    # resolves the same folder -- same convention as
                    # analyzer_path for the canonical cache.
                    display_folder = Path(tmp) / "display.zarr"
                    build_analyzer(
                        curated_sorting,
                        recording,
                        compute_key,
                        sorter_row=sorter_row,
                        job_kwargs=analyzer_job_kwargs,
                        analyzer_folder=display_folder,
                        waveform_params=display_waveform_params,
                    )
                    display_analyzer = si.load_sorting_analyzer(display_folder)
                    metric_analyzer = None
                    if wants_pc:
                        metric_folder = Path(tmp) / "metric.zarr"
                        build_analyzer(
                            curated_sorting,
                            recording,
                            compute_key,
                            sorter_row=sorter_row,
                            job_kwargs=analyzer_job_kwargs,
                            analyzer_folder=metric_folder,
                            waveform_params=metric_waveform_params,
                        )
                        metric_analyzer = si.load_sorting_analyzer(
                            metric_folder
                        )
                    metrics_df, labels_by_unit, merge_groups = (
                        self._evaluate_analyzers(
                            display_analyzer,
                            metric_analyzer,
                            metric_names=metric_names,
                            metric_kwargs=metric_kwargs,
                            skip_pc_metrics=skip_pc_metrics,
                            metric_job_kwargs=metric_job_kwargs,
                            template_metric_columns=template_metric_columns,
                            auto_merge_preset=auto_merge_preset,
                            auto_merge_kwargs=auto_merge_kwargs,
                            rule_rows=rule_rows,
                            expected_unit_ids=expected_unit_ids,
                        )
                    )

            # On the fast path the metrics were computed over the canonical
            # raw-sort analyzer -- regeneratable scratch not pinned in the
            # schema -- so snapshot its content hash for stale detection. The
            # merged path's temp analyzer is pinned by the committed curation +
            # recipe (both reachable), so it records NULL.
            # Snapshot EVERY canonical analyzer the metrics were computed over on
            # the fast path: the display analyzer always, plus the whitened metric
            # analyzer when PC/NN metrics consumed it (metric_analyzer is None
            # otherwise). The merged path's temp analyzers are pinned by the
            # committed curation + recipe, so None.
            source_analyzer_hashes = (
                analyzer_role_hashes(display_analyzer, metric_analyzer)
                if use_fast_path
                else None
            )
            # Self-describing provenance: the evaluation inputs + the source
            # provenance the row stores (analyzer-hash manifest, SI version) plus
            # the upstream recording content hash (see ``base_provenance``).
            object_ids = write_analyzer_curation_tables(
                abs_path,
                metrics_df=metrics_df,
                merge_groups=merge_groups,
                labels_by_unit=labels_by_unit,
                unit_ids=[int(u) for u in metrics_df.index],
                provenance_tables=_provenance_tables(source_analyzer_hashes),
            )
            return CurationEvaluationComputed(
                analysis_file_name,
                *object_ids,
                nwb_file_name,
                spikeinterface_version,
                source_analyzer_hashes,
            )
        except Exception:
            self._cleanup_staged_file(analysis_file_name)
            raise

    def make_insert(
        self,
        key,
        analysis_file_name,
        metrics_object_id,
        merge_suggestions_object_id,
        proposed_labels_object_id,
        nwb_file_name,
        spikeinterface_version,
        source_analyzer_hashes,
    ) -> None:
        """Register the analysis file and insert the row (atomic).

        ``AnalysisNwbfile().add`` + ``insert1`` run inside
        ``transaction_or_noop`` so a failed insert never orphans a registered
        AnalysisNwbfile row (mirrors Sorting); the staged analysis file
        is removed on failure.
        """
        from spyglass.spikesorting.v2.utils import transaction_or_noop

        try:
            with transaction_or_noop(self.connection):
                AnalysisNwbfile().add(nwb_file_name, analysis_file_name)
                self.insert1(
                    {
                        **key,
                        "analysis_file_name": analysis_file_name,
                        "metrics_object_id": metrics_object_id,
                        "merge_suggestions_object_id": (
                            merge_suggestions_object_id
                        ),
                        "proposed_labels_object_id": proposed_labels_object_id,
                        "spikeinterface_version": spikeinterface_version,
                        "source_analyzer_hashes": source_analyzer_hashes,
                    }
                )
        except Exception:
            self._cleanup_staged_file(analysis_file_name)
            raise

    @classmethod
    def detect_stale_source(cls, key) -> dict:
        """Flag whether an evaluation's recorded source provenance still holds.

        Compares the stored ``spikeinterface_version`` to the running SI version
        and, for a fast-path evaluation, re-hashes EVERY canonical analyzer the
        evaluation recorded consuming (``source_analyzer_hashes`` -- ``"display"``
        always, plus ``"metric"`` when PC/NN metrics were requested) and compares
        each to its stored hash, so a regenerated/mutated analyzer cache or a
        library upgrade surfaces as drift. A merged-curation evaluation stored
        ``source_analyzer_hashes=NULL`` (its temp analyzers are pinned by the
        committed curation + recipe, both reachable), so only the SI version is
        compared.

        Parameters
        ----------
        key : dict
            Restriction selecting a single ``CurationEvaluation`` row.

        Returns
        -------
        dict
            ``{"stale": bool, "reasons": list[str],
            "spikeinterface_version": {"stored", "current"},
            "source_analyzer_hashes": {"stored", "current"}}``. ``reasons`` names
            each drifted field: ``source_analyzer_hash:<role>`` for a content
            mismatch, or ``source_analyzer_missing:<role>`` /
            ``source_analyzer_invalid:<role>`` when a regeneratable analyzer cache
            was reclaimed or corrupted (reported as stale, never raised, since the
            stored metrics can no longer be reproduced from it). ``current`` is
            the re-hashed manifest (an absent/invalid role maps to ``None``; empty
            on the merged path).
        """
        import spikeinterface as si

        from spyglass.spikesorting.v2._recompute import analyzer_hash_for_role
        from spyglass.spikesorting.v2.exceptions import (
            AnalyzerFolderInvalidError,
            AnalyzerFolderMissingError,
            ZeroUnitAnalyzerError,
        )
        from spyglass.spikesorting.v2.sorting import Sorting

        row = (cls & key).fetch1()
        sel = (CurationEvaluationSelection & key).fetch1()
        reasons: list[str] = []

        current_version = si.__version__
        if row["spikeinterface_version"] != current_version:
            reasons.append("spikeinterface_version")

        # Re-hash each canonical analyzer the evaluation recorded consuming and
        # compare per role, using the shared analyzer_hash_for_role primitive so
        # the store and compare sides apply ONE role -> hashed-extensions mapping
        # (the metric role includes principal_components). The "display" role
        # reloads the default recipe, "metric" the whitened metric recipe. These
        # analyzers are regeneratable scratch, so a reclaimed/corrupt cache is
        # reported as stale per role -- NOT raised, which would abort the caller.
        # The merged path stored None, so only the SI version is checked there.
        stored_hashes = row["source_analyzer_hashes"]
        current_hashes: dict[str, str | None] = {}
        if stored_hashes:
            sort_key = {"sorting_id": str(sel["sorting_id"])}
            recipe_for = {
                "display": None,
                "metric": sel["metric_waveform_params_name"],
            }
            for role, stored in stored_hashes.items():
                try:
                    analyzer = Sorting().get_analyzer(
                        sort_key,
                        waveform_params_name=recipe_for[role],
                        rebuild=False,
                    )
                # AnalyzerFolderInvalidError subclasses AnalyzerFolderMissingError,
                # so catch the invalid/zero-unit cases first.
                except (AnalyzerFolderInvalidError, ZeroUnitAnalyzerError):
                    current_hashes[role] = None
                    reasons.append(f"source_analyzer_invalid:{role}")
                    continue
                except AnalyzerFolderMissingError:
                    current_hashes[role] = None
                    reasons.append(f"source_analyzer_missing:{role}")
                    continue
                current = analyzer_hash_for_role(analyzer, role)
                current_hashes[role] = current
                if current != stored:
                    reasons.append(f"source_analyzer_hash:{role}")

        return {
            "stale": bool(reasons),
            "reasons": reasons,
            "spikeinterface_version": {
                "stored": row["spikeinterface_version"],
                "current": current_version,
            },
            "source_analyzer_hashes": {
                "stored": stored_hashes,
                "current": current_hashes,
            },
        }

    # ---- compute helpers (DB-free; SI work) ------------------------------

    @staticmethod
    def _sorting_from_units_nwb(abs_path, recording_row, fs):
        """Reconstruct a ``NumpySorting`` from a units NWB (no DB).

        The same machinery ``CurationV2.get_sorting`` / ``Sorting.get_sorting``
        use internally, minus the DB fetch: new files reconstruct from stored
        sample frames; legacy files map absolute times to frames against the
        recording timestamps. Works for both the raw-sort and curated-units NWB
        (an applied-merge curated NWB already stores the MERGED unit set).
        """
        from spyglass.spikesorting.v2._units_nwb import (
            numpysorting_from_abs_times,
            numpysorting_from_sample_indices,
            read_units_abs_spike_times,
            read_units_spike_sample_indices,
        )

        sample_indices = read_units_spike_sample_indices(abs_path)
        if sample_indices is not None:
            return numpysorting_from_sample_indices(sample_indices, fs)
        abs_times = read_units_abs_spike_times(abs_path)
        return numpysorting_from_abs_times(abs_times, recording_row, fs)

    def _evaluate_analyzers(
        self,
        display_analyzer,
        metric_analyzer,
        *,
        metric_names,
        metric_kwargs,
        skip_pc_metrics,
        metric_job_kwargs,
        template_metric_columns,
        auto_merge_preset,
        auto_merge_kwargs,
        rule_rows,
        expected_unit_ids,
    ):
        """Compute metrics / labels / merge suggestions and enforce namespace.

        Reuses ``_compute_metrics`` / ``_compute_merge_groups`` and
        ``apply_label_rules`` over the curation's analyzers, then enforces
        the unit-namespace invariant BEFORE labels/merges are returned (and
        before the NWB write): the metric index must equal the curation's unit
        set, and every suggested merge member must be a unit in that set. This
        catches a stale temp analyzer, accidental raw-sort analyzer reuse, or a
        preview row that slipped past selection.
        """
        metrics_df = self._compute_metrics(
            display_analyzer,
            metric_analyzer,
            metric_names,
            metric_kwargs,
            skip_pc_metrics,
            metric_job_kwargs,
            template_metric_columns=template_metric_columns,
        )
        self._assert_unit_namespace(metrics_df, expected_unit_ids)
        labels_by_unit = apply_label_rules(metrics_df, rule_rows)
        merge_groups = self._compute_merge_groups(
            display_analyzer,
            auto_merge_preset,
            auto_merge_kwargs,
            metric_job_kwargs,
        )
        self._assert_merge_membership(merge_groups, expected_unit_ids)
        return metrics_df, labels_by_unit, merge_groups

    @staticmethod
    def _assert_unit_namespace(metrics_df, expected_unit_ids) -> None:
        """Raise unless the metric index equals the curation's unit set."""
        computed = {int(u) for u in metrics_df.index}
        expected = {int(u) for u in expected_unit_ids}
        if computed != expected:
            raise ValueError(
                "CurationEvaluation namespace invariant violated: computed "
                f"metric unit ids {sorted(computed)} != the curation's unit set "
                f"{sorted(expected)}. The metrics must be scored over the "
                "evaluated curation's own units (e.g. the merged unit set), not "
                "the raw-sort analyzer; this indicates a stale temp analyzer or "
                "accidental raw-analyzer reuse."
            )

    @staticmethod
    def _assert_merge_membership(merge_groups, expected_unit_ids) -> None:
        """Raise unless every suggested merge member is a curation unit."""
        expected = {int(u) for u in expected_unit_ids}
        for group in merge_groups:
            members = {int(u) for u in group}
            if not members <= expected:
                raise ValueError(
                    "CurationEvaluation merge-suggestion invariant violated: "
                    f"suggested merge {sorted(members)} contains units outside "
                    f"the curation's unit set {sorted(expected)}."
                )

    # ---- fetch helpers (read the persisted scratch tables) ---------------

    @classmethod
    def get_metrics(cls, key):
        """Return the quality-metrics table (DataFrame indexed by unit_id)."""
        row = (cls & key).fetch1()
        abs_path = AnalysisNwbfile.get_abs_path(row["analysis_file_name"])
        return read_quality_metrics(abs_path, row["metrics_object_id"])

    @classmethod
    def get_labels(cls, key) -> dict:
        """Return ``{unit_id: [label, ...]}`` for units with proposed labels."""
        row = (cls & key).fetch1()
        abs_path = AnalysisNwbfile.get_abs_path(row["analysis_file_name"])
        return read_proposed_labels(abs_path, row["proposed_labels_object_id"])

    @classmethod
    def get_suggested_merge_groups(cls, key) -> list:
        """Return proposed merge groups as a list of unit-id lists."""
        row = (cls & key).fetch1()
        abs_path = AnalysisNwbfile.get_abs_path(row["analysis_file_name"])
        return read_merge_suggestions(
            abs_path, row["merge_suggestions_object_id"]
        )

    # ---- acceptance helpers (evaluation outputs -> committed curation) ----

    def _evaluated_curation_key(self, key) -> dict:
        """Resolve a CurationEvaluation key to its evaluated curation key.

        Requires the ``CurationEvaluation`` row to be POPULATED, not merely
        selected. Acceptance writes ``curation_source='curation_evaluation'``
        children, so the workflow contract is "evaluate, THEN accept": minting
        an evaluation-sourced child from a bare selection (no computed
        metrics/proposals) would be a provenance lie. The check holds even when
        labels / merge groups are supplied explicitly -- the provenance tag
        claims an evaluation backs the child regardless of how the merges/labels
        were chosen.
        """
        if not (CurationEvaluation & key):
            raise ValueError(
                "CurationEvaluation acceptance requires a POPULATED evaluation "
                f"(curation_source='curation_evaluation'); no CurationEvaluation "
                f"row for {dict(key)}. Call CurationEvaluation.populate(key) "
                "before accepting (accept_evaluation_outputs / accept_merges / "
                "preview_merges / use_evaluation_labels / ...)."
            )
        sel = (CurationEvaluationSelection & key).fetch1()
        return {
            "sorting_id": sel["sorting_id"],
            "curation_id": int(sel["curation_id"]),
        }

    def _resolve_accepted_merges(
        self, key, merge_groups, use_all_suggested_merges
    ) -> list[list[int]]:
        """Resolve the merge groups to accept (explicit, all-suggested, none).

        Never applies all suggested merges implicitly: the caller must pass an
        explicit ``merge_groups`` OR ``use_all_suggested_merges=True``.

        CALLER-SUPPLIED ``merge_groups`` are returned VERBATIM (only coerced to
        ints) -- they are NOT silently filtered, so a singleton/empty group
        reaches ``CurationV2.insert_curation``'s >=2-member typo guard and
        raises instead of degrading into a labels-only child. Only the PERSISTED
        suggestions (``use_all_suggested_merges``) are filtered to the real
        (>=2-member) groups, since the stored suggestion set is not a caller
        typo. All ids are in the evaluated curation's own unit namespace.
        """
        if merge_groups is not None and use_all_suggested_merges:
            raise ValueError(
                "CurationEvaluation acceptance: pass either merge_groups or "
                "use_all_suggested_merges=True, not both."
            )
        if use_all_suggested_merges:
            return [
                [int(u) for u in group]
                for group in self.get_suggested_merge_groups(key)
                if len(group) >= 2
            ]
        if merge_groups is not None:
            return [[int(u) for u in group] for group in merge_groups]
        return []

    def accept_evaluation_outputs(
        self,
        key,
        *,
        merge_groups=None,
        use_all_suggested_merges: bool = False,
        labels: dict | None = None,
        label_policy: str = "replace",
        description: str = "accepted from curation evaluation",
        allow_custom_labels: bool = False,
        reuse_existing: bool = True,
    ) -> dict:
        """Accept selected evaluation outputs into a COMMITTED child curation.

        Creates a new committed child ``CurationV2`` row branched off the
        evaluated curation, in that curation's own unit namespace. The child is
        always committed (``assert_committed_curation`` true) -- it never leaves
        a preview row with unapplied proposed merges; use :meth:`preview_merges`
        for an explicit draft.

        Merges are applied ONLY when the caller is explicit: pass
        ``merge_groups`` (the exact accepted groups, in the evaluated
        curation's namespace) or ``use_all_suggested_merges=True`` (apply every
        >=2-member suggestion this evaluation proposed). Neither -> a
        labels-only committed child.

        Labels default to the evaluation's proposed labels
        (:meth:`get_labels`) and, with the default ``label_policy="replace"``,
        are the child's FULL label state -- the auto-curation verdict, not
        layered on the parent's labels. This matches v1 (a re-evaluation writes
        the complete label state), so a unit the rules no longer flag is not
        left carrying a stale ``reject`` / ``noise`` from an earlier curation
        (which would silently drop it from the matchable-unit set downstream).
        Pass ``label_policy="inherit"`` to instead overlay the proposals on the
        parent's labels, or ``labels`` to supply the state explicitly. Accepted
        children carry the ``curation_source='curation_evaluation'`` provenance.

        Parameters
        ----------
        key : dict
            Restriction selecting a single ``CurationEvaluation`` row.
        merge_groups : list[list[int]] or None
            Explicit accepted merge groups in the evaluated curation's unit
            namespace. ``None`` (with ``use_all_suggested_merges=False``) makes
            this a labels-only acceptance.
        use_all_suggested_merges : bool
            Apply every >=2-member suggested merge group instead of an explicit
            list. Mutually exclusive with ``merge_groups``.
        labels : dict or None
            ``{unit_id: [label, ...]}`` to write; ``None`` defaults to the
            evaluation's proposed labels. The proposed labels are in the
            evaluated curation's PRE-merge namespace: when this call also
            applies a merge, a label on a unit absorbed by that merge cannot
            attach to the fresh merged unit and is dropped with a warning -- the
            merged unit's labels come from RE-EVALUATING the merged child (the
            recommended accept-merge-then-evaluate workflow), not from the
            pre-merge proposals. Combine merges + labels in one call only when
            the labels are on surviving (non-absorbed) units.
        label_policy : str
            ``"replace"`` (default for acceptance: the proposed labels are the
            full label state) or ``"inherit"`` -- see
            ``CurationV2.insert_curation``.
        description : str
            Free-text description for the child curation.
        allow_custom_labels : bool
            Forwarded to ``CurationV2.insert_curation``.
        reuse_existing : bool
            Reuse an existing matching child instead of staging a new NWB
            (idempotent re-acceptance). Defaults True.

        Returns
        -------
        dict
            ``{"sorting_id", "curation_id"}`` of the committed child curation.
        """
        curation_key = self._evaluated_curation_key(key)
        accepted = self._resolve_accepted_merges(
            key, merge_groups, use_all_suggested_merges
        )
        effective_labels = self.get_labels(key) if labels is None else labels
        return CurationV2.insert_curation(
            {"sorting_id": curation_key["sorting_id"]},
            labels=effective_labels or None,
            merge_groups=accepted or None,
            apply_merge=bool(accepted),
            parent_curation_id=curation_key["curation_id"],
            description=description,
            curation_source="curation_evaluation",
            label_policy=label_policy,
            allow_custom_labels=allow_custom_labels,
            reuse_existing=reuse_existing,
        )

    def _require_merge_acceptance(
        self,
        key,
        merge_groups,
        use_all_suggested_merges: bool,
        *,
        action_name: str,
    ) -> list[list[int]]:
        """Resolve merge groups for an action method and require a real merge.

        Enforces the populated-evaluation contract BEFORE resolving suggestions:
        the ``use_all_suggested_merges`` path reads the evaluation NWB via
        ``get_suggested_merge_groups``, which on an unpopulated selection would fail with
        an opaque fetch error instead of the friendly "populate first" message.
        """
        self._evaluated_curation_key(key)  # populated-evaluation guard
        accepted = self._resolve_accepted_merges(
            key, merge_groups, use_all_suggested_merges
        )
        if not accepted:
            raise ValueError(
                f"CurationEvaluation.{action_name} needs at least one merge "
                "group. Pass merge_groups=[[...]] or use a selection with "
                "persisted merge suggestions."
            )
        return accepted

    def preview_merges(
        self,
        key,
        *,
        merge_groups=None,
        use_all_suggested_merges: bool = False,
        labels: dict | None = None,
        description: str = "draft merge(s) from curation evaluation",
        allow_custom_labels: bool = False,
        reuse_existing: bool = True,
    ) -> dict:
        """Draft selected merge suggestions without committing the merge.

        Action-oriented alias for the review-before-commit workflow. By default
        it drafts only merges and inherits the evaluated curation's labels; it
        does not apply pre-merge evaluation labels to the draft.
        """
        # _create_preview_curation (whose only caller is this method) resolves
        # the merge groups and enforces the populated-evaluation + non-empty
        # contracts itself, so pass the request through rather than resolving
        # the same suggestions twice.
        return self._create_preview_curation(
            key,
            merge_groups=merge_groups,
            use_all_suggested_merges=use_all_suggested_merges,
            labels={} if labels is None else labels,
            label_policy="inherit",
            description=description,
            allow_custom_labels=allow_custom_labels,
            reuse_existing=reuse_existing,
        )

    def accept_merges(
        self,
        key,
        *,
        merge_groups,
        description: str = "accepted merge(s) from curation evaluation",
        allow_custom_labels: bool = False,
        reuse_existing: bool = True,
    ) -> dict:
        """Commit selected merge groups into a child curation.

        This is the recommended merge-acceptance action: it commits the merged
        unit set and inherits existing labels, but deliberately does NOT apply
        pre-merge evaluation labels. Re-evaluate the merged child, then call
        :meth:`use_evaluation_labels` or :meth:`overlay_evaluation_labels`.
        ``allow_custom_labels`` is forwarded so an inherited custom (non-canonical)
        parent label does not fail the child insert.
        """
        accepted = self._require_merge_acceptance(
            key,
            merge_groups,
            False,
            action_name="accept_merges",
        )
        return self.accept_evaluation_outputs(
            key,
            merge_groups=accepted,
            labels={},
            label_policy="inherit",
            description=description,
            allow_custom_labels=allow_custom_labels,
            reuse_existing=reuse_existing,
        )

    def accept_all_suggested_merges(
        self,
        key,
        *,
        description: str = "accepted all suggested merges from curation evaluation",
        allow_custom_labels: bool = False,
        reuse_existing: bool = True,
    ) -> dict:
        """Commit every persisted >=2-unit merge suggestion from this evaluation.

        Inherits existing labels (``allow_custom_labels`` forwarded so an
        inherited custom parent label does not fail the child insert).
        """
        accepted = self._require_merge_acceptance(
            key,
            None,
            True,
            action_name="accept_all_suggested_merges",
        )
        return self.accept_evaluation_outputs(
            key,
            merge_groups=accepted,
            labels={},
            label_policy="inherit",
            description=description,
            allow_custom_labels=allow_custom_labels,
            reuse_existing=reuse_existing,
        )

    def use_evaluation_labels(
        self,
        key,
        *,
        labels: dict | None = None,
        description: str = "evaluation labels (replace)",
        allow_custom_labels: bool = False,
        reuse_existing: bool = True,
    ) -> dict:
        """Use the evaluation verdict as the child's full label state.

        The authoritative "use this evaluation's labels" path (label-only, no
        merges) -- the child's labels are exactly ``labels`` (defaulting to the
        evaluation's proposed labels), CLEARING any label the evaluation does
        not propose. A unit no longer flagged loses a stale ``reject`` /
        ``noise`` (v1 "final auto-curation writes the full label state"), so it
        is not silently dropped from the matchable-unit set. This is the default
        final-metrics path; use :meth:`overlay_evaluation_labels` to instead
        keep the current labels.

        Returns the child's ``{"sorting_id", "curation_id"}``.
        """
        return self.accept_evaluation_outputs(
            key,
            merge_groups=None,
            use_all_suggested_merges=False,
            labels=labels,
            label_policy="replace",
            description=description,
            allow_custom_labels=allow_custom_labels,
            reuse_existing=reuse_existing,
        )

    def overlay_evaluation_labels(
        self,
        key,
        *,
        labels: dict | None = None,
        description: str = "evaluation labels (overlay)",
        allow_custom_labels: bool = False,
        reuse_existing: bool = True,
    ) -> dict:
        """Overlay the evaluation's labels ON TOP of the curation's current ones.

        The manual-curation path (label-only, no merges): KEEP the evaluated
        curation's existing labels and add/override only the proposed ones.
        Deliberately a different method from :meth:`use_evaluation_labels` so
        the "keep my labels" choice is visible at the call site rather than a
        quiet flag -- overlaying can retain prior auto labels, which
        :meth:`use_evaluation_labels` (the default verdict path) clears.

        Returns the child's ``{"sorting_id", "curation_id"}``.
        """
        return self.accept_evaluation_outputs(
            key,
            merge_groups=None,
            use_all_suggested_merges=False,
            labels=labels,
            label_policy="inherit",
            description=description,
            allow_custom_labels=allow_custom_labels,
            reuse_existing=reuse_existing,
        )

    def _create_preview_curation(
        self,
        key,
        *,
        merge_groups=None,
        use_all_suggested_merges: bool = False,
        labels: dict | None = None,
        label_policy: str = "replace",
        description: str = "draft from curation evaluation",
        allow_custom_labels: bool = False,
        reuse_existing: bool = True,
    ) -> dict:
        """Create a DRAFT (preview) child from the evaluation's outputs.

        The explicit opt-in for drafting a merge for review before committing:
        the proposed merges are recorded in ``CurationV2.MergeGroup`` WITHOUT
        being applied (``apply_merge=False``), so the child is a preview --
        ``has_unapplied_proposed_merges`` is True and downstream consumers
        reject it until it is committed. Distinct from
        :meth:`accept_evaluation_outputs`, which only ever produces committed
        children. A preview is, by
        definition, an UNAPPLIED merge for review, so it must actually draft a
        merge: pass ``merge_groups`` or ``use_all_suggested_merges=True`` (and
        the latter must resolve at least one merge). With no merge this would
        otherwise produce a normal committed labels-only child, contradicting
        the "preview/draft" contract -- so it raises instead. For a committed
        labels-only child use :meth:`use_evaluation_labels` /
        :meth:`overlay_evaluation_labels`.

        Returns the child's ``{"sorting_id", "curation_id"}``.
        """
        curation_key = self._evaluated_curation_key(key)
        accepted = self._resolve_accepted_merges(
            key, merge_groups, use_all_suggested_merges
        )
        if not accepted:
            raise ValueError(
                "preview_merges drafts an UNAPPLIED merge for "
                "review, so it needs at least one merge: pass "
                "merge_groups=[[...]] or use_all_suggested_merges=True (with "
                "proposed merges present). For a committed labels-only child, "
                "call use_evaluation_labels() or overlay_evaluation_labels() "
                "instead."
            )
        effective_labels = self.get_labels(key) if labels is None else labels
        return CurationV2.insert_curation(
            {"sorting_id": curation_key["sorting_id"]},
            labels=effective_labels or None,
            merge_groups=accepted or None,
            apply_merge=False,
            parent_curation_id=curation_key["curation_id"],
            description=description,
            curation_source="curation_evaluation",
            label_policy=label_policy,
            allow_custom_labels=allow_custom_labels,
            reuse_existing=reuse_existing,
        )

    # ---- compute helpers (DB-light; SI work) -----------------------------

    @staticmethod
    def _compute_metrics(
        display_analyzer,
        metric_analyzer,
        metric_names,
        metric_kwargs,
        skip_pc_metrics,
        job_kwargs=None,
        template_metric_columns=None,
    ):
        """Compute quality metrics, routing PC/NN metrics to the whitened one.

        Voltage / spike-train metrics (``snr``, ``amplitude_*``,
        ``firing_rate``, ``num_spikes``, ``presence_ratio``, ``isi_violation``)
        compute on the unwhitened DISPLAY analyzer -- whitening normalizes
        per-channel variance, so SNR / amplitude on whitened traces would be
        meaningless. PC / cluster-separation metrics (the SI PCA-metric set)
        compute on the whitened METRIC analyzer, where decorrelated separation
        is meaningful. The two frames are merged by unit id. Spyglass's
        ``isi_violation`` fraction is added from the (recipe-independent) spike
        times. ``job_kwargs`` drive the heavy extension computation.

        ``template_metric_columns`` (SI output COLUMN names) are surfaced from
        the DISPLAY analyzer's ``template_metrics`` extension -- waveform SHAPE
        must come from real, unwhitened templates, exactly as ``snr`` does. The
        columns are selected directly (config already holds column names, so no
        name->column mapping) and joined onto the result by unit id; they are
        exposed for downstream cell typing, never thresholded here.
        """
        import numpy as np
        import pandas as pd
        from spikeinterface.metrics.quality import compute_quality_metrics

        from spyglass.spikesorting.v2._sorting_analyzer import (
            ensure_extensions,
        )

        metric_kwargs = metric_kwargs or {}
        pc_names = _requested_pc_metrics(metric_names)
        pc_set = set(pc_names)
        voltage_names = [m for m in metric_names if m not in pc_set]

        frames = []
        # Voltage / spike-train metrics -> unwhitened display analyzer.
        if voltage_names:
            # Compute each requested voltage metric's display-safe extension
            # dependencies (read from SI's registry, not hardcoded) beyond the
            # default curation set -- otherwise SI silently skips a metric whose
            # extension is absent (e.g. ``drift`` needs ``spike_locations``),
            # leaving the column missing and any rule thresholding it never
            # firing. ``principal_components`` is excluded defensively: it is
            # metric-analyzer-only and voltage metrics never depend on it.
            base_present = {
                "random_spikes",
                "noise_levels",
                "templates",
                "waveforms",
                *_CURATION_EXTENSIONS,
            }
            extra_extensions = [
                ext
                for ext in required_extensions_for_metrics(
                    voltage_names, base_present
                )
                if ext != "principal_components"
            ]
            ensure_extensions(
                display_analyzer,
                list(_CURATION_EXTENSIONS) + extra_extensions,
                job_kwargs=job_kwargs,
            )
            voltage_df = compute_quality_metrics(
                display_analyzer,
                metric_names=voltage_names,
                metric_params={
                    k: v for k, v in metric_kwargs.items() if k in voltage_names
                }
                or None,
                skip_pc_metrics=True,
                # The analyzer is shared across curations; SI preserves the
                # stored quality_metrics by default, so a prior curation's
                # columns would leak into this result (and an auto-rule could
                # threshold a stale metric). Compute only THIS row's metrics.
                delete_existing_metrics=True,
            )
            voltage_df.index = voltage_df.index.astype(int)
            frames.append(voltage_df)

        # PC / cluster-separation metrics -> whitened metric analyzer.
        if pc_names and not skip_pc_metrics:
            if metric_analyzer is None:
                raise ValueError(
                    "PC/NN metrics were requested but no whitened metric "
                    "analyzer was provided -- make_compute must build it when "
                    "PC metrics are requested."
                )
            # nn_noise_overlap needs a 'median' template AND SI 0.104.3's metric
            # is broken for sparse analyzers (it derives the peak channel from a
            # dense median but indexes the sparse noise cluster -> IndexError,
            # swallowed as NaN). Ensure the median operator, then install the
            # sparse fix. The PC compute below runs n_jobs=1 so the fix (a
            # main-process monkeypatch) is the code that actually runs.
            from spyglass.spikesorting.v2._si_metric_patches import (
                patch_nn_noise_overlap_sparsity,
            )

            ensure_extensions(
                metric_analyzer, ["templates"], job_kwargs=job_kwargs
            )
            templates_operators = list(
                metric_analyzer.get_extension("templates").params.get(
                    "operators"
                )
                or []
            )
            if "median" not in templates_operators:
                metric_analyzer.compute(
                    "templates", operators=[*templates_operators, "median"]
                )
            patch_nn_noise_overlap_sparsity()
            # Enforce the pinned PCA params even if a stale/manual analyzer
            # already carries principal_components computed with different ones
            # (ensure_extensions skips a present extension without checking its
            # params): drop a mismatched extension so it recomputes pinned.
            if metric_analyzer.has_extension("principal_components"):
                existing_pca = dict(
                    metric_analyzer.get_extension("principal_components").params
                )
                if not _pca_params_match(existing_pca):
                    logger.warning(
                        "CurationEvaluation: principal_components on the metric "
                        f"analyzer has params {existing_pca} != pinned "
                        f"{_PCA_EXTENSION_PARAMS}; deleting and recomputing "
                        "with the pinned params so PC/NN metrics are consistent."
                    )
                    metric_analyzer.delete_extension("principal_components")
            ensure_extensions(
                metric_analyzer,
                ["principal_components"],
                job_kwargs=job_kwargs,
                extension_params={
                    "principal_components": _PCA_EXTENSION_PARAMS
                },
            )
            pc_df = compute_quality_metrics(
                metric_analyzer,
                metric_names=pc_names,
                metric_params={
                    k: v for k, v in metric_kwargs.items() if k in pc_names
                }
                or None,
                skip_pc_metrics=False,
                # As above: compute only this row's PC metrics, never inherit a
                # prior curation's stored quality_metrics on this analyzer.
                delete_existing_metrics=True,
                # nn_noise_overlap's sparse fix (patch_nn_noise_overlap_sparsity)
                # is a main-process monkeypatch; SI parallelises nn_advanced with
                # spawned workers that re-import SI and would not see it, so pin
                # the PC/NN metric compute to the main process.
                n_jobs=1,
            )
            pc_df.index = pc_df.index.astype(int)
            frames.append(pc_df)

        if not frames:
            raise ValueError(
                "_compute_metrics: no metrics to compute -- metric_names "
                f"{sorted(metric_names)} contains only PC/NN metrics but "
                "skip_pc_metrics=True. Set skip_pc_metrics=False to compute "
                "them, or include a voltage-based metric."
            )
        if len(frames) > 1:
            # The display and metric analyzers derive from the SAME canonical
            # sorting, so the voltage and PC frames must share a unit-id index.
            # Concat defaults to an OUTER join that would silently NaN-fill a
            # mismatched unit (and a label rule would then never fire on the
            # NaN); assert the set invariant loudly. Ordering differences are
            # harmless, so reindex PC metrics to the display order before
            # concatenating.
            if set(frames[0].index) != set(frames[1].index):
                raise ValueError(
                    "voltage and PC metric frames have mismatched unit ids "
                    f"(voltage={sorted(frames[0].index)}, "
                    f"pc={sorted(frames[1].index)}); both derive from the same "
                    "canonical sorting, so this indicates an analyzer build "
                    "divergence and the metrics cannot be safely merged."
                )
            frames[1] = frames[1].reindex(frames[0].index)
            metrics_df = pd.concat(frames, axis=1)
        else:
            metrics_df = frames[0]

        if (
            "isi_violation" in metric_names
            and "isi_violations_count" in metrics_df.columns
        ):
            counts = metrics_df["isi_violations_count"].to_numpy()
            n_by_unit = display_analyzer.sorting.count_num_spikes_per_unit()
            n_spikes = np.array(
                [n_by_unit[int(u)] for u in metrics_df.index], dtype=float
            )
            metrics_df["isi_violation"] = isi_violation_fraction(
                counts, n_spikes
            )

        # Surfacing the configured shape columns must not depend on a voltage
        # metric having grown the display extensions: a PC-only row (e.g.
        # metric_names=["nn_advanced"], skip_pc_metrics=False) skips the voltage
        # branch above, so ensure template_metrics on the display analyzer
        # whenever shape columns are requested -- otherwise they would be
        # silently dropped rather than surfaced.
        if template_metric_columns and not display_analyzer.has_extension(
            "template_metrics"
        ):
            ensure_extensions(
                display_analyzer, ["template_metrics"], job_kwargs=job_kwargs
            )

        metrics_df = CurationEvaluation._surface_template_columns(
            metrics_df, display_analyzer, template_metric_columns
        )
        return metrics_df

    @staticmethod
    def _surface_template_columns(
        metrics_df, display_analyzer, template_metric_columns
    ):
        """Join configured waveform-shape columns onto the metric frame.

        Reads the already-computed ``template_metrics`` extension from the
        DISPLAY (unwhitened) analyzer per the display-vs-metric routing
        contract, selects the configured output COLUMNS directly (no
        name->column mapping), and joins them by unit id so a unit missing from
        either frame yields ``NaN`` rather than a misaligned row. Surfaces, does
        not threshold. A no-op when no columns are configured or the extension
        is absent (e.g. a PC-only row that never grew the display extensions).
        """
        if not template_metric_columns:
            return metrics_df
        if not display_analyzer.has_extension("template_metrics"):
            return metrics_df
        tm_df = display_analyzer.get_extension("template_metrics").get_data()
        # Select the configured columns directly (config holds column names, so
        # no name->column mapping), never shadowing a same-named quality-metric
        # column with a template one.
        present = [
            c
            for c in template_metric_columns
            if c in tm_df.columns and c not in metrics_df.columns
        ]
        missing = [c for c in template_metric_columns if c not in tm_df.columns]
        if missing:
            # Validation guarantees the configured columns are real
            # single-channel output columns, so this only fires if SI's default
            # template_metrics compute omits a validated column -- an
            # upstream-version drift signal, surfaced loudly, not silent.
            logger.warning(
                "template_metric_columns %s absent from computed "
                "template_metrics columns %s; surfacing %s only.",
                missing,
                list(tm_df.columns),
                present,
            )
        # Copy the selected columns before retyping the index so the analyzer's
        # cached template_metrics frame is never mutated in place.
        selected = tm_df[present].copy()
        selected.index = selected.index.astype(int)
        return metrics_df.join(selected)

    @staticmethod
    def _compute_merge_groups(
        analyzer, auto_merge_preset, auto_merge_kwargs, job_kwargs=None
    ):
        """Return proposed merge groups for a preset (``[]`` for 'none')."""
        if auto_merge_preset == "none":
            return []
        from spyglass.spikesorting.v2._sorting_analyzer import (
            ensure_extensions,
        )
        from spikeinterface.curation import compute_merge_unit_groups

        extensions = list(_CURATION_EXTENSIONS)
        extensions.extend(
            _AUTO_MERGE_EXTRA_EXTENSIONS.get(auto_merge_preset, ())
        )
        ensure_extensions(analyzer, extensions, job_kwargs=job_kwargs)
        merge_job_kwargs = {
            key: value
            for key, value in (job_kwargs or {}).items()
            if key != "random_seed"
        }
        compute_kwargs = dict(auto_merge_kwargs or {})
        compute_kwargs.update(merge_job_kwargs)
        groups = compute_merge_unit_groups(
            analyzer,
            preset=auto_merge_preset,
            compute_needed_extensions=False,
            **compute_kwargs,
        )
        return [[int(u) for u in group] for group in groups]

    @staticmethod
    def _write_empty(abs_path, *, provenance_tables=None):
        import pandas as pd

        return write_analyzer_curation_tables(
            abs_path,
            metrics_df=pd.DataFrame(),
            merge_groups=[],
            labels_by_unit={},
            unit_ids=[],
            provenance_tables=provenance_tables,
        )

    @staticmethod
    def _cleanup_staged_file(analysis_file_name) -> None:
        """Best-effort removal of a staged analysis file on failure."""
        from pathlib import Path

        try:
            abs_path = AnalysisNwbfile.get_abs_path(analysis_file_name)
            Path(abs_path).unlink(missing_ok=True)
        except Exception as err:  # noqa: BLE001 - cleanup must not mask cause
            logger.error(
                f"CurationEvaluation: failed to clean staged file "
                f"{analysis_file_name}: {err}"
            )

    # ---- visualization (notebook-facing) ---------------------------------

    def get_waveforms(self, key, fetch_all: bool = False):
        """Return a waveform accessor over the sort's SortingAnalyzer.

        The returned object exposes SI's ``get_waveforms_one_unit(unit_id)``
        and a v1-style ``get_waveforms(unit_id)`` over the analyzer's
        ``waveforms`` extension, replacing v1 ``MetricCuration.get_waveforms``.
        ``fetch_all`` is accepted for v1 signature parity; the sort-time
        waveform subsample is returned (a full re-extract is out of scope).
        """
        if fetch_all:
            logger.warning(
                "CurationEvaluation.get_waveforms(fetch_all=True): returning "
                "the sort-time waveform subsample (full re-extraction is not "
                "supported)."
            )
        sel = (CurationEvaluationSelection & key).fetch1()
        _assert_curation_in_raw_namespace(
            sel["sorting_id"],
            int(sel["curation_id"]),
            context="CurationEvaluation.get_waveforms",
        )
        analyzer = Sorting().get_analyzer({"sorting_id": sel["sorting_id"]})
        return _WaveformsAccessor(analyzer)

    def _display_analyzer_key(self, key):
        """Resolve + namespace-validate the DISPLAY-analyzer key for a plot.

        Returns the ``{"sorting_id": ...}`` the sort's DISPLAY (unwhitened)
        analyzer loads from. This is the RAW sort's display analyzer, so it is
        REJECTED for a merged curation (or a label-only child of one) whose unit
        namespace differs from the raw sort -- those would mix namespaces. The
        routed ``get_metrics()`` / ``get_suggested_merge_groups()`` accessors
        carry the curation's own namespace. Shared by the read-only
        ``_analyzer_for`` and the write-locking ``_locked_display_analyzer``.
        """
        sel = (CurationEvaluationSelection & key).fetch1()
        _assert_curation_in_raw_namespace(
            sel["sorting_id"],
            int(sel["curation_id"]),
            context="CurationEvaluation analyzer-backed plot",
        )
        return {"sorting_id": sel["sorting_id"]}

    def _analyzer_for(self, key):
        """Return the sort's DISPLAY (unwhitened) analyzer, read-only.

        All burst-pair legs, peak amplitudes, and the notebook plots load
        through here, so they all read real waveforms / amplitudes / positions
        -- never the whitened metric analyzer (which is built only for the PC/NN
        cluster-separation metrics). ``get_analyzer`` with no recipe name
        resolves the sort's stored display recipe.

        The returned analyzer MUST NOT be mutated: it is loaded WITHOUT holding
        ``analyzer_cache_lock``, so any accessor that computes-and-persists an
        extension must go through ``_locked_display_analyzer`` instead, or it
        could write the shared canonical zarr store concurrently with a locked
        populate / rebuild / recompute writer and corrupt it.
        """
        return Sorting().get_analyzer(self._display_analyzer_key(key))

    @contextmanager
    def _locked_display_analyzer(self, key):
        """Yield the DISPLAY analyzer while holding ``analyzer_cache_lock``.

        The plot accessors that compute-and-persist extensions into the
        canonical display analyzer (``plot_units_qc``,
        ``plot_burst_pair_metrics``) must hold the per-sort cache lock across the
        load + mutate -- the same lock the populate / rebuild / recompute writers
        take -- so two processes never write the shared zarr store at once.
        """
        from spyglass.spikesorting.v2._analyzer_cache import (
            analyzer_cache_lock,
        )

        sorting_key = self._display_analyzer_key(key)
        with analyzer_cache_lock(sorting_key["sorting_id"]):
            yield Sorting().get_analyzer(sorting_key)

    def plot_units_qc(
        self, key, *, metric_names=None, color_metric: str = "snr", axes=None
    ):
        """Static population QC overview: metric histograms + depth scatter.

        The at-a-glance "do these units look reasonable as a population?"
        view (complement to the per-unit ``describe_units`` table). Renders one
        histogram per quality metric (NaN values dropped) and a scatter placing
        each unit at its estimated probe position colored by ``color_metric``.
        A zero-unit sort returns an empty, labeled axes rather than raising.
        Pass ``axes`` to draw into a notebook/dashboard layout; otherwise a
        figure is created.

        Returns
        -------
        dict[str, matplotlib.axes.Axes]
            Axes keyed by metric name plus ``"scatter"``. A zero-unit sort
            returns ``{"empty": ax}``.
        """
        from spyglass.spikesorting.v2._metric_curation_plots import (
            plot_units_qc_figure,
        )

        from spyglass.spikesorting.v2._sorting_analyzer import (
            ensure_extensions,
        )

        metrics = self.get_metrics(key)
        try:
            with self._locked_display_analyzer(key) as analyzer:
                ensure_extensions(analyzer, ["unit_locations"])
                locations = analyzer.get_extension("unit_locations").get_data()
                unit_ids = list(analyzer.unit_ids)
        except ZeroUnitAnalyzerError:
            locations, unit_ids = None, []
        return plot_units_qc_figure(
            metrics,
            locations,
            unit_ids,
            metric_names=metric_names,
            color_metric=color_metric,
            axes=axes,
        )

    def get_correlograms(self, key, *, window_ms=100.0, bin_ms=5.0):
        """Return ``(ccgs, bins, unit_ids)`` from the correlograms extension."""
        from spyglass.spikesorting.v2._metric_curation_plots import (
            correlograms_from_analyzer,
        )

        return correlograms_from_analyzer(
            self._analyzer_for(key), window_ms=window_ms, bin_ms=bin_ms
        )

    def plot_correlograms(
        self, key, *, unit_ids=None, window_ms=100.0, bin_ms=5.0
    ):
        """Plot autocorrelograms (one panel per unit). Ported BurstPair view."""
        from spyglass.spikesorting.v2._metric_curation_plots import (
            plot_autocorrelograms_figure,
        )

        ccgs, bins, ids = self.get_correlograms(
            key, window_ms=window_ms, bin_ms=bin_ms
        )
        return plot_autocorrelograms_figure(ccgs, bins, ids, unit_ids=unit_ids)

    def investigate_pair_xcorrel(
        self, key, pairs, *, window_ms=100.0, bin_ms=5.0
    ):
        """Plot cross-correlograms for unit pairs (ported BurstPair view)."""
        from spyglass.spikesorting.v2._metric_curation_plots import (
            plot_pair_correlograms_figure,
            validate_unit_pairs,
        )

        ccgs, bins, ids = self.get_correlograms(
            key, window_ms=window_ms, bin_ms=bin_ms
        )
        used = validate_unit_pairs(ids, pairs)
        return plot_pair_correlograms_figure(ccgs, bins, ids, used)

    def investigate_pair_peaks(self, key, pairs):
        """Plot per-channel peak-amplitude histograms for unit pairs."""
        from spyglass.spikesorting.utils_burst import plot_burst_pair_peaks
        from spyglass.spikesorting.v2._metric_curation_plots import (
            peak_amplitudes_from_analyzer,
            validate_unit_pairs,
        )

        analyzer = self._analyzer_for(key)
        used = validate_unit_pairs(list(analyzer.unit_ids), pairs)
        peak_amps, _ = peak_amplitudes_from_analyzer(analyzer)
        return plot_burst_pair_peaks(used, peak_amps)

    def plot_peak_over_time(self, key, pairs, overlap: bool = True):
        """Plot peak amplitude over time for unit pairs (ported BurstPair view)."""
        from spyglass.spikesorting.utils_burst import plot_burst_peak_over_time
        from spyglass.spikesorting.v2._metric_curation_plots import (
            peak_amplitudes_from_analyzer,
            validate_unit_pairs,
        )

        analyzer = self._analyzer_for(key)
        used = validate_unit_pairs(list(analyzer.unit_ids), pairs)
        peak_amps, peak_times = peak_amplitudes_from_analyzer(analyzer)
        return plot_burst_peak_over_time(
            peak_amps, peak_times, used, overlap=overlap
        )

    def get_peak_amps(self, key):
        """Return ``(peak_amps, peak_times)`` per unit (v1 BurstPair analog).

        ``peak_amps[unit_id]`` is ``(n_spikes, n_channels)`` sampled at the
        waveform peak; ``peak_times[unit_id]`` is the seconds-timestamps of
        those SAME sampled spikes (the ``waveforms`` extension's
        ``random_spikes`` subset, not the full train), so the two arrays stay
        aligned. Reads the sort's SortingAnalyzer ``waveforms`` extension.
        """
        from spyglass.spikesorting.v2._metric_curation_plots import (
            peak_amplitudes_from_analyzer,
        )

        return peak_amplitudes_from_analyzer(self._analyzer_for(key))

    def plot_burst_pair_metrics(self, key, pairs=None):
        """Per-pair burst-metrics scatter for the sort (v1 BurstPair analog).

        Scatters waveform similarity vs cross-correlogram asymmetry, one point
        per unit pair, computed on the fly from the analyzer's extensions (no
        pair metrics are stored, though the required display extensions are
        computed into the analyzer under the cache lock). v1 laid out one panel
        per sort group; a v2 sort is a single sort group, so this renders that
        sort's pairs. ``pairs`` defaults to all ordered pairs.
        """
        from spyglass.spikesorting.utils_burst import plot_burst_metrics
        from spyglass.spikesorting.v2._metric_curation_plots import (
            burst_pair_metrics_from_analyzer,
        )

        with self._locked_display_analyzer(key) as analyzer:
            rows = burst_pair_metrics_from_analyzer(analyzer, pairs=pairs)
        return plot_burst_metrics(rows)

    # ---- SI metric / merge delegates (see v2.visualization facade) --------

    def plot_metrics(self, key, *, backend="matplotlib", **kwargs):
        """Delegate to ``visualization.plot_metrics`` for this curation.

        A local-discoverability one-liner over the Spyglass-routed
        ``get_metrics()`` table; the plotting lives in the ``v2.visualization``
        facade, which the notebook/docs teach as the primary surface.
        """
        from spyglass.spikesorting.v2 import visualization

        return visualization.plot_metrics(key, backend=backend, **kwargs)

    def plot_si_quality_metrics(
        self, key, *, compute_missing=False, backend="matplotlib", **kwargs
    ):
        """Delegate to ``visualization.plot_si_quality_metrics`` (raw SI view)."""
        from spyglass.spikesorting.v2 import visualization

        return visualization.plot_si_quality_metrics(
            key, compute_missing=compute_missing, backend=backend, **kwargs
        )

    def plot_si_template_metrics(
        self, key, *, compute_missing=False, backend="matplotlib", **kwargs
    ):
        """Delegate to ``visualization.plot_si_template_metrics`` (raw SI view)."""
        from spyglass.spikesorting.v2 import visualization

        return visualization.plot_si_template_metrics(
            key, compute_missing=compute_missing, backend=backend, **kwargs
        )

    def plot_suggested_merges(self, key, *, backend="ipywidgets", **kwargs):
        """Delegate to ``visualization.plot_suggested_merges`` for this curation.

        Defaults to the ``ipywidgets`` backend (SI's ``PotentialMergesWidget``
        has no matplotlib backend); see the facade.
        """
        from spyglass.spikesorting.v2 import visualization

        return visualization.plot_suggested_merges(
            key, backend=backend, **kwargs
        )

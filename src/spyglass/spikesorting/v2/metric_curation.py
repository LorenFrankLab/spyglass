"""Analyzer-driven quality-metric curation for spike sorting v2.

Replaces v1's ``MetricCuration`` + ``BurstPair`` with a single
``AnalyzerCuration`` computed table that walks a sort's ``SortingAnalyzer``
extensions to compute quality metrics, suggest merges, and propose
auto-curation labels. The proposed labels/merges are written to NWB; turning
them into a new ``CurationV2`` row is an explicit user action
(``materialize_curation``).

Tables
------
``QualityMetricParameters``
    Which SpikeInterface metrics to compute and their per-metric kwargs.
``AutoCurationRules`` (+ ``Rule`` part)
    An auto-merge preset plus ordered threshold rules that label units.
``AnalyzerCurationSelection``
    Pairs a ``CurationV2`` row with a metric-params and an auto-rules row.
``AnalyzerCuration``
    Computes metrics / merge suggestions / proposed labels and stores them in
    three NWB scratch tables.
"""

from __future__ import annotations

import uuid
from typing import NamedTuple

import datajoint as dj

from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.spikesorting.v2._metric_curation import (
    apply_label_rules,
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
    Sorting,
    SortingSelection,
)
from spyglass.spikesorting.v2.utils import (
    ImmutableParamsLookup,
    SelectionMasterInsertGuard,
    _jsonable_blob,
    _resolved_job_kwargs,
    _validate_params,
)
from spyglass.utils import SpyglassMixin, SpyglassMixinPart, logger

schema = dj.schema("spikesorting_v2_metric_curation")

# Extensions AnalyzerCuration adds to the sort-time analyzer before metrics and
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


class AnalyzerCurationFetched(NamedTuple):
    """DB inputs for ``AnalyzerCuration.make_compute`` (no SI/NWB I/O)."""

    sorting_id: str
    curation_id: int
    metric_waveform_params_name: str
    nwb_file_name: str
    metric_names: list[str]
    metric_kwargs: dict[str, dict]
    template_metric_columns: list[str]
    skip_pc_metrics: bool
    auto_merge_preset: str
    auto_merge_kwargs: dict
    rule_rows: list[dict]
    job_kwargs: dict


class AnalyzerCurationComputed(NamedTuple):
    """Compute -> insert carrier (DeepHash-stable strings only)."""

    analysis_file_name: str
    metrics_object_id: str
    merge_suggestions_object_id: str
    proposed_labels_object_id: str
    nwb_file_name: str


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

    def insert(self, rows, **kwargs):
        """Validate every row's params blob before a single bulk insert."""
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
        super().insert(validated, **kwargs)

    @classmethod
    def insert_default(cls):
        """Insert the default quality-metric parameter rows (idempotent)."""
        cls().insert(cls._default_rows(), skip_duplicates=True)

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
            ({"auto_curation_rules_name": "none", "auto_merge_preset": "none"}, []),
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
                        {k: rule[k] for k in AutoCurationRuleSchema.model_fields}
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


@schema
class AnalyzerCurationSelection(
    SelectionMasterInsertGuard, SpyglassMixin, dj.Manual
):
    """A CurationV2 row paired with metric and auto-curation parameters.

    Like the other deterministic-id selection masters, a raw ``insert`` /
    ``insert1`` is blocked (the PK is content-addressed from the logical
    identity); use ``insert_selection``, which passes
    ``allow_direct_insert=True`` for its already-validated insert.
    """

    definition = """
    analyzer_curation_id: uuid
    ---
    -> CurationV2
    -> QualityMetricParameters
    -> AutoCurationRules
    -> AnalyzerWaveformParameters.proj(metric_waveform_params_name="waveform_params_name")
    """

    @classmethod
    def insert_selection(cls, key: dict) -> dict:
        """Insert or find an analyzer-curation selection; return PK-only dict.

        The ``analyzer_curation_id`` PK is content-addressed (a ``uuid5`` of
        the logical identity ``(sorting_id, curation_id, metric_params_name,
        auto_curation_rules_name, metric_waveform_params_name)``) via the shared
        deterministic-selection contract, so the same logical selection always
        maps to one id. The PK uniqueness constraint -- not a check-then-insert
        dedup -- is the concurrency guard: two concurrent callers compute the
        same id, the DB accepts one, and the loser refetches it.

        ``metric_waveform_params_name`` (the whitened analyzer recipe the PC/NN
        metrics compute on) defaults to the sort's region metric row, resolved
        from the sort source's preprocessing recipe -- the same source-aware
        resolution the display recipe uses: single-recording sorts read
        ``RecordingSelection.preprocessing_params_name``, concat-backed sorts
        read the ``ConcatenatedRecordingSelection -> PreprocessingParameters``
        FK. Pass it explicitly in ``key`` to override.

        Raises ``DuplicateSelectionError`` if any existing row for this
        identity carries a non-deterministic id (a raw ``dj.insert`` bypass or
        a legacy row). Warns (does not raise) when the upstream ``CurationV2``
        was itself produced by auto-curation -- running metrics over post-merge
        templates is usually not intended, but is occasionally deliberate.
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
            # concat), the same way the display recipe is resolved. ``[1]`` is
            # the metric element of the (display, metric) pair.
            preproc = (
                SortingSelection.resolve_source_preprocessing_params_name(
                    {"sorting_id": key["sorting_id"]}
                )
            )
            metric_waveform_params_name = waveform_params_for_preprocessing(
                preproc
            )[1]

        # Reject a display/unwhitened recipe (e.g. a bad explicit override)
        # before folding it into the identity; the shipped default resolves a
        # metric row, so this only fires on a bad explicit value.
        _assert_is_metric_recipe(metric_waveform_params_name)

        identity = {
            "sorting_id": key["sorting_id"],
            "curation_id": key["curation_id"],
            "metric_params_name": key["metric_params_name"],
            "auto_curation_rules_name": key["auto_curation_rules_name"],
            "metric_waveform_params_name": metric_waveform_params_name,
        }
        analyzer_curation_id = deterministic_id("analyzer_curation", identity)
        assert_supplied_id_matches(
            key.get("analyzer_curation_id"),
            analyzer_curation_id,
            field="analyzer_curation_id",
        )

        existing = cls._find_existing_pk(identity, analyzer_curation_id)
        if existing is not None:
            return existing

        upstream_source = (
            CurationV2
            & {"sorting_id": key["sorting_id"], "curation_id": key["curation_id"]}
        ).fetch1("curation_source")
        if upstream_source == "analyzer_curation":
            logger.warning(
                "AnalyzerCuration is being inserted on a CurationV2 row with "
                "curation_source='analyzer_curation' (already auto-curated). "
                "Metrics will be computed over post-merge templates, which is "
                "usually not intended."
            )

        new_key = {**identity, "analyzer_curation_id": analyzer_curation_id}
        try:
            cls.insert1(new_key, allow_direct_insert=True)
        except Exception as exc:  # noqa: BLE001 - re-raised unless dup-PK race
            if not _is_duplicate_key_error(exc):
                raise
            existing = cls._find_existing_pk(identity, analyzer_curation_id)
            if existing is None:
                raise
            return existing
        return {"analyzer_curation_id": analyzer_curation_id}

    @classmethod
    def insert_by_curation_id(
        cls,
        sorting_id,
        curation_id,
        metric_params_name,
        auto_curation_rules_name,
    ) -> dict:
        """Insert an analyzer-curation selection for a curation (v1 analog).

        Convenience matching v1 ``BurstPairSelection.insert_by_curation_id``:
        assemble the content-addressed selection key from a curation
        (``sorting_id`` + ``curation_id``) and the named quality-metric /
        auto-rule rows, then delegate to :meth:`insert_selection` (the only
        sanctioned insert path). Returns the PK-only dict.
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
    def pc_requesting(cls):
        """Selections whose QualityMetricParameters request PC/NN metrics.

        Joined to ``QualityMetricParameters`` and restricted to
        ``skip_pc_metrics=0`` -- the EXACT set that actually builds a whitened
        metric analyzer (the schema validates skip_pc_metrics=False ⇔ a PCA
        metric is requested). The single source of truth for which metric
        recipes are "in use", so the recompute key_source and the orphan-folder
        audit cannot drift apart. Carries ``sorting_id`` (a secondary CurationV2
        FK attr) and ``metric_waveform_params_name`` for the caller to project /
        fetch.
        """
        return cls * QualityMetricParameters & "skip_pc_metrics = 0"

    @classmethod
    def _find_existing_pk(cls, identity, deterministic_id):
        """Return the PK-only dict for ``identity`` or None; guard bad ids."""
        from spyglass.spikesorting.v2.exceptions import (
            DuplicateSelectionError,
        )

        existing_ids = (cls & identity).fetch("analyzer_curation_id")
        bypassed = [
            aid
            for aid in existing_ids
            if uuid.UUID(str(aid)) != deterministic_id
        ]
        if bypassed:
            raise DuplicateSelectionError(
                "AnalyzerCurationSelection has duplicate selection rows for "
                f"{identity} with non-deterministic id(s) {sorted(map(str, bypassed))} "
                f"(expected the content-addressed {deterministic_id}). This is "
                "an integrity bug -- a row was inserted bypassing "
                "insert_selection."
            )
        if len(existing_ids):
            return {"analyzer_curation_id": deterministic_id}
        return None


@schema
class AnalyzerCuration(SpyglassMixin, dj.Computed):
    """Compute quality metrics, merge suggestions, and proposed labels.

    Walks the sort's ``SortingAnalyzer`` extensions, computes the requested
    quality metrics (replicating Spyglass's ``isi_violation`` fraction),
    proposes merges via the configured auto-merge preset, and applies the
    threshold rules to propose labels. Outputs are written to three NWB
    scratch tables; ``materialize_curation`` turns them into a child
    ``CurationV2`` row.
    """

    definition = """
    -> AnalyzerCurationSelection
    ---
    -> AnalysisNwbfile
    metrics_object_id: varchar(72)
    merge_suggestions_object_id: varchar(72)
    proposed_labels_object_id: varchar(72)
    """

    # Tri-part make so the metric/merge compute (and NWB write) run OUTSIDE the
    # DB transaction, mirroring Recording / Sorting.
    _parallel_make = True

    def make_fetch(self, key) -> AnalyzerCurationFetched:
        """Fetch the DB inputs (params + lineage); no SI/NWB I/O."""
        sel = (AnalyzerCurationSelection & key).fetch1()
        # Re-assert the stored metric recipe is a whitened metric row, even
        # though insert_selection already validated it: a row planted via
        # allow_direct_insert bypasses that guard, and building an unwhitened
        # metric analyzer here would silently compute PC/NN metrics in the wrong
        # space rather than fail.
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
        sorting_key = {"sorting_id": sel["sorting_id"]}
        return AnalyzerCurationFetched(
            sorting_id=str(sel["sorting_id"]),
            curation_id=int(sel["curation_id"]),
            metric_waveform_params_name=sel["metric_waveform_params_name"],
            nwb_file_name=_nwb_file_name_for_sorting(sorting_key),
            metric_names=list(qm["metric_names"]),
            metric_kwargs=dict(qm["metric_kwargs"] or {}),
            template_metric_columns=list(qm["template_metric_columns"] or []),
            skip_pc_metrics=bool(qm["skip_pc_metrics"]),
            auto_merge_preset=acr["auto_merge_preset"],
            auto_merge_kwargs=dict(acr["auto_merge_kwargs"] or {}),
            rule_rows=list(rule_rows),
            job_kwargs=_resolved_job_kwargs(
                qm["job_kwargs"], acr["job_kwargs"]
            ),
        )

    def make_compute(
        self,
        key,
        sorting_id,
        curation_id,
        metric_waveform_params_name,
        nwb_file_name,
        metric_names,
        metric_kwargs,
        template_metric_columns,
        skip_pc_metrics,
        auto_merge_preset,
        auto_merge_kwargs,
        rule_rows,
        job_kwargs,
    ) -> AnalyzerCurationComputed:
        """Compute metrics / merges / labels and write the NWB tables.

        Routing (see the display-vs-metric contract): the unwhitened DISPLAY
        analyzer carries voltage / spike-train metrics, amplitudes, and the
        merge engine; the whitened METRIC analyzer carries ONLY the PC/NN
        cluster-separation metrics. The whitened analyzer is built on demand
        (cache miss) and only when PC metrics are actually requested.
        """
        analysis_file_name = AnalysisNwbfile().create(nwb_file_name)
        abs_path = AnalysisNwbfile.get_abs_path(analysis_file_name)
        sorting_key = {"sorting_id": sorting_id}
        try:
            try:
                display_analyzer = Sorting().get_analyzer(sorting_key)
            except ZeroUnitAnalyzerError:
                logger.warning(
                    "AnalyzerCuration: zero-unit sort "
                    f"{sorting_id}; writing empty metric/merge/label tables."
                )
                object_ids = self._write_empty(abs_path)
                return AnalyzerCurationComputed(
                    analysis_file_name, *object_ids, nwb_file_name
                )

            # The whitened metric analyzer is only needed for PC/NN metrics;
            # building a second analyzer is expensive, so skip it when none are
            # requested. (Display get_analyzer above already raised for a
            # zero-unit sort, so we never build the metric analyzer for one.)
            metric_analyzer = None
            if not skip_pc_metrics and _requested_pc_metrics(metric_names):
                metric_analyzer = Sorting().get_analyzer(
                    sorting_key,
                    waveform_params_name=metric_waveform_params_name,
                )

            metrics_df = self._compute_metrics(
                display_analyzer,
                metric_analyzer,
                metric_names,
                metric_kwargs,
                skip_pc_metrics,
                job_kwargs,
                template_metric_columns=template_metric_columns,
            )
            labels_by_unit = apply_label_rules(metrics_df, rule_rows)
            merge_groups = self._compute_merge_groups(
                display_analyzer,
                auto_merge_preset,
                auto_merge_kwargs,
                job_kwargs,
            )
            object_ids = write_analyzer_curation_tables(
                abs_path,
                metrics_df=metrics_df,
                merge_groups=merge_groups,
                labels_by_unit=labels_by_unit,
                unit_ids=[int(u) for u in metrics_df.index],
            )
            return AnalyzerCurationComputed(
                analysis_file_name, *object_ids, nwb_file_name
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
    ) -> None:
        """Register the analysis file and insert the row.

        The ``AnalysisNwbfile().add`` registration and the ``insert1`` run
        inside ``transaction_or_noop`` so they commit (or roll back) atomically.
        Under the tri-part populate path the framework transaction is already
        open, so this is a no-op; the wrapper matters for a DIRECT (out-of-
        populate) reuse of ``make_insert`` -- without it, ``add`` could commit an
        AnalysisNwbfile row that the failed ``insert1`` then orphans (mirrors
        ``Sorting`` / ``ConcatenatedRecording``). ``nwb_file_name`` is threaded
        from ``make_fetch`` via the compute carrier rather than re-walking the
        recording lineage in the transaction phase. On failure the staged
        analysis file is removed before re-raising.
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
                    }
                )
        except Exception:
            self._cleanup_staged_file(analysis_file_name)
            raise

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
            # Enforce the pinned PCA params even if a stale/manual analyzer
            # already carries principal_components computed with different ones
            # (ensure_extensions skips a present extension without checking its
            # params): drop a mismatched extension so it recomputes pinned.
            if metric_analyzer.has_extension("principal_components"):
                existing_pca = dict(
                    metric_analyzer.get_extension(
                        "principal_components"
                    ).params
                )
                if not _pca_params_match(existing_pca):
                    logger.warning(
                        "AnalyzerCuration: principal_components on the metric "
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
            )
            pc_df.index = pc_df.index.astype(int)
            frames.append(pc_df)

        if not frames:
            raise ValueError(
                "No metrics to compute: metric_names contains only PC/NN "
                "metrics but skip_pc_metrics=True. Set skip_pc_metrics=False "
                "to compute them."
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

        metrics_df = AnalyzerCuration._surface_template_columns(
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
    def _write_empty(abs_path):
        import pandas as pd

        return write_analyzer_curation_tables(
            abs_path,
            metrics_df=pd.DataFrame(),
            merge_groups=[],
            labels_by_unit={},
            unit_ids=[],
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
                f"AnalyzerCuration: failed to clean staged file "
                f"{analysis_file_name}: {err}"
            )

    # ---- fetch / promote helpers (v1 MetricCuration parity) --------------

    @classmethod
    def get_metrics(cls, key):
        """Return the quality-metrics table (DataFrame indexed by unit_id).

        The frame carries the configured SpikeInterface quality metrics plus the
        row's surfaced waveform-shape (template) columns -- ``trough_half_width``
        by default, with ``peak_to_trough_duration`` and slope columns available
        opt-in via the row's ``template_metric_columns`` -- so downstream code can
        classify cell types (rate x spike width) with its own region-appropriate
        thresholds (the pipeline ships none). Non-finite values surface as
        ``None`` (the on-disk representation is HDF5-native NaN).
        """
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
    def get_merge_groups(cls, key) -> list:
        """Return proposed merge groups as a list of unit-id lists."""
        row = (cls & key).fetch1()
        abs_path = AnalysisNwbfile.get_abs_path(row["analysis_file_name"])
        return read_merge_suggestions(
            abs_path, row["merge_suggestions_object_id"]
        )

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
                "AnalyzerCuration.get_waveforms(fetch_all=True): returning the "
                "sort-time waveform subsample (full re-extraction is not "
                "supported)."
            )
        sorting_id = (AnalyzerCurationSelection & key).fetch1("sorting_id")
        analyzer = Sorting().get_analyzer({"sorting_id": sorting_id})
        return _WaveformsAccessor(analyzer)

    def materialize_curation(
        self,
        key,
        description: str = "auto-curation",
        allow_custom_labels: bool = False,
    ) -> dict:
        """Create a child ``CurationV2`` row from the proposed labels/merges.

        The explicit v2 analog of v1 ``CurationV1.insert_metric_curation``:
        auto-curation never silently writes a curation, so the user calls this
        to commit. Re-running with the same analyzer-curation content is
        idempotent: an existing child with the same parent, labels, proposed
        merges, description, and ``curation_source='analyzer_curation'`` is
        returned instead of creating another curation row. Returns the
        ``{sorting_id, curation_id}`` key.
        """
        sel = (AnalyzerCurationSelection & key).fetch1()
        sorting_key = {"sorting_id": sel["sorting_id"]}
        labels = self.get_labels(key)
        merge_groups = [g for g in self.get_merge_groups(key) if len(g) >= 2]
        # Idempotency is the child-reuse path in CurationV2.insert_curation
        # (reuse_existing=True): a matching analyzer_curation child is returned
        # instead of staging another curated-units NWB. Don't re-implement the
        # content match here -- one source of truth for the dedup.
        return CurationV2.insert_curation(
            sorting_key,
            labels=labels or None,
            merge_groups=merge_groups or None,
            parent_curation_id=int(sel["curation_id"]),
            apply_merge=False,
            description=description,
            curation_source="analyzer_curation",
            allow_custom_labels=allow_custom_labels,
            reuse_existing=True,
        )

    # ---- visualization (notebook-facing) ---------------------------------

    def _analyzer_for(self, key):
        """Return the sort's DISPLAY (unwhitened) analyzer.

        All burst-pair legs, peak amplitudes, the merge engine, and the
        notebook plots load through here, so they all read real waveforms /
        amplitudes / positions -- never the whitened metric analyzer (which is
        built only for the PC/NN cluster-separation metrics). ``get_analyzer``
        with no recipe name resolves the sort's stored display recipe.
        """
        sorting_id = (AnalyzerCurationSelection & key).fetch1("sorting_id")
        return Sorting().get_analyzer({"sorting_id": sorting_id})

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
            analyzer = self._analyzer_for(key)
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

    def plot_by_sort_group_ids(self, key, pairs=None):
        """Per-pair burst-metrics scatter for the sort (v1 BurstPair analog).

        Scatters waveform similarity vs cross-correlogram asymmetry, one point
        per unit pair, computed on the fly from the analyzer's extensions
        (nothing is stored). v1 laid out one panel per sort group; a v2 sort is
        a single sort group, so this renders that sort's pairs. ``pairs``
        defaults to all ordered pairs.
        """
        from spyglass.spikesorting.utils_burst import plot_burst_metrics
        from spyglass.spikesorting.v2._metric_curation_plots import (
            burst_pair_metrics_from_analyzer,
        )

        rows = burst_pair_metrics_from_analyzer(
            self._analyzer_for(key), pairs=pairs
        )
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

    def plot_potential_merges(self, key, *, backend="ipywidgets", **kwargs):
        """Delegate to ``visualization.plot_potential_merges`` for this curation.

        Defaults to the ``ipywidgets`` backend (SI's ``PotentialMergesWidget``
        has no matplotlib backend); see the facade.
        """
        from spyglass.spikesorting.v2 import visualization

        return visualization.plot_potential_merges(
            key, backend=backend, **kwargs
        )


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

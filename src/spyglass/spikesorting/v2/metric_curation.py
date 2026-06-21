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
)
from spyglass.spikesorting.v2.curation import CurationV2
from spyglass.spikesorting.v2.exceptions import (
    UnsupportedDirectInsertError,
    ZeroUnitAnalyzerError,
)
from spyglass.spikesorting.v2.recording import RecordingSelection
from spyglass.spikesorting.v2.sorting import (
    Sorting,
    SortingSelection,
)
from spyglass.spikesorting.v2.utils import (
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


def _nwb_file_name_for_sorting(sorting_key: dict) -> str:
    """Return the source ``nwb_file_name`` for a recording-backed sort."""
    recording_id = (SortingSelection.RecordingSource & sorting_key).fetch1(
        "recording_id"
    )
    return (RecordingSelection & {"recording_id": recording_id}).fetch1(
        "nwb_file_name"
    )


class AnalyzerCurationFetched(NamedTuple):
    """DB inputs for ``AnalyzerCuration.make_compute`` (no SI/NWB I/O)."""

    sorting_id: str
    curation_id: int
    nwb_file_name: str
    metric_names: list
    metric_kwargs: dict
    skip_pc_metrics: bool
    auto_merge_preset: str
    auto_merge_kwargs: dict
    rule_rows: list
    job_kwargs: dict


class AnalyzerCurationComputed(NamedTuple):
    """Compute -> insert carrier (DeepHash-stable strings only)."""

    analysis_file_name: str
    metrics_object_id: str
    merge_suggestions_object_id: str
    proposed_labels_object_id: str


@schema
class QualityMetricParameters(SpyglassMixin, dj.Lookup):
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
    metric_names: blob       # list[str] of SpikeInterface metric names
    metric_kwargs: blob      # dict[str, dict] per-metric kwargs
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
        return [
            {
                "metric_params_name": "franklab_default",
                "metric_names": full_metrics,
                "metric_kwargs": {
                    "snr": {"peak_sign": "neg"},
                    "isi_violation": isi_kwargs,
                    "nn_advanced": nn_kwargs,
                },
                "skip_pc_metrics": False,
            },
            {
                "metric_params_name": "neuropixels_default",
                "metric_names": full_metrics,
                "metric_kwargs": {
                    "snr": {"peak_sign": "neg"},
                    "isi_violation": isi_kwargs,
                    "nn_advanced": nn_kwargs,
                },
                "skip_pc_metrics": False,
            },
            {
                "metric_params_name": "minimal",
                "metric_names": ["snr", "isi_violation", "firing_rate"],
                "metric_kwargs": {
                    "snr": {"peak_sign": "neg"},
                    "isi_violation": isi_kwargs,
                },
                "skip_pc_metrics": True,
            },
        ]

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
            clean = _validate_params(QualityMetricParamsSchema, payload)
            validated.append(
                {
                    "metric_params_name": row["metric_params_name"],
                    "metric_names": clean["metric_names"],
                    "metric_kwargs": clean["metric_kwargs"],
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
    def show_available_metrics(cls) -> None:
        """Log the available SpikeInterface quality-metric names."""
        for name in cls.available_quality_metrics():
            logger.info(name)


@schema
class AutoCurationRules(SpyglassMixin, dj.Lookup):
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

    class Rule(SpyglassMixinPart):
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
        Idempotent when the master name already exists (returns its PK).
        """
        name = row["auto_curation_rules_name"]
        existing = cls & {"auto_curation_rules_name": name}
        if existing:
            if not skip_duplicates:
                logger.warning(
                    f"AutoCurationRules {name!r} already exists; returning it."
                )
            return {"auto_curation_rules_name": name}

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
        inst = cls()
        with inst.connection.transaction:
            # Bypass this class's raising insert override by calling the next
            # insert in the MRO (SpyglassMixin / dj.Table); the bulk insert
            # does not route back through the overridden insert1.
            super(AutoCurationRules, inst).insert([master])
            if rule_inserts:
                cls.Rule.insert(rule_inserts)
        return {"auto_curation_rules_name": name}

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
        ]

    @classmethod
    def insert_default(cls):
        """Insert the default auto-curation rule sets (idempotent)."""
        for master, rules in cls._default_payloads():
            cls.insert_rules(master, rules, skip_duplicates=True)

    @classmethod
    def check_rule_integrity(cls, restriction=True) -> list[dict]:
        """Return master rows whose Rule rows are malformed or missing.

        Direct inserts into ``AutoCurationRules.Rule`` bypass whole-payload
        validation; this surfaces masters with no rule rows (when a preset is
        ``none``, so no auto-merge either) or rule rows that fail the rule
        schema, for use in an integrity check.
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
            if master["auto_merge_preset"] == "none" and not rules:
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
class AnalyzerCurationSelection(SpyglassMixin, dj.Manual):
    """A CurationV2 row paired with metric and auto-curation parameters."""

    definition = """
    analyzer_curation_id: uuid
    ---
    -> CurationV2
    -> QualityMetricParameters
    -> AutoCurationRules
    """

    @classmethod
    def insert_selection(cls, key: dict) -> dict:
        """Insert or find an analyzer-curation selection; return PK-only dict.

        Warns (does not raise) when the upstream ``CurationV2`` was itself
        produced by auto-curation -- running metrics over post-merge templates
        is usually not intended, but is occasionally deliberate.
        """
        identity = {
            "sorting_id": key["sorting_id"],
            "curation_id": key["curation_id"],
            "metric_params_name": key["metric_params_name"],
            "auto_curation_rules_name": key["auto_curation_rules_name"],
        }
        existing = (cls & identity).fetch("KEY", as_dict=True)
        if len(existing) == 1:
            return existing[0]
        if len(existing) > 1:  # pragma: no cover - guarded by unique identity
            from spyglass.spikesorting.v2.exceptions import (
                DuplicateSelectionError,
            )

            raise DuplicateSelectionError(
                "AnalyzerCurationSelection has duplicate selection rows for "
                f"{identity}."
            )

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

        new_key = {**identity, "analyzer_curation_id": uuid.uuid4()}
        cls.insert1(new_key)
        return {"analyzer_curation_id": new_key["analyzer_curation_id"]}


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
            nwb_file_name=_nwb_file_name_for_sorting(sorting_key),
            metric_names=list(qm["metric_names"]),
            metric_kwargs=dict(qm["metric_kwargs"] or {}),
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
        nwb_file_name,
        metric_names,
        metric_kwargs,
        skip_pc_metrics,
        auto_merge_preset,
        auto_merge_kwargs,
        rule_rows,
        job_kwargs,
    ) -> AnalyzerCurationComputed:
        """Compute metrics / merges / labels and write the NWB tables."""
        analysis_file_name = AnalysisNwbfile().create(nwb_file_name)
        abs_path = AnalysisNwbfile.get_abs_path(analysis_file_name)
        sorting_key = {"sorting_id": sorting_id}
        try:
            try:
                analyzer = Sorting().get_analyzer(sorting_key)
            except ZeroUnitAnalyzerError:
                logger.warning(
                    "AnalyzerCuration: zero-unit sort "
                    f"{sorting_id}; writing empty metric/merge/label tables."
                )
                object_ids = self._write_empty(abs_path)
                return AnalyzerCurationComputed(analysis_file_name, *object_ids)

            metrics_df = self._compute_metrics(
                analyzer,
                metric_names,
                metric_kwargs,
                skip_pc_metrics,
                job_kwargs,
            )
            labels_by_unit = apply_label_rules(metrics_df, rule_rows)
            merge_groups = self._compute_merge_groups(
                analyzer, auto_merge_preset, auto_merge_kwargs, job_kwargs
            )
            object_ids = write_analyzer_curation_tables(
                abs_path,
                metrics_df=metrics_df,
                merge_groups=merge_groups,
                labels_by_unit=labels_by_unit,
                unit_ids=[int(u) for u in metrics_df.index],
            )
            return AnalyzerCurationComputed(analysis_file_name, *object_ids)
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
    ) -> None:
        """Register the analysis file and insert the row.

        Runs inside the framework's tri-part insert transaction (DataJoint
        opens it around ``make_insert``), so no explicit transaction is opened
        here. On failure the staged analysis file is removed before re-raising.
        """
        sorting_id = (AnalyzerCurationSelection & key).fetch1("sorting_id")
        nwb_file_name = _nwb_file_name_for_sorting({"sorting_id": sorting_id})
        try:
            AnalysisNwbfile().add(nwb_file_name, analysis_file_name)
            self.insert1(
                {
                    **key,
                    "analysis_file_name": analysis_file_name,
                    "metrics_object_id": metrics_object_id,
                    "merge_suggestions_object_id": merge_suggestions_object_id,
                    "proposed_labels_object_id": proposed_labels_object_id,
                }
            )
        except Exception:
            self._cleanup_staged_file(analysis_file_name)
            raise

    # ---- compute helpers (DB-light; SI work) -----------------------------

    @staticmethod
    def _compute_metrics(
        analyzer, metric_names, metric_kwargs, skip_pc_metrics, job_kwargs=None
    ):
        """Compute extensions + quality metrics, adding Spyglass isi_violation.

        ``job_kwargs`` (resolved n_jobs / chunk_duration / progress_bar) are
        forwarded to the heavy extension computation so chronic runs honor the
        configured concurrency.
        """
        import numpy as np
        from spikeinterface.metrics.quality import compute_quality_metrics

        # ``random_seed`` is an extension param, not a ChunkRecordingExecutor
        # job kwarg (mirrors build_analyzer), so drop it before compute().
        compute_kwargs = {
            k: v for k, v in (job_kwargs or {}).items() if k != "random_seed"
        }
        extensions = list(_CURATION_EXTENSIONS)
        if not skip_pc_metrics:
            extensions.append("principal_components")
        # Add only extensions not already present: recomputing one would
        # cascade-delete its children, and re-populating the same sort under a
        # second metric-params row must be idempotent (mirrors add_extensions).
        to_add = [e for e in extensions if not analyzer.has_extension(e)]
        if to_add:
            analyzer.compute(to_add, **compute_kwargs)
        metrics_df = compute_quality_metrics(
            analyzer,
            metric_names=metric_names,
            metric_params=metric_kwargs or None,
            skip_pc_metrics=skip_pc_metrics,
        )
        metrics_df.index = metrics_df.index.astype(int)
        if (
            "isi_violation" in metric_names
            and "isi_violations_count" in metrics_df.columns
        ):
            counts = metrics_df["isi_violations_count"].to_numpy()
            n_by_unit = analyzer.sorting.count_num_spikes_per_unit()
            n_spikes = np.array(
                [n_by_unit[int(u)] for u in metrics_df.index], dtype=float
            )
            metrics_df["isi_violation"] = isi_violation_fraction(
                counts, n_spikes
            )
        return metrics_df

    @staticmethod
    def _compute_merge_groups(
        analyzer, auto_merge_preset, auto_merge_kwargs, job_kwargs=None
    ):
        """Return proposed merge groups for a preset (``[]`` for 'none')."""
        if auto_merge_preset == "none":
            return []
        from spikeinterface.curation import compute_merge_unit_groups

        groups = compute_merge_unit_groups(
            analyzer,
            preset=auto_merge_preset,
            compute_needed_extensions=False,
            job_kwargs=job_kwargs or {},
            **(auto_merge_kwargs or {}),
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

        Non-finite metric values surface as ``None`` (the on-disk
        representation is HDF5-native NaN).
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
        to commit. Returns the new ``{sorting_id, curation_id}`` key.
        """
        sel = (AnalyzerCurationSelection & key).fetch1()
        sorting_key = {"sorting_id": sel["sorting_id"]}
        labels = self.get_labels(key)
        merge_groups = [g for g in self.get_merge_groups(key) if len(g) >= 2]
        return CurationV2.insert_curation(
            sorting_key,
            labels=labels or None,
            merge_groups=merge_groups or None,
            parent_curation_id=int(sel["curation_id"]),
            apply_merge=False,
            description=description,
            curation_source="analyzer_curation",
            allow_custom_labels=allow_custom_labels,
        )

    # ---- visualization (notebook-facing) ---------------------------------

    def _analyzer_for(self, key):
        sorting_id = (AnalyzerCurationSelection & key).fetch1("sorting_id")
        return Sorting().get_analyzer({"sorting_id": sorting_id})

    def plot_units_qc(
        self, key, *, metric_names=None, color_metric: str = "snr"
    ):
        """Static population QC overview: metric histograms + depth scatter.

        The at-a-glance "do these units look reasonable as a population?"
        view (complement to the per-unit ``describe_units`` table). Renders one
        histogram per quality metric (NaN values dropped) and a scatter placing
        each unit at its estimated probe position colored by ``color_metric``.
        A zero-unit sort returns an empty, labeled figure rather than raising.

        Returns
        -------
        matplotlib.figure.Figure
        """
        from spyglass.spikesorting.v2._metric_curation_plots import (
            plot_units_qc_figure,
        )

        metrics = self.get_metrics(key)
        try:
            analyzer = self._analyzer_for(key)
            if not analyzer.has_extension("unit_locations"):
                analyzer.compute("unit_locations")
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

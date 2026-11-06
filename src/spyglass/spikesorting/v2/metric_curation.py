"""Analyzer-driven quality-metric curation for spike sorting v2.

``CurationEvaluation`` computes quality metrics, merge suggestions and
proposed labels in a committed ``CurationV2`` row's own unit namespace (a
merged unit is scored over its merged spike train). Committing them to a child
``CurationV2`` is an explicit user action (``accept_evaluation_outputs`` /
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
from spyglass.spikesorting.v2._curation import (
    evaluation_acceptance as _evaluation_acceptance,
    evaluation_analyzers as _evaluation_analyzers,
    metric_fetch as _metric_curation_fetch,
)
from spyglass.spikesorting.v2._curation.metrics import (
    _requested_pc_metrics,
    rules_payloads_match,
)
from spyglass.spikesorting.v2._storage.metrics_nwb import (
    read_analyzer_curation_tables,
    read_merge_suggestions,
    read_proposed_labels,
    read_quality_metrics,
    write_analyzer_curation_tables,
)
from spyglass.spikesorting.v2._params.metric_curation import (
    prepare_auto_curation_rules,
    prepare_quality_metric_row,
)
from spyglass.spikesorting.v2._core.recipe_catalog import (
    auto_curation_default_payloads,
    quality_metric_default_rows,
    waveform_params_for_preprocessing,
)
from spyglass.spikesorting.v2._recording.source import EffectiveTraces
from spyglass.spikesorting.v2._storage.staged_outputs import (
    StagedOutputCleanupMixin,
    StagedOutputs,
    staged_analysis_file,
)
from spyglass.spikesorting.v2._storage.units_nwb import StoredUnits
from spyglass.spikesorting.v2.curation import CurationV2
from spyglass.spikesorting.v2.exceptions import (
    UnsupportedDirectInsertError,
    ZeroUnitAnalyzerError,
)
from spyglass.spikesorting.v2.sorting import (
    AnalyzerWaveformParameters,
    SortingSelection,
)
from spyglass.spikesorting.v2._core.table_integrity import (
    ImmutableParamsLookup,
    SelectionMasterInsertGuard,
)
from spyglass.spikesorting.v2._core.lookup_validation import (
    _jsonable_blob,
    reject_duplicate_quality_metric_content,
    reject_stale_quality_metric_defaults,
)
from spyglass.utils import SpyglassMixin, SpyglassMixinPart, logger

schema = dj.schema("spikesorting_v2_metric_curation")


def _assert_is_metric_recipe(waveform_params_name: str) -> None:
    """Raise unless ``waveform_params_name`` is a whitened metric recipe.

    PC/NN cluster-separation metrics must compute in whitened space.
    Checked at ``insert_selection`` and again in ``make_fetch``, since
    ``allow_direct_insert`` bypasses the selection guard.

    Raises
    ------
    ValueError
        If the recipe is not ``purpose='metric'`` with ``whiten=True``.
    """
    from spyglass.spikesorting.v2._sorting.analyzer import fetch_waveform_params

    recipe = fetch_waveform_params(waveform_params_name)
    if not recipe.get("whiten") or recipe.get("purpose") != "metric":
        raise ValueError(
            f"metric_waveform_params_name={waveform_params_name!r} is not a "
            "whitened metric recipe (purpose='metric', whiten=True). PC/NN "
            "cluster-separation metrics must compute on a whitened analyzer; "
            "pass a metric recipe (e.g. franklab_cortex_metric_waveforms) or "
            "omit it to use the sort's resolved region metric row."
        )


class EvaluationRecordingInputs(NamedTuple):
    """Recording reconstruction and artifact-mask inputs."""

    nwb_file_name: str
    source_kind: str
    recording_id: str | None
    # From the sort's lineage, not the (possibly motion-corrected) traces row.
    concat_recording_id: str | None
    artifact_detection_id: str | None
    artifact_valid_times: object  # np.ndarray | None (DeepHashed, not ==)
    # The sort's effective traces; make_fetch rebuilds a missing file.
    traces: EffectiveTraces
    traces_abs_path: str
    # (start, end) frames; analyzers estimate noise and whitening from them.
    statistics_spans: tuple[tuple[int, int], ...]


class EvaluationSortingInputs(NamedTuple):
    """Committed unit namespace and raw/curated spike-train inputs."""

    sorting_id: str
    curation_id: int
    raw_units: StoredUnits
    raw_n_units: int
    curated_units: StoredUnits
    expected_unit_ids: list[int]
    use_fast_path: bool


class EvaluationAnalyzerInputs(NamedTuple):
    """Resolved display/metric analyzer recipes and execution settings."""

    display_waveform_params_name: str
    display_waveform_params: dict
    display_analyzer_folder: str
    metric_waveform_params_name: str
    metric_waveform_params: dict
    metric_analyzer_folder: str
    sorter_row: dict
    analyzer_job_kwargs: dict
    # Merged curation's display analyzer cache slot; None when the sort's
    # cached analyzers are used.
    merged_display_cache: dict | None = None


class EvaluationMetricInputs(NamedTuple):
    """Metric computation, merge suggestions, and automatic label rules."""

    metric_params_name: str
    auto_curation_rules_name: str
    metric_names: list[str]
    metric_kwargs: dict[str, dict]
    template_metric_columns: list[str]
    skip_pc_metrics: bool
    auto_merge_preset: str
    auto_merge_kwargs: dict
    rule_rows: list[dict]
    metric_job_kwargs: dict
    peak_sign: str
    # Matches the display analyzer's noise_levels samples.
    noise_seed: int
    observed_presence_bin_duration_s: float = 60.0
    # nn metrics' neighbor-search threads; None uses resolve_search_threads().
    search_threads: int | None = None


class CurationEvaluationFetched(NamedTuple):
    """Resolved inputs, unpacked into ``make_compute``."""

    recording_inputs: EvaluationRecordingInputs
    sorting_inputs: EvaluationSortingInputs
    analyzer_inputs: EvaluationAnalyzerInputs
    metric_inputs: EvaluationMetricInputs


class CurationEvaluationComputed(NamedTuple):
    """Compute -> insert carrier (DeepHash-stable strings only)."""

    analysis_file_name: str
    metrics_object_id: str
    merge_suggestions_object_id: str
    proposed_labels_object_id: str
    nwb_file_name: str
    # Provenance only. Analyzer hashes are None on the merged-curation path.
    spikeinterface_version: str
    source_analyzer_hashes: dict | None

    def staged_outputs(self) -> StagedOutputs:
        """The staged analysis file ``make_insert`` registers."""
        return StagedOutputs(analysis_file_names=(self.analysis_file_name,))


@schema
class QualityMetricParameters(ImmutableParamsLookup, SpyglassMixin, dj.Lookup):
    """Which quality metrics to compute and their per-metric kwargs.

    ``metric_names`` is validated against SpikeInterface's metric list;
    ``metric_kwargs`` is passed as ``compute_quality_metrics(metric_params=)``.
    A row feeding an ``nn_advanced`` rule must set ``skip_pc_metrics=False``.
    """

    definition = """
    metric_params_name: varchar(64)
    ---
    metric_names: blob              # list[str] of SpikeInterface metric names
    metric_kwargs: blob             # dict[str, dict] per-metric kwargs
    template_metric_columns: blob   # list[str] of SI template output columns
    skip_pc_metrics=1: bool
    observed_presence_bin_duration_s=60: double
    params_schema_version=2: int
    job_kwargs=null: blob
    """

    @classmethod
    def get_isi_threshold_ms(cls, metric_params_name: str) -> float:
        """Resolve the recipe's refractory window, including SI's default."""
        from spikeinterface.metrics.quality import (
            get_default_quality_metrics_params,
        )

        kwargs = (cls & {"metric_params_name": metric_params_name}).fetch1(
            "metric_kwargs"
        )
        defaults = get_default_quality_metrics_params()["isi_violation"]
        params = {**defaults, **((kwargs or {}).get("isi_violation") or {})}
        return float(params["isi_threshold_ms"])

    @classmethod
    def _default_rows(cls) -> list[dict]:
        """The shipped rows (``_core.recipe_catalog.quality_metric_default_rows``)."""
        return quality_metric_default_rows()

    def insert(self, rows, *, allow_duplicate_params=False, **kwargs):
        """Validate every row's params, reject duplicate content, then insert.

        Raises
        ------
        DuplicateParameterContentError
            If a row duplicates an existing row's content under a new name,
            unless ``allow_duplicate_params=True``.
        """
        if isinstance(rows, dict):
            rows = [rows]
        validated = [prepare_quality_metric_row(row) for row in rows]
        reject_duplicate_quality_metric_content(
            self.fetch(as_dict=True),
            validated,
            allow_duplicate_params=allow_duplicate_params,
        )
        super().insert(validated, **kwargs)

    @classmethod
    def insert_default(cls):
        """Insert the default quality-metric parameter rows (idempotent).

        Raises
        ------
        DuplicateParameterContentError
            If a stored row with a shipped name holds different content.
        """
        # franklab_default and neuropixels_default intentionally share content.
        default_rows = [
            prepare_quality_metric_row(row) for row in cls._default_rows()
        ]
        reject_stale_quality_metric_defaults(
            cls().fetch(as_dict=True), default_rows
        )
        cls().insert(
            default_rows,
            skip_duplicates=True,
            allow_duplicate_params=True,
        )


@schema
class AutoCurationRules(ImmutableParamsLookup, SpyglassMixin, dj.Lookup):
    """Auto-merge preset + ordered threshold rules that label units.

    Insert only through ``insert_rules(row, rule_rows)``, which validates
    the master and its ordered ``Rule`` rows together.

    Each rule's ``missing_policy`` covers units SpikeInterface cannot assess
    for that metric (e.g. fewer spikes than ``nn_advanced``'s ``min_spikes``;
    see ``expected_missing_units`` in :mod:`._curation.metrics`):

    - ``error``: raise.
    - ``fail``: apply the rule label.
    - ``pass``: leave the unit unlabelled by that rule.

    Any other NaN raises regardless of policy, including any NaN in a column
    with no registered conditions (template, custom, ``observed_*``).
    ``pass`` and ``fail`` warn when a metric is missing for every unit. A
    SpikeInterface error in a referenced metric aborts the evaluation; one in
    an unreferenced metric is logged. This is distinct from SpikeInterface's
    ``nan_policy``.
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
        threshold: double
        label: varchar(32)
        missing_policy='error': enum('error', 'fail', 'pass')
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

        Inserts both in one transaction. Idempotent for an identical payload.

        Returns
        -------
        dict
            ``{"auto_curation_rules_name": ...}``.

        Raises
        ------
        ValueError
            If the name already exists with different rules.
        """
        name = row["auto_curation_rules_name"]
        master, rule_inserts = prepare_auto_curation_rules(row, rule_rows)
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
            # Skip the raising insert overrides via the next insert in the MRO.
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
                    "missing_policy",
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
        """The shipped rule sets (``_core.recipe_catalog``)."""
        return auto_curation_default_payloads()

    @classmethod
    def insert_default(cls):
        """Insert the default auto-curation rule sets (idempotent)."""
        for master, rules in cls._default_payloads():
            cls.insert_rules(master, rules, skip_duplicates=True)


class _WaveformsAccessor:
    """``WaveformExtractor``-shaped view over a SortingAnalyzer's waveforms."""

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

    A preview curation (``apply_merge=False`` with an unapplied proposed
    merge) is rejected, since its units are not the final merged set. Insert
    through ``insert_selection``.
    """

    definition = """
    curation_evaluation_id: uuid
    ---
    -> CurationV2
    -> QualityMetricParameters
    -> AutoCurationRules
    -> AnalyzerWaveformParameters.proj(metric_waveform_params_name="waveform_params_name")
    observation_version: int  # observed-time contract the evaluation was selected under
    """

    @classmethod
    def insert_selection(cls, key: dict) -> dict:
        """Insert or find a curation-evaluation selection; return PK-only dict.

        ``curation_evaluation_id`` is a deterministic ``uuid5`` of the
        selection fields. ``metric_waveform_params_name`` (the whitened recipe
        for PC/NN metrics) defaults to the sort's region metric row.

        Raises
        ------
        ValueError
            If the parent curation is a preview (unapplied proposed merges) or
            the metric recipe is not whitened.
        DuplicateSelectionError
            If an existing row for this identity has a non-deterministic id.
        """
        from spyglass.spikesorting.v2._core.selection_identity import (
            assert_supplied_id_matches,
            deterministic_id,
        )

        metric_waveform_params_name = key.get("metric_waveform_params_name")
        if metric_waveform_params_name is None:
            # [1] is the metric element of the (display, metric) pair.
            preproc = SortingSelection.resolve_source_preprocessing_params_name(
                {"sorting_id": key["sorting_id"]}
            )
            metric_waveform_params_name = waveform_params_for_preprocessing(
                preproc
            )[1]

        _assert_is_metric_recipe(metric_waveform_params_name)

        parent_key = {
            "sorting_id": key["sorting_id"],
            "curation_id": key["curation_id"],
        }
        # Before the find-existing return, so a bypassed preview row is not
        # handed back.
        CurationV2.assert_committed_curation(
            parent_key, context="CurationEvaluation"
        )

        from spyglass.spikesorting.v2._core.observed_time import (
            OBSERVATION_VERSION,
        )

        identity = {
            "sorting_id": key["sorting_id"],
            "curation_id": key["curation_id"],
            "metric_params_name": key["metric_params_name"],
            "auto_curation_rules_name": key["auto_curation_rules_name"],
            "metric_waveform_params_name": metric_waveform_params_name,
            "observation_version": OBSERVATION_VERSION,
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
        except dj.errors.DuplicateError:
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
        """Call :meth:`insert_selection` from positional args (v1 analog).

        Returns
        -------
        dict
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
        from spyglass.spikesorting.v2._core.selection_identity import (
            existing_selection_pk,
        )

        existing_ids = (cls & identity).fetch("curation_evaluation_id")
        return existing_selection_pk(
            [uuid.UUID(str(cid)) for cid in existing_ids],
            deterministic_id,
            pk_field="curation_evaluation_id",
            bypass_message=lambda bypassed: (
                "CurationEvaluationSelection has duplicate selection rows for "
                f"{identity} with non-deterministic id(s) "
                f"{sorted(map(str, bypassed))} (expected the content-addressed "
                f"{deterministic_id}). This is an integrity bug -- a row was "
                "inserted bypassing insert_selection."
            ),
        )

    @classmethod
    def pc_requesting(cls):
        """Selections whose QualityMetricParameters request PC/NN metrics.

        These build a whitened metric analyzer. Single source of truth for
        the recompute key_source and the orphan-folder audit.
        """
        return cls * QualityMetricParameters & "skip_pc_metrics = 0"


@schema
class CurationEvaluation(StagedOutputCleanupMixin, SpyglassMixin, dj.Computed):
    """Quality metrics / merge suggestions / labels over a committed curation.

    Scores the curation's own units (merged units over their merged spike
    trains). Outputs are proposals in three NWB tables; the acceptance helpers
    (``accept_evaluation_outputs``, ``use_evaluation_labels``,
    ``overlay_evaluation_labels``, ``preview_merges``) commit them to a child
    ``CurationV2`` row.

    - Root or label-only curation: reuses the sort's cached analyzers.
    - Applied-merge curation: uses the curation's merged display-analyzer
      cache and temporary metric analyzers.
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
    # source_analyzer_hashes lets detect_stale_source flag drift in the
    # regeneratable fast-path analyzers ("display", plus "metric" for PC/NN).
    _parallel_make = True

    def make_fetch(self, key) -> CurationEvaluationFetched:
        """Resolve every DB input make_compute needs (no SI/NWB compute here).

        Re-asserts a committed parent and a whitened metric recipe, so a row
        planted via ``allow_direct_insert`` cannot reach compute.
        """

        return _metric_curation_fetch.fetch_evaluation_inputs(key)

    def make_compute(
        self,
        key,
        recording_inputs: EvaluationRecordingInputs,
        sorting_inputs: EvaluationSortingInputs,
        analyzer_inputs: EvaluationAnalyzerInputs,
        metric_inputs: EvaluationMetricInputs,
    ) -> CurationEvaluationComputed:
        """Compute metrics / merges / labels over the committed curation.

        No DB writes; every input comes from ``make_fetch``. The only DB
        access stages the output file.
        """
        import spikeinterface as si

        from spyglass.spikesorting.v2._storage.provenance import (
            CURATION_EVALUATION_PROVENANCE,
            build_provenance_table,
        )
        from spyglass.spikesorting.v2._recording.source import (
            read_effective_recording,
        )

        spikeinterface_version = si.__version__
        with staged_analysis_file(
            recording_inputs.nwb_file_name,
            context="CurationEvaluation.make_compute",
        ) as analysis_file_name:
            abs_path = AnalysisNwbfile.get_abs_path(analysis_file_name)
            wants_pc = (not metric_inputs.skip_pc_metrics) and bool(
                _requested_pc_metrics(metric_inputs.metric_names)
            )

            # recording_content_hash is of the traces actually scored (the
            # corrected recording's, when motion-corrected).
            corrected_id = recording_inputs.traces.key.get(
                "motion_corrected_recording_id"
            )

            # Shared by the populated and zero-unit paths.
            base_provenance = {
                "sorting_id": str(sorting_inputs.sorting_id),
                "curation_id": int(sorting_inputs.curation_id),
                "metric_params_name": metric_inputs.metric_params_name,
                "auto_curation_rules_name": metric_inputs.auto_curation_rules_name,
                "display_waveform_params_name": analyzer_inputs.display_waveform_params_name,
                "metric_waveform_params_name": analyzer_inputs.metric_waveform_params_name,
                "metric_names": list(metric_inputs.metric_names),
                "metric_kwargs": metric_inputs.metric_kwargs,
                "auto_merge_preset": metric_inputs.auto_merge_preset,
                "auto_merge_kwargs": metric_inputs.auto_merge_kwargs,
                "auto_curation_rules": metric_inputs.rule_rows,
                "source_kind": recording_inputs.source_kind,
                "recording_id": recording_inputs.recording_id,
                "concat_recording_id": recording_inputs.concat_recording_id,
                "recording_content_hash": recording_inputs.traces.row[
                    "content_hash"
                ],
                "spikeinterface_version": spikeinterface_version,
            }
            if corrected_id is not None:
                base_provenance["motion_corrected_recording_id"] = str(
                    corrected_id
                )

            from spyglass.spikesorting.v2._core.observed_time import (
                OBSERVATION_VERSION,
            )

            base_provenance["observation_version"] = OBSERVATION_VERSION
            base_provenance["observed_presence_bin_duration_s"] = (
                metric_inputs.observed_presence_bin_duration_s
            )

            def _provenance_tables(source_analyzer_hashes):
                from spyglass.spikesorting.v2._core.runtime import (
                    runtime_environment_provenance,
                )

                return [
                    build_provenance_table(
                        CURATION_EVALUATION_PROVENANCE,
                        {
                            **base_provenance,
                            "source_analyzer_hashes": source_analyzer_hashes,
                            **runtime_environment_provenance(
                                job_kwargs=metric_inputs.metric_job_kwargs,
                                execution_params={
                                    "stage": "curation_evaluation",
                                    "analyzer_job_kwargs": analyzer_inputs.analyzer_job_kwargs,
                                },
                            ),
                        },
                    )
                ]

            # SI cannot build an analyzer over zero units; write empty tables.
            if not sorting_inputs.expected_unit_ids:
                logger.warning(
                    "CurationEvaluation: curation "
                    f"(sorting_id={sorting_inputs.sorting_id}, curation_id={sorting_inputs.curation_id}) has "
                    "zero units; writing empty metric/merge/label tables."
                )
                object_ids = self._write_empty(
                    abs_path, provenance_tables=_provenance_tables(None)
                )
                return CurationEvaluationComputed(
                    analysis_file_name,
                    *object_ids,
                    recording_inputs.nwb_file_name,
                    spikeinterface_version,
                    None,
                )

            recording = read_effective_recording(
                recording_inputs.traces_abs_path,
                recording_inputs.traces,
                artifact_valid_times=recording_inputs.artifact_valid_times,
                artifact_detection_id=recording_inputs.artifact_detection_id,
                recording_id=recording_inputs.recording_id,
            )
            statistics_spans = list(recording_inputs.statistics_spans)

            from spyglass.spikesorting.v2._storage.observation_io import (
                observation_metrics_from_nwb,
            )

            observation_metrics, interval_hash = observation_metrics_from_nwb(
                sorting_inputs.curated_units.abs_path,
                recording,
                bin_duration_s=metric_inputs.observed_presence_bin_duration_s,
            )
            base_provenance["observation_intervals_hash"] = interval_hash

            evaluate = (
                _evaluation_analyzers.evaluate_cached_analyzers
                if sorting_inputs.use_fast_path
                else _evaluation_analyzers.evaluate_temporary_analyzers
            )
            metrics_df, labels_by_unit, merge_groups, source_analyzer_hashes = (
                evaluate(
                    recording,
                    sorting_inputs=sorting_inputs,
                    analyzer_inputs=analyzer_inputs,
                    metric_inputs=metric_inputs,
                    wants_pc=wants_pc,
                    observation_metrics=observation_metrics,
                    statistics_spans=statistics_spans,
                    # Concat traces use a synthetic clock; the spans keep
                    # the member/acquisition joins timestamps cannot recover.
                    continuity_spans=(
                        recording_inputs.traces.row["continuity_spans"]
                        if recording_inputs.source_kind
                        == "concatenated_recording"
                        else None
                    ),
                )
            )
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
                recording_inputs.nwb_file_name,
                spikeinterface_version,
                source_analyzer_hashes,
            )

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

        A direct call (outside ``populate``) must remove a failed attempt's
        staged file itself.
        """

        with self._safe_context():
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

    @classmethod
    def detect_stale_source(cls, key) -> dict:
        """Flag whether an evaluation's recorded source provenance still holds.

        Compares the stored ``spikeinterface_version`` with the running one
        and, on the fast path, re-hashes each analyzer in
        ``source_analyzer_hashes``. Merged-curation evaluations compare only
        the SI version.

        Parameters
        ----------
        key : dict
            Restriction selecting a single ``CurationEvaluation`` row.

        Returns
        -------
        dict
            ``{"stale": bool, "reasons": list[str],
            "spikeinterface_version": {"stored", "current"},
            "source_analyzer_hashes": {"stored", "current"}}``. ``reasons``
            entries are ``source_analyzer_hash:<role>`` (content mismatch) or
            ``source_analyzer_missing:<role>`` /
            ``source_analyzer_invalid:<role>`` (cache reclaimed or corrupt;
            reported, not raised). ``current`` maps
            an absent/invalid role to ``None`` and is empty on the merged path.
        """

        return _metric_curation_fetch.detect_stale_source(cls, key)

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

    @classmethod
    def _read_outputs(cls, key) -> tuple:
        """Return ``(metrics, merge_groups, labels)`` from one file open.

        Same values as :meth:`get_metrics`, :meth:`get_suggested_merge_groups`
        and :meth:`get_labels`.
        """
        row = (cls & key).fetch1()
        abs_path = AnalysisNwbfile.get_abs_path(row["analysis_file_name"])
        return read_analyzer_curation_tables(
            abs_path,
            metrics_object_id=row["metrics_object_id"],
            merge_suggestions_object_id=row["merge_suggestions_object_id"],
            proposed_labels_object_id=row["proposed_labels_object_id"],
        )

    # ---- acceptance helpers (evaluation outputs -> committed curation) ----

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

        The child is always committed; use :meth:`preview_merges` for a draft.
        Merges apply only if ``merge_groups`` or ``use_all_suggested_merges``
        is given; otherwise the child is labels-only.

        With ``label_policy="replace"`` the proposed labels are the child's
        full label state, so a unit the rules no longer flag drops a stale
        ``reject`` / ``noise``. The child records
        ``curation_source='curation_evaluation'``.

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
            ``{unit_id: [label, ...]}`` to write; ``None`` uses the
            evaluation's proposed labels. These are in the pre-merge namespace,
            so a label on a unit absorbed by a merge in this call is dropped
            with a warning. Re-evaluate the merged child to label merged units.
        label_policy : str
            ``"replace"`` (default) or ``"inherit"``; see
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

        return _evaluation_acceptance.accept_evaluation_outputs(
            self,
            key,
            merge_groups=merge_groups,
            use_all_suggested_merges=use_all_suggested_merges,
            labels=labels,
            label_policy=label_policy,
            description=description,
            allow_custom_labels=allow_custom_labels,
            reuse_existing=reuse_existing,
        )

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

        Inherits the evaluated curation's labels; pre-merge evaluation labels
        are not applied.
        """
        return _evaluation_acceptance.create_preview_curation(
            self,
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

        Inherits existing labels; pre-merge evaluation labels are not applied.
        Re-evaluate the merged child, then call :meth:`use_evaluation_labels`
        or :meth:`overlay_evaluation_labels`.
        """
        accepted = _evaluation_acceptance.require_merge_acceptance(
            self,
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

        Inherits existing labels.
        """
        accepted = _evaluation_acceptance.require_merge_acceptance(
            self,
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
        description: str = "evaluation labels (replace)",
        allow_custom_labels: bool = False,
        reuse_existing: bool = True,
    ) -> dict:
        """Use the evaluation verdict as the child's full label state.

        Label-only. Labels the evaluation does not propose are cleared, so a
        unit no longer flagged loses a stale ``reject`` / ``noise``. Use
        :meth:`overlay_evaluation_labels` to keep current labels.

        Returns
        -------
        dict
            The child's ``{"sorting_id", "curation_id"}``.
        """
        return self.accept_evaluation_outputs(
            key,
            merge_groups=None,
            use_all_suggested_merges=False,
            labels=None,
            label_policy="replace",
            description=description,
            allow_custom_labels=allow_custom_labels,
            reuse_existing=reuse_existing,
        )

    def overlay_evaluation_labels(
        self,
        key,
        *,
        description: str = "evaluation labels (overlay)",
        allow_custom_labels: bool = False,
        reuse_existing: bool = True,
    ) -> dict:
        """Overlay the evaluation's labels on the curation's current ones.

        Label-only. Keeps existing labels and adds or overrides the proposed
        ones, so prior auto labels can survive (unlike
        :meth:`use_evaluation_labels`).

        Returns
        -------
        dict
            The child's ``{"sorting_id", "curation_id"}``.
        """
        return self.accept_evaluation_outputs(
            key,
            merge_groups=None,
            use_all_suggested_merges=False,
            labels=None,
            label_policy="inherit",
            description=description,
            allow_custom_labels=allow_custom_labels,
            reuse_existing=reuse_existing,
        )

    # ---- compute helpers (DB-light; SI work) -----------------------------

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

    # ---- visualization (notebook-facing) ---------------------------------

    def get_waveforms(self, key, fetch_all: bool = False):
        """Return a waveform accessor over this curation's display analyzer.

        Exposes ``get_waveforms_one_unit(unit_id)`` and v1-style
        ``get_waveforms(unit_id)``. ``fetch_all`` is accepted for v1 parity
        but only the sort-time waveform subsample is available.
        """
        if fetch_all:
            logger.warning(
                "CurationEvaluation.get_waveforms(fetch_all=True): returning "
                "the sort-time waveform subsample (full re-extraction is not "
                "supported)."
            )
        return _WaveformsAccessor(self._analyzer_for(key))

    def _display_analyzer_key(self, key):
        """Resolve the curation + DISPLAY recipe request for every plot.

        Raw-unit curations reuse the sort analyzer; committed merged
        curations use their own.
        """
        from spyglass.spikesorting.v2._curation.analyzer import (
            display_analyzer_request,
        )

        sel = (CurationEvaluationSelection & key).fetch1()
        return display_analyzer_request(
            {
                "sorting_id": sel["sorting_id"],
                "curation_id": int(sel["curation_id"]),
            }
        )

    def _analyzer_for(self, key):
        """Return this curation's DISPLAY analyzer for a controlled read.

        Never the whitened metric analyzer, so plots show real waveforms.
        Do not mutate it; use ``_display_analyzer`` to add extensions.
        """
        from spyglass.spikesorting.v2._curation.analyzer import (
            _resolve_curation_analyzer,
        )

        return _resolve_curation_analyzer(**self._display_analyzer_key(key))

    @contextmanager
    def _display_analyzer(self, key, *, extra_extensions=None):
        """Yield a display analyzer without mutating its published cache.

        A missing extension is computed into a disk-backed derivative, never
        into the published cache.
        """
        from spyglass.spikesorting.v2._curation.analyzer import (
            curation_analyzer_with_extensions,
        )

        with curation_analyzer_with_extensions(
            **self._display_analyzer_key(key),
            extra_extensions=extra_extensions,
        ) as analyzer:
            yield analyzer

    def plot_units_qc(
        self, key, *, metric_names=None, color_metric: str = "snr", axes=None
    ):
        """Static population QC overview: metric histograms + depth scatter.

        One histogram per quality metric (NaNs dropped) plus a scatter of unit
        positions colored by ``color_metric``. Pass ``axes`` to draw into an
        existing layout; otherwise a figure is created.

        Returns
        -------
        dict[str, matplotlib.axes.Axes]
            Axes keyed by metric name plus ``"scatter"``. A zero-unit sort
            returns ``{"empty": ax}``.
        """
        from spyglass.spikesorting.v2._curation.metric_plots import (
            plot_units_qc_figure,
        )

        metrics = self.get_metrics(key)
        try:
            with self._display_analyzer(
                key, extra_extensions={"unit_locations": {}}
            ) as analyzer:
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
        from spyglass.spikesorting.v2._curation.metric_plots import (
            correlograms_from_analyzer,
        )

        return correlograms_from_analyzer(
            self._analyzer_for(key), window_ms=window_ms, bin_ms=bin_ms
        )

    def plot_correlograms(
        self, key, *, unit_ids=None, window_ms=100.0, bin_ms=5.0
    ):
        """Plot autocorrelograms (one panel per unit). Ported BurstPair view."""
        from spyglass.spikesorting.v2._curation.metric_plots import (
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
        from spyglass.spikesorting.v2._curation.metric_plots import (
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
        from spyglass.spikesorting.v2._curation.metric_plots import (
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
        from spyglass.spikesorting.v2._curation.metric_plots import (
            peak_amplitudes_from_analyzer,
            validate_unit_pairs,
        )

        analyzer = self._analyzer_for(key)
        used = validate_unit_pairs(list(analyzer.unit_ids), pairs)
        peak_amps, peak_times = peak_amplitudes_from_analyzer(analyzer)
        return plot_burst_peak_over_time(
            peak_amps, peak_times, used, overlap=overlap
        )

    def get_burst_pair_metrics(
        self,
        key,
        pairs=None,
        *,
        isi_threshold_ms: float | None = None,
        window_ms: float = 100.0,
        bin_ms: float = 5.0,
    ):
        """Per-pair burst-merge diagnostics as a DataFrame (v1 BurstPair data).

        Computed from the display analyzer, not stored. Columns:

        - ``wf_similarity``: SI cosine template similarity.
        - ``isi_violation``: violating intervals / (spikes - 1) of the merged
          train (cross-unit duplicates removed, source clock).
        - ``xcorrel_asymm``: directional, so ordered pairs are separate rows.
        - ``unit_distance``: distance between unit locations.

        ``pairs`` defaults to every ordered pair. ``isi_threshold_ms``
        defaults to the evaluation's metric recipe.

        Returns
        -------
        pd.DataFrame
            Shape ``(n_pairs, 4)``, MultiIndex ``(unit1, unit2)``.
        """
        from spyglass.spikesorting.v2._curation.metric_plots import (
            burst_pair_metrics_frame,
        )

        if isi_threshold_ms is None:
            metric_params_name = (CurationEvaluationSelection & key).fetch1(
                "metric_params_name"
            )
            isi_threshold_ms = QualityMetricParameters.get_isi_threshold_ms(
                metric_params_name
            )

        with self._display_analyzer(
            key,
            extra_extensions={
                "template_similarity": {},
                "unit_locations": {},
            },
        ) as analyzer:
            return burst_pair_metrics_frame(
                analyzer,
                pairs=pairs,
                isi_threshold_ms=isi_threshold_ms,
                window_ms=window_ms,
                bin_ms=bin_ms,
                continuity_spans=self._isi_continuity_spans(key),
            )

    def _isi_continuity_spans(self, key):
        """The evaluated sort's continuity spans when its clock is synthetic.

        A concat source's synthetic clock hides its joins; other sources
        return ``None``.
        """
        sorting_id = (CurationEvaluationSelection & key).fetch1("sorting_id")
        lineage, traces = SortingSelection.resolve_effective_source(
            {"sorting_id": sorting_id}
        )
        if lineage.kind != "concatenated_recording":
            return None
        return traces.row["continuity_spans"]

    def plot_burst_pair_metrics(self, key, pairs=None, **kwargs):
        """Per-pair burst-metrics scatter for the sort (v1 BurstPair analog).

        Waveform similarity vs cross-correlogram asymmetry, one point per
        pair. ``kwargs`` go to ``get_burst_pair_metrics``.
        """
        from spyglass.spikesorting.utils_burst import plot_burst_metrics

        frame = self.get_burst_pair_metrics(key, pairs=pairs, **kwargs)
        # plot_burst_metrics reads unit1/unit2 per record, which the
        # MultiIndex would otherwise drop from to_dict("records").
        return plot_burst_metrics(frame.reset_index().to_dict("records"))

    # ---- SI metric / merge delegates (see v2.visualization facade) --------

    def plot_metrics(self, key, *, backend="matplotlib", **kwargs):
        """Delegate to ``visualization.plot_metrics`` for this curation."""
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

        SI's ``PotentialMergesWidget`` has no matplotlib backend.
        """
        from spyglass.spikesorting.v2 import visualization

        return visualization.plot_suggested_merges(
            key, backend=backend, **kwargs
        )

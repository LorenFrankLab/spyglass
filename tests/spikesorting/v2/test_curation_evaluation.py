"""Tests for committed-curation evaluation and final merged-unit metrics.

``CurationEvaluation`` scores an existing committed ``CurationV2`` row in that
curation's own unit namespace: merged units get metrics recomputed over their
merged spike trains/templates, not inherited from a contributor. Preview
(draft) curations -- ``apply_merge=False`` with a real proposed merge group --
are rejected at the evaluation boundary.

Tiers:

- ``slow`` / ``integration`` end-to-end tests populate ``CurationEvaluation``
  on the planted two-unit sort and the shared MEArec smoke sort.
- hermetic compute-level tests pin the merged-template metric contract with
  controlled planted templates (no DB populate).
"""

from __future__ import annotations

import pandas as pd
import pytest

# ---------- committed-curation predicate ------------------------------------


@pytest.mark.slow
@pytest.mark.integration
def test_is_committed_curation_distinguishes_preview(planted_two_unit_sort):
    """Root, label-only, and applied-merge rows are committed; a preview is not.

    A preview (``apply_merge=False`` with a real >=2-unit proposed merge group)
    is a draft, not a final downstream curation; ``assert_committed_curation``
    raises for it and is a no-op for the committed states.
    """
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting

    sorting_key = dict(planted_two_unit_sort)
    unit_ids = sorted(
        int(u) for u in (Sorting.Unit & sorting_key).fetch("unit_id")
    )
    assert len(unit_ids) >= 2, "need >=2 units to plant a merge"

    clear_curations_for(planted_two_unit_sort)
    try:
        root = CurationV2.insert_curation(sorting_key=sorting_key)
        assert CurationV2.is_committed_curation(root) is True
        CurationV2.assert_committed_curation(root)  # no raise

        label_only = CurationV2.insert_curation(
            sorting_key,
            parent_curation_id=root["curation_id"],
            labels={unit_ids[0]: ["noise"]},
        )
        assert CurationV2.is_committed_curation(label_only) is True
        CurationV2.assert_committed_curation(label_only)

        merged = CurationV2.create_merged_curation(
            sorting_key,
            merge_groups=[[unit_ids[0], unit_ids[1]]],
            parent_curation_id=root["curation_id"],
        )
        assert CurationV2.is_committed_curation(merged) is True
        CurationV2.assert_committed_curation(merged)

        preview = CurationV2.propose_merge_curation(
            sorting_key,
            merge_groups=[[unit_ids[0], unit_ids[1]]],
            parent_curation_id=root["curation_id"],
        )
        assert CurationV2.is_committed_curation(preview) is False
        with pytest.raises(ValueError, match="preview"):
            CurationV2.assert_committed_curation(
                preview, context="CurationEvaluation"
            )
    finally:
        clear_curations_for(planted_two_unit_sort)


# ---------- DB-free resolved reconstruction helpers -------------------------
#
# CurationEvaluation.make_compute runs in a parallel worker with no DB access,
# so it reconstructs the recording / analyzer from inputs make_fetch resolved.
# These parity tests pin the DB-free helpers against the existing DB-coupled
# paths: a divergence here would silently feed metrics the wrong recording.


def _resolved_recording_inputs(sorting_key):
    """Resolve (the way make_fetch will) the DB-free recording inputs."""
    from spyglass.spikesorting.v2._artifact_intervals import (
        read_artifact_removed_intervals,
    )
    from spyglass.spikesorting.v2.recording import Recording, RecordingSelection
    from spyglass.spikesorting.v2.session_group import ConcatenatedRecording
    from spyglass.spikesorting.v2.sorting import SortingSelection

    source = SortingSelection.resolve_source(sorting_key)
    artifact_detection_id = SortingSelection.resolve_artifact_detection(
        sorting_key
    )
    if source.kind == "recording":
        rec_row = (
            Recording & {"recording_id": source.key["recording_id"]}
        ).fetch1()
        recording_id = source.key["recording_id"]
    else:
        rec_row = (ConcatenatedRecording & source.key).fetch1()
        recording_id = None
    valid_times = None
    if source.kind == "recording" and artifact_detection_id is not None:
        nwb = (RecordingSelection & {"recording_id": recording_id}).fetch1(
            "nwb_file_name"
        )
        valid_times = read_artifact_removed_intervals(
            {"artifact_detection_id": artifact_detection_id}, as_dict=True
        )[nwb]
    return source, rec_row, recording_id, artifact_detection_id, valid_times


@pytest.mark.slow
@pytest.mark.integration
def test_resolved_recording_reconstruction_matches_db_path(populated_sorting):
    """The DB-free recording reconstruction equals the DB-coupled path.

    ``reconstruct_recording_for_sorting_from_resolved`` reads the cached
    recording NWB + applies the artifact mask from make_fetch-resolved inputs,
    with no DB access; it must reproduce the same channels/samples/traces as the
    DB-coupled ``reconstruct_recording_and_sorting``.
    """
    import numpy as np

    from spyglass.spikesorting.v2._sorting_analyzer import (
        reconstruct_recording_and_sorting,
        reconstruct_recording_for_sorting_from_resolved,
    )
    from spyglass.spikesorting.v2.sorting import Sorting

    sorting_key = dict(populated_sorting)
    ref_rec, _ = reconstruct_recording_and_sorting(Sorting(), sorting_key)

    source, rec_row, recording_id, artifact_id, valid_times = (
        _resolved_recording_inputs(sorting_key)
    )
    out_rec = reconstruct_recording_for_sorting_from_resolved(
        recording_row=rec_row,
        source_kind=source.kind,
        artifact_valid_times=valid_times,
        artifact_detection_id=artifact_id,
        recording_id=recording_id,
    )

    assert out_rec.get_num_channels() == ref_rec.get_num_channels()
    assert out_rec.get_num_samples() == ref_rec.get_num_samples()
    assert bool(out_rec.get_annotation("is_filtered")) is True
    np.testing.assert_allclose(
        out_rec.get_traces(start_frame=0, end_frame=200),
        ref_rec.get_traces(start_frame=0, end_frame=200),
    )


@pytest.mark.slow
@pytest.mark.integration
def test_resolved_analyzer_loader_matches_get_analyzer(populated_sorting):
    """The DB-free analyzer loader reproduces the cached display analyzer.

    ``load_or_rebuild_analyzer_from_resolved`` loads (or rebuilds) the canonical
    display analyzer from make_fetch-resolved inputs without any DB read; it
    must return the same unit ids and templates as ``Sorting().get_analyzer``.
    """
    import numpy as np

    from spyglass.spikesorting.v2._analyzer_cache import analyzer_path
    from spyglass.spikesorting.v2._sorting_analyzer import (
        fetch_waveform_params,
        load_or_rebuild_analyzer_from_resolved,
        reconstruct_recording_and_sorting,
        resolve_display_waveform_params_name,
    )
    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        Sorting,
        SortingSelection,
    )
    from spyglass.spikesorting.v2.utils import _resolved_job_kwargs

    sorting_key = dict(populated_sorting)
    ref = Sorting().get_analyzer(sorting_key)
    ref_unit_ids = sorted(int(u) for u in ref.unit_ids)

    sorting_id, n_units = (Sorting & sorting_key).fetch1(
        "sorting_id", "n_units"
    )
    display_name = resolve_display_waveform_params_name(Sorting(), sorting_id)
    display_params = fetch_waveform_params(display_name)
    recording, raw_sorting = reconstruct_recording_and_sorting(
        Sorting(), sorting_key
    )
    sorter_row = (
        SorterParameters
        & (
            (SortingSelection & sorting_key).proj(
                "sorter", "sorter_params_name"
            )
        )
    ).fetch1()
    out = load_or_rebuild_analyzer_from_resolved(
        sorting_id=sorting_id,
        n_units=int(n_units),
        analyzer_folder=analyzer_path(sorting_id, display_name),
        waveform_params=display_params,
        recording=recording,
        sorting=raw_sorting,
        sorter_row=sorter_row,
        job_kwargs=_resolved_job_kwargs(sorter_row["job_kwargs"]),
    )
    assert sorted(int(u) for u in out.unit_ids) == ref_unit_ids
    np.testing.assert_allclose(
        out.get_extension("templates").get_data(),
        ref.get_extension("templates").get_data(),
    )


# ---------- CurationEvaluationSelection -------------------------------------
# ``curation_evaluation_defaults`` (metric/auto-rule/waveform Lookup rows) is a
# shared fixture in conftest.py.


@pytest.mark.slow
@pytest.mark.integration
def test_curation_evaluation_selection_rejects_preview(
    planted_two_unit_sort, curation_evaluation_defaults
):
    """A preview curation is rejected at CurationEvaluationSelection.insert.

    A preview (``apply_merge=False`` with a real proposed merge) is a draft;
    evaluating it would score the unmerged units. ``insert_selection`` must
    raise rather than accept it.
    """
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.metric_curation import (
        CurationEvaluationSelection,
    )
    from spyglass.spikesorting.v2.sorting import Sorting

    sorting_key = dict(planted_two_unit_sort)
    unit_ids = sorted(
        int(u) for u in (Sorting.Unit & sorting_key).fetch("unit_id")
    )
    clear_curations_for(planted_two_unit_sort)
    try:
        root = CurationV2.insert_curation(sorting_key=sorting_key)
        preview = CurationV2.propose_merge_curation(
            sorting_key,
            merge_groups=[[unit_ids[0], unit_ids[1]]],
            parent_curation_id=root["curation_id"],
        )
        with pytest.raises(ValueError, match="preview"):
            CurationEvaluationSelection.insert_selection(
                {
                    **preview,
                    "metric_params_name": "minimal",
                    "auto_curation_rules_name": "none",
                }
            )
        # A committed (applied-merge) curation is accepted.
        merged = CurationV2.create_merged_curation(
            sorting_key,
            merge_groups=[[unit_ids[0], unit_ids[1]]],
            parent_curation_id=root["curation_id"],
        )
        sel = CurationEvaluationSelection.insert_selection(
            {
                **merged,
                "metric_params_name": "minimal",
                "auto_curation_rules_name": "none",
            }
        )
        assert sel
    finally:
        clear_curations_for(planted_two_unit_sort)


# ---------- CurationEvaluation end-to-end (committed-curation metrics) -------


def _ensure_counts_metric_params():
    """A skip-PC metric row with spike-train metrics for the merged-unit test."""
    from spyglass.spikesorting.v2.metric_curation import (
        QualityMetricParameters,
    )

    name = "counts_minimal"
    if not (QualityMetricParameters & {"metric_params_name": name}):
        QualityMetricParameters.insert1(
            {
                "metric_params_name": name,
                "metric_names": ["num_spikes", "firing_rate", "snr"],
                "metric_kwargs": {"snr": {"peak_sign": "neg"}},
                "skip_pc_metrics": True,
            }
        )
    return name


@pytest.mark.slow
@pytest.mark.integration
def test_final_metrics_recomputed_for_merged_unit(
    planted_two_unit_sort, curation_evaluation_defaults
):
    """Final metrics over a committed applied-merge curation are recomputed.

    The merged unit appears with metrics computed over its merged spike train;
    the absorbed contributors are absent, and ``num_spikes`` equals the merged
    train length (== ``CurationV2.Unit.n_spikes``). This fails on the old
    raw-sort analyzer path, which would score the original contributor units.
    """
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.metric_curation import (
        CurationEvaluation,
        CurationEvaluationSelection,
    )
    from spyglass.spikesorting.v2.sorting import Sorting

    sorting_key = dict(planted_two_unit_sort)
    unit_ids = sorted(
        int(u) for u in (Sorting.Unit & sorting_key).fetch("unit_id")
    )
    metric_params = _ensure_counts_metric_params()
    clear_curations_for(planted_two_unit_sort)
    try:
        root = CurationV2.insert_curation(sorting_key=sorting_key)
        merged = CurationV2.create_merged_curation(
            sorting_key,
            merge_groups=[[unit_ids[0], unit_ids[1]]],
            parent_curation_id=root["curation_id"],
        )
        merged_units = (CurationV2.Unit & merged).fetch(
            "unit_id", "n_spikes", as_dict=True
        )
        assert (
            len(merged_units) == 1
        ), "apply_merge should yield one merged unit"
        merged_uid = int(merged_units[0]["unit_id"])
        merged_n = int(merged_units[0]["n_spikes"])

        sel = CurationEvaluationSelection.insert_selection(
            {
                **merged,
                "metric_params_name": metric_params,
                "auto_curation_rules_name": "none",
            }
        )
        CurationEvaluation.populate(sel, reserve_jobs=False)
        metrics = CurationEvaluation.get_metrics(sel)

        assert merged_uid in metrics.index
        for contributor in unit_ids:
            assert (
                contributor not in metrics.index
            ), "absorbed contributor must not appear in merged metrics"
        assert int(metrics.loc[merged_uid, "num_spikes"]) == merged_n
    finally:
        clear_curations_for(planted_two_unit_sort)


def test_curation_evaluation_records_source_provenance(
    populated_sorting_with_curation, curation_evaluation_defaults, monkeypatch
):
    """A fast-path (root-curation) evaluation records the SI version and the
    canonical raw-sort analyzer's content hash; detect_stale_source is clean in
    the same environment but flags a library-version drift and an analyzer-
    content drift. The evaluated curation identity + recipe names live on the
    selection FK, not duplicated on the row."""
    import spikeinterface as si

    from spyglass.spikesorting.v2.metric_curation import (
        CurationEvaluation,
        CurationEvaluationSelection,
    )

    sel = CurationEvaluationSelection.insert_selection(
        {
            **populated_sorting_with_curation,
            "metric_params_name": "minimal",
            "auto_curation_rules_name": "none",
        }
    )
    CurationEvaluation.populate(sel, reserve_jobs=False)
    row = (CurationEvaluation & sel).fetch1()
    assert row["spikeinterface_version"] == si.__version__
    # Fast path consumes the canonical display analyzer -> recorded as a manifest
    # ({role: hash}). "minimal" requests no PC metrics, so no metric analyzer.
    assert set(row["source_analyzer_hashes"]) == {"display"}
    assert row["source_analyzer_hashes"]["display"]
    # Evaluated identity is reachable via the selection FK, not duplicated.
    fetched_sel = (CurationEvaluationSelection & sel).fetch1()
    assert str(fetched_sel["sorting_id"]) == str(
        populated_sorting_with_curation["sorting_id"]
    )

    # Same environment -> not stale.
    assert CurationEvaluation.detect_stale_source(sel)["stale"] is False

    # A SpikeInterface-version drift is flagged.
    monkeypatch.setattr(si, "__version__", "0.0.0-test")
    version_drift = CurationEvaluation.detect_stale_source(sel)
    assert version_drift["stale"] is True
    assert "spikeinterface_version" in version_drift["reasons"]
    monkeypatch.undo()  # restore version before the analyzer-hash check

    # An analyzer-content drift (a different re-hash) is flagged per role.
    import spyglass.spikesorting.v2._recompute as rc

    monkeypatch.setattr(
        rc, "hash_extension_data", lambda analyzer, **k: {"x": "deadbeef"}
    )
    hash_drift = CurationEvaluation.detect_stale_source(sel)
    assert hash_drift["stale"] is True
    assert "source_analyzer_hash:display" in hash_drift["reasons"]
    monkeypatch.undo()

    # A reclaimed/missing analyzer cache (regeneratable scratch) is reported as
    # stale, not raised -- the stored metrics can no longer be reproduced.
    from spyglass.spikesorting.v2.exceptions import AnalyzerFolderMissingError
    from spyglass.spikesorting.v2.sorting import Sorting

    def _raise_missing(self, key, *args, **kwargs):
        raise AnalyzerFolderMissingError("analyzer folder reclaimed")

    monkeypatch.setattr(Sorting, "get_analyzer", _raise_missing)
    missing = CurationEvaluation.detect_stale_source(sel)
    assert missing["stale"] is True
    assert "source_analyzer_missing:display" in missing["reasons"]


def test_curation_evaluation_nwb_carries_inputs(
    populated_sorting_with_curation, curation_evaluation_defaults
):
    """The CurationEvaluation NWB embeds the evaluation inputs + source provenance.

    Alongside the metric/merge/label result tables, the artifact carries the
    metric param set + names, the display/metric recipe names, the auto-merge
    preset, the evaluated sorting/curation, and re-emits the source
    provenance the row stores (analyzer-hash manifest, SI version) plus the
    upstream recording content hash -- so the file is interpretable without the
    DB.
    """
    import spikeinterface as si

    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2._nwb_provenance import (
        CURATION_EVALUATION_PROVENANCE,
        read_provenance_values,
    )
    from spyglass.spikesorting.v2.metric_curation import (
        CurationEvaluation,
        CurationEvaluationSelection,
        QualityMetricParameters,
    )
    from spyglass.spikesorting.v2.recording import Recording
    from spyglass.spikesorting.v2.sorting import SortingSelection

    sel = CurationEvaluationSelection.insert_selection(
        {
            **populated_sorting_with_curation,
            "metric_params_name": "minimal",
            "auto_curation_rules_name": "none",
        }
    )
    CurationEvaluation.populate(sel, reserve_jobs=False)
    row = (CurationEvaluation & sel).fetch1()
    fetched_sel = (CurationEvaluationSelection & sel).fetch1()
    abs_path = AnalysisNwbfile.get_abs_path(row["analysis_file_name"])

    prov = read_provenance_values(abs_path, CURATION_EVALUATION_PROVENANCE)

    # Evaluated curation identity.
    assert prov["sorting_id"] == str(
        populated_sorting_with_curation["sorting_id"]
    )
    assert prov["curation_id"] == int(
        populated_sorting_with_curation["curation_id"]
    )
    # Recipe / param names.
    assert prov["metric_params_name"] == "minimal"
    assert prov["auto_curation_rules_name"] == "none"
    assert (
        prov["metric_waveform_params_name"]
        == fetched_sel["metric_waveform_params_name"]
    )
    assert prov["display_waveform_params_name"]
    # Metric param set + auto-merge preset.
    expected_metric_names = list(
        (QualityMetricParameters & {"metric_params_name": "minimal"}).fetch1(
            "metric_names"
        )
    )
    assert prov["metric_names"] == expected_metric_names
    assert isinstance(prov["auto_merge_preset"], str)
    # Source provenance, re-emitted from the row.
    assert (
        prov["spikeinterface_version"]
        == row["spikeinterface_version"]
        == si.__version__
    )
    assert prov["source_analyzer_hashes"] == row["source_analyzer_hashes"]
    # Upstream recording content hash + source kind.
    assert prov["source_kind"] == "recording"
    # A single-recording source carries no concat id.
    assert prov["concat_recording_id"] is None
    recording_id = SortingSelection.resolve_source(
        {"sorting_id": populated_sorting_with_curation["sorting_id"]}
    ).key["recording_id"]
    assert prov["recording_content_hash"] == (
        Recording & {"recording_id": recording_id}
    ).fetch1("content_hash")


def test_curation_evaluation_pc_eval_records_both_roles(
    populated_sorting_with_curation, curation_evaluation_defaults, monkeypatch
):
    """A PC/NN evaluation consumes BOTH the display and whitened metric analyzer,
    so the provenance manifest records both roles and detect_stale_source flags
    drift in the metric (PC) analyzer per role -- not just the display role."""
    from spyglass.spikesorting.v2.metric_curation import (
        CurationEvaluation,
        CurationEvaluationSelection,
    )

    # franklab_default requests PC/NN metrics (skip_pc_metrics=False) -> the
    # whitened metric analyzer is built + principal_components computed on it.
    sel = CurationEvaluationSelection.insert_selection(
        {
            **populated_sorting_with_curation,
            "metric_params_name": "franklab_default",
            "auto_curation_rules_name": "none",
        }
    )
    CurationEvaluation.populate(sel, reserve_jobs=False)
    row = (CurationEvaluation & sel).fetch1()
    assert set(row["source_analyzer_hashes"]) == {"display", "metric"}
    assert CurationEvaluation.detect_stale_source(sel)["stale"] is False

    # Perturb only the metric analyzer's hash (the one carrying
    # principal_components): the metric role is flagged, the display role is not.
    import spyglass.spikesorting.v2._recompute as rc

    real_hash = rc.hash_extension_data

    def fake_hash(analyzer, **kwargs):
        if analyzer.has_extension("principal_components"):
            return {"x": "deadbeef"}
        return real_hash(analyzer, **kwargs)

    monkeypatch.setattr(rc, "hash_extension_data", fake_hash)
    drift = CurationEvaluation.detect_stale_source(sel)
    assert drift["stale"] is True
    assert "source_analyzer_hash:metric" in drift["reasons"]
    assert "source_analyzer_hash:display" not in drift["reasons"]


def test_nn_noise_overlap_is_finite_not_silently_all_nan(
    populated_sorting_with_curation, curation_evaluation_defaults
):
    """nn_noise_overlap must be a real (finite) metric, not silently all-NaN.

    SI 0.104's nearest_neighbors_noise_overlap requires a ``median`` template
    operator. If the whitened metric analyzer builds ``templates`` without it,
    SI swallows the per-unit error and every ``nn_noise_overlap`` is NaN -- so
    the shipped auto-curation noise/reject rules (which threshold
    ``nn_noise_overlap``) silently fire on nothing. Assert, independent of any
    persisted/derived value, that at least one unit has a finite
    ``nn_noise_overlap`` so an all-NaN regression fails loudly rather than
    passing as "no units flagged".
    """
    from spyglass.spikesorting.v2.metric_curation import (
        CurationEvaluation,
        CurationEvaluationSelection,
    )

    sel = CurationEvaluationSelection.insert_selection(
        {
            **populated_sorting_with_curation,
            "metric_params_name": "franklab_default",
            "auto_curation_rules_name": "none",
        }
    )
    CurationEvaluation.populate(sel, reserve_jobs=False)
    metrics = CurationEvaluation.get_metrics(sel)
    assert "nn_noise_overlap" in metrics.columns
    assert metrics["nn_noise_overlap"].notna().any(), (
        "nn_noise_overlap is all-NaN: the whitened metric analyzer's "
        "'templates' extension is missing the 'median' operator that SI's "
        "nearest_neighbors_noise_overlap requires."
    )


def _two_distinct_template_inputs():
    """A 4-channel recording + two-unit and merged sortings, distinct templates.

    Unit 0 peaks on channel 0, unit 1 on channel 2 (distinct shapes, equal
    amplitude). The merged unit's template is the spike-count-weighted average,
    so its extremum (~half) differs from BOTH contributors -- the predictable
    setup for "merged-unit waveform metrics are recomputed, not inherited".
    """
    import numpy as np
    import spikeinterface as si
    from probeinterface import Probe

    fs = 30000.0
    n = 7000
    w = 15
    rng = np.random.default_rng(0)
    traces = (rng.standard_normal((n, 4)) * 2.0).astype("float32")
    shape = np.exp(-(np.arange(-w, w + 1) ** 2) / (2 * 5.0**2)).astype(
        "float32"
    )
    spikes0 = np.array([500, 1200, 1900, 2600, 3300])
    spikes1 = np.array([4000, 4700, 5400, 6100, 6500])
    for f in spikes0:
        traces[f - w : f + w + 1, 0] += -100.0 * shape
        traces[f - w : f + w + 1, [1, 2, 3]] += (-10.0 * shape)[:, None]
    for f in spikes1:
        traces[f - w : f + w + 1, 2] += -100.0 * shape
        traces[f - w : f + w + 1, [0, 1, 3]] += (-10.0 * shape)[:, None]

    rec = si.NumpyRecording(traces_list=[traces], sampling_frequency=fs)
    rec.set_channel_gains(1.0)
    rec.set_channel_offsets(0.0)
    probe = Probe(ndim=2)
    probe.set_contacts(
        positions=[[0.0, 0.0], [0.0, 20.0], [20.0, 0.0], [20.0, 20.0]],
        shapes="circle",
        shape_params={"radius": 5},
    )
    probe.set_device_channel_indices([0, 1, 2, 3])
    rec = rec.set_probe(probe)

    two = si.NumpySorting.from_unit_dict(
        [{0: spikes0, 1: spikes1}], sampling_frequency=fs
    )
    merged = si.NumpySorting.from_unit_dict(
        [{2: np.sort(np.concatenate([spikes0, spikes1]))}],
        sampling_frequency=fs,
    )
    return rec, two, merged


@pytest.mark.slow
@pytest.mark.integration
def test_merged_unit_waveform_metric_recomputed_not_inherited(
    dj_conn, tmp_path
):
    """A waveform metric (snr) on the merged unit is computed over the MERGED
    template, not inherited from a contributor.

    Drives the exact compute path CurationEvaluation uses for an applied-merge
    curation -- ``build_analyzer`` over the merged sorting, then
    ``CurationEvaluation._compute_metrics`` -- with two planted contributors whose
    distinct templates make the merged-template extremum predictable. The merged
    unit's snr must differ from BOTH contributors' pre-merge snr (it cannot have
    been inherited) and be smaller (the merged template extremum is diluted).
    """
    import spikeinterface as si

    from spyglass.spikesorting.v2._sorting_analyzer import build_analyzer
    from spyglass.spikesorting.v2.metric_curation import CurationEvaluation

    rec, two, merged = _two_distinct_template_inputs()
    waveform_params = {
        "ms_before": 1.0,
        "ms_after": 2.0,
        "max_spikes_per_unit": 20000,
        "whiten": False,
        "purpose": "display",
    }

    def _snr(sorting, name):
        folder = tmp_path / f"{name}.zarr"
        build_analyzer(
            sorting,
            rec,
            {"sorting_id": name},
            sorter_row={"job_kwargs": None},
            job_kwargs={},
            analyzer_folder=folder,
            waveform_params=waveform_params,
        )
        analyzer = si.load_sorting_analyzer(folder)
        # The planted recording is an in-memory NumpyRecording (not persistable),
        # so the reloaded zarr analyzer has no recording; reattach it so the
        # waveform extensions can compute. Production recordings are NWB-backed
        # and persist into the zarr, so the real path needs no reattach (the
        # DB-backed merged tests cover that).
        analyzer.set_temporary_recording(rec)
        df = CurationEvaluation._compute_metrics(
            analyzer,
            None,
            ["snr"],
            {"snr": {"peak_sign": "neg"}},
            True,
            {},
            template_metric_columns=[],
        )
        return df["snr"]

    contributors = _snr(two, "two_unit")
    merged_snr = _snr(merged, "merged_unit")

    assert set(int(u) for u in merged_snr.index) == {2}
    merged_value = float(merged_snr.loc[2])
    snr0 = float(contributors.loc[0])
    snr1 = float(contributors.loc[1])
    # Not inherited: differs from both contributors (data-dependent band).
    assert abs(merged_value - snr0) / snr0 > 0.1
    assert abs(merged_value - snr1) / snr1 > 0.1
    # The merged template extremum is the count-weighted average, so smaller.
    assert merged_value < snr0
    assert merged_value < snr1


@pytest.mark.slow
@pytest.mark.integration
def test_curation_evaluation_rejects_preview_at_make_fetch(
    planted_two_unit_sort, curation_evaluation_defaults
):
    """A preview row planted via allow_direct_insert is rejected at make_fetch.

    insert_selection guards previews, but a row planted directly bypasses it;
    make_fetch re-asserts the curation is committed so the preview compute path
    is unreachable.
    """
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    from spyglass.spikesorting.v2._selection_identity import deterministic_id
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.metric_curation import (
        CurationEvaluation,
        CurationEvaluationSelection,
    )
    from spyglass.spikesorting.v2.sorting import Sorting

    sorting_key = dict(planted_two_unit_sort)
    unit_ids = sorted(
        int(u) for u in (Sorting.Unit & sorting_key).fetch("unit_id")
    )
    clear_curations_for(planted_two_unit_sort)
    try:
        root = CurationV2.insert_curation(sorting_key=sorting_key)
        preview = CurationV2.propose_merge_curation(
            sorting_key,
            merge_groups=[[unit_ids[0], unit_ids[1]]],
            parent_curation_id=root["curation_id"],
        )
        identity = {
            "sorting_id": preview["sorting_id"],
            "curation_id": preview["curation_id"],
            "metric_params_name": "minimal",
            "auto_curation_rules_name": "none",
            "metric_waveform_params_name": "franklab_cortex_metric_waveforms",
        }
        bypass_id = deterministic_id("curation_evaluation", identity)
        CurationEvaluationSelection().insert1(
            {**identity, "curation_evaluation_id": bypass_id},
            allow_direct_insert=True,
        )
        with pytest.raises(ValueError, match="preview"):
            CurationEvaluation().make_fetch(
                {"curation_evaluation_id": bypass_id}
            )
    finally:
        clear_curations_for(planted_two_unit_sort)


@pytest.mark.slow
@pytest.mark.integration
def test_metric_namespace_matches_curation_units(
    planted_two_unit_sort, curation_evaluation_defaults
):
    """Root, label-only, and merged evaluations all index exactly CurationV2.Unit."""
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.metric_curation import (
        CurationEvaluation,
        CurationEvaluationSelection,
    )
    from spyglass.spikesorting.v2.sorting import Sorting

    sorting_key = dict(planted_two_unit_sort)
    unit_ids = sorted(
        int(u) for u in (Sorting.Unit & sorting_key).fetch("unit_id")
    )
    clear_curations_for(planted_two_unit_sort)
    try:
        root = CurationV2.insert_curation(sorting_key=sorting_key)
        label_only = CurationV2.insert_curation(
            sorting_key,
            parent_curation_id=root["curation_id"],
            labels={unit_ids[0]: ["noise"]},
        )
        merged = CurationV2.create_merged_curation(
            sorting_key,
            merge_groups=[[unit_ids[0], unit_ids[1]]],
            parent_curation_id=root["curation_id"],
        )
        for curation in (root, label_only, merged):
            sel = CurationEvaluationSelection.insert_selection(
                {
                    **curation,
                    "metric_params_name": "minimal",
                    "auto_curation_rules_name": "none",
                }
            )
            CurationEvaluation.populate(sel, reserve_jobs=False)
            metrics = CurationEvaluation.get_metrics(sel)
            expected = {
                int(u) for u in (CurationV2.Unit & curation).fetch("unit_id")
            }
            assert set(int(u) for u in metrics.index) == expected
    finally:
        clear_curations_for(planted_two_unit_sort)


@pytest.mark.slow
@pytest.mark.integration
def test_root_curation_uses_cached_raw_analyzer_fast_path(
    planted_two_unit_sort, curation_evaluation_defaults, monkeypatch
):
    """A root curation evaluates through the DB-free resolved raw-analyzer loader;
    a merged curation builds a temp analyzer instead; neither calls get_analyzer.

    Asserts the committed-state routing: the fast path uses
    ``load_or_rebuild_analyzer_from_resolved`` (cached canonical analyzer), the
    merged path does not, and make_compute never reaches ``Sorting.get_analyzer``
    (a DB read) on either path.
    """
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    import spyglass.spikesorting.v2._sorting_analyzer as sa_mod
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.metric_curation import (
        CurationEvaluation,
        CurationEvaluationSelection,
    )
    from spyglass.spikesorting.v2.sorting import Sorting

    sorting_key = dict(planted_two_unit_sort)
    unit_ids = sorted(
        int(u) for u in (Sorting.Unit & sorting_key).fetch("unit_id")
    )

    calls = {"resolved": 0}
    real_resolved = sa_mod.load_or_rebuild_analyzer_from_resolved

    def _spy_resolved(**kwargs):
        calls["resolved"] += 1
        return real_resolved(**kwargs)

    monkeypatch.setattr(
        sa_mod, "load_or_rebuild_analyzer_from_resolved", _spy_resolved
    )

    def _boom(*args, **kwargs):
        raise AssertionError(
            "make_compute called Sorting.get_analyzer (a DB read)"
        )

    monkeypatch.setattr(Sorting, "get_analyzer", _boom)

    clear_curations_for(planted_two_unit_sort)
    try:
        # Fast path: root curation -> resolved loader, no get_analyzer.
        root = CurationV2.insert_curation(sorting_key=sorting_key)
        root_sel = CurationEvaluationSelection.insert_selection(
            {
                **root,
                "metric_params_name": "minimal",
                "auto_curation_rules_name": "none",
            }
        )
        CurationEvaluation.populate(root_sel, reserve_jobs=False)
        assert calls["resolved"] >= 1

        # Merged path: temp analyzer (build_analyzer), NOT the resolved loader.
        calls["resolved"] = 0
        merged = CurationV2.create_merged_curation(
            sorting_key,
            merge_groups=[[unit_ids[0], unit_ids[1]]],
            parent_curation_id=root["curation_id"],
        )
        merged_sel = CurationEvaluationSelection.insert_selection(
            {
                **merged,
                "metric_params_name": "minimal",
                "auto_curation_rules_name": "none",
            }
        )
        CurationEvaluation.populate(merged_sel, reserve_jobs=False)
        assert calls["resolved"] == 0
    finally:
        clear_curations_for(planted_two_unit_sort)


@pytest.mark.slow
@pytest.mark.integration
def test_final_snr_peak_sign_uses_sorter_polarity(
    planted_two_unit_sort, curation_evaluation_defaults
):
    """make_fetch injects the sorter's resolved peak_sign into snr kwargs.

    Same SIG-2 fix as CurationEvaluation: the planted MS5 sort carries
    ``detect_sign=-1`` -> ``'neg'`` (the regression-pinned value), confirming the
    helper is wired into CurationEvaluation's DB-fetch stage.
    """
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.metric_curation import (
        CurationEvaluation,
        CurationEvaluationSelection,
    )

    sorting_key = dict(planted_two_unit_sort)
    clear_curations_for(planted_two_unit_sort)
    try:
        root = CurationV2.insert_curation(sorting_key=sorting_key)
        sel = CurationEvaluationSelection.insert_selection(
            {
                **root,
                "metric_params_name": "minimal",
                "auto_curation_rules_name": "none",
            }
        )
        fetched = CurationEvaluation().make_fetch(sel)
        assert "snr" in fetched.metric_names
        assert fetched.metric_kwargs["snr"]["peak_sign"] == "neg"
    finally:
        clear_curations_for(planted_two_unit_sort)


@pytest.fixture(scope="package")
def planted_zero_unit_sort(dj_conn):
    """A populated Sorting with ZERO units (an empty planted sorting).

    Mirrors ``planted_two_unit_sort`` but the monkeypatched ``_run_sorter``
    returns an empty ``NumpySorting``, so the Sorting row commits with
    ``n_units=0`` and a root curation has zero units -- the input for the
    zero-unit CurationEvaluation contract.
    """
    from pathlib import Path

    import spikeinterface as si

    from tests.spikesorting.v2._ingest_helpers import (
        _clean_session_v2,
        copy_and_insert_nwb,
    )

    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "mearec_polymer_smoke.nwb"
    )
    if not fixture.exists():
        pytest.skip(f"Fixture {fixture.name} not found.")

    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetection,
        ArtifactDetectionSelection,
    )
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
        SortGroupV2,
    )
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection

    nwb = copy_and_insert_nwb(fixture, dest_name="mearec_zero_unit.nwb")
    session = {"nwb_file_name": nwb}
    _clean_session_v2(session)
    initialize_v2_defaults()
    LabTeam.insert1(
        {"team_name": "v2_test_team", "team_description": "v2 zero-unit"},
        skip_duplicates=True,
    )
    if not (SortGroupV2 & session):
        SortGroupV2.set_group_by_shank(nwb_file_name=nwb)
    sg = int(sorted((SortGroupV2 & session).fetch("sort_group_id"))[0])
    rec_pk = RecordingSelection.insert_selection(
        {
            "nwb_file_name": nwb,
            "sort_group_id": sg,
            "interval_list_name": "raw data valid times",
            "preprocessing_params_name": "default",
            "team_name": "v2_test_team",
        }
    )
    if not (Recording & rec_pk):
        Recording.populate(rec_pk, reserve_jobs=False)
    art_pk = ArtifactDetectionSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "artifact_detection_params_name": "none",
        }
    )
    if not (ArtifactDetection & art_pk):
        ArtifactDetection.populate(art_pk, reserve_jobs=False)
    sort_pk = SortingSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "sorter": "mountainsort5",
            "sorter_params_name": "franklab_30khz_ms5_2026_06",
            "artifact_detection_id": art_pk["artifact_detection_id"],
        }
    )
    (Sorting & sort_pk).super_delete(warn=False)

    def _plant_empty(
        sorter,
        sorter_params,
        recording,
        sorting_id,
        *,
        job_kwargs=None,
        execution_params=None,
    ):
        return si.NumpySorting.from_unit_dict(
            [{}], sampling_frequency=recording.get_sampling_frequency()
        )

    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(Sorting, "_run_sorter", staticmethod(_plant_empty))
        Sorting.populate(sort_pk, reserve_jobs=False)
    finally:
        mp.undo()
    if int((Sorting & sort_pk).fetch1("n_units")) != 0:
        pytest.skip("planted sort did not yield zero units")
    yield sort_pk
    _clean_session_v2(session)


@pytest.mark.slow
@pytest.mark.integration
def test_zero_unit_curation_evaluation_writes_empty_tables(
    planted_zero_unit_sort, curation_evaluation_defaults
):
    """A zero-unit committed curation writes empty metric/label/merge tables."""
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.metric_curation import (
        CurationEvaluation,
        CurationEvaluationSelection,
    )

    sorting_key = dict(planted_zero_unit_sort)
    clear_curations_for(planted_zero_unit_sort)
    try:
        root = CurationV2.insert_curation(sorting_key=sorting_key)
        assert len(CurationV2.Unit & root) == 0
        sel = CurationEvaluationSelection.insert_selection(
            {
                **root,
                "metric_params_name": "minimal",
                "auto_curation_rules_name": "none",
            }
        )
        CurationEvaluation.populate(sel, reserve_jobs=False)
        assert CurationEvaluation & sel
        assert CurationEvaluation.get_metrics(sel).empty
        assert CurationEvaluation.get_labels(sel) == {}
        assert CurationEvaluation.get_suggested_merge_groups(sel) == []
        # The zero-unit artifact is still self-describing: the provenance header
        # is written even though there are no metrics (no analyzer is built, so
        # the source-analyzer manifest is None).
        from spyglass.common.common_nwbfile import AnalysisNwbfile
        from spyglass.spikesorting.v2._nwb_provenance import (
            CURATION_EVALUATION_PROVENANCE,
            read_provenance_values,
        )

        abs_path = AnalysisNwbfile.get_abs_path(
            (CurationEvaluation & sel).fetch1("analysis_file_name")
        )
        prov = read_provenance_values(abs_path, CURATION_EVALUATION_PROVENANCE)
        assert prov["sorting_id"] == str(sorting_key["sorting_id"])
        assert prov["metric_params_name"] == "minimal"
        assert prov["source_analyzer_hashes"] is None
    finally:
        clear_curations_for(planted_zero_unit_sort)


# ---------- CurationEvaluation acceptance helpers ---------------------------


@pytest.mark.slow
@pytest.mark.integration
def test_evaluation_acceptance_creates_committed_child(
    planted_two_unit_sort, curation_evaluation_defaults
):
    """``accept_evaluation_outputs(..., merge_groups=accepted)`` commits a child.

    ``use_evaluation_labels`` makes a committed labels-only child. Neither leaves
    a preview row with unapplied proposed merges.
    """
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.metric_curation import (
        CurationEvaluation,
        CurationEvaluationSelection,
    )
    from spyglass.spikesorting.v2.sorting import Sorting

    sorting_key = dict(planted_two_unit_sort)
    unit_ids = sorted(
        int(u) for u in (Sorting.Unit & sorting_key).fetch("unit_id")
    )
    clear_curations_for(planted_two_unit_sort)
    try:
        root = CurationV2.insert_curation(sorting_key=sorting_key)
        sel = CurationEvaluationSelection.insert_selection(
            {
                **root,
                "metric_params_name": "minimal",
                "auto_curation_rules_name": "none",
            }
        )
        CurationEvaluation.populate(sel, reserve_jobs=False)

        # Accept an explicit merge -> committed merged child (not a preview).
        merged_child = CurationEvaluation().accept_evaluation_outputs(
            sel, merge_groups=[[unit_ids[0], unit_ids[1]]]
        )
        assert CurationV2.is_committed_curation(merged_child)
        assert not CurationV2.has_unapplied_proposed_merges(merged_child)
        assert bool((CurationV2 & merged_child).fetch1("merges_applied"))
        child_units = set(
            int(u) for u in (CurationV2.Unit & merged_child).fetch("unit_id")
        )
        assert child_units == {max(unit_ids) + 1}

        # Labels-only acceptance -> committed child, full unit set preserved.
        labeled_child = CurationEvaluation().use_evaluation_labels(
            sel, labels={unit_ids[0]: ["mua"]}
        )
        assert CurationV2.is_committed_curation(labeled_child)
        assert not bool((CurationV2 & labeled_child).fetch1("merges_applied"))
        labeled_units = set(
            int(u) for u in (CurationV2.Unit & labeled_child).fetch("unit_id")
        )
        assert labeled_units == set(unit_ids)
        labels = {
            (int(r["unit_id"]), r["curation_label"])
            for r in (CurationV2.UnitLabel & labeled_child).fetch(as_dict=True)
        }
        assert (unit_ids[0], "mua") in labels
    finally:
        clear_curations_for(planted_two_unit_sort)


@pytest.mark.slow
@pytest.mark.integration
def test_evaluation_acceptance_requires_explicit_merge_choice(
    planted_two_unit_sort, curation_evaluation_defaults, monkeypatch
):
    """Suggested merges are not silently applied without explicit intent.

    Even when the evaluation HAS a proposed merge,
    ``accept_evaluation_outputs`` applies it only when the caller passes
    ``merge_groups`` or
    ``use_all_suggested_merges=True``; the bare call is labels-only.
    """
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.metric_curation import (
        CurationEvaluation,
        CurationEvaluationSelection,
    )
    from spyglass.spikesorting.v2.sorting import Sorting

    sorting_key = dict(planted_two_unit_sort)
    unit_ids = sorted(
        int(u) for u in (Sorting.Unit & sorting_key).fetch("unit_id")
    )
    clear_curations_for(planted_two_unit_sort)
    try:
        root = CurationV2.insert_curation(sorting_key=sorting_key)
        sel = CurationEvaluationSelection.insert_selection(
            {
                **root,
                "metric_params_name": "minimal",
                "auto_curation_rules_name": "none",
            }
        )
        CurationEvaluation.populate(sel, reserve_jobs=False)

        # Substitute a deterministic proposed merge so the acceptance decision
        # is exercised independently of the auto-merge detector.
        suggestion = [[unit_ids[0], unit_ids[1]]]
        monkeypatch.setattr(
            CurationEvaluation,
            "get_suggested_merge_groups",
            classmethod(lambda cls, key: suggestion),
        )

        # Bare acceptance: the suggestion is NOT applied (labels-only child).
        default_child = CurationEvaluation().accept_evaluation_outputs(sel)
        assert not bool((CurationV2 & default_child).fetch1("merges_applied"))
        assert set(
            int(u) for u in (CurationV2.Unit & default_child).fetch("unit_id")
        ) == set(unit_ids)

        # Opt in: use_all_suggested_merges applies the suggestion.
        all_child = CurationEvaluation().accept_evaluation_outputs(
            sel, use_all_suggested_merges=True
        )
        assert bool((CurationV2 & all_child).fetch1("merges_applied"))
        assert set(
            int(u) for u in (CurationV2.Unit & all_child).fetch("unit_id")
        ) == {max(unit_ids) + 1}

        # Passing both is a contradiction.
        with pytest.raises(ValueError, match="not both"):
            CurationEvaluation().accept_evaluation_outputs(
                sel,
                merge_groups=suggestion,
                use_all_suggested_merges=True,
            )
    finally:
        clear_curations_for(planted_two_unit_sort)


@pytest.mark.slow
@pytest.mark.integration
def test_evaluation_child_source_is_curation_evaluation(
    planted_two_unit_sort, curation_evaluation_defaults
):
    """Accepted evaluation children carry the curation_evaluation provenance."""
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.metric_curation import (
        CurationEvaluation,
        CurationEvaluationSelection,
    )

    sorting_key = dict(planted_two_unit_sort)
    clear_curations_for(planted_two_unit_sort)
    try:
        root = CurationV2.insert_curation(sorting_key=sorting_key)
        sel = CurationEvaluationSelection.insert_selection(
            {
                **root,
                "metric_params_name": "minimal",
                "auto_curation_rules_name": "none",
            }
        )
        CurationEvaluation.populate(sel, reserve_jobs=False)

        child = CurationEvaluation().use_evaluation_labels(sel)
        source = (CurationV2 & child).fetch1("curation_source")
        assert source == "curation_evaluation"
        assert source != "analyzer_curation"
    finally:
        clear_curations_for(planted_two_unit_sort)


@pytest.mark.slow
@pytest.mark.integration
def test_evaluation_preview_merges_is_a_rejected_draft(
    planted_two_unit_sort, curation_evaluation_defaults
):
    """``preview_merges`` produces a DRAFT (preview) child, not a
    committed one, and that draft is then rejected at the evaluation boundary.
    """
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.metric_curation import (
        CurationEvaluation,
        CurationEvaluationSelection,
    )
    from spyglass.spikesorting.v2.sorting import Sorting

    sorting_key = dict(planted_two_unit_sort)
    unit_ids = sorted(
        int(u) for u in (Sorting.Unit & sorting_key).fetch("unit_id")
    )
    clear_curations_for(planted_two_unit_sort)
    try:
        root = CurationV2.insert_curation(sorting_key=sorting_key)
        sel = CurationEvaluationSelection.insert_selection(
            {
                **root,
                "metric_params_name": "minimal",
                "auto_curation_rules_name": "none",
            }
        )
        CurationEvaluation.populate(sel, reserve_jobs=False)

        preview = CurationEvaluation().preview_merges(
            sel, merge_groups=[[unit_ids[0], unit_ids[1]]]
        )
        # It is a draft, not committed: every unit preserved + proposed merge.
        assert CurationV2.is_committed_curation(preview) is False
        assert CurationV2.has_unapplied_proposed_merges(preview) is True
        assert (CurationV2 & preview).fetch1("curation_source") == (
            "curation_evaluation"
        )
        # And evaluating that draft is rejected (would score unmerged units).
        with pytest.raises(ValueError, match="preview"):
            CurationEvaluationSelection.insert_selection(
                {
                    **preview,
                    "metric_params_name": "minimal",
                    "auto_curation_rules_name": "none",
                }
            )
    finally:
        clear_curations_for(planted_two_unit_sort)


def test_analyzer_curation_table_removed():
    """AnalyzerCuration / AnalyzerCurationSelection are deleted (no shim).

    CurationEvaluation fully replaces the legacy raw-sort auto-curation table;
    importing the old names must fail so no caller can keep using them.
    """
    import spyglass.spikesorting.v2.metric_curation as mc

    assert not hasattr(mc, "AnalyzerCuration")
    assert not hasattr(mc, "AnalyzerCurationSelection")
    with pytest.raises(ImportError):
        from spyglass.spikesorting.v2.metric_curation import (  # noqa: F401
            AnalyzerCuration,
        )


# ---------- review round 2: routing / label-state / namespace guards --------


@pytest.mark.slow
@pytest.mark.integration
def test_workflow_evaluate_accept_merge_reevaluate_use_evaluation_labels_final_child(
    planted_two_unit_sort, curation_evaluation_defaults
):
    """Common curation workflow ends on final metrics for the final child.

    Evaluate the root, accept an explicit merge with
    ``accept_evaluation_outputs``, re-evaluate the committed merged child,
    accept the final label verdict with
    ``use_evaluation_labels``, then re-evaluate that final child. The label-only final
    child has ``merges_applied=False`` but a merged unit namespace, so routing
    must key on the actual curation unit set rather than blindly reusing the raw
    analyzer whenever ``merges_applied`` is false.
    """
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.metric_curation import (
        CurationEvaluation,
        CurationEvaluationSelection,
    )
    from spyglass.spikesorting.v2.sorting import Sorting

    sorting_key = dict(planted_two_unit_sort)
    unit_ids = sorted(
        int(u) for u in (Sorting.Unit & sorting_key).fetch("unit_id")
    )
    merged_id = max(unit_ids) + 1
    metric_params = _ensure_counts_metric_params()
    clear_curations_for(planted_two_unit_sort)

    def _eval(curation_key):
        sel = CurationEvaluationSelection.insert_selection(
            {
                **curation_key,
                "metric_params_name": metric_params,
                "auto_curation_rules_name": "none",
            }
        )
        CurationEvaluation.populate(sel, reserve_jobs=False)
        return sel

    try:
        root = CurationV2.insert_curation(sorting_key=sorting_key)
        # Accept a merge into a committed merged child; evaluate it.
        merged = CurationEvaluation().accept_evaluation_outputs(
            _eval(root), merge_groups=[[unit_ids[0], unit_ids[1]]]
        )
        assert CurationV2.is_committed_curation(merged)
        assert (CurationV2 & merged).fetch1("curation_source") == (
            "curation_evaluation"
        )
        assert set(
            int(u) for u in (CurationV2.Unit & merged).fetch("unit_id")
        ) == {merged_id}
        eval_merged = _eval(merged)
        merged_metrics = CurationEvaluation.get_metrics(eval_merged)
        assert set(merged_metrics.index) == {merged_id}
        assert int(merged_metrics.loc[merged_id, "num_spikes"]) == int(
            (CurationV2.Unit & merged & {"unit_id": merged_id}).fetch1(
                "n_spikes"
            )
        )

        # Accept labels into a label-only grandchild (merges_applied=False, but
        # its namespace is the MERGED set {merged_id}).
        final_child = CurationEvaluation().use_evaluation_labels(
            eval_merged, labels={merged_id: ["mua"]}
        )
        assert CurationV2.is_committed_curation(final_child)
        assert (CurationV2 & final_child).fetch1("curation_source") == (
            "curation_evaluation"
        )
        assert not bool((CurationV2 & final_child).fetch1("merges_applied"))
        assert set(
            int(u) for u in (CurationV2.Unit & final_child).fetch("unit_id")
        ) == {merged_id}

        labels = {
            (int(r["unit_id"]), r["curation_label"])
            for r in (CurationV2.UnitLabel & final_child).fetch(as_dict=True)
        }
        assert labels == {(merged_id, "mua")}

        final_eval = _eval(final_child)
        final_metrics = CurationEvaluation.get_metrics(final_eval)
        assert set(final_metrics.index) == {merged_id}
        assert int(final_metrics.loc[merged_id, "num_spikes"]) == int(
            (CurationV2.Unit & final_child & {"unit_id": merged_id}).fetch1(
                "n_spikes"
            )
        )
    finally:
        clear_curations_for(planted_two_unit_sort)


@pytest.mark.slow
@pytest.mark.integration
def test_label_actions_clear_stale_auto_and_preserve_manual_labels(
    planted_two_unit_sort, curation_evaluation_defaults
):
    """The two label-acceptance actions encode distinct user intent.

    ``use_evaluation_labels`` writes the evaluation verdict as the full label state, so
    a prior auto label the new evaluation does not propose is cleared.
    ``overlay_evaluation_labels`` preserves the user's current/manual labels
    and overlays only the evaluation's proposals.
    """
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.metric_curation import (
        CurationEvaluation,
        CurationEvaluationSelection,
    )
    from spyglass.spikesorting.v2.sorting import Sorting

    sorting_key = dict(planted_two_unit_sort)
    unit_ids = sorted(
        int(u) for u in (Sorting.Unit & sorting_key).fetch("unit_id")
    )
    clear_curations_for(planted_two_unit_sort)
    try:
        # A curation with a prior reject on unit 0.
        root = CurationV2.insert_curation(
            sorting_key=sorting_key, labels={unit_ids[0]: ["reject"]}
        )
        # Evaluate it with the inert "none" rules -> NO proposed labels.
        sel = CurationEvaluationSelection.insert_selection(
            {
                **root,
                "metric_params_name": "minimal",
                "auto_curation_rules_name": "none",
            }
        )
        CurationEvaluation.populate(sel, reserve_jobs=False)
        # use_evaluation_labels writes the empty auto verdict, clearing the stale
        # reject -- so the unit is not silently excluded downstream.
        child = CurationEvaluation().use_evaluation_labels(sel)
        child_labels = {
            (int(r["unit_id"]), r["curation_label"])
            for r in (CurationV2.UnitLabel & child).fetch(as_dict=True)
        }
        assert child_labels == set(), child_labels
        # overlay_evaluation_labels keeps the prior reject.
        inherited = CurationEvaluation().overlay_evaluation_labels(sel)
        inh_labels = {
            (int(r["unit_id"]), r["curation_label"])
            for r in (CurationV2.UnitLabel & inherited).fetch(as_dict=True)
        }
        assert (unit_ids[0], "reject") in inh_labels
    finally:
        clear_curations_for(planted_two_unit_sort)


@pytest.mark.slow
@pytest.mark.integration
def test_acceptance_merge_drops_absorbed_contributor_label(
    planted_two_unit_sort, curation_evaluation_defaults
):
    """When accepting a merge, an absorbed contributor label is dropped.

    It does not silently attach to the merged unit -- re-evaluate the merged
    child to label it. The drop is warned, not silent.
    """
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.metric_curation import (
        CurationEvaluation,
        CurationEvaluationSelection,
    )
    from spyglass.spikesorting.v2.sorting import Sorting

    sorting_key = dict(planted_two_unit_sort)
    unit_ids = sorted(
        int(u) for u in (Sorting.Unit & sorting_key).fetch("unit_id")
    )
    merged_id = max(unit_ids) + 1
    clear_curations_for(planted_two_unit_sort)
    try:
        root = CurationV2.insert_curation(sorting_key=sorting_key)
        sel = CurationEvaluationSelection.insert_selection(
            {
                **root,
                "metric_params_name": "minimal",
                "auto_curation_rules_name": "none",
            }
        )
        CurationEvaluation.populate(sel, reserve_jobs=False)
        # A label on unit 0, which the accepted merge absorbs -> dropped; the
        # merged unit is unlabeled (label it by re-evaluating the merged child).
        # unit 0 is absorbed by the merge -> its label warn-drops (not a
        # truly-stray key, so no permissive flag needed).
        child = CurationEvaluation().accept_evaluation_outputs(
            sel,
            merge_groups=[[unit_ids[0], unit_ids[1]]],
            labels={unit_ids[0]: ["noise"]},
        )
        assert set(
            int(u) for u in (CurationV2.Unit & child).fetch("unit_id")
        ) == {merged_id}
        assert len(CurationV2.UnitLabel & child) == 0
    finally:
        clear_curations_for(planted_two_unit_sort)


@pytest.mark.slow
@pytest.mark.integration
def test_analyzer_backed_helpers_reject_merged_curation(
    planted_two_unit_sort, curation_evaluation_defaults
):
    """The raw-analyzer notebook/SI helpers refuse a merged curation rather
    than silently rendering it in the wrong (raw) unit namespace.
    """
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    from spyglass.spikesorting.v2 import visualization as ssviz
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.metric_curation import (
        CurationEvaluation,
        CurationEvaluationSelection,
    )
    from spyglass.spikesorting.v2.sorting import Sorting

    sorting_key = dict(planted_two_unit_sort)
    unit_ids = sorted(
        int(u) for u in (Sorting.Unit & sorting_key).fetch("unit_id")
    )
    clear_curations_for(planted_two_unit_sort)
    try:
        root = CurationV2.insert_curation(sorting_key=sorting_key)
        merged = CurationV2.create_merged_curation(
            sorting_key,
            merge_groups=[[unit_ids[0], unit_ids[1]]],
            parent_curation_id=root["curation_id"],
        )
        sel = CurationEvaluationSelection.insert_selection(
            {
                **merged,
                "metric_params_name": "minimal",
                "auto_curation_rules_name": "none",
            }
        )
        # No populate needed: the namespace guard fires before any analyzer load.
        with pytest.raises(ValueError, match="namespace"):
            CurationEvaluation().get_waveforms(sel)
        with pytest.raises(ValueError, match="namespace"):
            ssviz.plot_si_quality_metrics(sel)
    finally:
        clear_curations_for(planted_two_unit_sort)


@pytest.mark.slow
@pytest.mark.integration
def test_acceptance_rejects_caller_singleton_merge_group(
    planted_two_unit_sort, curation_evaluation_defaults
):
    """A caller-supplied singleton merge group is rejected (the >=2-member typo
    guard), not silently degraded into a labels-only child.
    """
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.metric_curation import (
        CurationEvaluation,
        CurationEvaluationSelection,
    )
    from spyglass.spikesorting.v2.sorting import Sorting

    sorting_key = dict(planted_two_unit_sort)
    unit_ids = sorted(
        int(u) for u in (Sorting.Unit & sorting_key).fetch("unit_id")
    )
    clear_curations_for(planted_two_unit_sort)
    try:
        root = CurationV2.insert_curation(sorting_key=sorting_key)
        sel = CurationEvaluationSelection.insert_selection(
            {
                **root,
                "metric_params_name": "minimal",
                "auto_curation_rules_name": "none",
            }
        )
        CurationEvaluation.populate(sel, reserve_jobs=False)

        # labels={} so the singleton reaches insert_curation's >=2-member guard
        # (the evaluation is populated; get_labels is skipped for the merge).
        with pytest.raises(ValueError, match="at least 2 units"):
            CurationEvaluation().accept_evaluation_outputs(
                sel, merge_groups=[[unit_ids[0]]], labels={}
            )
    finally:
        clear_curations_for(planted_two_unit_sort)


@pytest.mark.slow
@pytest.mark.integration
def test_preview_merges_requires_a_merge(
    planted_two_unit_sort, curation_evaluation_defaults
):
    """A preview with no merge is rejected, not silently committed.

    ``preview_merges`` is the explicit opt-in for drafting an
    UNAPPLIED merge for review. Called with no merge it would otherwise produce
    a normal committed labels-only child (apply_merge=False, no proposed
    merge), contradicting the "preview/draft" contract -- so it raises and
    points the caller at use_evaluation_labels/overlay_evaluation_labels.
    """
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.metric_curation import (
        CurationEvaluation,
        CurationEvaluationSelection,
    )

    sorting_key = dict(planted_two_unit_sort)
    clear_curations_for(planted_two_unit_sort)
    try:
        root = CurationV2.insert_curation(sorting_key=sorting_key)
        sel = CurationEvaluationSelection.insert_selection(
            {
                **root,
                "metric_params_name": "minimal",
                "auto_curation_rules_name": "none",
            }
        )
        CurationEvaluation.populate(sel, reserve_jobs=False)
        # No merge_groups and use_all_suggested_merges defaults False -> raises
        # before creating any child (no committed labels-only fallback).
        with pytest.raises(ValueError, match="needs at least one merge"):
            CurationEvaluation().preview_merges(sel)
    finally:
        clear_curations_for(planted_two_unit_sort)


@pytest.mark.slow
@pytest.mark.integration
def test_preview_merges_action_drafts_merge_without_fetching_labels(
    planted_two_unit_sort, curation_evaluation_defaults, monkeypatch
):
    """``preview_merges`` is the review action and does not apply eval labels."""
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.metric_curation import (
        CurationEvaluation,
        CurationEvaluationSelection,
    )
    from spyglass.spikesorting.v2.sorting import Sorting

    def _unexpected_get_labels(cls, key):
        raise AssertionError("preview_merges should not fetch eval labels")

    monkeypatch.setattr(
        CurationEvaluation, "get_labels", classmethod(_unexpected_get_labels)
    )
    sorting_key = dict(planted_two_unit_sort)
    unit_ids = sorted(
        int(u) for u in (Sorting.Unit & sorting_key).fetch("unit_id")
    )
    clear_curations_for(planted_two_unit_sort)
    try:
        root = CurationV2.insert_curation(sorting_key=sorting_key)
        sel = CurationEvaluationSelection.insert_selection(
            {
                **root,
                "metric_params_name": "minimal",
                "auto_curation_rules_name": "none",
            }
        )
        CurationEvaluation.populate(sel, reserve_jobs=False)

        preview = CurationEvaluation().preview_merges(
            sel, merge_groups=[[unit_ids[0], unit_ids[1]]]
        )
        assert CurationV2.is_committed_curation(preview) is False
        assert CurationV2.has_unapplied_proposed_merges(preview) is True
        assert (CurationV2 & preview).fetch1("curation_source") == (
            "curation_evaluation"
        )
    finally:
        clear_curations_for(planted_two_unit_sort)


@pytest.mark.slow
@pytest.mark.integration
def test_accept_merges_action_commits_without_fetching_labels(
    planted_two_unit_sort, curation_evaluation_defaults, monkeypatch
):
    """``accept_merges`` commits selected groups and leaves labels inherited."""
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.metric_curation import (
        CurationEvaluation,
        CurationEvaluationSelection,
    )
    from spyglass.spikesorting.v2.sorting import Sorting

    def _unexpected_get_labels(cls, key):
        raise AssertionError("accept_merges should not fetch eval labels")

    monkeypatch.setattr(
        CurationEvaluation, "get_labels", classmethod(_unexpected_get_labels)
    )
    sorting_key = dict(planted_two_unit_sort)
    unit_ids = sorted(
        int(u) for u in (Sorting.Unit & sorting_key).fetch("unit_id")
    )
    clear_curations_for(planted_two_unit_sort)
    try:
        root = CurationV2.insert_curation(
            sorting_key=sorting_key, labels={unit_ids[0]: ["mua"]}
        )
        sel = CurationEvaluationSelection.insert_selection(
            {
                **root,
                "metric_params_name": "minimal",
                "auto_curation_rules_name": "none",
            }
        )
        CurationEvaluation.populate(sel, reserve_jobs=False)

        child = CurationEvaluation().accept_merges(
            sel, merge_groups=[[unit_ids[0], unit_ids[1]]]
        )
        assert CurationV2.is_committed_curation(child)
        assert bool((CurationV2 & child).fetch1("merges_applied")) is True
        assert (CurationV2 & child).fetch1("curation_source") == (
            "curation_evaluation"
        )
        merged_id = max(unit_ids) + 1
        labels = {
            (int(r["unit_id"]), r["curation_label"])
            for r in (CurationV2.UnitLabel & child).fetch(as_dict=True)
        }
        assert labels == {(merged_id, "mua")}
    finally:
        clear_curations_for(planted_two_unit_sort)


@pytest.mark.slow
@pytest.mark.integration
def test_accept_all_suggested_merges_action_uses_persisted_suggestions(
    planted_two_unit_sort, curation_evaluation_defaults, monkeypatch
):
    """``accept_all_suggested_merges`` is explicit about using suggestions."""
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.metric_curation import (
        CurationEvaluation,
        CurationEvaluationSelection,
    )
    from spyglass.spikesorting.v2.sorting import Sorting

    sorting_key = dict(planted_two_unit_sort)
    unit_ids = sorted(
        int(u) for u in (Sorting.Unit & sorting_key).fetch("unit_id")
    )
    suggestion = [[unit_ids[0], unit_ids[1]]]
    monkeypatch.setattr(
        CurationEvaluation,
        "get_suggested_merge_groups",
        classmethod(lambda cls, key: suggestion),
    )
    clear_curations_for(planted_two_unit_sort)
    try:
        root = CurationV2.insert_curation(sorting_key=sorting_key)
        sel = CurationEvaluationSelection.insert_selection(
            {
                **root,
                "metric_params_name": "minimal",
                "auto_curation_rules_name": "none",
            }
        )
        CurationEvaluation.populate(sel, reserve_jobs=False)

        child = CurationEvaluation().accept_all_suggested_merges(sel)
        assert CurationV2.is_committed_curation(child)
        assert bool((CurationV2 & child).fetch1("merges_applied")) is True
        assert set(
            int(u) for u in (CurationV2.Unit & child).fetch("unit_id")
        ) == {max(unit_ids) + 1}
    finally:
        clear_curations_for(planted_two_unit_sort)


@pytest.mark.slow
@pytest.mark.integration
def test_accept_merge_actions_require_a_real_merge(
    planted_two_unit_sort, curation_evaluation_defaults, monkeypatch
):
    """Friendly merge actions raise rather than making labels-only children."""
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.metric_curation import (
        CurationEvaluation,
        CurationEvaluationSelection,
    )

    monkeypatch.setattr(
        CurationEvaluation,
        "get_suggested_merge_groups",
        classmethod(lambda cls, key: []),
    )
    sorting_key = dict(planted_two_unit_sort)
    clear_curations_for(planted_two_unit_sort)
    try:
        root = CurationV2.insert_curation(sorting_key=sorting_key)
        sel = CurationEvaluationSelection.insert_selection(
            {
                **root,
                "metric_params_name": "minimal",
                "auto_curation_rules_name": "none",
            }
        )
        # Populated so the actions reach the "needs a real merge" guard rather
        # than the populated-evaluation guard (get_suggested_merge_groups is monkeypatched
        # empty above to drive the no-merge path).
        CurationEvaluation.populate(sel, reserve_jobs=False)

        with pytest.raises(ValueError, match="accept_merges"):
            CurationEvaluation().accept_merges(sel, merge_groups=[])
        with pytest.raises(ValueError, match="accept_all_suggested_merges"):
            CurationEvaluation().accept_all_suggested_merges(sel)
        with pytest.raises(ValueError, match="preview_merges"):
            CurationEvaluation().preview_merges(sel)
    finally:
        clear_curations_for(planted_two_unit_sort)


@pytest.mark.slow
@pytest.mark.integration
def test_label_action_names_replace_and_overlay(
    planted_two_unit_sort, curation_evaluation_defaults
):
    """The intuitive label action names preserve replace/overlay semantics."""
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.metric_curation import (
        CurationEvaluation,
        CurationEvaluationSelection,
    )
    from spyglass.spikesorting.v2.sorting import Sorting

    sorting_key = dict(planted_two_unit_sort)
    unit_ids = sorted(
        int(u) for u in (Sorting.Unit & sorting_key).fetch("unit_id")
    )
    clear_curations_for(planted_two_unit_sort)
    try:
        root = CurationV2.insert_curation(
            sorting_key=sorting_key, labels={unit_ids[0]: ["reject"]}
        )
        sel = CurationEvaluationSelection.insert_selection(
            {
                **root,
                "metric_params_name": "minimal",
                "auto_curation_rules_name": "none",
            }
        )
        CurationEvaluation.populate(sel, reserve_jobs=False)

        replacement = CurationEvaluation().use_evaluation_labels(sel, labels={})
        replacement_labels = {
            (int(r["unit_id"]), r["curation_label"])
            for r in (CurationV2.UnitLabel & replacement).fetch(as_dict=True)
        }
        assert replacement_labels == set()

        overlay = CurationEvaluation().overlay_evaluation_labels(sel, labels={})
        overlay_label_rows = {
            (int(r["unit_id"]), r["curation_label"])
            for r in (CurationV2.UnitLabel & overlay).fetch(as_dict=True)
        }
        assert overlay_label_rows == {(unit_ids[0], "reject")}
    finally:
        clear_curations_for(planted_two_unit_sort)


@pytest.mark.slow
@pytest.mark.integration
def test_acceptance_requires_populated_evaluation(
    planted_two_unit_sort, curation_evaluation_defaults
):
    """Every acceptance entry point rejects a SELECTED-but-not-populated
    evaluation: minting a ``curation_source='curation_evaluation'`` child from a
    bare selection (no computed proposals) would be a provenance lie. The
    ``use_all_suggested_merges`` path raises the SAME friendly error, not an
    opaque NWB fetch -- it is guarded before resolving persisted suggestions.
    """
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.metric_curation import (
        CurationEvaluation,
        CurationEvaluationSelection,
    )
    from spyglass.spikesorting.v2.sorting import Sorting

    sorting_key = dict(planted_two_unit_sort)
    unit_ids = sorted(
        int(u) for u in (Sorting.Unit & sorting_key).fetch("unit_id")
    )
    clear_curations_for(planted_two_unit_sort)
    try:
        root = CurationV2.insert_curation(sorting_key=sorting_key)
        sel = CurationEvaluationSelection.insert_selection(
            {
                **root,
                "metric_params_name": "minimal",
                "auto_curation_rules_name": "none",
            }
        )
        assert not (CurationEvaluation & sel)  # selected, NOT populated
        ev = CurationEvaluation()
        merge = [[unit_ids[0], unit_ids[1]]]
        # Every public acceptance entry point -- including the all-suggested
        # path that reads the evaluation NWB -- raises the friendly guard.
        for call in (
            lambda: ev.accept_evaluation_outputs(sel, labels={}),
            lambda: ev.use_evaluation_labels(sel, labels={}),
            lambda: ev.overlay_evaluation_labels(sel, labels={}),
            lambda: ev.accept_merges(sel, merge_groups=merge),
            lambda: ev.accept_all_suggested_merges(sel),
            lambda: ev.preview_merges(sel, merge_groups=merge),
        ):
            with pytest.raises(ValueError, match="POPULATED evaluation"):
                call()
    finally:
        clear_curations_for(planted_two_unit_sort)

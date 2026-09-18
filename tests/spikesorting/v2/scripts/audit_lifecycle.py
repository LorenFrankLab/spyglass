"""Opt-in lifecycle acceptance probes; run explicitly with pytest.

Uses the v2 test database and MEArec tetrode fixture. Position is a controlled
input; the sorter, curation, population selection, detector, and serialization
are real. This directory does not inherit decoding's mock NetCDF writer.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import datajoint as dj
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def workflow(dj_conn, tmp_path_factory):
    from spyglass.common import LabTeam
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.curation_api import CurationRef
    from spyglass.spikesorting.v2.pipeline import run_v2_pipeline
    from spyglass.spikesorting.v2.recording import SortGroupV2
    from spyglass.spikesorting.v2.sorting import Sorting
    from tests.spikesorting.v2._ingest_helpers import copy_and_insert_nwb

    path = (
        Path(__file__).resolve().parents[1] / "fixtures/mearec_tetrode_60s.nwb"
    )
    assert path.exists(), "This acceptance probe requires the tetrode fixture"
    name = copy_and_insert_nwb(path, dest_name="lifecycle_tetrode.nwb")
    initialize_v2_defaults()
    LabTeam.insert1({"team_name": "lifecycle_audit"}, skip_duplicates=True)
    SortGroupV2.set_group_by_shank(nwb_file_name=name)
    group = min((SortGroupV2 & {"nwb_file_name": name}).fetch("sort_group_id"))
    run = run_v2_pipeline(
        nwb_file_name=name,
        sort_group_id=int(group),
        interval_list_name="raw data valid times",
        team_name="lifecycle_audit",
        pipeline_preset="franklab_tetrode_hippocampus_30khz_ms5_2026_06",
        manual_excluded_times=[[20.0, 21.0]],
        require_units=True,
    )
    root = run.root_curation
    ids = sorted(map(int, (Sorting.Unit & root.as_key()).fetch("unit_id")))
    assert len(ids) >= 2, "A real merge needs at least two sorted units"
    child = CurationRef.from_key(
        CurationV2.insert_curation(
            {"sorting_id": root.sorting_id},
            parent_curation_id=root.curation_id,
            labels={u: ["accept"] for u in ids},
            merge_groups=[ids[:2]],
            apply_merge=True,
            description="Lifecycle audit: controlled merge, not a biological judgment",
        )
    )
    # Verify and label the resulting unit namespace after the merge, including
    # the newly allocated merged unit; contributor IDs are no longer units.
    merged_ids = (CurationV2.Unit & child.as_key()).fetch("unit_id")
    child = CurationRef.from_key(
        CurationV2.insert_curation(
            {"sorting_id": root.sorting_id},
            parent_curation_id=child.curation_id,
            labels={int(u): ["accept"] for u in merged_ids},
            description="Lifecycle audit: verified merged population",
        )
    )
    out = tmp_path_factory.mktemp("lifecycle")
    config_file = out / "db.json"
    dj.config.save(str(config_file))
    config_file.chmod(0o600)
    yield {
        "run": run,
        "root": root,
        "child": child,
        "name": name,
        "unit_ids": ids,
        "out": out,
        "config": config_file,
    }
    config_file.unlink(missing_ok=True)


@pytest.mark.parametrize("estimate", [False, True])
def test_real_decoder_round_trip(workflow, monkeypatch, estimate):
    import h5py
    import xarray as xr
    from non_local_detector import ContFragSortedSpikesClassifier
    from non_local_detector.environment import Environment
    from non_local_detector.models.base import SortedSpikesDetector

    from spyglass.common import IntervalList
    from spyglass.decoding.v1.core import DecodingParameters, PositionGroup
    from spyglass.decoding.v1.sorted_spikes import (
        SortedSpikesDecodingSelection,
        SortedSpikesDecodingV1,
    )
    from spyglass.spikesorting.analysis.v1 import group as group_module
    from spyglass.spikesorting.v2.analysis_selection import (
        select_units_for_analysis,
    )

    monkeypatch.setattr(group_module, "test_mode", False)
    receipt = select_units_for_analysis(workflow["child"])
    spikes, ids = receipt.fetch_spike_data(return_unit_ids=True)
    assert len(spikes) == len(workflow["unit_ids"]) - 1
    observation = receipt.observation
    assert not observation.contains(np.array([20.5])).item()
    name = workflow["name"]
    time = np.arange(0.1, 59.9, 0.05)
    position = pd.DataFrame(
        {"position_x": 50 + 40 * np.sin(time / 3)}, index=time
    )
    pos_key = {
        "nwb_file_name": name,
        "position_group_name": "controlled_position",
    }
    PositionGroup.insert1(
        {**pos_key, "position_variables": ["position_x"]}, skip_duplicates=True
    )
    # This is the only substituted data provider. All mask construction, model
    # fitting, prediction, database insertion, and file I/O use production code.
    monkeypatch.setattr(
        PositionGroup,
        "fetch_position_info",
        lambda *a, **k: (position.copy(), ["position_x"]),
    )
    interval = {
        "nwb_file_name": name,
        "interval_list_name": "lifecycle_analysis",
    }
    IntervalList.insert1(
        {**interval, "valid_times": np.array([[0.1, 59.9]])},
        skip_duplicates=True,
    )
    model = ContFragSortedSpikesClassifier(
        sampling_frequency=20,
        environments=Environment(place_bin_size=10),
        sorted_spikes_algorithm_params={
            "position_std": 8.0,
            "block_size": 1000,
        },
    )
    params_name = f"lifecycle_real_{estimate}"
    kwargs = {
        "is_training": time < 40,
        "is_missing": (time >= 30) & (time < 30.2),
    }
    if estimate:
        kwargs["max_iter"] = 2
    DecodingParameters.insert1(
        {
            "decoding_param_name": params_name,
            "decoding_params": model,
            "decoding_kwargs": kwargs,
        },
        skip_duplicates=True,
    )
    key = {
        **receipt.groups[0].group_key,
        **pos_key,
        "decoding_param_name": params_name,
        "encoding_interval": interval["interval_list_name"],
        "decoding_interval": interval["interval_list_name"],
        "estimate_decoding_params": estimate,
    }
    SortedSpikesDecodingSelection.insert1(key, skip_duplicates=True)
    SortedSpikesDecodingV1.populate(key)
    table = SortedSpikesDecodingV1 & key
    result = table.fetch_results()
    restored = table.fetch_model()
    assert isinstance(restored, SortedSpikesDetector)
    observed = observation.contains(result.time.values)
    if estimate:
        assert np.all(result.interval_labels.values[~observed] == -1)
    else:
        assert observed.all()
    posterior = result.acausal_posterior
    assert np.isfinite(posterior.values).any()
    np.testing.assert_allclose(posterior.sum("state_bins"), 1, atol=1e-5)
    np.testing.assert_array_equal(
        json.loads(result.attrs["spyglass_observation_intervals"]),
        observation.intervals,
    )
    paths = table.fetch1("results_path", "classifier_path")
    assert h5py.is_hdf5(paths[0]), "Result must be real NetCDF/HDF5, not pickle"
    fresh = SortedSpikesDetector.load_results(paths[0])
    xr.testing.assert_identical(result, fresh)
    # Check model usability after deserialization, not merely its Python type.
    t = time[(time > 45) & (time < 46)]
    prediction = restored.predict(
        position_time=t,
        position=position.loc[t].values,
        spike_times=[s[observation.contains(s)] for s in spikes],
        time=t,
    )
    np.testing.assert_allclose(
        prediction.acausal_posterior.sum("state_bins"), 1, atol=1e-5
    )
    print(
        "DECODER",
        json.dumps(
            {
                "estimate": estimate,
                "units": len(ids),
                "samples": len(result.time),
                "masked_samples": int((~observed).sum()),
                "result_bytes": Path(paths[0]).stat().st_size,
            }
        ),
    )


def test_two_browser_drafts_preserve_both_edits(
    planted_two_unit_sort, curation_evaluation_defaults, tmp_path
):
    from playwright.sync_api import sync_playwright

    from spyglass.spikesorting.v2._review_delivery import stop_review_servers
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.curation_api import CurationRef
    from spyglass.spikesorting.v2.review_profile import CurationReviewProfile
    from spyglass.spikesorting.v2.sorting import Sorting
    from tests.spikesorting.v2 import _browser_review as ui

    CurationReviewProfile.insert1(
        {
            "review_profile_name": "lifecycle_review",
            "metric_params_name": "minimal",
            "auto_curation_rules_name": "none",
            "displayed_unit_properties": ["snr"],
            "label_options": ["accept", "noise"],
            "label_import_mode": "replace",
        },
        skip_duplicates=True,
    )
    root = CurationRef.from_key(
        CurationV2.insert_curation(planted_two_unit_sort, reuse_existing=True)
    )
    review = root.start_review("lifecycle_review")
    a, b = sorted(
        map(int, (Sorting.Unit & planted_two_unit_sort).fetch("unit_id"))
    )
    url = review.open(open_browser=False)
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            contexts = [
                browser.new_context(viewport={"width": 1280, "height": 720})
                for _ in range(2)
            ]
            pages = [context.new_page() for context in contexts]
            for page in pages:
                page.goto(url)
                ui.start_curating(page)
            ui.select_units(pages[0], a)
            ui.set_label(pages[0], "accept", True)
            assert ui.save_annotations(pages[0]) in (200, 201)
            before = review.preview_import()
            ui.select_units(pages[1], b)
            ui.set_label(pages[1], "noise", True)
            status = ui.save_annotations(pages[1])
            after = review.preview_import()
            evidence = {
                "second_save_status": status,
                "before": dict(before.labels_after),
                "after": dict(after.labels_after),
            }
            (tmp_path / "two-drafts.json").write_text(
                json.dumps(evidence, default=list, indent=2)
            )
            print("TWO_DRAFTS", json.dumps(evidence, default=list))
            for context in contexts:
                context.close()
            browser.close()
            assert status == 409 or after.labels_after.get(a) == ("accept",), (
                "A stale second browser silently overwrote the first scientist's saved label"
            )
    finally:
        stop_review_servers()


def test_masked_sort_review(workflow):
    from spyglass.spikesorting.v2.review_profile import FRANKLAB_REVIEW_PROFILE

    workflow["root"].start_review(FRANKLAB_REVIEW_PROFILE)


def test_fresh_process_rebuild_preserves_population(workflow, monkeypatch):
    from spyglass.spikesorting.analysis.v1 import group as group_module
    from spyglass.spikesorting.v2._analyzer_cache import remove_analyzer_cache
    from spyglass.spikesorting.v2.analysis_selection import (
        select_units_for_analysis,
    )

    monkeypatch.setattr(group_module, "test_mode", False)
    receipt = select_units_for_analysis(workflow["child"])
    before, ids = receipt.fetch_spike_data(return_unit_ids=True)
    ref = workflow["child"]
    remove_analyzer_cache(ref.sorting_id)
    payload = workflow["out"] / "reopen.json"
    code = """
import json, sys, uuid
import datajoint as dj
dj.config.load(sys.argv[1])
from spyglass.spikesorting.analysis.v1 import group as gm
gm.test_mode = False
from spyglass.spikesorting.v2.curation_api import CurationRef
from spyglass.spikesorting.v2.analysis_selection import select_units_for_analysis
from spyglass.spikesorting.v2.curation import CurationV2
ref = CurationRef.from_key({'sorting_id': uuid.UUID(sys.argv[2]), 'curation_id': int(sys.argv[3])})
receipt = select_units_for_analysis(ref)
spikes, ids = receipt.fetch_spike_data(return_unit_ids=True)
with ref.open_analyzer() as analyzer:
    analyzer_ids = list(map(int, analyzer.unit_ids))
json.dump({'uuid': str(ref.curation_uuid), 'spikes': [s.tolist() for s in spikes],
    'ids': [r['unit_id'] for r in ids], 'observation': receipt.observation.intervals.tolist(),
    'analyzer_ids': analyzer_ids}, open(sys.argv[4], 'w'))
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(workflow["config"]),
            str(ref.sorting_id),
            str(ref.curation_id),
            str(payload),
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=180,
        env=os.environ.copy(),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    after = json.loads(payload.read_text())
    assert after["uuid"] == str(ref.curation_uuid)
    assert after["ids"] == [r["unit_id"] for r in ids]
    assert after["analyzer_ids"] == after["ids"]
    for expected, actual in zip(before, after["spikes"], strict=True):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(
        after["observation"], receipt.observation.intervals
    )

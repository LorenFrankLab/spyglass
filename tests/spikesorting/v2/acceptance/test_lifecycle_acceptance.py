"""End-to-end lifecycle acceptance probes (run with ``--run-acceptance``).

Uses the v2 test database and MEArec tetrode fixture. Position is a controlled
input; the sorter, curation, population selection, detector, and serialization
are real. This directory does not inherit decoding's mock NetCDF writer.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.acceptance


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

    from spyglass.spikesorting.v2._review.delivery import stop_review_servers
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
            assert status == 409 or after.labels_after.get(a) == (
                "accept",
            ), "A stale second browser silently overwrote the first scientist's saved label"
    finally:
        stop_review_servers()


def _bundle_units_table_unit_ids(bundle: Path) -> list[list[int]]:
    """Unit ids of every FigPack ``UnitsTable`` in an offline review bundle.

    ``figpack_spike_sorting`` stores each table's rows as JSON bytes in a
    ``rows_data`` uint8 array. A saved FigPack bundle is not a plain zarr
    directory: its metadata lives only in the consolidated ``.zmetadata`` and
    its chunks are packed into ``_consolidated_<n>.dat`` files, located by
    ``.zmetadata["refs"][key] = [file, offset, size]``. Rebuild an in-memory
    zarr v2 store from those pieces and let zarr decode the arrays.
    """
    import zarr

    store_dir = bundle / "data.zarr"
    consolidated = json.loads((store_dir / ".zmetadata").read_text())
    metadata = consolidated["metadata"]
    store = {key: json.dumps(value).encode() for key, value in metadata.items()}
    for key, (name, offset, size) in consolidated.get("refs", {}).items():
        with open(store_dir / name, "rb") as packed:
            packed.seek(offset)
            store[key] = packed.read(size)
    root = zarr.open_group(store, mode="r")
    tables = []
    for key, attrs in metadata.items():
        if not key.endswith(".zattrs"):
            continue
        if attrs.get("view_type") != "spike_sorting.UnitsTable":
            continue
        group = key[: -len(".zattrs")].rstrip("/")
        rows_path = f"{group}/rows_data" if group else "rows_data"
        rows = json.loads(bytes(root[rows_path][:]).decode("utf-8"))
        tables.append([int(row["unitId"]) for row in rows])
    return tables


def test_masked_sort_review(workflow):
    """The standard review of a sort with an excluded span shows every unit
    and round-trips a saved draft."""
    import urllib.parse
    import urllib.request

    from spyglass.spikesorting.v2._review.annotations import (
        labels_and_merges_to_annotations,
    )
    from spyglass.spikesorting.v2._review.delivery import stop_review_servers
    from spyglass.spikesorting.v2.review_api import FigPackReview
    from spyglass.spikesorting.v2.review_profile import FRANKLAB_REVIEW_PROFILE
    from spyglass.spikesorting.v2.sorting import Sorting

    root = workflow["root"]
    # Derived from the sort's units NWB, independently of the review bundle.
    sorted_unit_ids = sorted(
        map(
            int,
            Sorting().get_sorting({"sorting_id": root.sorting_id}).unit_ids,
        )
    )
    assert sorted_unit_ids == workflow["unit_ids"]

    review = root.start_review(FRANKLAB_REVIEW_PROFILE)
    tables = _bundle_units_table_unit_ids(Path(review.uri))
    assert tables, "The review bundle has no units table"
    for table_unit_ids in tables:
        assert len(table_unit_ids) == len(sorted_unit_ids), (
            f"The review lists {len(table_unit_ids)} units; the masked sort "
            f"has {len(sorted_unit_ids)}"
        )
        assert sorted(table_unit_ids) == sorted_unit_ids

    # Alternate two palette labels so a draft that lost or shifted a unit
    # cannot compare equal.
    options = review.profile.label_options[:2]
    draft_labels = {
        unit: (options[i % 2],) for i, unit in enumerate(sorted_unit_ids)
    }
    draft = labels_and_merges_to_annotations(
        {unit: list(labels) for unit, labels in draft_labels.items()},
        [],
        label_options=list(review.profile.label_options),
    )
    url = urllib.parse.urljoin(
        review.open(open_browser=False), "annotations.json"
    )
    try:
        # Save the draft the way the browser does: PUT with the revision
        # returned by the last read.
        with urllib.request.urlopen(url, timeout=30) as response:
            revision = response.headers["ETag"]
        request = urllib.request.Request(
            url,
            data=json.dumps(draft).encode("utf-8"),
            method="PUT",
            headers={"Content-Type": "application/json", "If-Match": revision},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            assert response.status == 200
    finally:
        stop_review_servers()

    reloaded = FigPackReview.resume(review.review_id).preview_import()
    assert reloaded.unit_count_before == len(sorted_unit_ids)
    assert dict(reloaded.labels_after) == draft_labels
    assert reloaded.merge_groups == ()


def test_fresh_process_rebuild_preserves_population(workflow, monkeypatch):
    from spyglass.spikesorting.analysis.v1 import group as group_module
    from spyglass.spikesorting.v2._storage.analyzer_cache import (
        remove_analyzer_cache,
    )
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

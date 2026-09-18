"""Bounded unit-count benchmark of the production review layout and browser.

Run one unit count per fresh process. Uses a three-second synthetic recording;
this measures unit-count scaling, not long-recording capacity or sorting quality.
"""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ["SPYGLASS_BASE_DIR"] = str(args.out / "tests/data")

    import numpy as np
    import spikeinterface as si
    from playwright.sync_api import sync_playwright

    from spyglass.spikesorting.v2._figpack_curation import (
        labels_and_merges_to_annotations,
    )
    from spyglass.spikesorting.v2._review_delivery import (
        serve_review_bundle,
        stop_review_servers,
    )
    from spyglass.spikesorting.v2._review_inspection import inspection_view
    from spyglass.spikesorting.v2._review_profile import ReviewDisplayOptions
    from spyglass.spikesorting.v2._review_view import (
        coerce_units_table_ids,
        compose_review_layout,
        curation_control,
    )
    from tests.spikesorting.v2 import _browser_review as ui
    from tests.spikesorting.v2.scripts.measure_release_workflow import (
        TreeMonitor,
    )

    measurements = {
        "units": args.units,
        "channels": 16,
        "duration_s": 3,
        "rss_definition": "sampled sum of RSS for Python and descendants; not deduplicated",
    }
    bundle = args.out / "review.figpack"
    with TreeMonitor([args.out]) as monitor:
        start = perf_counter()
        recording, sorting = si.generate_ground_truth_recording(
            durations=[3.0],
            num_channels=16,
            num_units=args.units,
            sampling_frequency=30000,
            generate_unit_locations_kwargs={"minimum_distance": None},
            seed=42,
        )
        analyzer = si.create_sorting_analyzer(sorting, recording, sparse=False)
        analyzer.compute(
            [
                "random_spikes",
                "waveforms",
                "templates",
                "noise_levels",
                "spike_amplitudes",
                "correlograms",
                "unit_locations",
                "template_similarity",
            ],
            n_jobs=1,
            progress_bar=False,
        )
        assert np.isfinite(analyzer.get_extension("templates").get_data()).all()
        measurements["analyzer_seconds"] = perf_counter() - start
        start = perf_counter()
        summary = inspection_view(
            analyzer, ReviewDisplayOptions(), displayed_unit_properties=[]
        )
        view = compose_review_layout(
            summary,
            curation_control(["accept", "noise"], {}),
            summary_title="Synthetic unit-count scaling",
        )
        coerce_units_table_ids(view)
        view.save(str(bundle), title="Review scaling")
        (bundle / "annotations.json").write_text(
            json.dumps(
                labels_and_merges_to_annotations(
                    {}, [], label_options=["accept", "noise"]
                )
            )
        )
        measurements["bundle_seconds"] = perf_counter() - start
        measurements["bundle_bytes"] = TreeMonitor.dir_bytes(bundle)
        url = serve_review_bundle(bundle)
        errors = []
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                page = browser.new_page(viewport={"width": 1280, "height": 720})
                page.set_default_timeout(90000)
                page.on("pageerror", lambda error: errors.append(str(error)))
                start = perf_counter()
                page.goto(url)
                ui.start_curating(page)
                ui.wait_for_visible_plot(page)
                measurements["browser_ready_seconds"] = perf_counter() - start
                start = perf_counter()
                ui.unit_row(page, 0).get_by_role("checkbox").check()
                ui.set_label(page, "accept", True)
                assert ui.save_annotations(page) in (200, 201)
                measurements["label_and_save_seconds"] = perf_counter() - start
                measurements["table_rows"] = page.get_by_role("row").count()
                page.screenshot(path=str(args.out / "review.png"))
                browser.close()
        finally:
            stop_review_servers()
            measurements["browser_errors"] = errors
    measurements["peak_tree_rss_gb"] = monitor.peak_rss / 1024**3
    (args.out / "measurement.json").write_text(
        json.dumps(measurements, indent=2)
    )
    print(json.dumps(measurements))


if __name__ == "__main__":
    main()

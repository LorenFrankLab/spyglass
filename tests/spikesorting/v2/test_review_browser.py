"""The review figure in a real browser (DB-free).

Composes the production review layout (``_review_view``) over an in-memory
SpikeInterface analyzer with the official-property unit table
(``_review_unit_properties``), saves a real FigPack bundle, serves it with
the production local delivery (``_review_delivery``), and drives headless
Chromium through the documented sequence: **Curate Figure**, select units
in the unit table, add / remove labels, propose a merge, **Save
Annotations**, reload. Assertions are on the ``annotations.json`` the
frontend wrote and Spyglass's own parser reads.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tests.spikesorting.v2 import _browser_review as browser

pytestmark = [pytest.mark.slow]

LABEL_OPTIONS = ["accept", "mua", "noise"]


@pytest.fixture(scope="module")
def review_bundle(tmp_path_factory):
    """A saved review bundle over a small synthetic 3-unit analyzer."""
    browser.require_browser()
    import spikeinterface.core as sc
    import spikeinterface.widgets as sw

    from spyglass.spikesorting.v2._figpack_curation import (
        labels_and_merges_to_annotations,
    )
    from spyglass.spikesorting.v2._review_unit_properties import (
        review_unit_properties,
    )
    from spyglass.spikesorting.v2._review_view import (
        coerce_units_table_ids,
        compose_review_layout,
        curation_control,
    )

    recording, sorting = sc.generate_ground_truth_recording(
        durations=[6.0], num_channels=4, num_units=3, seed=0
    )
    sorting = sorting.rename_units(np.array([1, 2, 3]))
    analyzer = sc.create_sorting_analyzer(sorting, recording, sparse=False)
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
        ]
    )
    # The "official" review columns: an evaluation metric, a proposal, an
    # incomplete boolean annotation (its gap must not render as False).
    table = pd.DataFrame(
        {
            "snr": [8.0, np.nan, 7.0],
            "proposed_labels": ["", "noise", ""],
            "burst_flag": pd.array([True, None, False], dtype="boolean"),
        },
        index=pd.Index([1, 2, 3], name="unit_id"),
    )
    from spyglass.spikesorting.v2._review_unit_properties import (
        missing_rule_metrics,
    )

    table["unavailable_qc"] = missing_rule_metrics(table, ["snr"])
    summary = sw.plot_sorting_summary(
        analyzer,
        backend="figpack",
        curation=False,
        displayed_unit_properties=[],
        extra_unit_properties=review_unit_properties(table, analyzer.unit_ids),
        generate_url=False,
        display=False,
    ).view
    view = compose_review_layout(
        summary,
        curation_control(LABEL_OPTIONS, {1: ["accept"], 2: ["noise"]}),
        summary_title="Sorting summary -- synthetic browser test",
    )
    coerce_units_table_ids(view)
    bundle = tmp_path_factory.mktemp("browser") / "review.figpack"
    view.save(str(bundle), title="Spyglass v2 browser test")
    (bundle / "annotations.json").write_text(
        json.dumps(
            labels_and_merges_to_annotations(
                {1: ["accept"], 2: ["noise"]}, [], label_options=LABEL_OPTIONS
            )
        )
    )
    yield bundle
    from spyglass.spikesorting.v2._review_delivery import stop_review_servers

    stop_review_servers(bundle)


def _saved_state(bundle):
    from spyglass.spikesorting.v2._figpack_curation import (
        curation_annotations_to_labels_and_merges,
    )

    return curation_annotations_to_labels_and_merges(
        json.loads((bundle / "annotations.json").read_text())
    )


@pytest.mark.parametrize("viewport", sorted(browser.VIEWPORTS))
def test_review_columns_and_controls_are_usable(
    review_bundle, viewport, tmp_path
):
    """Official columns appear in the selectable unit table in the requested
    order with their values (gaps empty), a metric row selects that unit for
    curation, and the curation controls are reachable at both viewports."""
    from spyglass.spikesorting.v2._review_delivery import serve_review_bundle

    url = serve_review_bundle(review_bundle)
    with browser.review_page(
        url, viewport=browser.VIEWPORTS[viewport], artifacts=tmp_path
    ) as page:
        # SI's unit table: checkbox | Unit | Labels | Similarity | <ours...>.
        # The review columns follow in the requested order; SI's default
        # metric columns (firing_rate, x, y, ...) are NOT added.
        headers = browser.row_texts(page.get_by_role("row").first)
        assert [h.lower() for h in headers] == [
            "",
            "unit",
            "labels",
            "similarity",
            "snr",
            "proposed_labels",
            "burst_flag",
            "unavailable_qc",
        ]
        assert browser.row_texts(browser.unit_row(page, 2)) == [
            "",
            "2",
            "noise",
            "",
            "",  # unavailable numeric evidence stays absent
            "noise",
            "",  # the boolean gap stays empty, not False
            "snr",
        ]
        assert browser.row_texts(browser.unit_row(page, 1))[-2] == "True"

        browser.start_curating(page)
        browser.select_units(page, 3)  # select via the metric-bearing row
        assert browser.label_checkbox(page, "accept").is_enabled()
        assert page.get_by_role("button", name="Finalize Curation").is_enabled()
        # The pane collapses to hand the views the full height, and expands
        # again with its controls reachable.
        merge = page.get_by_role("button", name="Merge Selected", exact=True)
        browser.toggle_curation_pane(page)
        merge.wait_for(state="hidden")
        browser.toggle_curation_pane(page)
        merge.wait_for(state="visible")
        assert browser.label_checkbox(page, "accept").is_enabled()


def test_browser_edits_reach_annotations_and_survive_reload(
    review_bundle, tmp_path
):
    """Label add/remove + a merge proposal saved by the browser land in the
    bundle's annotations.json as Spyglass reads them, and reload shows them."""
    from spyglass.spikesorting.v2._review_delivery import serve_review_bundle

    url = serve_review_bundle(review_bundle)
    with browser.review_page(url, artifacts=tmp_path) as page:
        browser.start_curating(page)
        browser.select_units(page, 3)
        browser.set_label(page, "accept", True)
        browser.select_units(page, 2)
        browser.set_label(page, "noise", False)
        browser.select_units(page, 1, 3)
        browser.merge_selected(page)
        page.get_by_text("Selected units in 1 merge group(s)").wait_for()
        assert browser.save_annotations(page) in (200, 201)

    labels, merges = _saved_state(review_bundle)
    # Unit 2's cleared label is an empty list in the saved state (the
    # frontend keeps the key); units 1 and 3 carry accept; one merge.
    assert {u: v for u, v in labels.items() if v} == {
        1: ["accept"],
        3: ["accept"],
    }
    assert labels.get(2, []) == []
    assert merges == [[1, 3]]

    with browser.review_page(url, artifacts=tmp_path) as page:
        assert browser.row_texts(browser.unit_row(page, 3))[2] == "accept"
        assert browser.row_texts(browser.unit_row(page, 2))[2] == ""
        assert "(1, 3)" in browser.row_texts(browser.unit_row(page, 1))[1]


@pytest.mark.parametrize("n_units", [64, 256])
def test_large_review_save_reload(n_units, tmp_path):
    """Measure actual browser interactions independently of sorter runtime."""
    from time import perf_counter

    import spikeinterface.core as sc
    import spikeinterface.widgets as sw

    from spyglass.spikesorting.v2._review_delivery import (
        serve_review_bundle,
        stop_review_servers,
    )
    from spyglass.spikesorting.v2._review_view import (
        coerce_units_table_ids,
        compose_review_layout,
        curation_control,
    )

    browser.require_browser()
    start = perf_counter()
    recording, sorting = sc.generate_ground_truth_recording(
        durations=[6.0], num_channels=16, num_units=n_units, seed=42
    )
    analyzer = sc.create_sorting_analyzer(sorting, recording, sparse=False)
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
        ]
    )
    summary = sw.plot_sorting_summary(
        analyzer,
        backend="figpack",
        curation=False,
        generate_url=False,
        display=False,
        max_amplitudes_per_unit=1000,
        min_similarity_for_correlograms=0.2,
    ).view
    view = compose_review_layout(
        summary,
        curation_control(LABEL_OPTIONS),
        summary_title="Sorting summary",
    )
    coerce_units_table_ids(view)
    bundle = tmp_path / "stress.figpack"
    view.save(str(bundle), title=f"{n_units} unit review")
    result = {
        "units": n_units,
        "channels": 16,
        "duration_s": 6,
        "generation_s": perf_counter() - start,
        "bundle_bytes": sum(
            path.stat().st_size for path in bundle.rglob("*") if path.is_file()
        ),
    }
    url = serve_review_bundle(bundle)
    unit_id = int(analyzer.unit_ids[-1])
    try:
        start = perf_counter()
        with browser.review_page(url, artifacts=tmp_path) as page:
            result["browser_start_and_load_s"] = perf_counter() - start
            assert page.get_by_text(
                "Commit in Python:", exact=False
            ).is_visible()
            browser.start_curating(page)
            start = perf_counter()
            browser.unit_row(page, unit_id).get_by_role("checkbox").check()
            browser.set_label(page, "accept", True)
            assert browser.save_annotations(page) in (200, 201)
            result["select_label_save_s"] = perf_counter() - start
        start = perf_counter()
        with browser.review_page(url, artifacts=tmp_path) as page:
            assert (
                browser.row_texts(browser.unit_row(page, unit_id))[2]
                == "accept"
            )
            result["browser_restart_reload_s"] = perf_counter() - start
        labels, merges = _saved_state(bundle)
        assert labels[unit_id] == ["accept"] and merges == []
    finally:
        stop_review_servers(bundle)
    (tmp_path / "measurements.json").write_text(json.dumps(result, indent=2))
    print("REVIEW_MEASUREMENT", json.dumps(result))

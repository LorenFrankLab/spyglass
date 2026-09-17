"""The documented local review journey, end to end, through a real browser.

Start a review over a real committed root curation, deliver it with
``review.open()``, edit through headless Chromium (labels + a merge), let
the browser save, preview and commit in Python, continue onto the merged
child, verify it with a no-change commit, and hand the exact reviewed
result to analysis selection. Delivery is restarted and the review resumed
in between, as after a kernel restart. Requires the curation extra, Python
Playwright and a database.
"""

from __future__ import annotations

import pytest

from tests.spikesorting.v2 import _browser_review as browser

pytestmark = [pytest.mark.slow, pytest.mark.integration]


@pytest.fixture(autouse=True)
def _browser_stack():
    browser.require_browser()
    yield
    from spyglass.spikesorting.v2._review_delivery import stop_review_servers

    stop_review_servers()


def _ensure_journey_profile() -> str:
    from spyglass.spikesorting.v2.review_profile import CurationReviewProfile

    name = "test_browser_journey_2026_09"
    CurationReviewProfile.insert1(
        {
            "review_profile_name": name,
            "metric_params_name": "minimal",
            "auto_curation_rules_name": "none",
            "displayed_unit_properties": ["snr", "firing_rate"],
            "label_options": ["accept", "mua", "noise"],
            "label_import_mode": "replace",
        },
        skip_duplicates=True,
    )
    return name


def test_browser_review_commit_verify_and_select(
    planted_two_unit_sort, curation_evaluation_defaults, tmp_path, monkeypatch
):
    from spyglass.spikesorting.analysis.v1 import group as group_module
    from spyglass.spikesorting.analysis.v1.group import SortedSpikesGroup
    from spyglass.spikesorting.v2._review_delivery import (
        served_review_bundles,
        stop_review_servers,
    )
    from spyglass.spikesorting.v2.analysis_selection import (
        select_units_for_analysis,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.curation_api import CurationRef
    from spyglass.spikesorting.v2.review_api import FigPackReview
    from spyglass.spikesorting.v2.sorting import Sorting
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    sorting_key = dict(planted_two_unit_sort)
    unit_a, unit_b = sorted(
        map(int, (Sorting.Unit & sorting_key).fetch("unit_id"))
    )
    clear_curations_for(sorting_key)
    monkeypatch.setattr(group_module, "test_mode", False)
    profile = _ensure_journey_profile()
    created_groups = []
    try:
        root = CurationRef.from_key(
            CurationV2.create_initial_curation(
                sorting_key, labels={unit_b: ["noise"]}
            )
        )
        review = root.start_review(profile, upload=False)
        detail = review.inspect_units([unit_a], time_range=(0, 0.2))
        assert [row.unit_id for row in detail.item1.view.rows] == [unit_a]
        assert "committed_labels" in [
            column.key for column in detail.item1.view.columns
        ]
        detail.save(
            str(tmp_path / "inspection.figpack"), title="Selected units"
        )
        url = review.open(open_browser=False)
        assert url.startswith("http://localhost:")
        assert review.open(open_browser=False) == url  # reused delivery

        # 1. Edit in the browser: accept both units (clearing the seeded
        #    noise label), propose merging them, save.
        with browser.review_page(url, artifacts=tmp_path / "edit") as page:
            # The official columns (profile order) are in the unit table.
            headers = [
                h.lower()
                for h in browser.row_texts(page.get_by_role("row").first)
            ]
            assert page.get_by_text(
                "Commit and review:", exact=False
            ).is_visible()
            # Evidence coverage, actions, then the profile's metrics in order.
            assert headers[4:10] == [
                "unavailable_qc",
                "proposed_labels",
                "proposed_merge_groups",
                "merged_from",
                "snr",
                "firing_rate",
            ]
            assert browser.row_texts(browser.unit_row(page, unit_b))[2] == (
                "noise"
            )
            browser.start_curating(page)
            browser.select_units(page, unit_a)
            browser.set_label(page, "accept", True)
            browser.select_units(page, unit_b)
            browser.set_label(page, "noise", False)
            browser.set_label(page, "accept", True)
            browser.select_units(page, unit_a, unit_b)
            browser.merge_selected(page)
            assert browser.save_annotations(page) in (200, 201)

        # 2. Preview reads the file the browser wrote; commit the merge.
        changes = review.preview_import()
        assert changes.has_changes
        assert changes.merge_groups == ((unit_a, unit_b),)
        assert changes.labels_after[unit_a] == ("accept",)
        assert changes.labels_after[unit_b] == ("accept",)
        assert not changes.label_conflicts  # both contributors: accept
        assert "proposed merges" in changes.summary()
        panel = review.commit_panel(open_browser=False)
        panel.button.click()
        receipt = panel.receipt
        assert receipt is not None
        assert panel.verification_review is not None
        merged = receipt.curation
        assert merged.parent == root
        assert receipt.needs_merge_verification
        merged_unit_id = max(unit_a, unit_b) + 1
        assert sorted(
            map(int, (CurationV2.Unit & merged.as_key()).fetch("unit_id"))
        ) == [merged_unit_id]

        # 3. Continue onto the merged child; restart delivery and resume, as
        #    after a kernel restart, then inspect the merged unit.
        continuation = receipt.continue_review()
        stop_review_servers()
        assert served_review_bundles() == {}
        resumed = FigPackReview.resume(continuation.review_id)
        merged_url = resumed.open(open_browser=False)
        with browser.review_page(
            merged_url, artifacts=tmp_path / "verify"
        ) as page:
            row = browser.row_texts(browser.unit_row(page, merged_unit_id))
            assert row[1] == str(merged_unit_id) and row[2] == "accept"
            # Applied merge provenance is a unit-table column.
            headers = browser.row_texts(page.get_by_role("row").first)
            merged_from = row[[h.lower() for h in headers].index("merged_from")]
            assert merged_from == f"{unit_a},{unit_b}"
        # The first review's saved edits are still there after the restart.
        with browser.review_page(
            review.open(open_browser=False), artifacts=tmp_path / "reopen"
        ) as page:
            assert "(" in browser.row_texts(browser.unit_row(page, unit_a))[1]

        # 4. No further edits: an explicit no-change verification.
        verification = resumed.preview_import()
        assert not verification.has_changes
        assert "confirm_no_changes" in verification.summary()
        verification_panel = resumed.commit_panel(open_browser=False)
        assert "Record reviewed" in verification_panel.button.description
        verification_panel.button.click()
        final_receipt = verification_panel.receipt
        final_curation = final_receipt.curation
        assert final_curation.parent == merged
        assert not final_receipt.needs_merge_verification

        # 5. Hand exactly that curation to analysis.
        selection = select_units_for_analysis(
            final_curation, policy="v2_accepted_single_units"
        )
        created_groups.extend(g.group_key for g in selection.groups)
        assert selection.curation == final_curation
        assert selection.included_unit_ids == (merged_unit_id,)
        assert selection.excluded_units == {}
        spikes, ids = selection.fetch_spike_data(return_unit_ids=True)
        assert [d["unit_id"] for d in ids] == [merged_unit_id]
        assert all(
            d["spikesorting_merge_id"] == final_curation.merge_id for d in ids
        )
        assert len(spikes) == 1 and len(spikes[0]) > 0

        # 6. Recovery: the merge was a mistake. The review it came from is
        #    on its receipt (the merge's own parent -- here the root); undo
        #    the proposal in the browser, save, and commit a replacement
        #    sibling; the labels saved along with the merge are kept, the
        #    merged branch stays as history. find() lists that review too
        #    (a fresh start_review would begin a new review now that a
        #    child exists).
        again = receipt.changes.review
        assert again.review_id == review.review_id
        assert [
            r.review_id for r in FigPackReview.find(root, profile=profile)
        ] == [review.review_id]
        assert FigPackReview.find(merged) == (resumed,)
        assert again.preview_import().has_changes  # the saved edits
        with browser.review_page(
            again.open(open_browser=False), artifacts=tmp_path / "recover"
        ) as page:
            browser.start_curating(page)
            browser.select_units(page, unit_a, unit_b)
            page.get_by_role(
                "button", name="Undo selected merge", exact=True
            ).click()
            page.get_by_text("Pending merges:", exact=False).wait_for(
                state="hidden"
            )
            assert browser.save_annotations(page) in (200, 201)
        recovery = again.preview_import()
        assert recovery.merge_groups == ()
        assert recovery.labels_after[unit_a] == ("accept",)
        assert recovery.labels_after[unit_b] == ("accept",)
        assert merged in recovery.newer_sibling_curations
        assert "differ from the reviewed parent" in recovery.next_step()
        replacement = recovery.commit().curation
        assert replacement.parent == root and replacement != merged
        assert sorted(
            map(int, (CurationV2.Unit & replacement.as_key()).fetch("unit_id"))
        ) == [unit_a, unit_b]
        assert {c.curation_id for c in root.children} >= {
            merged.curation_id,
            replacement.curation_id,
        }
        # The abandoned branch (merged child + its verification) can be
        # removed leaf-first once nothing downstream refers to it.
        for group_key in created_groups:
            (SortedSpikesGroup & dict(group_key)).super_delete(warn=False)
        created_groups.clear()
        preview = merged.preview_curation_delete()
        assert [r.curation_id for r in preview.leaf_first] == [
            final_curation.curation_id,
            merged.curation_id,
        ]
        merged.delete_subtree(safemode=False)
        assert {c.curation_id for c in root.children} == {
            replacement.curation_id
        }

        # 7. A merge-only mistake on a LATER curation: recover through that
        #    merge's own parent (the replacement, not the root) and, since
        #    unmerging restores the parent exactly, confirm the no-change
        #    commit.
        later = replacement.start_review(profile, upload=False)
        with browser.review_page(
            later.open(open_browser=False), artifacts=tmp_path / "later"
        ) as page:
            browser.start_curating(page)
            browser.select_units(page, unit_a, unit_b)
            browser.merge_selected(page)
            assert browser.save_annotations(page) in (200, 201)
        bad_receipt = later.preview_import().commit()
        bad_merge = bad_receipt.curation
        assert bad_merge.parent == replacement
        assert bad_receipt.needs_merge_verification
        review_of_bad = bad_receipt.changes.review
        assert review_of_bad.parent == replacement  # not the root
        with browser.review_page(
            review_of_bad.open(open_browser=False),
            artifacts=tmp_path / "later-undo",
        ) as page:
            browser.start_curating(page)
            browser.select_units(page, unit_a, unit_b)
            page.get_by_role(
                "button", name="Undo selected merge", exact=True
            ).click()
            assert browser.save_annotations(page) in (200, 201)
        undone = review_of_bad.preview_import()
        assert not undone.has_changes
        assert bad_merge in undone.newer_sibling_curations
        with pytest.raises(ValueError, match="confirm_no_changes"):
            undone.commit()
        replacement_receipt = undone.commit(confirm_no_changes=True)
        assert not replacement_receipt.needs_merge_verification
        verified_replacement = replacement_receipt.curation
        assert verified_replacement.parent == replacement
        assert sorted(
            map(
                int,
                (CurationV2.Unit & verified_replacement.as_key()).fetch(
                    "unit_id"
                ),
            )
        ) == [unit_a, unit_b]
        # The analysis handoff follows the replacement branch: both units
        # (still accept) from the replacement's merge id, not the bad merge.
        recovered = select_units_for_analysis(
            verified_replacement, policy="v2_accepted_single_units"
        )
        created_groups.extend(g.group_key for g in recovered.groups)
        assert recovered.curation == verified_replacement
        assert recovered.included_unit_ids == (unit_a, unit_b)
        assert recovered.groups[0].merge_id == verified_replacement.merge_id
        assert recovered.groups[0].merge_id != bad_merge.merge_id
        _, ids = recovered.fetch_spike_data(return_unit_ids=True)
        assert [d["unit_id"] for d in ids] == [unit_a, unit_b]
        for group_key in created_groups:
            (SortedSpikesGroup & dict(group_key)).super_delete(warn=False)
        created_groups.clear()
        bad_merge.delete_subtree(safemode=False)
    finally:
        for group_key in created_groups:
            (SortedSpikesGroup & dict(group_key)).super_delete(warn=False)
        clear_curations_for(sorting_key)


def test_connected_browser_commits_conflict_and_verifies_without_notebook(
    planted_two_unit_sort, curation_evaluation_defaults, tmp_path
):
    import json
    from time import perf_counter

    from playwright.sync_api import expect

    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.curation_api import CurationRef
    from spyglass.spikesorting.v2.sorting import Sorting
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    sorting_key = dict(planted_two_unit_sort)
    clear_curations_for(sorting_key)
    first, second = sorted(
        map(int, (Sorting.Unit & sorting_key).fetch("unit_id"))
    )
    root = CurationRef.from_key(CurationV2.insert_curation(sorting_key))
    review = root.start_review(
        _ensure_journey_profile(), display_options={"max_initial_points": 1}
    )
    initial_url = review.open(open_browser=False)
    measurements = {}
    with browser.review_page(
        initial_url, artifacts=tmp_path / "connected"
    ) as page:
        page.get_by_role(
            "button", name="Preview and commit", exact=True
        ).wait_for()
        for name in (
            "Waveforms",
            "Spike amplitudes",
            "Autocorrelograms",
            "Cross-correlograms",
            "Electrode geometry",
        ):
            expect(page.get_by_text(name, exact=True)).to_be_visible()
        page.get_by_text("Spike amplitudes", exact=True).click()
        expect(
            page.get_by_text("Time-based overview not loaded.", exact=True)
        ).to_be_visible()
        page.get_by_text("Waveforms", exact=True).click()
        browser.select_units(page, first)
        browser.set_label(page, "accept", True)
        # Focused evidence is loaded through the service without saving or
        # replacing the active draft. The trace request stays bounded.
        page.get_by_label("Start seconds", exact=True).fill("0.1")
        page.get_by_label("Stop seconds", exact=True).fill("0.3")
        started = perf_counter()
        page.get_by_role(
            "button", name="Inspect selected units / pairs", exact=True
        ).click()
        detail = page.get_by_role(
            "link", name="Open selected-unit inspection", exact=True
        )
        expect(detail).to_be_visible(timeout=120000)
        measurements["focused_bundle_preparation_s"] = perf_counter() - started
        with page.expect_popup() as popup:
            detail.click()
        focused = popup.value
        started = perf_counter()
        try:
            trace_tab = focused.get_by_text("Spikes on traces", exact=True)
            expect(trace_tab).to_be_visible(timeout=60000)
            trace_tab.click()
            # A visible tab alone says nothing about whether its image loaded.
            focused.wait_for_function(
                """() => [...document.images].some(image => {
                    const rect = image.getBoundingClientRect();
                    return image.complete && image.naturalWidth >= 100 &&
                        image.naturalHeight >= 100 && rect.width >= 100 &&
                        rect.height >= 100;
                })"""
            )
            measurements["focused_browser_load_s"] = perf_counter() - started
        finally:
            focused.screenshot(path=str(tmp_path / "focused-traces.png"))
            focused.close()
        assert browser.label_checkbox(page, "accept").is_checked()
        browser.select_units(page, second)
        browser.set_label(page, "noise", True)
        browser.select_units(page, first, second)
        browser.merge_selected(page)
        page.get_by_role(
            "button", name="Preview and commit", exact=True
        ).click()
        confirmation = page.get_by_label(
            "Use these final labels (empty is allowed)", exact=True
        )
        expect(confirmation).to_be_visible(timeout=120000)
        page.get_by_text("accept", exact=True).last.locator("..").get_by_role(
            "checkbox"
        ).check()
        confirmation.check()
        started = perf_counter()
        page.get_by_role(
            "button", name="Commit and inspect merged units", exact=True
        ).click()
        page.wait_for_url(lambda url: str(url) != initial_url, timeout=180000)
        expect(browser.unit_row(page, second + 1)).to_be_visible(timeout=60000)
        measurements["merge_evaluate_and_open_s"] = perf_counter() - started
        with pytest.raises(ValueError, match="not completed"):
            review.result()
        page.get_by_role(
            "button", name="Preview and commit", exact=True
        ).click()
        verify = page.get_by_role(
            "button", name="Record reviewed — no changes", exact=True
        )
        expect(verify).to_be_visible(timeout=120000)
        verify.click()
        expect(
            page.get_by_text(
                "Review complete. This curation is ready for analysis.",
                exact=True,
            )
        ).to_be_visible(timeout=120000)
        final = review.result()
        assert final.parent.parent == root
        assert [
            int(u) for u in (CurationV2.Unit & final.as_key()).fetch("unit_id")
        ] == [second + 1]
        page.reload()
        expect(
            page.get_by_text(
                "Review complete. This curation is ready for analysis.",
                exact=True,
            )
        ).to_be_visible()
        assert review.result() == final
        # Recover a mistaken merge through its original draft, then explicitly
        # replace the result branch. No notebook mutation is involved.
        page.get_by_role(
            "button", name="Review parent branch", exact=True
        ).click()
        page.wait_for_url(initial_url, timeout=120000)
        expect(page.get_by_text("Draft saved.", exact=False)).to_be_visible()
        expect(browser.unit_row(page, first)).to_be_visible()
        browser.select_units(page, first, second)
        page.get_by_role(
            "button", name="Undo selected merge", exact=True
        ).click()
        page.get_by_role(
            "button", name="Preview and commit", exact=True
        ).click()
        commit = page.get_by_role("button", name="Commit curation", exact=True)
        expect(commit).to_be_visible(timeout=120000)
        commit.click()
        expect(
            page.get_by_text(
                "Review complete. This curation is ready for analysis.",
                exact=True,
            )
        ).to_be_visible(timeout=120000)
        replacement = review.result()
        assert replacement.parent == root
        assert replacement != final
        assert sorted(
            map(int, (CurationV2.Unit & replacement.as_key()).fetch("unit_id"))
        ) == [first, second]
    (tmp_path / "connected-measurements.json").write_text(
        json.dumps(measurements, indent=2)
    )

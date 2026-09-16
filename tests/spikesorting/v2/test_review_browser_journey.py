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
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

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
                "Commit in Python:", exact=False
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
        receipt = changes.commit()
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
        final_receipt = verification.commit(confirm_no_changes=True)
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
                "button", name="Unmerge Selected", exact=True
            ).click()
            page.get_by_text("2 unmerged unit(s) selected").wait_for()
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
                "button", name="Unmerge Selected", exact=True
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

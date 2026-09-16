"""Playwright helpers for driving a served FigPack review in a real browser.

Shared by the DB-free view test and the database-backed review journey.
Every interaction is an ordinary user action (visible controls, normal
clicks, no DOM patching); a save is awaited as the browser's own ``PUT`` of
``annotations.json`` returning 2xx so the assertion is on the file the
importer reads, written by the frontend.
"""

from __future__ import annotations

import importlib.util
import os
import re
from contextlib import contextmanager
from pathlib import Path

import pytest

REQUIRE_BROWSER_ENV = "SPYGLASS_REQUIRE_BROWSER_TESTS"
DEFAULT_TIMEOUT_MS = 20_000

#: Laptop and large-display viewports the layout must stay usable at.
VIEWPORTS = {"laptop": (1280, 720), "desktop": (1920, 1080)}


def browser_unavailable_reason() -> str | None:
    """Why the browser tests cannot run here, or ``None`` when they can."""
    if importlib.util.find_spec("figpack") is None:
        return "requires the spikesorting-v2-curation extra (figpack)"
    if importlib.util.find_spec("playwright") is None:
        return "requires playwright (spikesorting-v2-curation-test extra)"
    return None


def require_browser() -> None:
    """Skip when the browser stack is absent -- or FAIL when the lane that
    is supposed to run these tests (``SPYGLASS_REQUIRE_BROWSER_TESTS=1``)
    is missing a dependency, so a green job cannot be made of skips."""
    reason = browser_unavailable_reason()
    if reason is None:
        return
    if os.environ.get(REQUIRE_BROWSER_ENV):
        pytest.fail(f"browser tests are required in this lane but {reason}")
    pytest.skip(reason)


@contextmanager
def review_page(url: str, *, viewport=VIEWPORTS["laptop"], artifacts: Path):
    """Open ``url`` in headless Chromium; on any error keep a screenshot.

    Yields the Playwright ``Page`` once the figure has rendered (the
    **Curate Figure** button is visible). Confirmation dialogs (the merge
    button asks) are accepted. On an exception, ``<artifacts>/failure.png``
    and the page's text are written before re-raising.
    """
    from playwright.sync_api import sync_playwright

    artifacts.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": viewport[0], "height": viewport[1]}
        )
        page = context.new_page()
        page.set_default_timeout(DEFAULT_TIMEOUT_MS)
        page.on("dialog", lambda dialog: dialog.accept())
        try:
            page.goto(url, wait_until="load")
            page.get_by_role("button", name="Curate Figure").wait_for()
            yield page
        except Exception:
            try:
                page.screenshot(
                    path=str(artifacts / "failure.png"), full_page=True
                )
                (artifacts / "failure.txt").write_text(
                    page.locator("body").inner_text()
                )
            except Exception:  # pragma: no cover - diagnostics only
                pass
            raise
        finally:
            context.close()
            browser.close()


def unit_row(page, unit_id: int):
    """The unit table row whose Unit cell is ``unit_id`` (or, once merged,
    ``"<unit_id> (a, b)"``)."""
    return page.get_by_role("row").filter(
        has=page.get_by_role("cell", name=re.compile(rf"^{unit_id}( \(.*\))?$"))
    )


def row_texts(row) -> list[str]:
    """Every cell's text in ``row`` (the leading checkbox cell is ``""``)."""
    cells = row.get_by_role("cell")
    return [cells.nth(i).inner_text().strip() for i in range(cells.count())]


def toggle_curation_pane(page) -> None:
    """Collapse / expand the Curation pane through its title toggle."""
    page.get_by_text(re.compile(r"^[▼▶]\s*Curation")).first.click()


def start_curating(page) -> None:
    """**Curate Figure** enables editing (the frontend's local-figure mode)."""
    page.get_by_role("button", name="Curate Figure", exact=True).click()
    page.get_by_text("Curating Local Figure").wait_for()


def select_units(page, *unit_ids: int) -> None:
    """Select exactly ``unit_ids`` through the unit table's row checkboxes."""
    rows = page.get_by_role("row")
    for index in range(1, rows.count()):  # row 0 is the header
        row = rows.nth(index)
        unit_id = int(row_texts(row)[1].split(" ")[0])
        box = row.get_by_role("checkbox")
        wanted = unit_id in unit_ids
        if box.is_checked() != wanted:
            box.set_checked(wanted)
    listed = ", ".join(str(u) for u in unit_ids)
    page.get_by_text(re.compile(rf"^Units: {listed}$")).wait_for()


def label_checkbox(page, label: str):
    """The Curation pane's checkbox for ``label`` (the last match on the
    page: the pane sits below the unit table, whose cells may also show the
    label text)."""
    return (
        page.get_by_text(label, exact=True)
        .last.locator("..")
        .get_by_role("checkbox")
    )


def set_label(page, label: str, on: bool) -> None:
    box = label_checkbox(page, label)
    box.check() if on else box.uncheck()


def merge_selected(page) -> None:
    page.get_by_role("button", name="Merge Selected", exact=True).click()


def save_annotations(page) -> int:
    """Click **Save Annotations**; return the status of the browser's PUT."""
    with page.expect_response(
        lambda response: response.request.method == "PUT"
        and response.url.endswith("/annotations.json")
    ) as saved:
        page.get_by_role("button", name="Save Annotations", exact=True).click()
    return saved.value.status

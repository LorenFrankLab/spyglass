"""Guided notebook commit, keeping all scientific writes in Python."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spyglass.spikesorting.v2.review_api import (
        FigPackReview,
        ReviewImportReceipt,
    )


class ReviewCommitPanel:
    """Preview a saved draft, resolve labels, and commit it explicitly.

    ``receipt`` remains available even if opening the verification review fails;
    clicking again reuses the committed child through the existing commit API.
    """

    def __init__(self, review: FigPackReview, *, open_browser: bool = True):
        import ipywidgets as widgets
        from IPython.display import clear_output, display

        self.receipt: ReviewImportReceipt | None = None
        self.verification_review: FigPackReview | None = None
        self.changes = review.preview_import()
        self.output = widgets.Output()
        self.resolutions = {}
        items = []
        for conflict in self.changes.label_conflicts:
            choices = tuple(
                dict.fromkeys(
                    [
                        *review.profile.label_options,
                        *(
                            label
                            for labels in conflict.contributor_labels.values()
                            for label in labels
                        ),
                    ]
                )
            )
            caption = widgets.Label(
                "Merged contributors: "
                + ", ".join(map(str, conflict.contributor_unit_ids))
            )
            labels = widgets.SelectMultiple(
                options=choices, description="Final labels"
            )
            confirmed = widgets.Checkbox(
                description="Use these labels (empty is allowed)"
            )
            self.resolutions[conflict.merged_unit_id] = (labels, confirmed)
            items.extend([caption, labels, confirmed])
        self.button = widgets.Button(
            description=(
                "Commit and review merges"
                if self.changes.merge_groups
                else (
                    "Commit curation"
                    if self.changes.has_changes
                    else "Record reviewed — no changes"
                )
            ),
            layout=widgets.Layout(width="280px"),
            button_style="primary",
        )

        def commit(_):
            self.button.disabled = True
            with self.output:
                clear_output(wait=True)
                try:
                    if any(
                        not confirmation.value
                        for _, confirmation in self.resolutions.values()
                    ):
                        raise ValueError(
                            "Confirm final labels for each merged group before committing."
                        )
                    print(
                        "Committing saved edits; merged units will be reevaluated…"
                    )
                    self.receipt = self.changes.commit(
                        conflict_resolutions={
                            unit: tuple(labels.value)
                            for unit, (labels, _) in self.resolutions.items()
                        },
                        confirm_no_changes=not self.changes.has_changes,
                    )
                    print(
                        f"Committed curation {self.receipt.curation.curation_id}."
                    )
                    if self.receipt.needs_merge_verification:
                        self.verification_review = (
                            self.receipt.continue_review()
                        )
                        from spyglass.spikesorting.v2._curation_transforms import (
                            allocate_merged_unit_ids,
                        )
                        from spyglass.spikesorting.v2.curation import (
                            CurationV2,
                        )

                        source_ids = (
                            CurationV2.Unit & review.parent.as_key()
                        ).fetch("unit_id")
                        merged = tuple(
                            allocate_merged_unit_ids(
                                source_ids, self.changes.merge_groups
                            )
                        )
                        url = self.verification_review.open(
                            open_browser=open_browser, focus_unit_ids=merged
                        )
                        print(f"Inspect merged units: {url}")
                        print(
                            "After inspecting and saving any edits, create a commit panel from this panel's verification_review."
                        )
                    else:
                        print(
                            "Ready for analysis: use this panel's receipt.curation."
                        )
                # Keep callback failures visible and preserve any committed receipt.
                except Exception as exc:  # noqa: BLE001
                    print(f"{type(exc).__name__}: {exc}")
                    print(
                        "If the saved draft changed, create a fresh review.commit_panel()."
                    )
                finally:
                    self.button.disabled = False

        self.button.on_click(commit)
        self.widget = widgets.VBox([*items, self.button, self.output])
        print(self.changes.summary())
        display(self.changes.changed_units())
        display(self.widget)
